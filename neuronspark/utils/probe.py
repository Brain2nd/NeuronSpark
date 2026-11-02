"""Validation probes of the dendritic neuron layers, read off a whole-row forward.

A :class:`DendriticProbe` attached to the mixers of a model's dendritic neuron layers (:func:`probing`, for the
duration of whole-row forwards under ``torch.no_grad()``) receives from each mixer the row's projections and the somatic
outputs it forms anyway, and accumulates per layer and per head, as means over the rows:

* ``retention``: ``exp(A sum_t dt_t)`` over the row, the product of the row's leak factors ``a_t = exp(dt_t A)`` -- the
  fraction of a compartment's potential at the row's first token that is still there after its last token;
* ``retention_horizon``: the same with the sum scaled by ``horizon / T``, the retention of a row ``horizon`` tokens long
  whose steps keep the row's mean (``horizon`` 32768 by default);
* ``exponent``: ``-A sum_t dt_t``, the row's total decay exponent (minus the log of the retention), readable where the
  retention underflows to 0; ``T / exponent`` is the head's e-fold length in tokens at the row's mean step (the rows
  of one probe have one length ``T``, kept in ``probe.T``);
* ``firing``: the fraction of the somatic outputs ``s_t`` released through the threshold (non-zero), and
  ``firing_by_segment``: the same per segment of the row (``segments`` equal parts; one segment by default);
* at the document boundaries of the rows (an EOS token ``eos_id`` followed by the next document's first token; the
  rows' tokens are given to the probe before each forward, :meth:`DendriticProbe.row`), as means over the
  boundaries: ``boundary_retention``, ``exp(A (dt_e + dt_e+1))``, what is left of the old document's potential after
  the new document's first token; ``boundary_step``, ``(dt_e + dt_e+1) / 2``, the mean step over the boundary's two
  tokens; and ``body_step``, the mean step over the other tokens (neither an EOS nor the token after one): a
  ``boundary_step`` well above ``body_step`` is a head that flushes its compartments at the boundaries; ``boundaries``
  counts the boundaries seen (0: the three are NaN).

The probe changes nothing in the forward, and a mixer without one (``probe is None``, the default) does no extra work.
"""
from __future__ import annotations

import contextlib
from typing import Dict, Iterator, List

import torch
import torch.nn.functional as F

PROBE_NAMES = ("retention", "retention_horizon", "exponent", "firing", "firing_by_segment", "boundary_retention",
               "boundary_step", "body_step", "boundaries")


class DendriticProbe:
    """Per-layer, per-head sums of the probed quantities; :meth:`summary` gives their means over the rows (over the
    boundaries for the boundary readings).  ``eos_id`` is the token that ends a document in the rows."""

    def __init__(self, eos_id: int, horizon: int = 32768, segments: int = 1) -> None:
        if int(horizon) <= 0 or int(segments) <= 0:
            raise ValueError(f"DendriticProbe: horizon and segments must be positive, got {horizon}, {segments}")
        self.eos_id, self.horizon, self.segments = int(eos_id), int(horizon), int(segments)
        self.T = None                                                # the rows' length, from the first row
        self._eos = None                                             # (B, T) bool, the EOS tokens of the rows of row()
        self._read = set()                                           # the layers that read the rows of row()
        self._sums: Dict[int, Dict[str, torch.Tensor]] = {}
        self._rows: Dict[int, int] = {}
        self._counts: Dict[int, Dict[str, int]] = {}                 # per layer: boundaries, body tokens

    def row(self, tokens: torch.Tensor) -> None:
        """The input tokens (B, T) of the forward about to run: its document boundaries are read off them.  Every
        probed forward needs its own call (a forward's mixers each read the rows once)."""
        if tokens.dim() != 2:
            raise ValueError(f"DendriticProbe.row: tokens must be (rows, T), got {tuple(tokens.shape)}")
        self._eos = tokens == self.eos_id
        self._read = set()

    @torch.no_grad()
    def dendritic(self, mixer, zxbcdt: torch.Tensor, xs: torch.Tensor) -> None:
        """Called by a dendritic mixer in its whole-row forward: ``zxbcdt`` (B, T, proj_dim) its projections (the dt
        rows last), ``xs`` (B T, d_inner) its somatic outputs."""
        B, T, _ = (int(s) for s in zxbcdt.shape)
        H, S = int(mixer.nheads), self.segments
        if T % S:
            raise RuntimeError(f"DendriticProbe: a row of {T} tokens does not split into {S} segments")
        if self.T is None:
            self.T = T
        elif self.T != T:
            raise RuntimeError(f"DendriticProbe: one probe reads rows of one length, got {T} after {self.T}")
        if self._eos is None or tuple(self._eos.shape) != (B, T):
            raise RuntimeError(f"DendriticProbe: probe.row(tokens) of the rows ({B}, {T}) must precede the probed "
                               f"forward" + ("" if self._eos is None else f", got {tuple(self._eos.shape)}"))
        if mixer.layer_idx in self._read:
            raise RuntimeError(f"DendriticProbe: layer {mixer.layer_idx} read the rows of probe.row(tokens) twice: a "
                               "new forward needs its own probe.row(tokens)")
        self._read.add(mixer.layer_idx)
        A = mixer.A()                                                        # (H,) < 0
        dt = F.softplus(zxbcdt[:, :, -H:].float() + mixer.dt_bias)         # (B, T, H) the steps
        log_r = A * dt.sum(dim=1)                                            # (B, H) <= 0: log of the row's retention
        P = int(mixer.head_dim)
        fired = (xs.reshape(B, S, T // S, H, P) != 0).sum(dim=(0, 2, 4), dtype=torch.float32) / (T // S * P)
        eos = self._eos.to(zxbcdt.device)
        span = eos[:, :-1]                                                   # (B, T-1) an EOS with a next token
        pair = (dt[:, :-1] + dt[:, 1:])[span]                                # (n, H) the step over a boundary's 2 tokens
        after = torch.cat([eos.new_zeros(B, 1), eos[:, :-1]], dim=1)         # (B, T) the token after an EOS
        body = dt[~(eos | after)]                                            # (n_body, H)
        acc = self._sums.get(mixer.layer_idx)                        # (fired: (S, H), the rows' sum of fractions)
        if acc is None:
            z = lambda *shape: torch.zeros(*shape, dtype=torch.float32, device=zxbcdt.device)
            acc = self._sums[mixer.layer_idx] = {"retention": z(H), "retention_horizon": z(H), "exponent": z(H),
                                                 "firing": z(S, H), "boundary_retention": z(H), "boundary_step": z(H),
                                                 "body_step": z(H)}
            self._rows[mixer.layer_idx] = 0
            self._counts[mixer.layer_idx] = {"boundaries": 0, "body": 0}
        acc["retention"] += torch.exp(log_r).sum(dim=0)
        acc["retention_horizon"] += torch.exp(log_r * (self.horizon / T)).sum(dim=0)
        acc["exponent"] += (-log_r).sum(dim=0)
        acc["firing"] += fired
        acc["boundary_retention"] += torch.exp(A * pair).sum(dim=0)
        acc["boundary_step"] += (pair / 2).sum(dim=0)
        acc["body_step"] += body.sum(dim=0)
        self._rows[mixer.layer_idx] += B
        self._counts[mixer.layer_idx]["boundaries"] += int(pair.shape[0])
        self._counts[mixer.layer_idx]["body"] += int(body.shape[0])

    def summary(self) -> Dict[int, Dict[str, torch.Tensor]]:
        """``{layer_idx: {name: fp32 on the CPU}}`` in layer order, the means over the rows seen: ``retention``,
        ``retention_horizon``, ``exponent``, ``firing`` (H,) and ``firing_by_segment`` (segments, H); over the
        boundaries seen: ``boundary_retention``, ``boundary_step``, ``body_step`` (H,; NaN when ``boundaries``, their
        count, is 0)."""
        out = {}
        for i, acc in sorted(self._sums.items()):
            n, nb, nbody = self._rows[i], self._counts[i]["boundaries"], self._counts[i]["body"]
            seg = (acc["firing"] / n).cpu()
            over = lambda v, k: (v / k).cpu() if k else torch.full_like(v, float("nan")).cpu()
            out[i] = {"retention": (acc["retention"] / n).cpu(), "retention_horizon": (acc["retention_horizon"] / n).cpu(),
                      "exponent": (acc["exponent"] / n).cpu(), "firing": seg.mean(dim=0), "firing_by_segment": seg,
                      "boundary_retention": over(acc["boundary_retention"], nb),
                      "boundary_step": over(acc["boundary_step"], nb), "body_step": over(acc["body_step"], nbody),
                      "boundaries": nb}
        return out


def dendritic_mixers(model) -> List:
    """The mixers of the model's dendritic neuron layers, in depth order (``model`` an LM head model or its backbone)."""
    backbone = getattr(model, "backbone", model)
    return [layer.mixer_a for layer in backbone.dendritic_layers]


@contextlib.contextmanager
def probing(model, probe: DendriticProbe) -> Iterator[DendriticProbe]:
    """``probe`` attached to the model's dendritic mixers inside the block, detached on exit (also on an error)."""
    mixers = dendritic_mixers(model)
    for m in mixers:
        if m.probe is not None:
            raise RuntimeError(f"probing: dendritic mixer of layer {m.layer_idx} already carries a probe")
    for m in mixers:
        m.probe = probe
    try:
        yield probe
    finally:
        for m in mixers:
            m.probe = None
