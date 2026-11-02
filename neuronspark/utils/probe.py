"""Validation probes of the dendritic neuron layers and of the hippocampal layers, read off a whole-row forward.

A probe is attached for the duration of whole-row forwards under ``torch.no_grad()`` (:func:`probing`); it receives
from the layers tensors they form anyway and accumulates sums; :meth:`summary` gives the means.  The rows' input
tokens are given to a probe before each forward (:meth:`DendriticProbe.row` / :meth:`HippocampalProbe.row`): the
document boundaries of the rows (an EOS token ``eos_id`` followed by the next document's first token) are read off
them.  A probe changes nothing in the forward, and a layer without one (``probe is None``, the default) does no extra
work.

:class:`DendriticProbe`, per dendritic neuron layer and per head, as means over the rows:

* ``retention``: ``exp(A sum_t dt_t)`` over the row, the product of the row's leak factors ``a_t = exp(dt_t A)`` -- the
  fraction of a compartment's potential at the row's first token that is still there after its last token;
* ``retention_horizon``: the same with the sum scaled by ``horizon / T``, the retention of a row ``horizon`` tokens long
  whose steps keep the row's mean (``horizon`` 32768 by default);
* ``exponent``: ``-A sum_t dt_t``, the row's total decay exponent (minus the log of the retention), readable where the
  retention underflows to 0; ``T / exponent`` is the head's e-fold length in tokens at the row's mean step (the rows
  of one probe have one length ``T``, kept in ``probe.T``);
* ``firing``: the fraction of the somatic outputs ``s_t`` released through the threshold (non-zero), and
  ``firing_by_segment``: the same per segment of the row (``segments`` equal parts; one segment by default);
* at the document boundaries, as means over the boundaries: ``boundary_retention``, ``exp(A (dt_e + dt_e+1))``, what
  is left of the old document's potential after the new document's first token; ``boundary_step``,
  ``(dt_e + dt_e+1) / 2``, the mean step over the boundary's two tokens; and ``body_step``, the mean step over the
  other tokens (neither an EOS nor the token after one): a ``boundary_step`` well above ``body_step`` is a head that
  flushes its compartments at the boundaries; ``boundaries`` counts the boundaries seen (0: the three are NaN).

:class:`HippocampalProbe`, per hippocampal layer, as means over the rows (the pool attention, the copy arbitration,
the spiking FFN and the product-key memory each hand it their tensors):

* ``decay_exponent`` (C,): ``-sum_t log a_t,c`` per decay channel, the channel's total decay exponent over the row
  (``T / decay_exponent`` its e-fold length in tokens); ``clock_exponent`` (H,): ``gamma_h sum_t -mean_c log a_t,c``,
  the head's pool clock run over the row (``T / clock_exponent`` the tokens over which the head's eligibility bias
  reaches one e-fold); at the boundaries ``clock_boundary_step`` and ``clock_body_step``, the mean clock increment
  ``-mean_c log a`` over a boundary's two tokens and elsewhere, the hippocampal counterpart of the dendritic
  ``boundary_step``;
* ``write`` (H,): the mean write tag ``sigmoid(g + wb - sparsity)`` of the head, the fraction of a token's value that
  enters the pool; ``gate_firing``: the fraction of non-zero write-gate spikes;
* ``copy_pi``, ``novelty``, ``read_gain``: the means of the copy confidence, the novelty factor and the read gain
  factor ``kappa pi c``;
* ``ffn_in_firing``, ``ffn_gate_firing``, ``ffn_up_firing``: the firing rates of the FFN input neuron and of the
  spiking FFN's gate and up neurons;
* ``pkm_entropy``: the mean entropy (nats) of a token's top-k slot weights per PKM head; ``pkm_slots_used``: the
  fraction of the PKM's slots read at least once over the rows; ``pkm_gate``: the mean PKM read gate ``sigmoid(glog)``;
* ``boundaries``: the boundaries seen.
"""
from __future__ import annotations

import contextlib
from typing import Dict, Iterator, List

import torch
import torch.nn.functional as F

PROBE_NAMES = ("retention", "retention_horizon", "exponent", "firing", "firing_by_segment", "boundary_retention",
               "boundary_step", "body_step", "boundaries")
HIPPOCAMPAL_NAMES = ("decay_exponent", "clock_exponent", "clock_boundary_step", "clock_body_step", "write",
                     "gate_firing", "copy_pi", "novelty", "read_gain", "ffn_in_firing", "ffn_gate_firing",
                     "ffn_up_firing", "pkm_entropy", "pkm_slots_used", "pkm_gate", "boundaries")


def _boundary_mask(tokens: torch.Tensor, eos_id: int, who: str) -> torch.Tensor:
    if tokens.dim() != 2:
        raise ValueError(f"{who}.row: tokens must be (rows, T), got {tuple(tokens.shape)}")
    return tokens == eos_id


def _boundary_parts(eos: torch.Tensor):
    """From the (B, T) EOS mask: ``span`` (B, T-1) the boundaries (an EOS with a next token) and ``body`` (B, T) the
    tokens that are neither an EOS nor the token after one."""
    B = int(eos.shape[0])
    after = torch.cat([eos.new_zeros(B, 1), eos[:, :-1]], dim=1)
    return eos[:, :-1], ~(eos | after)


def _firing(spikes: torch.Tensor) -> torch.Tensor:
    """(B,) the fraction of non-zero entries per row of (T, B, ...) spikes."""
    T, B = int(spikes.shape[0]), int(spikes.shape[1])
    return (spikes.reshape(T, B, -1) != 0).sum(dim=(0, 2), dtype=torch.float32) / (T * spikes[0, 0].numel())


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

    @staticmethod
    def targets(model) -> List:
        """The modules this probe attaches to: the mixers of the dendritic neuron layers."""
        return dendritic_mixers(model)

    def row(self, tokens: torch.Tensor) -> None:
        """The input tokens (B, T) of the forward about to run: its document boundaries are read off them.  Every
        probed forward needs its own call (a forward's mixers each read the rows once)."""
        self._eos = _boundary_mask(tokens, self.eos_id, "DendriticProbe")
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
        span, body_mask = _boundary_parts(self._eos.to(zxbcdt.device))
        pair = (dt[:, :-1] + dt[:, 1:])[span]                                # (n, H) the step over a boundary
        body = dt[body_mask]                                                 # (n_body, H)
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


class HippocampalProbe:
    """Per hippocampal layer sums of the probed quantities (see the module docstring); :meth:`summary` gives the
    means over the rows.  ``eos_id`` is the token that ends a document in the rows."""

    def __init__(self, eos_id: int) -> None:
        self.eos_id = int(eos_id)
        self.T = None
        self._eos = None
        self._read = set()
        self._sums: Dict[int, Dict[str, torch.Tensor]] = {}
        self._rows: Dict[int, Dict[str, int]] = {}                  # rows seen by each hook, per layer
        self._counts: Dict[int, Dict[str, int]] = {}                 # boundaries, body tokens, per layer

    @staticmethod
    def targets(model) -> List:
        """The modules this probe attaches to: the hippocampal layers."""
        return hippocampal_layers(model)

    def row(self, tokens: torch.Tensor) -> None:
        """The input tokens (B, T) of the forward about to run (see :meth:`DendriticProbe.row`)."""
        self._eos = _boundary_mask(tokens, self.eos_id, "HippocampalProbe")
        self._read = set()

    def _acc(self, layer) -> Dict[str, torch.Tensor]:
        acc = self._sums.get(layer.layer_idx)
        if acc is None:
            acc = self._sums[layer.layer_idx] = {}
            self._rows[layer.layer_idx] = {"attention": 0, "copy": 0, "ffn_input": 0, "ffn": 0, "pkm": 0}
            self._counts[layer.layer_idx] = {"boundaries": 0, "body": 0}
        return acc

    def _add(self, acc: Dict[str, torch.Tensor], name: str, value: torch.Tensor) -> None:
        if name in acc:
            acc[name] += value
        else:
            acc[name] = value.clone()

    @torch.no_grad()
    def attention(self, layer, a: torch.Tensor, gn, gate: torch.Tensor, gains: torch.Tensor) -> None:
        """From the pool attention: ``a`` (T, B, C, 1) the decay per channel, ``gn`` the H-tuple (T, B, dk) of the
        write-gate spikes plus the write bias, ``gate`` (T, B, H dk) the write-gate spikes, ``gains`` (H,) the heads'
        eligibility slopes gamma_h."""
        T, B = int(a.shape[0]), int(a.shape[1])
        if self.T is None:
            self.T = T
        elif self.T != T:
            raise RuntimeError(f"HippocampalProbe: one probe reads rows of one length, got {T} after {self.T}")
        if self._eos is None or tuple(self._eos.shape) != (B, T):
            raise RuntimeError(f"HippocampalProbe: probe.row(tokens) of the rows ({B}, {T}) must precede the probed "
                               f"forward" + ("" if self._eos is None else f", got {tuple(self._eos.shape)}"))
        if layer.layer_idx in self._read:
            raise RuntimeError(f"HippocampalProbe: layer {layer.layer_idx} read the rows of probe.row(tokens) twice: "
                               "a new forward needs its own probe.row(tokens)")
        self._read.add(layer.layer_idx)
        acc = self._acc(layer)
        la = a.reshape(T, B, -1).float().clamp_min(1e-6).log()              # (T, B, C) <= 0
        step = -la.mean(dim=-1)                                              # (T, B) >= 0 the clock increment
        self._add(acc, "decay_exponent", (-la).sum(dim=(0, 1)))              # (C,)
        self._add(acc, "clock_exponent", step.sum(dim=0).sum(dim=0) * gains.float().reshape(-1))     # (H,)
        eos = self._eos.to(a.device)
        span, body_mask = _boundary_parts(eos)
        st = step.transpose(0, 1)                                            # (B, T)
        pair = (st[:, :-1] + st[:, 1:])[span]                                # (n,)
        body = st[body_mask]                                                 # (n_body,)
        self._add(acc, "clock_boundary_step", (pair / 2).sum().reshape(1))
        self._add(acc, "clock_body_step", body.sum().reshape(1))
        write = torch.stack([torch.sigmoid(g.float() - layer.write_sparsity).mean() for g in gn])      # (H,)
        self._add(acc, "write", write * B)
        self._add(acc, "gate_firing", _firing(gate).sum().reshape(1))
        self._rows[layer.layer_idx]["attention"] += B
        self._counts[layer.layer_idx]["boundaries"] += int(pair.shape[0])
        self._counts[layer.layer_idx]["body"] += int(body.shape[0])

    @torch.no_grad()
    def copy(self, layer, pi: torch.Tensor, c: torch.Tensor, f: torch.Tensor) -> None:
        """From the copy arbitration: ``pi``, ``c``, ``f`` (T, B) the copy confidence, the novelty factor and the
        read gain factor."""
        acc = self._acc(layer)
        for name, v in (("copy_pi", pi), ("novelty", c), ("read_gain", f)):
            self._add(acc, name, v.float().mean(dim=0).sum().reshape(1))
        self._rows[layer.layer_idx]["copy"] += int(pi.shape[1])

    @torch.no_grad()
    def ffn_input(self, layer, zs: torch.Tensor) -> None:
        """From the FFN input neuron: ``zs`` (T, B, D) its spikes."""
        acc = self._acc(layer)
        self._add(acc, "ffn_in_firing", _firing(zs).sum().reshape(1))
        self._rows[layer.layer_idx]["ffn_input"] += int(zs.shape[1])

    @torch.no_grad()
    def ffn(self, layer, g: torch.Tensor, u: torch.Tensor) -> None:
        """From the spiking FFN: ``g``, ``u`` (T, B, D_ff) the gate and up neurons' spikes."""
        acc = self._acc(layer)
        self._add(acc, "ffn_gate_firing", _firing(g).sum().reshape(1))
        self._add(acc, "ffn_up_firing", _firing(u).sum().reshape(1))
        self._rows[layer.layer_idx]["ffn"] += int(g.shape[1])

    @torch.no_grad()
    def pkm(self, layer, w: torch.Tensor, u: torch.Tensor, glog: torch.Tensor) -> None:
        """From the product-key memory read: ``w`` (T, B, H, k) the slot weights, ``u`` (T, B, H, k) the slot ids,
        ``glog`` (T, B, dv) the read gate logits."""
        acc = self._acc(layer)
        wf = w.float()
        ent = -(wf * wf.clamp_min(1e-30).log()).sum(dim=-1)                  # (T, B, H)
        self._add(acc, "pkm_entropy", ent.mean(dim=(0, 2)).sum().reshape(1))
        self._add(acc, "pkm_gate", torch.sigmoid(glog.float()).mean(dim=(0, 2)).sum().reshape(1))
        used = acc.get("pkm_used")
        if used is None:
            used = acc["pkm_used"] = torch.zeros(int(layer.pkm_n) ** 2, dtype=torch.bool, device=w.device)
        used[u.reshape(-1)] = True
        self._rows[layer.layer_idx]["pkm"] += int(w.shape[1])

    def summary(self) -> Dict[int, Dict[str, torch.Tensor]]:
        """``{layer_idx: {name: fp32 on the CPU}}`` in layer order (the names of ``HIPPOCAMPAL_NAMES``): the means
        over the rows; the clock boundary / body steps over the boundaries / the body tokens (NaN when none);
        ``pkm_slots_used`` the fraction of the slots read; ``boundaries`` the count."""
        out = {}
        for i, acc in sorted(self._sums.items()):
            rows = self._rows[i]
            if len(set(rows.values())) != 1 or not rows["attention"]:
                raise RuntimeError(f"HippocampalProbe: layer {i} did not reach every hook on every row: {rows}")
            n, nb, nbody = rows["attention"], self._counts[i]["boundaries"], self._counts[i]["body"]
            over = lambda v, k: (v / k).cpu() if k else torch.full_like(v, float("nan")).cpu()
            s = {name: (acc[name] / n).cpu() for name in HIPPOCAMPAL_NAMES
                 if name not in ("clock_boundary_step", "clock_body_step", "pkm_slots_used", "boundaries")}
            s["clock_boundary_step"] = over(acc["clock_boundary_step"], nb)
            s["clock_body_step"] = over(acc["clock_body_step"], nbody)
            s["pkm_slots_used"] = acc["pkm_used"].float().mean().reshape(1).cpu()
            s["boundaries"] = nb
            out[i] = s
        return out


def dendritic_mixers(model) -> List:
    """The mixers of the model's dendritic neuron layers, in depth order (``model`` an LM head model or its backbone)."""
    backbone = getattr(model, "backbone", model)
    return [layer.mixer_a for layer in backbone.dendritic_layers]


def hippocampal_layers(model) -> List:
    """The model's hippocampal layers, in depth order (``model`` an LM head model or its backbone)."""
    backbone = getattr(model, "backbone", model)
    return list(backbone.hippocampal_layers)


@contextlib.contextmanager
def probing(model, *probes) -> Iterator:
    """The probes attached to their modules of ``model`` inside the block (each probe's ``targets``), detached on
    exit (also on an error).  Yields the probe when one is given, else the tuple of probes."""
    if not probes:
        raise ValueError("probing: give at least one probe")
    attached = []
    for p in probes:
        for m in p.targets(model):
            if m.probe is not None:
                raise RuntimeError(f"probing: layer {m.layer_idx} already carries a probe")
            attached.append(m)
    seen = set()
    for m in attached:
        if id(m) in seen:
            raise RuntimeError(f"probing: layer {m.layer_idx} is targeted by two probes")
        seen.add(id(m))
    try:
        for p in probes:
            for m in p.targets(model):
                m.probe = p
        yield probes[0] if len(probes) == 1 else probes
    finally:
        for m in attached:
            m.probe = None
