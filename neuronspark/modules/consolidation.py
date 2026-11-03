"""Online consolidation of the hippocampal pool's recall into the product-key memory (experiment; the design is
``docs/experiments/consolidation_level2.md``).

A :class:`Consolidation` attached to a hippocampal layer (``layer.consolidation``; None by default and the layer is
then unchanged) holds a delta table ``dV`` (slots, dv) fp32 that the PKM read adds to the values, and teaches the PKM,
chunk by chunk along the row, to reproduce what the pool recalled for each query:

    pool's contribution to the stream   r (1 + f)                 (read gain, hippocampal layer step 15)
    the part recalled from the context  r f                        (f = 4 pi c: copy confidence x novelty, the layer's
                                                                    own estimate that this token is read from context)
    PKM alone must supply it            y* = y + r f / rho         (rho: the PKM copy gate of the same token)
    in the value space                  r* = y* pinv(W_o)^T,  d = r* / ((1 + g_norm) sigmoid(glog)),
                                        val* = d rms(val_0) / rms(d)     (val_0: the read of V alone -- the scale anchor)
    write (the addressed slots only)    dV[u_m] += w_m (val* - val) / (lam + sum_j w_j^2 + usage[u_m]),
                                        usage[u_m] += w_m^2

The write is the regularised least-squares step for one token's read ``sum_m w_m dV[u_m] = val* - val`` with a per-slot
precision (``usage``: the squared weights a slot has already absorbed) in place of the full inverse correlation --
slots that already carry much move less.  No gradient, no global error: the teacher is the same layer's pool read.
Only the gated part ``r f`` is taught: the ungated residual ``r`` is the pool's ordinary contribution on any text and
is still there after a reset; teaching it too rewrote the memory on every token and drifted the pretrained read
(dev cross entropy +0.15 nat on the development checkpoint).

Two knobs against forgetting (the pretrained read is the thing to protect): ``write_topm`` writes only the ``m``
strongest-addressed slots of each head (the read keeps all k), so a token perturbs 4 m instead of 64 slots; a usage
prior (:meth:`prime`: a pass over pretraining-distribution text with ``collect=True`` accumulates the squared weights
each slot receives, scaled by ``prior_scale``) gives the slots the pretrained model relies on a high precision before
the first write, so they move least -- the RLS prior from the pretraining distribution.
``val`` is the read of ``V + dV`` at the time of the write, so a repeated query converges to its target instead of
accumulating.  The ZCN of the read fixes the magnitude of ``r``, so only the direction of ``val`` is expressible: the
target is the direction of ``r*`` at the rms of the base read ``val_0`` (of ``V`` alone; anchoring the scale to the read
of ``V + dV`` fed back on itself and diverged), and ``captured`` (the cosine of ``y*`` with its projection onto the
column space of ``W_o``) reports what the PKM can express at all.  A token whose error exceeds ``max_ratio`` times the
base read's norm is clipped to it; a non-finite target or table raises (fail loud, nothing silently skipped).

Regimes.  Whole row: the read runs chunk by chunk (``chunk`` tokens of the row); the reads of a chunk see the ``dV``
written by the previous chunks of the same row and by earlier rows / calls (``dV`` persists until :meth:`reset`).  The
gradient of a training forward flows through the reads into the PKM values as before; the writes are constants of the
call.  ``write=False`` keeps ``dV`` as it is (reads add it, nothing is written).  The inference regimes (prefill,
decode) do not consult the consolidation yet.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

__all__ = ["Consolidation"]


class Consolidation:
    def __init__(self, layer, *, lam: float = 1.0, chunk: int = 256, write: bool = True, max_ratio: float = 4.0,
                 write_topm: Optional[int] = None) -> None:
        if not hasattr(layer, "pkm_values") or not hasattr(layer, "pkm_o"):
            raise TypeError("Consolidation: the layer has no product-key memory")
        if not float(lam) > 0.0:
            raise ValueError(f"Consolidation: lam must be positive, got {lam}")
        if isinstance(chunk, bool) or int(chunk) < 1:
            raise ValueError(f"Consolidation: chunk must be a positive integer, got {chunk!r}")
        V = layer.pkm_values.detach()
        self.layer = layer
        self.lam = float(lam)
        self.chunk = int(chunk)
        self.write = bool(write)
        self.max_ratio = float(max_ratio)
        self.write_topm = None if write_topm is None else int(write_topm)
        self.collect = False                                       # True: accumulate usage only (the prior), no writes
        self.log_f: List[float] = []                               # per written chunk: mean f
        self.dV = torch.zeros_like(V, dtype=torch.float32)
        self.usage = torch.zeros(int(V.shape[0]), dtype=torch.float32, device=V.device)
        with torch.no_grad():
            self.pinv = torch.linalg.pinv(layer.pkm_o.weight.detach().float())        # (dv, D): r* = y* pinv^T
        self.writes = 0                                            # tokens written (every token of every write chunk)
        self.log: List[Dict[str, float]] = []                      # per written chunk: captured, |e| / |val|, |dV| rms

    # ------------------------------------------------------------------------------------------------ state
    def reset(self, usage: bool = True) -> None:
        """Clear the delta table (and the usage unless ``usage=False``: a primed prior stays)."""
        self.dV.zero_()
        if usage:
            self.usage.zero_()

    def prime(self, scale: float) -> None:
        """Turn the usage collected under ``collect=True`` into the prior: usage *= scale, dV untouched."""
        self.usage.mul_(float(scale))

    def values(self) -> torch.Tensor:
        """The values the read uses: ``V + dV`` (fp32; differentiable in V)."""
        return self.layer.pkm_values + self.dV

    def state_dict(self) -> Dict[str, torch.Tensor]:
        return {"dV": self.dV.detach().cpu(), "usage": self.usage.detach().cpu()}

    def load_state_dict(self, d: Dict[str, torch.Tensor]) -> None:
        self.dV.copy_(d["dV"].to(self.dV.device))
        self.usage.copy_(d["usage"].to(self.usage.device))

    # ------------------------------------------------------------------------------------------------ the target
    def target(self, val0: torch.Tensor, glog: torch.Tensor, y: torch.Tensor, r_pool: torch.Tensor, f: torch.Tensor,
               rho: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """``(val*, captured)`` for N tokens: val0 (N, dv) fp32 the read of V alone (the scale anchor), glog (N, dv), y
        (N, D) the PKM output, r_pool (N, D) the pool residual, f / rho (N,) the read gain factor and the PKM copy gate
        (autocast disabled by the caller)."""
        layer = self.layer
        y_star = y.float() + r_pool.float() * (f.float() / rho.float().clamp_min(1e-3)).unsqueeze(-1)
        r_star = y_star @ self.pinv.t()                                                   # (N, dv)
        back = r_star @ layer.pkm_o.weight.detach().float().t()                           # its image in D
        captured = F.cosine_similarity(back, y_star, dim=-1)
        gains = (1.0 + layer.pkm_norm_gain.detach().float()) * torch.sigmoid(glog.float())
        d = r_star / gains.clamp_min(1e-6)
        rms_val = val0.float().pow(2).mean(dim=-1, keepdim=True).sqrt()
        rms_d = d.pow(2).mean(dim=-1, keepdim=True).sqrt().clamp_min(1e-12)
        return d * (rms_val / rms_d), captured

    # ------------------------------------------------------------------------------------------------ the write
    @torch.no_grad()
    def write_chunk(self, u: torch.Tensor, w: torch.Tensor, val: torch.Tensor, val_star: torch.Tensor,
                    captured: Optional[torch.Tensor] = None, val0: Optional[torch.Tensor] = None) -> None:
        """One chunk of N tokens: u (N, Hk) int64 slot ids, w (N, Hk) fp32 weights, val (the read of V + dV) / val*
        (N, dv) fp32; val0 the read of V alone (the clip scale; val when None)."""
        if self.collect:
            self.usage.index_add_(0, u.reshape(-1), w.reshape(-1).pow(2))
            return
        if not self.write:
            return
        if not bool(torch.isfinite(val_star).all()):
            raise RuntimeError("Consolidation: a non-finite write target")
        if self.write_topm is not None:                                                  # the m strongest slots per head
            H = self.layer.pkm_heads
            wh = w.reshape(w.shape[0], H, -1)
            keep = torch.zeros_like(wh, dtype=torch.bool)
            keep.scatter_(-1, wh.topk(min(self.write_topm, wh.shape[-1]), dim=-1).indices, True)
            w = torch.where(keep.reshape(w.shape), w, torch.zeros_like(w))
        e = (val_star - val).float()                                                     # (N, dv)
        scale = (val if val0 is None else val0).float().norm(dim=-1, keepdim=True) * self.max_ratio
        e = e * torch.minimum(torch.ones_like(scale), scale / e.norm(dim=-1, keepdim=True).clamp_min(1e-12))
        denom = self.lam + w.pow(2).sum(dim=-1, keepdim=True) + self.usage[u]            # (N, Hk)
        coef = w / denom                                                                 # (N, Hk)
        self.dV.index_add_(0, u.reshape(-1), (coef.unsqueeze(-1) * e.unsqueeze(1)).reshape(-1, e.shape[-1]))
        self.usage.index_add_(0, u.reshape(-1), w.reshape(-1).pow(2))
        if not bool(torch.isfinite(self.dV).all()):
            raise RuntimeError("Consolidation: the delta table became non-finite")
        self.writes += int(e.shape[0])
        self.log.append({"captured": float(captured.mean()) if captured is not None else float("nan"),
                         "err_over_val": float(e.norm(dim=-1).mean() / val.float().norm(dim=-1).mean().clamp_min(1e-12)),
                         "dV_rms": float(self.dV.pow(2).mean().sqrt())})

    # ------------------------------------------------------------------------------------------------ whole row
    def read_write(self, ops, w: torch.Tensor, u: torch.Tensor, glog: torch.Tensor, x_hat: torch.Tensor,
                   r_pool: torch.Tensor, gates: Optional[Tuple[torch.Tensor, torch.Tensor]]) -> torch.Tensor:
        """The layer's whole-row PKM read with consolidation: w / u (T, B, Hk), glog (T, B, dv), x_hat (T, B, D), r_pool
        (T, B, D) bf16, gates = (f, rho) each (T, B) fp32 or None (then nothing is written) -> y (T, B, D) bf16.  Chunks
        of ``self.chunk`` tokens along T: read with V + dV, target, write, next chunk."""
        from neuronspark.modules.pkm import PKM_EPS
        layer = self.layer
        T, B, Hk = (int(s) for s in w.shape)
        ys = []
        for t0 in range(0, T, self.chunk):
            t1 = min(T, t0 + self.chunk)
            wc, uc, gc = w[t0:t1], u[t0:t1], glog[t0:t1]
            val = ops.pkm_value_read(wc, uc, self.values())                             # (n, B, dv) fp32
            val0 = (ops.pkm_value_read(wc, uc, layer.pkm_values.detach())
                    if (gates is not None and (self.write or self.collect)) else None)
            r = val * torch.rsqrt(val.pow(2).mean(dim=-1, keepdim=True) + PKM_EPS) * (
                1.0 + layer.pkm_norm_gain.to(torch.float32))
            r = r * torch.sigmoid(gc.to(torch.float32))
            y = layer._pkm_out(r, x_hat[t0:t1])                                          # (n, B, D) bf16
            ys.append(y)
            if gates is not None and (self.write or self.collect):
                f, rho = gates
                n = t1 - t0
                self.log_f.append(float(f[t0:t1].float().mean()))
                v0 = val0.reshape(n * B, -1)
                val_star, captured = self.target(v0, gc.detach().reshape(n * B, -1), y.detach().reshape(n * B, -1),
                                                 r_pool[t0:t1].detach().reshape(n * B, -1), f[t0:t1].reshape(-1),
                                                 rho[t0:t1].reshape(-1))
                self.write_chunk(uc.detach().reshape(n * B, Hk), wc.detach().reshape(n * B, Hk),
                                 val.detach().reshape(n * B, -1), val_star, captured, val0=v0)
        return torch.cat(ys, dim=0)
