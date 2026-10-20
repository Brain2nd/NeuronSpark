"""Product-key memory (PKM) read of the memory layer.

From the FFN input x_hat = ffn_norm(h') (T, B, D) fp32 (H = 4 heads, n = 128 sub-keys per half, k = 16, dh = 64,
dv = 768, one shared table of n^2 = 16384 slots):

    q     = bf16(x_hat @ pkm_q^T)  (T, B, H * 2 * dh);   glog = bf16(x_hat @ pkm_g^T)  (T, B, dv)
    -- fp32, autocast disabled, from here --
    qh    = normalize(f32(q).view(T, B, H, 2, dh));       kh = normalize(pkm_subkeys)  (H, 2, n, dh)
    beta  = sqrt(dh) exp(pkm_log_gain).view(H, 2)
    s     = einsum("tbhad,hand->tbhan", qh, kh) * beta    (TF32 in the training regime, forward and backward;
                                                           IEEE fp32 in inference)
    sc, u = exact product top-k of the two halves (k of n^2 slots per head; torch.topk tie semantics)
    w     = softmax(sc)                                    (T, B, H, k)
    o     = sum_{m < H k} w_m pkm_values[u_m]              (T, B, dv): the heads summed over the shared table
    r     = o rsqrt(mean(o^2) + 1e-6) (1 + pkm_norm_gain) * sigmoid(f32(glog))
    y     = bf16(r @ pkm_o^T)                              (T, B, D) bf16

Ops per regime:

* whole row: projections on the (T * B, D) view; the score einsum through
  :func:`neuronspark.utils.precision.regime_product` (its backward GEMMs under the forward's TF32 setting);
  ``ops.pkm_topk_train`` (pruned exact top-k, with backward); ``ops.pkm_value_read`` (exact fp32 fma chain,
  deterministic backward); torch normalisation / softmax / ZCN;
* prefill (T >= 2): ``ops.pkm_topk_prefill`` (exact top-k) + torch softmax; ``ops.pkm_weighted_sum``;
  ``ops.rmsnorm_rows`` for the ZCN;
* decode (T == 1): ``ops.pkm_topk_softmax_t1`` (the gain multiply, the exact top-k and the softmax in one kernel);
  ``F.embedding`` + an fp32 einsum for the value sum; torch ZCN.

Parameters (keys, flat on the layer): ``pkm_q.weight`` (512, 1536), ``pkm_g.weight`` (768, 1536), ``pkm_o.weight``
(1536, 768), ``pkm_values`` (16384, 768), ``pkm_subkeys`` (4, 2, 128, 64), ``pkm_norm_gain`` (768,) = 0,
``pkm_log_gain`` (8,) = 0.  Persistent scalar buffers: ``pkm_copy_strength`` (the copy gate's fixed strength, 4.0),
``pkm_copy_tau`` (the copy detector's sink, 1.0 until calibrated), ``pkm_calibrated`` (1 once the launch calibration
has run), ``pkm_launch_rms_res``, ``pkm_launch_rms_y``, ``pkm_launch_entropy`` (calibration records).

Launch: the PKM tensors are built and drawn on a forked stream seeded ``mechanism_init_seed(996_000, layer_idx)``
(the three projections' PyTorch default draws, pkm_q / pkm_g Xavier-uniform, pkm_o N(0, out_std), pkm_values
N(0, dv^-0.5), pkm_subkeys N(0, 1)).  The launch calibration (:mod:`neuronspark.utils.calibration`) then sets the
sub-keys (spherical k-means of the launch half-queries), ``pkm_copy_tau``, the pkm_o scale and the records.  A layer
whose PKM was neither calibrated nor loaded from a calibrated checkpoint refuses to run outside a calibration.

Calibration capture protocol (the ``calib`` object handed to the whole-row forward; duck-typed):

* ``calib.pkm_mode(layer)`` -> ``None`` | ``"queries"`` | ``"rms"``: the pass this layer is calibrated in (None: not
  calibrated in this call; an unready PKM then adds nothing to the layer);
* ``calib.record(layer, name, value)`` receives, in mode ``"queries"``: ``"copy_max"`` (the flattened (T, B) copy
  maxima, CPU fp32) and ``"pkm_queries"`` ((T * B, H, 2, dh) unit half-queries, CPU fp32); in mode ``"rms"``:
  ``"pkm_usage"`` (``(u, w, ii, jj)`` detached (T, B, H, k) device tensors), ``"ms_y"`` (per-token mean square of the
  centred read, CPU float64) and ``"ms_res"`` (per-token mean square of the attention residual, CPU float64).  In
  both modes the PKM adds nothing to the layer output.
"""
from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from neuronspark.modules.linear import Linear
from neuronspark.modules.plif import forked_rng, linear_init, rows_rmsnorm
from neuronspark.utils.precision import regime_product

__all__ = ["PKMMixin", "PKM_SEED_BASE", "PKM_EPS", "PKM_CALIB_MODES"]

# forked launch stream of the PKM tensors
PKM_SEED_BASE = 996_000
# eps of the read's ZCN
PKM_EPS = 1e-6
# calibration passes of the PKM launch calibration
PKM_CALIB_MODES = (None, "queries", "rms")


def _half_query_scores(qh: torch.Tensor, kh: torch.Tensor) -> torch.Tensor:
    """(T, B, H, 2, n) fp32 dot products of the unit half-queries (T, B, H, 2, dh) with the unit sub-keys (H, 2, n, dh)."""
    return torch.einsum("tbhad,hand->tbhan", qh, kh)


class PKMMixin:
    """Mixin of :class:`neuronspark.modules.memory_layer.MemoryLayer` (the product-key memory read)."""

    def _build_pkm(self, config, layer_idx: int, device, dtype, init: str) -> None:
        D = int(config.D)
        H, n = int(config.memory_pkm_heads), int(config.memory_pkm_subkeys)
        dq, dv = int(config.memory_pkm_query_dim), int(config.memory_pkm_value_dim)
        f32 = torch.float32 if dtype is None else dtype
        self.pkm_heads, self.pkm_n, self.pkm_k = H, n, int(config.memory_pkm_topk)
        self.pkm_half, self.pkm_dv = dq // 2, dv
        self.pkm_launch_ratio = float(config.memory_pkm_launch_ratio)
        self.pkm_kmeans_iters = int(config.memory_pkm_kmeans_iters)
        li = linear_init(init)
        if init == "launch":
            with forked_rng(device, PKM_SEED_BASE, layer_idx):
                self.pkm_q = Linear(D, H * dq, device=device, dtype=dtype, init=li)
                self.pkm_g = Linear(D, dv, device=device, dtype=dtype, init=li)
                self.pkm_o = Linear(dv, D, device=device, dtype=dtype, init=li)
                nn.init.xavier_uniform_(self.pkm_q.weight)
                nn.init.xavier_uniform_(self.pkm_g.weight)
                self.pkm_o.weight.normal_(std=float(config.out_std))
                self.pkm_values = nn.Parameter(torch.empty((n * n, dv), device=device, dtype=f32).normal_(
                    mean=0.0, std=dv ** -0.5))
                self.pkm_subkeys = nn.Parameter(torch.empty((H, 2, n, dq // 2), device=device, dtype=f32).normal_(
                    mean=0.0, std=1.0))
            self.pkm_norm_gain = nn.Parameter(torch.zeros(dv, device=device, dtype=f32))
            self.pkm_log_gain = nn.Parameter(torch.zeros(2 * H, device=device, dtype=f32))
        else:
            self.pkm_q = Linear(D, H * dq, device=device, dtype=dtype, init=li)
            self.pkm_g = Linear(D, dv, device=device, dtype=dtype, init=li)
            self.pkm_o = Linear(dv, D, device=device, dtype=dtype, init=li)
            self.pkm_values = nn.Parameter(torch.empty((n * n, dv), device=device, dtype=f32))
            self.pkm_subkeys = nn.Parameter(torch.empty((H, 2, n, dq // 2), device=device, dtype=f32))
            self.pkm_norm_gain = nn.Parameter(torch.empty(dv, device=device, dtype=f32))
            self.pkm_log_gain = nn.Parameter(torch.empty(2 * H, device=device, dtype=f32))
        scalar = dict(device=device, dtype=torch.float32)
        self.register_buffer("pkm_copy_strength",
                             torch.tensor(float(config.memory_pkm_copy_gate_strength), **scalar))
        self.register_buffer("pkm_copy_tau", torch.ones((), **scalar))
        self.register_buffer("pkm_calibrated", torch.zeros((), **scalar))
        self.register_buffer("pkm_launch_rms_res", torch.zeros((), **scalar))
        self.register_buffer("pkm_launch_rms_y", torch.zeros((), **scalar))
        self.register_buffer("pkm_launch_entropy", torch.zeros((), **scalar))
        self._pkm_ready = False

    # ------------------------------------------------------------------------------------------------ readiness
    @property
    def pkm_ready(self) -> bool:
        return bool(self._pkm_ready)

    def mark_pkm_ready(self) -> None:
        """Set by the launch calibration once the PKM buffers are written."""
        self._pkm_ready = True

    def _require_pkm_ready(self) -> None:
        if self._pkm_ready:
            return
        if float(self.pkm_calibrated) > 0.5:
            self._pkm_ready = True
            return
        raise RuntimeError("memory layer: the product-key memory was neither calibrated (neuronspark.utils.calibration)"
                           " nor loaded from a calibrated checkpoint (pkm_calibrated = 0)")

    # ------------------------------------------------------------------------------------------------ shared pieces
    def _pkm_queries(self, q: torch.Tensor, T: int, B: int) -> torch.Tensor:
        """(T, B, H, 2, dh) fp32 unit half-queries (autocast disabled by the caller)."""
        return F.normalize(q.to(torch.float32).view(T, B, self.pkm_heads, 2, self.pkm_half), dim=-1)

    def _pkm_gains(self) -> torch.Tensor:
        """(H, 2) fp32 beta = sqrt(dh) exp(pkm_log_gain)."""
        return math.sqrt(self.pkm_half) * torch.exp(self.pkm_log_gain.to(torch.float32)).view(self.pkm_heads, 2)

    def _pkm_raw_scores(self, qh: torch.Tensor) -> torch.Tensor:
        """(T, B, H, 2, n) fp32 cosines of the unit half-queries with the unit sub-keys (before the gain).  With autograd
        the einsum's backward runs under the TF32 setting of this call (:func:`regime_product`)."""
        kh = F.normalize(self.pkm_subkeys.to(torch.float32), dim=-1)
        return regime_product(_half_query_scores, qh, kh)

    def _pkm_out(self, r: torch.Tensor, x_hat: torch.Tensor) -> torch.Tensor:
        return self._ops.linear(r.to(x_hat.dtype), self.pkm_o.weight)

    # ------------------------------------------------------------------------------------------------ whole row
    def _pkm_read_train(self, x_hat: torch.Tensor, res_attn: torch.Tensor, calib=None,
                        mode: Optional[str] = None) -> Optional[torch.Tensor]:
        """Whole-row read y (T, B, D) bf16, or None when nothing is added (a launch-calibration call: an unready layer,
        or a layer captured by the calibration's ``mode``)."""
        if mode not in PKM_CALIB_MODES:
            raise RuntimeError(f"PKM: unknown calibration mode {mode!r}")
        if mode is None and not self._pkm_ready:
            if calib is not None:
                return None                                  # an unready layer inside a calibration: nothing added
            self._require_pkm_ready()
        o = self._ops
        T, B, D = (int(s) for s in x_hat.shape)
        H, k = self.pkm_heads, self.pkm_k
        xr = x_hat.reshape(T * B, D)
        q = o.linear(xr, self.pkm_q.weight).view(T, B, -1)
        if mode == "queries":
            with torch.autocast(x_hat.device.type, enabled=False):
                qh = F.normalize(q.to(torch.float32).view(T * B, H, 2, self.pkm_half), dim=-1)
            calib.record(self, "pkm_queries", qh.detach().to(device="cpu", dtype=torch.float32))
            return None
        glog = o.linear(xr, self.pkm_g.weight).view(T, B, -1)
        with torch.autocast(x_hat.device.type, enabled=False):
            qh = self._pkm_queries(q, T, B)
            s = self._pkm_raw_scores(qh) * self._pkm_gains().view(1, 1, H, 2, 1)
            sc, u, ii, jj = o.pkm_topk_train(s, k)
            w = torch.softmax(sc, dim=-1)
            val = o.pkm_value_read(w.reshape(T, B, H * k), u.reshape(T, B, H * k), self.pkm_values)
            r = val * torch.rsqrt(val.pow(2).mean(dim=-1, keepdim=True) + PKM_EPS) * (
                1.0 + self.pkm_norm_gain.to(torch.float32))
            r = r * torch.sigmoid(glog.to(torch.float32))
        y = self._pkm_out(r, x_hat)
        if mode == "rms":
            calib.record(self, "pkm_usage", (u.detach(), w.detach(), ii.detach(), jj.detach()))
            yc = y.detach().float()
            yc = yc - yc.mean(dim=-1, keepdim=True)
            calib.record(self, "ms_y", yc.pow(2).mean(dim=-1).reshape(-1).double().cpu())
            calib.record(self, "ms_res", res_attn.detach().float().pow(2).mean(dim=-1).reshape(-1).double().cpu())
            return None
        return y

    # ------------------------------------------------------------------------------------------------ prefill
    def _pkm_read_prefill(self, x_hat: torch.Tensor) -> torch.Tensor:
        """Prefill read (T >= 2, no autograd) y (T, B, D) bf16."""
        self._require_pkm_ready()
        o = self._ops
        T, B, _ = (int(s) for s in x_hat.shape)
        H, k = self.pkm_heads, self.pkm_k
        q = o.linear(x_hat, self.pkm_q.weight)
        glog = o.linear(x_hat, self.pkm_g.weight)
        with torch.autocast(x_hat.device.type, enabled=False):
            qh = self._pkm_queries(q, T, B)
            s = self._pkm_raw_scores(qh) * self._pkm_gains().view(1, 1, H, 2, 1)
            sc, u, ii, jj = o.pkm_topk_prefill(s[..., 0, :], s[..., 1, :], k)
            w = torch.softmax(sc, dim=-1)
            val = o.pkm_weighted_sum(w.reshape(T, B, H * k), u.reshape(T, B, H * k), self.pkm_values)
            r = rows_rmsnorm(o, val, PKM_EPS, weight=self.pkm_norm_gain, zero_centered=True,
                             out_dtype=torch.float32)
            r = r * torch.sigmoid(glog.to(torch.float32))
        return self._pkm_out(r, x_hat)

    # ------------------------------------------------------------------------------------------------ decode
    def _pkm_read_step(self, x_hat: torch.Tensor) -> torch.Tensor:
        """Single-token read (T == 1, no autograd) y (1, B, D) bf16."""
        self._require_pkm_ready()
        o = self._ops
        T, B, _ = (int(s) for s in x_hat.shape)
        H, k = self.pkm_heads, self.pkm_k
        q = o.linear(x_hat, self.pkm_q.weight)
        glog = o.linear(x_hat, self.pkm_g.weight)
        with torch.autocast(x_hat.device.type, enabled=False):
            qh = self._pkm_queries(q, T, B)
            s_raw = self._pkm_raw_scores(qh)
            sc, w, u, ii, jj = o.pkm_topk_softmax_t1(s_raw[0], self._pkm_gains().contiguous(), k)
            w, u = w.unsqueeze(0), u.unsqueeze(0)
            vals = F.embedding(u.reshape(T, B, H * k), self.pkm_values)
            val = torch.einsum("tbm,tbmd->tbd", w.reshape(T, B, H * k), vals.to(torch.float32))
            r = val * torch.rsqrt(val.pow(2).mean(dim=-1, keepdim=True) + PKM_EPS) * (
                1.0 + self.pkm_norm_gain.to(torch.float32))
            r = r * torch.sigmoid(glog.to(torch.float32))
        return self._pkm_out(r, x_hat)
