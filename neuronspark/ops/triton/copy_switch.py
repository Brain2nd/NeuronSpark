"""Copy-switch auxiliary loss: the supervised training term of the hippocampal layers' novelty predictor.

For token ids x (B, T) and pre-shifted next-token targets y (B, T) (``-100`` ignored), the copy label of position t is

    label[t] = any_{j < t} (x_j == x_t and y_j == y_t and y_j != -100) and y_t != -100,   valid[t] = y_t != -100

(the next token is an induction copy of an earlier continuation).  For each hippocampal layer l with the detached
predictor inputs z (T, B, D) (the layer's normed input), r (T, B, D) (its attention residual) and pi (T, B) (the copy
detector's confidence) and its parameters w, v (D,) and b (1,):

    logit = (z . w + b) + rsqrt(mean(r^2) + 1e-6) (r . v)
    term_l = sum_t,b wt bce(logit, label) / max(sum wt, 1),   wt = pi * valid
    aux    = lam * mean_l term_l                              (lam = 0.5 in the default configuration)

with bce(x, y) = (1 - y) x - log_sigmoid(x).  The gradient reaches w, v and b of every layer only.

Kernels (one autograd Function, :class:`CopySwitchAuxFn`; custom ops ``neuronspark::copy_switch_fwd`` /
``neuronspark::copy_switch_bwd``):

* labels: one program per (64 positions, row), the j < t scan in blocks of 64; stored as fp32 0 / 1 at t * B + b;
* forward, per layer: 16 rows per program, columns in chunks of 512, the three row dots (z . w, r . r, r . v) as fp32
  tree sums, libdevice rsqrt / exp / log1p, per-program partial sums of wt bce and wt (deterministic); one finishing
  kernel: term_l = sum / max(den, 1), aux = lam * (sum_l term_l) / L in a fixed order;
* backward, per layer: dl = (sigmoid(logit) - label) (wt g lam / L / den_l) per row; column partial sums over 128-row
  chunks of z^T dl, r^T (rs dl) and sum dl, then one fixed-order reduction of the partials (no atomics).

Numerics: the labels are exact; aux / dw / dv / db sum in other orders than the eager chain (cuBLAS products, ATen
reductions), with rs factored out of rf . v and libdevice routines for the ATen functors (fp32 reduction-order class).
The call site stays inside the training autocast region (the kernels do not depend on it).

Pure-torch references (same signatures): :func:`copy_switch_aux_ref` (the labels, the detached predictor logit, the
weighted cross entropy) and :func:`copy_switch_labels_ref` ((B, T, T) booleans).  The kernels are checked against them
with the per-operator tolerance rule of :mod:`neuronspark.utils.testing`.  The per-layer operands are passed as
explicit ``feats`` / ``weights`` lists.
"""
from __future__ import annotations

from typing import List, Sequence, Tuple

import torch
import torch.nn.functional as F
import triton
import triton.language as tl
from triton.language.extra import libdevice as _ld

from neuronspark.ops.triton import require_cuda

__all__ = [
    "LAM",
    "CopySwitchAuxFn",
    "copy_switch_aux",
    "copy_switch_aux_ref",
    "copy_switch_labels",
    "copy_switch_labels_ref",
]

# ================================================================================================ fixed constants
#: weight of the auxiliary term (the model's hippocampal_copy_novelty_gate_aux)
LAM = 0.5
#: eps of the residual's RMS
EPS = 1e-6
#: labels kernel: positions per program, block of earlier positions j, warps
LAB_BT, LAB_BJ, LAB_WARPS = 64, 64, 4
#: forward: rows per program, column chunk, warps
BR_F, BD_F, WARPS_F = 16, 512, 4
#: backward: rows per chunk, row block, column block, warps
RCH_B, BR_B, BD_B, WARPS_B = 128, 32, 128, 4
#: backward reduction: columns per program, warps; finishing kernel warps
BD_R, WARPS_R = 128, 4
WARPS_FINAL = 4


def _need(cond: bool, msg: str) -> None:
    if not cond:
        raise RuntimeError("neuronspark.ops.copy_switch: " + msg)


def _pow2(n: int) -> int:
    p = 1
    while p < n:
        p *= 2
    return p


# ================================================================================================ launchers
def _labels(token_ids: torch.Tensor, target_ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """(label, valid) as fp32 (T * B,) buffers, index t * B + b."""
    require_cuda(token_ids, "copy_switch_labels")
    _need(token_ids.dim() == 2 and tuple(token_ids.shape) == tuple(target_ids.shape)
          and token_ids.device == target_ids.device,
          f"token / target ids must be two (B, T) tensors on one device, got {tuple(token_ids.shape)} / "
          f"{tuple(target_ids.shape)}")
    _need(not token_ids.is_floating_point() and not target_ids.is_floating_point(), "integer token / target ids")
    Bsz, T = int(token_ids.shape[0]), int(token_ids.shape[1])
    yo = torch.empty(T * Bsz, device=token_ids.device, dtype=torch.float32)
    vo = torch.empty(T * Bsz, device=token_ids.device, dtype=torch.float32)
    if T * Bsz:
        _cs_labels_kernel[(triton.cdiv(T, LAB_BT), Bsz)](token_ids, target_ids, yo, vo, T, Bsz,
                                                         token_ids.stride(0), token_ids.stride(1),
                                                         target_ids.stride(0), target_ids.stride(1),
                                                         BT=LAB_BT, BJ=LAB_BJ, num_warps=LAB_WARPS)
    return yo, vo


def _check_layers(T: int, Bsz: int, dev: torch.device, zs, rs, pis, ws, vs, bs) -> int:
    L = len(zs)
    _need(L >= 1 and all(len(t) == L for t in (rs, pis, ws, vs, bs)),
          "one (z, r, pi, w, v, b) set per hippocampal layer")
    D = int(zs[0].shape[-1])
    for li in range(L):
        z, r, pi, w, v, b = zs[li], rs[li], pis[li], ws[li], vs[li], bs[li]
        _need(z.dim() == 3 and tuple(z.shape) == (T, Bsz, D) and tuple(r.shape) == (T, Bsz, D)
              and tuple(pi.shape) == (T, Bsz) and z.device == dev and r.device == dev and pi.device == dev,
              f"layer {li}: operands z {tuple(z.shape)} r {tuple(r.shape)} pi {tuple(pi.shape)} do not match "
              f"(T, B, D) = ({T}, {Bsz}, {D}) on {dev}")
        _need(w.dtype == torch.float32 and v.dtype == torch.float32 and b.dtype == torch.float32
              and tuple(w.shape) == (D,) and tuple(v.shape) == (D,) and b.numel() == 1
              and w.is_contiguous() and v.is_contiguous() and w.device == dev and v.device == dev
              and b.device == dev,
              f"layer {li}: the predictor's w / v ({D},) and b (1,) must be contiguous fp32 on {dev}")
    return D


def _copy_switch_fwd(token_ids, target_ids, zs, rs, pis, ws, vs, bs, lam: float):
    """(aux () fp32, label, valid, logit (L, N), rs (L, N), den (L,)) with N = T * B."""
    Bsz, T = int(token_ids.shape[0]), int(token_ids.shape[1])
    N = T * Bsz
    dev = token_ids.device
    yo, vo = _labels(token_ids, target_ids)
    D = _check_layers(T, Bsz, dev, zs, rs, pis, ws, vs, bs)
    L = len(zs)
    NP = triton.cdiv(N, BR_F)
    logit = torch.empty((L, N), device=dev, dtype=torch.float32)
    rso = torch.empty((L, N), device=dev, dtype=torch.float32)
    part = torch.empty((L, NP, 2), device=dev, dtype=torch.float32)
    for li in range(L):
        z, r, pi = zs[li], rs[li], pis[li]
        _cs_fwd_kernel[(NP,)](z, r, pi, ws[li], vs[li], bs[li], yo, vo, logit[li], rso[li], part[li], N, Bsz, D,
                              z.stride(0), z.stride(1), z.stride(2), r.stride(0), r.stride(1), r.stride(2),
                              pi.stride(0), pi.stride(1), 1.0 / D, EPS, BR=BR_F, BD=BD_F, num_warps=WARPS_F)
    terms = torch.empty(L, device=dev, dtype=torch.float32)
    dens = torch.empty(L, device=dev, dtype=torch.float32)
    out = torch.empty((), device=dev, dtype=torch.float32)
    _cs_final_kernel[(1,)](part, NP, terms, dens, out, float(lam), 1.0 / L, L=L, BP=_pow2(NP), num_warps=WARPS_FINAL)
    return out, yo, vo, logit, rso, dens


def _copy_switch_bwd(gout, yo, vo, logit, rso, dens, zs, rs, pis, lam: float) -> List[torch.Tensor]:
    """[dw_0, dv_0, db_0, dw_1, ...] fp32 for the upstream gradient of aux."""
    L = len(zs)
    T, Bsz, D = (int(s) for s in zs[0].shape)
    N = T * Bsz
    g = gout.detach().to(torch.float32).contiguous()
    NCH = triton.cdiv(N, RCH_B)
    grads: List[torch.Tensor] = []
    for li in range(L):
        z, r, pi = zs[li], rs[li], pis[li]
        pw = torch.empty((NCH, D), device=z.device, dtype=torch.float32)
        pv = torch.empty((NCH, D), device=z.device, dtype=torch.float32)
        pb = torch.empty((NCH,), device=z.device, dtype=torch.float32)
        _cs_bwd_kernel[(triton.cdiv(D, BD_B), NCH)](z, r, pi, yo, vo, logit[li], rso[li], dens, g, pw, pv, pb, li,
                                                   N, Bsz, D, z.stride(0), z.stride(1), z.stride(2),
                                                   r.stride(0), r.stride(1), r.stride(2), pi.stride(0),
                                                   pi.stride(1), lam / L, RCH=RCH_B, BR=BR_B, BD=BD_B,
                                                   num_warps=WARPS_B)
        dw = torch.empty((D,), device=z.device, dtype=torch.float32)
        dv = torch.empty((D,), device=z.device, dtype=torch.float32)
        db = torch.empty((1,), device=z.device, dtype=torch.float32)
        _cs_bwd_reduce_kernel[(triton.cdiv(D, BD_R),)](pw, pv, pb, NCH, D, dw, dv, db, BCH=_pow2(NCH), BD=BD_R,
                                                      num_warps=WARPS_R)
        grads += [dw, dv, db]
    return grads


@torch.library.custom_op("neuronspark::copy_switch_labels", mutates_args=())
def _copy_switch_labels_op(token_ids: torch.Tensor, target_ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    return _labels(token_ids, target_ids)


@_copy_switch_labels_op.register_fake
def _copy_switch_labels_fake(token_ids, target_ids):
    n = token_ids.shape[0] * token_ids.shape[1]
    return token_ids.new_empty((n,), dtype=torch.float32), token_ids.new_empty((n,), dtype=torch.float32)


@torch.library.custom_op("neuronspark::copy_switch_fwd", mutates_args=())
def _copy_switch_fwd_op(token_ids: torch.Tensor, target_ids: torch.Tensor, zs: List[torch.Tensor],
                        rs: List[torch.Tensor], pis: List[torch.Tensor], ws: List[torch.Tensor],
                        vs: List[torch.Tensor], bs: List[torch.Tensor], lam: float
                        ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,
                                   torch.Tensor]:
    return _copy_switch_fwd(token_ids, target_ids, zs, rs, pis, ws, vs, bs, lam)


@_copy_switch_fwd_op.register_fake
def _copy_switch_fwd_fake(token_ids, target_ids, zs, rs, pis, ws, vs, bs, lam):
    n = token_ids.shape[0] * token_ids.shape[1]
    L = len(zs)
    f = torch.float32
    return (token_ids.new_empty((), dtype=f), token_ids.new_empty((n,), dtype=f), token_ids.new_empty((n,), dtype=f),
            token_ids.new_empty((L, n), dtype=f), token_ids.new_empty((L, n), dtype=f),
            token_ids.new_empty((L,), dtype=f))


@torch.library.custom_op("neuronspark::copy_switch_bwd", mutates_args=())
def _copy_switch_bwd_op(gout: torch.Tensor, yo: torch.Tensor, vo: torch.Tensor, logit: torch.Tensor,
                        rso: torch.Tensor, dens: torch.Tensor, zs: List[torch.Tensor], rs: List[torch.Tensor],
                        pis: List[torch.Tensor], lam: float) -> List[torch.Tensor]:
    return _copy_switch_bwd(gout, yo, vo, logit, rso, dens, zs, rs, pis, lam)


@_copy_switch_bwd_op.register_fake
def _copy_switch_bwd_fake(gout, yo, vo, logit, rso, dens, zs, rs, pis, lam):
    out = []
    for z in zs:
        D = z.shape[-1]
        out += [z.new_empty((D,), dtype=torch.float32), z.new_empty((D,), dtype=torch.float32),
                z.new_empty((1,), dtype=torch.float32)]
    return out


# ================================================================================================ autograd Function
class CopySwitchAuxFn(torch.autograd.Function):
    """lam * mean_l(sum(wt bce(logit_l, label)) / max(sum wt, 1)); ``flat`` = (z, r, pi, w, v, b) per layer, z / r / pi
    detached; gradients to every layer's w, v, b only."""

    @staticmethod
    def forward(ctx, lam: float, token_ids, target_ids, *flat):
        _need(len(flat) % 6 == 0 and len(flat) >= 6, "six operands (z, r, pi, w, v, b) per layer")
        zs, rs, pis = list(flat[0::6]), list(flat[1::6]), list(flat[2::6])
        ws, vs, bs = list(flat[3::6]), list(flat[4::6]), list(flat[5::6])
        out, yo, vo, logit, rso, dens = torch.ops.neuronspark.copy_switch_fwd(token_ids, target_ids, zs, rs, pis, ws,
                                                                              vs, bs, float(lam))
        ctx.save_for_backward(yo, vo, logit, rso, dens, *zs, *rs, *pis)
        ctx.meta = (float(lam), len(zs))
        return out

    @staticmethod
    def backward(ctx, gout):
        lam, L = ctx.meta
        yo, vo, logit, rso, dens, *saved = ctx.saved_tensors
        zs, rs, pis = saved[0:L], saved[L:2 * L], saved[2 * L:3 * L]
        g = torch.ops.neuronspark.copy_switch_bwd(gout, yo, vo, logit, rso, dens, list(zs), list(rs), list(pis), lam)
        grads = []
        for li in range(L):
            grads += [None, None, None, g[3 * li], g[3 * li + 1], g[3 * li + 2]]
        return (None, None, None, *grads)


# ================================================================================================ public functions
def _flat(feats: Sequence[Sequence[torch.Tensor]], weights: Sequence[Sequence[torch.Tensor]]) -> List[torch.Tensor]:
    _need(len(feats) == len(weights) and len(feats) >= 1,
          f"one (z, r, pi) and one (w, v, b) per hippocampal layer, got {len(feats)} / {len(weights)}")
    flat: List[torch.Tensor] = []
    for li, (f, wt) in enumerate(zip(feats, weights)):
        _need(len(f) == 3 and len(wt) == 3, f"layer {li}: feats (z, r, pi) and weights (w, v, b) expected")
        z, r, pi = f
        w, v, b = wt
        flat += [z.detach(), r.detach(), pi.detach(), w, v, b]
    return flat


def copy_switch_aux(ids: torch.Tensor, targets: torch.Tensor, feats: Sequence[Sequence[torch.Tensor]],
                    weights: Sequence[Sequence[torch.Tensor]], lam: float = LAM) -> torch.Tensor:
    """The copy-switch auxiliary loss, a () fp32 tensor.

    ``ids`` / ``targets``: (B, T) int64 token ids and pre-shifted next-token targets (-100 ignored); ``feats``: per
    hippocampal layer (z (T, B, D), r (T, B, D), pi (T, B)) (detached here); ``weights``: per hippocampal layer the
    predictor's
    (w (D,), v (D,), b (1,)) fp32 contiguous.  Differentiable in the weights."""
    return CopySwitchAuxFn.apply(float(lam), ids, targets, *_flat(feats, weights))


def copy_switch_labels(ids: torch.Tensor, targets: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """(label (T, B) fp32 0 / 1, valid (T, B) bool) of the copy switch, by the labels kernel."""
    Bsz, T = int(ids.shape[0]), int(ids.shape[1])
    yo, vo = torch.ops.neuronspark.copy_switch_labels(ids, targets)
    return yo.view(T, Bsz), vo.view(T, Bsz) != 0


# ================================================================================================ pure-torch references
def copy_switch_labels_ref(ids: torch.Tensor, targets: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Pure-torch reference of :func:`copy_switch_labels` ((B, T, T) booleans)."""
    x, tg = ids, targets
    B, T = x.shape
    past = torch.ones(T, T, device=x.device, dtype=torch.bool).tril(-1)
    eq_tok = x.unsqueeze(2) == x.unsqueeze(1)
    eq_nxt = tg.unsqueeze(2) == tg.unsqueeze(1)
    valid = tg != -100
    y = (eq_tok & eq_nxt & past.unsqueeze(0) & valid.unsqueeze(2) & valid.unsqueeze(1)).any(-1)
    return y.transpose(0, 1).to(torch.float32), valid.transpose(0, 1)


def copy_switch_aux_ref(ids: torch.Tensor, targets: torch.Tensor, feats: Sequence[Sequence[torch.Tensor]],
                        weights: Sequence[Sequence[torch.Tensor]], lam: float = LAM) -> torch.Tensor:
    """Pure-torch reference of :func:`copy_switch_aux`: the labels, the detached predictor logit on the normed input z
    and the attention residual r, and the pi-weighted binary cross entropy per layer, outside autocast."""
    y, valid = copy_switch_labels_ref(ids, targets)
    terms = []
    for (z, r, pi), (w, v, b) in zip(feats, weights):
        with torch.autocast(z.device.type, enabled=False):
            logit = z.detach().float() @ w.float() + b.float()
            rf = r.detach().float()
            rf = rf * torch.rsqrt(rf.pow(2).mean(dim=-1, keepdim=True) + EPS)
            logit = logit + rf @ v.float()
            wt = pi.detach().float() * valid.to(torch.float32)
            den = wt.sum().clamp(min=1.0)
            bce = F.binary_cross_entropy_with_logits(logit, y, reduction="none")
            terms.append((wt * bce).sum() / den)
    _need(len(terms) >= 1, "at least one hippocampal layer")
    return lam * torch.stack(terms).mean()


# ================================================================================================ kernels

@triton.jit
def _cs_labels_kernel(X, Y, YO, VO, T, B, xs0, xs1, ys0, ys1, BT: tl.constexpr, BJ: tl.constexpr):
    """y[t, b] = any_{j < t}(X[b, j] == X[b, t] and Y[b, j] == Y[b, t] and Y[b, j] != -100) and Y[b, t] != -100;
    valid[t, b] = Y[b, t] != -100; stored at t * B + b as fp32 0 / 1."""
    pid_t = tl.program_id(0)
    b = tl.program_id(1).to(tl.int64)
    t = pid_t.to(tl.int64) * BT + tl.arange(0, BT).to(tl.int64)
    tm = t < T
    xt = tl.load(X + b * xs0 + t * xs1, mask=tm, other=0)
    yt = tl.load(Y + b * ys0 + t * ys1, mask=tm, other=-100)
    vt = yt != -100
    hit = tl.zeros((BT,), dtype=tl.int32)
    t_hi = tl.minimum((pid_t + 1) * BT, T)
    for j0 in range(0, t_hi, BJ):
        j = j0 + tl.arange(0, BJ).to(tl.int64)
        jm = j < T
        xj = tl.load(X + b * xs0 + j * xs1, mask=jm, other=0)
        yj = tl.load(Y + b * ys0 + j * ys1, mask=jm, other=-100)
        m = ((xj[None, :] == xt[:, None]) & (yj[None, :] == yt[:, None]) & (j[None, :] < t[:, None])
             & (yj[None, :] != -100) & jm[None, :])
        hit = tl.maximum(hit, tl.max(m.to(tl.int32), axis=1))
    y = (hit > 0) & vt
    o = t * B + b
    tl.store(YO + o, y.to(tl.float32), mask=tm)
    tl.store(VO + o, vt.to(tl.float32), mask=tm)


@triton.jit
def _cs_fwd_kernel(Z, R, PI, W, V, BIAS, YO, VO, LOGIT, RSO, PART, NROWS, B, D,
                   zs0, zs1, zs2, rs0, rs1, rs2, ps0, ps1, inv_d, eps,
                   BR: tl.constexpr, BD: tl.constexpr):
    """One hippocampal layer, rows n = t * B + b: logit = (z . w + b) + rsqrt(mean(r^2) + eps) (r . v); per-program
    partial
    sums of wt * bce(logit, y) and wt (wt = pi * valid) into PART[pid, 0 / 1]; logit and rs per row for the backward."""
    pid = tl.program_id(0)
    n = pid.to(tl.int64) * BR + tl.arange(0, BR).to(tl.int64)
    nm = n < NROWS
    t = n // B
    b = n - t * B
    acc_a = tl.zeros((BR,), dtype=tl.float32)
    acc_ss = tl.zeros((BR,), dtype=tl.float32)
    acc_rv = tl.zeros((BR,), dtype=tl.float32)
    for d0 in range(0, D, BD):
        c = d0 + tl.arange(0, BD).to(tl.int64)
        cm = c < D
        m = nm[:, None] & cm[None, :]
        z = tl.load(Z + t[:, None] * zs0 + b[:, None] * zs1 + c[None, :] * zs2, mask=m, other=0.0).to(tl.float32)
        r = tl.load(R + t[:, None] * rs0 + b[:, None] * rs1 + c[None, :] * rs2, mask=m, other=0.0).to(tl.float32)
        w = tl.load(W + c, mask=cm, other=0.0)
        v = tl.load(V + c, mask=cm, other=0.0)
        acc_a += tl.sum(z * w[None, :], axis=1)
        acc_ss += tl.sum(r * r, axis=1)
        acc_rv += tl.sum(r * v[None, :], axis=1)
    rsq = _ld.rsqrt(acc_ss * inv_d + eps)
    logit = (acc_a + tl.load(BIAS)) + rsq * acc_rv
    y = tl.load(YO + n, mask=nm, other=0.0)
    vv = tl.load(VO + n, mask=nm, other=0.0)
    pi = tl.load(PI + t * ps0 + b * ps1, mask=nm, other=0.0).to(tl.float32)
    wt = pi * vv
    ls = tl.minimum(logit, 0.0) - _ld.log1p(_ld.exp(-tl.abs(logit)))
    bce = (1.0 - y) * logit - ls
    tl.store(LOGIT + n, logit, mask=nm)
    tl.store(RSO + n, rsq, mask=nm)
    tl.store(PART + pid * 2, tl.sum(tl.where(nm, wt * bce, 0.0), axis=0))
    tl.store(PART + pid * 2 + 1, tl.sum(tl.where(nm, wt, 0.0), axis=0))


@triton.jit
def _cs_final_kernel(PART, NP, TERMS, DENS, OUT, lam, inv_l, L: tl.constexpr, BP: tl.constexpr):
    """term_l = sum_p PART[l, p, 0] / max(sum_p PART[l, p, 1], 1); OUT = lam * (sum_l term_l) / L (fixed order)."""
    p = tl.arange(0, BP)
    pm = p < NP
    tot = tl.sum(tl.zeros((BP,), dtype=tl.float32), axis=0)
    for l in tl.static_range(L):
        s1 = tl.sum(tl.load(PART + (l * NP + p) * 2, mask=pm, other=0.0), axis=0)
        s0 = tl.sum(tl.load(PART + (l * NP + p) * 2 + 1, mask=pm, other=0.0), axis=0)
        den = tl.maximum(s0, 1.0)
        term = s1 / den
        tl.store(TERMS + l, term)
        tl.store(DENS + l, den)
        tot = tot + term
    tl.store(OUT, lam * (tot * inv_l))


@triton.jit
def _cs_bwd_kernel(Z, R, PI, YO, VO, LOGIT, RSO, DENS, GOUT, PW, PV, PB, layer, NROWS, B, D,
                   zs0, zs1, zs2, rs0, rs1, rs2, ps0, ps1, lam_inv_l,
                   RCH: tl.constexpr, BR: tl.constexpr, BD: tl.constexpr):
    """Column block cb, row chunk ch: PW[ch, c] = sum_n z[n, c] dl[n], PV[ch, c] = sum_n r[n, c] rs[n] dl[n],
    PB[ch] = sum_n dl[n] (cb == 0), dl = (sigmoid(logit) - y) (wt (g lam / L / den))."""
    cb = tl.program_id(0)
    ch = tl.program_id(1)
    c = cb.to(tl.int64) * BD + tl.arange(0, BD).to(tl.int64)
    cm = c < D
    q = (tl.load(GOUT).to(tl.float32) * lam_inv_l) / tl.load(DENS + layer)
    acc_w = tl.zeros((BD,), dtype=tl.float32)
    acc_v = tl.zeros((BD,), dtype=tl.float32)
    acc_b = tl.zeros((BR,), dtype=tl.float32)
    for r0 in range(0, RCH, BR):
        n = ch.to(tl.int64) * RCH + r0 + tl.arange(0, BR).to(tl.int64)
        nm = n < NROWS
        t = n // B
        b = n - t * B
        x = tl.load(LOGIT + n, mask=nm, other=0.0)
        y = tl.load(YO + n, mask=nm, other=0.0)
        vv = tl.load(VO + n, mask=nm, other=0.0)
        pi = tl.load(PI + t * ps0 + b * ps1, mask=nm, other=0.0).to(tl.float32)
        rsq = tl.load(RSO + n, mask=nm, other=0.0)
        sig = 1.0 / (1.0 + _ld.exp(-x))
        dl = tl.where(nm, (sig - y) * ((pi * vv) * q), 0.0)
        m = nm[:, None] & cm[None, :]
        z = tl.load(Z + t[:, None] * zs0 + b[:, None] * zs1 + c[None, :] * zs2, mask=m, other=0.0).to(tl.float32)
        r = tl.load(R + t[:, None] * rs0 + b[:, None] * rs1 + c[None, :] * rs2, mask=m, other=0.0).to(tl.float32)
        acc_w += tl.sum(z * dl[:, None], axis=0)
        acc_v += tl.sum(r * (rsq * dl)[:, None], axis=0)
        acc_b += dl
    tl.store(PW + ch * D + c, acc_w, mask=cm)
    tl.store(PV + ch * D + c, acc_v, mask=cm)
    if cb == 0:
        tl.store(PB + ch, tl.sum(acc_b, axis=0))


@triton.jit
def _cs_bwd_reduce_kernel(PW, PV, PB, NCH, D, DW, DV, DB, BCH: tl.constexpr, BD: tl.constexpr):
    """DW[c] = sum_ch PW[ch, c], DV[c] = sum_ch PV[ch, c], DB = sum_ch PB[ch] (one tree per output, fixed order)."""
    pid = tl.program_id(0)
    c = pid.to(tl.int64) * BD + tl.arange(0, BD).to(tl.int64)
    cm = c < D
    k = tl.arange(0, BCH).to(tl.int64)
    km = k < NCH
    m = km[:, None] & cm[None, :]
    tl.store(DW + c, tl.sum(tl.load(PW + k[:, None] * D + c[None, :], mask=m, other=0.0), axis=0), mask=cm)
    tl.store(DV + c, tl.sum(tl.load(PV + k[:, None] * D + c[None, :], mask=m, other=0.0), axis=0), mask=cm)
    if pid == 0:
        tl.store(DB, tl.sum(tl.load(PB + k, mask=km, other=0.0), axis=0))
