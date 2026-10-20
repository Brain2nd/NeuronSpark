"""Gluon backward of the training PLIF population (:func:`neuronspark.ops.triton.plif.plif_scan_train`).

One kernel per launch computes, for every lane (b, c) walking the tokens from t = T - 1 down in tiles of 16 (the next
two tiles' loads issued ahead), the adjoint of the bipolar soft-reset scan and the backward of its gate chain:

    phase 1  a = 1 - eta; gp = sigmoid(alpha (pre - th)), gn = sigmoid(alpha (-pre - th)) (RN sigmoid);
             g_mem = gp + gn, g_th = gp - gn; a~ = a (1 - g_mem), c~ = a (g_mem gS)
    phase 2  the adjoint chain lam_{t-1} = a~_t lam_t + c~_t as an associative scan over the tile (combine
             x_cur = RN(RN(a~ x_next) + c~)); lam_t (= dL/dv_t) is the chain value before row t
    phase 3  dL/dpre = lam + g_mem (gS - lam); dL/dth += g_th (lam - gS) (per-lane sum over T);
             v_{t-1} from the stored pre_{t-1}; gA = dL/dpre v_{t-1}; g_eta = dL/dpre xw + (-gA);
             gz = (g_eta (1 - eta)) eta; gzp = bf16(gz); gx = dL/dpre eta (stored in x.dtype)

Every arithmetic step is one RN inline-PTX operation (helpers in :mod:`neuronspark.ops.gluon._ptx`).  The membrane
at t = -1 and the gradient of the last membrane are zero (whole rows), the kernel's HAS_V0 / HAS_GVLAST constexprs are
0.  Launch: 32 lanes per program, 4 warps, grid cdiv(B * C, 32); requires compute capability >= 10.0.

Public: :func:`plif_k1_bwd` (custom op ``neuronspark::plif_k1_bwd``) and the pure-torch :func:`plif_k1_bwd_ref`;
the kernel is checked against the reference with the per-operator tolerance rule of :mod:`neuronspark.utils.testing`.
"""
from __future__ import annotations

from typing import Tuple

import torch
import triton
from triton.experimental import gluon
from triton.experimental.gluon import language as gl

from neuronspark.ops.gluon._ptx import (  # noqa: F401  (resolved by the kernel below through this module's globals)
    _g_chain_combine,
    _g_pick_last,
    _g_rn_add,
    _g_rn_mul,
    _g_rn_sub,
    _g_sigmoid_rn,
)
from neuronspark.ops.triton import require_cuda, require_sm100
from neuronspark.ops.triton._ptx import gate_guard

__all__ = ["BWD_TT", "BWD_DEPTH", "BWD_WARPS", "BWD_LANES", "gate_slope_ok", "plif_k1_bwd", "plif_k1_bwd_ref"]

# ================================================================================================ fixed constants
# Backward launch constants: 16 tokens per tile, loads two tiles ahead, 4 warps, 32 lanes per program.
BWD_TT = 16
BWD_DEPTH = 2
BWD_WARPS = 4
BWD_LANES = 32


def _need(cond: bool, msg: str) -> None:
    if not cond:
        raise RuntimeError("neuronspark.ops.plif_k1_bwd: " + msg)


def gate_slope_ok(alpha: float) -> bool:
    """True when the kernel's surrogate form applies: a positive slope below the guarded-sigmoid threshold (64)."""
    return float(alpha) > 0.0 and gate_guard(float(alpha), "plain") == 0


def _tb_ok(t: torch.Tensor) -> bool:
    if t.dim() != 3 or int(t.stride(2)) != 1:
        return False
    T, B, C = (int(s) for s in t.shape)
    s0, s1 = int(t.stride(0)), int(t.stride(1))
    tmaj = s1 == C and s0 == B * C
    bmaj = s0 == C and s1 == T * C
    return (tmaj or bmaj or B == 1) and T * B * C < 2 ** 31


def _plif_k1_bwd(eta: torch.Tensor, x: torch.Tensor, v_th: torch.Tensor, pre: torch.Tensor, gS: torch.Tensor,
                 alpha: float) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """(gz fp32, gzp bf16, gx x.dtype, gth (B, C) fp32 per-lane sums over T), all contiguous."""
    require_cuda(pre, "plif_k1_bwd")
    _need(pre.dim() == 3, f"pre must be (T, B, C), got {tuple(pre.shape)}")
    T, B, C = (int(s) for s in pre.shape)
    _need(pre.dtype == torch.float32 and pre.is_contiguous() and T >= 2,
          "pre must be a contiguous fp32 (T >= 2, B, C) tensor")
    _need(eta.dtype == torch.float32 and tuple(eta.shape) == (T, B, C) and eta.is_contiguous(),
          "eta must be a contiguous fp32 tensor of pre's shape")
    _need(x.dtype in (torch.float32, torch.bfloat16) and tuple(x.shape) == (T, B, C) and _tb_ok(x),
          "x must be fp32 / bf16 (T, B, C) with channel stride 1 (T-major or a view of B-major storage)")
    _need(v_th.dtype == torch.float32 and tuple(v_th.shape) == (C,) and v_th.is_contiguous(),
          f"v_th must be a contiguous fp32 ({C},) tensor")
    _need(gS.dtype in (torch.float32, torch.bfloat16) and tuple(gS.shape) == (T, B, C) and gS.is_contiguous(),
          "gS must be a contiguous fp32 / bf16 tensor of pre's shape")
    _need(gate_slope_ok(alpha), f"the surrogate slope must satisfy 0 < alpha < 64, got {alpha}")
    for name, t in (("eta", eta), ("x", x), ("v_th", v_th), ("gS", gS)):
        _need(t.device == pre.device, f"{name} is on {t.device}, pre on {pre.device}")
    require_sm100(pre.device, "plif_k1_bwd")
    dev = pre.device
    gz = torch.empty((T, B, C), dtype=torch.float32, device=dev)
    gzp = torch.empty((T, B, C), dtype=torch.bfloat16, device=dev)
    gx = torch.empty((T, B, C), dtype=x.dtype, device=dev)
    gth = torch.empty((B, C), dtype=torch.float32, device=dev)
    gv0 = torch.empty((B, C), dtype=torch.float32, device=dev)          # written by the kernel, not returned
    _k1_bwd_tt16_nw4_d2_kernel[(triton.cdiv(B * C, BWD_LANES),)](
        eta, x, v_th, pre, gv0, gS, gv0, gz, gzp, gx, gth, gv0, float(alpha), T, B, C,
        int(x.stride(0)), int(x.stride(1)),
        BC=BWD_LANES, X_BF16=int(x.dtype == torch.bfloat16), HAS_V0=0, HAS_GVLAST=0, num_warps=BWD_WARPS)
    return gz, gzp, gx, gth


@torch.library.custom_op("neuronspark::plif_k1_bwd", mutates_args=())
def _plif_k1_bwd_op(eta: torch.Tensor, x: torch.Tensor, v_th: torch.Tensor, pre: torch.Tensor, gS: torch.Tensor,
                    alpha: float) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """The training backward launcher (the autograd formula of the PLIF Function calls it)."""
    return _plif_k1_bwd(eta, x, v_th, pre, gS, alpha)


@_plif_k1_bwd_op.register_fake
def _plif_k1_bwd_fake(eta, x, v_th, pre, gS, alpha):
    T, B, C = pre.shape
    return (pre.new_empty((T, B, C), dtype=torch.float32), pre.new_empty((T, B, C), dtype=torch.bfloat16),
            pre.new_empty((T, B, C), dtype=x.dtype), pre.new_empty((B, C), dtype=torch.float32))


def plif_k1_bwd(eta: torch.Tensor, x: torch.Tensor, v_th: torch.Tensor, pre: torch.Tensor, gS: torch.Tensor,
                alpha: float = 8.0) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Backward of the PLIF population scan + gate for the upstream spike gradient ``gS``.

    ``eta`` / ``pre``: contiguous fp32 (T, B, C) from the forward (the gate and the pre-release membrane); ``x``: the
    forward's input (fp32 / bf16, channel stride 1); ``v_th``: fp32 (C,); ``gS``: contiguous fp32 / bf16 (T, B, C).
    Returns (gz fp32 = dL/dz of the gate logit, gzp = bf16(gz), gx = the drive edge's dL/dx in x.dtype, gth (B, C) fp32
    = per-lane sums over T of dL/dth)."""
    return torch.ops.neuronspark.plif_k1_bwd(eta, x, v_th, pre, gS, float(alpha))


@torch.no_grad()
def plif_k1_bwd_ref(eta: torch.Tensor, x: torch.Tensor, v_th: torch.Tensor, pre: torch.Tensor, gS: torch.Tensor,
                    alpha: float = 8.0) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pure-torch reference of :func:`plif_k1_bwd`: the same recurrences in fp32, the adjoint chain sequential over t
    (the backward of the bipolar release surrogate, of the soft-reset scan and of the gate chain, in the form the kernel
    evaluates)."""
    f32 = torch.float32
    T, B, C = (int(s) for s in pre.shape)
    xw = x.to(f32)
    gs = gS.to(f32)
    th = v_th.to(f32)
    a = 1.0 - eta
    gate = torch.sigmoid((pre - th) * alpha)
    gate_neg = torch.sigmoid((-pre - th) * alpha)
    g_mem = gate + gate_neg
    g_th = gate - gate_neg
    a_tilde = a * (1.0 - g_mem)
    c_tilde = a * (g_mem * gs)
    s_all = torch.clamp_min(pre - th, 0.0) - torch.clamp_min(-pre - th, 0.0)
    v_all = pre - s_all
    lam = torch.zeros((B, C), dtype=f32, device=pre.device)
    grad_pre = torch.empty_like(pre)
    gth = torch.zeros((B, C), dtype=f32, device=pre.device)
    for t in range(T - 1, -1, -1):
        grad_pre[t] = lam + g_mem[t] * (gs[t] - lam)
        gth = gth + g_th[t] * (lam - gs[t])
        lam = a_tilde[t] * lam + c_tilde[t]
    v_prev = torch.cat([torch.zeros_like(v_all[:1]), v_all[:-1]], dim=0)
    ga = grad_pre * v_prev
    g_eta = grad_pre * xw + (-ga)
    gz = (g_eta * (1.0 - eta)) * eta
    gx = (grad_pre * eta).to(x.dtype)
    return gz, gz.to(torch.bfloat16), gx, gth


# ================================================================================================ kernels
# _k1_bwd_tt16_nw4_d2_kernel: TT 16 tokens per tile, NW 4 warps, DEPTH 2 tiles loaded ahead (the prefetch is written out
# as three register buffers rotated per tile).  Helpers _g_rn_mul / _g_rn_add / _g_rn_sub / _g_sigmoid_rn /
# _g_chain_combine / _g_pick_last are imported above from neuronspark.ops.gluon._ptx.

@gluon.jit
def _g_neg_f32(a):
    """neg.f32 (sign flip, exact for +-0), the instruction of ``neuronspark.ops.triton._ptx._pg_neg``; Gluon's unary
    minus lowers to 0 - x, which maps +0 to +0."""
    return gl.inline_asm_elementwise(asm="neg.f32 $0, $1;", constraints="=f,f", args=[a], dtype=gl.float32,
                                     is_pure=True, pack=1)


@gluon.jit
def _k1_bwd_tt16_nw4_d2_kernel(
    ETA, X, TH, PRE, V0, GS, GVLAST, GZ_OUT, GZP_OUT, GX_OUT, GTH_OUT, GV0_OUT,
    ALPHA, T, B, DN, xs0, xs1,
    BC: gl.constexpr, X_BF16: gl.constexpr, HAS_V0: gl.constexpr, HAS_GVLAST: gl.constexpr,
):
    """Phase-split backward of the PLIF scan and of the eta gate, TT=16 tokens per tile, 4 warps, loads DEPTH=2 tiles
    ahead."""
    P2: gl.constexpr = gl.BlockedLayout([1, 1], [1, 32], [4, 1], [1, 0])
    LANE: gl.constexpr = gl.SliceLayout(0, P2)
    ROW: gl.constexpr = gl.SliceLayout(1, P2)
    Q: gl.constexpr = gl.BlockedLayout([16, 1], [1, 32], [1, 4], [1, 0])
    SH: gl.constexpr = gl.SwizzledSharedLayout(1, 1, 1, [1, 0])
    sm_a = gl.allocate_shared_memory(gl.float32, [16, BC], SH)
    sm_c = gl.allocate_shared_memory(gl.float32, [16, BC], SH)
    sm_v = gl.allocate_shared_memory(gl.float32, [16, BC], SH)
    sm_x = gl.allocate_shared_memory(gl.float32, [16, BC], SH)
    pid = gl.program_id(0)
    lane = pid * BC + gl.arange(0, BC, layout=LANE)
    lmask = lane < B * DN
    c_l = lane % DN
    xlane2 = gl.expand_dims((lane // DN) * xs1 + c_l, 0)
    rows = gl.arange(0, 16, layout=ROW)
    lane2 = gl.expand_dims(lane, 0)
    lmask2 = gl.expand_dims(lmask, 0)
    th2 = gl.expand_dims(gl.load(TH + c_l, mask=lmask, other=0.0), 0)
    if HAS_V0:
        v0_2 = gl.expand_dims(gl.load(V0 + lane, mask=lmask, other=0.0), 0)
    else:
        v0_2 = gl.zeros([1, BC], gl.float32, P2)
    QL: gl.constexpr = gl.SliceLayout(0, Q)
    lane_q = pid * BC + gl.arange(0, BC, layout=QL)
    lmask_q = lane_q < B * DN
    rowq = gl.expand_dims(gl.arange(0, 16, layout=gl.SliceLayout(1, Q)), 1) + gl.zeros([16, BC], gl.int32, Q)
    first = rowq == 0
    if HAS_GVLAST:
        x = gl.expand_dims(gl.load(GVLAST + lane_q, mask=lmask_q, other=0.0), 0)
    else:
        x = gl.zeros([1, BC], gl.float32, Q)
    alpha = ALPHA.to(gl.float32)
    gth_acc = gl.zeros([BC], gl.float32, LANE)
    tt0 = T - 1 - (0) - gl.expand_dims(rows, 1)
    mk0 = lmask2 & (tt0 >= 0)
    of0 = tt0 * (B * DN) + lane2
    pre0 = gl.load(PRE + of0, mask=mk0, other=0.0)
    gs0 = gl.load(GS + of0, mask=mk0, other=0.0).to(gl.float32)
    e0 = gl.load(ETA + of0, mask=mk0, other=0.0)
    ox0 = tt0 * xs0 + xlane2
    if X_BF16:
        xw0 = gl.load(X + ox0, mask=mk0, other=0.0).to(gl.float32)
    else:
        xw0 = gl.load(X + ox0, mask=mk0, other=0.0)
    pp0 = gl.load(PRE + of0 - B * DN, mask=lmask2 & (tt0 - 1 >= 0), other=0.0)
    tt1 = T - 1 - (16) - gl.expand_dims(rows, 1)
    mk1 = lmask2 & (tt1 >= 0)
    of1 = tt1 * (B * DN) + lane2
    pre1 = gl.load(PRE + of1, mask=mk1, other=0.0)
    gs1 = gl.load(GS + of1, mask=mk1, other=0.0).to(gl.float32)
    e1 = gl.load(ETA + of1, mask=mk1, other=0.0)
    ox1 = tt1 * xs0 + xlane2
    if X_BF16:
        xw1 = gl.load(X + ox1, mask=mk1, other=0.0).to(gl.float32)
    else:
        xw1 = gl.load(X + ox1, mask=mk1, other=0.0)
    pp1 = gl.load(PRE + of1 - B * DN, mask=lmask2 & (tt1 - 1 >= 0), other=0.0)
    for tile in range(0, T, 16):
        tt2 = T - 1 - (tile + 32) - gl.expand_dims(rows, 1)
        mk2 = lmask2 & (tt2 >= 0)
        of2 = tt2 * (B * DN) + lane2
        pre2 = gl.load(PRE + of2, mask=mk2, other=0.0)
        gs2 = gl.load(GS + of2, mask=mk2, other=0.0).to(gl.float32)
        e2 = gl.load(ETA + of2, mask=mk2, other=0.0)
        ox2 = tt2 * xs0 + xlane2
        if X_BF16:
            xw2 = gl.load(X + ox2, mask=mk2, other=0.0).to(gl.float32)
        else:
            xw2 = gl.load(X + ox2, mask=mk2, other=0.0)
        pp2 = gl.load(PRE + of2 - B * DN, mask=lmask2 & (tt2 - 1 >= 0), other=0.0)
        # ---- phase 1: per-token coefficients (a = 1 - eta)
        tt = T - 1 - tile - gl.expand_dims(rows, 1)
        mk = lmask2 & (tt >= 0)
        of = tt * (B * DN) + lane2
        pre = pre0
        gs = gs0
        e = e0
        xw = xw0
        a = _g_rn_sub(1.0, e)
        th = th2
        d = _g_rn_sub(pre, th)
        gate = _g_sigmoid_rn(_g_rn_mul(d, alpha))
        nd = _g_rn_sub(-pre, th)
        gate_neg = _g_sigmoid_rn(_g_rn_mul(nd, alpha))
        g_mem = _g_rn_add(gate, gate_neg)
        g_th = _g_rn_sub(gate, gate_neg)
        cc = _g_rn_mul(g_mem, gs)
        nra = a
        a_tilde = _g_rn_mul(nra, _g_rn_sub(1.0, g_mem))
        c_tilde = _g_rn_mul(nra, cc)
        # ---- phase 2: the adjoint chain
        va = gl.where(mk, 1.0, 0.0)
        sm_a.store(a_tilde)
        sm_c.store(c_tilde)
        sm_v.store(va)
        gl.thread_barrier()
        aq = sm_a.load(Q)
        cq = sm_c.load(Q)
        vq = sm_v.load(Q)
        x0 = gl.where(vq != 0.0, _g_rn_add(_g_rn_mul(aq, x), cq), x)
        e_x = gl.where(first, x0, cq)
        e_p = gl.where(first, x, 0.0)
        s_a, s_x, s_p, s_v = gl.associative_scan((aq, e_x, e_p, vq), 0, _g_chain_combine)
        x_last, _i_last = gl.reduce((s_x, rowq), 0, _g_pick_last)
        x = gl.expand_dims(x_last, 0)
        # ---- phase 3: outputs (grad_pre, ga; then the eta-gate backward)
        sm_x.store(s_p)
        gl.thread_barrier()
        total = sm_x.load(P2)
        grad_pre = _g_rn_add(total, _g_rn_mul(g_mem, _g_rn_sub(gs, total)))
        gth = _g_rn_mul(g_th, _g_rn_sub(total, gs))
        pre_prev = pp0
        th_prev = th2
        s_prev = gl.maximum(_g_rn_sub(pre_prev, th_prev), 0.0)
        s_prev = _g_rn_sub(s_prev, gl.maximum(_g_rn_sub(-pre_prev, th_prev), 0.0))
        v_prev = gl.where(tt > 0, _g_rn_sub(pre_prev, s_prev), v0_2)
        ga = _g_rn_mul(grad_pre, v_prev)
        gth_acc = _g_rn_add(gth_acc, gl.sum(gl.where(mk, gth, 0.0), axis=0))
        g_eta = _g_rn_add(_g_rn_mul(grad_pre, xw), _g_neg_f32(ga))      # neg.f32 (unary minus lowers to 0 - x: +0 for +0)
        gz = _g_rn_mul(_g_rn_mul(g_eta, _g_rn_sub(1.0, e)), e)
        gx = _g_rn_mul(grad_pre, e)
        gl.store(GZ_OUT + of, gz, mask=mk)
        gl.store(GZP_OUT + of, gz.to(gl.bfloat16), mask=mk)
        if X_BF16:
            gl.store(GX_OUT + of, gx.to(gl.bfloat16), mask=mk)
        else:
            gl.store(GX_OUT + of, gx, mask=mk)
        # rotate the register buffers
        pre0 = pre1
        gs0 = gs1
        e0 = e1
        xw0 = xw1
        pp0 = pp1
        pre1 = pre2
        gs1 = gs2
        e1 = e2
        xw1 = xw2
        pp1 = pp2
    gl.store(GV0_OUT + gl.expand_dims(lane_q, 0), x, mask=gl.expand_dims(lmask_q, 0))
    gl.store(GTH_OUT + lane, gth_acc, mask=lmask)
