"""PLIF population operators: the projected-gate, bipolar soft-reset PLIF neuron of the hippocampal layers.

Every PLIF population (per hippocampal layer: ``gate_neuron`` and ``input_neuron2`` with C = 1536, the latter with a
centred projection input; ``snn_ffn.gate_neuron`` and ``snn_ffn.up_neuron`` with C = 768) computes, per row b and
channel c, with ``xw = fp32(x)`` and every fp32 operation rounded once (round to nearest even, no FMA contraction):

    z     = ((w_eta * xw) + b_eta) + fp32(zp)        zp = eta_U(eta_V(xc)), (T, B, C) bf16, formed by the caller
    eta   = 1 / (1 + expf(-z))                        libdevice expf, add.rn, div.rn (the ATen fp32 sigmoid)
    a     = 1 - eta;   i = eta * xw
    pre_t = (a_t * v_{t-1}) + i_t                      v_{-1} = 0 (training) or the carried membrane (inference)
    S_t   = max(pre_t - th, 0) - max(-pre_t - th, 0)   th = v_th (C,): the bipolar release
    v_t   = pre_t - S_t                               (the membrane stays in [-th, th])

Training backward (sigmoid surrogate of slope alpha = 8.0): gp = sigmoid(alpha (pre - th)), gn = sigmoid(alpha (-pre -
th)), g_mem = gp + gn; the adjoint lam_t = dL/dv_t runs from t = T - 1 down, dL/dpre_t = lam_t + g_mem (gS_t - lam_t),
lam_{t-1} = a_t dL/dpre_t evaluated as (a_t (1 - g_mem)) lam_t + a_t (g_mem gS_t) (the kernel's rounding),
dL/dth += (gp - gn)(lam_t - gS_t); then dL/da_t = dL/dpre_t v_{t-1}, dL/deta = dL/dpre xw - dL/da,
gz = (dL/deta (1 - eta)) eta, the drive edge of x gets dL/dpre eta.

Operators (regime: public function -> what runs):

* train   :func:`plif_scan_train` -> :class:`PLIFScanFn`.  Forward: custom op ``neuronspark::plif_k1x_scan`` = the eta
  kernel (w_eta * x formed in registers, eta stored) followed by the scan forward (a / i formed in registers
  from eta and x; S, and the pre-release membrane ``pre`` when a backward will run).  Backward: custom op
  ``neuronspark::plif_k1_bwd`` (one Gluon kernel: the adjoint scan and the gate backward, see
  :mod:`neuronspark.ops.gluon.plif_k1_bwd`), then torch reductions: grad b_eta = gz.sum((0, 1)); grad w_eta =
  (gz * xw).sum((0, 1)); grad v_th = the kernel's per-lane T sums (B, C) summed over B; x receives two gradients,
  gz * w_eta (the multiply's edge, cast to x.dtype) and the kernel's drive-edge gradient (x.dtype), which autograd adds
  (x is passed to the Function twice, as the multiply's and the drive's operand).
* train   :func:`plif_centred_input` -> :class:`CentreFn` on ``ops.cumsum0``: the strictly-past centring of
  input_neuron2's projection input, xc_t = (x_t - (sum_{j<t} x_j) / max(t, 1)) * [t > 0], returned contiguous in
  x.dtype; backward: A = g * [t > 0] on x, q_t = -g_{t+1} / (t + 1) (0 at t = T - 1) on the running sum.
* prefill :func:`plif_gate_prefill` (custom op ``neuronspark::plif_gate_prefill``): (a, drive) fp32 from x, w_eta, b_eta
  and the projection, any T >= 1; :func:`plif_scan_prefill` (custom op ``neuronspark::plif_scan_prefill``, mutates
  ``v0``): the scan from the carried membrane v0, which it overwrites with the last membrane.
* decode: the modules' torch T = 1 pointwise step (no op here).

Numerics.  The training forward S / pre are the operations of the eager gate chain and the sequential scan in the same
order; gz / gzp / gx follow the phase-split backward above; the per-channel sums for b_eta, w_eta and v_th are
reassociated against autograd's (fp32 reduction-order class).  The prefill a / drive / S are the eager chain's
operations.  Tests: the per-operator tolerance rule of :mod:`neuronspark.utils.testing` against the references.

Pure-torch references (same signatures): :func:`plif_scan_train_ref` (the eager forward; the phase-split backward's
expressions through :func:`neuronspark.ops.gluon.plif_k1_bwd.plif_k1_bwd_ref`), :func:`plif_centred_input_ref`,
:func:`plif_gate_prefill_ref`, :func:`plif_scan_prefill_ref`.
"""
from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice

from neuronspark.ops.gluon.plif_k1_bwd import gate_slope_ok, plif_k1_bwd, plif_k1_bwd_ref  # noqa: F401  (registers neuronspark::plif_k1_bwd)
from neuronspark.ops.triton import require_cuda, require_sm100
from neuronspark.ops.triton._ptx import (  # noqa: F401  (resolved by the kernels below through this module's globals)
    _add_rn,
    _div_rn,
    _mul_rn,
    _pg_add,
    _pg_div,
    _pg_mul,
    _pg_n,
    _pg_neg,
    _pg_sigmoid,
    _pg_sub,
    _pg_w,
    _rn_add_f32,
    _rn_mul_f32,
    _sub_rn,
)

__all__ = [
    "ALPHA",
    "PLIFScanFn",
    "plif_scan_train",
    "plif_scan_train_ref",
    "CentreFn",
    "plif_centred_input",
    "plif_centred_input_ref",
    "plif_gate_prefill",
    "plif_gate_prefill_ref",
    "plif_scan_prefill",
    "plif_scan_prefill_ref",
]

# ================================================================================================ fixed constants
# Launch configuration of the training and inference operators (fixed; no run-time selection).
#: surrogate slope of the training backward (the model's plif_surrogate_alpha)
ALPHA = 8.0
#: eta kernel and centring kernels: one program per (t, b) row and channel block of 1024, 4 warps
ROW_BLOCK = 1024
ROW_WARPS = 4
#: scan forward of the training op: TILE 32, DEPTH 1, AHEAD 0, one lane per thread, 1 warp
K1L_BLOCK_C = 32
K1L_WARPS = 1
#: prefill gate kernel: 2048 elements per program, 4 warps
GATE_BLOCK = 2048
GATE_WARPS = 4
#: prefill scan: TILE 32, DEPTH 1, 32 lanes per program, 1 warp
SCAN_BLOCK_C = 32
SCAN_WARPS = 1

_DTYPES: Dict[str, torch.dtype] = {"float32": torch.float32, "bfloat16": torch.bfloat16}
# documented first-call cache: gradient strides of the centring backward per (shape, strides, dtype)
_CENTRE_LAYOUT: Dict[tuple, Tuple[Tuple[int, ...], Tuple[int, ...]]] = {}


def _need(cond: bool, msg: str) -> None:
    if not cond:
        raise RuntimeError("neuronspark.ops.plif: " + msg)


def _dtype_name(dt: torch.dtype) -> str:
    for name, d in _DTYPES.items():
        if d == dt:
            return name
    raise RuntimeError(f"neuronspark.ops.plif: spikes are stored in float32 or bfloat16, got {dt}")


def _tb_ok(t: torch.Tensor) -> bool:
    """(T, B, C) with channel stride 1 and non-overlapping (t, b) strides (T-major contiguous or a (T, B, C) view of
    B-major storage), int32-addressable."""
    if t.dim() != 3 or int(t.stride(2)) != 1:
        return False
    T, B, C = (int(s) for s in t.shape)
    s0, s1 = int(t.stride(0)), int(t.stride(1))
    tmaj = s1 == C and s0 == B * C
    bmaj = s0 == C and s1 == T * C
    return (tmaj or bmaj or B == 1) and T * B * C < 2 ** 31


def _check_dense(name: str, t: torch.Tensor) -> None:
    _need(t.dim() == 3 and int(t.stride(2)) == 1,
          f"{name} must be 3-D with a dense channel axis, got {tuple(t.shape)} {tuple(t.stride())}")


# ================================================================================================ kernels (training)
@triton.jit
def _eta_x_kernel(W, X, BE, ZP, ETA, B, C, s_x0, s_x1, s_zp0, s_zp1,
                  X_BF16: tl.constexpr, BLOCK: tl.constexpr):
    """eta = sigmoid(((w_eta * xw) + b_eta) + fp32(zp)) for one (t, b) row and a block of channels: the multiply as one
    mul.rn of the (C,) weight and the widened x, then add.rn, add.rn and the ATen sigmoid; eta stored contiguous."""
    r = tl.program_id(0)
    blk = tl.program_id(1)
    t = r // B
    b = r - t * B
    c = blk * BLOCK + tl.arange(0, BLOCK)
    m = c < C
    t64 = t.to(tl.int64)
    b64 = b.to(tl.int64)
    if X_BF16:
        xw = _pg_w(tl.load(X + t64 * s_x0 + b64 * s_x1 + c, mask=m, other=0.0))
    else:
        xw = tl.load(X + t64 * s_x0 + b64 * s_x1 + c, mask=m, other=0.0)
    zwx = _pg_mul(tl.load(W + c, mask=m, other=0.0), xw)
    be = tl.load(BE + c, mask=m, other=0.0)
    zp = _pg_w(tl.load(ZP + t64 * s_zp0 + b64 * s_zp1 + c, mask=m, other=0.0))
    eta = _pg_sigmoid(_pg_add(_pg_add(zwx, be), zp))
    tl.store(ETA + r.to(tl.int64) * C + c, eta, mask=m)


@triton.jit
def _centre_fwd_kernel(XF, XC, OUT, B, C, s_xf0, s_xf1, s_xc0, s_xc1,
                       OUT_BF16: tl.constexpr, BLOCK: tl.constexpr):
    """out_t = ((xf_t - xc_{t-1} / max(t, 1)) * [t > 0]) (div.rn, sub.rn, mul.rn; cvt.rn.bf16 for a bf16 output);
    xc is the inclusive running sum, so xc_{t-1} is the strictly-past sum."""
    r = tl.program_id(0)
    blk = tl.program_id(1)
    t = r // B
    b = r - t * B
    c = blk * BLOCK + tl.arange(0, BLOCK)
    m = c < C
    t64 = t.to(tl.int64)
    b64 = b.to(tl.int64)
    xf = tl.load(XF + t64 * s_xf0 + b64 * s_xf1 + c, mask=m, other=0.0)
    has = t > 0
    tp = tl.where(has, t - 1, 0).to(tl.int64)
    xcs = tl.load(XC + tp * s_xc0 + b64 * s_xc1 + c, mask=m & has, other=0.0)
    xcs = tl.where(has, xcs, 0.0)
    xn = tl.where(has, t.to(tl.float32), 1.0)
    hp = tl.where(has, 1.0, 0.0)
    v = _pg_mul(_pg_sub(xf, _pg_div(xcs, xn)), hp)
    o = r.to(tl.int64) * C + c
    if OUT_BF16:
        tl.store(OUT + o, _pg_n(v), mask=m)
    else:
        tl.store(OUT + o, v, mask=m)


@triton.jit
def _centre_bwd_kernel(G, AO, QS, T, B, C, s_g0, s_g1, s_a0, s_a1, s_q0, s_q1,
                       G_BF16: tl.constexpr, BLOCK: tl.constexpr):
    """A_t = g_t * [t > 0] (the x edge); q_t = (-(g_{t+1} * 1)) / (t + 1) for t < T - 1 and 0 at t = T - 1 (the
    running-sum edge)."""
    r = tl.program_id(0)
    blk = tl.program_id(1)
    t = r // B
    b = r - t * B
    c = blk * BLOCK + tl.arange(0, BLOCK)
    m = c < C
    t64 = t.to(tl.int64)
    b64 = b.to(tl.int64)
    if G_BF16:
        g = _pg_w(tl.load(G + t64 * s_g0 + b64 * s_g1 + c, mask=m, other=0.0))
    else:
        g = tl.load(G + t64 * s_g0 + b64 * s_g1 + c, mask=m, other=0.0)
    hp = tl.where(t > 0, 1.0, 0.0)
    a = _pg_mul(g, hp)
    tl.store(AO + t64 * s_a0 + b64 * s_a1 + c, a, mask=m)
    nx = t + 1 < T
    tn = tl.where(nx, t + 1, t).to(tl.int64)
    if G_BF16:
        g1 = _pg_w(tl.load(G + tn * s_g0 + b64 * s_g1 + c, mask=m & nx, other=0.0))
    else:
        g1 = tl.load(G + tn * s_g0 + b64 * s_g1 + c, mask=m & nx, other=0.0)
    q = _pg_div(_pg_neg(_pg_mul(g1, 1.0)), (t + 1).to(tl.float32))
    q = tl.where(nx, q, 0.0)
    tl.store(QS + t64 * s_q0 + b64 * s_q1 + c, q, mask=m)


# ================================================================================================ kernels (prefill)
@triton.jit
def _gate_prefill_kernel(X, Z, W, B, A, D, N, C, BLOCK: tl.constexpr):
    """a = 1 - eta and drive = eta * xw with eta = 1 / (1 + expf(-(((w * xw) + b) + z))), every fp32 operation through
    inline PTX without flush-to-zero (ATen keeps subnormals); z is the projection, widened exactly when stored bf16."""
    pid = tl.program_id(axis=0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    m = offs < N
    c = offs % C
    xf = tl.load(X + offs, mask=m, other=0.0).to(tl.float32)
    z = tl.load(Z + offs, mask=m, other=0.0).to(tl.float32)
    w = tl.load(W + c, mask=m, other=0.0)
    b = tl.load(B + c, mask=m, other=0.0)
    one = tl.full(xf.shape, 1.0, tl.float32)
    wx = _mul_rn(w, xf)
    z1 = _add_rn(wx, b)
    z2 = _add_rn(z1, z)
    e = libdevice.exp(-z2)
    eta = _div_rn(one, _add_rn(one, e))
    a = _sub_rn(one, eta)
    d = _mul_rn(eta, xf)
    tl.store(A + offs, a, mask=m)
    tl.store(D + offs, d, mask=m)


# ================================================================================================ launchers (training)
def _check_scan_train(w_eta, b_eta, zp, x, v_th) -> Tuple[int, int, int]:
    require_cuda(x, "plif_scan_train")
    _need(x.dim() == 3, f"x must be (T, B, C), got {tuple(x.shape)}")
    T, B, C = (int(s) for s in x.shape)
    _need(T >= 2, f"the training scan needs T >= 2 tokens, got T = {T}")
    _need(x.dtype in (torch.float32, torch.bfloat16) and _tb_ok(x),
          f"x must be fp32 / bf16 (T, B, C) with channel stride 1 (T-major or a view of B-major storage), got "
          f"{x.dtype} {tuple(x.stride())}")
    for name, t in (("w_eta", w_eta), ("b_eta", b_eta), ("v_th", v_th)):
        _need(t.dtype == torch.float32 and tuple(t.shape) == (C,) and t.is_contiguous() and t.device == x.device,
              f"{name} must be a contiguous fp32 ({C},) tensor on {x.device}, got {t.dtype} {tuple(t.shape)}")
    _need(zp.dtype == torch.bfloat16 and tuple(zp.shape) == (T, B, C) and zp.is_contiguous()
          and zp.device == x.device,
          f"zp must be a contiguous bf16 ({T}, {B}, {C}) tensor, got {zp.dtype} {tuple(zp.shape)} {tuple(zp.stride())}")
    return T, B, C


def _eta_x(w_eta: torch.Tensor, x: torch.Tensor, b_eta: torch.Tensor, zp: torch.Tensor) -> torch.Tensor:
    """eta (T, B, C) fp32 contiguous with the multiply w_eta * x formed in the kernel."""
    T, B, C = (int(s) for s in x.shape)
    _check_dense("x", x)
    _check_dense("zp", zp)
    eta = torch.empty((T, B, C), dtype=torch.float32, device=x.device)
    _eta_x_kernel[(T * B, triton.cdiv(C, ROW_BLOCK))](
        w_eta, x, b_eta, zp, eta, B, C, x.stride(0), x.stride(1), zp.stride(0), zp.stride(1),
        X_BF16=int(x.dtype == torch.bfloat16), BLOCK=ROW_BLOCK, num_warps=ROW_WARPS, enable_fp_fusion=False)
    return eta


def _plif_k1x_fwd(w_eta, b_eta, zp, x, v_th, store_pre: bool, out_dtype: torch.dtype):
    """(S out_dtype, pre fp32, eta fp32): pre / eta (T, B, C) contiguous when ``store_pre`` (the backward's operands),
    else empty (0,) tensors."""
    T, B, C = _check_scan_train(w_eta, b_eta, zp, x, v_th)
    eta = _eta_x(w_eta, x, b_eta, zp)
    dev = x.device
    S = torch.empty((T, B, C), dtype=out_dtype, device=dev)
    vlast = torch.empty((B, C), dtype=torch.float32, device=dev)
    pre = (torch.empty((T, B, C), dtype=torch.float32, device=dev) if store_pre
           else x.new_empty((0,), dtype=torch.float32))
    _k1l_fwd_t32_d1_a0_kernel[(triton.cdiv(B * C, K1L_BLOCK_C),)](
        eta, x, v_th, vlast, S, vlast, pre, T, B, C, int(x.stride(0)), int(x.stride(1)),
        STORE_PRE=int(bool(store_pre)), X_BF16=int(x.dtype == torch.bfloat16), BLOCK_C=K1L_BLOCK_C, BIPOLAR=1,
        HAS_V0=0, num_warps=K1L_WARPS)
    return S, pre, (eta if store_pre else x.new_empty((0,), dtype=torch.float32))


@torch.library.custom_op("neuronspark::plif_k1x_scan", mutates_args=())
def _plif_k1x_scan_op(w_eta: torch.Tensor, b_eta: torch.Tensor, zp: torch.Tensor, x: torch.Tensor,
                      v_th: torch.Tensor, store_pre: bool, out_dtype: str
                      ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """The training forward launcher: (S, pre, eta)."""
    return _plif_k1x_fwd(w_eta, b_eta, zp, x, v_th, bool(store_pre), _DTYPES[out_dtype])


@_plif_k1x_scan_op.register_fake
def _plif_k1x_scan_fake(w_eta, b_eta, zp, x, v_th, store_pre, out_dtype):
    T, B, C = x.shape
    shp = (T, B, C) if store_pre else (0,)
    return (x.new_empty((T, B, C), dtype=_DTYPES[out_dtype]), x.new_empty(shp, dtype=torch.float32),
            x.new_empty(shp, dtype=torch.float32))


# ================================================================================================ autograd (training scan)
class PLIFScanFn(torch.autograd.Function):
    """S of the PLIF population; gradients for w_eta, b_eta, zp, x (two edges: the multiply's ``xm`` and the drive's
    ``x``, the same tensor) and v_th."""

    @staticmethod
    def forward(ctx, w_eta, b_eta, zp, xm, x, v_th, alpha: float, out_dtype: str):
        S, pre, eta = torch.ops.neuronspark.plif_k1x_scan(w_eta, b_eta, zp, x, v_th, True, out_dtype)
        ctx.set_materialize_grads(False)
        ctx.save_for_backward(eta, x, v_th, pre, w_eta, xm)
        ctx.alpha = float(alpha)
        return S

    @staticmethod
    def backward(ctx, gS):
        eta, x, v_th, pre, w_eta, xm = ctx.saved_tensors
        need_w, need_b, need_zp, need_xm, need_x, need_th = ctx.needs_input_grad[:6]
        gS_ = pre.new_zeros(pre.shape) if gS is None else gS.contiguous()
        if gS_.dtype not in (torch.float32, torch.bfloat16):
            gS_ = gS_.float()
        gz, gzp, gx, gth = torch.ops.neuronspark.plif_k1_bwd(eta, x, v_th, pre, gS_, ctx.alpha)
        # the in-kernel multiply's autograd (zwx = w_eta.float() * xm.float()): grad w = sum over (T, B) of gz * xm,
        # grad xm = gz * w (in xm's dtype)
        gw = (gz * xm.to(torch.float32)).sum(dim=(0, 1)) if need_w else None
        gxm = None
        if need_xm:
            gxm = gz * w_eta.to(torch.float32)
            if xm.dtype != torch.float32:
                gxm = gxm.to(xm.dtype)
        return (gw, gz.sum(dim=(0, 1)) if need_b else None, gzp if need_zp else None, gxm,
                gx if need_x else None, gth.sum(dim=0) if need_th else None, None, None)


def plif_scan_train(w_eta: torch.Tensor, b_eta: torch.Tensor, zp: torch.Tensor, x: torch.Tensor,
                    v_th: torch.Tensor, *, alpha: float = ALPHA,
                    out_dtype: Optional[torch.dtype] = None) -> torch.Tensor:
    """Spikes S (T, B, C) of one PLIF population over whole rows (membrane zero at t = 0).

    ``w_eta``, ``b_eta``, ``v_th``: contiguous fp32 (C,); ``zp``: contiguous bf16 (T, B, C) (the eta projection of the
    caller); ``x``: fp32 or bf16 (T, B, C) with channel stride 1, T-major or a (T, B, C) view of B-major storage;
    T >= 2.  ``out_dtype``: dtype of S (float32 / bfloat16), default ``x.dtype``.  When a gradient is needed, the
    backward runs a Gluon kernel (compute capability >= 10.0, checked here); ``alpha`` is the surrogate slope
    (0 < alpha < 64: the plain RN sigmoid form of the kernel).
    """
    _check_scan_train(w_eta, b_eta, zp, x, v_th)
    od = _dtype_name(x.dtype if out_dtype is None else out_dtype)
    grad = torch.is_grad_enabled() and any(t.requires_grad for t in (w_eta, b_eta, zp, x, v_th))
    if grad:
        _need(gate_slope_ok(alpha), f"surrogate slope alpha must satisfy 0 < alpha < 64, got {alpha}")
        require_sm100(x.device, "plif_scan_train (backward)")
        return PLIFScanFn.apply(w_eta, b_eta, zp, x, x, v_th, float(alpha), od)
    S, _pre, _eta = torch.ops.neuronspark.plif_k1x_scan(w_eta, b_eta, zp, x, v_th, False, od)
    return S


# ================================================================================================ centring (training)
def _centre_layouts(g: torch.Tensor) -> Tuple[Tuple[int, ...], Tuple[int, ...]]:
    """The strides the eager centring's backward gives its two gradients (A = g.float() * hp; q zeros-like): the same
    ops on meta tensors of the same strides, cached per signature."""
    key = (tuple(g.shape), tuple(g.stride()), g.dtype)
    st = _CENTRE_LAYOUT.get(key)
    if st is None:
        f32 = torch.float32
        T = int(g.shape[0])
        g_m = torch.empty_strided(tuple(g.shape), tuple(g.stride()), dtype=g.dtype, device="meta").to(f32)
        hp = torch.empty((T, 1, 1), dtype=f32, device="meta")
        h = g_m * hp
        q_s = torch.zeros(tuple(g.shape), dtype=f32, device="meta")
        st = (tuple(h.stride()), tuple(q_s.stride()))
        _CENTRE_LAYOUT[key] = st
    return st


def _centre_fwd(xf: torch.Tensor, xc: torch.Tensor, out_dtype: torch.dtype) -> torch.Tensor:
    T, B, C = (int(s) for s in xf.shape)
    _check_dense("xf", xf)
    _check_dense("xc", xc)
    _need(xf.dtype == torch.float32 and xc.dtype == torch.float32 and tuple(xc.shape) == (T, B, C) and T >= 2,
          "the centring needs fp32 xf / running sum of one shape with T >= 2")
    out = torch.empty((T, B, C), dtype=out_dtype, device=xf.device)
    _centre_fwd_kernel[(T * B, triton.cdiv(C, ROW_BLOCK))](
        xf, xc, out, B, C, xf.stride(0), xf.stride(1), xc.stride(0), xc.stride(1),
        OUT_BF16=int(out_dtype == torch.bfloat16), BLOCK=ROW_BLOCK, num_warps=ROW_WARPS, enable_fp_fusion=False)
    return out


def _centre_bwd(g: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    T, B, C = (int(s) for s in g.shape)
    _check_dense("g", g)
    _need(g.dtype in (torch.float32, torch.bfloat16), f"the centring gradient must be fp32 / bf16, got {g.dtype}")
    st_a, st_q = _centre_layouts(g)
    a = torch.empty_strided((T, B, C), st_a, dtype=torch.float32, device=g.device)
    qs = torch.empty_strided((T, B, C), st_q, dtype=torch.float32, device=g.device)
    _check_dense("A", a)
    _check_dense("q", qs)
    _centre_bwd_kernel[(T * B, triton.cdiv(C, ROW_BLOCK))](
        g, a, qs, T, B, C, g.stride(0), g.stride(1), a.stride(0), a.stride(1), qs.stride(0), qs.stride(1),
        G_BF16=int(g.dtype == torch.bfloat16), BLOCK=ROW_BLOCK, num_warps=ROW_WARPS, enable_fp_fusion=False)
    return a, qs


@torch.library.custom_op("neuronspark::plif_centre_fwd", mutates_args=())
def _plif_centre_fwd_op(xf: torch.Tensor, xc: torch.Tensor, out_dtype: str) -> torch.Tensor:
    return _centre_fwd(xf, xc, _DTYPES[out_dtype])


@_plif_centre_fwd_op.register_fake
def _plif_centre_fwd_fake(xf, xc, out_dtype):
    return xf.new_empty(tuple(xf.shape), dtype=_DTYPES[out_dtype])


@torch.library.custom_op("neuronspark::plif_centre_bwd", mutates_args=())
def _plif_centre_bwd_op(g: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    return _centre_bwd(g)


@_plif_centre_bwd_op.register_fake
def _plif_centre_bwd_fake(g):
    st_a, st_q = _centre_layouts(g)
    return (g.new_empty_strided(tuple(g.shape), st_a, dtype=torch.float32),
            g.new_empty_strided(tuple(g.shape), st_q, dtype=torch.float32))


class CentreFn(torch.autograd.Function):
    """xc (contiguous, ``out_dtype``) from xf and its inclusive running sum; gradients on both."""

    @staticmethod
    def forward(ctx, xf, xsum, out_dtype: str):
        return torch.ops.neuronspark.plif_centre_fwd(xf, xsum, out_dtype)

    @staticmethod
    def backward(ctx, g):
        a, qs = torch.ops.neuronspark.plif_centre_bwd(g)
        return a, qs, None


def plif_centred_input(x: torch.Tensor) -> torch.Tensor:
    """``x_t - mean_{j<t} x_j`` (zero at t = 0) in x.dtype, contiguous: input_neuron2's projection input.

    ``x``: fp32 (or bf16) (T, B, C) CUDA tensor with a dense channel axis, T >= 2.  The running sum is ``ops.cumsum0``
    of ``x.float()``; the centring and its backward are one kernel each (:class:`CentreFn`)."""
    require_cuda(x, "plif_centred_input")
    _need(x.dim() == 3 and int(x.shape[0]) >= 2, f"x must be (T >= 2, B, C), got {tuple(x.shape)}")
    _need(x.dtype in (torch.float32, torch.bfloat16), f"x must be fp32 or bf16, got {x.dtype}")
    _check_dense("x", x)
    from neuronspark.ops.triton.cumsum import cumsum0
    xf = x.float()
    xsum = cumsum0(xf)
    return CentreFn.apply(xf, xsum, _dtype_name(x.dtype))


# ================================================================================================ prefill
def _gate_prefill(x: torch.Tensor, w_eta: torch.Tensor, b_eta: torch.Tensor, zp: torch.Tensor):
    require_cuda(x, "plif_gate_prefill")
    _need(x.dim() == 3 and tuple(zp.shape) == tuple(x.shape) and zp.dtype in (torch.float32, torch.bfloat16),
          f"x (T, B, C) and the projection zp (fp32 / bf16) of the same shape expected, got {tuple(x.shape)} "
          f"{x.dtype} / {tuple(zp.shape)} {zp.dtype}")
    _need(x.dtype in (torch.float32, torch.bfloat16), f"x must be fp32 or bf16, got {x.dtype}")
    C = int(x.shape[-1])
    for name, t in (("w_eta", w_eta), ("b_eta", b_eta)):
        _need(t.dtype == torch.float32 and tuple(t.shape) == (C,) and t.device == x.device,
              f"{name} must be fp32 ({C},), got {t.dtype} {tuple(t.shape)}")
    xc = x if x.is_contiguous() else x.contiguous()
    zc = zp if zp.is_contiguous() else zp.contiguous()
    wc = w_eta if w_eta.is_contiguous() else w_eta.contiguous()
    bc = b_eta if b_eta.is_contiguous() else b_eta.contiguous()
    a = torch.empty(x.shape, dtype=torch.float32, device=x.device)
    d = torch.empty(x.shape, dtype=torch.float32, device=x.device)
    N = xc.numel()
    if N:
        _gate_prefill_kernel[(triton.cdiv(N, GATE_BLOCK),)](xc, zc, wc, bc, a, d, N, C, BLOCK=GATE_BLOCK,
                                                            num_warps=GATE_WARPS, enable_fp_fusion=False)
    return a, d


@torch.library.custom_op("neuronspark::plif_gate_prefill", mutates_args=())
def _plif_gate_prefill_op(x: torch.Tensor, w_eta: torch.Tensor, b_eta: torch.Tensor,
                          zp: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    return _gate_prefill(x, w_eta, b_eta, zp)


@_plif_gate_prefill_op.register_fake
def _plif_gate_prefill_fake(x, w_eta, b_eta, zp):
    return x.new_empty(tuple(x.shape), dtype=torch.float32), x.new_empty(tuple(x.shape), dtype=torch.float32)


def _scan_prefill(a: torch.Tensor, drive: torch.Tensor, v_th: torch.Tensor, v0: torch.Tensor,
                  out_dtype: torch.dtype) -> torch.Tensor:
    require_cuda(drive, "plif_scan_prefill")
    _need(drive.dim() == 3, f"drive must be (T, B, C), got {tuple(drive.shape)}")
    T, B, C = (int(s) for s in drive.shape)
    _need(T >= 1, "at least one token")
    _need(a.dtype == torch.float32 and drive.dtype == torch.float32 and tuple(a.shape) == (T, B, C),
          f"a / drive must be fp32 ({T}, {B}, {C}), got {a.dtype} {tuple(a.shape)} / {drive.dtype}")
    _need(v_th.dtype == torch.float32 and tuple(v_th.shape) == (C,), f"v_th must be fp32 ({C},)")
    _need(v0.dtype == torch.float32 and tuple(v0.shape) == (B, C) and v0.is_contiguous(),
          f"v0 must be a contiguous fp32 ({B}, {C}) membrane, got {v0.dtype} {tuple(v0.shape)}")
    _need(T * B * C < 2 ** 31, "too many elements for int32 addressing")
    a = a.contiguous()
    drive = drive.contiguous()
    th = v_th.contiguous()
    dev = drive.device
    S = torch.empty((T, B, C), dtype=out_dtype, device=dev)
    vlast = torch.empty((B, C), dtype=torch.float32, device=dev)
    pre = drive.new_empty((0,), dtype=torch.float32)
    no_reset = torch.empty((1,), dtype=torch.int32, device=dev)          # not read (no reset)
    _plif2_fwd_t32_d1_kernel[(triton.cdiv(B * C, SCAN_BLOCK_C),)](
        a, drive, th, no_reset, v0, S, vlast, pre, T, B, C,
        A_PER_CHANNEL=0, TH_PER_CHANNEL=1, STORE_PRE=0, BLOCK_C=SCAN_BLOCK_C, BIPOLAR=1, HAS_RESET=0, HAS_V0=1,
        num_warps=SCAN_WARPS)
    v0.copy_(vlast)
    return S


@torch.library.custom_op("neuronspark::plif_scan_prefill", mutates_args=("v0",))
def _plif_scan_prefill_op(a: torch.Tensor, drive: torch.Tensor, v_th: torch.Tensor, v0: torch.Tensor,
                          out_dtype: str) -> torch.Tensor:
    return _scan_prefill(a, drive, v_th, v0, _DTYPES[out_dtype])


@_plif_scan_prefill_op.register_fake
def _plif_scan_prefill_fake(a, drive, v_th, v0, out_dtype):
    return drive.new_empty(tuple(drive.shape), dtype=_DTYPES[out_dtype])


def plif_gate_prefill(x: torch.Tensor, w_eta: torch.Tensor, b_eta: torch.Tensor,
                      zp: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """(a, drive), each fp32 (T, B, C) contiguous, of the inference gate chain: ``x`` (T, B, C) fp32 / bf16, ``w_eta``
    / ``b_eta`` fp32 (C,), ``zp`` the eta projection (T, B, C) (bf16, or its exact fp32 widening)."""
    return torch.ops.neuronspark.plif_gate_prefill(x, w_eta, b_eta, zp)


def plif_scan_prefill(a: torch.Tensor, drive: torch.Tensor, v_th: torch.Tensor, v0: torch.Tensor,
                      out_dtype: Optional[torch.dtype] = None) -> torch.Tensor:
    """Spikes S (T, B, C) in ``out_dtype`` (default float32) of the scan from the carried membrane ``v0`` (B, C) fp32
    contiguous, which is overwritten in place with the membrane after the last token.  ``a`` / ``drive`` fp32
    (T, B, C), ``v_th`` fp32 (C,); any T >= 1 (the model uses it for T >= 2)."""
    od = _dtype_name(torch.float32 if out_dtype is None else out_dtype)
    return torch.ops.neuronspark.plif_scan_prefill(a, drive, v_th, v0, od)


# ================================================================================================ pure-torch references
class _PLIFScanRef(torch.autograd.Function):
    """Pure-torch training scan: forward = the eager gate chain and the sequential scan with the bipolar release;
    backward = the training backward in torch (:func:`plif_k1_bwd_ref`: the phase-split adjoint
    ``lam_{t-1} = (a (1 - g_mem)) lam_t + a (g_mem gS)`` sequential over t, then the gate backward) followed by the
    op's torch reductions (grad w_eta = (gz * xm).sum((0, 1)), grad b_eta = gz.sum((0, 1)), grad v_th = the per-lane
    T sums summed over B, grad xm = gz * w_eta, the drive edge's gx on x)."""

    @staticmethod
    def forward(ctx, w_eta, b_eta, zp, xm, x, v_th, alpha: float, out_dtype: torch.dtype):
        f32 = torch.float32
        T, B, C = (int(s) for s in x.shape)
        z = w_eta.to(f32) * xm.to(f32) + b_eta.to(f32)
        z = z + zp.to(f32)
        eta = torch.sigmoid(z)
        a = 1.0 - eta
        drive = eta * x.to(f32)
        th = v_th.to(f32)
        prev = drive.new_zeros((B, C))
        spikes = []
        pres = []
        for t in range(T):
            v_pre = a[t] * prev + drive[t]
            s = torch.clamp_min(v_pre - th, 0.0) - torch.clamp_min(-v_pre - th, 0.0)
            prev = v_pre - s
            spikes.append(s)
            pres.append(v_pre)
        ctx.set_materialize_grads(False)
        ctx.save_for_backward(eta, x, v_th, torch.stack(pres, dim=0), w_eta, xm)
        ctx.alpha = float(alpha)
        return torch.stack(spikes, dim=0).to(out_dtype)

    @staticmethod
    def backward(ctx, gS):
        eta, x, v_th, pre, w_eta, xm = ctx.saved_tensors
        need_w, need_b, need_zp, need_xm, need_x, need_th = ctx.needs_input_grad[:6]
        gS_ = pre.new_zeros(pre.shape) if gS is None else gS.contiguous()
        if gS_.dtype not in (torch.float32, torch.bfloat16):
            gS_ = gS_.float()
        gz, gzp, gx, gth = plif_k1_bwd_ref(eta, x, v_th, pre, gS_, ctx.alpha)
        gw = (gz * xm.to(torch.float32)).sum(dim=(0, 1)) if need_w else None
        gxm = None
        if need_xm:
            gxm = gz * w_eta.to(torch.float32)
            if xm.dtype != torch.float32:
                gxm = gxm.to(xm.dtype)
        return (gw, gz.sum(dim=(0, 1)) if need_b else None, gzp if need_zp else None, gxm,
                gx if need_x else None, gth.sum(dim=0) if need_th else None, None, None)


def plif_scan_train_ref(w_eta: torch.Tensor, b_eta: torch.Tensor, zp: torch.Tensor, x: torch.Tensor,
                        v_th: torch.Tensor, *, alpha: float = ALPHA,
                        out_dtype: Optional[torch.dtype] = None) -> torch.Tensor:
    """Pure-torch reference of :func:`plif_scan_train`.

    Forward: the eager gate chain, the sequential scan and the bipolar release.  Backward: the expressions of the
    phase-split backward kernel (:mod:`neuronspark.ops.gluon.plif_k1_bwd`, phases 1-3) evaluated sequentially in torch,
    and the reductions of the op's autograd formula.  The kernel's adjoint is ``lam_{t-1} = RN(RN(a~ lam_t) + c~)``
    with ``a~ = a (1 - g_mem)``, ``c~ = a (g_mem gS)``; autograd through the eager chain rounds it as ``a (lam +
    g_mem (gS - lam))`` instead, a different rounding at every step of a chain as long as the slowest channel's memory.
    x enters twice (the multiply's and the drive's operand), as in :func:`plif_scan_train`."""
    od = x.dtype if out_dtype is None else out_dtype
    return _PLIFScanRef.apply(w_eta, b_eta, zp, x, x, v_th, float(alpha), od)


def plif_centred_input_ref(x: torch.Tensor) -> torch.Tensor:
    """Pure-torch reference of :func:`plif_centred_input`: the strictly-past centring with ATen's ``cumsum(dim=0)``,
    cast to x.dtype, contiguous."""
    T = int(x.shape[0])
    xf = x.float()
    xc = xf.cumsum(dim=0)
    xc = torch.cat([torch.zeros_like(xc[:1]), xc[:-1]], dim=0)
    xn = torch.arange(T, device=x.device, dtype=torch.float32).clamp_min(1.0).view(T, 1, 1)
    has_past = (torch.arange(T, device=x.device) > 0).view(T, 1, 1).to(torch.float32)
    return ((xf - xc / xn) * has_past).to(x.dtype).contiguous()


def plif_gate_prefill_ref(x: torch.Tensor, w_eta: torch.Tensor, b_eta: torch.Tensor,
                          zp: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Pure-torch reference of :func:`plif_gate_prefill`: the eager chain."""
    z_eta = w_eta.to(torch.float32) * x.to(torch.float32) + b_eta.to(torch.float32)
    z_eta = z_eta + zp.to(torch.float32)
    eta = torch.sigmoid(z_eta)
    a_eta = 1.0 - eta
    xw = x.to(torch.float32)
    return a_eta, eta * xw


@torch.no_grad()
def plif_scan_prefill_ref(a: torch.Tensor, drive: torch.Tensor, v_th: torch.Tensor, v0: torch.Tensor,
                          out_dtype: Optional[torch.dtype] = None) -> torch.Tensor:
    """Pure-torch reference of :func:`plif_scan_prefill`: the sequential scan from ``v0`` (forward only); ``v0`` is
    overwritten with the last membrane."""
    T = int(drive.shape[0])
    th = v_th.to(torch.float32)
    prev = v0.to(torch.float32)
    spikes = []
    for t in range(T):
        v_pre = a[t] * prev + drive[t]
        s = torch.clamp_min(v_pre - th, 0.0) - torch.clamp_min(-v_pre - th, 0.0)
        prev = v_pre - s
        spikes.append(s)
    v0.copy_(prev)
    return torch.stack(spikes, dim=0).to(torch.float32 if out_dtype is None else out_dtype)


# ================================================================================================ scan kernels
# The two scan kernels below are written out per token of their 32-token tile.  Their helpers (_pg_mul, _pg_sub, _pg_w,
# _rn_add_f32, _rn_mul_f32) are imported above from neuronspark.ops.triton._ptx.
#
# _k1l_fwd_t32_d1_a0_kernel: the training scan forward, TILE 32, DEPTH 1, AHEAD 0 (a = 1 - eta and i = eta * xw formed
#   from eta and x in registers).
@triton.jit
def _k1l_fwd_t32_d1_a0_kernel(
    ETA, X, TH, V0, S_OUT, VLAST_OUT, PRE_OUT,
    T, B, DN, xs0, xs1,
    STORE_PRE: tl.constexpr, X_BF16: tl.constexpr, BLOCK_C: tl.constexpr, BIPOLAR: tl.constexpr,
    HAS_V0: tl.constexpr,
):
    """Training scan forward: a / i from eta and x in registers, TILE=32, DEPTH=1, AHEAD=0."""
    pid = tl.program_id(0)
    offs = pid * BLOCK_C + tl.arange(0, BLOCK_C)
    cmask = offs < B * DN
    b = offs // DN
    c = offs % DN
    if HAS_V0:
        v = tl.load(V0 + offs, mask=cmask, other=0.0)
    else:
        v = tl.zeros([BLOCK_C], dtype=tl.float32)
    th_const = tl.load(TH + c, mask=cmask, other=0.0)
    lm_0 = cmask & (0 + 0 < T)
    lp_0 = (0 + 0) * (B * DN) + b * DN + c
    lx_0 = (0 + 0) * xs0 + b * xs1 + c
    lm_1 = cmask & (0 + 1 < T)
    lp_1 = (0 + 1) * (B * DN) + b * DN + c
    lx_1 = (0 + 1) * xs0 + b * xs1 + c
    lm_2 = cmask & (0 + 2 < T)
    lp_2 = (0 + 2) * (B * DN) + b * DN + c
    lx_2 = (0 + 2) * xs0 + b * xs1 + c
    lm_3 = cmask & (0 + 3 < T)
    lp_3 = (0 + 3) * (B * DN) + b * DN + c
    lx_3 = (0 + 3) * xs0 + b * xs1 + c
    lm_4 = cmask & (0 + 4 < T)
    lp_4 = (0 + 4) * (B * DN) + b * DN + c
    lx_4 = (0 + 4) * xs0 + b * xs1 + c
    lm_5 = cmask & (0 + 5 < T)
    lp_5 = (0 + 5) * (B * DN) + b * DN + c
    lx_5 = (0 + 5) * xs0 + b * xs1 + c
    lm_6 = cmask & (0 + 6 < T)
    lp_6 = (0 + 6) * (B * DN) + b * DN + c
    lx_6 = (0 + 6) * xs0 + b * xs1 + c
    lm_7 = cmask & (0 + 7 < T)
    lp_7 = (0 + 7) * (B * DN) + b * DN + c
    lx_7 = (0 + 7) * xs0 + b * xs1 + c
    lm_8 = cmask & (0 + 8 < T)
    lp_8 = (0 + 8) * (B * DN) + b * DN + c
    lx_8 = (0 + 8) * xs0 + b * xs1 + c
    lm_9 = cmask & (0 + 9 < T)
    lp_9 = (0 + 9) * (B * DN) + b * DN + c
    lx_9 = (0 + 9) * xs0 + b * xs1 + c
    lm_10 = cmask & (0 + 10 < T)
    lp_10 = (0 + 10) * (B * DN) + b * DN + c
    lx_10 = (0 + 10) * xs0 + b * xs1 + c
    lm_11 = cmask & (0 + 11 < T)
    lp_11 = (0 + 11) * (B * DN) + b * DN + c
    lx_11 = (0 + 11) * xs0 + b * xs1 + c
    lm_12 = cmask & (0 + 12 < T)
    lp_12 = (0 + 12) * (B * DN) + b * DN + c
    lx_12 = (0 + 12) * xs0 + b * xs1 + c
    lm_13 = cmask & (0 + 13 < T)
    lp_13 = (0 + 13) * (B * DN) + b * DN + c
    lx_13 = (0 + 13) * xs0 + b * xs1 + c
    lm_14 = cmask & (0 + 14 < T)
    lp_14 = (0 + 14) * (B * DN) + b * DN + c
    lx_14 = (0 + 14) * xs0 + b * xs1 + c
    lm_15 = cmask & (0 + 15 < T)
    lp_15 = (0 + 15) * (B * DN) + b * DN + c
    lx_15 = (0 + 15) * xs0 + b * xs1 + c
    lm_16 = cmask & (0 + 16 < T)
    lp_16 = (0 + 16) * (B * DN) + b * DN + c
    lx_16 = (0 + 16) * xs0 + b * xs1 + c
    lm_17 = cmask & (0 + 17 < T)
    lp_17 = (0 + 17) * (B * DN) + b * DN + c
    lx_17 = (0 + 17) * xs0 + b * xs1 + c
    lm_18 = cmask & (0 + 18 < T)
    lp_18 = (0 + 18) * (B * DN) + b * DN + c
    lx_18 = (0 + 18) * xs0 + b * xs1 + c
    lm_19 = cmask & (0 + 19 < T)
    lp_19 = (0 + 19) * (B * DN) + b * DN + c
    lx_19 = (0 + 19) * xs0 + b * xs1 + c
    lm_20 = cmask & (0 + 20 < T)
    lp_20 = (0 + 20) * (B * DN) + b * DN + c
    lx_20 = (0 + 20) * xs0 + b * xs1 + c
    lm_21 = cmask & (0 + 21 < T)
    lp_21 = (0 + 21) * (B * DN) + b * DN + c
    lx_21 = (0 + 21) * xs0 + b * xs1 + c
    lm_22 = cmask & (0 + 22 < T)
    lp_22 = (0 + 22) * (B * DN) + b * DN + c
    lx_22 = (0 + 22) * xs0 + b * xs1 + c
    lm_23 = cmask & (0 + 23 < T)
    lp_23 = (0 + 23) * (B * DN) + b * DN + c
    lx_23 = (0 + 23) * xs0 + b * xs1 + c
    lm_24 = cmask & (0 + 24 < T)
    lp_24 = (0 + 24) * (B * DN) + b * DN + c
    lx_24 = (0 + 24) * xs0 + b * xs1 + c
    lm_25 = cmask & (0 + 25 < T)
    lp_25 = (0 + 25) * (B * DN) + b * DN + c
    lx_25 = (0 + 25) * xs0 + b * xs1 + c
    lm_26 = cmask & (0 + 26 < T)
    lp_26 = (0 + 26) * (B * DN) + b * DN + c
    lx_26 = (0 + 26) * xs0 + b * xs1 + c
    lm_27 = cmask & (0 + 27 < T)
    lp_27 = (0 + 27) * (B * DN) + b * DN + c
    lx_27 = (0 + 27) * xs0 + b * xs1 + c
    lm_28 = cmask & (0 + 28 < T)
    lp_28 = (0 + 28) * (B * DN) + b * DN + c
    lx_28 = (0 + 28) * xs0 + b * xs1 + c
    lm_29 = cmask & (0 + 29 < T)
    lp_29 = (0 + 29) * (B * DN) + b * DN + c
    lx_29 = (0 + 29) * xs0 + b * xs1 + c
    lm_30 = cmask & (0 + 30 < T)
    lp_30 = (0 + 30) * (B * DN) + b * DN + c
    lx_30 = (0 + 30) * xs0 + b * xs1 + c
    lm_31 = cmask & (0 + 31 < T)
    lp_31 = (0 + 31) * (B * DN) + b * DN + c
    lx_31 = (0 + 31) * xs0 + b * xs1 + c
    e_0_0 = tl.load(ETA + lp_0, mask=lm_0, other=0.0)
    if X_BF16:
        x_0_0 = _pg_w(tl.load(X + lx_0, mask=lm_0, other=0.0))
    else:
        x_0_0 = tl.load(X + lx_0, mask=lm_0, other=0.0)
    e_0_1 = tl.load(ETA + lp_1, mask=lm_1, other=0.0)
    if X_BF16:
        x_0_1 = _pg_w(tl.load(X + lx_1, mask=lm_1, other=0.0))
    else:
        x_0_1 = tl.load(X + lx_1, mask=lm_1, other=0.0)
    e_0_2 = tl.load(ETA + lp_2, mask=lm_2, other=0.0)
    if X_BF16:
        x_0_2 = _pg_w(tl.load(X + lx_2, mask=lm_2, other=0.0))
    else:
        x_0_2 = tl.load(X + lx_2, mask=lm_2, other=0.0)
    e_0_3 = tl.load(ETA + lp_3, mask=lm_3, other=0.0)
    if X_BF16:
        x_0_3 = _pg_w(tl.load(X + lx_3, mask=lm_3, other=0.0))
    else:
        x_0_3 = tl.load(X + lx_3, mask=lm_3, other=0.0)
    e_0_4 = tl.load(ETA + lp_4, mask=lm_4, other=0.0)
    if X_BF16:
        x_0_4 = _pg_w(tl.load(X + lx_4, mask=lm_4, other=0.0))
    else:
        x_0_4 = tl.load(X + lx_4, mask=lm_4, other=0.0)
    e_0_5 = tl.load(ETA + lp_5, mask=lm_5, other=0.0)
    if X_BF16:
        x_0_5 = _pg_w(tl.load(X + lx_5, mask=lm_5, other=0.0))
    else:
        x_0_5 = tl.load(X + lx_5, mask=lm_5, other=0.0)
    e_0_6 = tl.load(ETA + lp_6, mask=lm_6, other=0.0)
    if X_BF16:
        x_0_6 = _pg_w(tl.load(X + lx_6, mask=lm_6, other=0.0))
    else:
        x_0_6 = tl.load(X + lx_6, mask=lm_6, other=0.0)
    e_0_7 = tl.load(ETA + lp_7, mask=lm_7, other=0.0)
    if X_BF16:
        x_0_7 = _pg_w(tl.load(X + lx_7, mask=lm_7, other=0.0))
    else:
        x_0_7 = tl.load(X + lx_7, mask=lm_7, other=0.0)
    e_0_8 = tl.load(ETA + lp_8, mask=lm_8, other=0.0)
    if X_BF16:
        x_0_8 = _pg_w(tl.load(X + lx_8, mask=lm_8, other=0.0))
    else:
        x_0_8 = tl.load(X + lx_8, mask=lm_8, other=0.0)
    e_0_9 = tl.load(ETA + lp_9, mask=lm_9, other=0.0)
    if X_BF16:
        x_0_9 = _pg_w(tl.load(X + lx_9, mask=lm_9, other=0.0))
    else:
        x_0_9 = tl.load(X + lx_9, mask=lm_9, other=0.0)
    e_0_10 = tl.load(ETA + lp_10, mask=lm_10, other=0.0)
    if X_BF16:
        x_0_10 = _pg_w(tl.load(X + lx_10, mask=lm_10, other=0.0))
    else:
        x_0_10 = tl.load(X + lx_10, mask=lm_10, other=0.0)
    e_0_11 = tl.load(ETA + lp_11, mask=lm_11, other=0.0)
    if X_BF16:
        x_0_11 = _pg_w(tl.load(X + lx_11, mask=lm_11, other=0.0))
    else:
        x_0_11 = tl.load(X + lx_11, mask=lm_11, other=0.0)
    e_0_12 = tl.load(ETA + lp_12, mask=lm_12, other=0.0)
    if X_BF16:
        x_0_12 = _pg_w(tl.load(X + lx_12, mask=lm_12, other=0.0))
    else:
        x_0_12 = tl.load(X + lx_12, mask=lm_12, other=0.0)
    e_0_13 = tl.load(ETA + lp_13, mask=lm_13, other=0.0)
    if X_BF16:
        x_0_13 = _pg_w(tl.load(X + lx_13, mask=lm_13, other=0.0))
    else:
        x_0_13 = tl.load(X + lx_13, mask=lm_13, other=0.0)
    e_0_14 = tl.load(ETA + lp_14, mask=lm_14, other=0.0)
    if X_BF16:
        x_0_14 = _pg_w(tl.load(X + lx_14, mask=lm_14, other=0.0))
    else:
        x_0_14 = tl.load(X + lx_14, mask=lm_14, other=0.0)
    e_0_15 = tl.load(ETA + lp_15, mask=lm_15, other=0.0)
    if X_BF16:
        x_0_15 = _pg_w(tl.load(X + lx_15, mask=lm_15, other=0.0))
    else:
        x_0_15 = tl.load(X + lx_15, mask=lm_15, other=0.0)
    e_0_16 = tl.load(ETA + lp_16, mask=lm_16, other=0.0)
    if X_BF16:
        x_0_16 = _pg_w(tl.load(X + lx_16, mask=lm_16, other=0.0))
    else:
        x_0_16 = tl.load(X + lx_16, mask=lm_16, other=0.0)
    e_0_17 = tl.load(ETA + lp_17, mask=lm_17, other=0.0)
    if X_BF16:
        x_0_17 = _pg_w(tl.load(X + lx_17, mask=lm_17, other=0.0))
    else:
        x_0_17 = tl.load(X + lx_17, mask=lm_17, other=0.0)
    e_0_18 = tl.load(ETA + lp_18, mask=lm_18, other=0.0)
    if X_BF16:
        x_0_18 = _pg_w(tl.load(X + lx_18, mask=lm_18, other=0.0))
    else:
        x_0_18 = tl.load(X + lx_18, mask=lm_18, other=0.0)
    e_0_19 = tl.load(ETA + lp_19, mask=lm_19, other=0.0)
    if X_BF16:
        x_0_19 = _pg_w(tl.load(X + lx_19, mask=lm_19, other=0.0))
    else:
        x_0_19 = tl.load(X + lx_19, mask=lm_19, other=0.0)
    e_0_20 = tl.load(ETA + lp_20, mask=lm_20, other=0.0)
    if X_BF16:
        x_0_20 = _pg_w(tl.load(X + lx_20, mask=lm_20, other=0.0))
    else:
        x_0_20 = tl.load(X + lx_20, mask=lm_20, other=0.0)
    e_0_21 = tl.load(ETA + lp_21, mask=lm_21, other=0.0)
    if X_BF16:
        x_0_21 = _pg_w(tl.load(X + lx_21, mask=lm_21, other=0.0))
    else:
        x_0_21 = tl.load(X + lx_21, mask=lm_21, other=0.0)
    e_0_22 = tl.load(ETA + lp_22, mask=lm_22, other=0.0)
    if X_BF16:
        x_0_22 = _pg_w(tl.load(X + lx_22, mask=lm_22, other=0.0))
    else:
        x_0_22 = tl.load(X + lx_22, mask=lm_22, other=0.0)
    e_0_23 = tl.load(ETA + lp_23, mask=lm_23, other=0.0)
    if X_BF16:
        x_0_23 = _pg_w(tl.load(X + lx_23, mask=lm_23, other=0.0))
    else:
        x_0_23 = tl.load(X + lx_23, mask=lm_23, other=0.0)
    e_0_24 = tl.load(ETA + lp_24, mask=lm_24, other=0.0)
    if X_BF16:
        x_0_24 = _pg_w(tl.load(X + lx_24, mask=lm_24, other=0.0))
    else:
        x_0_24 = tl.load(X + lx_24, mask=lm_24, other=0.0)
    e_0_25 = tl.load(ETA + lp_25, mask=lm_25, other=0.0)
    if X_BF16:
        x_0_25 = _pg_w(tl.load(X + lx_25, mask=lm_25, other=0.0))
    else:
        x_0_25 = tl.load(X + lx_25, mask=lm_25, other=0.0)
    e_0_26 = tl.load(ETA + lp_26, mask=lm_26, other=0.0)
    if X_BF16:
        x_0_26 = _pg_w(tl.load(X + lx_26, mask=lm_26, other=0.0))
    else:
        x_0_26 = tl.load(X + lx_26, mask=lm_26, other=0.0)
    e_0_27 = tl.load(ETA + lp_27, mask=lm_27, other=0.0)
    if X_BF16:
        x_0_27 = _pg_w(tl.load(X + lx_27, mask=lm_27, other=0.0))
    else:
        x_0_27 = tl.load(X + lx_27, mask=lm_27, other=0.0)
    e_0_28 = tl.load(ETA + lp_28, mask=lm_28, other=0.0)
    if X_BF16:
        x_0_28 = _pg_w(tl.load(X + lx_28, mask=lm_28, other=0.0))
    else:
        x_0_28 = tl.load(X + lx_28, mask=lm_28, other=0.0)
    e_0_29 = tl.load(ETA + lp_29, mask=lm_29, other=0.0)
    if X_BF16:
        x_0_29 = _pg_w(tl.load(X + lx_29, mask=lm_29, other=0.0))
    else:
        x_0_29 = tl.load(X + lx_29, mask=lm_29, other=0.0)
    e_0_30 = tl.load(ETA + lp_30, mask=lm_30, other=0.0)
    if X_BF16:
        x_0_30 = _pg_w(tl.load(X + lx_30, mask=lm_30, other=0.0))
    else:
        x_0_30 = tl.load(X + lx_30, mask=lm_30, other=0.0)
    e_0_31 = tl.load(ETA + lp_31, mask=lm_31, other=0.0)
    if X_BF16:
        x_0_31 = _pg_w(tl.load(X + lx_31, mask=lm_31, other=0.0))
    else:
        x_0_31 = tl.load(X + lx_31, mask=lm_31, other=0.0)
    for tile in range(0, T, 32):
        lm_0 = cmask & (tile + 32 + 0 < T)
        lp_0 = (tile + 32 + 0) * (B * DN) + b * DN + c
        lx_0 = (tile + 32 + 0) * xs0 + b * xs1 + c
        lm_1 = cmask & (tile + 32 + 1 < T)
        lp_1 = (tile + 32 + 1) * (B * DN) + b * DN + c
        lx_1 = (tile + 32 + 1) * xs0 + b * xs1 + c
        lm_2 = cmask & (tile + 32 + 2 < T)
        lp_2 = (tile + 32 + 2) * (B * DN) + b * DN + c
        lx_2 = (tile + 32 + 2) * xs0 + b * xs1 + c
        lm_3 = cmask & (tile + 32 + 3 < T)
        lp_3 = (tile + 32 + 3) * (B * DN) + b * DN + c
        lx_3 = (tile + 32 + 3) * xs0 + b * xs1 + c
        lm_4 = cmask & (tile + 32 + 4 < T)
        lp_4 = (tile + 32 + 4) * (B * DN) + b * DN + c
        lx_4 = (tile + 32 + 4) * xs0 + b * xs1 + c
        lm_5 = cmask & (tile + 32 + 5 < T)
        lp_5 = (tile + 32 + 5) * (B * DN) + b * DN + c
        lx_5 = (tile + 32 + 5) * xs0 + b * xs1 + c
        lm_6 = cmask & (tile + 32 + 6 < T)
        lp_6 = (tile + 32 + 6) * (B * DN) + b * DN + c
        lx_6 = (tile + 32 + 6) * xs0 + b * xs1 + c
        lm_7 = cmask & (tile + 32 + 7 < T)
        lp_7 = (tile + 32 + 7) * (B * DN) + b * DN + c
        lx_7 = (tile + 32 + 7) * xs0 + b * xs1 + c
        lm_8 = cmask & (tile + 32 + 8 < T)
        lp_8 = (tile + 32 + 8) * (B * DN) + b * DN + c
        lx_8 = (tile + 32 + 8) * xs0 + b * xs1 + c
        lm_9 = cmask & (tile + 32 + 9 < T)
        lp_9 = (tile + 32 + 9) * (B * DN) + b * DN + c
        lx_9 = (tile + 32 + 9) * xs0 + b * xs1 + c
        lm_10 = cmask & (tile + 32 + 10 < T)
        lp_10 = (tile + 32 + 10) * (B * DN) + b * DN + c
        lx_10 = (tile + 32 + 10) * xs0 + b * xs1 + c
        lm_11 = cmask & (tile + 32 + 11 < T)
        lp_11 = (tile + 32 + 11) * (B * DN) + b * DN + c
        lx_11 = (tile + 32 + 11) * xs0 + b * xs1 + c
        lm_12 = cmask & (tile + 32 + 12 < T)
        lp_12 = (tile + 32 + 12) * (B * DN) + b * DN + c
        lx_12 = (tile + 32 + 12) * xs0 + b * xs1 + c
        lm_13 = cmask & (tile + 32 + 13 < T)
        lp_13 = (tile + 32 + 13) * (B * DN) + b * DN + c
        lx_13 = (tile + 32 + 13) * xs0 + b * xs1 + c
        lm_14 = cmask & (tile + 32 + 14 < T)
        lp_14 = (tile + 32 + 14) * (B * DN) + b * DN + c
        lx_14 = (tile + 32 + 14) * xs0 + b * xs1 + c
        lm_15 = cmask & (tile + 32 + 15 < T)
        lp_15 = (tile + 32 + 15) * (B * DN) + b * DN + c
        lx_15 = (tile + 32 + 15) * xs0 + b * xs1 + c
        lm_16 = cmask & (tile + 32 + 16 < T)
        lp_16 = (tile + 32 + 16) * (B * DN) + b * DN + c
        lx_16 = (tile + 32 + 16) * xs0 + b * xs1 + c
        lm_17 = cmask & (tile + 32 + 17 < T)
        lp_17 = (tile + 32 + 17) * (B * DN) + b * DN + c
        lx_17 = (tile + 32 + 17) * xs0 + b * xs1 + c
        lm_18 = cmask & (tile + 32 + 18 < T)
        lp_18 = (tile + 32 + 18) * (B * DN) + b * DN + c
        lx_18 = (tile + 32 + 18) * xs0 + b * xs1 + c
        lm_19 = cmask & (tile + 32 + 19 < T)
        lp_19 = (tile + 32 + 19) * (B * DN) + b * DN + c
        lx_19 = (tile + 32 + 19) * xs0 + b * xs1 + c
        lm_20 = cmask & (tile + 32 + 20 < T)
        lp_20 = (tile + 32 + 20) * (B * DN) + b * DN + c
        lx_20 = (tile + 32 + 20) * xs0 + b * xs1 + c
        lm_21 = cmask & (tile + 32 + 21 < T)
        lp_21 = (tile + 32 + 21) * (B * DN) + b * DN + c
        lx_21 = (tile + 32 + 21) * xs0 + b * xs1 + c
        lm_22 = cmask & (tile + 32 + 22 < T)
        lp_22 = (tile + 32 + 22) * (B * DN) + b * DN + c
        lx_22 = (tile + 32 + 22) * xs0 + b * xs1 + c
        lm_23 = cmask & (tile + 32 + 23 < T)
        lp_23 = (tile + 32 + 23) * (B * DN) + b * DN + c
        lx_23 = (tile + 32 + 23) * xs0 + b * xs1 + c
        lm_24 = cmask & (tile + 32 + 24 < T)
        lp_24 = (tile + 32 + 24) * (B * DN) + b * DN + c
        lx_24 = (tile + 32 + 24) * xs0 + b * xs1 + c
        lm_25 = cmask & (tile + 32 + 25 < T)
        lp_25 = (tile + 32 + 25) * (B * DN) + b * DN + c
        lx_25 = (tile + 32 + 25) * xs0 + b * xs1 + c
        lm_26 = cmask & (tile + 32 + 26 < T)
        lp_26 = (tile + 32 + 26) * (B * DN) + b * DN + c
        lx_26 = (tile + 32 + 26) * xs0 + b * xs1 + c
        lm_27 = cmask & (tile + 32 + 27 < T)
        lp_27 = (tile + 32 + 27) * (B * DN) + b * DN + c
        lx_27 = (tile + 32 + 27) * xs0 + b * xs1 + c
        lm_28 = cmask & (tile + 32 + 28 < T)
        lp_28 = (tile + 32 + 28) * (B * DN) + b * DN + c
        lx_28 = (tile + 32 + 28) * xs0 + b * xs1 + c
        lm_29 = cmask & (tile + 32 + 29 < T)
        lp_29 = (tile + 32 + 29) * (B * DN) + b * DN + c
        lx_29 = (tile + 32 + 29) * xs0 + b * xs1 + c
        lm_30 = cmask & (tile + 32 + 30 < T)
        lp_30 = (tile + 32 + 30) * (B * DN) + b * DN + c
        lx_30 = (tile + 32 + 30) * xs0 + b * xs1 + c
        lm_31 = cmask & (tile + 32 + 31 < T)
        lp_31 = (tile + 32 + 31) * (B * DN) + b * DN + c
        lx_31 = (tile + 32 + 31) * xs0 + b * xs1 + c
        e_1_0 = tl.load(ETA + lp_0, mask=lm_0, other=0.0)
        if X_BF16:
            x_1_0 = _pg_w(tl.load(X + lx_0, mask=lm_0, other=0.0))
        else:
            x_1_0 = tl.load(X + lx_0, mask=lm_0, other=0.0)
        e_1_1 = tl.load(ETA + lp_1, mask=lm_1, other=0.0)
        if X_BF16:
            x_1_1 = _pg_w(tl.load(X + lx_1, mask=lm_1, other=0.0))
        else:
            x_1_1 = tl.load(X + lx_1, mask=lm_1, other=0.0)
        e_1_2 = tl.load(ETA + lp_2, mask=lm_2, other=0.0)
        if X_BF16:
            x_1_2 = _pg_w(tl.load(X + lx_2, mask=lm_2, other=0.0))
        else:
            x_1_2 = tl.load(X + lx_2, mask=lm_2, other=0.0)
        e_1_3 = tl.load(ETA + lp_3, mask=lm_3, other=0.0)
        if X_BF16:
            x_1_3 = _pg_w(tl.load(X + lx_3, mask=lm_3, other=0.0))
        else:
            x_1_3 = tl.load(X + lx_3, mask=lm_3, other=0.0)
        e_1_4 = tl.load(ETA + lp_4, mask=lm_4, other=0.0)
        if X_BF16:
            x_1_4 = _pg_w(tl.load(X + lx_4, mask=lm_4, other=0.0))
        else:
            x_1_4 = tl.load(X + lx_4, mask=lm_4, other=0.0)
        e_1_5 = tl.load(ETA + lp_5, mask=lm_5, other=0.0)
        if X_BF16:
            x_1_5 = _pg_w(tl.load(X + lx_5, mask=lm_5, other=0.0))
        else:
            x_1_5 = tl.load(X + lx_5, mask=lm_5, other=0.0)
        e_1_6 = tl.load(ETA + lp_6, mask=lm_6, other=0.0)
        if X_BF16:
            x_1_6 = _pg_w(tl.load(X + lx_6, mask=lm_6, other=0.0))
        else:
            x_1_6 = tl.load(X + lx_6, mask=lm_6, other=0.0)
        e_1_7 = tl.load(ETA + lp_7, mask=lm_7, other=0.0)
        if X_BF16:
            x_1_7 = _pg_w(tl.load(X + lx_7, mask=lm_7, other=0.0))
        else:
            x_1_7 = tl.load(X + lx_7, mask=lm_7, other=0.0)
        e_1_8 = tl.load(ETA + lp_8, mask=lm_8, other=0.0)
        if X_BF16:
            x_1_8 = _pg_w(tl.load(X + lx_8, mask=lm_8, other=0.0))
        else:
            x_1_8 = tl.load(X + lx_8, mask=lm_8, other=0.0)
        e_1_9 = tl.load(ETA + lp_9, mask=lm_9, other=0.0)
        if X_BF16:
            x_1_9 = _pg_w(tl.load(X + lx_9, mask=lm_9, other=0.0))
        else:
            x_1_9 = tl.load(X + lx_9, mask=lm_9, other=0.0)
        e_1_10 = tl.load(ETA + lp_10, mask=lm_10, other=0.0)
        if X_BF16:
            x_1_10 = _pg_w(tl.load(X + lx_10, mask=lm_10, other=0.0))
        else:
            x_1_10 = tl.load(X + lx_10, mask=lm_10, other=0.0)
        e_1_11 = tl.load(ETA + lp_11, mask=lm_11, other=0.0)
        if X_BF16:
            x_1_11 = _pg_w(tl.load(X + lx_11, mask=lm_11, other=0.0))
        else:
            x_1_11 = tl.load(X + lx_11, mask=lm_11, other=0.0)
        e_1_12 = tl.load(ETA + lp_12, mask=lm_12, other=0.0)
        if X_BF16:
            x_1_12 = _pg_w(tl.load(X + lx_12, mask=lm_12, other=0.0))
        else:
            x_1_12 = tl.load(X + lx_12, mask=lm_12, other=0.0)
        e_1_13 = tl.load(ETA + lp_13, mask=lm_13, other=0.0)
        if X_BF16:
            x_1_13 = _pg_w(tl.load(X + lx_13, mask=lm_13, other=0.0))
        else:
            x_1_13 = tl.load(X + lx_13, mask=lm_13, other=0.0)
        e_1_14 = tl.load(ETA + lp_14, mask=lm_14, other=0.0)
        if X_BF16:
            x_1_14 = _pg_w(tl.load(X + lx_14, mask=lm_14, other=0.0))
        else:
            x_1_14 = tl.load(X + lx_14, mask=lm_14, other=0.0)
        e_1_15 = tl.load(ETA + lp_15, mask=lm_15, other=0.0)
        if X_BF16:
            x_1_15 = _pg_w(tl.load(X + lx_15, mask=lm_15, other=0.0))
        else:
            x_1_15 = tl.load(X + lx_15, mask=lm_15, other=0.0)
        e_1_16 = tl.load(ETA + lp_16, mask=lm_16, other=0.0)
        if X_BF16:
            x_1_16 = _pg_w(tl.load(X + lx_16, mask=lm_16, other=0.0))
        else:
            x_1_16 = tl.load(X + lx_16, mask=lm_16, other=0.0)
        e_1_17 = tl.load(ETA + lp_17, mask=lm_17, other=0.0)
        if X_BF16:
            x_1_17 = _pg_w(tl.load(X + lx_17, mask=lm_17, other=0.0))
        else:
            x_1_17 = tl.load(X + lx_17, mask=lm_17, other=0.0)
        e_1_18 = tl.load(ETA + lp_18, mask=lm_18, other=0.0)
        if X_BF16:
            x_1_18 = _pg_w(tl.load(X + lx_18, mask=lm_18, other=0.0))
        else:
            x_1_18 = tl.load(X + lx_18, mask=lm_18, other=0.0)
        e_1_19 = tl.load(ETA + lp_19, mask=lm_19, other=0.0)
        if X_BF16:
            x_1_19 = _pg_w(tl.load(X + lx_19, mask=lm_19, other=0.0))
        else:
            x_1_19 = tl.load(X + lx_19, mask=lm_19, other=0.0)
        e_1_20 = tl.load(ETA + lp_20, mask=lm_20, other=0.0)
        if X_BF16:
            x_1_20 = _pg_w(tl.load(X + lx_20, mask=lm_20, other=0.0))
        else:
            x_1_20 = tl.load(X + lx_20, mask=lm_20, other=0.0)
        e_1_21 = tl.load(ETA + lp_21, mask=lm_21, other=0.0)
        if X_BF16:
            x_1_21 = _pg_w(tl.load(X + lx_21, mask=lm_21, other=0.0))
        else:
            x_1_21 = tl.load(X + lx_21, mask=lm_21, other=0.0)
        e_1_22 = tl.load(ETA + lp_22, mask=lm_22, other=0.0)
        if X_BF16:
            x_1_22 = _pg_w(tl.load(X + lx_22, mask=lm_22, other=0.0))
        else:
            x_1_22 = tl.load(X + lx_22, mask=lm_22, other=0.0)
        e_1_23 = tl.load(ETA + lp_23, mask=lm_23, other=0.0)
        if X_BF16:
            x_1_23 = _pg_w(tl.load(X + lx_23, mask=lm_23, other=0.0))
        else:
            x_1_23 = tl.load(X + lx_23, mask=lm_23, other=0.0)
        e_1_24 = tl.load(ETA + lp_24, mask=lm_24, other=0.0)
        if X_BF16:
            x_1_24 = _pg_w(tl.load(X + lx_24, mask=lm_24, other=0.0))
        else:
            x_1_24 = tl.load(X + lx_24, mask=lm_24, other=0.0)
        e_1_25 = tl.load(ETA + lp_25, mask=lm_25, other=0.0)
        if X_BF16:
            x_1_25 = _pg_w(tl.load(X + lx_25, mask=lm_25, other=0.0))
        else:
            x_1_25 = tl.load(X + lx_25, mask=lm_25, other=0.0)
        e_1_26 = tl.load(ETA + lp_26, mask=lm_26, other=0.0)
        if X_BF16:
            x_1_26 = _pg_w(tl.load(X + lx_26, mask=lm_26, other=0.0))
        else:
            x_1_26 = tl.load(X + lx_26, mask=lm_26, other=0.0)
        e_1_27 = tl.load(ETA + lp_27, mask=lm_27, other=0.0)
        if X_BF16:
            x_1_27 = _pg_w(tl.load(X + lx_27, mask=lm_27, other=0.0))
        else:
            x_1_27 = tl.load(X + lx_27, mask=lm_27, other=0.0)
        e_1_28 = tl.load(ETA + lp_28, mask=lm_28, other=0.0)
        if X_BF16:
            x_1_28 = _pg_w(tl.load(X + lx_28, mask=lm_28, other=0.0))
        else:
            x_1_28 = tl.load(X + lx_28, mask=lm_28, other=0.0)
        e_1_29 = tl.load(ETA + lp_29, mask=lm_29, other=0.0)
        if X_BF16:
            x_1_29 = _pg_w(tl.load(X + lx_29, mask=lm_29, other=0.0))
        else:
            x_1_29 = tl.load(X + lx_29, mask=lm_29, other=0.0)
        e_1_30 = tl.load(ETA + lp_30, mask=lm_30, other=0.0)
        if X_BF16:
            x_1_30 = _pg_w(tl.load(X + lx_30, mask=lm_30, other=0.0))
        else:
            x_1_30 = tl.load(X + lx_30, mask=lm_30, other=0.0)
        e_1_31 = tl.load(ETA + lp_31, mask=lm_31, other=0.0)
        if X_BF16:
            x_1_31 = _pg_w(tl.load(X + lx_31, mask=lm_31, other=0.0))
        else:
            x_1_31 = tl.load(X + lx_31, mask=lm_31, other=0.0)
        a_0_0 = _pg_sub(1.0, e_0_0)
        i_0_0 = _pg_mul(e_0_0, x_0_0)
        a_0_1 = _pg_sub(1.0, e_0_1)
        i_0_1 = _pg_mul(e_0_1, x_0_1)
        a_0_2 = _pg_sub(1.0, e_0_2)
        i_0_2 = _pg_mul(e_0_2, x_0_2)
        a_0_3 = _pg_sub(1.0, e_0_3)
        i_0_3 = _pg_mul(e_0_3, x_0_3)
        a_0_4 = _pg_sub(1.0, e_0_4)
        i_0_4 = _pg_mul(e_0_4, x_0_4)
        a_0_5 = _pg_sub(1.0, e_0_5)
        i_0_5 = _pg_mul(e_0_5, x_0_5)
        a_0_6 = _pg_sub(1.0, e_0_6)
        i_0_6 = _pg_mul(e_0_6, x_0_6)
        a_0_7 = _pg_sub(1.0, e_0_7)
        i_0_7 = _pg_mul(e_0_7, x_0_7)
        a_0_8 = _pg_sub(1.0, e_0_8)
        i_0_8 = _pg_mul(e_0_8, x_0_8)
        a_0_9 = _pg_sub(1.0, e_0_9)
        i_0_9 = _pg_mul(e_0_9, x_0_9)
        a_0_10 = _pg_sub(1.0, e_0_10)
        i_0_10 = _pg_mul(e_0_10, x_0_10)
        a_0_11 = _pg_sub(1.0, e_0_11)
        i_0_11 = _pg_mul(e_0_11, x_0_11)
        a_0_12 = _pg_sub(1.0, e_0_12)
        i_0_12 = _pg_mul(e_0_12, x_0_12)
        a_0_13 = _pg_sub(1.0, e_0_13)
        i_0_13 = _pg_mul(e_0_13, x_0_13)
        a_0_14 = _pg_sub(1.0, e_0_14)
        i_0_14 = _pg_mul(e_0_14, x_0_14)
        a_0_15 = _pg_sub(1.0, e_0_15)
        i_0_15 = _pg_mul(e_0_15, x_0_15)
        a_0_16 = _pg_sub(1.0, e_0_16)
        i_0_16 = _pg_mul(e_0_16, x_0_16)
        a_0_17 = _pg_sub(1.0, e_0_17)
        i_0_17 = _pg_mul(e_0_17, x_0_17)
        a_0_18 = _pg_sub(1.0, e_0_18)
        i_0_18 = _pg_mul(e_0_18, x_0_18)
        a_0_19 = _pg_sub(1.0, e_0_19)
        i_0_19 = _pg_mul(e_0_19, x_0_19)
        a_0_20 = _pg_sub(1.0, e_0_20)
        i_0_20 = _pg_mul(e_0_20, x_0_20)
        a_0_21 = _pg_sub(1.0, e_0_21)
        i_0_21 = _pg_mul(e_0_21, x_0_21)
        a_0_22 = _pg_sub(1.0, e_0_22)
        i_0_22 = _pg_mul(e_0_22, x_0_22)
        a_0_23 = _pg_sub(1.0, e_0_23)
        i_0_23 = _pg_mul(e_0_23, x_0_23)
        a_0_24 = _pg_sub(1.0, e_0_24)
        i_0_24 = _pg_mul(e_0_24, x_0_24)
        a_0_25 = _pg_sub(1.0, e_0_25)
        i_0_25 = _pg_mul(e_0_25, x_0_25)
        a_0_26 = _pg_sub(1.0, e_0_26)
        i_0_26 = _pg_mul(e_0_26, x_0_26)
        a_0_27 = _pg_sub(1.0, e_0_27)
        i_0_27 = _pg_mul(e_0_27, x_0_27)
        a_0_28 = _pg_sub(1.0, e_0_28)
        i_0_28 = _pg_mul(e_0_28, x_0_28)
        a_0_29 = _pg_sub(1.0, e_0_29)
        i_0_29 = _pg_mul(e_0_29, x_0_29)
        a_0_30 = _pg_sub(1.0, e_0_30)
        i_0_30 = _pg_mul(e_0_30, x_0_30)
        a_0_31 = _pg_sub(1.0, e_0_31)
        i_0_31 = _pg_mul(e_0_31, x_0_31)
        m_0_0 = cmask & (tile + 0 < T)
        p_0_0 = (tile + 0) * (B * DN) + b * DN + c
        pre = _rn_add_f32(_rn_mul_f32(a_0_0, v), i_0_0)
        s = tl.maximum(pre - th_const, 0.0)
        if BIPOLAR:
            s = s - tl.maximum(-pre - th_const, 0.0)
        v = tl.where(m_0_0, pre - s, v)
        tl.store(S_OUT + p_0_0, s, mask=m_0_0)
        if STORE_PRE:
            tl.store(PRE_OUT + p_0_0, pre, mask=m_0_0)
        m_0_1 = cmask & (tile + 1 < T)
        p_0_1 = (tile + 1) * (B * DN) + b * DN + c
        pre = _rn_add_f32(_rn_mul_f32(a_0_1, v), i_0_1)
        s = tl.maximum(pre - th_const, 0.0)
        if BIPOLAR:
            s = s - tl.maximum(-pre - th_const, 0.0)
        v = tl.where(m_0_1, pre - s, v)
        tl.store(S_OUT + p_0_1, s, mask=m_0_1)
        if STORE_PRE:
            tl.store(PRE_OUT + p_0_1, pre, mask=m_0_1)
        m_0_2 = cmask & (tile + 2 < T)
        p_0_2 = (tile + 2) * (B * DN) + b * DN + c
        pre = _rn_add_f32(_rn_mul_f32(a_0_2, v), i_0_2)
        s = tl.maximum(pre - th_const, 0.0)
        if BIPOLAR:
            s = s - tl.maximum(-pre - th_const, 0.0)
        v = tl.where(m_0_2, pre - s, v)
        tl.store(S_OUT + p_0_2, s, mask=m_0_2)
        if STORE_PRE:
            tl.store(PRE_OUT + p_0_2, pre, mask=m_0_2)
        m_0_3 = cmask & (tile + 3 < T)
        p_0_3 = (tile + 3) * (B * DN) + b * DN + c
        pre = _rn_add_f32(_rn_mul_f32(a_0_3, v), i_0_3)
        s = tl.maximum(pre - th_const, 0.0)
        if BIPOLAR:
            s = s - tl.maximum(-pre - th_const, 0.0)
        v = tl.where(m_0_3, pre - s, v)
        tl.store(S_OUT + p_0_3, s, mask=m_0_3)
        if STORE_PRE:
            tl.store(PRE_OUT + p_0_3, pre, mask=m_0_3)
        m_0_4 = cmask & (tile + 4 < T)
        p_0_4 = (tile + 4) * (B * DN) + b * DN + c
        pre = _rn_add_f32(_rn_mul_f32(a_0_4, v), i_0_4)
        s = tl.maximum(pre - th_const, 0.0)
        if BIPOLAR:
            s = s - tl.maximum(-pre - th_const, 0.0)
        v = tl.where(m_0_4, pre - s, v)
        tl.store(S_OUT + p_0_4, s, mask=m_0_4)
        if STORE_PRE:
            tl.store(PRE_OUT + p_0_4, pre, mask=m_0_4)
        m_0_5 = cmask & (tile + 5 < T)
        p_0_5 = (tile + 5) * (B * DN) + b * DN + c
        pre = _rn_add_f32(_rn_mul_f32(a_0_5, v), i_0_5)
        s = tl.maximum(pre - th_const, 0.0)
        if BIPOLAR:
            s = s - tl.maximum(-pre - th_const, 0.0)
        v = tl.where(m_0_5, pre - s, v)
        tl.store(S_OUT + p_0_5, s, mask=m_0_5)
        if STORE_PRE:
            tl.store(PRE_OUT + p_0_5, pre, mask=m_0_5)
        m_0_6 = cmask & (tile + 6 < T)
        p_0_6 = (tile + 6) * (B * DN) + b * DN + c
        pre = _rn_add_f32(_rn_mul_f32(a_0_6, v), i_0_6)
        s = tl.maximum(pre - th_const, 0.0)
        if BIPOLAR:
            s = s - tl.maximum(-pre - th_const, 0.0)
        v = tl.where(m_0_6, pre - s, v)
        tl.store(S_OUT + p_0_6, s, mask=m_0_6)
        if STORE_PRE:
            tl.store(PRE_OUT + p_0_6, pre, mask=m_0_6)
        m_0_7 = cmask & (tile + 7 < T)
        p_0_7 = (tile + 7) * (B * DN) + b * DN + c
        pre = _rn_add_f32(_rn_mul_f32(a_0_7, v), i_0_7)
        s = tl.maximum(pre - th_const, 0.0)
        if BIPOLAR:
            s = s - tl.maximum(-pre - th_const, 0.0)
        v = tl.where(m_0_7, pre - s, v)
        tl.store(S_OUT + p_0_7, s, mask=m_0_7)
        if STORE_PRE:
            tl.store(PRE_OUT + p_0_7, pre, mask=m_0_7)
        m_0_8 = cmask & (tile + 8 < T)
        p_0_8 = (tile + 8) * (B * DN) + b * DN + c
        pre = _rn_add_f32(_rn_mul_f32(a_0_8, v), i_0_8)
        s = tl.maximum(pre - th_const, 0.0)
        if BIPOLAR:
            s = s - tl.maximum(-pre - th_const, 0.0)
        v = tl.where(m_0_8, pre - s, v)
        tl.store(S_OUT + p_0_8, s, mask=m_0_8)
        if STORE_PRE:
            tl.store(PRE_OUT + p_0_8, pre, mask=m_0_8)
        m_0_9 = cmask & (tile + 9 < T)
        p_0_9 = (tile + 9) * (B * DN) + b * DN + c
        pre = _rn_add_f32(_rn_mul_f32(a_0_9, v), i_0_9)
        s = tl.maximum(pre - th_const, 0.0)
        if BIPOLAR:
            s = s - tl.maximum(-pre - th_const, 0.0)
        v = tl.where(m_0_9, pre - s, v)
        tl.store(S_OUT + p_0_9, s, mask=m_0_9)
        if STORE_PRE:
            tl.store(PRE_OUT + p_0_9, pre, mask=m_0_9)
        m_0_10 = cmask & (tile + 10 < T)
        p_0_10 = (tile + 10) * (B * DN) + b * DN + c
        pre = _rn_add_f32(_rn_mul_f32(a_0_10, v), i_0_10)
        s = tl.maximum(pre - th_const, 0.0)
        if BIPOLAR:
            s = s - tl.maximum(-pre - th_const, 0.0)
        v = tl.where(m_0_10, pre - s, v)
        tl.store(S_OUT + p_0_10, s, mask=m_0_10)
        if STORE_PRE:
            tl.store(PRE_OUT + p_0_10, pre, mask=m_0_10)
        m_0_11 = cmask & (tile + 11 < T)
        p_0_11 = (tile + 11) * (B * DN) + b * DN + c
        pre = _rn_add_f32(_rn_mul_f32(a_0_11, v), i_0_11)
        s = tl.maximum(pre - th_const, 0.0)
        if BIPOLAR:
            s = s - tl.maximum(-pre - th_const, 0.0)
        v = tl.where(m_0_11, pre - s, v)
        tl.store(S_OUT + p_0_11, s, mask=m_0_11)
        if STORE_PRE:
            tl.store(PRE_OUT + p_0_11, pre, mask=m_0_11)
        m_0_12 = cmask & (tile + 12 < T)
        p_0_12 = (tile + 12) * (B * DN) + b * DN + c
        pre = _rn_add_f32(_rn_mul_f32(a_0_12, v), i_0_12)
        s = tl.maximum(pre - th_const, 0.0)
        if BIPOLAR:
            s = s - tl.maximum(-pre - th_const, 0.0)
        v = tl.where(m_0_12, pre - s, v)
        tl.store(S_OUT + p_0_12, s, mask=m_0_12)
        if STORE_PRE:
            tl.store(PRE_OUT + p_0_12, pre, mask=m_0_12)
        m_0_13 = cmask & (tile + 13 < T)
        p_0_13 = (tile + 13) * (B * DN) + b * DN + c
        pre = _rn_add_f32(_rn_mul_f32(a_0_13, v), i_0_13)
        s = tl.maximum(pre - th_const, 0.0)
        if BIPOLAR:
            s = s - tl.maximum(-pre - th_const, 0.0)
        v = tl.where(m_0_13, pre - s, v)
        tl.store(S_OUT + p_0_13, s, mask=m_0_13)
        if STORE_PRE:
            tl.store(PRE_OUT + p_0_13, pre, mask=m_0_13)
        m_0_14 = cmask & (tile + 14 < T)
        p_0_14 = (tile + 14) * (B * DN) + b * DN + c
        pre = _rn_add_f32(_rn_mul_f32(a_0_14, v), i_0_14)
        s = tl.maximum(pre - th_const, 0.0)
        if BIPOLAR:
            s = s - tl.maximum(-pre - th_const, 0.0)
        v = tl.where(m_0_14, pre - s, v)
        tl.store(S_OUT + p_0_14, s, mask=m_0_14)
        if STORE_PRE:
            tl.store(PRE_OUT + p_0_14, pre, mask=m_0_14)
        m_0_15 = cmask & (tile + 15 < T)
        p_0_15 = (tile + 15) * (B * DN) + b * DN + c
        pre = _rn_add_f32(_rn_mul_f32(a_0_15, v), i_0_15)
        s = tl.maximum(pre - th_const, 0.0)
        if BIPOLAR:
            s = s - tl.maximum(-pre - th_const, 0.0)
        v = tl.where(m_0_15, pre - s, v)
        tl.store(S_OUT + p_0_15, s, mask=m_0_15)
        if STORE_PRE:
            tl.store(PRE_OUT + p_0_15, pre, mask=m_0_15)
        m_0_16 = cmask & (tile + 16 < T)
        p_0_16 = (tile + 16) * (B * DN) + b * DN + c
        pre = _rn_add_f32(_rn_mul_f32(a_0_16, v), i_0_16)
        s = tl.maximum(pre - th_const, 0.0)
        if BIPOLAR:
            s = s - tl.maximum(-pre - th_const, 0.0)
        v = tl.where(m_0_16, pre - s, v)
        tl.store(S_OUT + p_0_16, s, mask=m_0_16)
        if STORE_PRE:
            tl.store(PRE_OUT + p_0_16, pre, mask=m_0_16)
        m_0_17 = cmask & (tile + 17 < T)
        p_0_17 = (tile + 17) * (B * DN) + b * DN + c
        pre = _rn_add_f32(_rn_mul_f32(a_0_17, v), i_0_17)
        s = tl.maximum(pre - th_const, 0.0)
        if BIPOLAR:
            s = s - tl.maximum(-pre - th_const, 0.0)
        v = tl.where(m_0_17, pre - s, v)
        tl.store(S_OUT + p_0_17, s, mask=m_0_17)
        if STORE_PRE:
            tl.store(PRE_OUT + p_0_17, pre, mask=m_0_17)
        m_0_18 = cmask & (tile + 18 < T)
        p_0_18 = (tile + 18) * (B * DN) + b * DN + c
        pre = _rn_add_f32(_rn_mul_f32(a_0_18, v), i_0_18)
        s = tl.maximum(pre - th_const, 0.0)
        if BIPOLAR:
            s = s - tl.maximum(-pre - th_const, 0.0)
        v = tl.where(m_0_18, pre - s, v)
        tl.store(S_OUT + p_0_18, s, mask=m_0_18)
        if STORE_PRE:
            tl.store(PRE_OUT + p_0_18, pre, mask=m_0_18)
        m_0_19 = cmask & (tile + 19 < T)
        p_0_19 = (tile + 19) * (B * DN) + b * DN + c
        pre = _rn_add_f32(_rn_mul_f32(a_0_19, v), i_0_19)
        s = tl.maximum(pre - th_const, 0.0)
        if BIPOLAR:
            s = s - tl.maximum(-pre - th_const, 0.0)
        v = tl.where(m_0_19, pre - s, v)
        tl.store(S_OUT + p_0_19, s, mask=m_0_19)
        if STORE_PRE:
            tl.store(PRE_OUT + p_0_19, pre, mask=m_0_19)
        m_0_20 = cmask & (tile + 20 < T)
        p_0_20 = (tile + 20) * (B * DN) + b * DN + c
        pre = _rn_add_f32(_rn_mul_f32(a_0_20, v), i_0_20)
        s = tl.maximum(pre - th_const, 0.0)
        if BIPOLAR:
            s = s - tl.maximum(-pre - th_const, 0.0)
        v = tl.where(m_0_20, pre - s, v)
        tl.store(S_OUT + p_0_20, s, mask=m_0_20)
        if STORE_PRE:
            tl.store(PRE_OUT + p_0_20, pre, mask=m_0_20)
        m_0_21 = cmask & (tile + 21 < T)
        p_0_21 = (tile + 21) * (B * DN) + b * DN + c
        pre = _rn_add_f32(_rn_mul_f32(a_0_21, v), i_0_21)
        s = tl.maximum(pre - th_const, 0.0)
        if BIPOLAR:
            s = s - tl.maximum(-pre - th_const, 0.0)
        v = tl.where(m_0_21, pre - s, v)
        tl.store(S_OUT + p_0_21, s, mask=m_0_21)
        if STORE_PRE:
            tl.store(PRE_OUT + p_0_21, pre, mask=m_0_21)
        m_0_22 = cmask & (tile + 22 < T)
        p_0_22 = (tile + 22) * (B * DN) + b * DN + c
        pre = _rn_add_f32(_rn_mul_f32(a_0_22, v), i_0_22)
        s = tl.maximum(pre - th_const, 0.0)
        if BIPOLAR:
            s = s - tl.maximum(-pre - th_const, 0.0)
        v = tl.where(m_0_22, pre - s, v)
        tl.store(S_OUT + p_0_22, s, mask=m_0_22)
        if STORE_PRE:
            tl.store(PRE_OUT + p_0_22, pre, mask=m_0_22)
        m_0_23 = cmask & (tile + 23 < T)
        p_0_23 = (tile + 23) * (B * DN) + b * DN + c
        pre = _rn_add_f32(_rn_mul_f32(a_0_23, v), i_0_23)
        s = tl.maximum(pre - th_const, 0.0)
        if BIPOLAR:
            s = s - tl.maximum(-pre - th_const, 0.0)
        v = tl.where(m_0_23, pre - s, v)
        tl.store(S_OUT + p_0_23, s, mask=m_0_23)
        if STORE_PRE:
            tl.store(PRE_OUT + p_0_23, pre, mask=m_0_23)
        m_0_24 = cmask & (tile + 24 < T)
        p_0_24 = (tile + 24) * (B * DN) + b * DN + c
        pre = _rn_add_f32(_rn_mul_f32(a_0_24, v), i_0_24)
        s = tl.maximum(pre - th_const, 0.0)
        if BIPOLAR:
            s = s - tl.maximum(-pre - th_const, 0.0)
        v = tl.where(m_0_24, pre - s, v)
        tl.store(S_OUT + p_0_24, s, mask=m_0_24)
        if STORE_PRE:
            tl.store(PRE_OUT + p_0_24, pre, mask=m_0_24)
        m_0_25 = cmask & (tile + 25 < T)
        p_0_25 = (tile + 25) * (B * DN) + b * DN + c
        pre = _rn_add_f32(_rn_mul_f32(a_0_25, v), i_0_25)
        s = tl.maximum(pre - th_const, 0.0)
        if BIPOLAR:
            s = s - tl.maximum(-pre - th_const, 0.0)
        v = tl.where(m_0_25, pre - s, v)
        tl.store(S_OUT + p_0_25, s, mask=m_0_25)
        if STORE_PRE:
            tl.store(PRE_OUT + p_0_25, pre, mask=m_0_25)
        m_0_26 = cmask & (tile + 26 < T)
        p_0_26 = (tile + 26) * (B * DN) + b * DN + c
        pre = _rn_add_f32(_rn_mul_f32(a_0_26, v), i_0_26)
        s = tl.maximum(pre - th_const, 0.0)
        if BIPOLAR:
            s = s - tl.maximum(-pre - th_const, 0.0)
        v = tl.where(m_0_26, pre - s, v)
        tl.store(S_OUT + p_0_26, s, mask=m_0_26)
        if STORE_PRE:
            tl.store(PRE_OUT + p_0_26, pre, mask=m_0_26)
        m_0_27 = cmask & (tile + 27 < T)
        p_0_27 = (tile + 27) * (B * DN) + b * DN + c
        pre = _rn_add_f32(_rn_mul_f32(a_0_27, v), i_0_27)
        s = tl.maximum(pre - th_const, 0.0)
        if BIPOLAR:
            s = s - tl.maximum(-pre - th_const, 0.0)
        v = tl.where(m_0_27, pre - s, v)
        tl.store(S_OUT + p_0_27, s, mask=m_0_27)
        if STORE_PRE:
            tl.store(PRE_OUT + p_0_27, pre, mask=m_0_27)
        m_0_28 = cmask & (tile + 28 < T)
        p_0_28 = (tile + 28) * (B * DN) + b * DN + c
        pre = _rn_add_f32(_rn_mul_f32(a_0_28, v), i_0_28)
        s = tl.maximum(pre - th_const, 0.0)
        if BIPOLAR:
            s = s - tl.maximum(-pre - th_const, 0.0)
        v = tl.where(m_0_28, pre - s, v)
        tl.store(S_OUT + p_0_28, s, mask=m_0_28)
        if STORE_PRE:
            tl.store(PRE_OUT + p_0_28, pre, mask=m_0_28)
        m_0_29 = cmask & (tile + 29 < T)
        p_0_29 = (tile + 29) * (B * DN) + b * DN + c
        pre = _rn_add_f32(_rn_mul_f32(a_0_29, v), i_0_29)
        s = tl.maximum(pre - th_const, 0.0)
        if BIPOLAR:
            s = s - tl.maximum(-pre - th_const, 0.0)
        v = tl.where(m_0_29, pre - s, v)
        tl.store(S_OUT + p_0_29, s, mask=m_0_29)
        if STORE_PRE:
            tl.store(PRE_OUT + p_0_29, pre, mask=m_0_29)
        m_0_30 = cmask & (tile + 30 < T)
        p_0_30 = (tile + 30) * (B * DN) + b * DN + c
        pre = _rn_add_f32(_rn_mul_f32(a_0_30, v), i_0_30)
        s = tl.maximum(pre - th_const, 0.0)
        if BIPOLAR:
            s = s - tl.maximum(-pre - th_const, 0.0)
        v = tl.where(m_0_30, pre - s, v)
        tl.store(S_OUT + p_0_30, s, mask=m_0_30)
        if STORE_PRE:
            tl.store(PRE_OUT + p_0_30, pre, mask=m_0_30)
        m_0_31 = cmask & (tile + 31 < T)
        p_0_31 = (tile + 31) * (B * DN) + b * DN + c
        pre = _rn_add_f32(_rn_mul_f32(a_0_31, v), i_0_31)
        s = tl.maximum(pre - th_const, 0.0)
        if BIPOLAR:
            s = s - tl.maximum(-pre - th_const, 0.0)
        v = tl.where(m_0_31, pre - s, v)
        tl.store(S_OUT + p_0_31, s, mask=m_0_31)
        if STORE_PRE:
            tl.store(PRE_OUT + p_0_31, pre, mask=m_0_31)
        e_0_0 = e_1_0
        x_0_0 = x_1_0
        e_0_1 = e_1_1
        x_0_1 = x_1_1
        e_0_2 = e_1_2
        x_0_2 = x_1_2
        e_0_3 = e_1_3
        x_0_3 = x_1_3
        e_0_4 = e_1_4
        x_0_4 = x_1_4
        e_0_5 = e_1_5
        x_0_5 = x_1_5
        e_0_6 = e_1_6
        x_0_6 = x_1_6
        e_0_7 = e_1_7
        x_0_7 = x_1_7
        e_0_8 = e_1_8
        x_0_8 = x_1_8
        e_0_9 = e_1_9
        x_0_9 = x_1_9
        e_0_10 = e_1_10
        x_0_10 = x_1_10
        e_0_11 = e_1_11
        x_0_11 = x_1_11
        e_0_12 = e_1_12
        x_0_12 = x_1_12
        e_0_13 = e_1_13
        x_0_13 = x_1_13
        e_0_14 = e_1_14
        x_0_14 = x_1_14
        e_0_15 = e_1_15
        x_0_15 = x_1_15
        e_0_16 = e_1_16
        x_0_16 = x_1_16
        e_0_17 = e_1_17
        x_0_17 = x_1_17
        e_0_18 = e_1_18
        x_0_18 = x_1_18
        e_0_19 = e_1_19
        x_0_19 = x_1_19
        e_0_20 = e_1_20
        x_0_20 = x_1_20
        e_0_21 = e_1_21
        x_0_21 = x_1_21
        e_0_22 = e_1_22
        x_0_22 = x_1_22
        e_0_23 = e_1_23
        x_0_23 = x_1_23
        e_0_24 = e_1_24
        x_0_24 = x_1_24
        e_0_25 = e_1_25
        x_0_25 = x_1_25
        e_0_26 = e_1_26
        x_0_26 = x_1_26
        e_0_27 = e_1_27
        x_0_27 = x_1_27
        e_0_28 = e_1_28
        x_0_28 = x_1_28
        e_0_29 = e_1_29
        x_0_29 = x_1_29
        e_0_30 = e_1_30
        x_0_30 = x_1_30
        e_0_31 = e_1_31
        x_0_31 = x_1_31
    tl.store(VLAST_OUT + offs, v, mask=cmask)


# _plif2_fwd_t32_d1_kernel: the software-pipelined scan forward, TILE 32, DEPTH 1 (launched with A_PER_CHANNEL 0,
#   TH_PER_CHANNEL 1, HAS_RESET 0, STORE_PRE 0).
@triton.jit
def _plif2_fwd_t32_d1_kernel(
    A, I, TH, R, V0, S_OUT, VLAST_OUT, PRE_OUT,
    T, B, DN,
    A_PER_CHANNEL: tl.constexpr, TH_PER_CHANNEL: tl.constexpr, STORE_PRE: tl.constexpr,
    BLOCK_C: tl.constexpr, BIPOLAR: tl.constexpr, HAS_RESET: tl.constexpr, HAS_V0: tl.constexpr,
):
    """Forward scan, TILE=32 tokens per tile, loads issued DEPTH=1 tile ahead."""
    pid = tl.program_id(0)
    offs = pid * BLOCK_C + tl.arange(0, BLOCK_C)
    cmask = offs < B * DN
    b = offs // DN
    c = offs % DN
    if HAS_V0:
        v = tl.load(V0 + offs, mask=cmask, other=0.0)
    else:
        v = tl.zeros([BLOCK_C], dtype=tl.float32)
    a_const = tl.load(A + c, mask=cmask & (A_PER_CHANNEL != 0), other=0.0)
    th_const = tl.load(TH + c, mask=cmask & (TH_PER_CHANNEL != 0), other=0.0)
    lm_0 = cmask & (0 + 0 < T)
    lp_0 = (0 + 0) * (B * DN) + b * DN + c
    lm_1 = cmask & (0 + 1 < T)
    lp_1 = (0 + 1) * (B * DN) + b * DN + c
    lm_2 = cmask & (0 + 2 < T)
    lp_2 = (0 + 2) * (B * DN) + b * DN + c
    lm_3 = cmask & (0 + 3 < T)
    lp_3 = (0 + 3) * (B * DN) + b * DN + c
    lm_4 = cmask & (0 + 4 < T)
    lp_4 = (0 + 4) * (B * DN) + b * DN + c
    lm_5 = cmask & (0 + 5 < T)
    lp_5 = (0 + 5) * (B * DN) + b * DN + c
    lm_6 = cmask & (0 + 6 < T)
    lp_6 = (0 + 6) * (B * DN) + b * DN + c
    lm_7 = cmask & (0 + 7 < T)
    lp_7 = (0 + 7) * (B * DN) + b * DN + c
    lm_8 = cmask & (0 + 8 < T)
    lp_8 = (0 + 8) * (B * DN) + b * DN + c
    lm_9 = cmask & (0 + 9 < T)
    lp_9 = (0 + 9) * (B * DN) + b * DN + c
    lm_10 = cmask & (0 + 10 < T)
    lp_10 = (0 + 10) * (B * DN) + b * DN + c
    lm_11 = cmask & (0 + 11 < T)
    lp_11 = (0 + 11) * (B * DN) + b * DN + c
    lm_12 = cmask & (0 + 12 < T)
    lp_12 = (0 + 12) * (B * DN) + b * DN + c
    lm_13 = cmask & (0 + 13 < T)
    lp_13 = (0 + 13) * (B * DN) + b * DN + c
    lm_14 = cmask & (0 + 14 < T)
    lp_14 = (0 + 14) * (B * DN) + b * DN + c
    lm_15 = cmask & (0 + 15 < T)
    lp_15 = (0 + 15) * (B * DN) + b * DN + c
    lm_16 = cmask & (0 + 16 < T)
    lp_16 = (0 + 16) * (B * DN) + b * DN + c
    lm_17 = cmask & (0 + 17 < T)
    lp_17 = (0 + 17) * (B * DN) + b * DN + c
    lm_18 = cmask & (0 + 18 < T)
    lp_18 = (0 + 18) * (B * DN) + b * DN + c
    lm_19 = cmask & (0 + 19 < T)
    lp_19 = (0 + 19) * (B * DN) + b * DN + c
    lm_20 = cmask & (0 + 20 < T)
    lp_20 = (0 + 20) * (B * DN) + b * DN + c
    lm_21 = cmask & (0 + 21 < T)
    lp_21 = (0 + 21) * (B * DN) + b * DN + c
    lm_22 = cmask & (0 + 22 < T)
    lp_22 = (0 + 22) * (B * DN) + b * DN + c
    lm_23 = cmask & (0 + 23 < T)
    lp_23 = (0 + 23) * (B * DN) + b * DN + c
    lm_24 = cmask & (0 + 24 < T)
    lp_24 = (0 + 24) * (B * DN) + b * DN + c
    lm_25 = cmask & (0 + 25 < T)
    lp_25 = (0 + 25) * (B * DN) + b * DN + c
    lm_26 = cmask & (0 + 26 < T)
    lp_26 = (0 + 26) * (B * DN) + b * DN + c
    lm_27 = cmask & (0 + 27 < T)
    lp_27 = (0 + 27) * (B * DN) + b * DN + c
    lm_28 = cmask & (0 + 28 < T)
    lp_28 = (0 + 28) * (B * DN) + b * DN + c
    lm_29 = cmask & (0 + 29 < T)
    lp_29 = (0 + 29) * (B * DN) + b * DN + c
    lm_30 = cmask & (0 + 30 < T)
    lp_30 = (0 + 30) * (B * DN) + b * DN + c
    lm_31 = cmask & (0 + 31 < T)
    lp_31 = (0 + 31) * (B * DN) + b * DN + c
    i_0_0 = tl.load(I + lp_0, mask=lm_0, other=0.0).to(tl.float32)
    i_0_1 = tl.load(I + lp_1, mask=lm_1, other=0.0).to(tl.float32)
    i_0_2 = tl.load(I + lp_2, mask=lm_2, other=0.0).to(tl.float32)
    i_0_3 = tl.load(I + lp_3, mask=lm_3, other=0.0).to(tl.float32)
    i_0_4 = tl.load(I + lp_4, mask=lm_4, other=0.0).to(tl.float32)
    i_0_5 = tl.load(I + lp_5, mask=lm_5, other=0.0).to(tl.float32)
    i_0_6 = tl.load(I + lp_6, mask=lm_6, other=0.0).to(tl.float32)
    i_0_7 = tl.load(I + lp_7, mask=lm_7, other=0.0).to(tl.float32)
    i_0_8 = tl.load(I + lp_8, mask=lm_8, other=0.0).to(tl.float32)
    i_0_9 = tl.load(I + lp_9, mask=lm_9, other=0.0).to(tl.float32)
    i_0_10 = tl.load(I + lp_10, mask=lm_10, other=0.0).to(tl.float32)
    i_0_11 = tl.load(I + lp_11, mask=lm_11, other=0.0).to(tl.float32)
    i_0_12 = tl.load(I + lp_12, mask=lm_12, other=0.0).to(tl.float32)
    i_0_13 = tl.load(I + lp_13, mask=lm_13, other=0.0).to(tl.float32)
    i_0_14 = tl.load(I + lp_14, mask=lm_14, other=0.0).to(tl.float32)
    i_0_15 = tl.load(I + lp_15, mask=lm_15, other=0.0).to(tl.float32)
    i_0_16 = tl.load(I + lp_16, mask=lm_16, other=0.0).to(tl.float32)
    i_0_17 = tl.load(I + lp_17, mask=lm_17, other=0.0).to(tl.float32)
    i_0_18 = tl.load(I + lp_18, mask=lm_18, other=0.0).to(tl.float32)
    i_0_19 = tl.load(I + lp_19, mask=lm_19, other=0.0).to(tl.float32)
    i_0_20 = tl.load(I + lp_20, mask=lm_20, other=0.0).to(tl.float32)
    i_0_21 = tl.load(I + lp_21, mask=lm_21, other=0.0).to(tl.float32)
    i_0_22 = tl.load(I + lp_22, mask=lm_22, other=0.0).to(tl.float32)
    i_0_23 = tl.load(I + lp_23, mask=lm_23, other=0.0).to(tl.float32)
    i_0_24 = tl.load(I + lp_24, mask=lm_24, other=0.0).to(tl.float32)
    i_0_25 = tl.load(I + lp_25, mask=lm_25, other=0.0).to(tl.float32)
    i_0_26 = tl.load(I + lp_26, mask=lm_26, other=0.0).to(tl.float32)
    i_0_27 = tl.load(I + lp_27, mask=lm_27, other=0.0).to(tl.float32)
    i_0_28 = tl.load(I + lp_28, mask=lm_28, other=0.0).to(tl.float32)
    i_0_29 = tl.load(I + lp_29, mask=lm_29, other=0.0).to(tl.float32)
    i_0_30 = tl.load(I + lp_30, mask=lm_30, other=0.0).to(tl.float32)
    i_0_31 = tl.load(I + lp_31, mask=lm_31, other=0.0).to(tl.float32)
    if not A_PER_CHANNEL:
        a_0_0 = tl.load(A + lp_0, mask=lm_0, other=0.0)
        a_0_1 = tl.load(A + lp_1, mask=lm_1, other=0.0)
        a_0_2 = tl.load(A + lp_2, mask=lm_2, other=0.0)
        a_0_3 = tl.load(A + lp_3, mask=lm_3, other=0.0)
        a_0_4 = tl.load(A + lp_4, mask=lm_4, other=0.0)
        a_0_5 = tl.load(A + lp_5, mask=lm_5, other=0.0)
        a_0_6 = tl.load(A + lp_6, mask=lm_6, other=0.0)
        a_0_7 = tl.load(A + lp_7, mask=lm_7, other=0.0)
        a_0_8 = tl.load(A + lp_8, mask=lm_8, other=0.0)
        a_0_9 = tl.load(A + lp_9, mask=lm_9, other=0.0)
        a_0_10 = tl.load(A + lp_10, mask=lm_10, other=0.0)
        a_0_11 = tl.load(A + lp_11, mask=lm_11, other=0.0)
        a_0_12 = tl.load(A + lp_12, mask=lm_12, other=0.0)
        a_0_13 = tl.load(A + lp_13, mask=lm_13, other=0.0)
        a_0_14 = tl.load(A + lp_14, mask=lm_14, other=0.0)
        a_0_15 = tl.load(A + lp_15, mask=lm_15, other=0.0)
        a_0_16 = tl.load(A + lp_16, mask=lm_16, other=0.0)
        a_0_17 = tl.load(A + lp_17, mask=lm_17, other=0.0)
        a_0_18 = tl.load(A + lp_18, mask=lm_18, other=0.0)
        a_0_19 = tl.load(A + lp_19, mask=lm_19, other=0.0)
        a_0_20 = tl.load(A + lp_20, mask=lm_20, other=0.0)
        a_0_21 = tl.load(A + lp_21, mask=lm_21, other=0.0)
        a_0_22 = tl.load(A + lp_22, mask=lm_22, other=0.0)
        a_0_23 = tl.load(A + lp_23, mask=lm_23, other=0.0)
        a_0_24 = tl.load(A + lp_24, mask=lm_24, other=0.0)
        a_0_25 = tl.load(A + lp_25, mask=lm_25, other=0.0)
        a_0_26 = tl.load(A + lp_26, mask=lm_26, other=0.0)
        a_0_27 = tl.load(A + lp_27, mask=lm_27, other=0.0)
        a_0_28 = tl.load(A + lp_28, mask=lm_28, other=0.0)
        a_0_29 = tl.load(A + lp_29, mask=lm_29, other=0.0)
        a_0_30 = tl.load(A + lp_30, mask=lm_30, other=0.0)
        a_0_31 = tl.load(A + lp_31, mask=lm_31, other=0.0)
    if not TH_PER_CHANNEL:
        t_0_0 = tl.load(TH + lp_0, mask=lm_0, other=0.0)
        t_0_1 = tl.load(TH + lp_1, mask=lm_1, other=0.0)
        t_0_2 = tl.load(TH + lp_2, mask=lm_2, other=0.0)
        t_0_3 = tl.load(TH + lp_3, mask=lm_3, other=0.0)
        t_0_4 = tl.load(TH + lp_4, mask=lm_4, other=0.0)
        t_0_5 = tl.load(TH + lp_5, mask=lm_5, other=0.0)
        t_0_6 = tl.load(TH + lp_6, mask=lm_6, other=0.0)
        t_0_7 = tl.load(TH + lp_7, mask=lm_7, other=0.0)
        t_0_8 = tl.load(TH + lp_8, mask=lm_8, other=0.0)
        t_0_9 = tl.load(TH + lp_9, mask=lm_9, other=0.0)
        t_0_10 = tl.load(TH + lp_10, mask=lm_10, other=0.0)
        t_0_11 = tl.load(TH + lp_11, mask=lm_11, other=0.0)
        t_0_12 = tl.load(TH + lp_12, mask=lm_12, other=0.0)
        t_0_13 = tl.load(TH + lp_13, mask=lm_13, other=0.0)
        t_0_14 = tl.load(TH + lp_14, mask=lm_14, other=0.0)
        t_0_15 = tl.load(TH + lp_15, mask=lm_15, other=0.0)
        t_0_16 = tl.load(TH + lp_16, mask=lm_16, other=0.0)
        t_0_17 = tl.load(TH + lp_17, mask=lm_17, other=0.0)
        t_0_18 = tl.load(TH + lp_18, mask=lm_18, other=0.0)
        t_0_19 = tl.load(TH + lp_19, mask=lm_19, other=0.0)
        t_0_20 = tl.load(TH + lp_20, mask=lm_20, other=0.0)
        t_0_21 = tl.load(TH + lp_21, mask=lm_21, other=0.0)
        t_0_22 = tl.load(TH + lp_22, mask=lm_22, other=0.0)
        t_0_23 = tl.load(TH + lp_23, mask=lm_23, other=0.0)
        t_0_24 = tl.load(TH + lp_24, mask=lm_24, other=0.0)
        t_0_25 = tl.load(TH + lp_25, mask=lm_25, other=0.0)
        t_0_26 = tl.load(TH + lp_26, mask=lm_26, other=0.0)
        t_0_27 = tl.load(TH + lp_27, mask=lm_27, other=0.0)
        t_0_28 = tl.load(TH + lp_28, mask=lm_28, other=0.0)
        t_0_29 = tl.load(TH + lp_29, mask=lm_29, other=0.0)
        t_0_30 = tl.load(TH + lp_30, mask=lm_30, other=0.0)
        t_0_31 = tl.load(TH + lp_31, mask=lm_31, other=0.0)
    if HAS_RESET:
        r_0_0 = tl.load(R + (0 + 0) * B + b, mask=lm_0, other=0).to(tl.int32)
        r_0_1 = tl.load(R + (0 + 1) * B + b, mask=lm_1, other=0).to(tl.int32)
        r_0_2 = tl.load(R + (0 + 2) * B + b, mask=lm_2, other=0).to(tl.int32)
        r_0_3 = tl.load(R + (0 + 3) * B + b, mask=lm_3, other=0).to(tl.int32)
        r_0_4 = tl.load(R + (0 + 4) * B + b, mask=lm_4, other=0).to(tl.int32)
        r_0_5 = tl.load(R + (0 + 5) * B + b, mask=lm_5, other=0).to(tl.int32)
        r_0_6 = tl.load(R + (0 + 6) * B + b, mask=lm_6, other=0).to(tl.int32)
        r_0_7 = tl.load(R + (0 + 7) * B + b, mask=lm_7, other=0).to(tl.int32)
        r_0_8 = tl.load(R + (0 + 8) * B + b, mask=lm_8, other=0).to(tl.int32)
        r_0_9 = tl.load(R + (0 + 9) * B + b, mask=lm_9, other=0).to(tl.int32)
        r_0_10 = tl.load(R + (0 + 10) * B + b, mask=lm_10, other=0).to(tl.int32)
        r_0_11 = tl.load(R + (0 + 11) * B + b, mask=lm_11, other=0).to(tl.int32)
        r_0_12 = tl.load(R + (0 + 12) * B + b, mask=lm_12, other=0).to(tl.int32)
        r_0_13 = tl.load(R + (0 + 13) * B + b, mask=lm_13, other=0).to(tl.int32)
        r_0_14 = tl.load(R + (0 + 14) * B + b, mask=lm_14, other=0).to(tl.int32)
        r_0_15 = tl.load(R + (0 + 15) * B + b, mask=lm_15, other=0).to(tl.int32)
        r_0_16 = tl.load(R + (0 + 16) * B + b, mask=lm_16, other=0).to(tl.int32)
        r_0_17 = tl.load(R + (0 + 17) * B + b, mask=lm_17, other=0).to(tl.int32)
        r_0_18 = tl.load(R + (0 + 18) * B + b, mask=lm_18, other=0).to(tl.int32)
        r_0_19 = tl.load(R + (0 + 19) * B + b, mask=lm_19, other=0).to(tl.int32)
        r_0_20 = tl.load(R + (0 + 20) * B + b, mask=lm_20, other=0).to(tl.int32)
        r_0_21 = tl.load(R + (0 + 21) * B + b, mask=lm_21, other=0).to(tl.int32)
        r_0_22 = tl.load(R + (0 + 22) * B + b, mask=lm_22, other=0).to(tl.int32)
        r_0_23 = tl.load(R + (0 + 23) * B + b, mask=lm_23, other=0).to(tl.int32)
        r_0_24 = tl.load(R + (0 + 24) * B + b, mask=lm_24, other=0).to(tl.int32)
        r_0_25 = tl.load(R + (0 + 25) * B + b, mask=lm_25, other=0).to(tl.int32)
        r_0_26 = tl.load(R + (0 + 26) * B + b, mask=lm_26, other=0).to(tl.int32)
        r_0_27 = tl.load(R + (0 + 27) * B + b, mask=lm_27, other=0).to(tl.int32)
        r_0_28 = tl.load(R + (0 + 28) * B + b, mask=lm_28, other=0).to(tl.int32)
        r_0_29 = tl.load(R + (0 + 29) * B + b, mask=lm_29, other=0).to(tl.int32)
        r_0_30 = tl.load(R + (0 + 30) * B + b, mask=lm_30, other=0).to(tl.int32)
        r_0_31 = tl.load(R + (0 + 31) * B + b, mask=lm_31, other=0).to(tl.int32)
    for tile in range(0, T, 32):
        lm_0 = cmask & (tile + 32 + 0 < T)
        lp_0 = (tile + 32 + 0) * (B * DN) + b * DN + c
        lm_1 = cmask & (tile + 32 + 1 < T)
        lp_1 = (tile + 32 + 1) * (B * DN) + b * DN + c
        lm_2 = cmask & (tile + 32 + 2 < T)
        lp_2 = (tile + 32 + 2) * (B * DN) + b * DN + c
        lm_3 = cmask & (tile + 32 + 3 < T)
        lp_3 = (tile + 32 + 3) * (B * DN) + b * DN + c
        lm_4 = cmask & (tile + 32 + 4 < T)
        lp_4 = (tile + 32 + 4) * (B * DN) + b * DN + c
        lm_5 = cmask & (tile + 32 + 5 < T)
        lp_5 = (tile + 32 + 5) * (B * DN) + b * DN + c
        lm_6 = cmask & (tile + 32 + 6 < T)
        lp_6 = (tile + 32 + 6) * (B * DN) + b * DN + c
        lm_7 = cmask & (tile + 32 + 7 < T)
        lp_7 = (tile + 32 + 7) * (B * DN) + b * DN + c
        lm_8 = cmask & (tile + 32 + 8 < T)
        lp_8 = (tile + 32 + 8) * (B * DN) + b * DN + c
        lm_9 = cmask & (tile + 32 + 9 < T)
        lp_9 = (tile + 32 + 9) * (B * DN) + b * DN + c
        lm_10 = cmask & (tile + 32 + 10 < T)
        lp_10 = (tile + 32 + 10) * (B * DN) + b * DN + c
        lm_11 = cmask & (tile + 32 + 11 < T)
        lp_11 = (tile + 32 + 11) * (B * DN) + b * DN + c
        lm_12 = cmask & (tile + 32 + 12 < T)
        lp_12 = (tile + 32 + 12) * (B * DN) + b * DN + c
        lm_13 = cmask & (tile + 32 + 13 < T)
        lp_13 = (tile + 32 + 13) * (B * DN) + b * DN + c
        lm_14 = cmask & (tile + 32 + 14 < T)
        lp_14 = (tile + 32 + 14) * (B * DN) + b * DN + c
        lm_15 = cmask & (tile + 32 + 15 < T)
        lp_15 = (tile + 32 + 15) * (B * DN) + b * DN + c
        lm_16 = cmask & (tile + 32 + 16 < T)
        lp_16 = (tile + 32 + 16) * (B * DN) + b * DN + c
        lm_17 = cmask & (tile + 32 + 17 < T)
        lp_17 = (tile + 32 + 17) * (B * DN) + b * DN + c
        lm_18 = cmask & (tile + 32 + 18 < T)
        lp_18 = (tile + 32 + 18) * (B * DN) + b * DN + c
        lm_19 = cmask & (tile + 32 + 19 < T)
        lp_19 = (tile + 32 + 19) * (B * DN) + b * DN + c
        lm_20 = cmask & (tile + 32 + 20 < T)
        lp_20 = (tile + 32 + 20) * (B * DN) + b * DN + c
        lm_21 = cmask & (tile + 32 + 21 < T)
        lp_21 = (tile + 32 + 21) * (B * DN) + b * DN + c
        lm_22 = cmask & (tile + 32 + 22 < T)
        lp_22 = (tile + 32 + 22) * (B * DN) + b * DN + c
        lm_23 = cmask & (tile + 32 + 23 < T)
        lp_23 = (tile + 32 + 23) * (B * DN) + b * DN + c
        lm_24 = cmask & (tile + 32 + 24 < T)
        lp_24 = (tile + 32 + 24) * (B * DN) + b * DN + c
        lm_25 = cmask & (tile + 32 + 25 < T)
        lp_25 = (tile + 32 + 25) * (B * DN) + b * DN + c
        lm_26 = cmask & (tile + 32 + 26 < T)
        lp_26 = (tile + 32 + 26) * (B * DN) + b * DN + c
        lm_27 = cmask & (tile + 32 + 27 < T)
        lp_27 = (tile + 32 + 27) * (B * DN) + b * DN + c
        lm_28 = cmask & (tile + 32 + 28 < T)
        lp_28 = (tile + 32 + 28) * (B * DN) + b * DN + c
        lm_29 = cmask & (tile + 32 + 29 < T)
        lp_29 = (tile + 32 + 29) * (B * DN) + b * DN + c
        lm_30 = cmask & (tile + 32 + 30 < T)
        lp_30 = (tile + 32 + 30) * (B * DN) + b * DN + c
        lm_31 = cmask & (tile + 32 + 31 < T)
        lp_31 = (tile + 32 + 31) * (B * DN) + b * DN + c
        i_1_0 = tl.load(I + lp_0, mask=lm_0, other=0.0).to(tl.float32)
        i_1_1 = tl.load(I + lp_1, mask=lm_1, other=0.0).to(tl.float32)
        i_1_2 = tl.load(I + lp_2, mask=lm_2, other=0.0).to(tl.float32)
        i_1_3 = tl.load(I + lp_3, mask=lm_3, other=0.0).to(tl.float32)
        i_1_4 = tl.load(I + lp_4, mask=lm_4, other=0.0).to(tl.float32)
        i_1_5 = tl.load(I + lp_5, mask=lm_5, other=0.0).to(tl.float32)
        i_1_6 = tl.load(I + lp_6, mask=lm_6, other=0.0).to(tl.float32)
        i_1_7 = tl.load(I + lp_7, mask=lm_7, other=0.0).to(tl.float32)
        i_1_8 = tl.load(I + lp_8, mask=lm_8, other=0.0).to(tl.float32)
        i_1_9 = tl.load(I + lp_9, mask=lm_9, other=0.0).to(tl.float32)
        i_1_10 = tl.load(I + lp_10, mask=lm_10, other=0.0).to(tl.float32)
        i_1_11 = tl.load(I + lp_11, mask=lm_11, other=0.0).to(tl.float32)
        i_1_12 = tl.load(I + lp_12, mask=lm_12, other=0.0).to(tl.float32)
        i_1_13 = tl.load(I + lp_13, mask=lm_13, other=0.0).to(tl.float32)
        i_1_14 = tl.load(I + lp_14, mask=lm_14, other=0.0).to(tl.float32)
        i_1_15 = tl.load(I + lp_15, mask=lm_15, other=0.0).to(tl.float32)
        i_1_16 = tl.load(I + lp_16, mask=lm_16, other=0.0).to(tl.float32)
        i_1_17 = tl.load(I + lp_17, mask=lm_17, other=0.0).to(tl.float32)
        i_1_18 = tl.load(I + lp_18, mask=lm_18, other=0.0).to(tl.float32)
        i_1_19 = tl.load(I + lp_19, mask=lm_19, other=0.0).to(tl.float32)
        i_1_20 = tl.load(I + lp_20, mask=lm_20, other=0.0).to(tl.float32)
        i_1_21 = tl.load(I + lp_21, mask=lm_21, other=0.0).to(tl.float32)
        i_1_22 = tl.load(I + lp_22, mask=lm_22, other=0.0).to(tl.float32)
        i_1_23 = tl.load(I + lp_23, mask=lm_23, other=0.0).to(tl.float32)
        i_1_24 = tl.load(I + lp_24, mask=lm_24, other=0.0).to(tl.float32)
        i_1_25 = tl.load(I + lp_25, mask=lm_25, other=0.0).to(tl.float32)
        i_1_26 = tl.load(I + lp_26, mask=lm_26, other=0.0).to(tl.float32)
        i_1_27 = tl.load(I + lp_27, mask=lm_27, other=0.0).to(tl.float32)
        i_1_28 = tl.load(I + lp_28, mask=lm_28, other=0.0).to(tl.float32)
        i_1_29 = tl.load(I + lp_29, mask=lm_29, other=0.0).to(tl.float32)
        i_1_30 = tl.load(I + lp_30, mask=lm_30, other=0.0).to(tl.float32)
        i_1_31 = tl.load(I + lp_31, mask=lm_31, other=0.0).to(tl.float32)
        if not A_PER_CHANNEL:
            a_1_0 = tl.load(A + lp_0, mask=lm_0, other=0.0)
            a_1_1 = tl.load(A + lp_1, mask=lm_1, other=0.0)
            a_1_2 = tl.load(A + lp_2, mask=lm_2, other=0.0)
            a_1_3 = tl.load(A + lp_3, mask=lm_3, other=0.0)
            a_1_4 = tl.load(A + lp_4, mask=lm_4, other=0.0)
            a_1_5 = tl.load(A + lp_5, mask=lm_5, other=0.0)
            a_1_6 = tl.load(A + lp_6, mask=lm_6, other=0.0)
            a_1_7 = tl.load(A + lp_7, mask=lm_7, other=0.0)
            a_1_8 = tl.load(A + lp_8, mask=lm_8, other=0.0)
            a_1_9 = tl.load(A + lp_9, mask=lm_9, other=0.0)
            a_1_10 = tl.load(A + lp_10, mask=lm_10, other=0.0)
            a_1_11 = tl.load(A + lp_11, mask=lm_11, other=0.0)
            a_1_12 = tl.load(A + lp_12, mask=lm_12, other=0.0)
            a_1_13 = tl.load(A + lp_13, mask=lm_13, other=0.0)
            a_1_14 = tl.load(A + lp_14, mask=lm_14, other=0.0)
            a_1_15 = tl.load(A + lp_15, mask=lm_15, other=0.0)
            a_1_16 = tl.load(A + lp_16, mask=lm_16, other=0.0)
            a_1_17 = tl.load(A + lp_17, mask=lm_17, other=0.0)
            a_1_18 = tl.load(A + lp_18, mask=lm_18, other=0.0)
            a_1_19 = tl.load(A + lp_19, mask=lm_19, other=0.0)
            a_1_20 = tl.load(A + lp_20, mask=lm_20, other=0.0)
            a_1_21 = tl.load(A + lp_21, mask=lm_21, other=0.0)
            a_1_22 = tl.load(A + lp_22, mask=lm_22, other=0.0)
            a_1_23 = tl.load(A + lp_23, mask=lm_23, other=0.0)
            a_1_24 = tl.load(A + lp_24, mask=lm_24, other=0.0)
            a_1_25 = tl.load(A + lp_25, mask=lm_25, other=0.0)
            a_1_26 = tl.load(A + lp_26, mask=lm_26, other=0.0)
            a_1_27 = tl.load(A + lp_27, mask=lm_27, other=0.0)
            a_1_28 = tl.load(A + lp_28, mask=lm_28, other=0.0)
            a_1_29 = tl.load(A + lp_29, mask=lm_29, other=0.0)
            a_1_30 = tl.load(A + lp_30, mask=lm_30, other=0.0)
            a_1_31 = tl.load(A + lp_31, mask=lm_31, other=0.0)
        if not TH_PER_CHANNEL:
            t_1_0 = tl.load(TH + lp_0, mask=lm_0, other=0.0)
            t_1_1 = tl.load(TH + lp_1, mask=lm_1, other=0.0)
            t_1_2 = tl.load(TH + lp_2, mask=lm_2, other=0.0)
            t_1_3 = tl.load(TH + lp_3, mask=lm_3, other=0.0)
            t_1_4 = tl.load(TH + lp_4, mask=lm_4, other=0.0)
            t_1_5 = tl.load(TH + lp_5, mask=lm_5, other=0.0)
            t_1_6 = tl.load(TH + lp_6, mask=lm_6, other=0.0)
            t_1_7 = tl.load(TH + lp_7, mask=lm_7, other=0.0)
            t_1_8 = tl.load(TH + lp_8, mask=lm_8, other=0.0)
            t_1_9 = tl.load(TH + lp_9, mask=lm_9, other=0.0)
            t_1_10 = tl.load(TH + lp_10, mask=lm_10, other=0.0)
            t_1_11 = tl.load(TH + lp_11, mask=lm_11, other=0.0)
            t_1_12 = tl.load(TH + lp_12, mask=lm_12, other=0.0)
            t_1_13 = tl.load(TH + lp_13, mask=lm_13, other=0.0)
            t_1_14 = tl.load(TH + lp_14, mask=lm_14, other=0.0)
            t_1_15 = tl.load(TH + lp_15, mask=lm_15, other=0.0)
            t_1_16 = tl.load(TH + lp_16, mask=lm_16, other=0.0)
            t_1_17 = tl.load(TH + lp_17, mask=lm_17, other=0.0)
            t_1_18 = tl.load(TH + lp_18, mask=lm_18, other=0.0)
            t_1_19 = tl.load(TH + lp_19, mask=lm_19, other=0.0)
            t_1_20 = tl.load(TH + lp_20, mask=lm_20, other=0.0)
            t_1_21 = tl.load(TH + lp_21, mask=lm_21, other=0.0)
            t_1_22 = tl.load(TH + lp_22, mask=lm_22, other=0.0)
            t_1_23 = tl.load(TH + lp_23, mask=lm_23, other=0.0)
            t_1_24 = tl.load(TH + lp_24, mask=lm_24, other=0.0)
            t_1_25 = tl.load(TH + lp_25, mask=lm_25, other=0.0)
            t_1_26 = tl.load(TH + lp_26, mask=lm_26, other=0.0)
            t_1_27 = tl.load(TH + lp_27, mask=lm_27, other=0.0)
            t_1_28 = tl.load(TH + lp_28, mask=lm_28, other=0.0)
            t_1_29 = tl.load(TH + lp_29, mask=lm_29, other=0.0)
            t_1_30 = tl.load(TH + lp_30, mask=lm_30, other=0.0)
            t_1_31 = tl.load(TH + lp_31, mask=lm_31, other=0.0)
        if HAS_RESET:
            r_1_0 = tl.load(R + (tile + 32 + 0) * B + b, mask=lm_0, other=0).to(tl.int32)
            r_1_1 = tl.load(R + (tile + 32 + 1) * B + b, mask=lm_1, other=0).to(tl.int32)
            r_1_2 = tl.load(R + (tile + 32 + 2) * B + b, mask=lm_2, other=0).to(tl.int32)
            r_1_3 = tl.load(R + (tile + 32 + 3) * B + b, mask=lm_3, other=0).to(tl.int32)
            r_1_4 = tl.load(R + (tile + 32 + 4) * B + b, mask=lm_4, other=0).to(tl.int32)
            r_1_5 = tl.load(R + (tile + 32 + 5) * B + b, mask=lm_5, other=0).to(tl.int32)
            r_1_6 = tl.load(R + (tile + 32 + 6) * B + b, mask=lm_6, other=0).to(tl.int32)
            r_1_7 = tl.load(R + (tile + 32 + 7) * B + b, mask=lm_7, other=0).to(tl.int32)
            r_1_8 = tl.load(R + (tile + 32 + 8) * B + b, mask=lm_8, other=0).to(tl.int32)
            r_1_9 = tl.load(R + (tile + 32 + 9) * B + b, mask=lm_9, other=0).to(tl.int32)
            r_1_10 = tl.load(R + (tile + 32 + 10) * B + b, mask=lm_10, other=0).to(tl.int32)
            r_1_11 = tl.load(R + (tile + 32 + 11) * B + b, mask=lm_11, other=0).to(tl.int32)
            r_1_12 = tl.load(R + (tile + 32 + 12) * B + b, mask=lm_12, other=0).to(tl.int32)
            r_1_13 = tl.load(R + (tile + 32 + 13) * B + b, mask=lm_13, other=0).to(tl.int32)
            r_1_14 = tl.load(R + (tile + 32 + 14) * B + b, mask=lm_14, other=0).to(tl.int32)
            r_1_15 = tl.load(R + (tile + 32 + 15) * B + b, mask=lm_15, other=0).to(tl.int32)
            r_1_16 = tl.load(R + (tile + 32 + 16) * B + b, mask=lm_16, other=0).to(tl.int32)
            r_1_17 = tl.load(R + (tile + 32 + 17) * B + b, mask=lm_17, other=0).to(tl.int32)
            r_1_18 = tl.load(R + (tile + 32 + 18) * B + b, mask=lm_18, other=0).to(tl.int32)
            r_1_19 = tl.load(R + (tile + 32 + 19) * B + b, mask=lm_19, other=0).to(tl.int32)
            r_1_20 = tl.load(R + (tile + 32 + 20) * B + b, mask=lm_20, other=0).to(tl.int32)
            r_1_21 = tl.load(R + (tile + 32 + 21) * B + b, mask=lm_21, other=0).to(tl.int32)
            r_1_22 = tl.load(R + (tile + 32 + 22) * B + b, mask=lm_22, other=0).to(tl.int32)
            r_1_23 = tl.load(R + (tile + 32 + 23) * B + b, mask=lm_23, other=0).to(tl.int32)
            r_1_24 = tl.load(R + (tile + 32 + 24) * B + b, mask=lm_24, other=0).to(tl.int32)
            r_1_25 = tl.load(R + (tile + 32 + 25) * B + b, mask=lm_25, other=0).to(tl.int32)
            r_1_26 = tl.load(R + (tile + 32 + 26) * B + b, mask=lm_26, other=0).to(tl.int32)
            r_1_27 = tl.load(R + (tile + 32 + 27) * B + b, mask=lm_27, other=0).to(tl.int32)
            r_1_28 = tl.load(R + (tile + 32 + 28) * B + b, mask=lm_28, other=0).to(tl.int32)
            r_1_29 = tl.load(R + (tile + 32 + 29) * B + b, mask=lm_29, other=0).to(tl.int32)
            r_1_30 = tl.load(R + (tile + 32 + 30) * B + b, mask=lm_30, other=0).to(tl.int32)
            r_1_31 = tl.load(R + (tile + 32 + 31) * B + b, mask=lm_31, other=0).to(tl.int32)
        # ---- token 0 of the tile (sequential, registers only)
        m_0_0 = cmask & (tile + 0 < T)
        p_0_0 = (tile + 0) * (B * DN) + b * DN + c
        i = i_0_0
        a = a_const if A_PER_CHANNEL else a_0_0
        th = th_const if TH_PER_CHANNEL else t_0_0
        if HAS_RESET:
            pre = tl.where(r_0_0 != 0, i, _rn_add_f32(_rn_mul_f32(a, v), i))
        else:
            pre = _rn_add_f32(_rn_mul_f32(a, v), i)
        s = tl.maximum(pre - th, 0.0)
        if BIPOLAR:
            s = s - tl.maximum(-pre - th, 0.0)
        v = tl.where(m_0_0, pre - s, v)
        tl.store(S_OUT + p_0_0, s, mask=m_0_0)
        if STORE_PRE:
            tl.store(PRE_OUT + p_0_0, pre, mask=m_0_0)
        # ---- token 1 of the tile (sequential, registers only)
        m_0_1 = cmask & (tile + 1 < T)
        p_0_1 = (tile + 1) * (B * DN) + b * DN + c
        i = i_0_1
        a = a_const if A_PER_CHANNEL else a_0_1
        th = th_const if TH_PER_CHANNEL else t_0_1
        if HAS_RESET:
            pre = tl.where(r_0_1 != 0, i, _rn_add_f32(_rn_mul_f32(a, v), i))
        else:
            pre = _rn_add_f32(_rn_mul_f32(a, v), i)
        s = tl.maximum(pre - th, 0.0)
        if BIPOLAR:
            s = s - tl.maximum(-pre - th, 0.0)
        v = tl.where(m_0_1, pre - s, v)
        tl.store(S_OUT + p_0_1, s, mask=m_0_1)
        if STORE_PRE:
            tl.store(PRE_OUT + p_0_1, pre, mask=m_0_1)
        # ---- token 2 of the tile (sequential, registers only)
        m_0_2 = cmask & (tile + 2 < T)
        p_0_2 = (tile + 2) * (B * DN) + b * DN + c
        i = i_0_2
        a = a_const if A_PER_CHANNEL else a_0_2
        th = th_const if TH_PER_CHANNEL else t_0_2
        if HAS_RESET:
            pre = tl.where(r_0_2 != 0, i, _rn_add_f32(_rn_mul_f32(a, v), i))
        else:
            pre = _rn_add_f32(_rn_mul_f32(a, v), i)
        s = tl.maximum(pre - th, 0.0)
        if BIPOLAR:
            s = s - tl.maximum(-pre - th, 0.0)
        v = tl.where(m_0_2, pre - s, v)
        tl.store(S_OUT + p_0_2, s, mask=m_0_2)
        if STORE_PRE:
            tl.store(PRE_OUT + p_0_2, pre, mask=m_0_2)
        # ---- token 3 of the tile (sequential, registers only)
        m_0_3 = cmask & (tile + 3 < T)
        p_0_3 = (tile + 3) * (B * DN) + b * DN + c
        i = i_0_3
        a = a_const if A_PER_CHANNEL else a_0_3
        th = th_const if TH_PER_CHANNEL else t_0_3
        if HAS_RESET:
            pre = tl.where(r_0_3 != 0, i, _rn_add_f32(_rn_mul_f32(a, v), i))
        else:
            pre = _rn_add_f32(_rn_mul_f32(a, v), i)
        s = tl.maximum(pre - th, 0.0)
        if BIPOLAR:
            s = s - tl.maximum(-pre - th, 0.0)
        v = tl.where(m_0_3, pre - s, v)
        tl.store(S_OUT + p_0_3, s, mask=m_0_3)
        if STORE_PRE:
            tl.store(PRE_OUT + p_0_3, pre, mask=m_0_3)
        # ---- token 4 of the tile (sequential, registers only)
        m_0_4 = cmask & (tile + 4 < T)
        p_0_4 = (tile + 4) * (B * DN) + b * DN + c
        i = i_0_4
        a = a_const if A_PER_CHANNEL else a_0_4
        th = th_const if TH_PER_CHANNEL else t_0_4
        if HAS_RESET:
            pre = tl.where(r_0_4 != 0, i, _rn_add_f32(_rn_mul_f32(a, v), i))
        else:
            pre = _rn_add_f32(_rn_mul_f32(a, v), i)
        s = tl.maximum(pre - th, 0.0)
        if BIPOLAR:
            s = s - tl.maximum(-pre - th, 0.0)
        v = tl.where(m_0_4, pre - s, v)
        tl.store(S_OUT + p_0_4, s, mask=m_0_4)
        if STORE_PRE:
            tl.store(PRE_OUT + p_0_4, pre, mask=m_0_4)
        # ---- token 5 of the tile (sequential, registers only)
        m_0_5 = cmask & (tile + 5 < T)
        p_0_5 = (tile + 5) * (B * DN) + b * DN + c
        i = i_0_5
        a = a_const if A_PER_CHANNEL else a_0_5
        th = th_const if TH_PER_CHANNEL else t_0_5
        if HAS_RESET:
            pre = tl.where(r_0_5 != 0, i, _rn_add_f32(_rn_mul_f32(a, v), i))
        else:
            pre = _rn_add_f32(_rn_mul_f32(a, v), i)
        s = tl.maximum(pre - th, 0.0)
        if BIPOLAR:
            s = s - tl.maximum(-pre - th, 0.0)
        v = tl.where(m_0_5, pre - s, v)
        tl.store(S_OUT + p_0_5, s, mask=m_0_5)
        if STORE_PRE:
            tl.store(PRE_OUT + p_0_5, pre, mask=m_0_5)
        # ---- token 6 of the tile (sequential, registers only)
        m_0_6 = cmask & (tile + 6 < T)
        p_0_6 = (tile + 6) * (B * DN) + b * DN + c
        i = i_0_6
        a = a_const if A_PER_CHANNEL else a_0_6
        th = th_const if TH_PER_CHANNEL else t_0_6
        if HAS_RESET:
            pre = tl.where(r_0_6 != 0, i, _rn_add_f32(_rn_mul_f32(a, v), i))
        else:
            pre = _rn_add_f32(_rn_mul_f32(a, v), i)
        s = tl.maximum(pre - th, 0.0)
        if BIPOLAR:
            s = s - tl.maximum(-pre - th, 0.0)
        v = tl.where(m_0_6, pre - s, v)
        tl.store(S_OUT + p_0_6, s, mask=m_0_6)
        if STORE_PRE:
            tl.store(PRE_OUT + p_0_6, pre, mask=m_0_6)
        # ---- token 7 of the tile (sequential, registers only)
        m_0_7 = cmask & (tile + 7 < T)
        p_0_7 = (tile + 7) * (B * DN) + b * DN + c
        i = i_0_7
        a = a_const if A_PER_CHANNEL else a_0_7
        th = th_const if TH_PER_CHANNEL else t_0_7
        if HAS_RESET:
            pre = tl.where(r_0_7 != 0, i, _rn_add_f32(_rn_mul_f32(a, v), i))
        else:
            pre = _rn_add_f32(_rn_mul_f32(a, v), i)
        s = tl.maximum(pre - th, 0.0)
        if BIPOLAR:
            s = s - tl.maximum(-pre - th, 0.0)
        v = tl.where(m_0_7, pre - s, v)
        tl.store(S_OUT + p_0_7, s, mask=m_0_7)
        if STORE_PRE:
            tl.store(PRE_OUT + p_0_7, pre, mask=m_0_7)
        # ---- token 8 of the tile (sequential, registers only)
        m_0_8 = cmask & (tile + 8 < T)
        p_0_8 = (tile + 8) * (B * DN) + b * DN + c
        i = i_0_8
        a = a_const if A_PER_CHANNEL else a_0_8
        th = th_const if TH_PER_CHANNEL else t_0_8
        if HAS_RESET:
            pre = tl.where(r_0_8 != 0, i, _rn_add_f32(_rn_mul_f32(a, v), i))
        else:
            pre = _rn_add_f32(_rn_mul_f32(a, v), i)
        s = tl.maximum(pre - th, 0.0)
        if BIPOLAR:
            s = s - tl.maximum(-pre - th, 0.0)
        v = tl.where(m_0_8, pre - s, v)
        tl.store(S_OUT + p_0_8, s, mask=m_0_8)
        if STORE_PRE:
            tl.store(PRE_OUT + p_0_8, pre, mask=m_0_8)
        # ---- token 9 of the tile (sequential, registers only)
        m_0_9 = cmask & (tile + 9 < T)
        p_0_9 = (tile + 9) * (B * DN) + b * DN + c
        i = i_0_9
        a = a_const if A_PER_CHANNEL else a_0_9
        th = th_const if TH_PER_CHANNEL else t_0_9
        if HAS_RESET:
            pre = tl.where(r_0_9 != 0, i, _rn_add_f32(_rn_mul_f32(a, v), i))
        else:
            pre = _rn_add_f32(_rn_mul_f32(a, v), i)
        s = tl.maximum(pre - th, 0.0)
        if BIPOLAR:
            s = s - tl.maximum(-pre - th, 0.0)
        v = tl.where(m_0_9, pre - s, v)
        tl.store(S_OUT + p_0_9, s, mask=m_0_9)
        if STORE_PRE:
            tl.store(PRE_OUT + p_0_9, pre, mask=m_0_9)
        # ---- token 10 of the tile (sequential, registers only)
        m_0_10 = cmask & (tile + 10 < T)
        p_0_10 = (tile + 10) * (B * DN) + b * DN + c
        i = i_0_10
        a = a_const if A_PER_CHANNEL else a_0_10
        th = th_const if TH_PER_CHANNEL else t_0_10
        if HAS_RESET:
            pre = tl.where(r_0_10 != 0, i, _rn_add_f32(_rn_mul_f32(a, v), i))
        else:
            pre = _rn_add_f32(_rn_mul_f32(a, v), i)
        s = tl.maximum(pre - th, 0.0)
        if BIPOLAR:
            s = s - tl.maximum(-pre - th, 0.0)
        v = tl.where(m_0_10, pre - s, v)
        tl.store(S_OUT + p_0_10, s, mask=m_0_10)
        if STORE_PRE:
            tl.store(PRE_OUT + p_0_10, pre, mask=m_0_10)
        # ---- token 11 of the tile (sequential, registers only)
        m_0_11 = cmask & (tile + 11 < T)
        p_0_11 = (tile + 11) * (B * DN) + b * DN + c
        i = i_0_11
        a = a_const if A_PER_CHANNEL else a_0_11
        th = th_const if TH_PER_CHANNEL else t_0_11
        if HAS_RESET:
            pre = tl.where(r_0_11 != 0, i, _rn_add_f32(_rn_mul_f32(a, v), i))
        else:
            pre = _rn_add_f32(_rn_mul_f32(a, v), i)
        s = tl.maximum(pre - th, 0.0)
        if BIPOLAR:
            s = s - tl.maximum(-pre - th, 0.0)
        v = tl.where(m_0_11, pre - s, v)
        tl.store(S_OUT + p_0_11, s, mask=m_0_11)
        if STORE_PRE:
            tl.store(PRE_OUT + p_0_11, pre, mask=m_0_11)
        # ---- token 12 of the tile (sequential, registers only)
        m_0_12 = cmask & (tile + 12 < T)
        p_0_12 = (tile + 12) * (B * DN) + b * DN + c
        i = i_0_12
        a = a_const if A_PER_CHANNEL else a_0_12
        th = th_const if TH_PER_CHANNEL else t_0_12
        if HAS_RESET:
            pre = tl.where(r_0_12 != 0, i, _rn_add_f32(_rn_mul_f32(a, v), i))
        else:
            pre = _rn_add_f32(_rn_mul_f32(a, v), i)
        s = tl.maximum(pre - th, 0.0)
        if BIPOLAR:
            s = s - tl.maximum(-pre - th, 0.0)
        v = tl.where(m_0_12, pre - s, v)
        tl.store(S_OUT + p_0_12, s, mask=m_0_12)
        if STORE_PRE:
            tl.store(PRE_OUT + p_0_12, pre, mask=m_0_12)
        # ---- token 13 of the tile (sequential, registers only)
        m_0_13 = cmask & (tile + 13 < T)
        p_0_13 = (tile + 13) * (B * DN) + b * DN + c
        i = i_0_13
        a = a_const if A_PER_CHANNEL else a_0_13
        th = th_const if TH_PER_CHANNEL else t_0_13
        if HAS_RESET:
            pre = tl.where(r_0_13 != 0, i, _rn_add_f32(_rn_mul_f32(a, v), i))
        else:
            pre = _rn_add_f32(_rn_mul_f32(a, v), i)
        s = tl.maximum(pre - th, 0.0)
        if BIPOLAR:
            s = s - tl.maximum(-pre - th, 0.0)
        v = tl.where(m_0_13, pre - s, v)
        tl.store(S_OUT + p_0_13, s, mask=m_0_13)
        if STORE_PRE:
            tl.store(PRE_OUT + p_0_13, pre, mask=m_0_13)
        # ---- token 14 of the tile (sequential, registers only)
        m_0_14 = cmask & (tile + 14 < T)
        p_0_14 = (tile + 14) * (B * DN) + b * DN + c
        i = i_0_14
        a = a_const if A_PER_CHANNEL else a_0_14
        th = th_const if TH_PER_CHANNEL else t_0_14
        if HAS_RESET:
            pre = tl.where(r_0_14 != 0, i, _rn_add_f32(_rn_mul_f32(a, v), i))
        else:
            pre = _rn_add_f32(_rn_mul_f32(a, v), i)
        s = tl.maximum(pre - th, 0.0)
        if BIPOLAR:
            s = s - tl.maximum(-pre - th, 0.0)
        v = tl.where(m_0_14, pre - s, v)
        tl.store(S_OUT + p_0_14, s, mask=m_0_14)
        if STORE_PRE:
            tl.store(PRE_OUT + p_0_14, pre, mask=m_0_14)
        # ---- token 15 of the tile (sequential, registers only)
        m_0_15 = cmask & (tile + 15 < T)
        p_0_15 = (tile + 15) * (B * DN) + b * DN + c
        i = i_0_15
        a = a_const if A_PER_CHANNEL else a_0_15
        th = th_const if TH_PER_CHANNEL else t_0_15
        if HAS_RESET:
            pre = tl.where(r_0_15 != 0, i, _rn_add_f32(_rn_mul_f32(a, v), i))
        else:
            pre = _rn_add_f32(_rn_mul_f32(a, v), i)
        s = tl.maximum(pre - th, 0.0)
        if BIPOLAR:
            s = s - tl.maximum(-pre - th, 0.0)
        v = tl.where(m_0_15, pre - s, v)
        tl.store(S_OUT + p_0_15, s, mask=m_0_15)
        if STORE_PRE:
            tl.store(PRE_OUT + p_0_15, pre, mask=m_0_15)
        # ---- token 16 of the tile (sequential, registers only)
        m_0_16 = cmask & (tile + 16 < T)
        p_0_16 = (tile + 16) * (B * DN) + b * DN + c
        i = i_0_16
        a = a_const if A_PER_CHANNEL else a_0_16
        th = th_const if TH_PER_CHANNEL else t_0_16
        if HAS_RESET:
            pre = tl.where(r_0_16 != 0, i, _rn_add_f32(_rn_mul_f32(a, v), i))
        else:
            pre = _rn_add_f32(_rn_mul_f32(a, v), i)
        s = tl.maximum(pre - th, 0.0)
        if BIPOLAR:
            s = s - tl.maximum(-pre - th, 0.0)
        v = tl.where(m_0_16, pre - s, v)
        tl.store(S_OUT + p_0_16, s, mask=m_0_16)
        if STORE_PRE:
            tl.store(PRE_OUT + p_0_16, pre, mask=m_0_16)
        # ---- token 17 of the tile (sequential, registers only)
        m_0_17 = cmask & (tile + 17 < T)
        p_0_17 = (tile + 17) * (B * DN) + b * DN + c
        i = i_0_17
        a = a_const if A_PER_CHANNEL else a_0_17
        th = th_const if TH_PER_CHANNEL else t_0_17
        if HAS_RESET:
            pre = tl.where(r_0_17 != 0, i, _rn_add_f32(_rn_mul_f32(a, v), i))
        else:
            pre = _rn_add_f32(_rn_mul_f32(a, v), i)
        s = tl.maximum(pre - th, 0.0)
        if BIPOLAR:
            s = s - tl.maximum(-pre - th, 0.0)
        v = tl.where(m_0_17, pre - s, v)
        tl.store(S_OUT + p_0_17, s, mask=m_0_17)
        if STORE_PRE:
            tl.store(PRE_OUT + p_0_17, pre, mask=m_0_17)
        # ---- token 18 of the tile (sequential, registers only)
        m_0_18 = cmask & (tile + 18 < T)
        p_0_18 = (tile + 18) * (B * DN) + b * DN + c
        i = i_0_18
        a = a_const if A_PER_CHANNEL else a_0_18
        th = th_const if TH_PER_CHANNEL else t_0_18
        if HAS_RESET:
            pre = tl.where(r_0_18 != 0, i, _rn_add_f32(_rn_mul_f32(a, v), i))
        else:
            pre = _rn_add_f32(_rn_mul_f32(a, v), i)
        s = tl.maximum(pre - th, 0.0)
        if BIPOLAR:
            s = s - tl.maximum(-pre - th, 0.0)
        v = tl.where(m_0_18, pre - s, v)
        tl.store(S_OUT + p_0_18, s, mask=m_0_18)
        if STORE_PRE:
            tl.store(PRE_OUT + p_0_18, pre, mask=m_0_18)
        # ---- token 19 of the tile (sequential, registers only)
        m_0_19 = cmask & (tile + 19 < T)
        p_0_19 = (tile + 19) * (B * DN) + b * DN + c
        i = i_0_19
        a = a_const if A_PER_CHANNEL else a_0_19
        th = th_const if TH_PER_CHANNEL else t_0_19
        if HAS_RESET:
            pre = tl.where(r_0_19 != 0, i, _rn_add_f32(_rn_mul_f32(a, v), i))
        else:
            pre = _rn_add_f32(_rn_mul_f32(a, v), i)
        s = tl.maximum(pre - th, 0.0)
        if BIPOLAR:
            s = s - tl.maximum(-pre - th, 0.0)
        v = tl.where(m_0_19, pre - s, v)
        tl.store(S_OUT + p_0_19, s, mask=m_0_19)
        if STORE_PRE:
            tl.store(PRE_OUT + p_0_19, pre, mask=m_0_19)
        # ---- token 20 of the tile (sequential, registers only)
        m_0_20 = cmask & (tile + 20 < T)
        p_0_20 = (tile + 20) * (B * DN) + b * DN + c
        i = i_0_20
        a = a_const if A_PER_CHANNEL else a_0_20
        th = th_const if TH_PER_CHANNEL else t_0_20
        if HAS_RESET:
            pre = tl.where(r_0_20 != 0, i, _rn_add_f32(_rn_mul_f32(a, v), i))
        else:
            pre = _rn_add_f32(_rn_mul_f32(a, v), i)
        s = tl.maximum(pre - th, 0.0)
        if BIPOLAR:
            s = s - tl.maximum(-pre - th, 0.0)
        v = tl.where(m_0_20, pre - s, v)
        tl.store(S_OUT + p_0_20, s, mask=m_0_20)
        if STORE_PRE:
            tl.store(PRE_OUT + p_0_20, pre, mask=m_0_20)
        # ---- token 21 of the tile (sequential, registers only)
        m_0_21 = cmask & (tile + 21 < T)
        p_0_21 = (tile + 21) * (B * DN) + b * DN + c
        i = i_0_21
        a = a_const if A_PER_CHANNEL else a_0_21
        th = th_const if TH_PER_CHANNEL else t_0_21
        if HAS_RESET:
            pre = tl.where(r_0_21 != 0, i, _rn_add_f32(_rn_mul_f32(a, v), i))
        else:
            pre = _rn_add_f32(_rn_mul_f32(a, v), i)
        s = tl.maximum(pre - th, 0.0)
        if BIPOLAR:
            s = s - tl.maximum(-pre - th, 0.0)
        v = tl.where(m_0_21, pre - s, v)
        tl.store(S_OUT + p_0_21, s, mask=m_0_21)
        if STORE_PRE:
            tl.store(PRE_OUT + p_0_21, pre, mask=m_0_21)
        # ---- token 22 of the tile (sequential, registers only)
        m_0_22 = cmask & (tile + 22 < T)
        p_0_22 = (tile + 22) * (B * DN) + b * DN + c
        i = i_0_22
        a = a_const if A_PER_CHANNEL else a_0_22
        th = th_const if TH_PER_CHANNEL else t_0_22
        if HAS_RESET:
            pre = tl.where(r_0_22 != 0, i, _rn_add_f32(_rn_mul_f32(a, v), i))
        else:
            pre = _rn_add_f32(_rn_mul_f32(a, v), i)
        s = tl.maximum(pre - th, 0.0)
        if BIPOLAR:
            s = s - tl.maximum(-pre - th, 0.0)
        v = tl.where(m_0_22, pre - s, v)
        tl.store(S_OUT + p_0_22, s, mask=m_0_22)
        if STORE_PRE:
            tl.store(PRE_OUT + p_0_22, pre, mask=m_0_22)
        # ---- token 23 of the tile (sequential, registers only)
        m_0_23 = cmask & (tile + 23 < T)
        p_0_23 = (tile + 23) * (B * DN) + b * DN + c
        i = i_0_23
        a = a_const if A_PER_CHANNEL else a_0_23
        th = th_const if TH_PER_CHANNEL else t_0_23
        if HAS_RESET:
            pre = tl.where(r_0_23 != 0, i, _rn_add_f32(_rn_mul_f32(a, v), i))
        else:
            pre = _rn_add_f32(_rn_mul_f32(a, v), i)
        s = tl.maximum(pre - th, 0.0)
        if BIPOLAR:
            s = s - tl.maximum(-pre - th, 0.0)
        v = tl.where(m_0_23, pre - s, v)
        tl.store(S_OUT + p_0_23, s, mask=m_0_23)
        if STORE_PRE:
            tl.store(PRE_OUT + p_0_23, pre, mask=m_0_23)
        # ---- token 24 of the tile (sequential, registers only)
        m_0_24 = cmask & (tile + 24 < T)
        p_0_24 = (tile + 24) * (B * DN) + b * DN + c
        i = i_0_24
        a = a_const if A_PER_CHANNEL else a_0_24
        th = th_const if TH_PER_CHANNEL else t_0_24
        if HAS_RESET:
            pre = tl.where(r_0_24 != 0, i, _rn_add_f32(_rn_mul_f32(a, v), i))
        else:
            pre = _rn_add_f32(_rn_mul_f32(a, v), i)
        s = tl.maximum(pre - th, 0.0)
        if BIPOLAR:
            s = s - tl.maximum(-pre - th, 0.0)
        v = tl.where(m_0_24, pre - s, v)
        tl.store(S_OUT + p_0_24, s, mask=m_0_24)
        if STORE_PRE:
            tl.store(PRE_OUT + p_0_24, pre, mask=m_0_24)
        # ---- token 25 of the tile (sequential, registers only)
        m_0_25 = cmask & (tile + 25 < T)
        p_0_25 = (tile + 25) * (B * DN) + b * DN + c
        i = i_0_25
        a = a_const if A_PER_CHANNEL else a_0_25
        th = th_const if TH_PER_CHANNEL else t_0_25
        if HAS_RESET:
            pre = tl.where(r_0_25 != 0, i, _rn_add_f32(_rn_mul_f32(a, v), i))
        else:
            pre = _rn_add_f32(_rn_mul_f32(a, v), i)
        s = tl.maximum(pre - th, 0.0)
        if BIPOLAR:
            s = s - tl.maximum(-pre - th, 0.0)
        v = tl.where(m_0_25, pre - s, v)
        tl.store(S_OUT + p_0_25, s, mask=m_0_25)
        if STORE_PRE:
            tl.store(PRE_OUT + p_0_25, pre, mask=m_0_25)
        # ---- token 26 of the tile (sequential, registers only)
        m_0_26 = cmask & (tile + 26 < T)
        p_0_26 = (tile + 26) * (B * DN) + b * DN + c
        i = i_0_26
        a = a_const if A_PER_CHANNEL else a_0_26
        th = th_const if TH_PER_CHANNEL else t_0_26
        if HAS_RESET:
            pre = tl.where(r_0_26 != 0, i, _rn_add_f32(_rn_mul_f32(a, v), i))
        else:
            pre = _rn_add_f32(_rn_mul_f32(a, v), i)
        s = tl.maximum(pre - th, 0.0)
        if BIPOLAR:
            s = s - tl.maximum(-pre - th, 0.0)
        v = tl.where(m_0_26, pre - s, v)
        tl.store(S_OUT + p_0_26, s, mask=m_0_26)
        if STORE_PRE:
            tl.store(PRE_OUT + p_0_26, pre, mask=m_0_26)
        # ---- token 27 of the tile (sequential, registers only)
        m_0_27 = cmask & (tile + 27 < T)
        p_0_27 = (tile + 27) * (B * DN) + b * DN + c
        i = i_0_27
        a = a_const if A_PER_CHANNEL else a_0_27
        th = th_const if TH_PER_CHANNEL else t_0_27
        if HAS_RESET:
            pre = tl.where(r_0_27 != 0, i, _rn_add_f32(_rn_mul_f32(a, v), i))
        else:
            pre = _rn_add_f32(_rn_mul_f32(a, v), i)
        s = tl.maximum(pre - th, 0.0)
        if BIPOLAR:
            s = s - tl.maximum(-pre - th, 0.0)
        v = tl.where(m_0_27, pre - s, v)
        tl.store(S_OUT + p_0_27, s, mask=m_0_27)
        if STORE_PRE:
            tl.store(PRE_OUT + p_0_27, pre, mask=m_0_27)
        # ---- token 28 of the tile (sequential, registers only)
        m_0_28 = cmask & (tile + 28 < T)
        p_0_28 = (tile + 28) * (B * DN) + b * DN + c
        i = i_0_28
        a = a_const if A_PER_CHANNEL else a_0_28
        th = th_const if TH_PER_CHANNEL else t_0_28
        if HAS_RESET:
            pre = tl.where(r_0_28 != 0, i, _rn_add_f32(_rn_mul_f32(a, v), i))
        else:
            pre = _rn_add_f32(_rn_mul_f32(a, v), i)
        s = tl.maximum(pre - th, 0.0)
        if BIPOLAR:
            s = s - tl.maximum(-pre - th, 0.0)
        v = tl.where(m_0_28, pre - s, v)
        tl.store(S_OUT + p_0_28, s, mask=m_0_28)
        if STORE_PRE:
            tl.store(PRE_OUT + p_0_28, pre, mask=m_0_28)
        # ---- token 29 of the tile (sequential, registers only)
        m_0_29 = cmask & (tile + 29 < T)
        p_0_29 = (tile + 29) * (B * DN) + b * DN + c
        i = i_0_29
        a = a_const if A_PER_CHANNEL else a_0_29
        th = th_const if TH_PER_CHANNEL else t_0_29
        if HAS_RESET:
            pre = tl.where(r_0_29 != 0, i, _rn_add_f32(_rn_mul_f32(a, v), i))
        else:
            pre = _rn_add_f32(_rn_mul_f32(a, v), i)
        s = tl.maximum(pre - th, 0.0)
        if BIPOLAR:
            s = s - tl.maximum(-pre - th, 0.0)
        v = tl.where(m_0_29, pre - s, v)
        tl.store(S_OUT + p_0_29, s, mask=m_0_29)
        if STORE_PRE:
            tl.store(PRE_OUT + p_0_29, pre, mask=m_0_29)
        # ---- token 30 of the tile (sequential, registers only)
        m_0_30 = cmask & (tile + 30 < T)
        p_0_30 = (tile + 30) * (B * DN) + b * DN + c
        i = i_0_30
        a = a_const if A_PER_CHANNEL else a_0_30
        th = th_const if TH_PER_CHANNEL else t_0_30
        if HAS_RESET:
            pre = tl.where(r_0_30 != 0, i, _rn_add_f32(_rn_mul_f32(a, v), i))
        else:
            pre = _rn_add_f32(_rn_mul_f32(a, v), i)
        s = tl.maximum(pre - th, 0.0)
        if BIPOLAR:
            s = s - tl.maximum(-pre - th, 0.0)
        v = tl.where(m_0_30, pre - s, v)
        tl.store(S_OUT + p_0_30, s, mask=m_0_30)
        if STORE_PRE:
            tl.store(PRE_OUT + p_0_30, pre, mask=m_0_30)
        # ---- token 31 of the tile (sequential, registers only)
        m_0_31 = cmask & (tile + 31 < T)
        p_0_31 = (tile + 31) * (B * DN) + b * DN + c
        i = i_0_31
        a = a_const if A_PER_CHANNEL else a_0_31
        th = th_const if TH_PER_CHANNEL else t_0_31
        if HAS_RESET:
            pre = tl.where(r_0_31 != 0, i, _rn_add_f32(_rn_mul_f32(a, v), i))
        else:
            pre = _rn_add_f32(_rn_mul_f32(a, v), i)
        s = tl.maximum(pre - th, 0.0)
        if BIPOLAR:
            s = s - tl.maximum(-pre - th, 0.0)
        v = tl.where(m_0_31, pre - s, v)
        tl.store(S_OUT + p_0_31, s, mask=m_0_31)
        if STORE_PRE:
            tl.store(PRE_OUT + p_0_31, pre, mask=m_0_31)
        # rotate the register buffers (slot s <- slot s+1)
        i_0_0 = i_1_0
        i_0_1 = i_1_1
        i_0_2 = i_1_2
        i_0_3 = i_1_3
        i_0_4 = i_1_4
        i_0_5 = i_1_5
        i_0_6 = i_1_6
        i_0_7 = i_1_7
        i_0_8 = i_1_8
        i_0_9 = i_1_9
        i_0_10 = i_1_10
        i_0_11 = i_1_11
        i_0_12 = i_1_12
        i_0_13 = i_1_13
        i_0_14 = i_1_14
        i_0_15 = i_1_15
        i_0_16 = i_1_16
        i_0_17 = i_1_17
        i_0_18 = i_1_18
        i_0_19 = i_1_19
        i_0_20 = i_1_20
        i_0_21 = i_1_21
        i_0_22 = i_1_22
        i_0_23 = i_1_23
        i_0_24 = i_1_24
        i_0_25 = i_1_25
        i_0_26 = i_1_26
        i_0_27 = i_1_27
        i_0_28 = i_1_28
        i_0_29 = i_1_29
        i_0_30 = i_1_30
        i_0_31 = i_1_31
        if not A_PER_CHANNEL:
            a_0_0 = a_1_0
            a_0_1 = a_1_1
            a_0_2 = a_1_2
            a_0_3 = a_1_3
            a_0_4 = a_1_4
            a_0_5 = a_1_5
            a_0_6 = a_1_6
            a_0_7 = a_1_7
            a_0_8 = a_1_8
            a_0_9 = a_1_9
            a_0_10 = a_1_10
            a_0_11 = a_1_11
            a_0_12 = a_1_12
            a_0_13 = a_1_13
            a_0_14 = a_1_14
            a_0_15 = a_1_15
            a_0_16 = a_1_16
            a_0_17 = a_1_17
            a_0_18 = a_1_18
            a_0_19 = a_1_19
            a_0_20 = a_1_20
            a_0_21 = a_1_21
            a_0_22 = a_1_22
            a_0_23 = a_1_23
            a_0_24 = a_1_24
            a_0_25 = a_1_25
            a_0_26 = a_1_26
            a_0_27 = a_1_27
            a_0_28 = a_1_28
            a_0_29 = a_1_29
            a_0_30 = a_1_30
            a_0_31 = a_1_31
        if not TH_PER_CHANNEL:
            t_0_0 = t_1_0
            t_0_1 = t_1_1
            t_0_2 = t_1_2
            t_0_3 = t_1_3
            t_0_4 = t_1_4
            t_0_5 = t_1_5
            t_0_6 = t_1_6
            t_0_7 = t_1_7
            t_0_8 = t_1_8
            t_0_9 = t_1_9
            t_0_10 = t_1_10
            t_0_11 = t_1_11
            t_0_12 = t_1_12
            t_0_13 = t_1_13
            t_0_14 = t_1_14
            t_0_15 = t_1_15
            t_0_16 = t_1_16
            t_0_17 = t_1_17
            t_0_18 = t_1_18
            t_0_19 = t_1_19
            t_0_20 = t_1_20
            t_0_21 = t_1_21
            t_0_22 = t_1_22
            t_0_23 = t_1_23
            t_0_24 = t_1_24
            t_0_25 = t_1_25
            t_0_26 = t_1_26
            t_0_27 = t_1_27
            t_0_28 = t_1_28
            t_0_29 = t_1_29
            t_0_30 = t_1_30
            t_0_31 = t_1_31
        if HAS_RESET:
            r_0_0 = r_1_0
            r_0_1 = r_1_1
            r_0_2 = r_1_2
            r_0_3 = r_1_3
            r_0_4 = r_1_4
            r_0_5 = r_1_5
            r_0_6 = r_1_6
            r_0_7 = r_1_7
            r_0_8 = r_1_8
            r_0_9 = r_1_9
            r_0_10 = r_1_10
            r_0_11 = r_1_11
            r_0_12 = r_1_12
            r_0_13 = r_1_13
            r_0_14 = r_1_14
            r_0_15 = r_1_15
            r_0_16 = r_1_16
            r_0_17 = r_1_17
            r_0_18 = r_1_18
            r_0_19 = r_1_19
            r_0_20 = r_1_20
            r_0_21 = r_1_21
            r_0_22 = r_1_22
            r_0_23 = r_1_23
            r_0_24 = r_1_24
            r_0_25 = r_1_25
            r_0_26 = r_1_26
            r_0_27 = r_1_27
            r_0_28 = r_1_28
            r_0_29 = r_1_29
            r_0_30 = r_1_30
            r_0_31 = r_1_31
    tl.store(VLAST_OUT + offs, v, mask=cmask)
