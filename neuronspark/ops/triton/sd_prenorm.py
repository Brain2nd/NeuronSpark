"""SD pre-norm: the RMSNorm of the residual stream in front of each SD mixer's in-projection, rounded to bf16 (W_in's
operand), and the fused residual addition + next pre-norm between two consecutive SD layers.  Training regime.

Ops (registry names ``sd_norm`` / ``sd_resid_norm``):

* ``sd_norm(h, gain, eps) -> hn``: ``hn = bf16((h * rstd) * gain)``, ``rstd = 1 / sqrt(sum_d h^2 / D + eps)`` per row.
  ``h`` (T, B, D) fp32 with unit last stride, contiguous or the (0, 1)-transposed view of a contiguous tensor (the
  stream's B-major storage); ``hn`` is bf16 in h's own layout (``torch.empty_like(h, dtype=bf16)``), so
  ``hn.transpose(0, 1)`` is the contiguous (B, T, D) operand of the in-projection.  Backward: ``dh`` fp32 in h's
  layout, ``dgain`` fp32 (D,).
* ``sd_resid_norm(h, upd, gain_next, eps) -> (h_new, hn_next)``: ``h_new = h + float(upd)`` (one fp32 add.rn, output
  strides = torch's add of the same operands) and ``hn_next = sd_norm(h_new, gain_next, eps)`` from the same registers.
  Backward for the incoming (g_h, g_hn): ``g = g_h + dx_norm`` (add.rn), ``dh = g`` fp32, ``dupd = bf16(g)`` (RNE)
  with upd's strides, ``dgain_next`` as ``sd_norm``.

Numerics.  The kernel bodies are flash-linear-attention 0.5.2's RMSNorm kernels (``layer_norm_fwd_kernel1`` /
``layer_norm_bwd_kernel1``, IS_RMS_NORM, weight, one group) line for line with only the addressing changed (rows read
and written through their strides): ``div.full.f32`` for ``/ D`` and ``1 / .``, ``sqrt.approx.ftz.f32``, FMA
contraction on (Triton's default), the bf16 rounding ``cvt.rn.bf16.f32`` and the exact widening ``cvt.f32.bf16`` as
inline PTX, the joining add as inline ``add.rn.f32`` (no contraction with the product next to it).  The bf16 pointers
of the reductions are not specialised on alignment, which keeps the reduction operands at 4 elements per thread (an
aligned bf16 access would widen the vector to 8 and reorder the row sums).  The gain gradient: per program s of the
NS = min(#SM, rows) programs of BS = ceil(rows / NS) consecutive rows, ``dw_partial[s] = sum_rows dy * xhat`` in row
order (columns independent), then ``dw_partial.view(1, -1, D).sum(1)`` (torch's reduction): the grouping depends on
the SM count of the GPU.  Launch constants: forward 8 warps, backward dx 4 warps, dw 128 columns x 1 warp,
num_stages 3.

Contract (anything else raises): CUDA fp32 ``h`` of 3 axes with unit last stride and both row strides divisible by
16, contiguous or the transposed view of a contiguous tensor; 512 < D <= 16384, D % 16 == 0; fewer than 2^31
elements; ``gain`` a contiguous fp32 (D,) tensor on h's device; the incoming hn gradient bf16 of h's shape with unit
last stride; for ``sd_resid_norm`` ``upd`` bf16 of h's shape with unit last stride, dense.

Fixed constants: forward / backward warps 8 / 4, separate dx and dw backward kernels, dw block 128 x 1 warp.
"""
from __future__ import annotations

from typing import Dict, Tuple

import torch
import triton
import triton.language as tl

from neuronspark.ops.triton import num_sms, require_cuda
from neuronspark.ops.triton._ptx import _rn_add_f32, _sr_bf16_to_f32, _sr_f32_to_bf16

__all__ = [
    "FWD_WARPS", "BWD_WARPS", "DW_BLOCK", "DW_WARPS", "NUM_STAGES",
    "SDNormFn", "SDResidNormFn", "sd_norm", "sd_norm_ref", "sd_resid_norm", "sd_resid_norm_ref", "bwd_grid",
]

# ------------------------------------------------------------------------------------------------ launch constants
FWD_WARPS = 8          # forward kernels
BWD_WARPS = 4          # backward dx kernels
DW_BLOCK = 128         # dw kernel: columns per program
DW_WARPS = 1           # dw kernel warps
NUM_STAGES = 3         # triton.Config's default num_stages (fla's configurations set none)


def bwd_grid(rows: int, device) -> Tuple[int, int]:
    """(NS, BS) of the gain-gradient partial sums: NS = min(#SM, rows) programs of BS = ceil(rows / NS) consecutive
    rows (fla layer_norm_bwd with one group).  Depends on the SM count of the device."""
    ns = min(triton.cdiv(num_sms(device), 1), rows // 1) * 1
    return ns, triton.cdiv(rows, ns)


# ------------------------------------------------------------------------------------------------ kernels
@triton.jit(do_not_specialize_on_alignment=["y"])
def _norm_fwd_kernel(x, y, w, rstd, eps, S1, sx0, sx1, sy0, sy1, D: tl.constexpr, BD: tl.constexpr):
    """Row i_t (T-major order) of y = bf16(rms_norm(x) * w), x and y addressed through their strides."""
    i_t = tl.program_id(0)
    i_0 = i_t // S1
    i_1 = i_t % S1
    x += i_0.to(tl.int64) * sx0 + i_1.to(tl.int64) * sx1
    y += i_0.to(tl.int64) * sy0 + i_1.to(tl.int64) * sy1

    o_d = tl.arange(0, BD)
    m_d = o_d < D
    b_x = tl.load(x + o_d, mask=m_d, other=0.0).to(tl.float32)
    b_xbar = tl.where(m_d, b_x, 0.0)
    b_var = tl.sum(b_xbar * b_xbar, axis=0) / D
    b_rstd = 1 / tl.sqrt(b_var + eps)
    tl.store(rstd + i_t, b_rstd)

    b_w = tl.load(w + o_d, mask=m_d).to(tl.float32)
    b_x_hat = b_x * b_rstd
    b_y = b_x_hat * b_w

    tl.store(y + o_d, _sr_f32_to_bf16(b_y), mask=m_d)


@triton.jit(do_not_specialize_on_alignment=["u", "y"])
def _resid_norm_fwd_kernel(h, u, o, y, w, rstd, eps, S1, sh0, sh1, su0, su1, so0, so1, sy0, sy1,
                           D: tl.constexpr, BD: tl.constexpr):
    """Row i_t (T-major order) of o = h + fp32(u) (add.rn) and, from the same registers, y = bf16(rms_norm(o) * w)."""
    i_t = tl.program_id(0)
    i_0 = i_t // S1
    i_1 = i_t % S1
    h += i_0.to(tl.int64) * sh0 + i_1.to(tl.int64) * sh1
    u += i_0.to(tl.int64) * su0 + i_1.to(tl.int64) * su1
    o += i_0.to(tl.int64) * so0 + i_1.to(tl.int64) * so1
    y += i_0.to(tl.int64) * sy0 + i_1.to(tl.int64) * sy1

    o_d = tl.arange(0, BD)
    m_d = o_d < D
    b_h = tl.load(h + o_d, mask=m_d, other=0.0)
    b_u = tl.load(u + o_d, mask=m_d, other=0.0)
    b_x = _rn_add_f32(b_h, _sr_bf16_to_f32(b_u))
    tl.store(o + o_d, b_x, mask=m_d)

    b_xbar = tl.where(m_d, b_x, 0.0)
    b_var = tl.sum(b_xbar * b_xbar, axis=0) / D
    b_rstd = 1 / tl.sqrt(b_var + eps)
    tl.store(rstd + i_t, b_rstd)

    b_w = tl.load(w + o_d, mask=m_d).to(tl.float32)
    b_x_hat = b_x * b_rstd
    b_y = b_x_hat * b_w

    tl.store(y + o_d, _sr_f32_to_bf16(b_y), mask=m_d)


@triton.jit(do_not_specialize_on_alignment=["dy"])
def _norm_bwd_dx_kernel(x, w, dy, dx, rstd, S1, sx0, sx1, sdy0, sdy1, sdx0, sdx1,
                        D: tl.constexpr, BD: tl.constexpr):
    """dx of row i_t: the loop body of fla's layer_norm_bwd_kernel1 without the dw accumulation."""
    i_t = tl.program_id(0)
    i_0 = i_t // S1
    i_1 = i_t % S1

    o_d = tl.arange(0, BD)
    mask = o_d < D

    b_w = tl.load(w + o_d, mask=mask).to(tl.float32)

    b_x = tl.load(x + i_0.to(tl.int64) * sx0 + i_1.to(tl.int64) * sx1 + o_d, mask=mask,
                  other=0).to(tl.float32)
    b_dy = _sr_bf16_to_f32(tl.load(dy + i_0.to(tl.int64) * sdy0 + i_1.to(tl.int64) * sdy1 + o_d,
                                   mask=mask, other=0))

    b_rstd = tl.load(rstd + i_t)
    # Compute dx
    b_xhat = b_x * b_rstd
    b_xhat = tl.where(mask, b_xhat, 0.0)
    b_wdy = b_dy * b_w
    b_c1 = tl.sum(b_xhat * b_wdy, axis=0) / D
    b_dx = (b_wdy - b_xhat * b_c1) * b_rstd
    # Write dx
    b_dx = tl.cast(b_dx, dtype=dx.dtype.element_ty, fp_downcast_rounding='rtne')
    tl.store(dx + i_0.to(tl.int64) * sdx0 + i_1.to(tl.int64) * sdx1 + o_d, b_dx, mask=mask)


@triton.jit(do_not_specialize_on_alignment=["dy", "gu"])
def _resid_norm_bwd_dx_kernel(x, w, dy, gp, gx, gu, rstd, S1, sx0, sx1, sdy0, sdy1, sgp0, sgp1, sgx0, sgx1, sgu0, sgu1,
                              D: tl.constexpr, BD: tl.constexpr, HAS_GP: tl.constexpr):
    """The dx body for row i_t, then g = gp + dx (add.rn) stored fp32 (gx) and as bf16 (gu, RNE)."""
    i_t = tl.program_id(0)
    i_0 = i_t // S1
    i_1 = i_t % S1

    o_d = tl.arange(0, BD)
    mask = o_d < D

    b_w = tl.load(w + o_d, mask=mask).to(tl.float32)

    b_x = tl.load(x + i_0.to(tl.int64) * sx0 + i_1.to(tl.int64) * sx1 + o_d, mask=mask,
                  other=0).to(tl.float32)
    b_dy = _sr_bf16_to_f32(tl.load(dy + i_0.to(tl.int64) * sdy0 + i_1.to(tl.int64) * sdy1 + o_d,
                                   mask=mask, other=0))

    b_rstd = tl.load(rstd + i_t)
    # Compute dx
    b_xhat = b_x * b_rstd
    b_xhat = tl.where(mask, b_xhat, 0.0)
    b_wdy = b_dy * b_w
    b_c1 = tl.sum(b_xhat * b_wdy, axis=0) / D
    b_dx = (b_wdy - b_xhat * b_c1) * b_rstd
    b_dx = tl.cast(b_dx, dtype=gx.dtype.element_ty, fp_downcast_rounding='rtne')
    if HAS_GP:
        b_gp = tl.load(gp + i_0.to(tl.int64) * sgp0 + i_1.to(tl.int64) * sgp1 + o_d, mask=mask, other=0.0)
        b_g = _rn_add_f32(b_gp, b_dx)
    else:
        b_g = b_dx
    tl.store(gx + i_0.to(tl.int64) * sgx0 + i_1.to(tl.int64) * sgx1 + o_d, b_g, mask=mask)
    tl.store(gu + i_0.to(tl.int64) * sgu0 + i_1.to(tl.int64) * sgu1 + o_d, _sr_f32_to_bf16(b_g), mask=mask)


@triton.jit
def _dw_row(x, dy, rstd, i_t, end, S1, sx0, sx1, sdy0, sdy1, o_d, mask):
    """Row i_t's (b_dy, b_xhat) of the dw accumulation (rows i_t >= end are loaded masked and never used)."""
    ok = i_t < end
    i_0 = i_t // S1
    i_1 = i_t % S1
    b_x = tl.load(x + i_0.to(tl.int64) * sx0 + i_1.to(tl.int64) * sx1 + o_d, mask=mask & ok,
                  other=0).to(tl.float32)
    b_dy = _sr_bf16_to_f32(tl.load(dy + i_0.to(tl.int64) * sdy0 + i_1.to(tl.int64) * sdy1 + o_d,
                                   mask=mask & ok, other=0))
    b_rstd = tl.load(rstd + i_t, mask=ok, other=0.0)
    b_xhat = b_x * b_rstd
    b_xhat = tl.where(mask, b_xhat, 0.0)
    return b_dy, b_xhat


@triton.jit
def _norm_bwd_dw_kernel(x, dy, dw, rstd, T, S1, sx0, sx1, sdy0, sdy1,
                        D: tl.constexpr, BS: tl.constexpr, BLOCK_D: tl.constexpr):
    """Program (i_s, i_d): columns [i_d BLOCK_D, (i_d + 1) BLOCK_D) of the partial sum of program i_s,
    b_dw += b_dy * b_xhat over rows i_s BS .. min(i_s BS + BS, T) - 1 in that order, from zero.  The rows are taken 8 at
    a time: the 8 rows' loads first, then the 8 accumulations in row order, each guarded by its row's validity."""
    i_s = tl.program_id(0)
    i_d = tl.program_id(1)

    o_d = i_d * BLOCK_D + tl.arange(0, BLOCK_D)
    mask = o_d < D

    b_dw = tl.zeros((BLOCK_D,), dtype=tl.float32)
    start = i_s * BS
    end = tl.minimum(start + BS, T)
    for i0 in range(start, end, 8):
        y0, h0 = _dw_row(x, dy, rstd, i0 + 0, end, S1, sx0, sx1, sdy0, sdy1, o_d, mask)
        y1, h1 = _dw_row(x, dy, rstd, i0 + 1, end, S1, sx0, sx1, sdy0, sdy1, o_d, mask)
        y2, h2 = _dw_row(x, dy, rstd, i0 + 2, end, S1, sx0, sx1, sdy0, sdy1, o_d, mask)
        y3, h3 = _dw_row(x, dy, rstd, i0 + 3, end, S1, sx0, sx1, sdy0, sdy1, o_d, mask)
        y4, h4 = _dw_row(x, dy, rstd, i0 + 4, end, S1, sx0, sx1, sdy0, sdy1, o_d, mask)
        y5, h5 = _dw_row(x, dy, rstd, i0 + 5, end, S1, sx0, sx1, sdy0, sdy1, o_d, mask)
        y6, h6 = _dw_row(x, dy, rstd, i0 + 6, end, S1, sx0, sx1, sdy0, sdy1, o_d, mask)
        y7, h7 = _dw_row(x, dy, rstd, i0 + 7, end, S1, sx0, sx1, sdy0, sdy1, o_d, mask)
        b_dw = tl.where(i0 + 0 < end, b_dw + y0 * h0, b_dw)
        b_dw = tl.where(i0 + 1 < end, b_dw + y1 * h1, b_dw)
        b_dw = tl.where(i0 + 2 < end, b_dw + y2 * h2, b_dw)
        b_dw = tl.where(i0 + 3 < end, b_dw + y3 * h3, b_dw)
        b_dw = tl.where(i0 + 4 < end, b_dw + y4 * h4, b_dw)
        b_dw = tl.where(i0 + 5 < end, b_dw + y5 * h5, b_dw)
        b_dw = tl.where(i0 + 6 < end, b_dw + y6 * h6, b_dw)
        b_dw = tl.where(i0 + 7 < end, b_dw + y7 * h7, b_dw)

    tl.store(dw + i_s * D + o_d, b_dw, mask=mask)


# ------------------------------------------------------------------------------------------------ launchers
def _launch_norm_fwd(h, gain, eps, y, rstd):
    S0, S1, D = (int(s) for s in h.shape)
    _norm_fwd_kernel[(S0 * S1,)](
        h, y, gain, rstd, float(eps), S1, h.stride(0), h.stride(1), y.stride(0), y.stride(1),
        D=D, BD=triton.next_power_of_2(D), num_warps=FWD_WARPS, num_stages=NUM_STAGES)


def _launch_resid_norm_fwd(h, upd, gain, eps, o, y, rstd):
    S0, S1, D = (int(s) for s in h.shape)
    _resid_norm_fwd_kernel[(S0 * S1,)](
        h, upd, o, y, gain, rstd, float(eps), S1, h.stride(0), h.stride(1), upd.stride(0), upd.stride(1),
        o.stride(0), o.stride(1), y.stride(0), y.stride(1), D=D, BD=triton.next_power_of_2(D),
        num_warps=FWD_WARPS, num_stages=NUM_STAGES)


def _launch_dw(x, dy, rstd, dw_partial, bs):
    S0, S1, D = (int(s) for s in x.shape)
    rows = S0 * S1
    ns = int(dw_partial.shape[0])
    _norm_bwd_dw_kernel[(ns, triton.cdiv(D, DW_BLOCK))](
        x, dy, dw_partial, rstd, rows, S1, x.stride(0), x.stride(1), dy.stride(0), dy.stride(1), D=D, BS=bs,
        BLOCK_D=DW_BLOCK, num_warps=DW_WARPS, num_stages=NUM_STAGES)


def _launch_norm_bwd(dy, h, gain, rstd, dx, dw_partial):
    """dy (S0, S1, D) bf16, h (S0, S1, D) fp32 (unit last strides), rstd (S0 S1,) -> dx (h's shape, unit last
    stride), dw_partial (NS, D) fp32."""
    S0, S1, D = (int(s) for s in h.shape)
    if (dy.dtype != torch.bfloat16 or tuple(dy.shape) != tuple(h.shape) or dy.stride(-1) != 1
            or h.dtype != torch.float32 or h.stride(-1) != 1 or dx.dtype != torch.float32
            or tuple(dx.shape) != tuple(h.shape) or dx.stride(-1) != 1 or rstd.dtype != torch.float32
            or rstd.numel() != S0 * S1 or not rstd.is_contiguous()):
        raise RuntimeError(
            "neuronspark.sd_norm backward: dy bf16 and dx fp32 of h's shape with unit last strides, h fp32, rstd fp32 "
            f"contiguous (rows,) required; got dy {dy.dtype} {tuple(dy.shape)} {tuple(dy.stride())}, h {h.dtype} "
            f"{tuple(h.shape)} {tuple(h.stride())}, dx {dx.dtype} {tuple(dx.stride())}, rstd {rstd.dtype} "
            f"{tuple(rstd.shape)}")
    rows = S0 * S1
    ns, bs = bwd_grid(rows, h.device)
    if tuple(dw_partial.shape) != (ns, D):
        raise RuntimeError(f"neuronspark.sd_norm: dw_partial must be ({ns}, {D}), got {tuple(dw_partial.shape)}")
    _norm_bwd_dx_kernel[(rows,)](
        h, gain, dy, dx, rstd, S1, h.stride(0), h.stride(1), dy.stride(0), dy.stride(1), dx.stride(0),
        dx.stride(1), D=D, BD=triton.next_power_of_2(D), num_warps=BWD_WARPS, num_stages=NUM_STAGES)
    _launch_dw(h, dy, rstd, dw_partial, bs)


def _launch_resid_norm_bwd(dy, gp, x, gain, rstd, gx, gu, dw_partial):
    """dy (S0, S1, D) bf16, gp fp32 of x's shape or None, x (S0, S1, D) fp32 (unit last strides), rstd (rows,) ->
    gx fp32 / gu bf16 (unit last strides), dw_partial (NS, D) fp32."""
    S0, S1, D = (int(s) for s in x.shape)
    rows = S0 * S1
    ns, bs = bwd_grid(rows, x.device)
    if tuple(dw_partial.shape) != (ns, D):
        raise RuntimeError(f"neuronspark.sd_resid_norm: dw_partial must be ({ns}, {D}), got {tuple(dw_partial.shape)}")
    has_gp = gp is not None
    g = gp if has_gp else gx
    _resid_norm_bwd_dx_kernel[(rows,)](
        x, gain, dy, g, gx, gu, rstd, S1, x.stride(0), x.stride(1), dy.stride(0), dy.stride(1),
        g.stride(0), g.stride(1), gx.stride(0), gx.stride(1), gu.stride(0), gu.stride(1),
        D=D, BD=triton.next_power_of_2(D), HAS_GP=has_gp, num_warps=BWD_WARPS, num_stages=NUM_STAGES)
    _launch_dw(x, dy, rstd, dw_partial, bs)


# torch's add strides for (h, upd) layouts: a documented first-call cache keyed by (shape, strides) (host-only)
_ADD_STRIDES: Dict[tuple, tuple] = {}


def _add_strides(h: torch.Tensor, upd: torch.Tensor) -> tuple:
    """The strides torch's ``h + upd.to(fp32)`` gives for these operand layouts (the same op on meta tensors)."""
    key = (tuple(h.shape), tuple(h.stride()), tuple(upd.stride()))
    st = _ADD_STRIDES.get(key)
    if st is None:
        hm = torch.empty_strided(h.shape, h.stride(), dtype=torch.float32, device="meta")
        um = torch.empty_strided(upd.shape, upd.stride(), dtype=torch.bfloat16, device="meta")
        st = tuple((hm + um.to(torch.float32)).stride())
        _ADD_STRIDES[key] = st
    if st[2] != 1:
        raise RuntimeError(f"neuronspark.sd_resid_norm: torch's add would give strides {st}; the kernel writes a dense "
                           "last dimension")
    return st


# ------------------------------------------------------------------------------------------------ custom ops
def _norm_fwd_alloc(h):
    S0, S1, _ = (int(s) for s in h.shape)
    y = torch.empty_like(h, dtype=torch.bfloat16)
    rstd = torch.empty((S0 * S1,), dtype=torch.float32, device=h.device)
    return y, rstd


@torch.library.custom_op("neuronspark::sd_norm_fwd", mutates_args=(),
                         schema="(Tensor h, Tensor gain, float eps) -> (Tensor, Tensor)")
def _sd_norm_fwd_op(h, gain, eps):
    """(hn bf16 in h's layout, rstd fp32 (S0 S1,))"""
    y, rstd = _norm_fwd_alloc(h)
    _launch_norm_fwd(h, gain, eps, y, rstd)
    return y, rstd


@_sd_norm_fwd_op.register_fake
def _sd_norm_fwd_fake(h, gain, eps):
    return _norm_fwd_alloc(h)


def _norm_bwd_alloc(h, gain):
    S0, S1, D = (int(s) for s in h.shape)
    ns, _ = bwd_grid(S0 * S1, h.device)
    return torch.empty_like(h), torch.empty((ns, D), dtype=torch.float32, device=gain.device)


@torch.library.custom_op("neuronspark::sd_norm_bwd", mutates_args=(),
                         schema="(Tensor dy, Tensor h, Tensor gain, Tensor rstd) -> (Tensor, Tensor)")
def _sd_norm_bwd_op(dy, h, gain, rstd):
    """(dh fp32 in h's layout, dw_partial fp32 (NS, D))"""
    dx, dwp = _norm_bwd_alloc(h, gain)
    _launch_norm_bwd(dy, h, gain, rstd, dx, dwp)
    return dx, dwp


@_sd_norm_bwd_op.register_fake
def _sd_norm_bwd_fake(dy, h, gain, rstd):
    return _norm_bwd_alloc(h, gain)


def _resid_fwd_alloc(h, upd):
    S0, S1, _ = (int(s) for s in h.shape)
    o = torch.empty_strided(tuple(h.shape), _add_strides(h, upd), dtype=torch.float32, device=h.device)
    y = torch.empty_like(o, dtype=torch.bfloat16)
    rstd = torch.empty((S0 * S1,), dtype=torch.float32, device=h.device)
    return o, y, rstd


@torch.library.custom_op("neuronspark::sd_resid_norm_fwd", mutates_args=(),
                         schema="(Tensor h, Tensor upd, Tensor gain, float eps) -> (Tensor, Tensor, Tensor)")
def _sd_resid_norm_fwd_op(h, upd, gain, eps):
    """(o = h + fp32(upd) fp32 with torch's add strides, y = bf16 rms_norm(o) in o's layout, rstd fp32 (rows,))"""
    o, y, rstd = _resid_fwd_alloc(h, upd)
    _launch_resid_norm_fwd(h, upd, gain, eps, o, y, rstd)
    return o, y, rstd


@_sd_resid_norm_fwd_op.register_fake
def _sd_resid_norm_fwd_fake(h, upd, gain, eps):
    return _resid_fwd_alloc(h, upd)


def _resid_bwd_alloc(x, gain, upd_stride):
    S0, S1, D = (int(s) for s in x.shape)
    ns, _ = bwd_grid(S0 * S1, x.device)
    gx = torch.empty_like(x)
    gu = torch.empty_strided(tuple(x.shape), tuple(int(s) for s in upd_stride), dtype=torch.bfloat16, device=x.device)
    dwp = torch.empty((ns, D), dtype=torch.float32, device=gain.device)
    return gx, gu, dwp


@torch.library.custom_op("neuronspark::sd_resid_norm_bwd", mutates_args=(),
                         schema="(Tensor dy, Tensor? gp, Tensor x, Tensor gain, Tensor rstd, int[] upd_stride) "
                                "-> (Tensor, Tensor, Tensor)")
def _sd_resid_norm_bwd_op(dy, gp, x, gain, rstd, upd_stride):
    """(gx = gp + dx fp32 in x's layout, gu = bf16(gx) with the update's strides, dw_partial fp32 (NS, D))"""
    gx, gu, dwp = _resid_bwd_alloc(x, gain, upd_stride)
    _launch_resid_norm_bwd(dy, gp, x, gain, rstd, gx, gu, dwp)
    return gx, gu, dwp


@_sd_resid_norm_bwd_op.register_fake
def _sd_resid_norm_bwd_fake(dy, gp, x, gain, rstd, upd_stride):
    return _resid_bwd_alloc(x, gain, upd_stride)


# ------------------------------------------------------------------------------------------------ autograd Functions
class SDNormFn(torch.autograd.Function):
    """hn = bf16(rms_norm(h) * gain) in h's layout; saves (h, gain, rstd).  The gain gradient is the partial sums
    reduced by ``dw_partial.view(1, -1, D).sum(1)``.  An undefined incoming gradient gives undefined gradients."""

    @staticmethod
    def forward(ctx, h, gain, eps):
        ctx.set_materialize_grads(False)
        y, rstd = torch.ops.neuronspark.sd_norm_fwd(h, gain, eps)
        ctx.save_for_backward(h, gain, rstd)
        return y

    @staticmethod
    def backward(ctx, dy):
        if dy is None:
            return None, None, None
        h, gain, rstd = ctx.saved_tensors
        if dy.dtype != torch.bfloat16 or tuple(dy.shape) != tuple(h.shape) or dy.stride(-1) != 1:
            raise RuntimeError(
                "neuronspark.sd_norm: the incoming gradient must be bfloat16 of h's shape with unit last stride; got "
                f"{dy.dtype} {tuple(dy.shape)} strides {tuple(dy.stride())}")
        dx, dwp = torch.ops.neuronspark.sd_norm_bwd(dy, h, gain, rstd)
        D = int(h.shape[-1])
        dw = dwp.view(1, -1, D).sum(1).to(gain).view_as(gain)
        need = ctx.needs_input_grad
        return (dx if need[0] else None, dw if need[1] else None, None)


class SDResidNormFn(torch.autograd.Function):
    """(h + fp32(upd), bf16(rms_norm(h + fp32(upd)) * gain)) as one node; saves (o, gain, rstd).  Backward (g_o,
    g_y): dh = g_o + dx, dupd = bf16(dh) with upd's strides, dgain from the partial sums."""

    @staticmethod
    def forward(ctx, h, upd, gain, eps):
        ctx.set_materialize_grads(False)
        o, y, rstd = torch.ops.neuronspark.sd_resid_norm_fwd(h, upd, gain, eps)
        ctx.save_for_backward(o, gain, rstd)
        ctx.upd_dtype = upd.dtype
        ctx.upd_stride = tuple(int(s) for s in upd.stride())
        return o, y

    @staticmethod
    def backward(ctx, g_o, g_y):
        need = ctx.needs_input_grad
        if g_y is None:
            # nothing differentiates through the pre-norm: the residual addition's backward alone
            if g_o is None:
                return None, None, None, None
            return (g_o if need[0] else None, g_o.to(ctx.upd_dtype) if need[1] else None, None, None)
        o, gain, rstd = ctx.saved_tensors
        if g_y.dtype != torch.bfloat16 or tuple(g_y.shape) != tuple(o.shape) or g_y.stride(-1) != 1:
            raise RuntimeError(
                "neuronspark.sd_resid_norm: the incoming gradient of the pre-norm must be bfloat16 of the stream's "
                f"shape with unit last stride; got {g_y.dtype} {tuple(g_y.shape)} strides {tuple(g_y.stride())}")
        if g_o is not None and (g_o.dtype != torch.float32 or tuple(g_o.shape) != tuple(o.shape)
                                or g_o.stride(-1) != 1):
            raise RuntimeError(
                "neuronspark.sd_resid_norm: the residual stream's gradient must be float32 of its shape with unit last "
                f"stride; got {g_o.dtype} {tuple(g_o.shape)} strides {tuple(g_o.stride())}")
        gx, gu, dwp = torch.ops.neuronspark.sd_resid_norm_bwd(g_y, g_o, o, gain, rstd, list(ctx.upd_stride))
        D = int(o.shape[-1])
        dw = dwp.view(1, -1, D).sum(1).to(gain).view_as(gain)
        return (gx if need[0] else None, gu if need[1] else None, dw if need[2] else None, None)


# ------------------------------------------------------------------------------------------------ checks
def _check_input(h: torch.Tensor, gain: torch.Tensor, what: str) -> None:
    if not isinstance(h, torch.Tensor) or not h.is_cuda or h.dtype != torch.float32 or h.dim() != 3:
        raise RuntimeError(f"neuronspark.{what}: h must be a CUDA float32 tensor of 3 axes, got "
                           f"{type(h).__name__} {getattr(h, 'dtype', None)} {tuple(getattr(h, 'shape', ()))} "
                           f"{getattr(h, 'device', None)}")
    S0, S1, D = (int(s) for s in h.shape)
    if S0 * S1 == 0 or h.stride(-1) != 1 or h.stride(0) % 16 or h.stride(1) % 16:
        raise RuntimeError(f"neuronspark.{what}: h must be non-empty with unit last stride and row strides divisible "
                           f"by 16, got shape {tuple(h.shape)} strides {tuple(h.stride())}")
    if not 512 < D <= 16384 or D % 16:
        raise RuntimeError(f"neuronspark.{what}: the feature width must satisfy 512 < D <= 16384 and D % 16 == 0, "
                           f"got D={D}")
    if h.numel() >= 2 ** 31 or max(h.stride(0) * (S0 - 1), 0) + max(h.stride(1) * (S1 - 1), 0) + D > 2 ** 31:
        raise RuntimeError(f"neuronspark.{what}: h must span fewer than 2^31 elements")
    if (not isinstance(gain, torch.Tensor) or gain.device != h.device or gain.dtype != torch.float32
            or gain.dim() != 1 or int(gain.shape[0]) != D or not gain.is_contiguous()):
        raise RuntimeError(f"neuronspark.{what}: the gain must be a contiguous float32 (D={D},) tensor on {h.device}, "
                           f"got {type(gain).__name__} {getattr(gain, 'dtype', None)} "
                           f"{tuple(getattr(gain, 'shape', ()))} {getattr(gain, 'device', None)}")


def _check_dense(name: str, t: torch.Tensor, what: str) -> None:
    """contiguous or the (0, 1)-transposed view of a contiguous tensor, unit last stride (empty_like keeps strides)"""
    if not (t.dim() == 3 and t.stride(-1) == 1 and (t.is_contiguous() or t.transpose(0, 1).is_contiguous())):
        raise RuntimeError(f"neuronspark.{what}: {name} must be non-overlapping and dense with a unit last stride, got "
                           f"shape {tuple(t.shape)} strides {tuple(t.stride())}")


def _check_eps(eps, what: str) -> float:
    if isinstance(eps, bool) or not isinstance(eps, (int, float)):
        raise RuntimeError(f"neuronspark.{what}: eps must be a number, got {eps!r}")
    return float(eps)


# ------------------------------------------------------------------------------------------------ public
def sd_norm(h: torch.Tensor, gain: torch.Tensor, eps: float) -> torch.Tensor:
    """bf16 SD pre-norm of the fp32 stream ``h`` (T, B, D), in h's layout (see the module docstring)."""
    require_cuda(h, "sd_norm")
    _check_input(h, gain, "sd_norm")
    _check_dense("h", h, "sd_norm")
    return SDNormFn.apply(h, gain, _check_eps(eps, "sd_norm"))


def sd_resid_norm(h: torch.Tensor, upd: torch.Tensor, gain_next: torch.Tensor, eps: float):
    """``(h + upd.float(), sd_norm(h + upd.float(), gain_next, eps))`` as one node (see the module docstring)."""
    require_cuda(h, "sd_resid_norm")
    if upd.device != h.device:
        raise RuntimeError(f"neuronspark.sd_resid_norm: CUDA tensors on one device required, got {h.device}, "
                           f"{upd.device}")
    if h.dtype != torch.float32 or upd.dtype != torch.bfloat16 or h.dim() != 3 or tuple(h.shape) != tuple(upd.shape):
        raise RuntimeError(f"neuronspark.sd_resid_norm: fp32 h and bf16 upd of one 3-D shape required, got "
                           f"{tuple(h.shape)} {h.dtype}, {tuple(upd.shape)} {upd.dtype}")
    if h.stride(2) != 1 or upd.stride(2) != 1:
        raise RuntimeError(f"neuronspark.sd_resid_norm: the last dimension must be dense, got strides {h.stride()} "
                           f"{upd.stride()}")
    _check_input(h, gain_next, "sd_resid_norm")
    _check_dense("h", h, "sd_resid_norm")
    _check_dense("upd", upd, "sd_resid_norm")
    return SDResidNormFn.apply(h, upd, gain_next, _check_eps(eps, "sd_resid_norm"))


# ------------------------------------------------------------------------------------------------ references
def sd_norm_ref(h: torch.Tensor, gain: torch.Tensor, eps: float) -> torch.Tensor:
    """Pure torch: ``(h * rsqrt(mean(h^2, -1) + eps) * gain).to(bf16)`` in fp32 (the RMSNorm the kernel computes,
    followed by autocast's bf16 cast of the in-projection operand)."""
    hf = h.float()
    rstd = torch.rsqrt(hf.square().mean(dim=-1, keepdim=True) + float(eps))
    return (hf * rstd * gain.float()).to(torch.bfloat16)


def sd_resid_norm_ref(h: torch.Tensor, upd: torch.Tensor, gain_next: torch.Tensor, eps: float):
    """Pure torch: ``h_new = h.to(fp32) + upd.to(fp32)`` (the residual add),
    ``hn_next = sd_norm_ref(h_new, gain_next, eps)``; returns (h_new, hn_next)."""
    h_new = h.to(torch.float32) + upd.to(torch.float32)
    return h_new, sd_norm_ref(h_new, gain_next, eps)
