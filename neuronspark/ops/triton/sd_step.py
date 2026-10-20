"""SD decode step (T == 1) and the SD pre-norm of the inference paths.

Ops (registry names; inference, no autograd):

* ``sd_prenorm_rows(x, gain, eps) -> hn``: ``hn = bf16((x * rstd) * gain)``, ``rstd = 1 / sqrt(sum_D x^2 / D +
  eps)`` per row; x (M, D) fp32 rows with unit last stride (the residual stream: B rows at decode, T * B rows at
  prefill), gain (D,) fp32; hn (M, D) bf16 contiguous (the in-projection's operand).  Decode and prefill.
* ``sd_step_core(proj, conv_state, ssm_state, conv_w, conv_b, A, dt_bias, D, theta, gamma, eps) -> xs``: one token of
  an SD mixer from the in-projection output to W_out's bf16 input; ``conv_state`` and ``ssm_state`` are advanced in
  place.  Decode.

The step, per row (1.27B: di 9216 = H 144 x P 64, N 128, conv_dim 9472 = di + 2 N, proj width 18832 = [z | xBC | dt])::

    win    = [conv_state[c, 0], conv_state[c, 1], conv_state[c, 2], xBC[c]]          lags t-3, t-2, t-1, t
    pre    = fma(w3, win3, fma(w2, win2, fma(w1, win1, fma(w0, win0, bias))))       conv_w[:, j] = tap j, fp32
    xbc    = bf16(pre * rcp(1 + ex2(-pre log2 e)))                                   the conv-update kernel's SiLU
    x, B, C = split(xbc, [di, N, N]);  dt = softplus_k(dt_raw[h] + dt_bias[h]);  dA = exp(A[h] dt)
    S      = ssm[h] * dA + (B dt) x[h, p]                                            (P, N) fp32 per head
    y      = sum_n S C + x D[h];  ssm[h] <- S (stored in the state's dtype)
    g      = f32(bf16(y)) * z * sigmoid(z)                                            z = proj[:, :di]
    n      = bf16(g / sqrt(sum(g^2) / di + eps))                                      the unit-weight gated RMSNorm
    xs     = bf16(S(n, theta) * gamma),  S(n, th) = max(n - th, 0) - max(-n - th, 0)  the bipolar soma
    conv_state <- win[1:]                                                             the rolled tail

``softplus_k(v) = where(v <= 20, log(exp(v) + 1), v)`` (the selective-scan kernels' form).

Kernels (two or three launches per step, routed by the decode batch b with a constant table):

* K1 -- conv update + SiLU + bf16 rounding + selective state update (mamba_ssm ``_selective_scan_update_kernel``'s
  lines, TIE_HDIM, dt bias, D, softplus, no z), the x channels' conv tails rolled by their owner program.  Three
  routes, the same per-element operations in the same order:
    * ``gated`` (b == 1): one kernel, grid (P / 16, H, 1), 4 warps; it also forms the gate ``f32(bf16(y)) * z *
      sigmoid(z)`` and stores it fp32;
    * ``split`` (b in {8, 32}): a conv kernel (grid (H + 2, b), 2 warps: each head's x channels with dt / dA, then the
      row's B and C channel sets once) into small fp32 buffers, and a streaming state kernel (grid (P / 16, H, b),
      2 warps, no register cap) that forms dB, updates the state tile, reduces y, rounds it to bf16 and forms the gate
      (fp32);
    * ``plain`` (every other b): one kernel, grid (P / 32, H, b), 4 warps; it stores y in bf16 (the gate is formed
      in K2).
* K2 -- the gated RMSNorm over the whole row in one program per (row, slice) (flash-linear-attention's
  ``layer_norm_fwd_kernel`` lines: BLOCK_N 16384, 8 warps, unit weight), n rounded to bf16, then the soma read-out of
  the program's slice with IEEE single-rounding PTX (sub.rn, NaN-propagating max, mul.rn, cvt.rn.bf16); the row's
  B / C conv tails are rolled by slice 0.  ``from_gated`` (after ``gated`` / ``split``: the fp32 gated input; n goes
  through a bf16 scratch row) or ``select`` (after ``plain``: it gates the bf16 y itself and selects its slice
  from the whole-row registers before the rounding).  Programs per row: 8 for b <= 16, 4 above (slice widths 2048 /
  4096: 5 / 3 programs).
* The pre-norm: flash-linear-attention's ``layer_norm_fwd_kernel1`` lines (IS_RMS_NORM, weight; D = 1536 at its
  cached 2-warp configuration); the fp32 result is written to a scratch row and re-read for the bf16 rounding, so that
  the rounding does not change the layout of the reduction (the bf16 accesses are shaped (BD / 8, 8) to keep the fp32
  row at 4 elements per thread).

Numerics.  The conv is the fast-math instruction sequence of the reference conv-update kernel (FFMA.FTZ chain,
``ex2.approx.ftz``, ``rcp.approx.ftz``), issued as inline PTX; the state update and the norm are the reference kernels'
Triton lines with their reduction layouts (the 128-wide state row inside one warp; one program per norm row); the soma
is the torch chain's IEEE single-rounding arithmetic given n.  The SSM state is loaded, updated in fp32 and stored in
its own dtype: with a bf16 state the compartments are rounded once per step; with an fp32 state nothing is rounded.
The conv tail is fp32 holding bf16 values.  Tests: the per-operator tolerance rule of
:mod:`neuronspark.utils.testing` against :func:`sd_step_core_ref`, for both state dtypes.

Launch constants (per decode batch b, read at call time, so a captured graph bakes the
route of its batch): K1 route ``{1: gated, 8: split, 32: split}``, otherwise ``plain``; K1 rows per program and
warps (16, 4) for b == 1, (32, 4) otherwise; split state kernel (16 rows, 2 warps, no register cap), split conv
kernel 2 warps; K2 programs per row 8 (b <= 16) / 4; pre-norm 2 warps.  Supported decode batches: 1 <= b <= 32.

Contract (anything else raises): CUDA tensors on one device; proj (b, 2 di + 2 N + H) bf16 with unit last stride;
conv_state (b, conv_dim, 3) fp32; ssm_state (b, H, P, N) bf16 or fp32 (the split route needs contiguous state rows);
conv_w (conv_dim, 4) fp32 contiguous; conv_b (conv_dim,), A / dt_bias / D (H,), theta / gamma (di,) fp32 contiguous;
N and P powers of two, P a multiple of the K1 tile; the gated and split routes are built for the 1.27B widths (di 9216,
H 144, N 128, P 64).  References: :func:`sd_prenorm_rows_ref`, :func:`sd_step_core_ref` (pure torch, fp32; the
reference mutates the states in place like the kernel).  The PTX helpers are :mod:`neuronspark.ops.triton._ptx`'s.  The
pre-norm and K2 norm lines follow fla-core 0.5.2 kernels (MIT; see NOTICE).
"""
from __future__ import annotations

from typing import Tuple

import torch
import triton
import triton.language as tl

from neuronspark.ops.triton import require_cuda
from neuronspark.ops.triton._ptx import (
    _fma_rn_ftz,
    _silu_fast_ftz,
    _softplus,
    _sr_bf16_to_f32,
    _sr_f32_to_bf16,
    _sr_mul_rn,
    _sr_release,
)
from neuronspark.ops.triton.sd_scan import softplus_ref
from neuronspark.ops.triton.sd_soma import sd_soma_readout_ref

__all__ = [
    "PRENORM_WARPS", "K1_ROUTE", "K1_TILE_B1", "K1_TILE", "K1_SPLIT_CONV_WARPS", "K1_SPLIT_STATE_TILE",
    "K1_SPLIT_MAXNREG", "K2_PROGRAMS_SMALL", "K2_PROGRAMS_LARGE", "K2_SMALL_MAX_BATCH", "MAX_DECODE_BATCH",
    "ROUTE_WIDTHS", "k1_route", "k1_tile", "k2_programs",
    "SDStepCore", "sd_prenorm_rows", "sd_prenorm_rows_ref", "sd_step_core", "sd_step_core_ref",
]

# ------------------------------------------------------------------------------------------------ launch constants
PRENORM_WARPS = 2                      # fla layer_norm_fwd_kernel1's cached configuration at D = 1536
K1_ROUTE = {1: "gated", 8: "split", 32: "split"}   # every other decode batch: "plain"
K1_TILE_B1 = (16, 4)                   # K1 (rows per program, warps) at b == 1 (the gated route requires it)
K1_TILE = (32, 4)                      # K1 (rows per program, warps) at b >= 2
K1_SPLIT_CONV_WARPS = 2                # split route: conv kernel warps per program
K1_SPLIT_STATE_TILE = (16, 2)          # split route: state kernel (rows per program, warps)
K1_SPLIT_MAXNREG = 0                   # split route: state kernel register cap (0 = none)
K2_PROGRAMS_SMALL = 8                  # K2 programs per row for b <= K2_SMALL_MAX_BATCH
K2_PROGRAMS_LARGE = 4                  # K2 programs per row above
K2_SMALL_MAX_BATCH = 16
MAX_DECODE_BATCH = 32                  # the decode batches the routes cover: 1 .. 32
ROUTE_WIDTHS = (9216, 144, 128, 64)    # (di, H, N, P) the gated and split routes are built for
CONV_WIDTH = 4                         # the conv operators are built for width 4 (a 3-column tail)


def k1_route(b: int) -> str:
    """The K1 route of decode batch ``b``: ``gated`` (b == 1), ``split`` (b in {8, 32}) or ``plain``."""
    return K1_ROUTE.get(int(b), "plain")


def k1_tile(b: int) -> Tuple[int, int]:
    """(rows per program, warps) of the one-kernel K1 routes at decode batch ``b``."""
    return K1_TILE_B1 if int(b) <= 1 else K1_TILE


def k2_programs(b: int) -> int:
    """K2 programs per row at decode batch ``b`` (the slice width is next_pow2(cdiv(di, programs)))."""
    return K2_PROGRAMS_SMALL if int(b) <= K2_SMALL_MAX_BATCH else K2_PROGRAMS_LARGE


# ================================================================================================ conv helper

@triton.jit
def _conv_ch(proj_row, cs_row, CW, CB, ch, s_cs_c, s_cs_l, DI: tl.constexpr):
    """one conv channel set ``ch`` (indices into conv_dim): the window [tail0, tail1, tail2, new] -> SiLU -> bf16 ->
    fp32; returns (value, tail1, tail2, new) (the caller stores the rolled tail)"""
    xn = tl.load(proj_row + DI + ch).to(tl.float32)
    s0 = tl.load(cs_row + ch * s_cs_c)
    s1 = tl.load(cs_row + ch * s_cs_c + s_cs_l)
    s2 = tl.load(cs_row + ch * s_cs_c + 2 * s_cs_l)
    w0 = tl.load(CW + ch * 4)
    w1 = tl.load(CW + ch * 4 + 1)
    w2 = tl.load(CW + ch * 4 + 2)
    w3 = tl.load(CW + ch * 4 + 3)
    acc = tl.load(CB + ch)
    acc = _fma_rn_ftz(w0, s0, acc)
    acc = _fma_rn_ftz(w1, s1, acc)
    acc = _fma_rn_ftz(w2, s2, acc)
    acc = _fma_rn_ftz(w3, xn, acc)
    v = _silu_fast_ftz(acc).to(tl.bfloat16).to(tl.float32)
    return v, s1, s2, xn


# ================================================================================================ K1 routes

@triton.jit
def _sd_k1_kernel(PROJ, CS, SS, CW, CB, A, DTB, DD, Y,
                  s_proj_b, s_cs_b, s_cs_c, s_cs_l, s_ss_b, s_ss_h, s_ss_p, s_ss_n, s_y_b,
                  DI: tl.constexpr, NS: tl.constexpr, P: tl.constexpr, BLOCK_M: tl.constexpr):
    """K1 ``plain`` route: y stored in bf16 (Y (b, di) bf16)."""
    pid_m = tl.program_id(axis=0)
    pid_h = tl.program_id(axis=1)
    pid_b = tl.program_id(axis=2)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, NS)
    proj_row = PROJ + pid_b * s_proj_b
    cs_row = CS + pid_b * s_cs_b

    # conv of this program's x channels (owned: their tails are rolled at the end, after a barrier -- a 1-D tensor of
    # BLOCK_M elements is replicated over the program's threads, so a replica's store must not overtake another
    # replica's load of the same tail)
    cx = pid_h * P + offs_m
    x, xs1, xs2, xn = _conv_ch(proj_row, cs_row, CW, CB, cx, s_cs_c, s_cs_l, DI)
    # conv of the shared B / C channels (every program of the row computes them; K2 rolls their tails)
    B, _b1, _b2, _bn = _conv_ch(proj_row, cs_row, CW, CB, DI + offs_n, s_cs_c, s_cs_l, DI)
    C, _c1, _c2, _cn = _conv_ch(proj_row, cs_row, CW, CB, DI + NS + offs_n, s_cs_c, s_cs_l, DI)

    # mamba_ssm _selective_scan_update_kernel, TIE_HDIM, HAS_DT_BIAS, HAS_D, DT_SOFTPLUS, no z: its lines
    state_ptrs = SS + pid_b * s_ss_b + pid_h * s_ss_h + (offs_m[:, None] * s_ss_p + offs_n[None, :] * s_ss_n)
    state = tl.load(state_ptrs)
    dt = tl.load(PROJ + pid_b * s_proj_b + 2 * DI + 2 * NS + pid_h).to(tl.float32)
    dt += tl.load(DTB + pid_h).to(tl.float32)
    dt = tl.where(dt <= 20.0, _softplus(dt), dt)
    Ah = tl.load(A + pid_h).to(tl.float32)
    dA = tl.exp(Ah * dt)  # scalar, not a matrix
    D = tl.load(DD + pid_h + offs_m * 0).to(tl.float32)
    dB = B * dt  # vector of size (dstate,)
    state = state * dA + dB * x[:, None]
    out = tl.sum(state * C[None, :], axis=1)
    out += x * D
    tl.debug_barrier()
    tl.store(state_ptrs, state)
    tl.store(Y + pid_b * s_y_b + pid_h * P + offs_m, out)
    tl.store(cs_row + cx * s_cs_c, xs1)
    tl.store(cs_row + cx * s_cs_c + s_cs_l, xs2)
    tl.store(cs_row + cx * s_cs_c + 2 * s_cs_l, xn)


@triton.jit
def _sd_k1_gated_kernel(PROJ, CS, SS, CW, CB, A, DTB, DD, Y,
                              s_proj_b, s_cs_b, s_cs_c, s_cs_l, s_ss_b, s_ss_h, s_ss_p, s_ss_n, s_y_b,
                              DI: tl.constexpr, NS: tl.constexpr, P: tl.constexpr, BLOCK_M: tl.constexpr):
    """K1 ``gated`` route: the gate of K2 formed here, Y (b, di) fp32 = f32(bf16(y)) * z * sigmoid(z)."""
    pid_m = tl.program_id(axis=0)
    pid_h = tl.program_id(axis=1)
    pid_b = tl.program_id(axis=2)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, NS)
    proj_row = PROJ + pid_b * s_proj_b
    cs_row = CS + pid_b * s_cs_b

    # conv of this program's x channels (owned: their tails are rolled at the end, after a barrier -- a 1-D tensor of
    # BLOCK_M elements is replicated over the program's threads, so a replica's store must not overtake another
    # replica's load of the same tail)
    cx = pid_h * P + offs_m
    x, xs1, xs2, xn = _conv_ch(proj_row, cs_row, CW, CB, cx, s_cs_c, s_cs_l, DI)
    # conv of the shared B / C channels (every program of the row computes them; K2 rolls their tails)
    B, _b1, _b2, _bn = _conv_ch(proj_row, cs_row, CW, CB, DI + offs_n, s_cs_c, s_cs_l, DI)
    C, _c1, _c2, _cn = _conv_ch(proj_row, cs_row, CW, CB, DI + NS + offs_n, s_cs_c, s_cs_l, DI)

    # mamba_ssm _selective_scan_update_kernel, TIE_HDIM, HAS_DT_BIAS, HAS_D, DT_SOFTPLUS, no z: its lines
    state_ptrs = SS + pid_b * s_ss_b + pid_h * s_ss_h + (offs_m[:, None] * s_ss_p + offs_n[None, :] * s_ss_n)
    state = tl.load(state_ptrs).to(tl.float32)                             # fp32 or bf16 storage
    dt = tl.load(PROJ + pid_b * s_proj_b + 2 * DI + 2 * NS + pid_h).to(tl.float32)
    dt += tl.load(DTB + pid_h).to(tl.float32)
    dt = tl.where(dt <= 20.0, _softplus(dt), dt)
    Ah = tl.load(A + pid_h).to(tl.float32)
    dA = tl.exp(Ah * dt)  # scalar, not a matrix
    D = tl.load(DD + pid_h + offs_m * 0).to(tl.float32)
    dB = B * dt  # vector of size (dstate,)
    state = state * dA + dB * x[:, None]
    out = tl.sum(state * C[None, :], axis=1)
    out += x * D
    tl.debug_barrier()
    tl.store(state_ptrs, state)
    # Keep the bf16 K1 output boundary before the gate.
    out16 = out.to(tl.bfloat16)
    gate_x = _sr_bf16_to_f32(out16)
    gate_z = tl.load(PROJ + pid_b * s_proj_b + pid_h * P + offs_m).to(tl.float32)
    gate_x *= gate_z * tl.sigmoid(gate_z)
    # Y is fp32: the gated value is stored without another rounding.
    tl.store(Y + pid_b * s_y_b + pid_h * P + offs_m, gate_x)
    tl.store(cs_row + cx * s_cs_c, xs1)
    tl.store(cs_row + cx * s_cs_c + s_cs_l, xs2)
    tl.store(cs_row + cx * s_cs_c + 2 * s_cs_l, xn)


@triton.jit
def _sd_k1_conv_kernel(PROJ, CS, CW, CB, A, DTB, XB, BB, CBO, DTO, DAB,
                       s_proj_b, s_cs_b, s_cs_c, s_cs_l, s_x_b,
                       H: tl.constexpr, DI: tl.constexpr, NS: tl.constexpr, P: tl.constexpr):
    """K1 ``split`` route, conv phase.  grid (H + 2, b): programs 0..H-1 = head h's P x channels (+ tails rolled) and
    its dt (softplus) / dA; program H = the row's B conv set, program H + 1 = the row's C conv set (each once per row);
    all stored fp32"""
    pid = tl.program_id(axis=0)
    pid_b = tl.program_id(axis=1)
    proj_row = PROJ + pid_b * s_proj_b
    cs_row = CS + pid_b * s_cs_b
    if pid < H:
        offs_p = tl.arange(0, P)
        cx = pid * P + offs_p
        x, xs1, xs2, xn = _conv_ch(proj_row, cs_row, CW, CB, cx, s_cs_c, s_cs_l, DI)
        dt_raw = tl.load(PROJ + pid_b * s_proj_b + 2 * DI + 2 * NS + pid).to(tl.float32)
        dt_bias = tl.load(DTB + pid).to(tl.float32)
        Ah = tl.load(A + pid).to(tl.float32)
        dt = dt_raw
        dt += dt_bias
        dt = tl.where(dt <= 20.0, _softplus(dt), dt)
        dA = tl.exp(Ah * dt)
        tl.store(XB + pid_b * s_x_b + cx, x)
        tl.store(DTO + pid_b * H + pid, dt)
        tl.store(DAB + pid_b * H + pid, dA)
        tl.store(cs_row + cx * s_cs_c, xs1)
        tl.store(cs_row + cx * s_cs_c + s_cs_l, xs2)
        tl.store(cs_row + cx * s_cs_c + 2 * s_cs_l, xn)
    else:
        offs_n = tl.arange(0, NS)
        ch = DI + (pid - H) * NS + offs_n                                   # B (pid == H) or C (pid == H + 1) channels
        v, _t1, _t2, _tn = _conv_ch(proj_row, cs_row, CW, CB, ch, s_cs_c, s_cs_l, DI)
        if pid == H:
            tl.store(BB + pid_b * NS + offs_n, v)
        else:
            tl.store(CBO + pid_b * NS + offs_n, v)


@triton.jit
def _sd_k1_state_kernel(PROJ, SS, XB, BB, CBO, DTO, DAB, DD, Y,
                        s_proj_b, s_ss_b, s_ss_h, s_ss_p, s_ss_n, s_x_b, s_y_b,
                        H: tl.constexpr, NS: tl.constexpr, P: tl.constexpr, BLOCK_M: tl.constexpr):
    """K1 ``split`` route, state phase.  grid (P / BLOCK_M, H, b): the state update on one BLOCK_M x N tile with the
    conv results read from the buffers (dB = B * dt formed here, the same multiply on the same fp32 operands); the
    state tile is loaded first."""
    pid_m = tl.program_id(axis=0)
    pid_h = tl.program_id(axis=1)
    pid_b = tl.program_id(axis=2)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, NS)
    cx = pid_h * P + offs_m
    state_ptrs = SS + pid_b * s_ss_b + pid_h * s_ss_h + (offs_m[:, None] * s_ss_p + offs_n[None, :] * s_ss_n)
    state = tl.load(state_ptrs).to(tl.float32)                           # fp32 or bf16 storage
    x = tl.load(XB + pid_b * s_x_b + cx)
    B = tl.load(BB + pid_b * NS + offs_n)
    C = tl.load(CBO + pid_b * NS + offs_n)
    dt = tl.load(DTO + pid_b * H + pid_h)
    dA = tl.load(DAB + pid_b * H + pid_h)
    D = tl.load(DD + pid_h + offs_m * 0).to(tl.float32)
    dB = B * dt
    state = state * dA + dB * x[:, None]
    out = tl.sum(state * C[None, :], axis=1)
    out += x * D
    tl.store(state_ptrs, state)
    out16 = out.to(tl.bfloat16)
    gate_x = _sr_bf16_to_f32(out16)
    gate_ptr = PROJ + pid_b * s_proj_b + pid_h * P + offs_m
    gate_z = tl.load(gate_ptr).to(tl.float32)
    gate_x *= gate_z * tl.sigmoid(gate_z)
    tl.store(Y + pid_b * s_y_b + pid_h * P + offs_m, gate_x)


# ================================================================================================ K2 forms

@triton.jit
def _sd_k2_fused_select_kernel(Y, PROJ, CS, TH, GA, XO,
                               s_y_b, s_proj_b, s_cs_b, s_cs_c, s_cs_l, s_xo_b,
                               N, eps,
                               DI: tl.constexpr, NS: tl.constexpr, BLOCK_N: tl.constexpr, SLICE: tl.constexpr):
    """K2 ``select`` form (after the ``plain`` K1): the gate on the bf16 y, the whole-row reduction, the program's
    slice selected from the whole-row registers before the bf16 rounding, the soma read-out of the slice."""
    row = tl.program_id(0)
    part = tl.program_id(1)
    # fla.modules.layernorm_gated.layer_norm_fwd_kernel (HAS_Z, not NORM_BEFORE_GATE, IS_RMS_NORM, no bias, the ones
    # weight): its lines; one program reduces the whole row
    cols = tl.arange(0, BLOCK_N)
    x = tl.load(Y + row * s_y_b + cols, mask=cols < N, other=0.).to(tl.float32)
    z = tl.load(PROJ + row * s_proj_b + cols, mask=cols < N).to(tl.float32)
    x *= z * tl.sigmoid(z)
    xbar = tl.where(cols < N, x, 0.)
    var = tl.sum(xbar * xbar, axis=0) / N
    rstd = 1 / tl.sqrt(var + eps)
    mask = cols < N
    x_hat = x * rstd
    y = x_hat  # * w, w = ones (exact)

    # Each contiguous SLICE is a group of the full-row registers; the group id changes only the register group of a
    # thread.  The slice is selected before the summation so that unused / padded groups contribute exact zeros.
    sc = part * SLICE + tl.arange(0, SLICE)
    sm = sc < N
    groups = tl.reshape(y, (BLOCK_N // SLICE, SLICE))
    group_id = tl.arange(0, BLOCK_N // SLICE)
    selected = tl.where(group_id[:, None] == part, groups, 0.)
    y_slice = tl.sum(selected, axis=0)
    norm_bf16 = y_slice.to(tl.bfloat16)

    th = tl.load(TH + sc, mask=sm, other=0.0)
    ga = tl.load(GA + sc, mask=sm, other=0.0)
    v = _sr_bf16_to_f32(norm_bf16)
    s, _a, _b = _sr_release(v, th)
    tl.store(XO + row * s_xo_b + sc, _sr_f32_to_bf16(_sr_mul_rn(s, ga)), mask=sm)

    # the B / C conv tails of the row (read by every K1 program of the row; rolled once here)
    if part == 0:
        cbc = DI + tl.arange(0, 2 * NS)
        p = CS + row * s_cs_b + cbc * s_cs_c
        t1 = tl.load(p + s_cs_l)
        t2 = tl.load(p + 2 * s_cs_l)
        xn = tl.load(PROJ + row * s_proj_b + DI + cbc).to(tl.float32)
        tl.debug_barrier()                  # every replica has loaded the tail before any replica stores into it
        tl.store(p, t1)
        tl.store(p + s_cs_l, t2)
        tl.store(p + 2 * s_cs_l, xn)


@triton.jit
def _sd_k2_from_gated_kernel(Y, PROJ, CS, TH, GA, XO, SCR,
                             s_y_b, s_proj_b, s_cs_b, s_cs_c, s_cs_l, s_xo_b, s_scr_b,
                             N, eps,
                             DI: tl.constexpr, NS: tl.constexpr, BLOCK_N: tl.constexpr, SLICE: tl.constexpr):
    """K2 ``from_gated`` form (after the ``gated`` / ``split`` K1): the fp32 gated input, the whole-row reduction,
    n through a bf16 scratch row, the soma read-out of the program's slice."""
    row = tl.program_id(0)
    part = tl.program_id(1)
    # fla.modules.layernorm_gated.layer_norm_fwd_kernel (HAS_Z, not NORM_BEFORE_GATE, IS_RMS_NORM, no bias, the ones
    # weight): its lines; one program reduces the whole row
    cols = tl.arange(0, BLOCK_N)
    x = tl.load(Y + row * s_y_b + cols, mask=cols < N, other=0.).to(tl.float32)
    # Y holds the fp32 gate result from K1; the reduction and the soma are unchanged.
    xbar = tl.where(cols < N, x, 0.)
    var = tl.sum(xbar * xbar, axis=0) / N
    rstd = 1 / tl.sqrt(var + eps)
    mask = cols < N
    x_hat = x * rstd
    y = x_hat  # * w, w = ones (exact)
    smask = mask & (cols >= part * SLICE) & (cols < (part + 1) * SLICE)
    tl.store(SCR + row * s_scr_b + cols, y.to(tl.bfloat16), mask=smask)
    tl.debug_barrier()
    # soma read-out of this program's slice (the prefill soma read-out kernel's ops)
    sc = part * SLICE + tl.arange(0, SLICE)
    sm = sc < N
    th = tl.load(TH + sc, mask=sm, other=0.0)
    ga = tl.load(GA + sc, mask=sm, other=0.0)
    v = _sr_bf16_to_f32(tl.load(SCR + row * s_scr_b + sc, mask=sm, other=0.0))
    s, _a, _b = _sr_release(v, th)
    tl.store(XO + row * s_xo_b + sc, _sr_f32_to_bf16(_sr_mul_rn(s, ga)), mask=sm)
    # the B / C conv tails of the row (read by every K1 program of the row; rolled once here)
    if part == 0:
        cbc = DI + tl.arange(0, 2 * NS)
        p = CS + row * s_cs_b + cbc * s_cs_c
        t1 = tl.load(p + s_cs_l)
        t2 = tl.load(p + 2 * s_cs_l)
        xn = tl.load(PROJ + row * s_proj_b + DI + cbc).to(tl.float32)
        tl.debug_barrier()                  # every replica has loaded the tail before any replica stores into it
        tl.store(p, t1)
        tl.store(p + s_cs_l, t2)
        tl.store(p + 2 * s_cs_l, xn)


# ================================================================================================ pre-norm

@triton.jit
def _sd_prenorm_kernel(H, W, YO, SCR, eps, s_h, s_y, s_scr, D: tl.constexpr, BD: tl.constexpr):
    # Layout note.  Triton's coalescing gives every memory op of one shape the largest vector width among them, so a
    # bf16 load / store of shape (BD,) would put the fp32 row loads -- and the reduction fed by them -- at 8 elements
    # per thread, while fla's kernel (all fp32) reduces at 4 per thread: another summation order.  The bf16 accesses are
    # therefore made with shape (BD // 8, 8); the (BD,) group holds the fp32 accesses only, as in fla's kernel.
    # Packed-fp32 note: with y rounded to bf16 in the same expression chain, LLVM pairs the y products for
    # cvt.rn.bf16x2 and its SLP vectorizer then forms the whole row in f32x2 registers, the sum of squares included
    # (two interleaved accumulators per thread) -- fla's kernel1 runs one scalar fma chain per thread.  So y is written
    # in fp32 exactly as fla's kernel writes it (its (BD,) store, scratch row SCR), and the bf16 rounding reads it back
    # after a barrier in the (BD // 8, 8) shape: the code up to the fp32 store is fla's.
    i_t = tl.program_id(0)
    o_d = tl.arange(0, BD)
    m_d = o_d < D
    o2 = tl.arange(0, BD // 8)[:, None] * 8 + tl.arange(0, 8)[None, :]
    m2 = o2 < D
    src = H + i_t * s_h
    # fla.modules.layernorm.layer_norm_fwd_kernel1 (IS_RMS_NORM, HAS_WEIGHT, no bias / residual): its lines
    b_x = tl.load(src + o_d, mask=m_d, other=0.0).to(tl.float32)
    b_xbar = tl.where(m_d, b_x, 0.0)
    b_var = tl.sum(b_xbar * b_xbar, axis=0) / D
    b_rstd = 1 / tl.sqrt(b_var + eps)
    b_w = tl.load(W + o_d, mask=m_d).to(tl.float32)
    b_x_hat = b_x * b_rstd
    b_y = b_x_hat * b_w
    tl.store(SCR + i_t * s_scr + o_d, b_y, mask=m_d)
    tl.debug_barrier()
    b_yr = tl.load(SCR + i_t * s_scr + o2, mask=m2, other=0.0)
    tl.store(YO + i_t * s_y + o2, b_yr.to(tl.bfloat16), mask=m2)


# ================================================================================================ launchers

def _launch_k1_plain(proj, conv_state, ssm_state, conv_w, conv_b, A, dt_bias, D, di, H, N, P, bm, nw):
    """y (b, di) bf16 (the ``plain`` route)."""
    b = int(proj.shape[0])
    y = torch.empty((b, di), dtype=torch.bfloat16, device=proj.device)
    _sd_k1_kernel[(P // bm, H, b)](
        proj, conv_state, ssm_state, conv_w, conv_b, A, dt_bias, D, y,
        proj.stride(0), conv_state.stride(0), conv_state.stride(1), conv_state.stride(2),
        ssm_state.stride(0), ssm_state.stride(1), ssm_state.stride(2), ssm_state.stride(3), y.stride(0),
        DI=di, NS=N, P=P, BLOCK_M=bm, num_warps=nw)
    return y


def _launch_k1_gated(proj, conv_state, ssm_state, conv_w, conv_b, A, dt_bias, D, di, H, N, P, bm, nw):
    """gated y (b, di) fp32 (the ``gated`` route; the b == 1 layout only)."""
    b = int(proj.shape[0])
    if (b, bm, nw, di, H, N, P) != (1, K1_TILE_B1[0], K1_TILE_B1[1]) + ROUTE_WIDTHS:
        raise RuntimeError(f"neuronspark.sd_step_core: the gated K1 route is built for b 1, tile {K1_TILE_B1} and "
                           f"widths {ROUTE_WIDTHS}; got b {b} tile ({bm}, {nw}) widths {(di, H, N, P)}")
    y = torch.empty((b, di), dtype=torch.float32, device=proj.device)
    _sd_k1_gated_kernel[(P // bm, H, b)](
        proj, conv_state, ssm_state, conv_w, conv_b, A, dt_bias, D, y,
        proj.stride(0), conv_state.stride(0), conv_state.stride(1), conv_state.stride(2),
        ssm_state.stride(0), ssm_state.stride(1), ssm_state.stride(2), ssm_state.stride(3), y.stride(0),
        DI=di, NS=N, P=P, BLOCK_M=bm, num_warps=nw)
    return y


def _launch_k1_split(proj, conv_state, ssm_state, conv_w, conv_b, A, dt_bias, D, di, H, N, P):
    """gated y (b, di) fp32 (the ``split`` route: conv kernel + streaming state kernel)."""
    b = int(proj.shape[0])
    if b not in (8, 32) or (di, H, N, P) != ROUTE_WIDTHS:
        raise RuntimeError(f"neuronspark.sd_step_core: the split K1 route is built for b 8 / 32 and widths "
                           f"{ROUTE_WIDTHS}; got b {b} widths {(di, H, N, P)}")
    if int(ssm_state.stride(3)) != 1:
        raise RuntimeError("neuronspark.sd_step_core: the split K1 route needs contiguous SSM state rows")
    state_bm, state_warps = K1_SPLIT_STATE_TILE
    if P % state_bm:
        raise RuntimeError(f"neuronspark.sd_step_core: P {P} is not a multiple of the split state tile {state_bm}")
    dev = proj.device
    xb = torch.empty((b, di), dtype=torch.float32, device=dev)
    bb = torch.empty((b, N), dtype=torch.float32, device=dev)
    cbo = torch.empty((b, N), dtype=torch.float32, device=dev)
    dto = torch.empty((b, H), dtype=torch.float32, device=dev)
    dab = torch.empty((b, H), dtype=torch.float32, device=dev)
    y = torch.empty((b, di), dtype=torch.float32, device=dev)
    _sd_k1_conv_kernel[(H + 2, b)](
        proj, conv_state, conv_w, conv_b, A, dt_bias, xb, bb, cbo, dto, dab,
        proj.stride(0), conv_state.stride(0), conv_state.stride(1), conv_state.stride(2), xb.stride(0),
        H=H, DI=di, NS=N, P=P, num_warps=K1_SPLIT_CONV_WARPS)
    options = {} if K1_SPLIT_MAXNREG <= 0 else {"maxnreg": K1_SPLIT_MAXNREG}
    _sd_k1_state_kernel[(P // state_bm, H, b)](
        proj, ssm_state, xb, bb, cbo, dto, dab, D, y,
        proj.stride(0), ssm_state.stride(0), ssm_state.stride(1), ssm_state.stride(2), ssm_state.stride(3),
        xb.stride(0), y.stride(0),
        H=H, NS=N, P=P, BLOCK_M=state_bm, num_warps=state_warps, **options)
    return y


def _k2_geometry(b: int, di: int) -> Tuple[int, int, int, int]:
    """(BLOCK_N, SLICE, programs per row, warps) of K2 at decode batch ``b``."""
    block_n = triton.next_power_of_2(di)
    slice_ = triton.next_power_of_2(triton.cdiv(di, k2_programs(b)))
    split = triton.cdiv(di, slice_)
    return block_n, slice_, split, min(max(block_n // 256, 1), 8)


def _launch_k2_select(y, proj, conv_state, theta, gamma, di, N, eps):
    """xs (b, di) bf16 from the bf16 y of the ``plain`` K1."""
    b = int(proj.shape[0])
    block_n, slice_, split, nw = _k2_geometry(b, di)
    xo = torch.empty((b, di), dtype=torch.bfloat16, device=proj.device)
    _sd_k2_fused_select_kernel[(b, split)](
        y, proj, conv_state, theta, gamma, xo,
        y.stride(0), proj.stride(0), conv_state.stride(0), conv_state.stride(1), conv_state.stride(2),
        xo.stride(0), di, float(eps),
        DI=di, NS=N, BLOCK_N=block_n, SLICE=slice_, num_warps=nw)
    return xo


def _launch_k2_from_gated(y, proj, conv_state, theta, gamma, di, N, eps):
    """xs (b, di) bf16 from the fp32 gated y of the ``gated`` / ``split`` K1."""
    b = int(proj.shape[0])
    if y.dtype != torch.float32 or tuple(y.shape) != (b, di) or y.stride(-1) != 1:
        raise RuntimeError("neuronspark.sd_step_core: the gated K1 output must be fp32 (b, di) with unit last stride")
    block_n, slice_, split, nw = _k2_geometry(b, di)
    xo = torch.empty((b, di), dtype=torch.bfloat16, device=proj.device)
    scr = torch.empty((b, di), dtype=torch.bfloat16, device=proj.device)
    _sd_k2_from_gated_kernel[(b, split)](
        y, proj, conv_state, theta, gamma, xo, scr,
        y.stride(0), proj.stride(0), conv_state.stride(0), conv_state.stride(1), conv_state.stride(2),
        xo.stride(0), scr.stride(0), di, float(eps),
        DI=di, NS=N, BLOCK_N=block_n, SLICE=slice_, num_warps=nw)
    return xo


def _widths(proj, conv_w, A, theta):
    """(di, conv_dim, H, N, P) from the operand shapes."""
    di = int(theta.shape[0])
    conv_dim = int(conv_w.shape[0])
    H = int(A.shape[0])
    N = (conv_dim - di) // 2
    P = di // H if H > 0 else 0
    return di, conv_dim, H, N, P


def _sd_step_core_fwd(proj, conv_state, ssm_state, conv_w, conv_b, A, dt_bias, D, theta, gamma, eps):
    """K1 (routed by b) + K2 -> xs (b, di) bf16; conv_state / ssm_state advanced in place (operands checked by the
    caller)."""
    b = int(proj.shape[0])
    di, conv_dim, H, N, P = _widths(proj, conv_w, A, theta)
    route = k1_route(b)
    bm, nw = k1_tile(b)
    with torch.cuda.device(proj.device.index):
        if route == "gated":
            y = _launch_k1_gated(proj, conv_state, ssm_state, conv_w, conv_b, A, dt_bias, D, di, H, N, P, bm, nw)
            return _launch_k2_from_gated(y, proj, conv_state, theta, gamma, di, N, eps)
        if route == "split":
            y = _launch_k1_split(proj, conv_state, ssm_state, conv_w, conv_b, A, dt_bias, D, di, H, N, P)
            return _launch_k2_from_gated(y, proj, conv_state, theta, gamma, di, N, eps)
        y = _launch_k1_plain(proj, conv_state, ssm_state, conv_w, conv_b, A, dt_bias, D, di, H, N, P, bm, nw)
        return _launch_k2_select(y, proj, conv_state, theta, gamma, di, N, eps)


def _sd_prenorm_rows_fwd(x, gain, eps):
    """hn (M, D) bf16 contiguous (operands checked by the caller)."""
    M, D = int(x.shape[0]), int(x.shape[1])
    bd = triton.next_power_of_2(D)
    yo = torch.empty((M, D), dtype=torch.bfloat16, device=x.device)
    scr = torch.empty((M, D), dtype=torch.float32, device=x.device)     # the fp32 y row (packed-fp32 note)
    with torch.cuda.device(x.device.index):
        _sd_prenorm_kernel[(M,)](x, gain, yo, scr, float(eps), x.stride(0), yo.stride(0), scr.stride(0),
                                 D=D, BD=bd, num_warps=PRENORM_WARPS)
    return yo


# ================================================================================================ custom ops

@torch.library.custom_op("neuronspark::sd_prenorm_rows", mutates_args=(),
                         schema="(Tensor x, Tensor gain, float eps) -> Tensor")
def _sd_prenorm_rows_op(x, gain, eps):
    """bf16 RMSNorm rows of an fp32 x (contiguous output)"""
    return _sd_prenorm_rows_fwd(x, gain, eps)


@_sd_prenorm_rows_op.register_fake
def _sd_prenorm_rows_fake(x, gain, eps):
    return torch.empty(tuple(x.shape), dtype=torch.bfloat16, device=x.device)


@torch.library.custom_op(
    "neuronspark::sd_step_core", mutates_args=("conv_state", "ssm_state"),
    schema="(Tensor proj, Tensor(a!) conv_state, Tensor(b!) ssm_state, Tensor conv_w, Tensor conv_b, Tensor A, "
           "Tensor dt_bias, Tensor D, Tensor theta, Tensor gamma, float eps) -> Tensor")
def _sd_step_core_op(proj, conv_state, ssm_state, conv_w, conv_b, A, dt_bias, D, theta, gamma, eps):
    """W_out's bf16 input (b, di) from the in-projection output; conv / SSM state advanced in place"""
    return _sd_step_core_fwd(proj, conv_state, ssm_state, conv_w, conv_b, A, dt_bias, D, theta, gamma, eps)


@_sd_step_core_op.register_fake
def _sd_step_core_fake(proj, conv_state, ssm_state, conv_w, conv_b, A, dt_bias, D, theta, gamma, eps):
    return torch.empty((proj.shape[0], theta.shape[0]), dtype=torch.bfloat16, device=proj.device)


# the registered operator ``neuronspark::sd_step_core`` (a torch.library CustomOpDef; it declares the two state
# mutations): ``SDStepCore(proj, conv_state, ssm_state, ...)`` runs the routed kernels without the operand checks of
# the public function below.
SDStepCore = _sd_step_core_op


# ================================================================================================ public functions

def _check(t, name: str, dtype, dim: int, last_contig: bool = True) -> None:
    if not isinstance(t, torch.Tensor) or not t.is_cuda or t.dtype != dtype or t.dim() != dim:
        raise RuntimeError(f"neuronspark.sd_step: {name} must be a CUDA {dtype} {dim}-D tensor, got "
                           f"{type(t).__name__} {getattr(t, 'dtype', None)} {tuple(getattr(t, 'shape', ()))}")
    if last_contig and t.stride(-1) != 1:
        raise RuntimeError(f"neuronspark.sd_step: {name} must have unit stride on its last axis, got "
                           f"{tuple(t.stride())}")


def _no_grad_inputs(where: str, *tensors) -> None:
    if torch.is_grad_enabled() and any(t.requires_grad for t in tensors):
        raise RuntimeError(f"neuronspark.{where}: an inference operator (no autograd): call it under torch.no_grad()")


def sd_prenorm_rows(x: torch.Tensor, gain: torch.Tensor, eps: float) -> torch.Tensor:
    """``hn = bf16((x * rstd) * gain)`` per row (see the module docstring): x (M, D) fp32 CUDA with unit last stride,
    gain (D,) fp32 contiguous on x's device -> hn (M, D) bf16 contiguous.  No autograd."""
    require_cuda(x, "sd_prenorm_rows")
    _check(x, "x", torch.float32, 2)
    _check(gain, "gain", torch.float32, 1)
    M, D = int(x.shape[0]), int(x.shape[1])
    if gain.device != x.device or int(gain.shape[0]) != D or not gain.is_contiguous():
        raise RuntimeError(f"neuronspark.sd_prenorm_rows: gain {tuple(gain.shape)} on {gain.device} vs width {D} on "
                           f"{x.device} (a contiguous (D,) gain on x's device needed)")
    if M < 1 or D < 8:
        raise RuntimeError(f"neuronspark.sd_prenorm_rows: rows M >= 1 and width D >= 8 expected, got ({M}, {D})")
    if (M - 1) * max(int(x.stride(0)), D) + D >= 2 ** 31:
        raise RuntimeError(f"neuronspark.sd_prenorm_rows: row offsets of ({M}, {D}) exceed 2^31 elements")
    _no_grad_inputs("sd_prenorm_rows", x, gain)
    return torch.ops.neuronspark.sd_prenorm_rows(x, gain, float(eps))


def sd_step_core(proj: torch.Tensor, conv_state: torch.Tensor, ssm_state: torch.Tensor, conv_w: torch.Tensor,
                 conv_b: torch.Tensor, A: torch.Tensor, dt_bias: torch.Tensor, D: torch.Tensor, theta: torch.Tensor,
                 gamma: torch.Tensor, eps: float) -> torch.Tensor:
    """One decode step of an SD mixer from the in-projection output (see the module docstring): returns xs (b, di)
    bf16 (W_out's input) and advances ``conv_state`` (b, conv_dim, 3) fp32 and ``ssm_state`` (b, H, P, N) bf16|fp32
    in place (the state keeps its dtype).  No autograd."""
    require_cuda(proj, "sd_step_core")
    _check(proj, "proj", torch.bfloat16, 2)
    b = int(proj.shape[0])
    if not 1 <= b <= MAX_DECODE_BATCH:
        raise RuntimeError(f"neuronspark.sd_step_core: decode batch {b} outside 1..{MAX_DECODE_BATCH} (the batches the "
                           "kernel routes cover)")
    _check(conv_w, "conv_w", torch.float32, 2)
    _check(A, "A", torch.float32, 1)
    _check(theta, "theta", torch.float32, 1)
    di, conv_dim, H, N, P = _widths(proj, conv_w, A, theta)
    if H < 1 or di != H * P or conv_dim != di + 2 * N or N < 1 or int(proj.shape[1]) != 2 * di + 2 * N + H:
        raise RuntimeError(f"neuronspark.sd_step_core: widths di {di} conv_dim {conv_dim} H {H} N {N} P {P} vs "
                           f"proj {tuple(proj.shape)} (proj = [z di | xBC conv_dim | dt H], conv_dim = di + 2 N)")
    _check(conv_state, "conv_state", torch.float32, 3, last_contig=False)
    if not isinstance(ssm_state, torch.Tensor) or ssm_state.dtype not in (torch.bfloat16, torch.float32):
        raise RuntimeError(f"neuronspark.sd_step_core: ssm_state must be a bf16 or fp32 tensor, got "
                           f"{getattr(ssm_state, 'dtype', type(ssm_state).__name__)}")
    _check(ssm_state, "ssm_state", ssm_state.dtype, 4, last_contig=False)
    if tuple(conv_state.shape) != (b, conv_dim, CONV_WIDTH - 1) or tuple(ssm_state.shape) != (b, H, P, N):
        raise RuntimeError(f"neuronspark.sd_step_core: state shapes {tuple(conv_state.shape)} {tuple(ssm_state.shape)}"
                           f" (expected ({b}, {conv_dim}, {CONV_WIDTH - 1}) and ({b}, {H}, {P}, {N}))")
    if tuple(conv_w.shape) != (conv_dim, CONV_WIDTH) or not conv_w.is_contiguous():
        raise RuntimeError(f"neuronspark.sd_step_core: conv_w {tuple(conv_w.shape)} (contiguous ({conv_dim}, "
                           f"{CONV_WIDTH}) needed)")
    for name, t, n_ in (("conv_b", conv_b, conv_dim), ("A", A, H), ("dt_bias", dt_bias, H), ("D", D, H),
                        ("theta", theta, di), ("gamma", gamma, di)):
        _check(t, name, torch.float32, 1)
        if int(t.shape[0]) != n_ or not t.is_contiguous():
            raise RuntimeError(f"neuronspark.sd_step_core: {name} {tuple(t.shape)} (contiguous ({n_},) needed)")
    for name, t in (("conv_state", conv_state), ("ssm_state", ssm_state), ("conv_w", conv_w), ("conv_b", conv_b),
                    ("A", A), ("dt_bias", dt_bias), ("D", D), ("theta", theta), ("gamma", gamma)):
        if t.device != proj.device:
            raise RuntimeError(f"neuronspark.sd_step_core: {name} is on {t.device}, proj on {proj.device}")
    if N & (N - 1) or P & (P - 1):
        raise RuntimeError(f"neuronspark.sd_step_core: N {N} and P {P} must be powers of two")
    bm, _nw = k1_tile(b)
    if P % bm:
        raise RuntimeError(f"neuronspark.sd_step_core: P {P} not a multiple of the K1 tile {bm}")
    _no_grad_inputs("sd_step_core", proj, conv_state, ssm_state, conv_w, conv_b, A, dt_bias, D, theta, gamma)
    return torch.ops.neuronspark.sd_step_core(proj, conv_state, ssm_state, conv_w, conv_b, A, dt_bias, D, theta,
                                              gamma, float(eps))


# ================================================================================================ references
# Pure torch, fp32 (any device).  The reductions run in torch's order and the fast-math SiLU is evaluated in IEEE
# arithmetic: the per-operator rule class of the kernels.

_NEG_LOG2E = -1.4426950408889634       # rounds to the kernel's fp32 constant 0xBFB8AA3B


def sd_prenorm_rows_ref(x: torch.Tensor, gain: torch.Tensor, eps: float) -> torch.Tensor:
    """``bf16((x * rstd) * gain)``, ``rstd = 1 / sqrt(sum(x^2) / D + eps)`` (fla ``rms_norm`` + the bf16 cast of the
    in-projection), fp32; contiguous."""
    xf = x.float()
    var = (xf * xf).sum(dim=-1, keepdim=True) / int(xf.shape[-1])
    rstd = 1.0 / torch.sqrt(var + float(eps))
    return ((xf * rstd) * gain.float()).to(torch.bfloat16).contiguous()


def _silu_fast_ref(v: torch.Tensor) -> torch.Tensor:
    """The conv-update kernel's SiLU sequence in IEEE fp32: ``v * (1 / (1 + exp2(v * -log2 e)))``."""
    t = torch.exp2(v * torch.tensor(_NEG_LOG2E, dtype=torch.float32, device=v.device))
    return v * (1.0 / (t + 1.0))


def sd_step_core_ref(proj: torch.Tensor, conv_state: torch.Tensor, ssm_state: torch.Tensor, conv_w: torch.Tensor,
                     conv_b: torch.Tensor, A: torch.Tensor, dt_bias: torch.Tensor, D: torch.Tensor,
                     theta: torch.Tensor, gamma: torch.Tensor, eps: float) -> torch.Tensor:
    """The decode step in pure torch (the chain the kernels replace: ``causal_conv1d_update`` on the bf16 conv input
    with the fp32 tail, the bf16 rounding of its output, ``selective_state_update`` with dt bias / D / softplus, fla's
    gated RMSNorm with unit weight on the bf16 y, the soma read-out): returns xs (b, di) bf16; ``conv_state`` and
    ``ssm_state`` are advanced in place (the state stored in its own dtype), as the kernel does."""
    b = int(proj.shape[0])
    di, conv_dim, H, N, P = _widths(proj, conv_w, A, theta)
    z = proj[:, :di].float()
    xBC = proj[:, di:di + conv_dim].float()
    dt_raw = proj[:, di + conv_dim:di + conv_dim + H].float()
    win = torch.cat([conv_state.float(), xBC[:, :, None]], dim=-1)                 # (b, conv_dim, 4): lags t-3 .. t
    w = conv_w.float()
    pre = conv_b.float()[None, :].expand(b, conv_dim)
    for j in range(CONV_WIDTH):
        pre = pre + w[None, :, j] * win[:, :, j]
    xbc = _silu_fast_ref(pre).to(torch.bfloat16).float()
    x = xbc[:, :di].reshape(b, H, P)
    Bv = xbc[:, di:di + N]
    Cv = xbc[:, di + N:di + 2 * N]
    dt = softplus_ref(dt_raw + dt_bias.float())                                    # (b, H)
    dA = torch.exp(A.float()[None, :] * dt)
    dB = Bv[:, None, :] * dt[:, :, None]                                           # (b, H, N)
    S = ssm_state.float() * dA[:, :, None, None] + dB[:, :, None, :] * x[:, :, :, None]
    y = (S * Cv[:, None, None, :]).sum(dim=-1) + x * D.float()[None, :, None]      # (b, H, P)
    ssm_state.copy_(S)
    conv_state.copy_(win[:, :, 1:])
    g = y.reshape(b, di).to(torch.bfloat16).float() * (z * torch.sigmoid(z))
    var = (g * g).sum(dim=-1, keepdim=True) / di
    rstd = 1.0 / torch.sqrt(var + float(eps))
    n = (g * rstd).to(torch.bfloat16)
    return sd_soma_readout_ref(n, theta, gamma)
