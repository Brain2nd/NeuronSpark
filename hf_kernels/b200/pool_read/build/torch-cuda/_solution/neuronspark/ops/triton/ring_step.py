"""Decode-step ring write of the hippocampal layer's pool attention (T = 1): everything between the step's projections
and the ring read, in one kernel.

``ring_write_step`` takes the step's projections of the normed layer input (qkv, the value-residual gate, the two GLU
maps, the decay logit), the write-gate spikes and the carried state, and per row b and head h

    qkv   = bf16(conv step over the carried three taps)          the fp32 FMA chain of the depthwise conv; taps rolled
    q, k, v = silu(qkv)                                          fp32 ``x / (1 + expf(-x))``, one rounding to bf16
    q, k  = rope(q), rope(k)                                     interleaved pairs at the row's position ``pos``, bf16
                                                                 arithmetic (every product and sum rounded to bf16)
    k_hat = k / max(|k|, 1e-12)                                  fp32, rounded to bf16 into ``pool_k``'s slot
    q_out = (q / max(|q|, 1e-12)) qgain_h                        fp32, rounded to bf16 (the read's query rounding)
    lam   = lam0 + tanh(f(gl_h))
    v'    = bf16(f(v) (1 + lam))                                 hippocampal layer 0 (no carry)
          = bf16(f(v) + lam f(carry))                            later layers
    v'    = bf16(f(v') + f(bf16(bf16(silu(f(ga)) f(gb)) @ bf16(U)^T)))      last head only: the GLU tail
    gv    = bf16(bf16(sigmoid(g + wbias_h - sparsity)) v')       into ``pool_gv``'s slot
    a     = exp(-f(rate) softplus(f(dlog))) base_decay;   la = mean_c log(max(a, 1e-6))
    cum   = run_h + la gains_h                                   into ``pool_cum``'s slot and over ``pool_run``

with the slot ``ring_ptr % W``, f(.) the exact widening to fp32 and every fp32 operation of the chain rounded once
(no FMA contraction outside the conv).  Returns the gained unit queries (B, H, dk) bf16 and the values after the
SiLU (B, D_value) bf16 (hippocampal layer 0 hands them to the later layers as their carry).

These are the operations of the torch expression chain of the step (``ring_write_step_ref``) at its rounding points;
the row sums of the two norms, of the GLU product and of the clock's mean are formed in another order.  Tests: the
per-operator tolerance rule of :mod:`neuronspark.utils.testing` against the reference.
"""
from __future__ import annotations; from ....neuronspark._namespace import NAMESPACE as _KERNEL_OP_NAMESPACE, OPS as _KERNEL_OPS

from typing import Optional, Tuple

import torch
import torch.nn.functional as F
import triton
import triton.language as tl
from triton.language.extra import libdevice

from ....neuronspark.modules.rope import apply_rope_interleaved_ref
from ....neuronspark.ops.linear import inference_weight
from ....neuronspark.ops.triton import require_cuda
from ....neuronspark.ops.triton._ptx import _add_rn, _div_rn, _mul_rn, _sub_rn





from ....neuronspark.ops.triton.short_conv import short_conv_step_ref

__all__ = ["RING_STEP_WARPS", "DECAY_FLOOR", "ring_write_step", "ring_write_step_ref"]

# ================================================================================================ fixed constants
#: one program per (row, head), warps.  The conv taps, the conv weight and the GLU map are not in the cache when the
#: step reaches them: the kernel's time is the number of loads a thread issues one after the other, so the program's
#: tiles are spread over a block of one thread per channel of a 256-wide head (the step at batch 1 / 8 / 32 on the
#: B200: 8 warps below both 4 and 16)
RING_STEP_WARPS = 8
#: floor of the decay inside the clock's log (the kernel's literal 1e-6; its softplus takes torch's threshold, 20)
DECAY_FLOOR = 1e-6


# ================================================================================================ kernel
@triton.jit
def _bf16(x):
    """Round an fp32 value to the nearest bf16 (ties to even) and widen it back through the bit pattern (finite
    values)."""
    u = x.to(tl.int32, bitcast=True)
    lsb = (u >> 16) & 1
    u = u + 0x7FFF + lsb
    u = u & -65536                     # 0xFFFF0000 as a signed int32
    return u.to(tl.float32, bitcast=True)


@triton.jit
def _conv_silu(X, HIST, WT, xrow, hrow, s_h_c, c, N: tl.constexpr):
    """The N channels ``c`` (consecutive) of one row: (silu(bf16(conv step)) as fp32 values that hold bf16, the
    taps after the roll: three (N,) fp32 tensors).  Nothing is stored here."""
    x = tl.load(X + xrow + c).to(tl.float32)
    hp = HIST + hrow + c.to(tl.int64) * s_h_c
    h0 = _bf16(tl.load(hp))
    h1 = _bf16(tl.load(hp + 1))
    h2 = _bf16(tl.load(hp + 2))
    # the channel's four taps as one row of a (N, 4) tile: one load per thread
    w4 = _bf16(tl.load(WT + (c * 4)[:, None] + tl.arange(0, 4)[None, :]))
    w_e, w_o = tl.split(tl.reshape(w4, [N, 2, 2]))                 # taps (0, 2) and (1, 3)
    w0, w2 = tl.split(w_e)
    w1, w3 = tl.split(w_o)
    acc = tl.zeros_like(x)
    acc = tl.math.fma(w0, h0, acc)
    acc = tl.math.fma(w1, h1, acc)
    acc = tl.math.fma(w2, h2, acc)
    acc = tl.math.fma(w3, x, acc)
    y = _bf16(acc)
    one = tl.full(y.shape, 1.0, tl.float32)
    return _bf16(_div_rn(y, _add_rn(one, libdevice.exp(-y)))), h1, h2, x


@triton.jit
def _roll(HIST, hrow, s_h_c, c, h1, h2, x):
    hp = HIST + hrow + c.to(tl.int64) * s_h_c
    tl.store(hp, h1)
    tl.store(hp + 1, h2)
    tl.store(hp + 2, x)


@triton.jit
def _ring_write_step_kernel(X, HIST, WT, COS, SIN, POS, PTR, G, GL, LAM0, CARRY, GA, GB, U, DLOG, RATE, GAINS, QGAIN,
                            WBIAS, PK, PGV, PCUM, RUN, QOUT, VOUT,
                            s_x_b, s_h_b, s_h_c, s_cs, s_g_b, s_gl_b, s_c_b, s_ga_b, s_gb_b, s_d_b, s_u,
                            H, W, DKEY, GR, base_decay, sparsity,
                            DK: tl.constexpr, HALF: tl.constexpr, DR: tl.constexpr, GRB: tl.constexpr,
                            HAS_CARRY: tl.constexpr, HAS_SPARSITY: tl.constexpr):
    """Every load of the program comes before the barrier, every store of state after it: a tile narrower than the
    block (RING_STEP_WARPS warps) is held by more than one thread, and a thread must not read state another has
    already advanced."""
    b = tl.program_id(0)
    h = tl.program_id(1)
    b64 = b.to(tl.int64)
    xrow = b64 * s_x_b
    hrow = b64 * s_h_b
    bh = b64 * H + h
    ring = bh * W + tl.load(PTR) % W                                  # this step's slot of head h's rings
    ii = tl.arange(0, HALF)
    jj = tl.arange(0, DK)
    zero_h = tl.zeros([HALF], tl.float32)
    one_d = tl.full([DK], 1.0, tl.float32)
    zero_d = tl.zeros([DK], tl.float32)
    cq = h * DK + jj
    ck = DKEY + cq
    cv = 2 * DKEY + cq
    # ---- q, k, v: conv step and SiLU
    q, q_h1, q_h2, q_x = _conv_silu(X, HIST, WT, xrow, hrow, s_h_c, cq, DK)
    k, k_h1, k_h2, k_x = _conv_silu(X, HIST, WT, xrow, hrow, s_h_c, ck, DK)
    v, v_h1, v_h2, v_x = _conv_silu(X, HIST, WT, xrow, hrow, s_h_c, cv, DK)
    # ---- q, k: interleaved RoPE at the row's position, normalisation, the match gain
    q_e, q_o = tl.split(tl.reshape(q, [HALF, 2]))
    k_e, k_o = tl.split(tl.reshape(k, [HALF, 2]))
    tb = tl.load(POS + b) * s_cs + h * HALF + ii
    c = tl.load(COS + tb).to(tl.float32)
    s = tl.load(SIN + tb).to(tl.float32)
    qr_e = _bf16(_bf16(q_e * c) - _bf16(q_o * s))
    qr_o = _bf16(_bf16(q_o * c) + _bf16(q_e * s))
    kr_e = _bf16(_bf16(k_e * c) - _bf16(k_o * s))
    kr_o = _bf16(_bf16(k_o * c) + _bf16(k_e * s))
    kn = zero_h + tl.maximum(tl.sqrt(tl.sum(kr_e * kr_e, axis=0) + tl.sum(kr_o * kr_o, axis=0)), 1e-12)
    qn = zero_h + tl.maximum(tl.sqrt(tl.sum(qr_e * qr_e, axis=0) + tl.sum(qr_o * qr_o, axis=0)), 1e-12)
    qg = zero_h + tl.load(QGAIN + h)
    k_hat = tl.reshape(tl.join(_div_rn(kr_e, kn), _div_rn(kr_o, kn)), [DK])   # the pairs back in channel order
    q_out = tl.reshape(tl.join(_mul_rn(_div_rn(qr_e, qn), qg), _mul_rn(_div_rn(qr_o, qn), qg)), [DK])
    # ---- v: the value-residual mix, the GLU tail of the last head
    gl = zero_d + tl.load(GL + b64 * s_gl_b + h).to(tl.float32)
    lam = _add_rn(tl.load(LAM0 + h * DK + jj), libdevice.tanh(gl))
    if HAS_CARRY:
        vm = _bf16(_add_rn(v, _mul_rn(lam, tl.load(CARRY + b64 * s_c_b + h * DK + jj).to(tl.float32))))
    else:
        vm = _bf16(_mul_rn(v, _add_rn(one_d, lam)))
    if h == H - 1:
        rr = tl.arange(0, GRB)
        rm = rr < GR
        ga = tl.load(GA + b64 * s_ga_b + rr, mask=rm, other=0.0).to(tl.float32)
        gb = tl.load(GB + b64 * s_gb_b + rr, mask=rm, other=0.0).to(tl.float32)
        one_r = tl.full([GRB], 1.0, tl.float32)
        gp = _bf16(_mul_rn(_div_rn(ga, _add_rn(one_r, libdevice.exp(-ga))), gb))
        u = tl.load(U + jj[:, None].to(tl.int64) * s_u + rr[None, :], mask=rm[None, :], other=0.0)   # (DK, GRB)
        vm = _bf16(_add_rn(vm, _bf16(tl.sum(_bf16(u.to(tl.float32)) * gp[None, :], axis=1))))
    # ---- the write tag and the tagged values
    g = _add_rn(tl.load(G + b64 * s_g_b + h * DK + jj).to(tl.float32), zero_d + tl.load(WBIAS + h))
    if HAS_SPARSITY:
        g = _sub_rn(g, zero_d + sparsity)
    sig = _bf16(_div_rn(one_d, _add_rn(one_d, libdevice.exp(-g))))
    gv = _mul_rn(sig, vm)
    # ---- the clock: the shared mean log decay times the head's eligibility slope, added to the carried clock
    dd = tl.arange(0, DR)
    x = tl.load(DLOG + b64 * s_d_b + dd).to(tl.float32)
    sp = tl.where(x > 20.0, x, libdevice.log1p(libdevice.exp(x)))
    a = libdevice.exp(_mul_rn(-tl.load(RATE + dd).to(tl.float32), sp))
    a = _mul_rn(a, tl.zeros([DR], tl.float32) + base_decay)
    la = tl.sum(libdevice.log(tl.maximum(a, 1e-6)), axis=0) / DR
    cum = tl.load(RUN + bh) + la * tl.load(GAINS + h)
    # ---- the writes
    tl.debug_barrier()
    _roll(HIST, hrow, s_h_c, cq, q_h1, q_h2, q_x)
    _roll(HIST, hrow, s_h_c, ck, k_h1, k_h2, k_x)
    _roll(HIST, hrow, s_h_c, cv, v_h1, v_h2, v_x)
    tl.store(PK + ring * DK + jj, k_hat.to(tl.bfloat16))
    tl.store(QOUT + bh * DK + jj, q_out.to(tl.bfloat16))
    tl.store(VOUT + bh * DK + jj, v.to(tl.bfloat16))
    tl.store(PGV + ring * DK + jj, gv.to(tl.bfloat16))
    tl.store(PCUM + ring, cum)
    tl.store(RUN + bh, cum)


# ================================================================================================ launcher
def _need(cond: bool, msg: str) -> None:
    if not cond:
        raise RuntimeError("neuronspark.ops.ring_write_step: " + msg)


def _pow2(n: int) -> bool:
    return n > 0 and (n & (n - 1)) == 0


def _ring_write_step_fwd(qkv, conv_hist, conv_weight, cos, sin, pos, ring_ptr, gate, gl, lam0, carry, ga, gb, glu_u,
                         dlog, rate, base_decay: float, gains, qgain, wbias, sparsity: float, pool_k, pool_gv,
                         pool_cum, pool_run) -> Tuple[torch.Tensor, torch.Tensor]:
    require_cuda(qkv, "ring_write_step")
    dev = qkv.device
    _need(pool_k.dim() == 4 and pool_k.dtype == torch.bfloat16 and pool_k.is_contiguous(),
          f"pool_k must be a contiguous bf16 (B, H, W, dk) ring, got {pool_k.dtype} {tuple(pool_k.shape)}")
    B, H, W, dk = (int(s) for s in pool_k.shape)
    Dk = H * dk
    C = 3 * Dk
    half = dk // 2
    _need(_pow2(dk) and dk >= 2, f"the head width must be a power of two, got {dk}")
    _need(tuple(pool_gv.shape) == (B, H, W, dk) and pool_gv.dtype == torch.bfloat16 and pool_gv.is_contiguous(),
          f"pool_gv must be a contiguous bf16 ({B}, {H}, {W}, {dk}) ring (value heads as wide as the key heads), got "
          f"{pool_gv.dtype} {tuple(pool_gv.shape)}")
    _need(tuple(pool_cum.shape) == (B, H, W) and pool_cum.dtype == torch.float32 and pool_cum.is_contiguous(),
          f"pool_cum must be a contiguous fp32 ({B}, {H}, {W}) ring")
    _need(tuple(pool_run.shape) == (B, H) and pool_run.dtype == torch.float32 and pool_run.is_contiguous(),
          f"pool_run must be a contiguous fp32 ({B}, {H}) tensor")
    _need(tuple(qkv.shape) == (B, C) and qkv.dtype == torch.bfloat16 and qkv.stride(1) == 1,
          f"qkv must be bf16 ({B}, {C}) with unit inner stride, got {qkv.dtype} {tuple(qkv.shape)}")
    _need(tuple(conv_hist.shape) == (B, C, 3) and conv_hist.dtype == torch.float32 and conv_hist.stride(2) == 1,
          f"conv_hist must be fp32 ({B}, {C}, 3), got {conv_hist.dtype} {tuple(conv_hist.shape)}")
    _need(tuple(conv_weight.shape) == (C, 1, 4) and conv_weight.dtype == torch.float32 and conv_weight.is_contiguous(),
          f"conv_weight must be a contiguous fp32 ({C}, 1, 4) tensor")
    for name, t in (("cos", cos), ("sin", sin)):
        _need(t.dim() == 2 and int(t.shape[1]) == Dk // 2 and t.dtype == torch.bfloat16 and t.stride(1) == 1,
              f"{name} must be a bf16 (positions, {Dk // 2}) table, got {t.dtype} {tuple(t.shape)}")
    _need(cos.stride(0) == sin.stride(0) and cos.shape == sin.shape, "cos / sin must share their layout")
    _need(tuple(pos.shape) == (B,) and pos.dtype == torch.int64 and pos.is_contiguous(),
          f"pos must be a contiguous ({B},) int64 tensor")
    _need(ring_ptr.numel() == 1 and ring_ptr.dtype == torch.int64, "ring_ptr must be a (1,) int64 tensor")
    _need(tuple(gate.shape) == (B, Dk) and gate.dtype in (torch.float32, torch.bfloat16) and gate.stride(1) == 1,
          f"the write-gate spikes must be fp32 / bf16 ({B}, {Dk}), got {gate.dtype} {tuple(gate.shape)}")
    _need(tuple(gl.shape) == (B, H) and gl.dtype == torch.bfloat16 and gl.stride(1) == 1,
          f"gl must be bf16 ({B}, {H}), got {gl.dtype} {tuple(gl.shape)}")
    _need(tuple(lam0.shape) == (Dk,) and lam0.dtype == torch.float32 and lam0.is_contiguous(),
          f"lam0 must be a contiguous fp32 ({Dk},) tensor")
    if carry is not None:
        _need(tuple(carry.shape) == (B, Dk) and carry.dtype == torch.bfloat16 and carry.stride(1) == 1,
              f"carry must be bf16 ({B}, {Dk}), got {carry.dtype} {tuple(carry.shape)}")
    _need(ga.dim() == 2 and int(ga.shape[0]) == B and tuple(gb.shape) == tuple(ga.shape)
          and ga.dtype == torch.bfloat16 and gb.dtype == torch.bfloat16 and ga.stride(1) == 1 and gb.stride(1) == 1,
          f"ga / gb must be bf16 ({B}, rank), got {tuple(ga.shape)} / {tuple(gb.shape)}")
    gr = int(ga.shape[1])
    _need(tuple(glu_u.shape) == (dk, gr) and glu_u.dtype in (torch.float32, torch.bfloat16)
          and glu_u.stride(1) == 1,
          f"glu_u must be an fp32 / bf16 ({dk}, {gr}) tensor with unit inner stride, got {glu_u.dtype} "
          f"{tuple(glu_u.shape)}")
    _need(dlog.dim() == 2 and int(dlog.shape[0]) == B and dlog.dtype == torch.bfloat16 and dlog.stride(1) == 1,
          f"dlog must be bf16 ({B}, decay width), got {dlog.dtype} {tuple(dlog.shape)}")
    dr = int(dlog.shape[1])
    _need(_pow2(dr), f"the decay width must be a power of two, got {dr}")
    _need(tuple(rate.shape) == (dr,) and rate.dtype == torch.bfloat16 and rate.is_contiguous(),
          f"rate must be a contiguous bf16 ({dr},) tensor")
    for name, t in (("gains", gains), ("qgain", qgain), ("wbias", wbias)):
        _need(tuple(t.shape) == (H,) and t.dtype == torch.float32 and t.is_contiguous(),
              f"{name} must be a contiguous fp32 ({H},) tensor, got {t.dtype} {tuple(t.shape)}")
    q_out = torch.empty((B, H, dk), dtype=torch.bfloat16, device=dev)
    v_out = torch.empty((B, Dk), dtype=torch.bfloat16, device=dev)
    _ring_write_step_kernel[(B, H)](
        qkv, conv_hist, conv_weight, cos, sin, pos, ring_ptr, gate, gl, lam0, carry if carry is not None else qkv,
        ga, gb, glu_u, dlog, rate, gains, qgain, wbias, pool_k, pool_gv, pool_cum, pool_run, q_out, v_out,
        int(qkv.stride(0)), int(conv_hist.stride(0)), int(conv_hist.stride(1)), int(cos.stride(0)),
        int(gate.stride(0)), int(gl.stride(0)), int(carry.stride(0)) if carry is not None else 0,
        int(ga.stride(0)), int(gb.stride(0)), int(dlog.stride(0)), int(glu_u.stride(0)),
        H, W, Dk, gr, float(base_decay), float(sparsity),
        DK=dk, HALF=half, DR=dr, GRB=triton.next_power_of_2(gr), HAS_CARRY=int(carry is not None),
        HAS_SPARSITY=int(float(sparsity) > 0.0), num_warps=RING_STEP_WARPS, enable_fp_fusion=False)
    return q_out, v_out


@torch.library.custom_op(_KERNEL_OP_NAMESPACE + "::ring_write_step",
                         mutates_args=("conv_hist", "pool_k", "pool_gv", "pool_cum", "pool_run"))
def _ring_write_step_op(qkv: torch.Tensor, conv_hist: torch.Tensor, conv_weight: torch.Tensor, cos: torch.Tensor,
                        sin: torch.Tensor, pos: torch.Tensor, ring_ptr: torch.Tensor, gate: torch.Tensor,
                        gl: torch.Tensor, lam0: torch.Tensor, carry: Optional[torch.Tensor], ga: torch.Tensor,
                        gb: torch.Tensor, glu_u: torch.Tensor, dlog: torch.Tensor, rate: torch.Tensor,
                        base_decay: float, gains: torch.Tensor, qgain: torch.Tensor, wbias: torch.Tensor,
                        sparsity: float, pool_k: torch.Tensor, pool_gv: torch.Tensor, pool_cum: torch.Tensor,
                        pool_run: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    return _ring_write_step_fwd(qkv, conv_hist, conv_weight, cos, sin, pos, ring_ptr, gate, gl, lam0, carry, ga, gb,
                                glu_u, dlog, rate, base_decay, gains, qgain, wbias, sparsity, pool_k, pool_gv,
                                pool_cum, pool_run)


@_ring_write_step_op.register_fake
def _ring_write_step_fake(qkv, conv_hist, conv_weight, cos, sin, pos, ring_ptr, gate, gl, lam0, carry, ga, gb, glu_u,
                          dlog, rate, base_decay, gains, qgain, wbias, sparsity, pool_k, pool_gv, pool_cum, pool_run):
    B, H, _, dk = pool_k.shape
    return (qkv.new_empty((B, H, dk), dtype=torch.bfloat16), qkv.new_empty((B, H * dk), dtype=torch.bfloat16))


# ================================================================================================ public function
def ring_write_step(qkv: torch.Tensor, conv_hist: torch.Tensor, conv_weight: torch.Tensor, cos: torch.Tensor,
                    sin: torch.Tensor, pos: torch.Tensor, ring_ptr: torch.Tensor, gate: torch.Tensor,
                    gl: torch.Tensor, lam0: torch.Tensor, carry: Optional[torch.Tensor], ga: torch.Tensor,
                    gb: torch.Tensor, glu_u: torch.Tensor, dlog: torch.Tensor, rate: torch.Tensor, base_decay: float,
                    gains: torch.Tensor, qgain: torch.Tensor, wbias: torch.Tensor, sparsity: float,
                    pool_k: torch.Tensor, pool_gv: torch.Tensor, pool_cum: torch.Tensor,
                    pool_run: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """The ring write of a decode step (module docstring).

    qkv (B, 3 H dk) bf16: the qkv projection; conv_hist (B, 3 H dk, 3) fp32: the conv's carried taps (rolled in
    place); conv_weight (3 H dk, 1, 4) fp32; cos / sin (positions, H dk / 2) bf16: the RoPE tables; pos (B,) int64:
    the rows' positions; ring_ptr (1,) int64; gate (B, H dk) fp32: the write-gate spikes; gl (B, H) bf16: the
    value-residual gate projection; lam0 (H dk,) fp32: the static mix; carry (B, H dk) bf16 or None (hippocampal layer
    0); ga / gb (B, g) bf16: the GLU maps; glu_u (dk, g) fp32 (its bf16 inference copy is read when it has one); dlog
    (B, dk) bf16: the decay logit; rate (dk,) bf16;
    base_decay; gains / qgain / wbias (H,) fp32; sparsity; pool_k / pool_gv (B, H, W, dk) bf16, pool_cum (B, H, W)
    fp32: the rings (this step's slot written); pool_run (B, H) fp32: the carried clocks (advanced in place).

    ``ring_ptr`` and ``pos`` are counts (>= 0; ``pos`` inside the tables): the kernel reads them on the device.  The
    state tensors (``conv_hist``, the rings, ``pool_run``) are written by one program per (row, head) and must not
    share memory with another argument.

    Returns (q (B, H, dk) bf16 gained unit queries, v (B, H dk) bf16 values after the SiLU)."""
    return _KERNEL_OPS.ring_write_step(qkv, conv_hist, conv_weight, cos, sin, pos, ring_ptr, gate, gl, lam0,
                                                 carry, ga, gb, inference_weight(glu_u), dlog, rate,
                                                 float(base_decay), gains, qgain,
                                                 wbias, float(sparsity), pool_k, pool_gv, pool_cum, pool_run)


# ================================================================================================ pure-torch reference
@torch.no_grad()
def ring_write_step_ref(qkv: torch.Tensor, conv_hist: torch.Tensor, conv_weight: torch.Tensor, cos: torch.Tensor,
                        sin: torch.Tensor, pos: torch.Tensor, ring_ptr: torch.Tensor, gate: torch.Tensor,
                        gl: torch.Tensor, lam0: torch.Tensor, carry: Optional[torch.Tensor], ga: torch.Tensor,
                        gb: torch.Tensor, glu_u: torch.Tensor, dlog: torch.Tensor, rate: torch.Tensor,
                        base_decay: float, gains: torch.Tensor, qgain: torch.Tensor, wbias: torch.Tensor,
                        sparsity: float, pool_k: torch.Tensor, pool_gv: torch.Tensor, pool_cum: torch.Tensor,
                        pool_run: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """The torch expression chain of the step's ring write under the inference forward's bf16 autocast.  Same
    arguments, state updates and outputs as :func:`ring_write_step`."""
    B, H, W, dk = (int(s) for s in pool_k.shape)
    Dk = H * dk
    with torch.autocast("cuda", dtype=torch.bfloat16):
        x = short_conv_step_ref(qkv, conv_hist, conv_weight)                      # (B, C) bf16, the taps rolled
        q, k, v = x.split([Dk, Dk, Dk], dim=-1)
        q, k, v = F.silu(q), F.silu(k), F.silu(v)
        # value-residual mix and the GLU tail of the last head
        lam = lam0.view(1, -1) + torch.tanh(gl.to(torch.float32)).repeat_interleave(dk, dim=-1)
        if carry is None:
            vm = (v.float() * (1.0 + lam)).to(v.dtype)
        else:
            vm = (v.float() + lam * carry.float()).to(v.dtype)
        gp = F.silu(ga.float()) * gb.float()
        add = F.linear(gp.to(torch.bfloat16), glu_u.to(torch.bfloat16)).float()
        vm = torch.cat([vm[:, :Dk - dk], (vm[:, Dk - dk:].float() + add).to(v.dtype)], dim=-1)
        # RoPE at the rows' positions, the unit keys and the gained unit queries
        c, s = cos[pos].to(q.dtype), sin[pos].to(q.dtype)
        q = apply_rope_interleaved_ref(q, c, s)
        k = apply_rope_interleaved_ref(k, c, s)
        k_hat = F.normalize(k.reshape(B, H, dk), dim=-1)
        q_h = F.normalize(q.reshape(B, H, dk), dim=-1)
        q_h = q_h * qgain.to(q_h.dtype).view(1, H, 1)
        # the write tag and the tagged values
        g_raw = gate.reshape(B, H, dk).float() + wbias.view(1, H, 1)
        if sparsity > 0.0:
            g_raw = g_raw - sparsity
        gv = torch.sigmoid(g_raw).to(vm.dtype) * vm.reshape(B, H, dk)
        # the clock
        a = torch.exp(-rate * F.softplus(dlog.float())) * base_decay
        la = a.float().clamp_min(DECAY_FLOOR).log().mean(dim=-1, keepdim=True)   # (B, 1)
        cum = pool_run + la * gains.view(1, H)
        slot = ring_ptr.reshape(1) % W
        pool_k.index_copy_(2, slot, k_hat.unsqueeze(2).to(pool_k.dtype))
        pool_gv.index_copy_(2, slot, gv.unsqueeze(2).to(pool_gv.dtype))
        pool_cum.index_copy_(2, slot, cum.unsqueeze(2))
        pool_run.copy_(cum)
    return q_h.to(torch.bfloat16), v
