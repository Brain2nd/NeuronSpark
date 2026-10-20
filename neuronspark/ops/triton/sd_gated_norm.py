"""SD gated RMSNorm (inference prefill): the normalisation of the SSD output gated by SiLU(z), unit weight, rounded to
bf16 (the soma read-out's input).

Op (registry name ``sd_gated_rmsnorm``; no autograd)::

    n = sd_gated_rmsnorm(y, z, eps)        # y, z (M, N) bf16 -> n (M, N) bf16 contiguous

    g    = y * (z * sigmoid(z))                         fp32
    rstd = 1 / sqrt(sum_N g^2 / N + eps)                one row
    n    = bf16((g * rstd) * 1)                         the unit weight of the SD gated norm

``y`` is the scan output (rows of B * T), ``z`` the gate slice of the in-projection read in place (a (M, N) view with
the projection's row stride and a unit last stride).  1MIX: N = 9216, eps 1e-5.

What the kernel reproduces.  The reference implementation of this call is flash-linear-attention's gated layer norm
(``fla.modules.layernorm_gated.layer_norm_fwd``, ``norm_before_gate=False``, ``is_rms_norm=True``, one group), which
for 8192 < N <= 16384 runs one 8-warp program per row with the whole row in registers (BLOCK_N 16384).  Triton's sum
over that [16384] tensor is (1) a sequential chain in each of the 256 threads over its register slots, (2) a 32-lane
xor butterfly inside each warp, (3) the 8 warp partials combined by the same butterfly; a padded slot adds an exact
0.0.  This kernel forms each thread's chain explicitly over the real columns in register order -- per rep of
256 x 8 columns, one 16-byte vector load per thread per operand, the thread's 8 slots peeled out of its registers with
``tl.split`` and added in order 0..7 -- and hands the 256 partials to ``tl.sum`` on a [256] tensor with the same 8
warps, which is stages (2) and (3) again.  The gated row of each rep stays in registers between the two passes (no
second load or sigmoid).  The division, square root, sigmoid and the final cast are the same Triton expressions as
fla's.  The weight operand is the unit vector (materialised per call), multiplied as in fla.  Tests: the
per-operator tolerance rule of :mod:`neuronspark.utils.testing` against the reference.

Launch constants: 8 warps (256 threads), VEC 8 elements per thread per rep, rep-outer chain order, the
register-resident two-pass form; one program per row.

Contract (anything else raises): CUDA bf16 ``y`` and ``z`` of the same (M, N) shape on one device, unit last strides,
8192 < N <= 16384 (the domain of the reproduced fla layout), M >= 1, row offsets below 2^31 elements.  Reference:
:func:`sd_gated_rmsnorm_ref` (pure torch, fp32, the same formula with torch's reduction order).  The operands are 2-D
views; the gate is read in place.  The reproduced kernel: fla-core 0.5.2 ``layer_norm_fwd_kernel`` (MIT; see NOTICE).
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl

from neuronspark.ops.triton import require_cuda

__all__ = [
    "BLOCK_N", "NUM_WARPS", "THREADS", "VEC", "MIN_N_EXCLUSIVE",
    "sd_gated_rmsnorm", "sd_gated_rmsnorm_ref",
]

# ------------------------------------------------------------------------------------------------ launch constants
BLOCK_N = 16384          # fla's BLOCK_N for 8192 < N <= 16384 (bf16: MAX_FUSED_SIZE 32768, next_power_of_2(N))
NUM_WARPS = 8            # fla: min(max(BLOCK_N // 256, 1), 8)
THREADS = 32 * NUM_WARPS
VEC = 8                  # fla's elements per thread per rep (16-byte bf16 vectors)
MIN_N_EXCLUSIVE = 8192   # the reproduced layout holds for 8192 < N <= BLOCK_N
_MAX_REPS = 8            # BLOCK_N // (THREADS * VEC): the unrolled register-resident reps


# ------------------------------------------------------------------------------------------------ kernel
@triton.jit
def _gated_rmsnorm_kernel(X, Z, W, Y, sx, sz, sy, N, eps,
                          VEC: tl.constexpr, T: tl.constexpr, NREP_EFF: tl.constexpr):
    row = tl.program_id(0)
    X += row * sx
    Z += row * sz
    Y += row * sy
    acc = tl.zeros([T], dtype=tl.float32)
    # pass 1: each thread's sequential chain over its register slots (rep, j) of fla's [BLOCK_N] layout.  The gated
    # fp32 row stays in registers between the passes (NREP_EFF x 8 values per thread): pass 1 loads x / z once, gates,
    # forms the chain; pass 2 scales the kept values -- no second load or sigmoid.
    tl.static_assert(VEC == 8, "the chain peels 8 slots")
    tl.static_assert(NREP_EFF <= 8, "the chain unrolls up to 8 reps")
    if NREP_EFF > 0:
        cols0 = 0 * (T * VEC) + tl.arange(0, T * VEC)
        m0 = cols0 < N
        g0 = tl.load(X + cols0, mask=m0, other=0.0).to(tl.float32)
        z0 = tl.load(Z + cols0, mask=m0, other=0.0).to(tl.float32)
        g0 *= z0 * tl.sigmoid(z0)
        xb0 = tl.where(m0, g0, 0.)
        p = xb0 * xb0
        pe, po = tl.split(tl.reshape(p, [T, 4, 2]))          # j even / j odd
        pee, peo = tl.split(tl.reshape(pe, [T, 2, 2]))       # {0, 4} / {2, 6}
        poe, poo = tl.split(tl.reshape(po, [T, 2, 2]))       # {1, 5} / {3, 7}
        p0, p4 = tl.split(pee)
        p2, p6 = tl.split(peo)
        p1, p5 = tl.split(poe)
        p3, p7 = tl.split(poo)
        acc = acc + p0
        acc = acc + p1
        acc = acc + p2
        acc = acc + p3
        acc = acc + p4
        acc = acc + p5
        acc = acc + p6
        acc = acc + p7
    if NREP_EFF > 1:
        cols1 = 1 * (T * VEC) + tl.arange(0, T * VEC)
        m1 = cols1 < N
        g1 = tl.load(X + cols1, mask=m1, other=0.0).to(tl.float32)
        z1 = tl.load(Z + cols1, mask=m1, other=0.0).to(tl.float32)
        g1 *= z1 * tl.sigmoid(z1)
        xb1 = tl.where(m1, g1, 0.)
        p = xb1 * xb1
        pe, po = tl.split(tl.reshape(p, [T, 4, 2]))
        pee, peo = tl.split(tl.reshape(pe, [T, 2, 2]))
        poe, poo = tl.split(tl.reshape(po, [T, 2, 2]))
        p0, p4 = tl.split(pee)
        p2, p6 = tl.split(peo)
        p1, p5 = tl.split(poe)
        p3, p7 = tl.split(poo)
        acc = acc + p0
        acc = acc + p1
        acc = acc + p2
        acc = acc + p3
        acc = acc + p4
        acc = acc + p5
        acc = acc + p6
        acc = acc + p7
    if NREP_EFF > 2:
        cols2 = 2 * (T * VEC) + tl.arange(0, T * VEC)
        m2 = cols2 < N
        g2 = tl.load(X + cols2, mask=m2, other=0.0).to(tl.float32)
        z2 = tl.load(Z + cols2, mask=m2, other=0.0).to(tl.float32)
        g2 *= z2 * tl.sigmoid(z2)
        xb2 = tl.where(m2, g2, 0.)
        p = xb2 * xb2
        pe, po = tl.split(tl.reshape(p, [T, 4, 2]))
        pee, peo = tl.split(tl.reshape(pe, [T, 2, 2]))
        poe, poo = tl.split(tl.reshape(po, [T, 2, 2]))
        p0, p4 = tl.split(pee)
        p2, p6 = tl.split(peo)
        p1, p5 = tl.split(poe)
        p3, p7 = tl.split(poo)
        acc = acc + p0
        acc = acc + p1
        acc = acc + p2
        acc = acc + p3
        acc = acc + p4
        acc = acc + p5
        acc = acc + p6
        acc = acc + p7
    if NREP_EFF > 3:
        cols3 = 3 * (T * VEC) + tl.arange(0, T * VEC)
        m3 = cols3 < N
        g3 = tl.load(X + cols3, mask=m3, other=0.0).to(tl.float32)
        z3 = tl.load(Z + cols3, mask=m3, other=0.0).to(tl.float32)
        g3 *= z3 * tl.sigmoid(z3)
        xb3 = tl.where(m3, g3, 0.)
        p = xb3 * xb3
        pe, po = tl.split(tl.reshape(p, [T, 4, 2]))
        pee, peo = tl.split(tl.reshape(pe, [T, 2, 2]))
        poe, poo = tl.split(tl.reshape(po, [T, 2, 2]))
        p0, p4 = tl.split(pee)
        p2, p6 = tl.split(peo)
        p1, p5 = tl.split(poe)
        p3, p7 = tl.split(poo)
        acc = acc + p0
        acc = acc + p1
        acc = acc + p2
        acc = acc + p3
        acc = acc + p4
        acc = acc + p5
        acc = acc + p6
        acc = acc + p7
    if NREP_EFF > 4:
        cols4 = 4 * (T * VEC) + tl.arange(0, T * VEC)
        m4 = cols4 < N
        g4 = tl.load(X + cols4, mask=m4, other=0.0).to(tl.float32)
        z4 = tl.load(Z + cols4, mask=m4, other=0.0).to(tl.float32)
        g4 *= z4 * tl.sigmoid(z4)
        xb4 = tl.where(m4, g4, 0.)
        p = xb4 * xb4
        pe, po = tl.split(tl.reshape(p, [T, 4, 2]))
        pee, peo = tl.split(tl.reshape(pe, [T, 2, 2]))
        poe, poo = tl.split(tl.reshape(po, [T, 2, 2]))
        p0, p4 = tl.split(pee)
        p2, p6 = tl.split(peo)
        p1, p5 = tl.split(poe)
        p3, p7 = tl.split(poo)
        acc = acc + p0
        acc = acc + p1
        acc = acc + p2
        acc = acc + p3
        acc = acc + p4
        acc = acc + p5
        acc = acc + p6
        acc = acc + p7
    if NREP_EFF > 5:
        cols5 = 5 * (T * VEC) + tl.arange(0, T * VEC)
        m5 = cols5 < N
        g5 = tl.load(X + cols5, mask=m5, other=0.0).to(tl.float32)
        z5 = tl.load(Z + cols5, mask=m5, other=0.0).to(tl.float32)
        g5 *= z5 * tl.sigmoid(z5)
        xb5 = tl.where(m5, g5, 0.)
        p = xb5 * xb5
        pe, po = tl.split(tl.reshape(p, [T, 4, 2]))
        pee, peo = tl.split(tl.reshape(pe, [T, 2, 2]))
        poe, poo = tl.split(tl.reshape(po, [T, 2, 2]))
        p0, p4 = tl.split(pee)
        p2, p6 = tl.split(peo)
        p1, p5 = tl.split(poe)
        p3, p7 = tl.split(poo)
        acc = acc + p0
        acc = acc + p1
        acc = acc + p2
        acc = acc + p3
        acc = acc + p4
        acc = acc + p5
        acc = acc + p6
        acc = acc + p7
    if NREP_EFF > 6:
        cols6 = 6 * (T * VEC) + tl.arange(0, T * VEC)
        m6 = cols6 < N
        g6 = tl.load(X + cols6, mask=m6, other=0.0).to(tl.float32)
        z6 = tl.load(Z + cols6, mask=m6, other=0.0).to(tl.float32)
        g6 *= z6 * tl.sigmoid(z6)
        xb6 = tl.where(m6, g6, 0.)
        p = xb6 * xb6
        pe, po = tl.split(tl.reshape(p, [T, 4, 2]))
        pee, peo = tl.split(tl.reshape(pe, [T, 2, 2]))
        poe, poo = tl.split(tl.reshape(po, [T, 2, 2]))
        p0, p4 = tl.split(pee)
        p2, p6 = tl.split(peo)
        p1, p5 = tl.split(poe)
        p3, p7 = tl.split(poo)
        acc = acc + p0
        acc = acc + p1
        acc = acc + p2
        acc = acc + p3
        acc = acc + p4
        acc = acc + p5
        acc = acc + p6
        acc = acc + p7
    if NREP_EFF > 7:
        cols7 = 7 * (T * VEC) + tl.arange(0, T * VEC)
        m7 = cols7 < N
        g7 = tl.load(X + cols7, mask=m7, other=0.0).to(tl.float32)
        z7 = tl.load(Z + cols7, mask=m7, other=0.0).to(tl.float32)
        g7 *= z7 * tl.sigmoid(z7)
        xb7 = tl.where(m7, g7, 0.)
        p = xb7 * xb7
        pe, po = tl.split(tl.reshape(p, [T, 4, 2]))
        pee, peo = tl.split(tl.reshape(pe, [T, 2, 2]))
        poe, poo = tl.split(tl.reshape(po, [T, 2, 2]))
        p0, p4 = tl.split(pee)
        p2, p6 = tl.split(peo)
        p1, p5 = tl.split(poe)
        p3, p7 = tl.split(poo)
        acc = acc + p0
        acc = acc + p1
        acc = acc + p2
        acc = acc + p3
        acc = acc + p4
        acc = acc + p5
        acc = acc + p6
        acc = acc + p7
    var = tl.sum(acc, axis=0) / N
    rstd = 1 / tl.sqrt(var + eps)
    # pass 2: the normalised, weighted kept values (element-wise)
    if NREP_EFF > 0:
        w0 = tl.load(W + cols0, mask=m0).to(tl.float32)
        x_hat0 = g0 * rstd
        y0 = x_hat0 * w0
        tl.store(Y + cols0, y0, mask=m0)
    if NREP_EFF > 1:
        w1 = tl.load(W + cols1, mask=m1).to(tl.float32)
        x_hat1 = g1 * rstd
        y1 = x_hat1 * w1
        tl.store(Y + cols1, y1, mask=m1)
    if NREP_EFF > 2:
        w2 = tl.load(W + cols2, mask=m2).to(tl.float32)
        x_hat2 = g2 * rstd
        y2 = x_hat2 * w2
        tl.store(Y + cols2, y2, mask=m2)
    if NREP_EFF > 3:
        w3 = tl.load(W + cols3, mask=m3).to(tl.float32)
        x_hat3 = g3 * rstd
        y3 = x_hat3 * w3
        tl.store(Y + cols3, y3, mask=m3)
    if NREP_EFF > 4:
        w4 = tl.load(W + cols4, mask=m4).to(tl.float32)
        x_hat4 = g4 * rstd
        y4 = x_hat4 * w4
        tl.store(Y + cols4, y4, mask=m4)
    if NREP_EFF > 5:
        w5 = tl.load(W + cols5, mask=m5).to(tl.float32)
        x_hat5 = g5 * rstd
        y5 = x_hat5 * w5
        tl.store(Y + cols5, y5, mask=m5)
    if NREP_EFF > 6:
        w6 = tl.load(W + cols6, mask=m6).to(tl.float32)
        x_hat6 = g6 * rstd
        y6 = x_hat6 * w6
        tl.store(Y + cols6, y6, mask=m6)
    if NREP_EFF > 7:
        w7 = tl.load(W + cols7, mask=m7).to(tl.float32)
        x_hat7 = g7 * rstd
        y7 = x_hat7 * w7
        tl.store(Y + cols7, y7, mask=m7)


# ------------------------------------------------------------------------------------------------ launcher
def _gated_rmsnorm_fwd(y2: torch.Tensor, w: torch.Tensor, z2: torch.Tensor, eps: float) -> torch.Tensor:
    """fla ``layer_norm_fwd(y2, w, None, eps, z=z2, group_size=N, norm_before_gate=False, is_rms_norm=True)[0]`` for
    the reproduced domain (operands checked by the caller); a contiguous (M, N) bf16 output."""
    M, N = (int(s) for s in y2.shape)
    out = torch.empty((M, N), dtype=y2.dtype, device=y2.device)
    with torch.cuda.device(y2.device.index):
        _gated_rmsnorm_kernel[(M,)](y2, z2, w, out, y2.stride(0), z2.stride(0), out.stride(0), N, float(eps),
                                    VEC=VEC, T=THREADS, NREP_EFF=triton.cdiv(N, THREADS * VEC),
                                    num_warps=NUM_WARPS)
    return out


@torch.library.custom_op("neuronspark::sd_gated_rmsnorm", mutates_args=(),
                         schema="(Tensor y, Tensor z, float eps) -> Tensor")
def _sd_gated_rmsnorm_op(y, z, eps):
    """n (M, N) bf16 contiguous; the unit weight is materialised here (N fp32 ones)"""
    w = torch.ones((int(y.shape[1]),), dtype=torch.float32, device=y.device)
    return _gated_rmsnorm_fwd(y, w, z, eps)


@_sd_gated_rmsnorm_op.register_fake
def _sd_gated_rmsnorm_fake(y, z, eps):
    return torch.empty(tuple(y.shape), dtype=y.dtype, device=y.device)


# ------------------------------------------------------------------------------------------------ public function
def _fail(msg: str) -> None:
    raise RuntimeError(f"neuronspark.sd_gated_rmsnorm: {msg}")


def sd_gated_rmsnorm(y: torch.Tensor, z: torch.Tensor, eps: float) -> torch.Tensor:
    """``n = bf16(rmsnorm(y * silu(z)))`` with unit weight (see the module docstring): y, z (M, N) bf16 with unit last
    strides (z may be the strided gate slice of the projection) -> n (M, N) bf16 contiguous.  No autograd."""
    for name, t in (("y", y), ("z", z)):
        if not isinstance(t, torch.Tensor):
            _fail(f"{name} must be a tensor, got {type(t).__name__}")
    require_cuda(y, "sd_gated_rmsnorm")
    if z.device != y.device:
        _fail(f"z is on {z.device}, y on {y.device}")
    if y.dim() != 2 or tuple(z.shape) != tuple(y.shape):
        _fail(f"y and z must be (M, N) of one shape, got {tuple(y.shape)} {tuple(z.shape)}")
    if y.dtype != torch.bfloat16 or z.dtype != torch.bfloat16:
        _fail(f"y and z must be bf16, got {y.dtype} {z.dtype}")
    M, N = (int(s) for s in y.shape)
    if M < 1 or not MIN_N_EXCLUSIVE < N <= BLOCK_N:
        _fail(f"rows M >= 1 and {MIN_N_EXCLUSIVE} < N <= {BLOCK_N} (the reproduced layout) expected, got ({M}, {N})")
    if y.stride(1) != 1 or z.stride(1) != 1:
        _fail(f"unit last strides expected, got y {tuple(y.stride())} z {tuple(z.stride())}")
    if (M - 1) * max(int(y.stride(0)), int(z.stride(0)), N) + N >= 2 ** 31:
        _fail(f"row offsets of ({M}, {N}) with strides {tuple(y.stride())} / {tuple(z.stride())} exceed 2^31 elements")
    if torch.is_grad_enabled() and (y.requires_grad or z.requires_grad):
        _fail("an inference operator (no autograd): call it under torch.no_grad()")
    return torch.ops.neuronspark.sd_gated_rmsnorm(y, z, float(eps))


# ------------------------------------------------------------------------------------------------ reference
def sd_gated_rmsnorm_ref(y: torch.Tensor, z: torch.Tensor, eps: float) -> torch.Tensor:
    """Pure torch, fp32 (fla ``layer_norm_fwd`` with ``norm_before_gate=False``, RMS, unit weight, one group):
    ``g = y * (z * sigmoid(z))``, ``rstd = 1 / sqrt(sum(g^2) / N + eps)``, ``bf16(g * rstd)``; contiguous."""
    g = y.float()
    zf = z.float()
    g = g * (zf * torch.sigmoid(zf))
    var = (g * g).sum(dim=-1, keepdim=True) / int(g.shape[-1])
    rstd = 1.0 / torch.sqrt(var + float(eps))
    return (g * rstd).to(torch.bfloat16).contiguous()
