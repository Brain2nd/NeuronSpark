"""dendritic-layer gated RMSNorm + soma read-out (the inference prefill and the training forward): the normalisation
of the SSD output gated by SiLU(z), unit weight, rounded to bf16, then the bipolar soft-threshold soma times gamma --
W_out's bf16 input -- in one kernel.

Op (registry name ``dendritic_gated_norm_soma``; no autograd)::

    x = dendritic_gated_norm_soma(y, z, theta, gamma, eps)   # y, z (M, N) bf16 -> x (M, N) bf16 contiguous

    g    = y * (z * sigmoid(z))                         fp32
    rstd = 1 / sqrt(sum_N g^2 / N + eps)                one row
    n    = bf16(g * rstd)                               the dendritic-layer gated norm (unit weight)
    x    = bf16(S(n, theta) * gamma),  S(v, th) = max(v - th, 0) - max(-v - th, 0)

``y`` is the scan output (rows of B * T), ``z`` the gate slice of the in-projection read in place (a (M, N) view with
the projection's row stride and a unit last stride), theta / gamma (N,) fp32.  Default configuration: N = 9216, eps
1e-5.  n never goes through memory: the soma is applied to the bf16-rounded n in registers with the IEEE single-rounding
PTX of the training read-out (:mod:`neuronspark.ops.triton.dendritic_soma`), so x equals the read-out of the stored n.

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
fla's.  The norm's weight is the unit vector: the product with it is not formed (x * 1 = x).  A narrower row
(2048 < N <= 8192: the model widths 512 and 1024, N = 3072 and 6144) runs the same kernel with fewer reps: the same
function with this kernel's summation order (fla itself would choose a smaller block there).  Tests: the
per-operator tolerance rule of :mod:`neuronspark.utils.testing` against the reference.

The training forward (``gated_norm_soma_fwd(x, eps, z, theta, gamma) -> (x_soma, rstd, n)``, internal to
:class:`neuronspark.ops.dendritic_combined.DendriticChunkScanSomaFn`) is the same kernel with n (bf16) and the rows'
rstd (fp32) stored for the backward; it takes rows of any N up to 16384 and row offsets of any size.

Launch constants: 8 warps (256 threads), VEC 8 elements per thread per rep, rep-outer chain order, the
register-resident two-pass form; one program per row.

Contract (anything else raises): CUDA bf16 ``y`` and ``z`` of the same (M, N) shape on one device, unit last strides,
2048 < N <= 16384 (the fla layout is reproduced for 8192 < N), M >= 1, row offsets below 2^31 elements; theta / gamma
contiguous fp32 (N,). Reference: :func:`dendritic_gated_norm_soma_ref` (pure torch, fp32, the same formulas with torch's
reduction order). The operands are 2-D views; the gate is read in place. The reproduced kernel: fla-core 0.5.2
``layer_norm_fwd_kernel`` (MIT; see NOTICE).
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl

from neuronspark.ops.triton import require_cuda
from neuronspark.ops.triton._ptx import _sr_bf16_to_f32, _sr_f32_to_bf16, _sr_mul_rn, _sr_release
from neuronspark.ops.triton.dendritic_soma import dendritic_soma_readout_ref

__all__ = [
    "BLOCK_N", "NUM_WARPS", "THREADS", "VEC", "MIN_N_EXCLUSIVE",
    "dendritic_gated_norm_soma", "gated_norm_soma_fwd", "dendritic_gated_norm_soma_ref",
    "dendritic_gated_rmsnorm_ref",
]

# ------------------------------------------------------------------------------------------------ launch constants
BLOCK_N = 16384          # fla's BLOCK_N for 8192 < N <= 16384 (bf16: MAX_FUSED_SIZE 32768, next_power_of_2(N))
NUM_WARPS = 8            # fla: min(max(BLOCK_N // 256, 1), 8)
THREADS = 32 * NUM_WARPS
VEC = 8                  # fla's elements per thread per rep (16-byte bf16 vectors)
MIN_N_EXCLUSIVE = 2048   # rows of more than one rep (THREADS * VEC columns) up to BLOCK_N
_MAX_REPS = 8            # BLOCK_N // (THREADS * VEC): the unrolled register-resident reps


# ------------------------------------------------------------------------------------------------ kernel
@triton.jit
def _rep_load(X, Z, k: tl.constexpr, N, T: tl.constexpr, VEC: tl.constexpr):
    """rep k of the row (T * VEC columns): the bf16 operands as loaded, zero past N"""
    cols = k * (T * VEC) + tl.arange(0, T * VEC)
    m = cols < N
    return tl.load(X + cols, mask=m, other=0.0), tl.load(Z + cols, mask=m, other=0.0)


@triton.jit
def _rep_gate(xr, zr, acc, k: tl.constexpr, N, T: tl.constexpr, VEC: tl.constexpr):
    """rep k: the gated fp32 values g = x * (z * sigmoid(z)) and each thread's chain continued over the rep's 8
    register slots in order 0..7 (the slots peeled out of the thread's registers with ``tl.split``)"""
    cols = k * (T * VEC) + tl.arange(0, T * VEC)
    g = xr.to(tl.float32)
    z = zr.to(tl.float32)
    g *= z * tl.sigmoid(z)
    xb = tl.where(cols < N, g, 0.)
    p = xb * xb
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
    return g, acc


@triton.jit
def _soma_store(Y, NOUT, TH, GA, y, k: tl.constexpr, N, T: tl.constexpr, VEC: tl.constexpr, KEEP: tl.constexpr):
    """rep k: the soma read-out of the bf16-rounded norm value, x = bf16(S(n, theta) * gamma); KEEP also stores n"""
    cols = k * (T * VEC) + tl.arange(0, T * VEC)
    m = cols < N
    if KEEP:
        tl.store(NOUT + cols, _sr_f32_to_bf16(y), mask=m)
    v = _sr_bf16_to_f32(_sr_f32_to_bf16(y))
    th = tl.load(TH + cols, mask=m, other=0.0)
    ga = tl.load(GA + cols, mask=m, other=0.0)
    sv, _a, _b = _sr_release(v, th)
    tl.store(Y + cols, _sr_f32_to_bf16(_sr_mul_rn(sv, ga)), mask=m)


@triton.jit
def _gated_norm_soma_kernel(X, Z, Y, NOUT, RSTD, TH, GA, sx, sz, sy, N, eps,
                            VEC: tl.constexpr, T: tl.constexpr, NREP_EFF: tl.constexpr, KEEP: tl.constexpr):
    """One row per program.  KEEP (the training forward): n and the row's rstd are stored as well."""
    row = tl.program_id(0).to(tl.int64)                  # int64: rows x row stride may exceed 2^31 elements
    X += row * sx
    Z += row * sz
    Y += row * sy
    NOUT += row * sy
    tl.static_assert(VEC == 8, "the chain peels 8 slots")
    tl.static_assert(NREP_EFF <= 8, "the chain unrolls up to 8 reps")
    # pass 1: each thread's sequential chain over its register slots (rep, j) of fla's [BLOCK_N] layout.  The gated
    # fp32 row stays in registers between the passes (NREP_EFF x 8 values per thread): pass 1 loads x / z once, gates,
    # forms the chain; pass 2 scales the kept values -- no second load or sigmoid.
    acc = tl.zeros([T], dtype=tl.float32)
    if NREP_EFF > 0:
        x0, z0 = _rep_load(X, Z, 0, N, T, VEC)
        g0, acc = _rep_gate(x0, z0, acc, 0, N, T, VEC)
    if NREP_EFF > 1:
        x1, z1 = _rep_load(X, Z, 1, N, T, VEC)
        g1, acc = _rep_gate(x1, z1, acc, 1, N, T, VEC)
    if NREP_EFF > 2:
        x2, z2 = _rep_load(X, Z, 2, N, T, VEC)
        g2, acc = _rep_gate(x2, z2, acc, 2, N, T, VEC)
    if NREP_EFF > 3:
        x3, z3 = _rep_load(X, Z, 3, N, T, VEC)
        g3, acc = _rep_gate(x3, z3, acc, 3, N, T, VEC)
    if NREP_EFF > 4:
        x4, z4 = _rep_load(X, Z, 4, N, T, VEC)
        g4, acc = _rep_gate(x4, z4, acc, 4, N, T, VEC)
    if NREP_EFF > 5:
        x5, z5 = _rep_load(X, Z, 5, N, T, VEC)
        g5, acc = _rep_gate(x5, z5, acc, 5, N, T, VEC)
    if NREP_EFF > 6:
        x6, z6 = _rep_load(X, Z, 6, N, T, VEC)
        g6, acc = _rep_gate(x6, z6, acc, 6, N, T, VEC)
    if NREP_EFF > 7:
        x7, z7 = _rep_load(X, Z, 7, N, T, VEC)
        g7, acc = _rep_gate(x7, z7, acc, 7, N, T, VEC)
    var = tl.sum(acc, axis=0) / N
    rstd = 1 / tl.sqrt(var + eps)
    # pass 2: the normalised kept values (element-wise; the unit weight is not multiplied: x * 1 = x)
    if NREP_EFF > 0:
        _soma_store(Y, NOUT, TH, GA, g0 * rstd, 0, N, T, VEC, KEEP)
    if NREP_EFF > 1:
        _soma_store(Y, NOUT, TH, GA, g1 * rstd, 1, N, T, VEC, KEEP)
    if NREP_EFF > 2:
        _soma_store(Y, NOUT, TH, GA, g2 * rstd, 2, N, T, VEC, KEEP)
    if NREP_EFF > 3:
        _soma_store(Y, NOUT, TH, GA, g3 * rstd, 3, N, T, VEC, KEEP)
    if NREP_EFF > 4:
        _soma_store(Y, NOUT, TH, GA, g4 * rstd, 4, N, T, VEC, KEEP)
    if NREP_EFF > 5:
        _soma_store(Y, NOUT, TH, GA, g5 * rstd, 5, N, T, VEC, KEEP)
    if NREP_EFF > 6:
        _soma_store(Y, NOUT, TH, GA, g6 * rstd, 6, N, T, VEC, KEEP)
    if NREP_EFF > 7:
        _soma_store(Y, NOUT, TH, GA, g7 * rstd, 7, N, T, VEC, KEEP)
    if KEEP:                                             # after the row's stores (between the passes it costs 9 %)
        tl.store(RSTD + row, rstd)


# ------------------------------------------------------------------------------------------------ launchers
def _launch(y2: torch.Tensor, z2: torch.Tensor, theta: torch.Tensor, gamma: torch.Tensor, eps: float, keep: bool):
    """one launch of the row kernel (operands checked by the caller): the read-out (M, N) bf16 contiguous and, with
    ``keep``, the norm n (M, N) bf16 it was formed from and the rows' rstd (M,) fp32 (else None, None)"""
    M, N = (int(s) for s in y2.shape)
    out = torch.empty((M, N), dtype=torch.bfloat16, device=y2.device)
    n_out = torch.empty((M, N), dtype=torch.bfloat16, device=y2.device) if keep else None
    rstd = torch.empty((M,), dtype=torch.float32, device=y2.device) if keep else None
    with torch.cuda.device(y2.device.index):
        _gated_norm_soma_kernel[(M,)](y2, z2, out, n_out if keep else out, rstd if keep else theta, theta, gamma,
                                      y2.stride(0), z2.stride(0), out.stride(0), N, float(eps), VEC=VEC, T=THREADS,
                                      NREP_EFF=triton.cdiv(N, THREADS * VEC), KEEP=bool(keep), num_warps=NUM_WARPS)
    return out, n_out, rstd


def _gated_norm_soma_fwd(y2: torch.Tensor, z2: torch.Tensor, theta: torch.Tensor, gamma: torch.Tensor,
                         eps: float, store_n: bool = False):
    """the read-out of fla ``layer_norm_fwd(y2, ones, None, eps, z=z2, group_size=N, norm_before_gate=False,
    is_rms_norm=True)[0]`` for the reproduced domain (operands checked by the caller); a contiguous (M, N) bf16 output,
    and with ``store_n`` also the norm n (M, N) bf16 that the read-out was formed from (for a stage-wise comparison)."""
    out, n_out, _ = _launch(y2, z2, theta, gamma, eps, bool(store_n))
    return (out, n_out) if store_n else out


def gated_norm_soma_fwd(x: torch.Tensor, eps: float, z: torch.Tensor, theta: torch.Tensor, gamma: torch.Tensor):
    """(x_soma (M, N) bf16, rstd (M,) fp32, n (M, N) bf16): the training forward of the gated norm + soma read-out,
    internal to :class:`neuronspark.ops.dendritic_combined.DendriticChunkScanSomaFn` -- the kernel of
    :func:`dendritic_gated_norm_soma` with n and rstd stored for the backward
    (:func:`neuronspark.ops.triton.dendritic_soma.gated_norm_soma_bwd`).  x, z (M, N) bf16 rows with unit last
    strides (z the strided gate slice of the projection; row offsets of any size), M >= 1, N <= 16384; theta / gamma
    (N,) fp32 contiguous.  No autograd."""
    require_cuda(x, "gated_norm_soma_fwd")
    if (x.dim() != 2 or x.dtype != torch.bfloat16 or x.stride(1) != 1 or not isinstance(z, torch.Tensor)
            or z.device != x.device or tuple(z.shape) != tuple(x.shape) or z.dtype != torch.bfloat16
            or z.stride(1) != 1):
        raise RuntimeError("neuronspark.gated_norm_soma_fwd: gated norm operands outside the supported form (bf16 "
                           "(M, N) rows of one shape on one device, unit last strides)")
    M, N = (int(s) for s in x.shape)
    if M < 1 or not 1 <= N <= BLOCK_N:
        raise RuntimeError(f"neuronspark.gated_norm_soma_fwd: rows M >= 1 and 1 <= N <= {BLOCK_N} expected, got "
                           f"({M}, {N})")
    for name, t in (("theta", theta), ("gamma", gamma)):
        if (t.dtype != torch.float32 or t.dim() != 1 or int(t.shape[0]) != N or not t.is_contiguous()
                or t.device != x.device):
            raise RuntimeError(f"neuronspark.gated_norm_soma_fwd: {name} must be a contiguous float32 ({N},) tensor "
                               f"on {x.device}, got {t.dtype} {tuple(t.shape)} {t.device}")
    xo, n, rstd = _launch(x, z, theta, gamma, eps, True)
    return xo, rstd, n


@torch.library.custom_op("neuronspark::dendritic_gated_norm_soma", mutates_args=(),
                         schema="(Tensor y, Tensor z, Tensor theta, Tensor gamma, float eps) -> Tensor")
def _dendritic_gated_norm_soma_op(y, z, theta, gamma, eps):
    """x (M, N) bf16 contiguous"""
    return _gated_norm_soma_fwd(y, z, theta, gamma, eps)


@_dendritic_gated_norm_soma_op.register_fake
def _dendritic_gated_norm_soma_fake(y, z, theta, gamma, eps):
    return torch.empty(tuple(y.shape), dtype=torch.bfloat16, device=y.device)


# ------------------------------------------------------------------------------------------------ public function
def _fail(msg: str) -> None:
    raise RuntimeError(f"neuronspark.dendritic_gated_norm_soma: {msg}")


def dendritic_gated_norm_soma(y: torch.Tensor, z: torch.Tensor, theta: torch.Tensor, gamma: torch.Tensor,
                       eps: float) -> torch.Tensor:
    """``x = bf16(S(bf16(rmsnorm(y * silu(z))), theta) * gamma)`` with unit norm weight (see the module docstring): y, z
    (M, N) bf16 with unit last strides (z may be the strided gate slice of the projection), theta / gamma (N,) fp32
    -> x (M, N) bf16 contiguous.  No autograd."""
    for name, t in (("y", y), ("z", z), ("theta", theta), ("gamma", gamma)):
        if not isinstance(t, torch.Tensor):
            _fail(f"{name} must be a tensor, got {type(t).__name__}")
    require_cuda(y, "dendritic_gated_norm_soma")
    if z.device != y.device:
        _fail(f"z is on {z.device}, y on {y.device}")
    if y.dim() != 2 or tuple(z.shape) != tuple(y.shape):
        _fail(f"y and z must be (M, N) of one shape, got {tuple(y.shape)} {tuple(z.shape)}")
    if y.dtype != torch.bfloat16 or z.dtype != torch.bfloat16:
        _fail(f"y and z must be bf16, got {y.dtype} {z.dtype}")
    M, N = (int(s) for s in y.shape)
    if M < 1 or not MIN_N_EXCLUSIVE < N <= BLOCK_N:
        _fail(f"rows M >= 1 and {MIN_N_EXCLUSIVE} < N <= {BLOCK_N} expected, got ({M}, {N})")
    if y.stride(1) != 1 or z.stride(1) != 1:
        _fail(f"unit last strides expected, got y {tuple(y.stride())} z {tuple(z.stride())}")
    if (M - 1) * max(int(y.stride(0)), int(z.stride(0)), N) + N >= 2 ** 31:
        _fail(f"row offsets of ({M}, {N}) with strides {tuple(y.stride())} / {tuple(z.stride())} exceed 2^31 elements")
    for name, t in (("theta", theta), ("gamma", gamma)):
        if t.device != y.device or t.dtype != torch.float32 or t.dim() != 1 or int(t.shape[0]) != N or not t.is_contiguous():
            _fail(f"{name} must be a contiguous float32 ({N},) tensor on {y.device}, got {t.dtype} {tuple(t.shape)}")
    if torch.is_grad_enabled() and (y.requires_grad or z.requires_grad):
        _fail("an inference operator (no autograd): call it under torch.no_grad()")
    return torch.ops.neuronspark.dendritic_gated_norm_soma(y, z, theta, gamma, float(eps))


# ------------------------------------------------------------------------------------------------ reference
def dendritic_gated_rmsnorm_ref(y: torch.Tensor, z: torch.Tensor, eps: float) -> torch.Tensor:
    """Pure torch, fp32 (fla ``layer_norm_fwd`` with ``norm_before_gate=False``, RMS, unit weight, one group):
    ``g = y * (z * sigmoid(z))``, ``rstd = 1 / sqrt(sum(g^2) / N + eps)``, ``bf16(g * rstd)``; contiguous."""
    g = y.float()
    zf = z.float()
    g = g * (zf * torch.sigmoid(zf))
    var = (g * g).sum(dim=-1, keepdim=True) / int(g.shape[-1])
    rstd = 1.0 / torch.sqrt(var + float(eps))
    return (g * rstd).to(torch.bfloat16).contiguous()


def dendritic_gated_norm_soma_ref(y: torch.Tensor, z: torch.Tensor, theta: torch.Tensor, gamma: torch.Tensor,
                           eps: float) -> torch.Tensor:
    """Pure torch: the soma read-out (:func:`neuronspark.ops.triton.dendritic_soma.dendritic_soma_readout_ref`) of
    :func:`dendritic_gated_rmsnorm_ref`."""
    return dendritic_soma_readout_ref(dendritic_gated_rmsnorm_ref(y, z, eps), theta, gamma).contiguous()
