"""dendritic-layer chunk scan kernels: the chunk-scan forward with TMA loads (inference prefill calls that end in a
partial chunk; the arithmetic reference of the Gluon forward), the ddA assembly of the training backward (its
intra-chunk pieces are the Gluon :mod:`neuronspark.ops.gluon.dendritic_dda`) and the float64 references of the chunk
scan's forward and input backward (the training kernels of both are Gluon:
:mod:`neuronspark.ops.gluon.dendritic_chunk_scan_fwd` / :mod:`neuronspark.ops.gluon.dendritic_chunk_scan_dx`).

Internal pieces (not registry ops), all for the dendritic-layer call of the SSD scan (ngroups 1, D of shape (nheads,),
no z, no seq_idx, dstate <= 128, headdim a power of two):

* ``chunk_scan_fwd_tma(cb, x, dt, dA_cumsum, C, states, D) -> out`` (b, l, h, P) in x's dtype, per chunk::

      out[m] = exp(dA_m) C_m states^T + sum_{k <= m} bf16(CB[m, k] exp(min(dA_m - dA_k, 0)) dt_k) x_k + D x_m

  rounded once to x's dtype; C[m] (1, BM, N), states_h (P, N), CB[m, k] (BM, BK) and x_h[k] (1, BK, P) are loaded
  through TMA descriptors, the k tiles run up to the diagonal under the causal mask.
* ``dda_asm(v) -> ddA_intra`` (b, h, c, chunk) fp32: ddA[k] = sum_{m >= k} sum_{n < k} T[m, n] from the V grid of the
  Gluon dda kernel.

Numerics: one bf16 rounding of the weighted CB tile before its dot (fp32 accumulation), fp32 elementwise; the ddA
partial sums are fp32-rounding class.  Checked with the per-operator tolerance rule of :mod:`neuronspark.utils.testing`
against the references below.

Launch constants: forward (BM 64, BK 32, 4 warps, 3 stages); the ddA assembly 1 warp over the (BS 32, NB 8) V grid.
TMA descriptors: sm_90+.
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F
import triton
import triton.language as tl
from triton.tools.tensor_descriptor import TensorDescriptor

from neuronspark.ops.triton import require_cuda
from neuronspark.ops.triton.dendritic_states import _chunked, _desc3, _ex2ftz_ref, _exp_f32

__all__ = [
    "FWD_CFG", "DDA_BS", "DDA_NB", "DDA_ASM_WARPS",
    "chunk_scan_fwd_tma", "dda_asm",
    "chunk_scan_fwd_tma_ref", "chunk_scan_dx_ref", "dda_ref", "ssd_forward_ref", "softplus_ref",
]

# ------------------------------------------------------------------------------------------------ launch constants
FWD_CFG = (64, 32, 4, 3)            # BM, BK, num_warps, num_stages
DDA_BS, DDA_NB = 32, 8              # the V grid of the assembly: sub-tile edge, sub-tiles per chunk side (chunk = BS NB)
DDA_ASM_WARPS = 1



# ------------------------------------------------------------------------------------------------ kernels
@triton.jit
def _dda_asm_kernel(V, OUT, svb, svh, svc, sob, soh, soc, BS: tl.constexpr, NB: tl.constexpr):
    """Per (head, chunk, batch), V layout tile = I * NB + J: every load issued first (the (NB, NB, BS) row-sum and
    column-sum grids of the lower tiles), one (NB, BS) store; for k in block K,
        ddA[k] = suffix(RS_K)[k] - suffix(cs(K, K))[k] + exclusive_prefix(sum_{I > K} cs(I, K))[k]
                 + sum_{I > K > J} total(I, J),
    RS_K = sum_{J <= K} rs(K, J), cs(K, K) the diagonal tile's column sums, suffix(v)[k] = sum_{j >= k} v[j];
    = sum_{m >= k} sum_{n < k} T[m, n] (the intra-chunk ddA)."""
    h = tl.program_id(0)
    c = tl.program_id(1)
    b = tl.program_id(2)
    r = tl.arange(0, BS)
    kb = tl.arange(0, NB)
    i3 = kb[:, None, None]
    j3 = kb[None, :, None]
    low = (j3 <= i3) & (r[None, None, :] >= 0)
    gb = V + b * svb + h * svh + c * svc + (i3 * NB + j3) * (2 * BS) + r[None, None, :]
    R3 = tl.load(gb, mask=low, other=0.)                       # rs(I, J)[m]
    C3 = tl.load(gb + BS, mask=low, other=0.)                  # cs(I, J)[n]
    o2 = kb[:, None] * BS + r[None, :]
    RS = tl.sum(R3, axis=1)                                    # (K = I, m): rows of block K over all n < m
    csd = tl.sum(tl.where(i3 == j3, C3, 0.), axis=0)           # (K = J, n): the diagonal tile's column sums
    CSb = tl.sum(tl.where(i3 > j3, C3, 0.), axis=0)            # (K = J, n): column sums of the tiles below
    tot = tl.sum(R3, axis=2)                                   # (I, J) tile totals
    k3 = kb[:, None, None]
    strad = tl.sum(tl.sum(tl.where((j3 > k3) & (k3 > kb[None, None, :]), tot[None, :, :], 0.), axis=2), axis=1)
    o = (tl.cumsum(RS, axis=1, reverse=True) - tl.cumsum(csd, axis=1, reverse=True)
         + (tl.cumsum(CSb, axis=1) - CSb) + strad[:, None])
    tl.store(OUT + b * sob + h * soh + c * soc + o2, o)


@triton.jit
def _fwd_kernel(CBD, XD, X, OUT, DT, DA, CD, SD, D,
                CS, T, NB, NC,
                sxb, sxl, sxh, sob, sol, soh, stb, sth, stc, sab, sah, sac,
                H: tl.constexpr, P: tl.constexpr, N: tl.constexpr, BM: tl.constexpr, BK: tl.constexpr):
    # program: rows [r0, r0 + BM) of chunk c of batch b, head h -- the inter-chunk read of the entering state plus the
    # causal intra-chunk sum, the D skip, one rounding to x's dtype
    b = tl.program_id(1) % NB
    c = tl.program_id(1) // NB
    h = tl.program_id(2)
    r0 = tl.program_id(0) * BM
    rows = r0 + tl.arange(0, BM)
    cols = tl.arange(0, P)
    t0 = c * CS
    n_valid = tl.minimum(CS, T - t0)
    da_base = DA + b * sab + c * sac + h * sah
    dt_base = DT + b * stb + c * stc + h * sth
    da_r = tl.load(da_base + rows, mask=rows < CS, other=0.0).to(tl.float32)
    # inter-chunk: exp(dA_r) C_r . S_h
    c_rows = CD.load([b, t0 + r0, 0]).reshape(BM, N)
    s_h = SD.load([((b * NC + c) * H + h) * P, 0])
    acc = tl.dot(c_rows, tl.trans(s_h)) * tl.exp(da_r)[:, None]
    # intra-chunk: sum over source positions k <= r of bf16(CB[r, k] exp(min(dA_r - dA_k, 0)) dt_k) x_k
    kk = tl.arange(0, BK)
    cb_row = (b * NC + c) * CS + r0
    k_end = tl.minimum(r0 + BM, n_valid)
    for k0 in range(0, k_end, BK):
        w = CBD.load([cb_row, k0])
        da_k = tl.load(da_base + k0 + kk, mask=kk < CS - k0, other=0.0).to(tl.float32)
        dt_k = tl.load(dt_base + k0 + kk, mask=kk < CS - k0, other=0.0).to(tl.float32)
        w = w * tl.exp(tl.minimum(da_r[:, None] - da_k[None, :], 0.0))
        w = w * dt_k[None, :]
        w = tl.where(rows[:, None] >= (k0 + kk)[None, :], w, 0.0).to(X.dtype.element_ty)
        x_k = XD.load([b, t0 + k0, h * P]).reshape(BK, P)
        acc += tl.dot(w, x_k)
    live = (rows[:, None] < n_valid) & (cols[None, :] < P)
    x_r = tl.load(X + b * sxb + (t0 + rows)[:, None] * sxl + h * sxh + cols[None, :], mask=live,
                  other=0.0).to(tl.float32)
    acc += x_r * tl.load(D + h).to(tl.float32)
    tl.store(OUT + b * sob + (t0 + rows)[:, None] * sol + h * soh + cols[None, :], acc, mask=live)


# ------------------------------------------------------------------------------------------------ launchers
def dda_asm(v):
    """The intra-chunk ddA (batch, nheads, nchunks, chunk) fp32 from the V grid of ``gluon.dendritic_dda.dda_gluon``."""
    BS, NB = DDA_BS, DDA_NB
    batch, nheads, nchunks, _ = v.shape
    out = torch.empty((batch, nheads, nchunks, BS * NB), dtype=torch.float32, device=v.device)
    with torch.cuda.device(v.device.index):
        _dda_asm_kernel[(nheads, nchunks, batch)](
            v, out, v.stride(0), v.stride(1), v.stride(2), out.stride(0), out.stride(1), out.stride(2),
            BS=BS, NB=NB, num_warps=DDA_ASM_WARPS)
    return out


def chunk_scan_fwd_tma(cb, x, dt, dA_cumsum, C, states, D):
    """out (batch, seqlen, nheads, headdim) in x's dtype: the chunk-scan forward of the module docstring for the
    dendritic-layer call (ngroups 1, D (nheads,), no gate). cb (b, nc, 1, chunk, chunk) fp32
    contiguous; x (b, l, h, P) with a contiguous (h, P) part; dt / dA_cumsum (b, h, nc, chunk) with unit last stride; C
    (b, l, 1, N) with unit last stride; states (b, nc, h, P, N) contiguous in C's dtype; D (h,) contiguous. Partial last
    chunks are
    supported (rows past seqlen are the descriptors' zero fill and are not stored)."""
    require_cuda(x, "chunk_scan_fwd_tma")
    batch, seqlen, nheads, headdim = x.shape
    _, _, nchunks, chunk_size = dt.shape
    _, _, ngroups, dstate = C.shape
    BM, BK, nw, st = FWD_CFG
    if ngroups != 1 or dstate > 128 or headdim not in (16, 32, 64, 128):
        raise RuntimeError(f"neuronspark.chunk_scan_fwd_tma: ngroups {ngroups} dstate {dstate} headdim {headdim} "
                           "(dendritic-layer call: 1 / <= 128 / pow2)")
    if C.shape != (batch, seqlen, ngroups, dstate):
        raise RuntimeError("neuronspark.chunk_scan_fwd_tma: C shape")
    if cb.shape != (batch, nchunks, ngroups, chunk_size, chunk_size) or cb.dtype != torch.float32:
        raise RuntimeError("neuronspark.chunk_scan_fwd_tma: cb must be fp32 (batch, nchunks, 1, chunk, chunk)")
    if dt.shape != (batch, nheads, nchunks, chunk_size) or dA_cumsum.shape != dt.shape:
        raise RuntimeError("neuronspark.chunk_scan_fwd_tma: dt / dA_cumsum shapes")
    if states.shape != (batch, nchunks, nheads, headdim, dstate):
        raise RuntimeError("neuronspark.chunk_scan_fwd_tma: states shape")
    if states.dtype != C.dtype or D is None or D.dim() != 1 or D.stride(-1) != 1:
        raise RuntimeError("neuronspark.chunk_scan_fwd_tma: states in C's dtype and D (nheads,) contiguous expected "
                           "(the dendritic-layer call)")
    if (chunk_size % BM) or (BM % BK) or C.stride(-1) != 1 or not cb.is_contiguous() or not states.is_contiguous():
        raise RuntimeError(f"neuronspark.chunk_scan_fwd_tma: chunk {chunk_size} BM {BM} BK {BK} or operand layouts not "
                           "supported")
    if dt.stride(-1) != 1 or dA_cumsum.stride(-1) != 1 or x.stride(-1) != 1:
        raise RuntimeError("neuronspark.chunk_scan_fwd_tma: unit last strides of dt / dA_cumsum / x expected")
    out = torch.empty(batch, seqlen, nheads, headdim, device=x.device, dtype=x.dtype)
    cd = TensorDescriptor(C, [batch, seqlen, dstate], [C.stride(0), C.stride(1), 1], [1, BM, dstate])
    sd = TensorDescriptor(states, [batch * nchunks * nheads * headdim, dstate], [dstate, 1], [headdim, dstate])
    cbd = TensorDescriptor(cb, [batch * nchunks * chunk_size, chunk_size], [chunk_size, 1], [BM, BK])
    xd = _desc3(x, BK)
    grid = (triton.cdiv(chunk_size, BM), batch * nchunks, nheads)
    with torch.cuda.device(x.device.index):
        _fwd_kernel[grid](
            cbd, xd, x, out, dt, dA_cumsum, cd, sd, D,
            chunk_size, seqlen, batch, nchunks,
            x.stride(0), x.stride(1), x.stride(2), out.stride(0), out.stride(1), out.stride(2),
            dt.stride(0), dt.stride(1), dt.stride(2), dA_cumsum.stride(0), dA_cumsum.stride(1), dA_cumsum.stride(2),
            H=nheads, P=headdim, N=dstate, BM=BM, BK=BK, num_warps=nw, num_stages=st)
    return out


# ------------------------------------------------------------------------------------------------ references
# Pure torch.  The piece references (``chunk_scan_fwd_tma_ref``, ``chunk_scan_dx_ref``, ``dda_ref``) evaluate the
# chunked scan in float64 between the kernels' rounding points, which they reproduce: the masked fp32 operand tile that
# a kernel rounds to bf16 before its dot is evaluated in fp32 with the kernel's expression order (the decay ``tl.exp``
# as ``exp2(t * log2 e)``, see ``dendritic_states._exp_f32``) and rounded the same way.  ``ssd_forward_ref`` is the
# float32 differentiable whole-forward math without the kernels' internal roundings.
def _tril(chunk: int, device) -> torch.Tensor:
    i = torch.arange(chunk, device=device)
    return i[:, None] >= i[None, :]


def chunk_scan_fwd_tma_ref(cb, x, dt, dA_cumsum, C, states, D):
    """Reference of ``chunk_scan_fwd_tma`` (``_fwd_kernel``'s expressions): out[m] = exp(dA_m) C_m states_c^T
    + sum_{k <= m} P[m, k] x_k + D x_m with the dot operand P = bf16(where(m >= k, (CB[m, k] exp(min(dA_m - dA_k, 0)))
    dt_k, 0)) formed in fp32 as the kernel forms it (``cb *= exp(..)``, ``cb *= dt_k``, mask, cast to x's dtype);
    float64 sums; the result in x's dtype."""
    batch, seqlen, nheads, headdim = x.shape
    _, _, nchunks, chunk = dt.shape
    xc = _chunked(x, nchunks, chunk)                                   # (b, c, l, h, p)
    Cc = _chunked(C[:, :, 0], nchunks, chunk)                          # (b, c, l, n)
    dA = dA_cumsum.float()                                             # (b, h, c, l)
    y_off = torch.einsum("bcln,bchpn,bhcl->bclhp", Cc, states.double(), _exp_f32(dA).double())
    seg = torch.clamp_max(dA[..., :, None] - dA[..., None, :], 0.0)    # (b, h, c, m, k) fp32: min(dA_m - dA_k, 0)
    Pm = cb[:, :, 0].float()[:, :, None] * _exp_f32(seg).permute(0, 2, 1, 3, 4)      # (b, c, h, m, k) fp32
    Pm = Pm * dt.float().permute(0, 2, 1, 3)[:, :, :, None, :]                        # * dt_k
    Pm = torch.where(_tril(chunk, x.device), Pm, torch.zeros_like(Pm)).to(x.dtype).double()
    y = y_off + torch.einsum("bchls,bcshp->bclhp", Pm, xc) + D.double()[None, None, None, :, None] * xc
    return y.reshape(batch, nchunks * chunk, nheads, headdim)[:, :seqlen].to(x.dtype)


def chunk_scan_dx_ref(x, dt, dA_cumsum, B, CB, dout, dstates, D, dx):
    """Reference of the chunk-scan input backward
    (:func:`neuronspark.ops.gluon.dendritic_chunk_scan_dx.chunk_scan_dx_gluon`, with the FTZ decay): acc[m] =
    exp(min(dA_last - dA_m, 0)) B_m dstates^T + sum_{k >= m} K[k, m] dout_k with the dot operand K = bf16(where(k >= m,
    CB[k, m] ex2ftz(min(dA_k - dA_m, 0) log2 e), 0)) formed in fp32 as the kernel forms it (cast to dout's dtype); dx =
    acc dt_m + D dout_m; ddt[m] = sum_p acc x_m; dD = sum dout x; float64 sums.  Returns (dx in x's dtype, ddt (b, h, c,
    l) fp32, dD (h,) fp32); ``dx`` (the output buffer argument of the kernel launcher) is ignored."""
    batch, seqlen, nheads, headdim = x.shape
    _, _, nchunks, chunk = dt.shape
    xc = _chunked(x, nchunks, chunk)
    do = _chunked(dout, nchunks, chunk)
    Bc = _chunked(B[:, :, 0], nchunks, chunk)
    dA = dA_cumsum.float()
    sc = _exp_f32(torch.clamp_max(dA[..., -1:] - dA, 0.0)).double()   # (b, h, c, l)
    acc = torch.einsum("bcln,bchpn,bhcl->bclhp", Bc, dstates.double(), sc)
    seg = torch.clamp_max(dA[..., :, None] - dA[..., None, :], 0.0)    # (b, h, c, k, m) fp32: min(dA_k - dA_m, 0)
    K = CB[:, :, 0].float()[:, :, None] * _ex2ftz_ref(seg).permute(0, 2, 1, 3, 4)    # (b, c, h, k, m) fp32
    K = torch.where(_tril(chunk, x.device), K, torch.zeros_like(K)).to(dout.dtype).double()
    acc = acc + torch.einsum("bchkm,bckhp->bcmhp", K, do)
    dtc = dt.double().permute(0, 2, 3, 1)                              # (b, c, l, h)
    dxr = acc * dtc[..., None] + do * D.double()[None, None, None, :, None]
    ddt = (acc * xc).sum(-1).permute(0, 3, 1, 2)                       # (b, h, c, l)
    dD = (do * xc).sum((0, 1, 2, 4))
    dxr = dxr.reshape(batch, nchunks * chunk, nheads, headdim)[:, :seqlen]
    return dxr.to(x.dtype), ddt.float(), dD.float()


def dda_ref(x, dt, dA_cumsum, dout, cb):
    """float64 reference of ``gluon.dendritic_dda.dda_gluon`` + ``dda_asm``: g_h[m, n] = (dout_h[m] . x_h[n]) dt_n
    exp(min(dA_m - dA_n, 0)); dCB[m, n] = sum_h g_h[m, n] for n <= m (0 above); ddA_intra[k] = sum_{m >= k} sum_{n <
    k} g_h[m, n] CB[m, n] (n < m).
    Returns (dCB (b, c, 1, chunk, chunk) fp32, ddA_intra (b, h, c, chunk) fp32)."""
    batch, seqlen, nheads, headdim = x.shape
    _, _, nchunks, chunk = dt.shape
    xc = _chunked(x, nchunks, chunk)
    do = _chunked(dout, nchunks, chunk)
    dA = dA_cumsum.double()
    tril = _tril(chunk, x.device)
    g = torch.einsum("bcmhp,bcnhp->bhcmn", do, xc)
    g = g * dt.double()[..., None, :] * torch.exp(torch.clamp_max(dA[..., :, None] - dA[..., None, :], 0.0))
    dcb = (g * tril).sum(1)[:, :, None]                                # (b, c, 1, m, n)
    strict = tril & ~torch.eye(chunk, dtype=torch.bool, device=x.device)
    G = g * cb[:, :, 0].double()[:, None] * strict                     # (b, h, c, m, n)
    Q = torch.cumsum(G, dim=-1) - G                                    # Q[m, k] = sum_{n < k} G[m, n]
    dda = (Q * tril).sum(-2)                                           # sum over m >= k
    return dcb.float(), dda.float()


def softplus_ref(v: torch.Tensor) -> torch.Tensor:
    """The write-strength softplus of the kernels: ``where(v <= 20, log(exp(v) + 1), v)``; the exponential is taken of
    ``min(v, 20)`` so that the unselected branch stays finite under autograd (the selected values are unchanged)."""
    return torch.where(v <= 20.0, torch.log(torch.exp(torch.clamp_max(v, 20.0)) + 1.0), v)


def ssd_forward_ref(x, dt_raw, A, B, C, chunk_size, D, dt_bias):
    """Pure-torch float32 chunked-scan forward of the dendritic-layer call (differentiable): dt = softplus(dt_raw +
    dt_bias) (kernel form), dA
    = dt A per step, chunked: intra-chunk y_diag[l] = sum_{s <= l} (C_l . B_s) exp(dA_cs[l] - dA_cs[s]) dt_s x_s, chunk
    states and their passing, the inter-chunk term exp(dA_cs[l]) C_l s_c^T, and D x. x (b, l, h, P), dt_raw (b, l, h), A
    / D / dt_bias (h,),
    B / C (b, l, 1, N); returns y (b, l, h, P) in x's dtype."""
    batch, seqlen, nheads, headdim = x.shape
    nchunks = math.ceil(seqlen / chunk_size)
    L = nchunks * chunk_size

    def pad(t):
        if int(t.shape[1]) == L:
            return t
        return F.pad(t, [0, 0] * (t.dim() - 2) + [0, L - int(t.shape[1])])

    dt = softplus_ref(dt_raw.float() + dt_bias.float())                # (b, l, h)
    dt = pad(dt).reshape(batch, nchunks, chunk_size, nheads)           # padded steps: dt 0
    xc = pad(x.float()).reshape(batch, nchunks, chunk_size, nheads, headdim)
    Bc = pad(B[:, :, 0].float()).reshape(batch, nchunks, chunk_size, -1)
    Cc = pad(C[:, :, 0].float()).reshape(batch, nchunks, chunk_size, -1)
    dA = dt * A.float()                                                # (b, c, l, h)
    dA_cs = torch.cumsum(dA, dim=2).permute(0, 3, 1, 2)                # (b, h, c, l)
    CB = torch.einsum("bcln,bcsn->bcls", Cc, Bc)
    tril = _tril(chunk_size, x.device)
    decay = torch.exp(torch.clamp_max(dA_cs[..., :, None] - dA_cs[..., None, :], 0.0)) * tril     # (b, h, c, l, s)
    M = CB[:, :, None] * decay.permute(0, 2, 1, 3, 4) * dt.permute(0, 1, 3, 2)[:, :, :, None, :]  # (b, c, h, l, s)
    y = torch.einsum("bchls,bcshp->bclhp", M, xc)
    w = torch.exp(torch.clamp_max(dA_cs[..., -1:] - dA_cs, 0.0)) * dt.permute(0, 3, 1, 2)       # (b, h, c, s)
    S = torch.einsum("bhcs,bcshp,bcsn->bchpn", w, xc, Bc)
    s = torch.zeros_like(S[:, 0])
    prev = []
    for c in range(nchunks):
        prev.append(s)
        s = torch.exp(dA_cs[:, :, c, -1])[..., None, None] * s + S[:, c]
    prev = torch.stack(prev, dim=1)                                    # (b, c, h, p, n)
    y = y + torch.einsum("bcln,bchpn,bhcl->bclhp", Cc, prev, torch.exp(dA_cs))
    y = y + D.float()[None, None, None, :, None] * xc
    return y.reshape(batch, L, nheads, headdim)[:, :seqlen].to(x.dtype)
