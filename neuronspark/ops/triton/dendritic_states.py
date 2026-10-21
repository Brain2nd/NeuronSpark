"""dendritic-layer chunk states: the dB / dC pieces of the SSD scan's training backward, and the pure-torch references
of the chunk states + state passing (the forward pair runs the Gluon kernels of
:mod:`neuronspark.ops.gluon.dendritic_chunk_states`, whose references ``csp_fused_ref`` / ``csp_prefill_ref`` are here;
the state gradients run :mod:`neuronspark.ops.gluon.dendritic_dstates`).

Internal pieces (not registry ops); ``db_tma`` and ``dc_tma`` are used by
:class:`neuronspark.ops.dendritic_combined.DendriticChunkScanSomaFn`.  For chunk c, position m of the chunk and state
column n (ngroups 1):

* ``db_tma(x, dt, dA_cumsum, dstates, B, ngroups) -> (dB parts, ddA_next workspace, n tiles)``::

      d_h[m, n]  = (x_h[m] . dstates_h[:, n]) exp(min(dA_h[last] - dA_h[m], 0)) dt_h[m]      (dot fp32, then the scale)
      dB[m, n]   = sum_h d_h[m, n]                     one partial per group of DB_HEADS heads
      ws_h[m+1]  = sum_n d_h[m, n] B[m, n]             per n tile (m < cs - 1; slot 0 is not written)

* ``dc_tma(prev_states, dA_cumsum, dout, C, ngroups) -> (dC parts, ddA_prev workspace, n tiles)``::

      e_h[m, n]  = (dout_h[m] . prev_states_h[:, n]) exp(dA_h[m])
      dC[m, n]   = sum_h e_h[m, n]                     one partial per group of DC_HEADS heads
      ws_h[m]    = sum_n e_h[m, n] C[m, n]             per n tile

  The group partials are summed by :func:`neuronspark.ops.triton.dendritic_glue.splits_sum`, the workspace tiles by
  :func:`neuronspark.ops.triton.dendritic_glue.ddA_combine` (fixed head groups: the partition does not depend on the
  device).  x_h / dout_h row blocks and the dstates_h / prev_states_h boxes are loaded through TMA descriptors (rows
  past the sequence end read as zeros).

Numerics: one bf16 dot per head and tile with fp32 accumulation, the fp32 row scale applied to the dot result, fp32
sums over heads in head order.  Checked with the per-operator tolerance rule of :mod:`neuronspark.utils.testing`
against the references below.

Launch constants: db and dc (BM 64, BN 64, 4 warps), DB_HEADS / DC_HEADS heads per program.  The TMA descriptor kernels
need a GPU with TMA (sm_90+); the package targets Blackwell (sm_100+).
"""
from __future__ import annotations

import math

import torch
import triton
import triton.language as tl
from triton.tools.tensor_descriptor import TensorDescriptor

from neuronspark.ops.triton import require_cuda

#: state columns per slot of the deterministic ddA tile workspaces of db_tma / dc_tma
WS_TILE_N = 32

__all__ = [
    "DB_CFG", "DC_CFG",
    "db_tma", "dc_tma",
    "csp_fused_ref", "csp_prefill_ref", "db_tma_ref", "dc_tma_ref",
]

# ------------------------------------------------------------------------------------------------ launch constants
DB_CFG = (64, 64, 4, 3)        # BM, BN, num_warps, num_stages
DC_CFG = (64, 64, 4, 3)        # BM, BN, num_warps, num_stages
DB_HEADS = 32                  # heads per program of db_tma (one dB partial per group)
DC_HEADS = 32                  # heads per program of dc_tma (one dC partial per group)


# ------------------------------------------------------------------------------------------------ kernels
@triton.jit
def _db_kernel(XD, SD, B, DT, DA, DB, WS, CS, NS, T, NB, NC, H,
               s_b_b, s_b_t, s_b_n, s_dt_b, s_dt_h, s_dt_c, s_da_b, s_da_h, s_da_c,
               s_o_b, s_o_t, s_o_g, s_w_b, s_w_h, s_w_c, s_w_l, s_w_t,
               P: tl.constexpr, HG: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr):
    # program: an (m, n) tile of chunk c of batch b over the heads of group grp
    n_ntile = NS // BN
    mt = tl.program_id(0) // n_ntile
    nt = tl.program_id(0) % n_ntile
    bi = tl.program_id(1) % NB                    # int32 descriptor coordinates; int64 pointer offsets
    ci = tl.program_id(1) // NB
    b = bi.to(tl.int64)
    c = ci.to(tl.int64)
    grp = tl.program_id(2)
    t0 = c * CS
    m = mt * BM + tl.arange(0, BM)
    n = nt * BN + tl.arange(0, BN)
    live = m < tl.minimum(CS, T - t0)
    bv = tl.load(B + b * s_b_b + (t0 + m)[:, None] * s_b_t + n[None, :] * s_b_n, mask=live[:, None],
                 other=0.0).to(tl.float32)
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    h_end = tl.minimum(grp * HG + HG, H)
    for h in range(grp * HG, h_end):
        xt = XD.load([bi, ci * CS + mt * BM, h * P]).reshape(BM, P)
        st = SD.load([((bi * NC + ci) * H + h) * P, nt * BN])
        d = tl.dot(xt, st)
        hoff = b * s_da_b + h * s_da_h + c * s_da_c
        last = tl.load(DA + hoff + (CS - 1)).to(tl.float32)
        da_m = tl.load(DA + hoff + m, mask=m < CS, other=0.0).to(tl.float32)
        dt_m = tl.load(DT + b * s_dt_b + h * s_dt_h + c * s_dt_c + m, mask=m < CS, other=0.0).to(tl.float32)
        d = d * (tl.exp(tl.minimum(last - da_m, 0.0)) * dt_m)[:, None]
        tl.store(WS + b * s_w_b + h * s_w_h + c * s_w_c + nt * s_w_t + (m + 1).to(tl.int64) * s_w_l,
                 tl.sum(d * bv, axis=1), mask=m < CS - 1)
        acc += d
    tl.store(DB + b * s_o_b + (t0 + m)[:, None] * s_o_t + grp * s_o_g + n[None, :], acc, mask=live[:, None])


@triton.jit
def _dc_kernel(OD, SD, C, DA, DC, WS, CS, NS, T, NB, NC, H,
               s_c_b, s_c_t, s_c_n, s_da_b, s_da_h, s_da_c,
               s_o_b, s_o_t, s_o_g, s_w_b, s_w_h, s_w_c, s_w_l, s_w_t,
               P: tl.constexpr, HG: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr):
    n_ntile = NS // BN
    mt = tl.program_id(0) // n_ntile
    nt = tl.program_id(0) % n_ntile
    bi = tl.program_id(1) % NB                    # int32 descriptor coordinates; int64 pointer offsets
    ci = tl.program_id(1) // NB
    b = bi.to(tl.int64)
    c = ci.to(tl.int64)
    grp = tl.program_id(2)
    t0 = c * CS
    m = mt * BM + tl.arange(0, BM)
    n = nt * BN + tl.arange(0, BN)
    live = m < tl.minimum(CS, T - t0)
    cv = tl.load(C + b * s_c_b + (t0 + m)[:, None] * s_c_t + n[None, :] * s_c_n, mask=live[:, None],
                 other=0.0).to(tl.float32)
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    h_end = tl.minimum(grp * HG + HG, H)
    for h in range(grp * HG, h_end):
        ot = OD.load([bi, ci * CS + mt * BM, h * P]).reshape(BM, P)
        st = SD.load([((bi * NC + ci) * H + h) * P, nt * BN])
        e = tl.dot(ot, st)
        da_m = tl.load(DA + b * s_da_b + h * s_da_h + c * s_da_c + m, mask=live, other=0.0).to(tl.float32)
        e = e * tl.exp(da_m)[:, None]
        tl.store(WS + b * s_w_b + h * s_w_h + c * s_w_c + nt * s_w_t + m.to(tl.int64) * s_w_l,
                 tl.sum(e * cv, axis=1), mask=m < CS)
        acc += e
    tl.store(DC + b * s_o_b + (t0 + m)[:, None] * s_o_t + grp * s_o_g + n[None, :], acc, mask=live[:, None])


# ------------------------------------------------------------------------------------------------ launchers
def _desc3(t4, block_rows):
    """(1, block_rows, P) boxes of a (batch, seqlen, H, P) view whose (H, P) part is contiguous: descriptor over
    (batch, seqlen, H * P) -- rows t >= seqlen read as zeros."""
    batch, seqlen, nh, p = t4.shape
    if t4.stride(3) != 1 or t4.stride(2) != p:
        raise RuntimeError(f"neuronspark.dendritic_states: (H, P) part of a {tuple(t4.shape)} view with strides "
                           f"{tuple(t4.stride())} is not contiguous")
    return TensorDescriptor(t4, [batch, seqlen, nh * p], [t4.stride(0), t4.stride(1), 1], [1, block_rows, p])


def _ws_tiles(dstate, BN):
    tile_count = math.ceil(dstate / WS_TILE_N)
    nvn = triton.cdiv(dstate, BN)
    if nvn > tile_count:
        raise RuntimeError(f"neuronspark.dendritic_states: {nvn} n tiles written, the workspace holds {tile_count}")
    return tile_count, nvn


def db_tma(x, dt, dA_cumsum, dstates, B, ngroups):
    """(dB parts (b, l, ceil(h / DB_HEADS), 1, n) fp32, ddA workspace (b, h, c, l, tiles) fp32, n written tiles)
    (module docstring).  Workspace slot l = 0 of each row is never written and never read."""
    require_cuda(x, "db_tma")
    batch, seqlen, nheads, headdim = x.shape
    _, _, nchunks, chunk_size = dt.shape
    dstate = dstates.shape[-1]
    BM, BN, nw, st = DB_CFG
    if ngroups != 1 or headdim not in (16, 32, 64, 128) or dstate > 256:
        raise RuntimeError(f"neuronspark.db_tma: ngroups {ngroups} headdim {headdim} dstate {dstate} (dendritic-layer "
                           "call: 1 / pow2 / <= 256)")
    if dt.shape != (batch, nheads, nchunks, chunk_size) or dA_cumsum.shape != dt.shape:
        raise RuntimeError("neuronspark.db_tma: dt / dA_cumsum shapes")
    if dstates.shape != (batch, nchunks, nheads, headdim, dstate) or B.shape != (batch, seqlen, ngroups, dstate):
        raise RuntimeError("neuronspark.db_tma: dstates / B shapes")
    if dstates.dtype != x.dtype:
        raise RuntimeError(f"neuronspark.db_tma: dstates {dstates.dtype} / x {x.dtype} must agree (one bf16 dot)")
    if not dstates.is_contiguous() or B.stride(-1) != 1 or x.stride(-1) != 1 or dt.stride(-1) != 1 \
            or dA_cumsum.stride(-1) != 1 or (chunk_size % BM) or (dstate % BN):
        raise RuntimeError(f"neuronspark.db_tma: operand layouts or chunk {chunk_size} BM {BM} dstate {dstate} BN {BN} "
                           "not supported")
    tile_count, nvn = _ws_tiles(dstate, BN)
    ws = torch.empty(batch, nheads, nchunks, chunk_size, tile_count, device=x.device, dtype=torch.float32)
    ngrp = triton.cdiv(nheads, DB_HEADS)
    dB = torch.empty(batch, seqlen, ngrp, ngroups, dstate, device=x.device, dtype=torch.float32)
    xd = _desc3(x, BM)
    sd = TensorDescriptor(dstates, [batch * nchunks * nheads * headdim, dstate], [dstate, 1], [headdim, BN])
    grid = (triton.cdiv(chunk_size, BM) * nvn, batch * nchunks, ngrp)
    with torch.cuda.device(x.device.index):
        _db_kernel[grid](
            xd, sd, B, dt, dA_cumsum, dB, ws, chunk_size, dstate, seqlen, batch, nchunks, nheads,
            B.stride(0), B.stride(1), B.stride(3), dt.stride(0), dt.stride(1), dt.stride(2),
            dA_cumsum.stride(0), dA_cumsum.stride(1), dA_cumsum.stride(2),
            dB.stride(0), dB.stride(1), dB.stride(2),
            ws.stride(0), ws.stride(1), ws.stride(2), ws.stride(3), ws.stride(4),
            P=headdim, HG=DB_HEADS, BM=BM, BN=BN, num_warps=nw, num_stages=st)
    return dB, ws, nvn


def dc_tma(prev_states, dA_cumsum, dout, C, ngroups):
    """(dC parts (b, l, ceil(h / DC_HEADS), 1, n) fp32, ddA_prev workspace (b, h, c, l, tiles) fp32, n written tiles)
    (module docstring)."""
    require_cuda(dout, "dc_tma")
    batch, nchunks, nheads, headdim, dstate = prev_states.shape
    _, seqlen, _, _ = dout.shape
    _, _, _, chunk_size = dA_cumsum.shape
    BM, BN, nw, st = DC_CFG
    if ngroups != 1 or headdim not in (16, 32, 64, 128) or dstate > 256:
        raise RuntimeError(f"neuronspark.dc_tma: ngroups {ngroups} headdim {headdim} dstate {dstate} (dendritic-layer "
                           "call: 1 / pow2 / <= 256)")
    if dA_cumsum.shape != (batch, nheads, nchunks, chunk_size) or dout.shape != (batch, seqlen, nheads, headdim) \
            or C.shape != (batch, seqlen, ngroups, dstate):
        raise RuntimeError("neuronspark.dc_tma: dA_cumsum / dout / C shapes")
    if prev_states.dtype != dout.dtype:
        raise RuntimeError(f"neuronspark.dc_tma: prev_states {prev_states.dtype} / dout {dout.dtype} must agree")
    if not prev_states.is_contiguous() or C.stride(-1) != 1 or dout.stride(-1) != 1 or dA_cumsum.stride(-1) != 1 \
            or (chunk_size % BM) or (dstate % BN):
        raise RuntimeError(f"neuronspark.dc_tma: operand layouts or chunk {chunk_size} BM {BM} dstate {dstate} BN {BN} "
                           "not supported")
    tile_count, nvn = _ws_tiles(dstate, BN)
    ws = torch.empty(batch, nheads, nchunks, chunk_size, tile_count, device=dout.device, dtype=torch.float32)
    ngrp = triton.cdiv(nheads, DC_HEADS)
    dC = torch.empty(batch, seqlen, ngrp, ngroups, dstate, device=dout.device, dtype=torch.float32)
    od = _desc3(dout, BM)
    sd = TensorDescriptor(prev_states, [batch * nchunks * nheads * headdim, dstate], [dstate, 1], [headdim, BN])
    grid = (triton.cdiv(chunk_size, BM) * nvn, batch * nchunks, ngrp)
    with torch.cuda.device(dout.device.index):
        _dc_kernel[grid](
            od, sd, C, dA_cumsum, dC, ws, chunk_size, dstate, seqlen, batch, nchunks, nheads,
            C.stride(0), C.stride(1), C.stride(3), dA_cumsum.stride(0), dA_cumsum.stride(1), dA_cumsum.stride(2),
            dC.stride(0), dC.stride(1), dC.stride(2),
            ws.stride(0), ws.stride(1), ws.stride(2), ws.stride(3), ws.stride(4),
            P=headdim, HG=DC_HEADS, BM=BM, BN=BN, num_warps=nw, num_stages=st)
    return dC, ws, nvn


# ------------------------------------------------------------------------------------------------ references
# Pure torch: the chunked-scan math in float64 between the kernels' rounding points, which are reproduced
# (an fp32 operand expression rounded to bf16 before a dot is evaluated in fp32 with the kernel's expression order and
# then rounded); outputs in the kernels' dtypes.  Partial last chunks are handled by zero padding (x, B, C, dout = 0
# past seqlen contribute nothing).
LOG2E = 1.4426950408889634


def _exp_f32(t: torch.Tensor) -> torch.Tensor:
    """fp32 ``exp`` in the form the kernels evaluate ``tl.exp`` on fp32 operands: ``exp2(t * log2 e)`` (the fp32
    product, then the base-2 exponential: ``tl.exp`` issues ex2 after its log2(e) multiply).  ``t`` is cast to
    fp32."""
    return torch.exp2(t.float() * LOG2E)


def _ex2ftz_ref(t: torch.Tensor) -> torch.Tensor:
    """:func:`_exp_f32` with results below 2^-126 flushed to zero (the kernels' ``ex2.approx.ftz`` decay)."""
    e = _exp_f32(t)
    return torch.where(e < 2.0 ** -126, torch.zeros_like(e), e)


def _pad_time(t: torch.Tensor, length: int) -> torch.Tensor:
    if int(t.shape[1]) == length:
        return t
    pad = [0, 0] * (t.dim() - 2) + [0, length - int(t.shape[1])]
    return torch.nn.functional.pad(t, pad)


def _chunked(t: torch.Tensor, nchunks: int, chunk: int) -> torch.Tensor:
    """(b, l, ...) -> (b, nchunks, chunk, ...) float64, zero padded"""
    t = _pad_time(t.double(), nchunks * chunk)
    return t.reshape(t.shape[0], nchunks, chunk, *t.shape[2:])


def _forward_states(states_pre: torch.Tensor, dA_last: torch.Tensor) -> torch.Tensor:
    """states entering each chunk: s_0 = 0, s_{c+1} = exp(dA_last[c]) s_c + states_pre[c] (the decay factor in the
    kernels' fp32 exp form); states_pre (b, nc, h, P, N), dA_last (b, h, nc) -> (b, nc, h, P, N) float64"""
    s = torch.zeros_like(states_pre[:, 0], dtype=torch.float64)
    decay = _exp_f32(dA_last).double()
    out = []
    for c in range(states_pre.shape[1]):
        out.append(s)
        s = decay[:, :, c][..., None, None] * s + states_pre[:, c].double()
    return torch.stack(out, dim=1)


def csp_fused_ref(B, x, dt, dA_cumsum, out_dtype):
    """The chunk states and their passing with the chunk-state kernel's operand rounding (``chunk_states_train``:
    ``scale = exp(min(dA_last - dA_l, 0)) dt_l`` in fp32, ``bf16(B_l scale)`` as the dot operand): states_pre[c] =
    sum_{l in c} x_l bf16(B_l scale_l)^T (float64 sums), then the passing recurrence; returns (states_pre fp32, states in
    out_dtype)."""
    batch, seqlen, nheads, headdim = x.shape
    _, _, nchunks, chunk = dt.shape
    xc = _chunked(x, nchunks, chunk)                                   # (b, c, l, h, p)
    Bc = _chunked(B[:, :, 0], nchunks, chunk).float()                  # (b, c, l, n) bf16-exact fp32
    dA = dA_cumsum.float()                                             # (b, h, c, l)
    scale = _exp_f32(torch.clamp_max(dA[..., -1:] - dA, 0.0)) * dt.float()        # (b, h, c, l) fp32
    bs = (Bc[:, :, :, None, :] * scale.permute(0, 2, 3, 1)[..., None]).to(B.dtype).double()   # (b, c, l, h, n)
    states_pre = torch.einsum("bclhp,bclhn->bchpn", xc, bs)
    states = _forward_states(states_pre.float(), dA[..., -1])
    return states_pre.float(), states.to(out_dtype)


def csp_prefill_ref(B, x, dt, dA_cumsum, initial_state, out_dtype):
    """Reference of the prefill chunk states (``ops.gluon.dendritic_chunk_states.chunk_states_prefill``), with the
    kernel's expressions and rounding points::

        chunk state:    scale_l = exp(min(dA_last - dA_l, 0)) dt_l                                   fp32
                        chunk_state[c] = sum_{l in c} x_l bf16(B_l scale_l)^T                          stored fp32
        state passing:  s_0 = initial_state,  states[c] = out_dtype(s_c),
                        s_{c+1} = exp(dA_last[c]) s_c + chunk_state[c]                                 fp32
        final_state = s_nchunks (fp32)

    ``dA_last[c] = dA_cumsum[:, :, c, -1]``; ``exp`` in the kernels' fp32 form (:func:`_exp_f32`); the sums of the dot
    and of each passing step are accumulated in float64 and rounded once to fp32 (the kernels' fp32 dot / fma order is
    the rule class).  Returns (states (b, nc, h, P, N) in out_dtype, final_state (b, h, P, N) fp32).  No kernel; any
    device."""
    batch, seqlen, nheads, headdim = x.shape
    _, _, nchunks, chunk = dt.shape
    xc = _chunked(x, nchunks, chunk)                                   # (b, c, l, h, p) float64
    Bc = _chunked(B[:, :, 0], nchunks, chunk).float()                  # (b, c, l, n) bf16-exact fp32
    dA = dA_cumsum.float()                                             # (b, h, c, l)
    scale = _exp_f32(torch.clamp_max(dA[..., -1:] - dA, 0.0)) * dt.float()        # (b, h, c, l) fp32
    bs = (Bc[:, :, :, None, :] * scale.permute(0, 2, 3, 1)[..., None]).to(B.dtype).double()   # (b, c, l, h, n)
    chunk_states = torch.einsum("bclhp,bclhn->bchpn", xc, bs).float()                 # the stored fp32 chunk states
    decay = _exp_f32(dA[..., -1])                                      # (b, h, c) fp32
    s = initial_state.float()
    entering = []
    for c in range(nchunks):
        entering.append(s.to(out_dtype))
        s = (decay[:, :, c].double()[..., None, None] * s.double() + chunk_states[:, c].double()).float()
    return torch.stack(entering, dim=1), s


def db_tma_ref(x, dt, dA_cumsum, dstates, B, ngroups):
    """float64 reference of ``db_tma`` after its reductions: returns
    (dB (b, l, 1, n) fp32 = sum over heads, ddA_next (b, h, c, l) fp32 = the workspace summed over its tiles, slot
    l = sum_n db_h[l - 1, n] B[l - 1, n] for l >= 1 and 0 at l = 0)."""
    batch, seqlen, nheads, headdim = x.shape
    _, _, nchunks, chunk = dt.shape
    xc = _chunked(x, nchunks, chunk)                                   # (b, c, l, h, p)
    Bc = _chunked(B[:, :, 0], nchunks, chunk)                          # (b, c, l, n)
    dA = dA_cumsum.double()
    scale = torch.exp(torch.clamp_max(dA[..., -1:] - dA, 0.0)) * dt.double()          # (b, h, c, l)
    dbh = torch.einsum("bclhp,bchpn,bhcl->bhcln", xc, dstates.double(), scale)       # per head
    dB = dbh.sum(1).reshape(batch, nchunks * chunk, 1, -1)[:, :seqlen]
    t = (dbh * Bc[:, None]).sum(-1)                                    # (b, h, c, l)
    nxt = torch.zeros_like(t)
    nxt[..., 1:] = t[..., :-1]
    return dB.float(), nxt.float()


def dc_tma_ref(prev_states, dA_cumsum, dout, C, ngroups):
    """float64 reference of ``dc_tma`` after its reductions: returns
    (dC (b, l, 1, n) fp32 = sum over heads of (dout_h prev_states_h) exp(dA_l), ddA_prev (b, h, c, l) fp32 =
    sum_n dc_h[l, n] C[l, n])."""
    batch, nchunks, nheads, headdim, dstate = prev_states.shape
    _, seqlen, _, _ = dout.shape
    chunk = dA_cumsum.shape[-1]
    do = _chunked(dout, nchunks, chunk)                                # (b, c, l, h, p)
    Cc = _chunked(C[:, :, 0], nchunks, chunk)                          # (b, c, l, n)
    dch = torch.einsum("bclhp,bchpn,bhcl->bhcln", do, prev_states.double(), torch.exp(dA_cumsum.double()))
    dC = dch.sum(1).reshape(batch, nchunks * chunk, 1, -1)[:, :seqlen]
    return dC.float(), (dch * Cc[:, None]).sum(-1).float()
