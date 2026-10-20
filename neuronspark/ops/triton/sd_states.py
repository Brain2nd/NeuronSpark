"""SD chunk states: the dB / dC pieces of the SSD scan's training backward, and the pure-torch references of the chunk
states + state passing (the forward pair runs the Gluon kernels of :mod:`neuronspark.ops.gluon.sd_chunk_states`, whose
references ``csp_fused_ref`` / ``csp_prefill_ref`` are here; the state gradients run
:mod:`neuronspark.ops.gluon.sd_dstates`).

Internal pieces (not registry ops); ``db_tma`` and ``dc_tma`` are used by
:class:`neuronspark.ops.sd_combined.SDChunkScanSomaFn`:
* ``db_tma(x, dt, dA_cumsum, dstates, B, ngroups) -> (dB parts, ddA_next workspace, n tiles)`` and
  ``dc_tma(prev_states, dA_cumsum, dout, C, ngroups) -> (dC parts, ddA_prev workspace, n tiles)``: mamba_ssm
  ``_chunk_state_bwd_db_kernel`` / ``_chunk_scan_bwd_dc_kernel`` (HAS_DDA_CS, deterministic tile workspaces, ngroups 1)
  with the per-head tiles loaded by TMA descriptors; the head-split partials (``nheads_per_program =
  ceil(batch nchunks nheads / #SM)``, SM-count dependent) are summed by :func:`neuronspark.ops.triton.sd_glue.splits_sum`.

Numerics: the stock kernels' per-head expressions and dot shapes in the stock head order.  Checked with the
per-operator tolerance rule of :mod:`neuronspark.utils.testing` against the stock mamba_ssm kernels they replace and
against the references below.

Launch constants: db and dc (BM 64, BN 64, 4 warps, 3 stages).  The TMA descriptor kernels need a GPU with TMA
(sm_90+); the package targets Blackwell (sm_100+).

The kernel bodies are mamba_ssm 2.3.2.post1's (``ssd_chunk_state.py`` ``_chunk_state_bwd_db_kernel``,
``ssd_chunk_scan.py`` ``_chunk_scan_bwd_dc_kernel``; Apache-2.0, see NOTICE) with the changes stated in each kernel's
docstring.
"""
from __future__ import annotations

import math

import torch
import triton
import triton.language as tl
from triton.tools.tensor_descriptor import TensorDescriptor

from neuronspark.ops.triton import num_sms, require_cuda
from neuronspark.ops.triton.mamba import CHUNK_SCAN_BWD_DC_MIN_BLOCK_N, CHUNK_STATE_BWD_DB_MIN_BLOCK_N

__all__ = [
    "DB_CFG", "DC_CFG",
    "db_tma", "dc_tma",
    "csp_fused_ref", "csp_prefill_ref", "db_tma_ref", "dc_tma_ref",
]

# ------------------------------------------------------------------------------------------------ launch constants
DB_CFG = (64, 64, 4, 3)        # BM, BN, num_warps, num_stages
DC_CFG = (64, 64, 4, 3)        # BM, BN, num_warps, num_stages


# ------------------------------------------------------------------------------------------------ kernels
@triton.jit
def _db_kernel(XD, DSD, B, DT, DA, DB, WS,
                chunk_size, dstate, seqlen, batch, nchunks, nheads, nheads_per_program,
                s_b_b, s_b_l, s_b_n, s_dt_b, s_dt_c, s_dt_h, s_dt_l, s_da_b, s_da_c, s_da_h, s_da_l,
                s_db_b, s_db_l, s_db_s, s_db_n, s_ws_b, s_ws_c, s_ws_h, s_ws_l, s_ws_t,
                H: tl.constexpr, P: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr):
    """mamba_ssm _chunk_state_bwd_db_kernel (HAS_DDA_CS, no seq_idx, DETERMINISTIC_REDUCTION, ngroups 1, BLOCK_SIZE_K
    = P) with x_h[m] and dstates_h[:, n] as descriptor boxes (see the module docstring)."""
    pid_bc = tl.program_id(axis=1)
    pid_s = tl.program_id(axis=2)
    num_pid_n = tl.cdiv(dstate, BN)
    pid_m = tl.program_id(axis=0) // num_pid_n
    pid_n = tl.program_id(axis=0) % num_pid_n
    pid_c = pid_bc // batch
    pid_b = pid_bc - pid_c * batch
    h0 = pid_s * nheads_per_program
    offs_m = pid_m * BM + tl.arange(0, BM)
    offs_n = pid_n * BN + tl.arange(0, BN)
    chunk_size_limit = min(chunk_size, seqlen - pid_c * chunk_size)
    t0 = pid_c * chunk_size
    b_ptrs = B + pid_b * s_b_b + (t0 + offs_m[:, None]) * s_b_l + offs_n[None, :] * s_b_n
    b = tl.load(b_ptrs, mask=(offs_m[:, None] < chunk_size_limit) & (offs_n[None, :] < dstate), other=0.0).to(tl.float32)
    dtb = DT + pid_b * s_dt_b + pid_c * s_dt_c + h0 * s_dt_h
    dab = DA + pid_b * s_da_b + pid_c * s_da_c + h0 * s_da_h
    wsb = WS + pid_b * s_ws_b + pid_c * s_ws_c + h0 * s_ws_h + pid_n * s_ws_t + offs_m * s_ws_l
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    nheads_iter = min(nheads_per_program, nheads - h0)
    for h in range(nheads_iter):
        x = XD.load([pid_b, t0 + pid_m * BM, (h0 + h) * P]).reshape(BM, P)
        dstates = DSD.load([((pid_b * nchunks + pid_c) * H + h0 + h) * P, pid_n * BN])
        db = tl.dot(x, dstates)
        dA_cs_last = tl.load(dab + h * s_da_h + (chunk_size - 1) * s_da_l).to(tl.float32)
        dA_cs_m = tl.load(dab + h * s_da_h + offs_m * s_da_l, mask=offs_m < chunk_size, other=0.0).to(tl.float32)
        dt_m = tl.load(dtb + h * s_dt_h + offs_m * s_dt_l, mask=offs_m < chunk_size, other=0.0).to(tl.float32)
        scale = tl.exp(tl.minimum((dA_cs_last - dA_cs_m), 0.0))
        db *= (scale * dt_m)[:, None]
        ddA_cs = tl.sum(db * b, axis=1)
        tl.store(wsb + h * s_ws_h + s_ws_l, ddA_cs, mask=offs_m < chunk_size - 1)
        acc += db
    db_ptrs = DB + pid_b * s_db_b + (t0 + offs_m[:, None]) * s_db_l + pid_s * s_db_s + offs_n[None, :] * s_db_n
    tl.store(db_ptrs, acc, mask=(offs_m[:, None] < chunk_size_limit) & (offs_n[None, :] < dstate))


@triton.jit
def _dc_kernel(DOD, SD, C, DA, DC, WS,
                chunk_size, dstate, seqlen, batch, nchunks, nheads, nheads_per_program,
                s_c_b, s_c_l, s_c_n, s_da_b, s_da_c, s_da_h, s_da_l,
                s_dc_b, s_dc_l, s_dc_s, s_dc_n, s_ws_b, s_ws_c, s_ws_h, s_ws_l, s_ws_t,
                H: tl.constexpr, P: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr):
    """mamba_ssm _chunk_scan_bwd_dc_kernel (HAS_DDA_CS, no seq_idx, DETERMINISTIC_REDUCTION, ngroups 1, BLOCK_SIZE_K
    = P) with dout_h[m] and prev_states_h[:, n] as descriptor boxes (see the module docstring)."""
    pid_bc = tl.program_id(axis=1)
    pid_s = tl.program_id(axis=2)
    num_pid_n = tl.cdiv(dstate, BN)
    pid_m = tl.program_id(axis=0) // num_pid_n
    pid_n = tl.program_id(axis=0) % num_pid_n
    pid_c = pid_bc // batch
    pid_b = pid_bc - pid_c * batch
    h0 = pid_s * nheads_per_program
    offs_m = pid_m * BM + tl.arange(0, BM)
    offs_n = pid_n * BN + tl.arange(0, BN)
    chunk_size_limit = min(chunk_size, seqlen - pid_c * chunk_size)
    t0 = pid_c * chunk_size
    c_ptrs = C + pid_b * s_c_b + (t0 + offs_m[:, None]) * s_c_l + offs_n[None, :] * s_c_n
    c = tl.load(c_ptrs, mask=(offs_m[:, None] < chunk_size_limit) & (offs_n[None, :] < dstate), other=0.0).to(tl.float32)
    dab = DA + pid_b * s_da_b + pid_c * s_da_c + h0 * s_da_h
    wsb = WS + pid_b * s_ws_b + pid_c * s_ws_c + h0 * s_ws_h + pid_n * s_ws_t + offs_m * s_ws_l
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    nheads_iter = min(nheads_per_program, nheads - h0)
    for h in range(nheads_iter):
        dout = DOD.load([pid_b, t0 + pid_m * BM, (h0 + h) * P]).reshape(BM, P)
        prev_states = SD.load([((pid_b * nchunks + pid_c) * H + h0 + h) * P, pid_n * BN])
        dc = tl.dot(dout, prev_states)
        dA_cs_m = tl.load(dab + h * s_da_h + offs_m * s_da_l, mask=offs_m < chunk_size_limit, other=0.0).to(tl.float32)
        scale = tl.exp(dA_cs_m)
        dc *= scale[:, None]
        ddA_cs = tl.sum(dc * c, axis=1)
        tl.store(wsb + h * s_ws_h, ddA_cs, mask=offs_m < chunk_size)
        acc += dc
    dc_ptrs = DC + pid_b * s_dc_b + (t0 + offs_m[:, None]) * s_dc_l + pid_s * s_dc_s + offs_n[None, :] * s_dc_n
    tl.store(dc_ptrs, acc, mask=(offs_m[:, None] < chunk_size_limit) & (offs_n[None, :] < dstate))


# ------------------------------------------------------------------------------------------------ launchers
def _desc3(t4, block_rows):
    """(1, block_rows, P) boxes of a (batch, seqlen, H, P) view whose (H, P) part is contiguous: descriptor over
    (batch, seqlen, H * P) -- rows t >= seqlen read as zeros (the reference's masks)."""
    batch, seqlen, nh, p = t4.shape
    if t4.stride(3) != 1 or t4.stride(2) != p:
        raise RuntimeError(f"neuronspark.sd_states: (H, P) part of a {tuple(t4.shape)} view with strides "
                           f"{tuple(t4.stride())} is not contiguous")
    return TensorDescriptor(t4, [batch, seqlen, nh * p], [t4.stride(0), t4.stride(1), 1], [1, block_rows, p])


def _split_geometry(x, dstate, BM, BN, min_block_n):
    tile_count = math.ceil(dstate / min_block_n)
    nvn = triton.cdiv(dstate, BN)
    if nvn > tile_count:
        raise RuntimeError(f"neuronspark.sd_states: {nvn} n tiles written, the workspace holds {tile_count}")
    return tile_count, nvn, num_sms(x.device)


def db_tma(x, dt, dA_cumsum, dstates, B, ngroups):
    """(dB parts (b, l, nsplits, 1, n) fp32, ddA workspace (b, h, c, l, tiles) fp32, n written tiles) of the stock
    ``_chunk_state_bwd_db_kernel`` (HAS_DDA_CS, no seq_idx, deterministic tile workspaces, ngroups 1, BLOCK_SIZE_K = P)
    with x_h[m] and dstates_h[:, n] as TMA boxes.  Workspace slot l = 0 of each row is never written (the kernel stores
    rows m < L - 1 at slot m + 1) and never read."""
    require_cuda(x, "db_tma")
    batch, seqlen, nheads, headdim = x.shape
    _, _, nchunks, chunk_size = dt.shape
    dstate = dstates.shape[-1]
    BM, BN, nw, st = DB_CFG
    if ngroups != 1 or headdim not in (16, 32, 64, 128) or dstate > 256:
        raise RuntimeError(f"neuronspark.db_tma: ngroups {ngroups} headdim {headdim} dstate {dstate} (SD call: 1 / pow2 "
                           "/ <= 256)")
    if dt.shape != (batch, nheads, nchunks, chunk_size) or dA_cumsum.shape != dt.shape:
        raise RuntimeError("neuronspark.db_tma: dt / dA_cumsum shapes")
    if dstates.shape != (batch, nchunks, nheads, headdim, dstate) or B.shape != (batch, seqlen, ngroups, dstate):
        raise RuntimeError("neuronspark.db_tma: dstates / B shapes")
    if dstates.dtype != x.dtype:
        raise RuntimeError(f"neuronspark.db_tma: dstates {dstates.dtype} / x {x.dtype} must agree (the stock kernel casts "
                           "dstates to x's dtype; the SD call passes bf16 both)")
    if not dstates.is_contiguous() or B.stride(-1) != 1 or x.stride(-1) != 1 or dt.stride(-1) != 1 \
            or dA_cumsum.stride(-1) != 1 or (chunk_size % BM) or (dstate % BN):
        raise RuntimeError(f"neuronspark.db_tma: operand layouts or chunk {chunk_size} BM {BM} dstate {dstate} BN {BN} "
                           "not supported")
    tile_count, nvn, sm_count = _split_geometry(x, dstate, BM, BN, CHUNK_STATE_BWD_DB_MIN_BLOCK_N)
    ws = torch.empty(batch, nheads, nchunks, chunk_size, tile_count, device=x.device, dtype=torch.float32)
    nheads_per_program = max(min(math.ceil(batch * nchunks * nheads / sm_count), nheads), 1)
    nsplits = triton.cdiv(nheads, nheads_per_program)
    dB = torch.empty(batch, seqlen, nsplits, ngroups, dstate, device=x.device, dtype=torch.float32)
    xd = _desc3(x, BM)
    dsd = TensorDescriptor(dstates, [batch * nchunks * nheads * headdim, dstate], [dstate, 1], [headdim, BN])
    grid = (triton.cdiv(chunk_size, BM) * nvn, batch * nchunks, nsplits)
    with torch.cuda.device(x.device.index):
        _db_kernel[grid](
            xd, dsd, B, dt, dA_cumsum, dB, ws,
            chunk_size, dstate, seqlen, batch, nchunks, nheads, nheads_per_program,
            B.stride(0), B.stride(1), B.stride(3),
            dt.stride(0), dt.stride(2), dt.stride(1), dt.stride(3),
            dA_cumsum.stride(0), dA_cumsum.stride(2), dA_cumsum.stride(1), dA_cumsum.stride(3),
            dB.stride(0), dB.stride(1), dB.stride(2), dB.stride(4),
            ws.stride(0), ws.stride(2), ws.stride(1), ws.stride(3), ws.stride(4),
            H=nheads, P=headdim, BM=BM, BN=BN, num_warps=nw, num_stages=st)
    return dB, ws, nvn


def dc_tma(prev_states, dA_cumsum, dout, C, ngroups):
    """(dC parts (b, l, nsplits, 1, n) fp32, ddA_prev workspace (b, h, c, l, tiles) fp32, n written tiles) of the stock
    ``_chunk_scan_bwd_dc_kernel`` (HAS_DDA_CS, no seq_idx, deterministic tile workspaces, ngroups 1, BLOCK_SIZE_K = P)
    with dout_h[m] and prev_states_h[:, n] as TMA boxes."""
    require_cuda(dout, "dc_tma")
    batch, nchunks, nheads, headdim, dstate = prev_states.shape
    _, seqlen, _, _ = dout.shape
    _, _, _, chunk_size = dA_cumsum.shape
    BM, BN, nw, st = DC_CFG
    if ngroups != 1 or headdim not in (16, 32, 64, 128) or dstate > 256:
        raise RuntimeError(f"neuronspark.dc_tma: ngroups {ngroups} headdim {headdim} dstate {dstate} (SD call: 1 / pow2 "
                           "/ <= 256)")
    if dA_cumsum.shape != (batch, nheads, nchunks, chunk_size) or dout.shape != (batch, seqlen, nheads, headdim) \
            or C.shape != (batch, seqlen, ngroups, dstate):
        raise RuntimeError("neuronspark.dc_tma: dA_cumsum / dout / C shapes")
    if prev_states.dtype != dout.dtype:
        raise RuntimeError(f"neuronspark.dc_tma: prev_states {prev_states.dtype} / dout {dout.dtype} must agree")
    if not prev_states.is_contiguous() or C.stride(-1) != 1 or dout.stride(-1) != 1 or dA_cumsum.stride(-1) != 1 \
            or (chunk_size % BM) or (dstate % BN):
        raise RuntimeError(f"neuronspark.dc_tma: operand layouts or chunk {chunk_size} BM {BM} dstate {dstate} BN {BN} "
                           "not supported")
    tile_count, nvn, sm_count = _split_geometry(dout, dstate, BM, BN, CHUNK_SCAN_BWD_DC_MIN_BLOCK_N)
    ws = torch.empty(batch, nheads, nchunks, chunk_size, tile_count, device=dout.device, dtype=torch.float32)
    nheads_per_program = max(min(math.ceil(batch * nchunks * nheads / sm_count), nheads), 1)
    nsplits = triton.cdiv(nheads, nheads_per_program)
    dC = torch.empty(batch, seqlen, nsplits, ngroups, dstate, device=dout.device, dtype=torch.float32)
    dod = _desc3(dout, BM)
    sd = TensorDescriptor(prev_states, [batch * nchunks * nheads * headdim, dstate], [dstate, 1], [headdim, BN])
    grid = (triton.cdiv(chunk_size, BM) * nvn, batch * nchunks, nsplits)
    with torch.cuda.device(dout.device.index):
        _dc_kernel[grid](
            dod, sd, C, dA_cumsum, dC, ws,
            chunk_size, dstate, seqlen, batch, nchunks, nheads, nheads_per_program,
            C.stride(0), C.stride(1), C.stride(3),
            dA_cumsum.stride(0), dA_cumsum.stride(2), dA_cumsum.stride(1), dA_cumsum.stride(3),
            dC.stride(0), dC.stride(1), dC.stride(2), dC.stride(4),
            ws.stride(0), ws.stride(2), ws.stride(1), ws.stride(3), ws.stride(4),
            H=nheads, P=headdim, BM=BM, BN=BN, num_warps=nw, num_stages=st)
    return dC, ws, nvn


# ------------------------------------------------------------------------------------------------ references
# Pure torch: the stock mamba_ssm math in float64 between the kernels' rounding points, which are reproduced
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
    """mamba_ssm ``chunk_state_ref`` / ``state_passing_ref`` math (ssd_chunk_state.py 1094-1122, ssd_state_passing.py
    327-350) with the chunk-state kernel's operand rounding (``chunk_states_train``, as the stock ``_chunk_state_fwd_kernel``:
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
    """Reference of the prefill chunk states (``ops.gluon.sd_chunk_states.chunk_states_prefill``): the replaced pair, mamba_ssm ``_chunk_state_fwd(.., states_in_fp32=True)`` then
    ``_state_passing_fwd(.., initial_states=initial_state, out_dtype)``, with their expressions and rounding points::

        chunk state (_chunk_state_fwd_kernel):  scale_l = exp(min(dA_last - dA_l, 0)) dt_l               fp32
                                                chunk_state[c] = sum_{l in c} x_l bf16(B_l scale_l)^T      stored fp32
        state passing (_state_passing_fwd_kernel):  s_0 = initial_state,  states[c] = out_dtype(s_c),
                                                    s_{c+1} = exp(dA_last[c]) s_c + chunk_state[c]          fp32
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
    """float64 reference of ``db_tma`` after its reductions (mamba_ssm ``_chunk_state_bwd_db`` math): returns
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
    """float64 reference of ``dc_tma`` after its reductions (mamba_ssm ``_chunk_scan_bwd_dC`` math): returns
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
