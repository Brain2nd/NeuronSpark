"""SD chunk states: the chunk-state + state-passing forward of the SSD scan (training, and the inference prefill form
from a carried state) and three pieces of the training backward.

Internal pieces (not registry ops); ``csp_fused``, ``spd_fused``, ``db_tma`` and ``dc_tma`` are used by
:class:`neuronspark.ops.sd_combined.SDChunkScanSomaFn`, ``csp_prefill`` by
:func:`neuronspark.ops.sd_prefill.sd_chunk_scan_prefill`:

* ``csp_fused(B, x, dt, dA_cumsum, out_dtype) -> (states_pre, states)``: mamba_ssm ``_chunk_state_fwd(..,
  states_in_fp32=True)`` followed by ``_state_passing_fwd(.., out_dtype)`` in one kernel per (N tile, b, h).
  ``states_pre`` (b, nc, h, P, N) fp32 = each chunk's own state, ``states`` (b, nc, h, P, N) in out_dtype = the state
  entering chunk c (states[0] = 0; the final state is not stored).
* ``csp_prefill(B, x, dt, dA_cumsum, initial_state, out_dtype) -> (states, final_state)``: the same pair from a
  carried ``initial_state`` (b, h, P, N) fp32, ``_state_passing_fwd(.., initial_states=initial_state, out_dtype)``:
  ``states`` (b, nc, h, P, N) in out_dtype = the state entering chunk c (states[0] = the initial state),
  ``final_state`` (b, h, P, N) fp32.  Each chunk's own fp32 state is consumed in registers by the passing step and
  never stored (the value the stock pair stores in fp32 and reads back: the same).
* ``spd_fused(states_pre, dA_chunk_cumsum, dout, C, dA_cumsum, out_dtype) -> (dstates, ddA_ws, n_valid,
  states_converted)``: mamba_ssm ``_chunk_scan_bwd_dstates`` (fp32 dstates_in) + ``_state_passing_fwd`` (fp32
  recompute) + ``_state_passing_bwd`` in one kernel per (P block, b, h).
* ``db_tma(x, dt, dA_cumsum, dstates, B, ngroups) -> (dB parts, ddA_next workspace, n tiles)`` and
  ``dc_tma(prev_states, dA_cumsum, dout, C, ngroups) -> (dC parts, ddA_prev workspace, n tiles)``: mamba_ssm
  ``_chunk_state_bwd_db_kernel`` / ``_chunk_scan_bwd_dc_kernel`` (HAS_DDA_CS, deterministic tile workspaces, ngroups 1)
  with the per-head tiles loaded by TMA descriptors; the head-split partials (``nheads_per_program =
  ceil(batch nchunks nheads / #SM)``, SM-count dependent) are summed by :func:`neuronspark.ops.triton.sd_glue.splits_sum`.

Numerics.  csp / csp_prefill: the stock chunk-state body (scale = exp(min(dA_last - dA_k, 0)) dt_k, B * scale rounded
to bf16, fp32 dot accumulation over K steps of BK tokens) and the stock state-passing step on the fp32 value, states
rounded to out_dtype on store (csp_prefill: K steps of 32 tokens, the K block of the stock chunk-state configuration
of the prefill; the final state fp32; the dot tiling and the fused passing are the rule class).  spd:
dstates / states_converted per element as the stock pair; the ddA workspace blocks are 512-element sums taken as
``tl.sum(axis=1)`` of a (P / 4, 512) reshape (fp32 reduction-order class vs the stock 1-D block sum).  db / dc: the
stock kernels' per-head expressions and dot shapes in the stock head order.  Checked with the per-operator
tolerance rule of :mod:`neuronspark.utils.testing` against the stock mamba_ssm kernels they replace (csp_prefill:
``_chunk_state_fwd`` + ``_state_passing_fwd`` at the prefill configurations of :mod:`neuronspark.ops.triton.mamba`)
and against the references below.

Launch constants: csp (BN 64, BK 64, 4 warps, 4 stages); csp_prefill (BN 64, BK 32, 4 warps, 4 stages); spd (BK 32, 4 warps, 3 stages, PB 64); db and dc (BM 64, BN 64, 4 warps, 3 stages).  The TMA descriptor
kernels need a GPU with TMA (sm_90+); the package targets Blackwell (sm_100+).  csp and csp_prefill unroll the chunk
loop: one compiled variant per chunk count (1 .. 64 chunks).

The kernel bodies are mamba_ssm 2.3.2.post1's (``ssd_chunk_state.py`` ``_chunk_state_fwd_kernel`` /
``_chunk_state_bwd_db_kernel``, ``ssd_state_passing.py`` ``_state_passing_fwd_kernel`` / ``_state_passing_bwd_kernel``,
``ssd_chunk_scan.py`` ``_chunk_scan_bwd_dstates_kernel`` / ``_chunk_scan_bwd_dc_kernel``; Apache-2.0, see NOTICE)
with the changes stated in each kernel's docstring.
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
    "CSP_CFG", "CSP_PREFILL_CFG", "CSP_MAX_CHUNKS", "SPD_CFG", "DB_CFG", "DC_CFG",
    "csp_fused", "csp_prefill", "spd_fused", "db_tma", "dc_tma",
    "csp_fused_ref", "csp_prefill_ref", "spd_fused_ref", "db_tma_ref", "dc_tma_ref",
]

# ------------------------------------------------------------------------------------------------ launch constants
CSP_CFG = (64, 64, 4, 4)       # BN, BK, num_warps, num_stages
CSP_PREFILL_CFG = (64, 32, 4, 4)   # BN, BK, num_warps, num_stages
CSP_MAX_CHUNKS = 64            # the chunk loop is unrolled: 1 .. 64 chunks per call
SPD_CFG = (32, 4, 3, 64)       # BK, num_warps, num_stages, PB (rows of P per program)
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


@triton.jit
def _spd_kernel(PRE, DAL, DOUT, C, DA, DST, CONV, WS,
                 chunk_size, seqlen, dstate,
                 s_pre_b, s_pre_c, s_pre_h, s_dal_b, s_dal_c, s_dal_h,
                 s_do_b, s_do_l, s_do_h, s_do_p, s_c_b, s_c_l, s_c_n, s_da_b, s_da_c, s_da_h, s_da_l,
                 s_dst_b, s_dst_c, s_dst_h, s_conv_b, s_conv_c, s_conv_h, s_ws_b, s_ws_c, s_ws_h,
                 NCHUNK: tl.constexpr, P: tl.constexpr, N: tl.constexpr, BK: tl.constexpr, RB: tl.constexpr,
                 PB: tl.constexpr):
    """mamba_ssm _state_passing_fwd_kernel (fp32 states recomputed in registers) followed by _state_passing_bwd_kernel
    (no dfinal / dinit / seq_idx, CONVERT_STATES) on a (PB, N) row block of the (P, N) state tile of one (b, h) (rows
    [pid_q PB, pid_q PB + PB)), with dstates_in[k] computed in place by the body of mamba_ssm
    _chunk_scan_bwd_dstates_kernel (no seq_idx; BLOCK_SIZE_M = PB rows of P, BLOCK_SIZE_N = N, BLOCK_SIZE_K = BK) for
    chunk k; ddA workspace block j (the stock's 512-element block j of the flattened (p n) state = rows
    [RB j, RB j + RB)) = row j - pid_q PB / RB of the (PB / RB, RB * N) reshape (RB * N = 512; PB is a multiple of RB, so
    no block straddles two programs)."""
    pid_q = tl.program_id(0)
    pid_b = tl.program_id(1)
    pid_h = tl.program_id(2)
    offs_p = pid_q * PB + tl.arange(0, PB)
    offs_n = tl.arange(0, N)
    offs_k = tl.arange(0, BK)
    offs_j = pid_q * (PB // RB) + tl.arange(0, PB // RB)
    pn = offs_p[:, None] * N + offs_n[None, :]
    pre = PRE + pid_b * s_pre_b + pid_h * s_pre_h + pn
    dal = DAL + pid_b * s_dal_b + pid_h * s_dal_h
    dst = DST + pid_b * s_dst_b + pid_h * s_dst_h + pn
    conv = CONV + pid_b * s_conv_b + pid_h * s_conv_h + pn
    ws = WS + pid_b * s_ws_b + pid_h * s_ws_h + offs_j
    dob = DOUT + pid_b * s_do_b + pid_h * s_do_h
    cb = C + pid_b * s_c_b
    dab = DA + pid_b * s_da_b + pid_h * s_da_h
    dstates = tl.zeros((PB, N), dtype=tl.float32)
    tl.store(dst + (NCHUNK - 1) * s_dst_c, dstates)
    for k in tl.static_range(NCHUNK - 1, 0, -1):
        # the stock forward's out[k] (fp32 states of chunk k): states = exp(dA_cs[c]) states + states_pre[c], c < k
        states = tl.zeros((PB, N), dtype=tl.float32)
        for ci in tl.static_range(k):
            new_states = tl.load(pre + ci * s_pre_c)
            dA_cs_f = tl.load(dal + ci * s_dal_c).to(tl.float32)
            scale_f = tl.exp(dA_cs_f)
            states = scale_f * states + new_states
        dA_cs = tl.load(dal + k * s_dal_c).to(tl.float32)
        scale = tl.exp(dA_cs)
        tl.store(conv + k * s_conv_c, states)
        prod = states * dstates
        ddA = tl.sum(tl.reshape(prod, (PB // RB, RB * N)), axis=1) * scale
        tl.store(ws + k * s_ws_c, ddA)
        # dstates_in[k]: _chunk_scan_bwd_dstates_kernel's body for chunk k (rows offs_p of P)
        t0 = k * chunk_size
        limit = min(chunk_size, seqlen - t0)
        dout_ptrs = dob + t0 * s_do_l + (offs_p[:, None] * s_do_p + offs_k[None, :] * s_do_l)
        c_ptrs = cb + t0 * s_c_l + (offs_n[None, :] * s_c_n + offs_k[:, None] * s_c_l)
        da_ptrs = dab + k * s_da_c + offs_k * s_da_l
        acc = tl.zeros((PB, N), dtype=tl.float32)
        for kk in range(0, limit, BK):
            dout = tl.load(dout_ptrs, mask=(offs_p[:, None] < P) & (offs_k[None, :] < limit - kk), other=0.0).to(tl.float32)
            dA_cs_k = tl.load(da_ptrs, mask=offs_k < chunk_size - kk, other=0.0).to(tl.float32)
            scale_k = tl.exp(dA_cs_k)
            dout = (dout * scale_k).to(DOUT.dtype.element_ty)
            ct = tl.load(c_ptrs, mask=(offs_k[:, None] < limit - kk) & (offs_n[None, :] < dstate), other=0.0).to(DOUT.dtype.element_ty)
            acc += tl.dot(dout, ct)
            dout_ptrs += BK * s_do_l
            c_ptrs += BK * s_c_l
            da_ptrs += BK * s_da_l
        dstates = scale * dstates + acc
        tl.store(dst + (k - 1) * s_dst_c, dstates)
    tl.store(conv, tl.zeros((PB, N), dtype=tl.float32))
    tl.store(ws, tl.zeros((PB // RB,), dtype=tl.float32))


@triton.jit
def _csp_kernel(XD, B, ST, OUT, DT, DA,
                 chunk_size, dstate, seqlen,
                 s_b_b, s_b_l, s_b_n, s_st_b, s_st_c, s_st_h, s_st_p, s_st_n, s_o_b, s_o_c, s_o_h, s_o_p, s_o_n,
                 s_dt_b, s_dt_c, s_dt_h, s_dt_l, s_da_b, s_da_c, s_da_h, s_da_l,
                 NCHUNK: tl.constexpr, P: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    """mamba_ssm _chunk_state_fwd_kernel's body (no seq_idx, BLOCK_SIZE_M = P; the x_h[k] tile as a (1, BK, P)
    descriptor box transposed in registers) for the chunks c = 0 .. NCHUNK-1 of one (b, h, N tile) in order, each
    chunk's fp32 states stored, followed by
    mamba_ssm _state_passing_fwd_kernel's step on them (no initial states / seq_idx): out[0] = 0, s = exp(dA_cs[c]) s +
    states[c] (dA_cs[c] = dA_cumsum[b, h, c, chunk_size - 1], the element the stock state passing reads), out[c + 1] =
    s for c < NCHUNK - 1 (the final state is not stored: the SD call drops it)."""
    pid_n = tl.program_id(axis=0)
    pid_b = tl.program_id(axis=1)
    pid_h = tl.program_id(axis=2)
    offs_m = tl.arange(0, P)
    offs_n = pid_n * BN + tl.arange(0, BN)
    offs_k = tl.arange(0, BK)
    pmn = offs_m[:, None] * s_st_p + offs_n[None, :] * s_st_n
    omn = offs_m[:, None] * s_o_p + offs_n[None, :] * s_o_n
    stb = ST + pid_b * s_st_b + pid_h * s_st_h + pmn
    ob = OUT + pid_b * s_o_b + pid_h * s_o_h + omn
    mk = (offs_m[:, None] < P) & (offs_n[None, :] < dstate)
    s = tl.zeros((P, BN), dtype=tl.float32)
    tl.store(ob, s, mask=mk)
    for c in tl.static_range(NCHUNK):
        t0 = c * chunk_size
        b_ptrs = B + pid_b * s_b_b + t0 * s_b_l + (offs_n[None, :] * s_b_n + offs_k[:, None] * s_b_l)
        dt_ptrs = DT + pid_b * s_dt_b + c * s_dt_c + pid_h * s_dt_h + offs_k * s_dt_l
        dab = DA + pid_b * s_da_b + c * s_da_c + pid_h * s_da_h
        dA_cs_last = tl.load(dab + (chunk_size - 1) * s_da_l).to(tl.float32)
        dA_cumsum_ptrs = dab + offs_k * s_da_l
        chunk_size_limit = min(chunk_size, seqlen - t0)
        acc = tl.zeros((P, BN), dtype=tl.float32)
        for k in range(0, chunk_size_limit, BK):
            xk = XD.load([pid_b, t0 + k, pid_h * P]).reshape(BK, P)
            x = tl.trans(xk)
            b = tl.load(b_ptrs, mask=(offs_k[:, None] < chunk_size_limit - k) & (offs_n[None, :] < dstate), other=0.0).to(tl.float32)
            dA_cs_k = tl.load(dA_cumsum_ptrs, mask=offs_k < chunk_size_limit - k, other=0.0).to(tl.float32)
            dt_k = tl.load(dt_ptrs, mask=offs_k < chunk_size_limit - k, other=0.0).to(tl.float32)
            scale = tl.exp(tl.minimum((dA_cs_last - dA_cs_k), 0.0)) * dt_k
            b *= scale[:, None]
            b = b.to(xk.dtype)
            acc += tl.dot(x, b)
            b_ptrs += BK * s_b_l
            dt_ptrs += BK * s_dt_l
            dA_cumsum_ptrs += BK * s_da_l
        states = acc.to(ST.dtype.element_ty)
        tl.store(stb + c * s_st_c, states, mask=mk)
        # _state_passing_fwd_kernel's step for chunk c on the fp32 value it would load
        new_states = states.to(tl.float32)
        scale_c = tl.exp(dA_cs_last)
        s = scale_c * s + new_states
        if c < NCHUNK - 1:
            tl.store(ob + (c + 1) * s_o_c, s, mask=mk)


@triton.jit
def _csp_prefill_kernel(XD, B, INIT, OUT, FIN, DT, DA,
                        chunk_size, dstate, seqlen,
                        s_b_b, s_b_l, s_b_n, s_i_b, s_i_h, s_i_p, s_i_n, s_o_b, s_o_c, s_o_h, s_o_p, s_o_n,
                        s_f_b, s_f_h, s_f_p, s_f_n, s_dt_b, s_dt_c, s_dt_h, s_dt_l, s_da_b, s_da_c, s_da_h, s_da_l,
                        NCHUNK: tl.constexpr, P: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    """mamba_ssm _chunk_state_fwd_kernel's body (no seq_idx, BLOCK_SIZE_M = P; the x_h[k] tile as a (1, BK, P)
    descriptor box transposed in registers) for the chunks c = 0 .. NCHUNK-1 of one (b, h, N tile) in order, each
    chunk's fp32 state consumed in registers by mamba_ssm _state_passing_fwd_kernel's step (initial states, no
    seq_idx): out[0] = the initial state (stored in out's dtype), s = exp(dA_cs[c]) s + chunk_state[c] (dA_cs[c] =
    dA_cumsum[b, h, c, chunk_size - 1], the element the stock state passing reads), out[c + 1] = s for c < NCHUNK - 1,
    the last s stored fp32 as the final state."""
    pid_n = tl.program_id(axis=0)
    pid_b = tl.program_id(axis=1)
    pid_h = tl.program_id(axis=2)
    offs_m = tl.arange(0, P)
    offs_n = pid_n * BN + tl.arange(0, BN)
    offs_k = tl.arange(0, BK)
    mk = (offs_m[:, None] < P) & (offs_n[None, :] < dstate)
    ob = OUT + pid_b * s_o_b + pid_h * s_o_h + offs_m[:, None] * s_o_p + offs_n[None, :] * s_o_n
    s = tl.load(INIT + pid_b * s_i_b + pid_h * s_i_h + offs_m[:, None] * s_i_p + offs_n[None, :] * s_i_n, mask=mk,
                other=0.0).to(tl.float32)
    tl.store(ob, s.to(OUT.dtype.element_ty), mask=mk)                    # out[0] = the initial state (out's dtype)
    for c in tl.static_range(NCHUNK):
        t0 = c * chunk_size
        b_ptrs = B + pid_b * s_b_b + t0 * s_b_l + (offs_n[None, :] * s_b_n + offs_k[:, None] * s_b_l)
        dt_ptrs = DT + pid_b * s_dt_b + c * s_dt_c + pid_h * s_dt_h + offs_k * s_dt_l
        dab = DA + pid_b * s_da_b + c * s_da_c + pid_h * s_da_h
        dA_cs_last = tl.load(dab + (chunk_size - 1) * s_da_l).to(tl.float32)
        dA_cumsum_ptrs = dab + offs_k * s_da_l
        chunk_size_limit = min(chunk_size, seqlen - t0)
        acc = tl.zeros((P, BN), dtype=tl.float32)
        for k in range(0, chunk_size_limit, BK):                          # the stock _chunk_state_fwd_kernel's K loop
            xk = XD.load([pid_b, t0 + k, pid_h * P]).reshape(BK, P)
            x = tl.trans(xk)
            b = tl.load(b_ptrs, mask=(offs_k[:, None] < chunk_size_limit - k) & (offs_n[None, :] < dstate),
                        other=0.0).to(tl.float32)
            dA_cs_k = tl.load(dA_cumsum_ptrs, mask=offs_k < chunk_size_limit - k, other=0.0).to(tl.float32)
            dt_k = tl.load(dt_ptrs, mask=offs_k < chunk_size_limit - k, other=0.0).to(tl.float32)
            scale = tl.exp(tl.minimum((dA_cs_last - dA_cs_k), 0.0)) * dt_k
            b *= scale[:, None]
            b = b.to(xk.dtype)
            acc += tl.dot(x, b)
            b_ptrs += BK * s_b_l
            dt_ptrs += BK * s_dt_l
            dA_cumsum_ptrs += BK * s_da_l
        s = tl.exp(dA_cs_last) * s + acc                                   # the stock _state_passing_fwd_kernel's step
        if c < NCHUNK - 1:
            tl.store(ob + (c + 1) * s_o_c, s.to(OUT.dtype.element_ty), mask=mk)
        else:
            tl.store(FIN + pid_b * s_f_b + pid_h * s_f_h + offs_m[:, None] * s_f_p + offs_n[None, :] * s_f_n, s,
                     mask=mk)


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


def csp_fused(B, x, dt, dA_cumsum, out_dtype):
    """(states_pre (batch, nchunks, nheads, P, N) fp32, states (batch, nchunks, nheads, P, N) in out_dtype) =
    (``_chunk_state_fwd(B, x, dt, dA_cumsum, seq_idx=None, states_in_fp32=True)``, the first output of
    ``_state_passing_fwd(states_pre flattened, dA_cumsum[:, :, :, -1], initial_states=None, seq_idx=None,
    out_dtype=out_dtype)`` unflattened) in one kernel (see _csp_kernel).  B (b, l, 1, N), x (b, l, h, P) with unit
    last strides and a contiguous (h, P) part; dt / dA_cumsum (b, h, nc, chunk) with unit last stride."""
    require_cuda(x, "csp_fused")
    batch, seqlen, nheads, headdim = x.shape
    _, _, nchunks, chunk_size = dt.shape
    _, _, ngroups, dstate = B.shape
    BN, BK, nw, st = CSP_CFG
    if ngroups != 1 or headdim not in (16, 32, 64, 128) or (dstate % BN) or (chunk_size % BK):
        raise RuntimeError(f"neuronspark.csp_fused: ngroups {ngroups} headdim {headdim} dstate {dstate} BN {BN} chunk "
                           f"{chunk_size} BK {BK}")
    if B.shape != (batch, seqlen, ngroups, dstate) or dt.shape != (batch, nheads, nchunks, chunk_size):
        raise RuntimeError(f"neuronspark.csp_fused: shapes B {tuple(B.shape)} dt {tuple(dt.shape)} x {tuple(x.shape)}")
    if dA_cumsum.shape != dt.shape:
        raise RuntimeError("neuronspark.csp_fused: dA_cumsum must have dt's shape")
    if B.stride(-1) != 1 or x.stride(-1) != 1 or dt.stride(-1) != 1 or dA_cumsum.stride(-1) != 1:
        raise RuntimeError("neuronspark.csp_fused: unit last strides of B / x / dt / dA_cumsum expected")
    if nchunks < 1 or nchunks > 64:
        raise RuntimeError(f"neuronspark.csp_fused: nchunks {nchunks} (the SD call has 1 .. 64 chunks)")
    states_pre = torch.empty((batch, nchunks, nheads, headdim, dstate), device=x.device, dtype=torch.float32)
    states = torch.empty((batch, nchunks, nheads, headdim, dstate), device=x.device, dtype=out_dtype)
    xd = _desc3(x, BK)
    grid = (triton.cdiv(dstate, BN), batch, nheads)
    with torch.cuda.device(x.device.index):
        _csp_kernel[grid](
            xd, B, states_pre, states, dt, dA_cumsum,
            chunk_size, dstate, seqlen,
            B.stride(0), B.stride(1), B.stride(-1),
            states_pre.stride(0), states_pre.stride(1), states_pre.stride(2), states_pre.stride(3), states_pre.stride(4),
            states.stride(0), states.stride(1), states.stride(2), states.stride(3), states.stride(4),
            dt.stride(0), dt.stride(2), dt.stride(1), dt.stride(3),
            dA_cumsum.stride(0), dA_cumsum.stride(2), dA_cumsum.stride(1), dA_cumsum.stride(3),
            NCHUNK=nchunks, P=headdim, BN=BN, BK=BK, num_warps=nw, num_stages=st)
    return states_pre, states


def csp_prefill(B, x, dt, dA_cumsum, initial_state, out_dtype):
    """(states (batch, nchunks, nheads, P, N) in out_dtype, final_state (batch, nheads, P, N) fp32) = (the first output
    unflattened, the final states) of ``_state_passing_fwd(_chunk_state_fwd(B, x, dt, dA_cumsum, seq_idx=None,
    states_in_fp32=True) flattened, dA_cumsum[:, :, :, -1], initial_states=initial_state flattened, seq_idx=None,
    out_dtype=out_dtype)`` in one kernel (see _csp_prefill_kernel).  B (b, l, 1, N), x (b, l, h, P) with unit last
    strides and a contiguous (h, P) part (any row / batch strides); dt / dA_cumsum (b, h, nc, chunk) with unit last
    stride; initial_state (b, h, P, N) fp32 (any strides).  Partial last chunks are supported (rows past seqlen are the
    descriptor's zero fill and masked loads)."""
    require_cuda(x, "csp_prefill")
    batch, seqlen, nheads, headdim = x.shape
    _, _, nchunks, chunk_size = dt.shape
    _, _, ngroups, dstate = B.shape
    BN, BK, nw, st = CSP_PREFILL_CFG
    if ngroups != 1 or headdim not in (16, 32, 64, 128) or (dstate % BN) or (chunk_size % BK):
        raise RuntimeError(f"neuronspark.csp_prefill: ngroups {ngroups} headdim {headdim} dstate {dstate} BN {BN} chunk "
                           f"{chunk_size} BK {BK}")
    if B.shape != (batch, seqlen, ngroups, dstate) or dt.shape != (batch, nheads, nchunks, chunk_size):
        raise RuntimeError(f"neuronspark.csp_prefill: shapes B {tuple(B.shape)} dt {tuple(dt.shape)} x {tuple(x.shape)}")
    if dA_cumsum.shape != dt.shape:
        raise RuntimeError("neuronspark.csp_prefill: dA_cumsum must have dt's shape")
    if tuple(initial_state.shape) != (batch, nheads, headdim, dstate) or initial_state.dtype != torch.float32:
        raise RuntimeError(f"neuronspark.csp_prefill: initial_state must be fp32 {(batch, nheads, headdim, dstate)}, got "
                           f"{initial_state.dtype} {tuple(initial_state.shape)}")
    if B.stride(-1) != 1 or x.stride(-1) != 1 or dt.stride(-1) != 1 or dA_cumsum.stride(-1) != 1:
        raise RuntimeError("neuronspark.csp_prefill: unit last strides of B / x / dt / dA_cumsum expected")
    if nchunks < 1 or nchunks > CSP_MAX_CHUNKS or not (nchunks - 1) * chunk_size < seqlen <= nchunks * chunk_size:
        raise RuntimeError(f"neuronspark.csp_prefill: nchunks {nchunks} for seqlen {seqlen} (1 .. {CSP_MAX_CHUNKS} "
                           "chunks of the sequence)")
    states = torch.empty((batch, nchunks, nheads, headdim, dstate), device=x.device, dtype=out_dtype)
    final = torch.empty((batch, nheads, headdim, dstate), device=x.device, dtype=torch.float32)
    xd = _desc3(x, BK)
    ini = initial_state
    grid = (triton.cdiv(dstate, BN), batch, nheads)
    with torch.cuda.device(x.device.index):
        _csp_prefill_kernel[grid](
            xd, B, ini, states, final, dt, dA_cumsum,
            chunk_size, dstate, seqlen,
            B.stride(0), B.stride(1), B.stride(-1),
            ini.stride(0), ini.stride(1), ini.stride(2), ini.stride(3),
            states.stride(0), states.stride(1), states.stride(2), states.stride(3), states.stride(4),
            final.stride(0), final.stride(1), final.stride(2), final.stride(3),
            dt.stride(0), dt.stride(2), dt.stride(1), dt.stride(3),
            dA_cumsum.stride(0), dA_cumsum.stride(2), dA_cumsum.stride(1), dA_cumsum.stride(3),
            NCHUNK=nchunks, P=headdim, BN=BN, BK=BK, num_warps=nw, num_stages=st)
    return states, final


def spd_fused(states_pre, dA_chunk_cumsum, dout, C, dA_cumsum, out_dtype):
    """(dstates (b, c, h, P N) in out_dtype, ddA workspace (b, h, c, n_blocks) fp32 (blocks < n_valid written),
    n_valid, states_converted (b, c, h, P N) in out_dtype) = the stock ``_chunk_scan_bwd_dstates(C, dA_cumsum, dout,
    dtype=fp32)`` fed with the stock fp32 state recompute to the stock ``_state_passing_bwd`` (see _spd_kernel).
    states_pre: (batch, nchunks, nheads, P N) fp32, unit last stride; dA_chunk_cumsum: (batch, nheads, nchunks) (any
    strides); dout: (batch, seqlen, nheads, P) with unit last stride; C: (batch, seqlen, 1, N) unit last stride;
    dA_cumsum: (batch, nheads, nchunks, chunk_size) unit last stride."""
    require_cuda(states_pre, "spd_fused")
    batch, nchunks, nheads, dim = states_pre.shape
    _, seqlen, _, headdim = dout.shape
    _, _, ngroups, dstate = C.shape
    chunk_size = dA_cumsum.shape[-1]
    BK, nw, st, PB = SPD_CFG
    if ngroups != 1 or dim != headdim * dstate or headdim not in (16, 32, 64, 128) or dstate not in (64, 128, 256) \
            or (chunk_size % BK) or (512 % dstate) or (headdim % (512 // dstate)) or (headdim % PB) \
            or (PB % (512 // dstate)) or PB < 16:
        raise RuntimeError(f"neuronspark.spd_fused: ngroups {ngroups} dim {dim} headdim {headdim} dstate {dstate} chunk "
                           f"{chunk_size} BK {BK} PB {PB}")
    if dA_chunk_cumsum.shape != (batch, nheads, nchunks) or dA_cumsum.shape != (batch, nheads, nchunks, chunk_size) \
            or dout.shape[2] != nheads:
        raise RuntimeError("neuronspark.spd_fused: shapes of dA_chunk_cumsum / dA_cumsum / dout")
    if states_pre.dtype != torch.float32 or states_pre.stride(-1) != 1 or dout.stride(-1) != 1 or C.stride(-1) != 1 \
            or dA_cumsum.stride(-1) != 1:
        raise RuntimeError("neuronspark.spd_fused: fp32 states_pre with unit last stride; unit last strides of dout / C "
                           "/ dA_cumsum")
    if nchunks < 2 or nchunks > 64:
        raise RuntimeError(f"neuronspark.spd_fused: nchunks {nchunks} (the SD call has 2 .. 64 chunks)")
    RB = 512 // dstate
    dstates = torch.empty((batch, nchunks, nheads, dim), device=states_pre.device, dtype=out_dtype)
    conv = torch.empty((batch, nchunks, nheads, dim), device=states_pre.device, dtype=out_dtype)
    n_blocks = (dim + 64 - 1) // 64
    ws = torch.empty(batch, nheads, nchunks, n_blocks, dtype=torch.float32, device=states_pre.device)
    n_valid = headdim // RB
    if n_valid != (dim + 511) // 512 or n_valid > n_blocks:
        raise RuntimeError(f"neuronspark.spd_fused: {n_valid} workspace blocks, the stock BLOCK 512 writes "
                           f"{(dim + 511) // 512}")
    grid = (headdim // PB, batch, nheads)
    with torch.cuda.device(states_pre.device.index):
        _spd_kernel[grid](
            states_pre, dA_chunk_cumsum, dout, C, dA_cumsum, dstates, conv, ws,
            chunk_size, seqlen, dstate,
            states_pre.stride(0), states_pre.stride(1), states_pre.stride(2),
            dA_chunk_cumsum.stride(0), dA_chunk_cumsum.stride(2), dA_chunk_cumsum.stride(1),
            dout.stride(0), dout.stride(1), dout.stride(2), dout.stride(3),
            C.stride(0), C.stride(1), C.stride(3),
            dA_cumsum.stride(0), dA_cumsum.stride(2), dA_cumsum.stride(1), dA_cumsum.stride(3),
            dstates.stride(0), dstates.stride(1), dstates.stride(2),
            conv.stride(0), conv.stride(1), conv.stride(2),
            ws.stride(0), ws.stride(2), ws.stride(1),
            NCHUNK=nchunks, P=headdim, N=dstate, BK=BK, RB=RB, PB=PB, num_warps=nw, num_stages=st)
    return dstates, ws, n_valid, conv


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
    327-350) with the chunk-state kernel's operand rounding (``_csp_kernel``, as the stock ``_chunk_state_fwd_kernel``:
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
    """Reference of ``csp_prefill``: the replaced pair, mamba_ssm ``_chunk_state_fwd(.., states_in_fp32=True)`` then
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


def spd_fused_ref(states_pre, dA_chunk_cumsum, dout, C, dA_cumsum, out_dtype):
    """Reference of ``spd_fused``: dstates_in[k] = sum_{l in k} bf16(dout_l exp(dA_l)) C_l^T (the stock
    _chunk_scan_bwd_dstates with its operand rounding, ``_spd_kernel``: the fp32 product ``dout exp(dA_l)`` rounded to
    dout's dtype before the dot; float64 sums), the forward states recomputed
    from states_pre, and the reverse scan of _state_passing_bwd: d_{nc-1} = 0, ws[k, j] = sum_{block j} (s_k d_k)
    exp(dA_last[k]), d_{k-1} = exp(dA_last[k]) d_k + dstates_in[k]; ws[0] = 0, states_converted[0] = 0.  Returns
    (dstates, ws (b, h, c, n_blocks) with blocks >= n_valid zero, n_valid, states_converted)."""
    batch, nchunks, nheads, dim = states_pre.shape
    _, seqlen, _, headdim = dout.shape
    dstate = C.shape[-1]
    chunk = dA_cumsum.shape[-1]
    pre = states_pre.float().view(batch, nchunks, nheads, headdim, dstate)
    dAl = dA_chunk_cumsum.float()                                      # (b, h, c)
    S = _forward_states(pre, dAl)                                      # (b, c, h, p, n) float64
    do = _chunked(dout, nchunks, chunk).float()                        # (b, c, l, h, p) bf16-exact fp32
    Cc = _chunked(C[:, :, 0], nchunks, chunk)                          # (b, c, l, n)
    do_s = (do * _exp_f32(dA_cumsum).permute(0, 2, 3, 1)[..., None]).to(dout.dtype).double()
    din = torch.einsum("bclhp,bcln->bchpn", do_s, Cc)
    n_blocks = (dim + 63) // 64
    n_valid = (dim + 511) // 512
    ws = torch.zeros(batch, nheads, nchunks, n_blocks, dtype=torch.float64, device=states_pre.device)
    d = torch.zeros_like(S[:, 0])
    dst = [None] * nchunks
    dst[nchunks - 1] = d
    decay = _exp_f32(dAl).double()
    for k in range(nchunks - 1, 0, -1):
        sc = decay[:, :, k]
        ws[:, :, k, :n_valid] = (S[:, k] * d).reshape(batch, nheads, n_valid, -1).sum(-1) * sc[..., None]
        d = sc[..., None, None] * d + din[:, k]
        dst[k - 1] = d
    dstates = torch.stack(dst, dim=1).reshape(batch, nchunks, nheads, dim)
    return dstates.to(out_dtype), ws.float(), n_valid, S.reshape(batch, nchunks, nheads, dim).to(out_dtype)


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
