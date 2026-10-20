# Copyright (c) 2024, Tri Dao, Albert Gu.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Vendored from mamba_ssm 2.3.2.post1, mamba_ssm/ops/triton/ssd_chunk_state.py
# (sha256 6d82771b2f62a7bf84c381c8ef8e4b579a95e0352021d117507a1072e4d71dcd).
# Modified by NeuronSpark: autotune removed (fixed configs), imports rewritten, unused functions deleted.
#   * _chunk_cumsum_fwd_kernel (source lines 39-86), _chunk_cumsum_bwd_kernel (source lines 101-174) and
#     _chunk_state_fwd_kernel (source lines 191-268): the @triton.jit bodies are byte-identical; their @triton.autotune
#     decorators (lines 27-38, 89-100, 177-190) are removed.  _chunk_cumsum_bwd_kernel has no launcher here: its only
#     caller (neuronspark.ops.triton.dendritic_glue.chunk_cumsum_bwd) launches it with a fixed configuration.
#   * _chunk_cumsum_fwd (source lines 718-741) and _chunk_state_fwd (source lines 812-842): the launchers take the
#     launch configuration as the keyword argument ``config = (kwargs, num_warps, num_stages)`` (the constants are in
#     neuronspark.ops.triton.mamba) and compute the grid from it; every other line is the source's.
#   * import: ``softplus`` from the vendored neuronspark.ops.triton.mamba.softplus (was mamba_ssm.ops.triton.softplus).
#   * removed: init_to_zero (its only users were the removed autotune configurations), the einops,
#     torch.nn.functional and mamba_ssm.utils.determinism imports, and every other kernel / function of the file.
# See PROVENANCE.md.

"""We want triton==2.1.0 or 2.2.0 for this
"""

import math
import torch

import triton
import triton.language as tl

from neuronspark.ops.triton.mamba.softplus import softplus


@triton.jit
def _chunk_cumsum_fwd_kernel(
    # Pointers to matrices
    dt_ptr, A_ptr, dt_bias_ptr, dt_out_ptr, dA_cumsum_ptr,
    # Matrix dimension
    batch, seqlen, nheads, chunk_size,
    dt_min, dt_max,
    # Strides
    stride_dt_batch, stride_dt_seqlen, stride_dt_head,
    stride_A_head,
    stride_dt_bias_head,
    stride_dt_out_batch, stride_dt_out_chunk, stride_dt_out_head, stride_dt_out_csize,
    stride_dA_cs_batch, stride_dA_cs_chunk, stride_dA_cs_head, stride_dA_cs_csize,
    # Meta-parameters
    DT_SOFTPLUS: tl.constexpr,
    HAS_DT_BIAS: tl.constexpr,
    BLOCK_SIZE_H: tl.constexpr, BLOCK_SIZE_CHUNK: tl.constexpr,
):
    pid_b = tl.program_id(axis=0)
    pid_c = tl.program_id(axis=1)
    pid_h = tl.program_id(axis=2)
    dt_ptr += pid_b * stride_dt_batch + pid_c * chunk_size * stride_dt_seqlen
    dt_out_ptr += pid_b * stride_dt_out_batch + pid_c * stride_dt_out_chunk
    dA_cumsum_ptr += pid_b * stride_dA_cs_batch + pid_c * stride_dA_cs_chunk

    offs_h = pid_h * BLOCK_SIZE_H + tl.arange(0, BLOCK_SIZE_H)
    offs_c = tl.arange(0, BLOCK_SIZE_CHUNK)
    dt_ptrs = dt_ptr + (offs_h[:, None] * stride_dt_head + offs_c[None, :] * stride_dt_seqlen)
    A_ptrs = A_ptr + offs_h * stride_A_head
    dt_out_ptrs = dt_out_ptr + (offs_h[:, None] * stride_dt_out_head + offs_c[None, :] * stride_dt_out_csize)
    dA_cs_ptrs = dA_cumsum_ptr + (offs_h[:, None] * stride_dA_cs_head + offs_c[None, :] * stride_dA_cs_csize)
    chunk_size_limit = min(chunk_size, seqlen - pid_c * chunk_size)

    dt = tl.load(dt_ptrs, mask=(offs_h[:, None] < nheads) & (offs_c[None, :] < chunk_size_limit), other=0.0).to(tl.float32)
    if HAS_DT_BIAS:
        dt_bias = tl.load(dt_bias_ptr + offs_h * stride_dt_bias_head, mask=offs_h < nheads, other=0.0).to(tl.float32)
        dt += dt_bias[:, None]
    if DT_SOFTPLUS:
        dt = tl.where(dt <= 20.0, softplus(dt), dt)
    # As of Triton 2.2.0, tl.clamp is not available yet
    # dt = tl.clamp(dt, dt_min, dt_max)
    dt = tl.minimum(tl.maximum(dt, dt_min), dt_max)
    dt = tl.where((offs_h[:, None] < nheads) & (offs_c[None, :] < chunk_size_limit), dt, 0.0)
    tl.store(dt_out_ptrs, dt, mask=(offs_h[:, None] < nheads) & (offs_c[None, :] < chunk_size))
    A = tl.load(A_ptrs, mask=offs_h < nheads, other=0.0).to(tl.float32)
    dA = dt * A[:, None]
    dA_cs = tl.cumsum(dA, axis=1)
    tl.store(dA_cs_ptrs, dA_cs, mask=(offs_h[:, None] < nheads) & (offs_c[None, :] < chunk_size))


@triton.jit
def _chunk_cumsum_bwd_kernel(
    # Pointers to matrices
    ddA_ptr, ddt_out_ptr, dt_ptr, A_ptr, dt_bias_ptr,
    ddt_ptr, dA_ptr, ddt_bias_ptr,
    # Matrix dimensions
    batch, seqlen, nheads, chunk_size,
    dt_min, dt_max,
    # Strides
    stride_ddA_batch, stride_ddA_chunk, stride_ddA_head, stride_ddA_csize,
    stride_ddt_out_batch, stride_ddt_out_chunk, stride_ddt_out_head, stride_ddt_out_csize,
    stride_dt_batch, stride_dt_seqlen, stride_dt_head,
    stride_A_head,
    stride_dt_bias_head,
    stride_ddt_batch, stride_ddt_seqlen, stride_ddt_head,
    stride_dA_batch, stride_dA_chunk, stride_dA_head,
    stride_ddt_bias_batch, stride_ddt_bias_chunk, stride_ddt_bias_head,
    # Meta-parameters
    DT_SOFTPLUS: tl.constexpr,
    HAS_DT_BIAS: tl.constexpr,
    BLOCK_SIZE_H: tl.constexpr, BLOCK_SIZE_CHUNK: tl.constexpr,
    DETERMINISTIC_REDUCTION: tl.constexpr,
):
    pid_b = tl.program_id(axis=0)
    pid_c = tl.program_id(axis=1)
    pid_h = tl.program_id(axis=2)
    ddt_out_ptr += pid_b * stride_ddt_out_batch + pid_c * stride_ddt_out_chunk
    ddA_ptr += pid_b * stride_ddA_batch + pid_c * stride_ddA_chunk
    dt_ptr += pid_b * stride_dt_batch + pid_c * chunk_size * stride_dt_seqlen
    ddt_ptr += pid_b * stride_ddt_batch + pid_c * chunk_size * stride_ddt_seqlen

    offs_h = pid_h * BLOCK_SIZE_H + tl.arange(0, BLOCK_SIZE_H)
    offs_c = tl.arange(0, BLOCK_SIZE_CHUNK)
    ddt_out_ptrs = ddt_out_ptr + (offs_h[:, None] * stride_ddt_out_head + offs_c[None, :] * stride_ddt_out_csize)
    ddA_ptrs = ddA_ptr + (offs_h[:, None] * stride_ddA_head + offs_c[None, :] * stride_ddA_csize)
    dt_ptrs = dt_ptr + (offs_h[:, None] * stride_dt_head + offs_c[None, :] * stride_dt_seqlen)
    ddt_ptrs = ddt_ptr + (offs_h[:, None] * stride_ddt_head + offs_c[None, :] * stride_ddt_seqlen)
    A_ptrs = A_ptr + offs_h * stride_A_head
    chunk_size_limit = min(chunk_size, seqlen - pid_c * chunk_size)

    ddA = tl.load(ddA_ptrs, mask=(offs_h[:, None] < nheads) & (offs_c[None, :] < chunk_size_limit), other=0.0).to(tl.float32)
    ddt_out = tl.load(ddt_out_ptrs, mask=(offs_h[:, None] < nheads) & (offs_c[None, :] < chunk_size_limit), other=0.0).to(tl.float32)
    A = tl.load(A_ptrs, mask=offs_h < nheads, other=0.0).to(tl.float32)
    ddt = ddA * A[:, None] + ddt_out
    dt = tl.load(dt_ptrs, mask=(offs_h[:, None] < nheads) & (offs_c[None, :] < chunk_size_limit), other=0.0).to(tl.float32)
    if HAS_DT_BIAS:
        dt_bias = tl.load(dt_bias_ptr + offs_h * stride_dt_bias_head, mask=offs_h < nheads, other=0.0).to(tl.float32)
        dt += dt_bias[:, None]
    if DT_SOFTPLUS:
        dt_presoftplus = dt
        dt = tl.where(dt <= 20.0, softplus(dt), dt)
    clamp_mask = (dt < dt_min) | (dt > dt_max)
    # As of Triton 2.2.0, tl.clamp is not available yet
    # dt = tl.clamp(dt, dt_min, dt_max)
    dt = tl.minimum(tl.maximum(dt, dt_min), dt_max)
    dt = tl.where((offs_h[:, None] < nheads) & (offs_c[None, :] < chunk_size_limit), dt, 0.0)
    ddt = tl.where((offs_h[:, None] < nheads) & (offs_c[None, :] < chunk_size_limit), ddt, 0.0)
    ddt = tl.where(clamp_mask, 0.0, ddt)
    if DT_SOFTPLUS:
        ddt = tl.where(dt_presoftplus <= 20.0, ddt * tl.sigmoid(dt_presoftplus), ddt)
    tl.store(ddt_ptrs, ddt, mask=(offs_h[:, None] < nheads) & (offs_c[None, :] < chunk_size_limit))
    dA = tl.sum(ddA * dt, axis=1)
    dA_ptr += pid_b * stride_dA_batch + pid_c * stride_dA_chunk
    if DETERMINISTIC_REDUCTION:
        tl.store(dA_ptr + offs_h * stride_dA_head, dA, mask=offs_h < nheads)
    else:
        tl.atomic_add(dA_ptr + offs_h * stride_dA_head, dA, mask=offs_h < nheads)
    if HAS_DT_BIAS:
        ddt_bias = tl.sum(ddt, axis=1)
        ddt_bias_ptr += pid_b * stride_ddt_bias_batch + pid_c * stride_ddt_bias_chunk
        if DETERMINISTIC_REDUCTION:
            tl.store(ddt_bias_ptr + offs_h * stride_ddt_bias_head, ddt_bias, mask=offs_h < nheads)
        else:
            tl.atomic_add(ddt_bias_ptr + offs_h * stride_ddt_bias_head, ddt_bias, mask=offs_h < nheads)


@triton.jit
def _chunk_state_fwd_kernel(
    # Pointers to matrices
    x_ptr, b_ptr, states_ptr, dt_ptr, dA_cumsum_ptr, seq_idx_ptr,
    # Matrix dimensions
    hdim, dstate, chunk_size,
    batch, seqlen, nheads_ngroups_ratio,
    # Strides
    stride_x_batch, stride_x_seqlen, stride_x_head, stride_x_hdim,
    stride_b_batch, stride_b_seqlen, stride_b_head, stride_b_dstate,
    stride_states_batch, stride_states_chunk, stride_states_head, stride_states_hdim, stride_states_dstate,
    stride_dt_batch, stride_dt_chunk, stride_dt_head, stride_dt_csize,
    stride_dA_cs_batch, stride_dA_cs_chunk, stride_dA_cs_head, stride_dA_cs_csize,
    stride_seq_idx_batch, stride_seq_idx_seqlen,
    # Meta-parameters
    HAS_SEQ_IDX: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
):
    pid_bc = tl.program_id(axis=1)
    pid_c = pid_bc // batch
    pid_b = pid_bc - pid_c * batch
    pid_h = tl.program_id(axis=2)
    num_pid_n = tl.cdiv(dstate, BLOCK_SIZE_N)
    pid_m = tl.program_id(axis=0) // num_pid_n
    pid_n = tl.program_id(axis=0) % num_pid_n
    b_ptr += pid_b * stride_b_batch + pid_c * chunk_size * stride_b_seqlen + (pid_h // nheads_ngroups_ratio) * stride_b_head
    x_ptr += pid_b * stride_x_batch + pid_c * chunk_size * stride_x_seqlen + pid_h * stride_x_head
    dt_ptr += pid_b * stride_dt_batch + pid_c * stride_dt_chunk + pid_h * stride_dt_head
    dA_cumsum_ptr += pid_b * stride_dA_cs_batch + pid_c * stride_dA_cs_chunk + pid_h * stride_dA_cs_head
    if HAS_SEQ_IDX:
        seq_idx_ptr += pid_b * stride_seq_idx_batch + pid_c * chunk_size * stride_seq_idx_seqlen

    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    x_ptrs = x_ptr + (offs_m[:, None] * stride_x_hdim + offs_k[None, :] * stride_x_seqlen)
    b_ptrs = b_ptr + (offs_n[None, :] * stride_b_dstate + offs_k[:, None] * stride_b_seqlen)
    dt_ptrs = dt_ptr + offs_k * stride_dt_csize
    dA_cs_last = tl.load(dA_cumsum_ptr + (chunk_size - 1) * stride_dA_cs_csize).to(tl.float32)
    dA_cumsum_ptrs = dA_cumsum_ptr + offs_k * stride_dA_cs_csize
    if HAS_SEQ_IDX:
        seq_idx_ptrs = seq_idx_ptr + offs_k * stride_seq_idx_seqlen

    chunk_size_limit = min(chunk_size, seqlen - pid_c * chunk_size)
    if HAS_SEQ_IDX:
        seq_idx_last = tl.load(seq_idx_ptr + (chunk_size_limit - 1) * stride_seq_idx_seqlen)

    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, chunk_size_limit, BLOCK_SIZE_K):
        x = tl.load(x_ptrs, mask=(offs_m[:, None] < hdim) & (offs_k[None, :] < chunk_size_limit - k), other=0.0)
        b = tl.load(b_ptrs, mask=(offs_k[:, None] < chunk_size_limit - k) & (offs_n[None, :] < dstate), other=0.0).to(tl.float32)
        dA_cs_k = tl.load(dA_cumsum_ptrs, mask=offs_k < chunk_size_limit - k, other=0.0).to(tl.float32)
        if HAS_SEQ_IDX:
            seq_idx_k = tl.load(seq_idx_ptrs, mask=offs_k < chunk_size_limit - k, other=-1)
        dt_k = tl.load(dt_ptrs, mask=offs_k < chunk_size_limit - k, other=0.0).to(tl.float32)
        if not HAS_SEQ_IDX:
            # scale = tl.exp((dA_cs_last - dA_cs_k)) * dt_k
            scale = tl.exp(tl.minimum((dA_cs_last - dA_cs_k), 0.0)) * dt_k
        else:
            # scale = tl.where(seq_idx_k == seq_idx_last, tl.exp((dA_cs_last - dA_cs_k)) * dt_k, 0.0)
            scale = tl.where((seq_idx_last >= 0) & (seq_idx_k == seq_idx_last), tl.exp(tl.minimum((dA_cs_last - dA_cs_k), 0.0)) * dt_k, 0.0)
        b *= scale[:, None]
        b = b.to(x_ptr.dtype.element_ty)
        acc += tl.dot(x, b)
        x_ptrs += BLOCK_SIZE_K * stride_x_seqlen
        b_ptrs += BLOCK_SIZE_K * stride_b_seqlen
        dt_ptrs += BLOCK_SIZE_K * stride_dt_csize
        dA_cumsum_ptrs += BLOCK_SIZE_K * stride_dA_cs_csize
        if HAS_SEQ_IDX:
            seq_idx_ptrs += BLOCK_SIZE_K * stride_seq_idx_seqlen
    states = acc.to(states_ptr.dtype.element_ty)

    states_ptr += pid_b * stride_states_batch + pid_c * stride_states_chunk + pid_h * stride_states_head
    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    states_ptrs = states_ptr + (offs_m[:, None] * stride_states_hdim + offs_n[None, :] * stride_states_dstate)
    c_mask = (offs_m[:, None] < hdim) & (offs_n[None, :] < dstate)
    tl.store(states_ptrs, states, mask=c_mask)


def _chunk_cumsum_fwd(dt, A, chunk_size, dt_bias=None, dt_softplus=False, dt_limit=(0.0, float("inf")), *, config):
    batch, seqlen, nheads = dt.shape
    assert A.shape == (nheads,)
    if dt_bias is not None:
        assert dt_bias.shape == (nheads,)
    nchunks = math.ceil(seqlen / chunk_size)
    dt_out = torch.empty(batch, nheads, nchunks, chunk_size, device=dt.device, dtype=torch.float32)
    dA_cumsum = torch.empty(batch, nheads, nchunks, chunk_size, device=dt.device, dtype=torch.float32)
    meta, num_warps, num_stages = config
    grid_chunk_cs = (batch, nchunks, triton.cdiv(nheads, meta['BLOCK_SIZE_H']))
    with torch.cuda.device(dt.device.index):
        _chunk_cumsum_fwd_kernel[grid_chunk_cs](
            dt, A, dt_bias, dt_out, dA_cumsum,
            batch, seqlen, nheads, chunk_size,
            dt_limit[0], dt_limit[1],
            dt.stride(0), dt.stride(1), dt.stride(2),
            A.stride(0),
            dt_bias.stride(0) if dt_bias is not None else 0,
            dt_out.stride(0), dt_out.stride(2), dt_out.stride(1), dt_out.stride(3),
            dA_cumsum.stride(0), dA_cumsum.stride(2), dA_cumsum.stride(1), dA_cumsum.stride(3),
            dt_softplus,
            HAS_DT_BIAS=dt_bias is not None,
            BLOCK_SIZE_CHUNK=triton.next_power_of_2(chunk_size),
            **meta, num_warps=num_warps, num_stages=num_stages,
        )
    return dA_cumsum, dt_out


def _chunk_state_fwd(B, x, dt, dA_cumsum, seq_idx=None, states=None, states_in_fp32=True, *, config):
    batch, seqlen, nheads, headdim = x.shape
    _, _, nchunks, chunk_size = dt.shape
    _, _, ngroups, dstate = B.shape
    assert nheads % ngroups == 0
    assert B.shape == (batch, seqlen, ngroups, dstate)
    assert dt.shape == (batch, nheads, nchunks, chunk_size)
    assert dA_cumsum.shape == dt.shape
    if seq_idx is not None:
        assert seq_idx.shape == (batch, seqlen)
    if states is not None:
        assert states.shape == (batch, nchunks, nheads, headdim, dstate)
    else:
        states_dtype = torch.float32 if states_in_fp32 else B.dtype
        states = torch.empty((batch, nchunks, nheads, headdim, dstate), device=x.device, dtype=states_dtype)
    meta, num_warps, num_stages = config
    grid = (triton.cdiv(headdim, meta['BLOCK_SIZE_M']) * triton.cdiv(dstate, meta['BLOCK_SIZE_N']),
            batch * nchunks, nheads)
    with torch.cuda.device(x.device.index):
        _chunk_state_fwd_kernel[grid](
            x, B, states, dt, dA_cumsum, seq_idx,
            headdim, dstate, chunk_size,
            batch, seqlen, nheads // ngroups,
            x.stride(0), x.stride(1), x.stride(2), x.stride(3),
            B.stride(0), B.stride(1), B.stride(2), B.stride(-1),
            states.stride(0), states.stride(1), states.stride(2), states.stride(3), states.stride(4),
            dt.stride(0), dt.stride(2), dt.stride(1), dt.stride(3),
            dA_cumsum.stride(0), dA_cumsum.stride(2), dA_cumsum.stride(1), dA_cumsum.stride(3),
            *((seq_idx.stride(0), seq_idx.stride(1)) if seq_idx is not None else (0, 0)),
            HAS_SEQ_IDX=seq_idx is not None,
            **meta, num_warps=num_warps, num_stages=num_stages,
        )
    return states
