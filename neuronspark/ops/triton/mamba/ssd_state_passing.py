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
# Vendored from mamba_ssm 2.3.2.post1, mamba_ssm/ops/triton/ssd_state_passing.py
# (sha256 ae1fab6c680cb5312ef746862a2cf8e74157033c80878490cc64fa1f24f88b70).
# Modified by NeuronSpark: autotune removed (fixed configs), imports rewritten, unused functions deleted.
#   * _state_passing_fwd_kernel (source lines 29-87): the @triton.jit body is byte-identical; its @triton.autotune
#     decorator (lines 18-28) is removed.
#   * _state_passing_fwd (source lines 196-224): the launcher takes the launch configuration as the keyword argument
#     ``config = (kwargs, num_warps, num_stages)`` (the constants are in neuronspark.ops.triton.mamba) and computes the
#     grid from it; every other line is the source's.
#   * removed: the einops, torch.nn.functional and mamba_ssm.utils.determinism imports, and every other kernel /
#     function of the file.
# See PROVENANCE.md.

"""We want triton==2.1.0 or 2.2.0 for this
"""

import torch

import triton
import triton.language as tl


@triton.jit
def _state_passing_fwd_kernel(
    # Pointers to matrices
    states_ptr, out_ptr, final_states_ptr, dA_cs_ptr, initstates_ptr, seq_idx_ptr,
    # Matrix dimensions
    dim, nchunks, seqlen, chunk_size,
    # Strides
    stride_states_batch, stride_states_chunk, stride_states_head, stride_states_dim,
    stride_out_batch, stride_out_chunk, stride_out_head, stride_out_dim,
    stride_final_states_batch, stride_final_states_head, stride_final_states_dim,
    stride_dA_cs_batch, stride_dA_cs_chunk, stride_dA_cs_head,
    stride_initstates_batch, stride_initstates_head, stride_initstates_dim,
    stride_seq_idx_batch, stride_seq_idx_seqlen,
    # Meta-parameters
    HAS_INITSTATES: tl.constexpr,
    HAS_SEQ_IDX: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid_b = tl.program_id(axis=1)
    pid_h = tl.program_id(axis=2)
    pid_m = tl.program_id(axis=0)
    states_ptr += pid_b * stride_states_batch + pid_h * stride_states_head
    dA_cs_ptr += pid_b * stride_dA_cs_batch + pid_h * stride_dA_cs_head
    out_ptr += pid_b * stride_out_batch + pid_h * stride_out_head
    final_states_ptr += pid_b * stride_final_states_batch + pid_h * stride_final_states_head
    if HAS_INITSTATES:
        initstates_ptr += pid_b * stride_initstates_batch + pid_h * stride_initstates_head
    if HAS_SEQ_IDX:
        seq_idx_ptr += pid_b * stride_seq_idx_batch

    offs_m = pid_m * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    states_ptrs = states_ptr + offs_m * stride_states_dim
    out_ptrs = out_ptr + offs_m * stride_out_dim
    final_states_ptrs = final_states_ptr + offs_m * stride_final_states_dim

    if not HAS_INITSTATES:
        states = tl.zeros((BLOCK_SIZE, ), dtype=tl.float32)
    else:
        initstates_ptrs = initstates_ptr + offs_m * stride_initstates_dim
        states = tl.load(initstates_ptrs, mask=offs_m < dim, other=0.0).to(tl.float32)
    tl.store(out_ptrs, states, mask=offs_m < dim)
    out_ptrs += stride_out_chunk
    seq_idx = 0
    for c in range(nchunks):
        new_states = tl.load(states_ptrs, mask=offs_m < dim, other=0.0).to(tl.float32)
        dA_cs = tl.load(dA_cs_ptr).to(tl.float32)
        scale = tl.exp(dA_cs)
        if HAS_SEQ_IDX:
            seq_idx_new = tl.load(seq_idx_ptr + (min((c + 1) * chunk_size, seqlen) - 1) * stride_seq_idx_seqlen)
            scale = tl.where(seq_idx_new == seq_idx, scale, 0.0)
            seq_idx = seq_idx_new
        states = scale * states + new_states
        if c < nchunks - 1:
            tl.store(out_ptrs, states, mask=offs_m < dim)
        else:
            tl.store(final_states_ptrs, states, mask=offs_m < dim)
        states_ptrs += stride_states_chunk
        dA_cs_ptr += stride_dA_cs_chunk
        out_ptrs += stride_out_chunk


def _state_passing_fwd(states, dA_chunk_cumsum, initial_states=None, seq_idx=None, chunk_size=None,
                       out_dtype=None, *, config):
    batch, nchunks, nheads, dim = states.shape
    assert dA_chunk_cumsum.shape == (batch, nheads, nchunks)
    if initial_states is not None:
        assert initial_states.shape == (batch, nheads, dim)
    if seq_idx is not None:
        assert chunk_size is not None
        seqlen = seq_idx.shape[-1]
        assert seq_idx.shape == (batch, seqlen)
    out_dtype = states.dtype if out_dtype is None else out_dtype
    out = torch.empty((batch, nchunks, nheads, dim), device=states.device, dtype=out_dtype)
    final_states = torch.empty((batch, nheads, dim), device=states.device, dtype=torch.float32)
    meta, num_warps, num_stages = config
    grid = (triton.cdiv(dim, meta['BLOCK_SIZE']), batch, nheads)
    with torch.cuda.device(states.device.index):
        _state_passing_fwd_kernel[grid](
            states, out, final_states, dA_chunk_cumsum, initial_states, seq_idx,
            dim, nchunks, seqlen if seq_idx is not None else 0, chunk_size if seq_idx is not None else 0,
            states.stride(0), states.stride(1), states.stride(2), states.stride(3),
            out.stride(0), out.stride(1), out.stride(2), out.stride(3),
            final_states.stride(0), final_states.stride(1), final_states.stride(2),
            dA_chunk_cumsum.stride(0), dA_chunk_cumsum.stride(2), dA_chunk_cumsum.stride(1),
            *((initial_states.stride(0), initial_states.stride(1), initial_states.stride(2))
              if initial_states is not None else (0, 0, 0)),
            *((seq_idx.stride(0), seq_idx.stride(1)) if seq_idx is not None else (0, 0)),
            HAS_INITSTATES=initial_states is not None,
            HAS_SEQ_IDX=seq_idx is not None,
            **meta, num_warps=num_warps, num_stages=num_stages,
        )
    return out, final_states
