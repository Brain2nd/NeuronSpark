"""SSD Triton kernels vendored from mamba_ssm 2.3.2.post1 (Apache-2.0; see LICENSE and PROVENANCE.md in this directory).

Only the kernels the SD layer runs are kept: ``_chunk_cumsum_fwd_kernel`` / ``_chunk_cumsum_bwd_kernel`` /
``_chunk_state_fwd_kernel`` (ssd_chunk_state.py), ``_bmm_chunk_fwd_kernel`` / ``_bmm_chunk_bwd_kernel`` (ssd_bmm.py),
``_state_passing_fwd_kernel`` (ssd_state_passing.py) and ``softplus`` (softplus.py).  Their bodies are the source's
byte for byte; the autotune decorators are removed and every launch passes one of the fixed launch configurations
below, ``(kwargs, num_warps, num_stages)``.  Nothing is tuned at run time and no environment variable is read.

Launch configurations (the derivations are in PROVENANCE.md):

* ``CHUNK_CUMSUM_FWD_TRAIN``  -- training forward of the SD layer (nheads 144).
* ``CHUNK_CUMSUM_FWD_PREFILL`` -- inference prefill (the single configuration mamba_ssm's deterministic mode keeps).
* ``CHUNK_CUMSUM_BWD_TRAIN``  -- training backward (launched by
  :func:`neuronspark.ops.triton.dendritic_glue.chunk_cumsum_bwd`).
* ``BMM_CHUNK_FWD``           -- CB = C B^T of the training forward and of the prefill (deterministic-mode configuration).
* ``BMM_CHUNK_BWD_TRAIN``     -- the two ``_bmm_chunk_bwd`` calls of the training backward.
* ``CHUNK_STATE_FWD_PREFILL`` / ``STATE_PASSING_FWD_PREFILL`` -- the prefill chunk states and state passing.

Workspace geometry constants: mamba_ssm sizes the deterministic tile workspaces of two backward kernels that the SD
layer replaces by its own kernels (``neuronspark.ops.triton.sd_states.db_tma`` / ``dc_tma``) from the smallest
``BLOCK_SIZE_N`` of the kernel's configuration list as mamba_ssm's deterministic mode leaves it (one configuration).
The replacement kernels keep that workspace geometry, so the values are fixed here: 32 for both
(``_chunk_state_bwd_db_kernel``: {M 32, N 32}; ``_chunk_scan_bwd_dc_kernel``: {M 32, N 32}).
"""
from __future__ import annotations

from neuronspark.ops.triton.mamba.softplus import softplus
from neuronspark.ops.triton.mamba.ssd_bmm import _bmm_chunk_bwd, _bmm_chunk_bwd_kernel, _bmm_chunk_fwd, _bmm_chunk_fwd_kernel
from neuronspark.ops.triton.mamba.ssd_chunk_state import (
    _chunk_cumsum_bwd_kernel,
    _chunk_cumsum_fwd,
    _chunk_cumsum_fwd_kernel,
    _chunk_state_fwd,
    _chunk_state_fwd_kernel,
)
from neuronspark.ops.triton.mamba.ssd_state_passing import _state_passing_fwd, _state_passing_fwd_kernel

__all__ = [
    "softplus",
    "_bmm_chunk_fwd", "_bmm_chunk_bwd", "_bmm_chunk_fwd_kernel", "_bmm_chunk_bwd_kernel",
    "_chunk_cumsum_fwd", "_chunk_cumsum_fwd_kernel", "_chunk_cumsum_bwd_kernel",
    "_chunk_state_fwd", "_chunk_state_fwd_kernel",
    "_state_passing_fwd", "_state_passing_fwd_kernel",
    "CHUNK_CUMSUM_FWD_TRAIN", "CHUNK_CUMSUM_FWD_PREFILL", "CHUNK_CUMSUM_BWD_TRAIN",
    "BMM_CHUNK_FWD", "BMM_CHUNK_BWD_TRAIN", "CHUNK_STATE_FWD_PREFILL", "STATE_PASSING_FWD_PREFILL",
    "CHUNK_STATE_BWD_DB_MIN_BLOCK_N", "CHUNK_SCAN_BWD_DC_MIN_BLOCK_N",
]

# (kwargs, num_warps, num_stages); see PROVENANCE.md for where each value comes from.
CHUNK_CUMSUM_FWD_TRAIN = ({"BLOCK_SIZE_H": 2}, 8, 3)
CHUNK_CUMSUM_FWD_PREFILL = ({"BLOCK_SIZE_H": 1}, 4, 3)
CHUNK_CUMSUM_BWD_TRAIN = ({"BLOCK_SIZE_H": 4}, 4, 3)
BMM_CHUNK_FWD = ({"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 32, "BLOCK_SIZE_K": 32}, 2, 5)
BMM_CHUNK_BWD_TRAIN = ({"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_CS": 32}, 8, 3)
CHUNK_STATE_FWD_PREFILL = ({"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 32}, 4, 4)
STATE_PASSING_FWD_PREFILL = ({"BLOCK_SIZE": 1024}, 4, 3)

# deterministic tile-workspace geometry of the replaced mamba_ssm backward kernels (see the module docstring)
CHUNK_STATE_BWD_DB_MIN_BLOCK_N = 32
CHUNK_SCAN_BWD_DC_MIN_BLOCK_N = 32
