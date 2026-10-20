# Provenance of the vendored mamba_ssm SSD kernels

Source: mamba_ssm `__version__ = "2.3.2.post1"` (state-spaces/mamba), as installed with Triton 3.5.1 and torch
2.9.1. For the four files below the installed copy is byte-identical to the upstream repository at commit `e9594ce1c732d97440f0332fdc43170a2294dbfa`. License: Apache License 2.0; the full text
is `LICENSE` in this directory (the upstream `LICENSE`, sha256
`760939b000194d04548ede6a857bbe735d1695d8422ec85955c8e2bd7f4b95c5`).

| vendored file | source file | source sha256 | kept objects (source lines) |
|---|---|---|---|
| `softplus.py` | `mamba_ssm/ops/triton/softplus.py` | `addcf96f306dd2c9bf1c9ef9feda42381a6e55ad906dd92e220598fcfec7b508` | whole file (1-15) |
| `ssd_bmm.py` | `mamba_ssm/ops/triton/ssd_bmm.py` | `5059b16f8fa269cd84e8159ee3c6880e4ccf90e4bedf9c2d733e6bc2baff9587` | `_bmm_chunk_fwd_kernel` (36-93), `_bmm_chunk_bwd_kernel` (110-162), `_bmm_chunk_fwd` (165-210), `_bmm_chunk_bwd` (213-264) |
| `ssd_chunk_state.py` | `mamba_ssm/ops/triton/ssd_chunk_state.py` | `6d82771b2f62a7bf84c381c8ef8e4b579a95e0352021d117507a1072e4d71dcd` | `_chunk_cumsum_fwd_kernel` (39-86), `_chunk_cumsum_bwd_kernel` (101-174), `_chunk_state_fwd_kernel` (191-268), `_chunk_cumsum_fwd` (718-741), `_chunk_state_fwd` (812-842) |
| `ssd_state_passing.py` | `mamba_ssm/ops/triton/ssd_state_passing.py` | `ae1fab6c680cb5312ef746862a2cf8e74157033c80878490cc64fa1f24f88b70` | `_state_passing_fwd_kernel` (29-87), `_state_passing_fwd` (196-224) |

## Modifications (the only ones)

1. Every `@triton.autotune(...)` decorator of a kept kernel is removed. The kernel bodies (from `@triton.jit` to the
   end of the function) are byte-identical to the source.
2. The kept launchers (`_bmm_chunk_fwd`, `_bmm_chunk_bwd`, `_chunk_cumsum_fwd`, `_chunk_state_fwd`,
   `_state_passing_fwd`) take a required keyword argument `config = (kwargs, num_warps, num_stages)`. They build the
   grid from `kwargs` (the source used `lambda META: ...`) and pass `**kwargs, num_warps=, num_stages=` to the
   kernel. Every other line of the launchers is the source's.
3. Imports: `softplus` comes from the vendored `neuronspark.ops.triton.mamba.softplus`; the unused imports (einops,
   `torch.nn.functional`, `mamba_ssm.utils.determinism`) are removed.
4. Unused functions and kernels are deleted (including `init_to_zero`, whose only users were the removed autotune
   configuration lists).
5. Each file has a header with the upstream copyright line, the Apache-2.0 notice and these modifications.

`_chunk_cumsum_bwd_kernel` keeps no launcher here: the spiking-layer backward
(`neuronspark.ops.triton.spiking_glue.chunk_cumsum_bwd`) launches it directly with `CHUNK_CUMSUM_BWD_TRAIN`.

## Launch configurations (`neuronspark/ops/triton/mamba/__init__.py`)

The configurations start from mamba_ssm's deterministic mode (`MAMBA_DETERMINISTIC=1`). In that mode
`mamba_ssm.utils.determinism.autotune_configs` (lines 59-77 of the 2.3.2.post1 copy; sha256
`cb6e1c30392c11200425c2a23ad9fa3d47f50b556d15e9b0caf79b7d483d6f1d`) keeps one configuration per kernel: the first
configuration of minimal cost, cost = product of the `BLOCK_SIZE_*` values x `num_stages` (`_estimate_config_cost`,
lines 35-41; `triton.Config` defaults num_warps 4, num_stages 3), unless `TRITON_AUTOTUNE_BLOCK_SIZE_*` or
`TRITON_CACHE_AUTOTUNING=1` are set. Some of those single configurations are replaced by configurations measured for
the spiking layer's shapes (nheads 144). The values used here:

| constant | kernel | (kwargs, num_warps, num_stages) | derivation |
|---|---|---|---|
| `CHUNK_CUMSUM_FWD_TRAIN` | `_chunk_cumsum_fwd_kernel` | `({BLOCK_SIZE_H: 2}, 8, 3)` | training: deterministic choice `{H 1}` (cost 1 x 3, the first of the 7 configurations) replaced by a configuration measured for nheads 144 |
| `CHUNK_CUMSUM_FWD_PREFILL` | `_chunk_cumsum_fwd_kernel` | `({BLOCK_SIZE_H: 1}, 4, 3)` | prefill: the deterministic choice |
| `CHUNK_CUMSUM_BWD_TRAIN` | `_chunk_cumsum_bwd_kernel` | `({BLOCK_SIZE_H: 4}, 4, 3)` | training: deterministic `{H 1}` replaced by a configuration measured for nheads 144 (`{H 4}`) |
| `BMM_CHUNK_FWD` | `_bmm_chunk_fwd_kernel` | `({BLOCK_SIZE_M: 64, BLOCK_SIZE_N: 32, BLOCK_SIZE_K: 32}, 2, 5)` | training and prefill: the deterministic choice. Costs of the 9 registered configurations (`ssd_bmm.py` lines 24-32): 6291456, 2097152, 2097152, 1048576, 1048576, 524288, **327680**, 327680, 524288; the first minimum is the 7th, `{64, 32, 32}`, 5 stages, 2 warps (not replaced) |
| `BMM_CHUNK_BWD_TRAIN` | `_bmm_chunk_bwd_kernel` | `({BLOCK_SIZE_M: 64, BLOCK_SIZE_N: 128, BLOCK_SIZE_CS: 32}, 8, 3)` | training: deterministic `{64, 32, 32}` 2 warps 5 stages replaced by a measured configuration |
| `CHUNK_STATE_FWD_PREFILL` | `_chunk_state_fwd_kernel` | `({BLOCK_SIZE_M: 64, BLOCK_SIZE_N: 128, BLOCK_SIZE_K: 32}, 4, 4)` | prefill: a measured configuration (stages 4, warps 4) |
| `STATE_PASSING_FWD_PREFILL` | `_state_passing_fwd_kernel` | `({BLOCK_SIZE: 1024}, 4, 3)` | prefill: a measured configuration (stages 3, warps 4) |

Neither the training forward nor the inference prefill runs `_chunk_state_fwd_kernel` or `_state_passing_fwd_kernel`
(both are replaced by the Gluon chunk-states kernel of `neuronspark.ops.gluon.spiking_chunk_states`; the two configurations
above are those of the vendored pair the tests compare it with); the training backward does not run any other mamba_ssm
kernel.

## Pre-hooks of the registered configurations

The autotune configurations of the source carry `pre_hook=init_to_zero([...])` only on these kept kernels:

| kernel | pre-hook in the source | here |
|---|---|---|
| `_chunk_cumsum_fwd_kernel` | none | -- |
| `_chunk_cumsum_bwd_kernel` | `init_to_zero(["dA_ptr", "ddt_bias_ptr"])` on every configuration | not replicated. The only launch (`spiking_glue.chunk_cumsum_bwd`) passes `DETERMINISTIC_REDUCTION=True` and per-(batch, chunk, head) workspaces of shape (batch, nchunks, nheads) for `dA_ptr` / `ddt_bias_ptr`; the grid (batch, nchunks, cdiv(nheads, BLOCK_SIZE_H)) stores every slot (`tl.store`, masked only by `offs_h < nheads`), so no slot is read unwritten |
| `_chunk_state_fwd_kernel` | none | -- |
| `_bmm_chunk_fwd_kernel`, `_bmm_chunk_bwd_kernel` | none | -- |
| `_state_passing_fwd_kernel` | none | -- |

## Workspace geometry constants

`CHUNK_STATE_BWD_DB_MIN_BLOCK_N` and `CHUNK_SCAN_BWD_DC_MIN_BLOCK_N` are mamba_ssm's `_*_MIN_BLOCK_N` module constants
(`ssd_chunk_state.py` line 500, `ssd_chunk_scan.py` line 621), computed at import as the smallest `BLOCK_SIZE_N` of the
kernel's configuration list.  In deterministic mode that list holds one configuration (`{M 32, N 32}` each), so each
value is 32. The kernels that replace those two mamba_ssm kernels keep the workspace shapes derived from them.
