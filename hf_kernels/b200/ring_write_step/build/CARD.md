---
library_name: kernels
license: cc-by-nc-sa-4.0
tags: [cuda, triton, neuronspark]
---

# neuronspark-ring-write-step

The operator `ring_write_step` of the NeuronSpark model (`neuronspark.ops.triton.ring_step`), as Triton kernels compiled at first use.

## Interface

* `run(qkv, conv_hist, conv_weight, cos, sin, pos, ring_ptr, gate, gl, lam0, carry, ga, gb, glu_u, dlog, rate, base_decay, gains, qgain, wbias, sparsity, pool_k, pool_gv, pool_cum, pool_run)`: the operator.
* `_entrypoint.ref(...)`: the pure-torch reference `ring_write_step_ref`, same parameters (the definition the operator is validated against; not part of the public API).

Inputs: qkv (B, 4608) bf16 projection, conv_hist (B, 4608, 3) fp32, conv_weight (4608, 1, 4), cos / sin (8192, 768) bf16 tables, pos (B,) int64, ring_ptr (1,) int64, gate (B, 1536) fp32 spikes, gl (B, 6) bf16, lam0 (1536,), carry (B, 1536) bf16 or None, ga / gb (B, 128) bf16, glu_u (256, 128), dlog (B, 256) bf16, rate (256,) bf16, base_decay, gains / qgain / wbias (6,), sparsity, the rings and the carried clocks.

Returns: (q (B, 6, 256) bf16 gained unit queries, v (B, 1536) bf16 values after the SiLU).

Regimes: decode.  Written in place: conv_hist, pool_k, pool_gv, pool_cum, pool_run.

```text
The ring write of a decode step (module docstring).

qkv (B, 3 H dk) bf16: the qkv projection; conv_hist (B, 3 H dk, 3) fp32: the conv's carried taps (rolled in
place); conv_weight (3 H dk, 1, 4) fp32; cos / sin (positions, H dk / 2) bf16: the RoPE tables; pos (B,) int64:
the rows' positions; ring_ptr (1,) int64; gate (B, H dk) fp32: the write-gate spikes; gl (B, H) bf16: the
value-residual gate projection; lam0 (H dk,) fp32: the static mix; carry (B, H dk) bf16 or None (hippocampal layer
0); ga / gb (B, g) bf16: the GLU maps; glu_u (dk, g) fp32 (its bf16 inference copy is read when it has one); dlog
(B, dk) bf16: the decay logit; rate (dk,) bf16;
base_decay; gains / qgain / wbias (H,) fp32; sparsity; pool_k / pool_gv (B, H, W, dk) bf16, pool_cum (B, H, W)
fp32: the rings (this step's slot written); pool_run (B, H) fp32: the carried clocks (advanced in place).

``ring_ptr`` and ``pos`` are counts (>= 0; ``pos`` inside the tables): the kernel reads them on the device.  The
state tensors (``conv_hist``, the rings, ``pool_run``) are written by one program per (row, head) and must not
share memory with another argument.

Returns (q (B, H, dk) bf16 gained unit queries, v (B, H dk) bf16 values after the SiLU).
```

## Requirements

* An NVIDIA GPU of compute capability 10.0 (B200); the kernels are compiled for the device at first use and are not written for other architectures.
* torch 2.9.1 (CUDA 13.0 build) and triton 3.5.1, the versions the repository pins; triton is installed with torch, the package declares no Python dependency (the loader knows none for triton).
* Tested with: Python 3.12, torch 2.9.1+cu130, triton 3.5.1, `kernels` 0.18.3 (package built with kernel-builder 0.18.3), CUDA 13.2 `ptxas` (`TRITON_PTXAS_PATH`), on an NVIDIA B200.

## Loading

```python
from pathlib import Path
from kernels import get_local_kernel

kernel = get_local_kernel(Path("hf_kernels/b200/ring_write_step"), trust_remote_code=True)     # in a clone of NeuronSpark
out = kernel.run(qkv, conv_hist, conv_weight, cos, sin, pos, ring_ptr, gate, gl, lam0, carry, ga, gb, glu_u, dlog, rate, base_decay, gains, qgain, wbias, sparsity, pool_k, pool_gv, pool_cum, pool_run)
expected = kernel._entrypoint.ref(...)      # the pure-torch reference, same parameters
```

The first call compiles the kernels for the device (Triton's cache directory keeps them).  The validated use is the one of the model: the inputs listed above, on the GPU, with the dtypes and memory layouts given.

The package's `torch.library` operators are registered under the kernel's unique identifier (`torch.ops.<id>`), so it can be loaded beside the NeuronSpark package and beside other versions of itself.

## Source

NeuronSpark, branch `b200`, commit `b9556379c89f943f95dbf31d0a22967c75efa159`; 13 files (the operator's module and the modules it imports; package files reduced to what the operator uses).

## License

Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International (`LICENSE.txt`), except for the third-party material listed in the repository's NOTICE, reproduced below and in `_solution/third_party_notices.py`.

```text
NeuronSpark
Licensed under the Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International license (see LICENSE),
except for the third-party material listed below, which keeps its own license.

-----------------------------------------------------------------------------------------------------------------------
1. flash-linear-attention (fla-core), version 0.5.2
   Copyright (c) the flash-linear-attention authors
   (https://github.com/fla-org/flash-linear-attention, LICENSE)
   Licensed under the MIT License.

   Kernel bodies transcribed from fla-core's normalization kernels (rms_norm / layer_norm_fwd), with the reduction
   layout and rounding points kept:
     neuronspark/ops/triton/dendritic_prenorm.py       dendritic-layer pre-norm forward / backward
     neuronspark/ops/triton/dendritic_gated_norm.py    gated RMSNorm of the dendritic-layer prefill path
     neuronspark/ops/triton/dendritic_step.py          pre-norm rows kernel of the prefill / decode path and the gated-norm
                                                lines of the decode step kernels

   MIT License text:
     Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
     documentation files (the "Software"), to deal in the Software without restriction, including without limitation
     the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and
     to permit persons to whom the Software is furnished to do so, subject to the following conditions:
     The above copyright notice and this permission notice shall be included in all copies or substantial portions of
     the Software.
     THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
     THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
     AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF
     CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
     IN THE SOFTWARE.

-----------------------------------------------------------------------------------------------------------------------
2. PyTorch, version 2.9.1
   Copyright (c) 2016- Facebook, Inc. and its affiliates, and the PyTorch contributors
   Licensed under the BSD 3-Clause License (https://github.com/pytorch/pytorch/blob/main/LICENSE).

   Kernels that reproduce the arithmetic order of ATen CUDA kernels (reduction trees, scans, top-k selection,
   softmax / cross-entropy sequences) follow the structure of those kernels:
     neuronspark/ops/triton/hippo_norms.py     row reductions of the RMS norms (at::native reduce kernel order)
     neuronspark/ops/triton/cumsum.py        outer-dimension inclusive scan
     neuronspark/ops/triton/pkm_topk.py      top-k selection (radix select + bitonic sort semantics)
```
