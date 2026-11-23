---
library_name: kernels
license: cc-by-nc-sa-4.0
tags: [cuda, triton, neuronspark]
---

# neuronspark-ring-write-prefill

The operator `ring_write_prefill` of the NeuronSpark model (`neuronspark.ops.triton.pool_ring`), as Triton kernels compiled at first use.

## Interface

* `run(k_h, q_h, g_h, v_h, qgain, wbias, sparsity, pool_k, pool_gv, ring_ptr, T, B, H, W)`: the operator.
* `_entrypoint.ref(...)`: the pure-torch reference `ring_write_prefill_ref`, same parameters (the definition the operator is validated against; not part of the public API).

Inputs: per-head k / q / gate / v of this call, qgain (H,) fp32 match gain, wbias (H,) fp32, sparsity 2.0, rings (B, H, W, 256) bf16.

Returns: q fp32: the gained unit queries of the call.

Regimes: prefill.  Written in place: pool_k, pool_gv.

```text
Write a prefill call's unit key tags and tagged values into the rings and form its gained unit queries.

k_h / q_h / v_h (T, B, H, 256) bf16, g_h (T, B, H, 256) fp32 (or bf16): the per-head key / query / value / gate
of the call (after RoPE, before any normalisation); qgain (H,) fp32 or None: the dendritic query gain (formed in
fp32, the dtype of the normalised query); wbias (H,) fp32 or None: the write bias; sparsity: the write threshold
(2.0); pool_k / pool_gv (B, H, W, 256) bf16 contiguous, written in place at slots ``(ring_ptr + t) % W``;
ring_ptr (1,) int64 at the start of the call.  Returns the gained unit queries (T, B, H, 256) fp32.

Operands with a unit channel stride run the kernel; operands whose channel dimension is strided (the T = 2 conv
form) run :func:`ring_write_prefill_ref`, the expression chain of the hippocampal layer's prefill on that layout.
The ring clocks are written by the caller.
```

## Requirements

* An NVIDIA GPU of compute capability 10.0 (B200); the kernels are compiled for the device at first use and are not written for other architectures.
* torch 2.9.1 (CUDA 13.0 build) and triton 3.5.1, the versions the repository pins; triton is installed with torch, the package declares no Python dependency (the loader knows none for triton).
* Tested with: Python 3.12, torch 2.9.1+cu130, triton 3.5.1, `kernels` 0.18.3 (package built with kernel-builder 0.18.3), CUDA 13.2 `ptxas` (`TRITON_PTXAS_PATH`), on an NVIDIA B200.

## Loading

```python
from pathlib import Path
from kernels import get_local_kernel

kernel = get_local_kernel(Path("hf_kernels/b200/ring_write_prefill"), trust_remote_code=True)     # in a clone of NeuronSpark
out = kernel.run(k_h, q_h, g_h, v_h, qgain, wbias, sparsity, pool_k, pool_gv, ring_ptr, T, B, H, W)
expected = kernel._entrypoint.ref(...)      # the pure-torch reference, same parameters
```

The first call compiles the kernels for the device (Triton's cache directory keeps them).  The validated use is the one of the model: the inputs listed above, on the GPU, with the dtypes and memory layouts given.

The package's `torch.library` operators are registered under the kernel's unique identifier (`torch.ops.<id>`), so it can be loaded beside the NeuronSpark package and beside other versions of itself.

## Source

NeuronSpark, branch `b200`, commit `93d2f380531b05cedf5fb5d199cd636b771669b4`; 6 files (the operator's module and the modules it imports; package files reduced to what the operator uses).

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
