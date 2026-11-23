---
library_name: kernels
license: cc-by-nc-sa-4.0
tags: [cuda, triton, neuronspark]
---

# neuronspark-ring-recall-prefill

The operator `ring_recall_prefill` of the NeuronSpark model (`neuronspark.ops.triton.pool_ring`), as Triton kernels compiled at first use.

## Interface

* `run(q4, k_ring, gv_ring, cum_ring, ct3, p0, ind4, scale, T, B, G, W, dk, dv, out_dtype, xq, hr)`: the operator.
* `_entrypoint.ref(...)`: the pure-torch reference `ring_recall_prefill_ref`, same parameters (the definition the operator is validated against; not part of the public API).

Inputs: queries, rings and clocks of the call, p0 first position, ind4 = (match matrices, beta), scale 16.0; xq = False / hr = 0 for the six heads, xq = True for the extra queries.

Returns: (T, B, G * dv) out_dtype.

Regimes: prefill.

```text
Pool read of every query row of a prefill call over the whole ring, after the call's ring write.

q4 (B, G, T, dk) float view (any strides; rounded to bf16 in the kernel): the queries -- the six heads' gained unit
queries (``xq=False``, group g reads ring head g) or the last head's extra queries (``xq=True``, every group reads
ring head ``hr``); k_ring / gv_ring (B, H, W, dk / dv) bf16 and cum_ring (B, H, W) fp32: the rings with this call's
events written; ct3 (B, G, T) fp32 view of the query clocks (``xq=False``) or None (``xq=True``: the clocks of this
call's slots of head ``hr``); p0 (1,) int64: the ring pointer at the start of the call; ind4: None or
``(matches, beta)`` with n (<= 4) bool (B, T, W) match matrices sharing one layout and beta (n, H) fp32 (the head's
column is read); scale: the match scale (16.0).  Returns (T, B, G * dv) in ``out_dtype`` (the bf16 read widened).
```

## Requirements

* An NVIDIA GPU of compute capability 10.0 (B200); the kernels are compiled for the device at first use and are not written for other architectures.
* torch 2.9.1 (CUDA 13.0 build) and triton 3.5.1, the versions the repository pins; triton is installed with torch, the package declares no Python dependency (the loader knows none for triton).
* Tested with: Python 3.12, torch 2.9.1+cu130, triton 3.5.1, `kernels` 0.18.3 (package built with kernel-builder 0.18.3), CUDA 13.2 `ptxas` (`TRITON_PTXAS_PATH`), on an NVIDIA B200.

## Loading

```python
from pathlib import Path
from kernels import get_local_kernel

kernel = get_local_kernel(Path("hf_kernels/b200/ring_recall_prefill"), trust_remote_code=True)     # in a clone of NeuronSpark
out = kernel.run(q4, k_ring, gv_ring, cum_ring, ct3, p0, ind4, scale, T, B, G, W, dk, dv, out_dtype, xq, hr)
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
