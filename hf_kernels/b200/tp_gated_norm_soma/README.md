---
library_name: kernels
license: cc-by-nc-sa-4.0
tags: [cuda, triton, neuronspark]
---

# neuronspark-tp-gated-norm-soma

The operator `tp_gated_norm_soma` of the NeuronSpark model (`neuronspark.ops.triton.tp_dendritic`), as Triton kernels compiled at first use.

## Interface

* `run(y, z, theta, gamma, eps, n_all, boxes, rank, ticket, rstd_out, n_out)`: the operator.
* `_entrypoint.ref(...)`: the pure-torch reference `tp_gated_norm_soma_ref`, same parameters (the definition the operator is validated against; not part of the public API).

Inputs: y (M, 4608) bf16, z (M, 4608) bf16 view with unit last stride, theta / gamma (4608,) fp32, eps 1e-5, n_all 9216, boxes N regions of N - 1 slots of M words, rank, ticket int32 (1,) zero (the programs take their rows from it; left zero), rstd_out (M,) fp32 or None, n_out (M, 4608) bf16 or None.

Returns: x (M, 4608) bf16 contiguous soma read-out, normalized over n_all channels.

Regimes: prefill.  Written in place: boxes, ticket, rstd_out, n_out.

```text
x (M, N) bf16 contiguous from y, z (M, N) bf16 views with unit last stride (see the module docstring);
``ticket`` int32 (1,) zero, left zero; ``rstd_out`` (M,) fp32: the rows' rstd stored there as well; ``n_out`` (M,
N) bf16 contiguous: the norm n the read-out was formed from (both for the stage-wise tests).
```

## Requirements

* An NVIDIA GPU of compute capability 10.0 (B200); the kernels are compiled for the device at first use and are not written for other architectures.
* torch 2.9.1 (CUDA 13.0 build) and triton 3.5.1, the versions the repository pins; triton is installed with torch, the package declares no Python dependency (the loader knows none for triton).
* Tested with: Python 3.12, torch 2.9.1+cu130, triton 3.5.1, `kernels` 0.18.3 (package built with kernel-builder 0.18.3), CUDA 13.2 `ptxas` (`TRITON_PTXAS_PATH`), on an NVIDIA B200.

## Loading

```python
from pathlib import Path
from kernels import get_local_kernel

kernel = get_local_kernel(Path("hf_kernels/b200/tp_gated_norm_soma"), trust_remote_code=True)     # in a clone of NeuronSpark
out = kernel.run(y, z, theta, gamma, eps, n_all, boxes, rank, ticket, rstd_out, n_out)
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
