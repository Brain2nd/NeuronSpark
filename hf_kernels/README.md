# Hugging Face kernels of the operators

Every registered operator of NeuronSpark as a Hugging Face kernel package (the format of the `kernels` library): the
operator's module and its static import closure, self-contained, so that one operator can be loaded and run without
installing this repository.

| directory | operators | exported from | how each was checked |
|---|---|---|---|
| `b200/` | 81 | `main` (the B200's operators): 74 packages of b955637, the seven that contain `neuronspark/ops/triton/pool_ring.py` of 93d2f38 (the only file that differs between the two) | `b200/TABLE.md` |
| `gb300/` | 51 | tag `gb300-operators-20261116` (5c2720b, the GB300's operators) | `gb300/TABLE.md` |

A package is `<set>/<operator>/`: `README.md` (its card: the interface, the requirements, the source commit, the
licence) and `build/torch-cuda/` (what the loader reads, with the digest of its files in `metadata.json`).

## Loading

```python
from pathlib import Path
from kernels import get_local_kernel          # the `kernels` library (tested with 0.18.3)

kernel = get_local_kernel(Path("hf_kernels/b200/pool_read"), trust_remote_code=True)
out = kernel.run(q, k, gv, cum, beta, qtok, ktok, orders, scale)               # the operator (its card lists the parameters)
ref = kernel._entrypoint.ref(q, k, gv, cum, beta, qtok, ktok, orders, scale)   # its pure-PyTorch reference
```

The kernels are Triton kernels, compiled for the device at first use.  They are written for NVIDIA GPUs of compute
capability 10.0 and were run on a B200 with torch 2.9.1 (CUDA 13.0 build) and triton 3.5.1.  A package registers its
`torch.library` operators under an identifier of its own, so it can be loaded beside the `neuronspark` package and
beside other versions of itself.

## How the packages were made and checked

Exported with the exporter of Kernel Design Agents (NVlabs/kda, `scripts/export_huggingface.py`) and kernel-builder
0.18.3.  Each package was loaded in a process without a GPU and without this repository, then run on a B200 against
the operator of the commit it was exported from: the outputs, the gradients and the arguments written in place under
the rule of `neuronspark/utils/testing.py`, and the kernel time of both.  The `TABLE.md` of each set has the row of
every operator.
