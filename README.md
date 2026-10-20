# NeuronSpark

![NeuronSpark](https://img.shields.io/badge/parameters-1.27B-blue)
[![arXiv](https://img.shields.io/badge/arXiv-2603.16148-b31b1b.svg)](https://arxiv.org/abs/2603.16148)

> **NeuronSpark: A Spiking Neural Network Language Model with Selective State Space Dynamics**\
> Zhengzheng Tang\
> Paper: https://arxiv.org/abs/2603.16148

## About

NeuronSpark is a spiking language model.  Its recurrent layers are neurons: each layer keeps the membrane potentials
of its dendritic compartments as a selective state space (input-dependent leak and write, as in Mamba), and reads them
out through a spiking soma.  Every fourth layer is a memory layer that adds attention-style retrieval, a spiking
feed-forward block of leaky integrate-and-fire neurons and a product-key memory.

This repository contains the NeuronSpark model (1.27B parameters) together with its GPU operators.  The Triton and Gluon
kernels cover training, prefill and single-token decoding.  The repository also has a mamba-style inference cache with
CUDA-graph decoding, plus tests and benchmarks.

## Architecture

NeuronSpark stacks two kinds of layers in a fixed 3:1 pattern: three spiking layers, then one memory layer.  The readout
is tied to the embedding.

| block | what it computes |
|---|---|
| **Spiking layer** | pre-norm → input projection → causal conv + SiLU → a selective state space holding the membrane potentials of each soma's dendritic compartments (chunked scan) → gated RMSNorm → bipolar soft-threshold soma `max(n − θ, 0) − max(−n − θ, 0)` → output projection |
| **Memory layer** (every fourth layer) | lexical read-out; pool attention (RoPE, PLIF write gate, gated decay, induction bias, value residual); copy gate; spiking FFN of PLIF neurons; product-key memory with top-k retrieval |
| **Model level** | hashed n-gram lexical memory, induction match matrices, copy-detector statistics, final RMSNorm, readout balance (low-rank logit corrections) |

The training loss is the token cross entropy plus a weighted copy-switch auxiliary loss.

Sizes and constants (depth, width, somata and compartments, heads, memory slots, n-gram orders, vocabulary, loss
weights) are configuration, not architecture: they are the fields of `NeuronSparkConfig`
(`neuronspark/models/config_neuronspark.py`), and `configs/neuronspark_1p3b.json` is the configuration of the released
1.27B checkpoint.  Each GPU operator states the shapes it supports in its docstring.

## Installation

Requirements:

- Linux and an NVIDIA Blackwell GPU (compute capability ≥ 10.0)
- Python ≥ 3.10
- PyTorch 2.9.1 (CUDA 13.0) and Triton 3.5.1

```bash
pip install -r requirements.txt
pip install -e .
```

All kernels are compiled by Triton the first time they are used.  If the `ptxas` bundled with Triton does not know your
GPU architecture (for example `sm_103a`), set `TRITON_PTXAS_PATH` to the one in your CUDA toolkit.

## Usage

### Model

```python
import torch
from neuronspark import NeuronSparkLMHeadModel

model = NeuronSparkLMHeadModel.from_pretrained("path/to/checkpoint", device="cuda")

ids = torch.randint(0, model.config.vocab_size, (8, 1024), device="cuda")  # (B, T), T a multiple of spiking_chunk_size
logits = model(ids).logits                                        # (B, T, V)

targets = torch.roll(ids, -1, dims=1)
targets[:, -1] = -100                                             # -100 is ignored
loss = model.loss(ids, targets).loss                              # cross entropy + copy-switch auxiliary loss
loss.backward()
```

A checkpoint directory holds `config.json` and `model.safetensors`.  `model.save_pretrained(path)` writes one.  To
recompute the spiking layers in the backward pass, set `model.gradient_checkpointing = True`.

To train from scratch, build `NeuronSparkLMHeadModel(config)` and calibrate it once on the first training rows with
`neuronspark.utils.calibration.calibrate_fresh_model(model, batches)`.  This sets the neuron thresholds and the
product-key memories.

### Generation

```python
prompt = ids[:1, :512]
out = model.generate(prompt, max_length=768)                                     # greedy
out = model.generate(prompt, max_length=768, top_k=50, top_p=0.9, temperature=0.8)
```

`generate` decodes with CUDA graphs.  `max_length` counts the prompt and the new tokens together.  A batch holds 1 to 32
prompts of equal length.

As in Mamba, you can also drive the inference cache yourself:

```python
with torch.no_grad():
    cache = model.allocate_inference_cache(batch_size=1, max_seqlen=2048)
    logits = model(prompt, inference_params=cache, num_last_tokens=1).logits    # prefill
    token = logits[:, -1].argmax(-1, keepdim=True)
    logits = model(token, inference_params=cache).logits                       # one decode step
```

The spiking-layer state in the cache is stored in bf16 by default.  To store it in fp32, pass
`spiking_state_dtype="float32"` to `allocate_inference_cache` or `generate`, or
`inference_spiking_state_dtype="float32"` to `from_pretrained`.  fp32 storage uses twice the memory.

## Operators

`neuronspark/ops` follows the layout of `mamba_ssm/ops`: one file per operator family.  Each family provides a public
function and a pure-PyTorch reference `*_ref` with the same signature.  Kernels are registered as `torch.library`
custom ops.  `neuronspark/ops/registry.py` lists every operator with its signature and the code paths that use it.
`NeuronSparkLMHeadModel(config, impl="ref")` builds the whole model on the references.

| family | file | used in |
|---|---|---|
| linear with fp32 weight gradient | `ops/linear.py` | train, prefill, decode |
| Spiking-layer chunk scan + soma | `ops/spiking_combined.py`, `ops/triton/spiking_{states,scan,glue,soma}.py`, `ops/gluon/spiking_{chunk_states,dstates,dda,chunk_scan_fwd,chunk_scan_dx}.py` | train |
| Spiking-layer prefill scan | `ops/spiking_prefill.py`, `ops/gluon/spiking_chunk_states.py` | prefill |
| Spiking-layer conv, norms, residual | `ops/triton/spiking_{conv,prenorm,gated_norm,residual}.py` | train, prefill |
| Spiking-layer decode step | `ops/triton/spiking_step.py` | decode |
| PLIF neurons | `ops/triton/plif.py`, `ops/gluon/plif_k1_bwd.py` | train, prefill |
| pool attention | `ops/triton/pool_{attn,bias}.py`, `ops/triton/mem_{heads,rope,norms,residual}.py` | train |
| inference rings, extra queries | `ops/triton/pool_ring.py`, `ops/triton/xq_query.py` | prefill, decode |
| short conv | `ops/triton/short_conv.py` | train, prefill, decode |
| cumulative sums | `ops/triton/cumsum.py` | train, prefill |
| product-key memory | `ops/triton/pkm_{topk,read}.py` | train, prefill, decode |
| lexical read, copy statistics, value residual | `ops/triton/{lex_read,copy_stats,vr_mix}.py` | prefill |
| copy-switch loss | `ops/triton/copy_switch.py` | train |
| readout balance | `ops/triton/balance_add.py` | scoring, prefill, decode |
| readout balance + cross entropy | `ops/triton/lm_tail.py` | train |

The SSD chunk kernels in `ops/triton/mamba/` are vendored from `mamba_ssm` (see `NOTICE`).

## Tests

Kernels are tested against their references with one tolerance rule, the default rule of flashinfer-bench
(`neuronspark/utils/testing.py`).  An element fails only if both its absolute and its relative error exceed `1e-2`.  A
tensor passes when no element fails.

```bash
pytest tests/ops tests/modules
pytest tests/models --ckpt path/to/checkpoint
```

## Benchmarks

```bash
python benchmarks/benchmark_train_step.py --ckpt path/to/checkpoint
python benchmarks/benchmark_generation.py --ckpt path/to/checkpoint
```

The first script times a full training step (8 × 1024 tokens: forward, backward, optimizer).  The second times prefill
at batch 1 and 8, CUDA-graph decoding at batch 1, 8 and 32, and end-to-end generation.

## Citation

```bibtex
@article{tang2026neuronspark,
  title={NeuronSpark: A Spiking Neural Network Language Model with Selective State Space Dynamics},
  author={Tang, Zhengzheng},
  journal={arXiv preprint arXiv:2603.16148},
  year={2026}
}
```

## License

CC BY-NC-SA 4.0 (see `LICENSE`).  Third-party code is listed in `NOTICE`.
