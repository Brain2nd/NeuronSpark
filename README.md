# NeuronSpark

![NeuronSpark](https://img.shields.io/badge/model-1MIX%201.27B-blue)
[![arXiv](https://img.shields.io/badge/arXiv-2603.16148-b31b1b.svg)](https://arxiv.org/abs/2603.16148)

> **NeuronSpark: A Spiking Neural Network Language Model with Selective State Space Dynamics**\
> Zhengzheng Tang\
> Paper: https://arxiv.org/abs/2603.16148

## About

NeuronSpark is a spiking language model.  Its recurrent layers are neurons: each layer keeps the membrane potentials
of its dendritic compartments as a selective state space (input-dependent leak and write, as in Mamba), and reads them
out through a spiking soma.  Every fourth layer is a memory layer that adds attention-style retrieval, a spiking
feed-forward block of leaky integrate-and-fire neurons and a product-key memory.

This repository contains the 1MIX model (1.27B parameters) together with its GPU operators.  The Triton and Gluon
kernels cover training, prefill and single-token decoding.  The repository also has a mamba-style inference cache with
CUDA-graph decoding, plus tests and benchmarks.

## Architecture

1MIX has 24 layers in a fixed 3:1 pattern, width 1536, and a vocabulary of 151,669 tokens.  The readout is tied to
the embedding.

| block | layers | what it computes |
|---|---|---|
| **SD layer** (selective dendrite) | 18 | pre-norm → input projection → causal conv + SiLU → a selective state space holding the membrane potentials of 128 compartments per soma (144 × 64 somata, chunked scan) → gated RMSNorm → bipolar soft-threshold soma `max(n − θ, 0) − max(−n − θ, 0)` → output projection |
| **Memory layer** (layers 3, 7, …, 23) | 6 | lexical read-out; pool attention (6 heads, RoPE, PLIF write gate, gated decay, induction bias, value residual); copy gate; spiking FFN of PLIF neurons; product-key memory (4 heads × 128² keys, top-16) |
| **Model level** | – | hashed n-gram lexical memory (orders 1–3), induction match matrices (orders 2–4), copy-detector statistics, final RMSNorm, readout balance (low-rank logit corrections of widths 16 and 32) |

The training loss is the token cross entropy plus a copy-switch auxiliary loss with weight 0.5.  The configuration is
in `configs/onemix_1p3b.json`.

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

ids = torch.randint(0, 151669, (8, 1024), device="cuda")        # (B, T), T a multiple of 256
logits = model(ids).logits                                        # (B, T, V)

targets = torch.roll(ids, -1, dims=1)
targets[:, -1] = -100                                             # -100 is ignored
loss = model.loss(ids, targets).loss                              # cross entropy + copy-switch auxiliary loss
loss.backward()
```

A checkpoint directory holds `config.json` and `model.safetensors`.  `model.save_pretrained(path)` writes one.  To
recompute the SD layers in the backward pass, set `model.gradient_checkpointing = True`.

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

The SD state in the cache is stored in bf16 by default.  To store it in fp32, pass `sd_state_dtype="float32"` to
`allocate_inference_cache` or `generate`, or `inference_sd_state_dtype="float32"` to `from_pretrained`.  fp32 storage
uses twice the memory.

## Operators

`neuronspark/ops` follows the layout of `mamba_ssm/ops`: one file per operator family.  Each family provides a public
function and a pure-PyTorch reference `*_ref` with the same signature.  Kernels are registered as `torch.library`
custom ops.  `neuronspark/ops/registry.py` lists every operator with its signature and the code paths that use it.
`NeuronSparkLMHeadModel(config, impl="ref")` builds the whole model on the references.

| family | file | used in |
|---|---|---|
| linear with fp32 weight gradient | `ops/linear.py` | train, prefill, decode |
| SD chunk scan + soma | `ops/sd_combined.py`, `ops/triton/sd_{states,scan,glue,soma}.py`, `ops/gluon/sd_{chunk_states,dstates,chunk_scan_fwd,chunk_scan_dx}.py` | train |
| SD prefill scan | `ops/sd_prefill.py`, `ops/gluon/sd_chunk_states.py` | prefill |
| SD conv, norms, residual | `ops/triton/sd_{conv,prenorm,gated_norm,residual}.py` | train, prefill |
| SD decode step | `ops/triton/sd_step.py` | decode |
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
