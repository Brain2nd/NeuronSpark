# NeuronSpark

[![arXiv](https://img.shields.io/badge/arXiv-2603.16148-b31b1b.svg)](https://arxiv.org/abs/2603.16148)

> **NeuronSpark: A Spiking Neural Network Language Model with Selective State Space Dynamics**\
> Zhengzheng Tang, Hao Zhu, Zhewei Chen, Dajiang Tao\
> Paper: https://arxiv.org/abs/2603.16148

> **NeuronSpark: Selective Spiking State Space Models with Adaptive Computation for Language Modeling**\
> Zhengzheng Tang, Hao Zhu, Dajiang Tao, Yuhui Zhang\
> NeurIPS 2026 submission: https://openreview.net/forum?id=P7pyNdvnfr

## About

NeuronSpark is a spiking language model built from neuron models.  Its dendritic neuron layers are populations of
neurons with dendritic compartments: every soma owns a set of compartments whose membrane potentials leak with an
input-dependent time constant and integrate each token's synaptic input, and the soma reads its dendrites out through a
bipolar soft threshold.  The membrane dynamics have the selective state-space form, which lets them run in parallel over
a sequence in training and one token at a time in decoding.  Every fourth layer is a hippocampal layer: an episodic
memory of past tokens, written through PLIF write-gate neurons and recalled by content, a copy pathway for verbatim
recall, a spiking feed-forward circuit of PLIF (parametric leaky integrate-and-fire) neurons and a product-key
associative memory.

This repository contains the NeuronSpark architecture and its GPU operators: the model code (configuration, layers and
an inference cache with CUDA-graph decoding), Triton and Gluon kernels for training, prefill and single-token decoding,
and tests and benchmarks.

## Architecture

NeuronSpark stacks two kinds of layers in a fixed 3:1 pattern: three dendritic neuron layers, then one hippocampal
layer.  The readout is tied to the embedding.

| block | neural mechanism |
|---|---|
| **Dendritic neuron layer** | *synaptic input*: the normalised residual stream is projected onto the dendrites through a short causal synaptic filter (conv + SiLU); *dendritic integration*: each soma's compartments hold membrane potentials `V` that leak with an input-dependent time constant and integrate the input, `V_t = exp(Δt_t A) V_{t−1} + Δt_t x_t B_tᵀ`, and the soma reads `V_t C_t` (selective state-space dynamics, a chunked scan in training); *somatic release*: the somatic potential is normalised under a multiplicative gate and released through the bipolar soft threshold `max(n − θ, 0) − max(−n − θ, 0)` with a learned threshold θ; the release projects back onto the residual stream |
| **Hippocampal layer** (every fourth layer) | *episodic memory*: each token is written into memory slots through PLIF write-gate neurons, older traces decay under a gate, and the slots are recalled by content with position (RoPE) and induction biases, plus a value residual across hippocampal layers; *lexical read-out* of the token's hashed context; *copy pathway*: a copy gate for verbatim recall; *spiking feed-forward circuit* of PLIF neurons; *associative memory*: a product-key memory with top-k recall |
| **Model level** | hashed n-gram lexical memory, induction match matrices, copy-detector statistics, final RMSNorm, readout balance (low-rank logit corrections) |

The training loss is the token cross entropy plus a weighted copy-switch auxiliary loss.

Sizes and constants (depth, width, somata and compartments, heads, memory slots, n-gram orders, vocabulary, loss
weights) are configuration, not architecture: they are the fields of `NeuronSparkConfig`
(`neuronspark/models/config_neuronspark.py`), whose defaults are one configuration of the architecture.  Each GPU
operator states the shapes it supports in its docstring.

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
from neuronspark import NeuronSparkConfig, NeuronSparkLMHeadModel
from neuronspark.utils.calibration import calibrate_fresh_model

config = NeuronSparkConfig()                                      # the default configuration
model = NeuronSparkLMHeadModel(config, device="cuda")

# launch calibration of a new model (neuron thresholds, product-key memories) on its first 64 training rows
first_rows = [torch.randint(0, config.vocab_size, (8, 1024), device="cuda") for _ in range(8)]
calibrate_fresh_model(model, first_rows)

ids = torch.randint(0, config.vocab_size, (8, 1024), device="cuda")  # (B, T); T % dendritic_chunk_size == 0
logits = model(ids).logits                                        # (B, T, V)

targets = torch.roll(ids, -1, dims=1)
targets[:, -1] = -100                                             # -100 is ignored
loss = model.loss(ids, targets).loss                              # cross entropy + copy-switch auxiliary loss
loss.backward()
```

`model.save_pretrained(path)` writes a checkpoint directory (`config.json` and `model.safetensors`) and
`NeuronSparkLMHeadModel.from_pretrained(path)` loads one.  To recompute the dendritic neuron layers in the backward
pass, set `model.gradient_checkpointing = True`.

### Generation

```python
prompt = ids[:1, :512]
out = model.generate(prompt, max_length=768)                                     # greedy
out = model.generate(prompt, max_length=768, top_k=50, top_p=0.9, temperature=0.8)
```

`generate` decodes with CUDA graphs.  `max_length` counts the prompt and the new tokens together.  A batch holds 1 to 32
prompts of equal length.

The inference cache can also be driven step by step:

```python
with torch.no_grad():
    cache = model.allocate_inference_cache(batch_size=1, max_seqlen=2048)
    logits = model(prompt, inference_params=cache, num_last_tokens=1).logits    # prefill
    token = logits[:, -1].argmax(-1, keepdim=True)
    logits = model(token, inference_params=cache).logits                       # one decode step
```

The dendritic-layer state in the cache is stored in bf16 by default.  To store it in fp32, pass
`dendritic_state_dtype="float32"` to `allocate_inference_cache` or `generate`, or
`inference_dendritic_state_dtype="float32"` to `from_pretrained`.  fp32 storage uses twice the memory.

## Operators

`neuronspark/ops` has one file per operator family.  Each family provides a public
function and a pure-PyTorch reference `*_ref` with the same signature.  Kernels are registered as `torch.library`
custom ops.  `neuronspark/ops/registry.py` lists every operator with its signature and the code paths that use it.
`NeuronSparkLMHeadModel(config, impl="ref")` builds the whole model on the references.

| family | file | used in |
|---|---|---|
| linear with fp32 weight gradient | `ops/linear.py` | train, prefill, decode |
| dendritic integration (chunked scan) + somatic release | `ops/dendritic_combined.py`, `ops/triton/dendritic_{states,scan,glue,soma}.py`, `ops/gluon/dendritic_{chunk_states,dstates,dda,chunk_scan_fwd,chunk_scan_dx}.py` | train |
| dendritic integration, prefill | `ops/dendritic_prefill.py`, `ops/gluon/dendritic_chunk_states.py` | prefill |
| synaptic filter (conv), norms, residual | `ops/triton/dendritic_{conv,prenorm,gated_norm,residual}.py` | train, prefill |
| dendritic neuron step | `ops/triton/dendritic_step.py` | decode |
| PLIF neurons | `ops/triton/plif.py`, `ops/gluon/plif_k1_bwd.py` | train, prefill |
| hippocampal recall (pool attention) | `ops/triton/pool_{attn,bias}.py`, `ops/triton/hippo_{heads,rope,norms,residual}.py` | train |
| hippocampal slot rings, extra queries | `ops/triton/pool_ring.py`, `ops/triton/xq_query.py` | prefill, decode |
| short conv | `ops/triton/short_conv.py` | train, prefill, decode |
| cumulative sums | `ops/triton/cumsum.py` | train, prefill |
| product-key memory | `ops/triton/pkm_{topk,read}.py` | train, prefill, decode |
| lexical read, copy statistics, value residual | `ops/triton/{lex_read,copy_stats,vr_mix}.py` | prefill |
| copy-switch loss | `ops/triton/copy_switch.py` | train |
| readout balance | `ops/triton/balance_add.py` | scoring, prefill, decode |
| readout balance + cross entropy | `ops/triton/lm_tail.py` | train |

The chunked-scan kernels in `ops/triton/mamba/` are vendored from `mamba_ssm` (see `NOTICE`).

## Tests

Kernels are tested against their references with one tolerance rule, the default rule of flashinfer-bench
(`neuronspark/utils/testing.py`).  An element fails only if both its absolute and its relative error exceed `1e-2`.  A
tensor passes when no element fails.

```bash
pytest tests
```

The model tests build a fresh model of the default configuration and calibrate it on random token rows; no checkpoint
is needed.

## Benchmarks

```bash
python benchmarks/benchmark_train_step.py
python benchmarks/benchmark_generation.py
```

The first script times a full training step (8 × 1024 tokens: forward, backward, optimizer).  The second times prefill
at batch 1 and 8, CUDA-graph decoding at batch 1, 8 and 32, and end-to-end generation.  Both run a fresh model of the
default configuration (calibrated on random rows); `--ckpt DIR` loads a checkpoint directory instead.

## Citation

```bibtex
@article{tang2026neuronspark,
  title={NeuronSpark: A Spiking Neural Network Language Model with Selective State Space Dynamics},
  author={Tang, Zhengzheng and Zhu, Hao and Chen, Zhewei and Tao, Dajiang},
  journal={arXiv preprint arXiv:2603.16148},
  year={2026}
}

@misc{tang2026selective,
  title={NeuronSpark: Selective Spiking State Space Models with Adaptive Computation for Language Modeling},
  author={Tang, Zhengzheng and Zhu, Hao and Tao, Dajiang and Zhang, Yuhui},
  year={2026},
  note={Submitted to NeurIPS 2026},
  url={https://openreview.net/forum?id=P7pyNdvnfr}
}
```

## License

CC BY-NC-SA 4.0 (see `LICENSE`).  Third-party code is listed in `NOTICE`.
