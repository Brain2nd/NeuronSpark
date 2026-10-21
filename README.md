# NeuronSpark

[![arXiv](https://img.shields.io/badge/arXiv-2603.16148-b31b1b.svg)](https://arxiv.org/abs/2603.16148)
[![OpenReview](https://img.shields.io/badge/OpenReview-NeurIPS%202026%20submission-8c1b13.svg)](https://openreview.net/forum?id=P7pyNdvnfr)

> **NeuronSpark: A Spiking Neural Network Language Model with Selective State Space Dynamics**\
> Zhengzheng Tang, Hao Zhu, Zhewei Chen, Dajiang Tao\
> Paper: https://arxiv.org/abs/2603.16148

> **NeuronSpark: Selective Spiking State Space Models with Adaptive Computation for Language Modeling**\
> Anonymous\
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

## Results

### Controlled comparison on BabyLM

NeuronSpark against a Transformer, Mamba-2 and RWKV-7 at three sizes, trained with one recipe in which only the
architecture differs:

- data: the BabyLM 2026 Strict corpus, tokenized with the Qwen3-VL tokenizer (vocabulary 151,669) and packed into
  1,024-token rows, in the same order for every model;
- training: one epoch, 19,904 updates of 8 rows (162.9M training tokens), AdamW (learning rate 3e-4, 1,000 warm-up
  updates, weight decay 0.1, gradient clipping 1.0), seed 2026;
- baselines: a Llama-style Transformer decoder and the flash-linear-attention implementations of Mamba-2 and RWKV-7,
  each with a tied embedding;
- metric: next-token cross entropy (nats) on a fixed set of 24 batches of the training stream (192 rows, 196,416
  target tokens; each row is seen once in the single epoch, the set is not a held-out split), perplexity = exp(loss).

| loss / perplexity | ~200M | ~500M | ~1.3B |
|---|---|---|---|
| Transformer | 3.0577 / 21.28 | 2.9796 / 19.68 | 2.9145 / 18.44 |
| Mamba-2 | 3.0513 / 21.14 | 2.9438 / 18.99 | 2.8738 / 17.70 |
| RWKV-7 | 2.9807 / 19.70 | 2.8877 / 17.95 | 2.8326 / 16.99 |
| **NeuronSpark** | **2.9420 / 18.95** | **2.8444 / 17.19** | **2.7751 / 16.04** |

Parameters: Transformer 199.6M / 501.4M / 1,312.0M; Mamba-2 199.6M / 501.1M / 1,312.3M; RWKV-7 199.6M / 501.6M /
1,312.2M; NeuronSpark 217.5M / 508.6M / 1,270.3M.

### Speed

1.3B-parameter models on one NVIDIA GB300.  NeuronSpark is measured with the operators of this repository in our
training stack and serving engine (the serving engine is not part of this repository).

**Inference.**  Online serving: requests with 1,024-token random prompts and 256 generated tokens, at concurrency 1, 8
and 32; output tokens per second.  Compared with: the 1.3B Transformer of the table above (a Hugging Face Llama
checkpoint) served by vLLM 0.30.0 on the same GPU with the same workload.

| output tok/s | concurrency 1 | concurrency 8 | concurrency 32 |
|---|---|---|---|
| Transformer 1.3B, served by vLLM | 517 | 3,864 | 10,401 |
| NeuronSpark 1.3B | 673.5 | 4,069.5 | 9,515.7 |
| NeuronSpark ÷ Transformer | 1.30× | 1.05× | 0.91× |

**Training.**  One step of 8 × 1,024 tokens (forward, backward, AdamW update), with and without activation
recomputation; tokens per second and time per step.  Compared with: the 1.3B Transformer with its fastest measured
stack: Hugging Face Llama, full-graph `torch.compile`, bf16 compute with fp32 master weights, Transformer Engine
FusedAdam.

| tok/s (ms / step) | with activation recomputation | without |
|---|---|---|
| Transformer 1.3B (`torch.compile` + TE FusedAdam) | 79,387 (103.1 ms) | 94,546 (86.6 ms) |
| NeuronSpark 1.3B | 67,882 (120.6 ms) | 77,164 (106.1 ms) |
| NeuronSpark ÷ Transformer | 0.86× | 0.82× |

The speed tables are updated with every operator improvement.

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
| linear with fp32 weight gradient (few-feature decode products in one kernel) | `ops/linear.py`, `ops/triton/linear_small_n.py` | train, prefill, decode |
| dendritic integration (chunked scan) + somatic release | `ops/dendritic_combined.py`, `ops/triton/dendritic_{chunk,states,scan,glue,soma}.py`, `ops/gluon/dendritic_{chunk_states,dstates,dda,chunk_scan_fwd,chunk_scan_dx}.py` | train |
| dendritic integration, prefill | `ops/dendritic_prefill.py`, `ops/triton/dendritic_chunk.py`, `ops/gluon/dendritic_chunk_states.py` | prefill |
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

## Tests

Kernels are tested against their references with one tolerance rule, the default rule of flashinfer-bench
(`neuronspark/utils/testing.py`).  An element fails only if both its absolute and its relative error exceed `1e-2`.  A
tensor passes when no element fails.

```bash
pytest tests
```

The model tests build a fresh model of the default configuration and calibrate it on random token rows; no checkpoint
is needed.  `tests/models/test_widths.py` also runs the model at D = 2048 (training against the references, CUDA-graph
decoding); the operator tests of the gated-norm soma, product-key read and decode-step kernels cover both widths.

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

@misc{anonymous2026neuronspark,
  title={NeuronSpark: Selective Spiking State Space Models with Adaptive Computation for Language Modeling},
  author={Anonymous},
  year={2026},
  url={https://openreview.net/forum?id=P7pyNdvnfr}
}
```

## License

CC BY-NC-SA 4.0 (see `LICENSE`).  Third-party code is listed in `NOTICE`.
