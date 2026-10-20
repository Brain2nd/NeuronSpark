#!/usr/bin/env python3
"""Training-step throughput of the NeuronSpark model (eager PyTorch: no CUDA graph, no torch.compile).

One step = ``model.loss(input_ids, targets)`` (whole-row regime: bf16 autocast, TF32 on) + ``backward()`` + global
gradient-norm clipping (1.0) + one fused ``torch.optim.AdamW`` step over the four parameter groups of
:func:`build_param_groups` (the learning rate follows :func:`schedule_lr`: linear warm-up, then cosine decay).  The
default shape is 8 rows x 1024 tokens; the input ids are drawn uniformly from the vocabulary (``--seed``) or taken
from a token file (``--tokens``).

Reported: the step time (CUDA events around each step, median per trial, mean of the trial medians over ``--trials``),
the tokens per second it gives (rows x tokens / step time), the peak allocated memory and, with ``--cupti``, the summed
device time of the step's kernels (CUPTI, L2 flushed before each step).  The weights are updated by the timed steps; a
loaded checkpoint is not written back.

The model is loaded from ``--ckpt`` (a checkpoint directory: ``config.json`` + ``model.safetensors``); without it a
fresh model is launched from ``configs/neuronspark_1p3b.json`` and calibrated on the first benchmark batches (the launch
calibration of a new training run).

Example::

    python benchmarks/benchmark_train_step.py --ckpt CHECKPOINT_DIR
    python benchmarks/benchmark_train_step.py --ckpt CHECKPOINT_DIR --gradient-checkpointing
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Dict, List, Optional

import torch

_HERE = Path(__file__).resolve().parent
for _p in (str(_HERE), str(_HERE.parent)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from _timing import cuda_event_time_ms, cupti_time_ms, format_ms  # noqa: E402

# optimiser settings: AdamW, warm-up + cosine learning-rate schedule
PEAK_LR = 3e-4
WARMUP_UPDATES = 1000
TOTAL_UPDATES = 19904
MIN_LR_RATIO = 0.1
BETAS = (0.9, 0.95)
EPS = 1e-8
WEIGHT_DECAY = 0.1
LEXICAL_LR_MULT = 8.0
GRAD_CLIP = 1.0
# spiking mixer vectors decayed with the matrices (their own group)
SPIKING_DECAYED_VECTORS = ("A_log", "dt_bias", "D", "gamma", "conv_bias", "pre_norm_gain")


def schedule_lr(update: int, total: int = TOTAL_UPDATES, peak: float = PEAK_LR, warmup: int = WARMUP_UPDATES,
              min_ratio: float = MIN_LR_RATIO) -> float:
    """Learning rate of update ``update`` (0-based): linear warm-up ``peak * (u + 1) / warmup`` for u < warmup, then a
    cosine from ``peak`` to ``min_ratio * peak`` over the remaining updates."""
    u = int(update)
    if u < warmup:
        return peak * (u + 1) / warmup
    span = max(1, total - warmup)
    return peak * (min_ratio + (1.0 - min_ratio) * 0.5 * (1.0 + math.cos(math.pi * (u - warmup) / span)))


def build_param_groups(model, lr: float = PEAK_LR, weight_decay: float = WEIGHT_DECAY,
                       lexical_lr_mult: float = LEXICAL_LR_MULT) -> List[Dict]:
    """Four AdamW groups (trainable parameters only; the PLIF ``w`` are frozen), in this order, each in
    ``model.parameters()`` order:

    0. matrices (``ndim >= 2``: projections, spiking-layer ``in_proj`` / ``conv_weight`` / ``W_out``, the tied
    embedding), weight
       decay ``weight_decay``;
    1. vectors and scalars (``ndim < 2``: norm gains, thresholds, gate biases, spiking-layer ``rho``, ...), no weight
    decay; 2. the lexical n-gram tables, no weight decay, learning rate x ``lexical_lr_mult``; 3. the spiking mixer
    vectors ``A_log``, ``dt_bias``, ``D``, ``gamma``, ``conv_bias``, ``pre_norm_gain``, weight decay
       ``weight_decay``.

    Each group carries ``lr_mult``; its ``lr`` is ``lr * lr_mult``."""
    named = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    lexical = {id(p) for p in model.backbone.lexical_memory.tables}
    spiking_vectors = {id(p) for n, p in named if ".mixer_a." in n and n.rsplit(".", 1)[-1] in SPIKING_DECAYED_VECTORS}
    groups = [
        {"params": [p for n, p in named if p.ndim >= 2 and id(p) not in lexical and id(p) not in spiking_vectors],
         "weight_decay": weight_decay, "lr_mult": 1.0, "name": "matrices"},
        {"params": [p for n, p in named if p.ndim < 2 and id(p) not in lexical and id(p) not in spiking_vectors],
         "weight_decay": 0.0, "lr_mult": 1.0, "name": "vectors"},
        {"params": [p for n, p in named if id(p) in lexical], "weight_decay": 0.0, "lr_mult": float(lexical_lr_mult),
         "name": "lexical_tables"},
        {"params": [p for n, p in named if id(p) in spiking_vectors], "weight_decay": weight_decay, "lr_mult": 1.0,
         "name": "spiking_vectors"},
    ]
    if sum(len(g["params"]) for g in groups) != len(named):
        raise RuntimeError("build_param_groups: the groups do not cover every trainable parameter exactly once")
    for g in groups:
        g["lr"] = lr * g["lr_mult"]
    return groups


def set_lr(optimizer, lr: float) -> None:
    for g in optimizer.param_groups:
        g["lr"] = lr * float(g.get("lr_mult", 1.0))


def make_batches(n: int, rows: int, seq_len: int, vocab: int, seed: int, tokens: Optional[Path], device):
    """``n`` (input_ids, targets) pairs of (rows, seq_len): windows of seq_len + 1 tokens, targets shifted by one."""
    out = []
    if tokens is not None:
        obj = torch.load(str(tokens), map_location="cpu", weights_only=True) if tokens.suffix != ".npy" else None
        if obj is None:
            import numpy as np

            stream = torch.from_numpy(np.load(str(tokens), mmap_mode="r").reshape(-1)[: n * rows * (seq_len + 1)]
                                      .astype("int64"))
        else:
            stream = (obj["tokens"] if isinstance(obj, dict) else obj).reshape(-1).to(torch.int64)
        need = n * rows * (seq_len + 1)
        if stream.numel() < need:
            raise SystemExit(f"{tokens}: {stream.numel()} tokens, {need} needed")
        w = stream[:need].reshape(n, rows, seq_len + 1)
    else:
        g = torch.Generator(device="cpu").manual_seed(int(seed))
        w = torch.randint(0, int(vocab), (n, rows, seq_len + 1), generator=g, dtype=torch.int64)
    for i in range(n):
        out.append((w[i, :, :-1].contiguous().to(device), w[i, :, 1:].contiguous().to(device)))
    return out


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0],
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ckpt", default=None, help="checkpoint directory (default: a fresh, calibrated launch)")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--seq-len", type=int, default=1024)
    p.add_argument("--gradient-checkpointing", action="store_true")
    p.add_argument("--warmup", type=int, default=3, help="untimed steps")
    p.add_argument("--steps", type=int, default=20, help="timed steps per trial")
    p.add_argument("--trials", type=int, default=3)
    p.add_argument("--cupti", action="store_true", help="also report the CUPTI device time of a step")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--tokens", default=None, help="token stream (torch.save tensor or .npy) instead of random ids")
    p.add_argument("--json", default=None, help="write the results here")
    a = p.parse_args(argv)

    from neuronspark import NeuronSparkConfig
    from neuronspark.models.neuronspark_lm import NeuronSparkLMHeadModel

    if not torch.cuda.is_available():
        raise SystemExit("needs a CUDA GPU")
    dev = torch.device("cuda", torch.cuda.current_device())
    B, T = int(a.batch_size), int(a.seq_len)
    if T % 256 != 0:
        raise SystemExit("--seq-len must be a multiple of 256 (the whole-row regime runs full scan chunks)")
    if a.ckpt is not None:
        model = NeuronSparkLMHeadModel.from_pretrained(a.ckpt, device=dev)
        vocab = int(model.config.vocab_size)
    else:
        from neuronspark.utils.calibration import CALIBRATION_ROWS, calibrate_fresh_model

        config = NeuronSparkConfig.from_json(_HERE.parent / "configs" / "neuronspark_1p3b.json")
        torch.manual_seed(a.seed)
        model = NeuronSparkLMHeadModel(config, device=dev, init="launch")
        vocab = int(config.vocab_size)
        n_cal = max(1, CALIBRATION_ROWS // B)
        cal = [x for x, _ in make_batches(n_cal, B, T, vocab, a.seed + 1, None, dev)]
        calibrate_fresh_model(model, cal, rows=n_cal * B)
    model.gradient_checkpointing = bool(a.gradient_checkpointing)
    optimizer = torch.optim.AdamW(build_param_groups(model), lr=PEAK_LR, betas=BETAS, eps=EPS,
                                  weight_decay=WEIGHT_DECAY, fused=True)
    params = [q for q in model.parameters() if q.requires_grad]
    batches = make_batches(4, B, T, vocab, a.seed, None if a.tokens is None else Path(a.tokens), dev)
    state = {"i": 0}
    last = {}

    def step():
        x, y = batches[state["i"] % len(batches)]
        set_lr(optimizer, schedule_lr(state["i"]))
        state["i"] += 1
        optimizer.zero_grad(set_to_none=True)
        out = model.loss(x, y)
        out.loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(params, GRAD_CLIP, foreach=True)
        optimizer.step()
        last["loss"], last["grad_norm"] = out.loss.detach(), gn.detach()

    torch.cuda.reset_peak_memory_stats(dev)
    mean_ms, medians, _ = cuda_event_time_ms(step, warmup=a.warmup, iters=a.steps, trials=a.trials, device=dev)
    tokens = B * T
    result = {
        "shape": [B, T],
        "gradient_checkpointing": bool(a.gradient_checkpointing),
        "step_ms": mean_ms,
        "step_ms_trial_medians": medians,
        "tokens_per_s": tokens / (mean_ms / 1e3),
        "peak_allocated_gib": torch.cuda.max_memory_allocated(dev) / 2**30,
        "last_loss": float(last["loss"]),
        "last_grad_norm": float(last["grad_norm"]),
        "device": torch.cuda.get_device_name(dev),
        "torch": torch.__version__,
    }
    if a.cupti:
        k_ms, k_medians, names = cupti_time_ms(step, warmup=1, iters=a.steps, trials=a.trials, device=dev)
        result.update({"kernel_ms": k_ms, "kernel_ms_trial_medians": k_medians, "kernels": len(names)})
    print(f"train step {B} x {T} (gradient checkpointing {result['gradient_checkpointing']}): "
          f"{format_ms(mean_ms, medians)} -> {result['tokens_per_s']:,.1f} tok/s; peak allocated "
          f"{result['peak_allocated_gib']:.2f} GiB; last loss {result['last_loss']:.4f}")
    if a.cupti:
        print(f"  kernel time (CUPTI, cold L2): {format_ms(result['kernel_ms'], result['kernel_ms_trial_medians'])}, "
              f"{result['kernels']} distinct kernels")
    if a.json:
        Path(a.json).write_text(json.dumps(result, indent=1))
        print(f"wrote {a.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
