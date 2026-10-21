#!/usr/bin/env python3
"""Inference latency of the NeuronSpark model: prefill and CUDA-graph decode (``neuronspark.utils.generation``).

* **prefill**: one call ``model(ids (b, L), inference_params=cache, num_last_tokens=1)`` on a cache reset to the row
  start (the reset is not timed), CUDA events around the call, at every b of ``--prefill-batches``;
* **decode step**: one greedy step through the captured single-token step (``update_graph_cache`` /
  ``capture_graph``: static input copy, graph replay, logits copy) and the argmax of its logits, after a prefill of the
  prompt, CUDA events around each of ``--decode-steps`` steps, at every b of ``--decode-batches``; with ``--cupti``
  also the summed device time of one replay's kernels (CUPTI, L2 flushed before each replay: a cold-cache figure, not
  the steady state);
* **generate**: ``decode(ids, model, L + steps, cg=True)`` end to end (host clock, device synchronised), giving the
  output tokens per second of greedy decoding at that batch size.

Every figure is the median over the calls of a trial, then the mean over ``--trials`` (the per-trial medians are
printed too).  The prompts are uniform random ids (``--seed``); the ring capacity of the caches is ``--ring``; the
dendritic-layer state storage is ``--dendritic-state-dtype`` (default: the configuration's, bfloat16).

By default a fresh model is launched from ``NeuronSparkConfig()`` (the default configuration) and calibrated on random
rows (the launch calibration; the timings do not depend on trained weights); ``--ckpt`` loads a local checkpoint
directory (``config.json`` + ``model.safetensors``, as written by ``save_pretrained``) instead.

Example::

    python benchmarks/benchmark_generation.py
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List, Optional

import torch

_HERE = Path(__file__).resolve().parent
for _p in (str(_HERE), str(_HERE.parent)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from _timing import cuda_event_time_ms, cupti_time_ms, format_ms, host_time_ms  # noqa: E402


def _sizes(text: str) -> List[int]:
    return [int(s) for s in text.split(",") if s.strip()]


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0],
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ckpt", default=None, help="a local checkpoint directory (config.json + model.safetensors); "
                                                "default: a fresh model of the default configuration")
    p.add_argument("--prompt-len", type=int, default=1024)
    p.add_argument("--prefill-batches", default="1,8")
    p.add_argument("--decode-batches", default="1,8,32")
    p.add_argument("--decode-steps", type=int, default=256, help="decode steps per trial (and generated tokens)")
    p.add_argument("--prefill-iters", type=int, default=5, help="timed prefill calls per trial")
    p.add_argument("--trials", type=int, default=3)
    p.add_argument("--ring", type=int, default=2048, help="ring capacity of the caches (a multiple of 512)")
    p.add_argument("--dendritic-state-dtype", default=None, choices=("bfloat16", "float32"))
    p.add_argument("--cupti", action="store_true", help="also report the CUPTI device time of a decode replay")
    p.add_argument("--no-generate", action="store_true", help="skip the end-to-end generate timing")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--json", default=None, help="write the results here")
    a = p.parse_args(argv)

    from neuronspark.models.neuronspark_lm import NeuronSparkLMHeadModel
    from neuronspark.utils.cache import resolve_dendritic_state_dtype
    from neuronspark.utils.generation import decode, update_graph_cache

    if not torch.cuda.is_available():
        raise SystemExit("needs a CUDA GPU")
    dev = torch.device("cuda", torch.cuda.current_device())
    L, steps, W = int(a.prompt_len), int(a.decode_steps), int(a.ring)
    if L + steps > W:
        raise SystemExit(f"prompt {L} + {steps} decode steps exceed the ring capacity {W}")
    if a.ckpt is not None:
        model = NeuronSparkLMHeadModel.from_pretrained(a.ckpt, device=dev,
                                                       inference_dendritic_state_dtype=a.dendritic_state_dtype)
    else:
        import dataclasses

        from neuronspark.models.config_neuronspark import NeuronSparkConfig
        from neuronspark.utils.calibration import CALIBRATION_ROWS, calibrate_fresh_model

        config = NeuronSparkConfig()
        if a.dendritic_state_dtype is not None:
            config = dataclasses.replace(config, inference_dendritic_state_dtype=a.dendritic_state_dtype)
        torch.manual_seed(int(a.seed))
        model = NeuronSparkLMHeadModel(config, device=dev, init="launch")
        gc = torch.Generator(device="cpu").manual_seed(int(a.seed) + 1)
        cal = [torch.randint(0, int(config.vocab_size), (8, 1024), generator=gc, dtype=torch.int64).to(dev)
               for _ in range(CALIBRATION_ROWS // 8)]
        calibrate_fresh_model(model, cal, rows=CALIBRATION_ROWS)
    dt = resolve_dendritic_state_dtype(model.config.inference_dendritic_state_dtype)
    model.prepare_inference()
    V = int(model.config.vocab_size)
    g = torch.Generator(device="cpu").manual_seed(int(a.seed))
    bmax = max(_sizes(a.prefill_batches) + _sizes(a.decode_batches))
    prompts = torch.randint(0, V, (bmax, L), generator=g, dtype=torch.int64).to(dev)
    result = {"prompt_len": L, "ring": W, "dendritic_state_dtype": str(dt), "device": torch.cuda.get_device_name(dev),
              "torch": torch.__version__, "prefill": {}, "decode": {}, "generate": {}}

    # ---- prefill
    for b in _sizes(a.prefill_batches):
        ids = prompts[:b]
        ip = model.allocate_inference_cache(b, W, dendritic_state_dtype=dt)

        def prefill(ids=ids, ip=ip):
            with torch.no_grad():
                model(ids, inference_params=ip, num_last_tokens=1)

        ms, med, _ = cuda_event_time_ms(prefill, warmup=2, iters=a.prefill_iters, trials=a.trials, setup=ip.reset,
                                        device=dev)
        result["prefill"][b] = {"ms": ms, "trial_medians": med, "tokens_per_s": b * L / (ms / 1e3)}
        print(f"prefill b={b} T={L}: {format_ms(ms, med)} ({b * L / (ms / 1e3):,.0f} tok/s)")
        del ip

    # ---- decode steps (CUDA-graph replay) and generate
    for b in _sizes(a.decode_batches):
        ids = prompts[:b]
        model._decoding_cache = update_graph_cache(model, model._decoding_cache, b, L, W, dendritic_state_dtype=dt)
        cache = model._decoding_cache
        ip = cache.inference_params
        trial_medians, every = [], []
        for _ in range(int(a.trials)):
            ip.reset()
            with torch.no_grad():
                tok = model(ids, inference_params=ip, num_last_tokens=1).logits[:, -1].argmax(-1, keepdim=True)
            state = {"tok": tok}

            def one_step(state=state):
                state["tok"] = cache.run(state["tok"])[:, -1].argmax(-1, keepdim=True)

            ms, med, times = cuda_event_time_ms(one_step, warmup=0, iters=steps, trials=1, device=dev)
            trial_medians.append(med[0])
            every.append(times[0])
        mean = sum(trial_medians) / len(trial_medians)
        entry = {"step_ms": mean, "trial_medians": trial_medians, "steps_per_trial": steps}
        print(f"decode b={b}: {format_ms(mean, trial_medians)} per step (graph replay)")
        if a.cupti:
            ip.reset()
            with torch.no_grad():
                tok = model(ids, inference_params=ip, num_last_tokens=1).logits[:, -1].argmax(-1, keepdim=True)
            iters = max(1, min(50, (W - L) // (a.trials + 1) - 1))
            k_ms, k_med, names = cupti_time_ms(lambda: cache.run(tok), warmup=1, iters=iters, trials=a.trials,
                                               device=dev)
            entry.update({"kernel_ms": k_ms, "kernel_trial_medians": k_med, "kernels": len(names)})
            print(f"  kernel time per replay (CUPTI, cold L2): {format_ms(k_ms, k_med)}, {len(names)} distinct "
                  "kernels")
        result["decode"][b] = entry
        if not a.no_generate:
            def gen(ids=ids):
                decode(ids, model, L + steps, cg=True, dendritic_state_dtype=dt)

            ms, med, _ = host_time_ms(gen, warmup=1, iters=1, trials=a.trials, device=dev)
            result["generate"][b] = {"ms": ms, "trial_medians": med, "output_tokens_per_s": b * steps / (ms / 1e3)}
            print(f"  generate b={b} {L} + {steps} tokens: {format_ms(ms, med)} -> "
                  f"{b * steps / (ms / 1e3):,.1f} output tok/s")
    model.release_inference()
    if a.json:
        Path(a.json).write_text(json.dumps(result, indent=1))
        print(f"wrote {a.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
