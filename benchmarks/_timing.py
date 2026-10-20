"""Timing helpers shared by the benchmarks.

Three clocks, each reported as the mean over trials of the per-trial median:

* :func:`cupti_time_ms` -- the device time of the kernels one call launches, read from CUPTI activity records through
  ``torch.profiler`` (Kineto's CUPTI activity API).  Before every call the L2 cache is flushed by a 256 MiB write
  (``uint8`` ``zero_``, a ``FillFunctor<unsigned char>`` kernel); the kernels recorded between two consecutive flush
  kernels are one call, so per-call setup issued before the flush is not counted, and an optional name filter keeps
  only the kernels of interest.  Warm-up calls first, no CUDA graph, the median over the calls of a trial.  This is the
  kernel timing method of flashinfer-bench (``bench/timing.py``: CUPTI, warm-up plus timed iterations, cold L2 before
  each call, median; the caller averages the medians of 3 trials).
* :func:`cuda_event_time_ms` -- the time between two CUDA events recorded around each call on the current stream (warm
  L2, no flush): the device-timeline wall time of the call including launch gaps.
* :func:`host_time_ms` -- host wall time (``time.perf_counter``) of each call, the device synchronised before and after:
  end-to-end latency including host work and synchronisation.

The benchmarks report speed only; correctness is established by the tests (the per-operator rule of
:mod:`neuronspark.utils.testing`).
"""
from __future__ import annotations

import statistics
import time
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import torch

__all__ = [
    "FLUSH_BYTES",
    "cupti_time_ms",
    "cuda_event_time_ms",
    "host_time_ms",
    "summary",
    "format_ms",
]

# size of the L2 flush written before every CUPTI-timed call (larger than the L2 of every supported GPU)
FLUSH_BYTES = 256 * 2**20
# name fragment of the flush kernel (torch's fill kernel for a uint8 zero_), the call separator of cupti_time_ms
_FLUSH_KERNEL_MARK = "FillFunctor<unsigned char>"
# documented first-call caches: one flush buffer per device; whether the profiler was primed in this process
_FLUSH_BUFFERS: Dict[torch.device, torch.Tensor] = {}
_PROFILER_PRIMED = {"done": False}


def _device(device: Optional[torch.device]) -> torch.device:
    if device is not None:
        return torch.device(device)
    if not torch.cuda.is_available():
        raise RuntimeError("the benchmarks time CUDA work: no CUDA device is available")
    return torch.device("cuda", torch.cuda.current_device())


def _flush_buffer(dev: torch.device) -> torch.Tensor:
    buf = _FLUSH_BUFFERS.get(dev)
    if buf is None:
        buf = torch.empty(FLUSH_BYTES, dtype=torch.uint8, device=dev)
        _FLUSH_BUFFERS[dev] = buf
    return buf


def cupti_time_ms(fn: Callable[[], object], kernel_filter: Optional[Callable[[str], bool]] = None, warmup: int = 10,
                  iters: int = 50, trials: int = 3, setup: Optional[Callable[[], object]] = None,
                  device: Optional[torch.device] = None) -> Tuple[float, List[float], List[str]]:
    """Per call, the summed device time of the kernels ``fn()`` launches (those for which ``kernel_filter(name)`` holds,
    when given), from CUPTI activity records; the L2 cache flushed before every call; ``setup()`` (when given) runs
    before the flush of every call and is not timed.  Median over the ``iters`` calls of a trial, mean over ``trials``.

    Returns ``(mean of the trial medians in ms, the trial medians, the kernel names seen)``.  Raises when the recorded
    kernels cannot be grouped into ``iters`` calls (the first event names are in the message)."""
    dev = _device(device)
    buf = _flush_buffer(dev)
    if not _PROFILER_PRIMED["done"]:
        # the first profiler session of a process can return no CUDA events: one throw-away session
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]):
            buf.zero_()
            torch.cuda.synchronize(dev)
        _PROFILER_PRIMED["done"] = True
    for _ in range(int(warmup)):
        if setup is not None:
            setup()
        buf.zero_()
        fn()
    torch.cuda.synchronize(dev)
    medians: List[float] = []
    seen = set()
    for _ in range(int(trials)):
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as prof:
            for _ in range(int(iters)):
                if setup is not None:
                    setup()
                buf.zero_()
                fn()
            buf.zero_()
            torch.cuda.synchronize(dev)
        kernels = sorted((e for e in prof.events() if e.device_type == torch.autograd.DeviceType.CUDA),
                         key=lambda e: e.time_range.start)
        calls: List[list] = []
        cur: Optional[list] = None
        for e in kernels:
            if _FLUSH_KERNEL_MARK in e.name:
                if cur is not None:
                    calls.append(cur)
                cur = []
            elif cur is not None and (kernel_filter is None or kernel_filter(e.name)):
                cur.append(e)
        if len(calls) != int(iters):
            names = [e.name[:80] for e in kernels[:12]]
            raise RuntimeError(f"cupti_time_ms: {len(calls)} calls between flush kernels for {iters} calls; "
                               f"{len(kernels)} CUDA events, first: {names}")
        for c in calls:
            seen.update(k.name for k in c)
        medians.append(statistics.median(sum(k.time_range.elapsed_us() for k in c) / 1000.0 for c in calls))
    return sum(medians) / len(medians), medians, sorted(seen)


def cuda_event_time_ms(fn: Callable[[], object], warmup: int = 3, iters: int = 20, trials: int = 3,
                       setup: Optional[Callable[[], object]] = None, device: Optional[torch.device] = None
                       ) -> Tuple[float, List[float], List[List[float]]]:
    """Per call, the time between two CUDA events recorded around ``fn()`` on the current stream; ``setup()`` (when
    given) runs before the first event of every call and is not timed.  Median over the ``iters`` calls of a trial,
    mean over ``trials``.  Returns ``(mean of the trial medians in ms, the trial medians, every call's time per
    trial)``."""
    dev = _device(device)
    for _ in range(int(warmup)):
        if setup is not None:
            setup()
        fn()
    torch.cuda.synchronize(dev)
    medians: List[float] = []
    every: List[List[float]] = []
    for _ in range(int(trials)):
        pairs = []
        for _ in range(int(iters)):
            if setup is not None:
                setup()
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            fn()
            end.record()
            pairs.append((start, end))
        torch.cuda.synchronize(dev)
        times = [s.elapsed_time(e) for s, e in pairs]
        every.append(times)
        medians.append(statistics.median(times))
    return sum(medians) / len(medians), medians, every


def host_time_ms(fn: Callable[[], object], warmup: int = 1, iters: int = 5, trials: int = 3,
                 setup: Optional[Callable[[], object]] = None, device: Optional[torch.device] = None
                 ) -> Tuple[float, List[float], List[List[float]]]:
    """Per call, the host wall time of ``fn()`` with the device synchronised before and after; ``setup()`` (when given)
    runs before the first synchronisation of every call and is not timed.  Median over the ``iters`` calls of a trial,
    mean over ``trials``.  Returns ``(mean of the trial medians in ms, the trial medians, every call's time per
    trial)``."""
    dev = _device(device)
    for _ in range(int(warmup)):
        if setup is not None:
            setup()
        fn()
    torch.cuda.synchronize(dev)
    medians: List[float] = []
    every: List[List[float]] = []
    for _ in range(int(trials)):
        times = []
        for _ in range(int(iters)):
            if setup is not None:
                setup()
            torch.cuda.synchronize(dev)
            t0 = time.perf_counter()
            fn()
            torch.cuda.synchronize(dev)
            times.append((time.perf_counter() - t0) * 1e3)
        every.append(times)
        medians.append(statistics.median(times))
    return sum(medians) / len(medians), medians, every


def summary(values: Sequence[float]) -> Dict[str, float]:
    """Median, mean, min, max and the 10 / 90 percentiles (nearest rank) of ``values``."""
    v = sorted(float(x) for x in values)
    if not v:
        return {"n": 0}
    n = len(v)
    return {"n": n, "median": statistics.median(v), "mean": sum(v) / n, "min": v[0], "max": v[-1],
            "p10": v[min(n - 1, n // 10)], "p90": v[min(n - 1, (9 * n) // 10)]}


def format_ms(mean_ms: float, medians: Sequence[float]) -> str:
    """``"<mean> ms (trial medians a / b / c)"``."""
    return f"{mean_ms:.3f} ms (trial medians " + " / ".join(f"{m:.3f}" for m in medians) + ")"
