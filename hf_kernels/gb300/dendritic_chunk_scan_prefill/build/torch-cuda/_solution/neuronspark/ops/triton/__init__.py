"""Triton kernels of the NeuronSpark operators (Triton 3.5.1).

This copy of ``neuronspark/ops/triton/__init__.py`` holds the definitions the exported operator uses,
and nothing else of the package file.
"""

from __future__ import annotations

import functools
from typing import Tuple, Union
import torch

__all__ = ["num_sms", "require_cuda", "require_sm100"]


SM100 = (10, 0)


_Device = Union[torch.device, str, int, torch.Tensor]


def _as_device(dev: _Device) -> torch.device:
    if isinstance(dev, torch.Tensor):
        dev = dev.device
    d = torch.device(dev) if not isinstance(dev, int) else torch.device("cuda", dev)
    if d.type != "cuda":
        return d
    return torch.device("cuda", torch.cuda.current_device() if d.index is None else d.index)


@functools.lru_cache(maxsize=None)
def _capability(index: int) -> Tuple[int, int]:
    return tuple(torch.cuda.get_device_capability(index))


@functools.lru_cache(maxsize=None)
def _sm_count(index: int) -> int:
    return int(torch.cuda.get_device_properties(index).multi_processor_count)


def device_capability(dev: _Device) -> Tuple[int, int]:
    """(major, minor) compute capability of a CUDA device (cached per device: a documented first-call cache)."""
    d = _as_device(dev)
    if d.type != "cuda":
        raise RuntimeError(f"neuronspark: a CUDA device is required, got {d}")
    return _capability(d.index)


def num_sms(dev: _Device) -> int:
    """Streaming-multiprocessor count of a CUDA device (cached per device)."""
    d = _as_device(dev)
    if d.type != "cuda":
        raise RuntimeError(f"neuronspark: a CUDA device is required, got {d}")
    return _sm_count(d.index)


def require_cuda(t: torch.Tensor, what: str) -> None:
    """Raise unless ``t`` is a CUDA tensor."""
    if not isinstance(t, torch.Tensor) or not t.is_cuda:
        raise RuntimeError(f"neuronspark.{what}: CUDA tensors are required")


def require_sm100(dev: _Device, what: str) -> None:
    """Raise unless the device is a datacenter Blackwell GPU (compute capability 10.x: B200 / GB200 10.0, B300 / GB300
    10.3): TMA / tcgen05 / TMEM / Gluon kernels.  Compute capability 12.x (RTX 50, RTX PRO 6000 Blackwell, GB10) has
    no tcgen05 or tensor memory."""
    cap = device_capability(dev)
    if cap[0] != SM100[0]:
        raise RuntimeError(f"neuronspark.{what}: requires a datacenter Blackwell GPU (compute capability 10.x: B200, "
                           f"GB200, B300, GB300), got {cap[0]}.{cap[1]}")
