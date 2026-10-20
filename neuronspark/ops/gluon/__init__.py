"""Gluon kernels (``triton.experimental.gluon``, Triton 3.5.1) for Blackwell GPUs (compute capability >= 10.0):
warp-specialised tcgen05 / TMEM / TMA kernels.  Every launcher calls
:func:`neuronspark.ops.triton.require_sm100` before launching.  Shared round-to-nearest helpers:
:mod:`neuronspark.ops.gluon._ptx`.
"""
from __future__ import annotations

from neuronspark.ops.triton import SM100, device_capability, num_sms, require_cuda, require_sm100

__all__ = ["SM100", "device_capability", "num_sms", "require_cuda", "require_sm100"]
