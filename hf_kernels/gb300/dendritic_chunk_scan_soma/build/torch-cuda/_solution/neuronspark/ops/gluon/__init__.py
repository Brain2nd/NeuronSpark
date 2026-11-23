"""Gluon kernels (``triton.experimental.gluon``, Triton 3.5.1) for datacenter Blackwell GPUs (compute capability 10.x):
warp-specialised tcgen05 / TMEM / TMA kernels.  Every launcher calls
:func:`neuronspark.ops.triton.require_sm100` before launching.  Shared round-to-nearest helpers:
:mod:`neuronspark.ops.gluon._ptx`.

This copy of ``neuronspark/ops/gluon/__init__.py`` is empty:
the exported operator imports modules of the package only.
"""
