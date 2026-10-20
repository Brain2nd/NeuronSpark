"""Rotary position tables of the memory layers and the interleaved RoPE reference.

Every memory head rotates its ``dk / 2`` pairs ``(x[2j], x[2j + 1])`` with the per-head frequency ladder
``theta_j = base^(-2j / dk)``, ``j = 0 .. dk/2 - 1`` (dk = 256): head ``h`` reads the table columns
``[h * dk/2, (h + 1) * dk/2)``, and all heads share the same ladder.  The tables are fp32 ``(max_seq_len, D_key / 2)``
= (8192, 768), computed on the CPU (device-independent values) and then moved to the requested device.  They
are non-persistent buffers of the memory layer (not in the checkpoint).

``apply_rope_interleaved_ref(x, cos, sin)``: ``y[2p] = x[2p] * c - x[2p+1] * s`` and ``y[2p+1] = x[2p+1] * c +
x[2p] * s`` evaluated in ``x``'s dtype (each product and each sum rounded to that dtype); the caller passes ``cos`` /
``sin`` already cast to ``x.dtype`` (bf16) and broadcastable to ``x[..., ::2]``.
"""
from __future__ import annotations

from typing import Optional, Tuple

import torch

__all__ = ["precompute_rope_freqs", "rope_head_band_tables", "apply_rope_interleaved_ref"]


def precompute_rope_freqs(dim: int, max_seq_len: int = 8192, base: float = 10000.0) -> Tuple[torch.Tensor, torch.Tensor]:
    """fp32 ``(max_seq_len, dim / 2)`` cos / sin of ``t * base^(-2j / dim)`` (CPU)."""
    freqs = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
    t = torch.arange(max_seq_len)
    freqs = torch.outer(t, freqs)
    return freqs.cos(), freqs.sin()


def rope_head_band_tables(D_key: int, heads: int, max_seq_len: int, base: float,
                          device: Optional[torch.device] = None) -> Tuple[torch.Tensor, torch.Tensor]:
    """fp32 ``(max_seq_len, D_key / 2)`` cos / sin tables: the head-width ladder ``precompute_rope_freqs(D_key / heads,
    max_seq_len, base)`` placed in every head's column slice.  Computed on the CPU, then moved to ``device``."""
    D_key, heads = int(D_key), int(heads)
    if D_key % heads != 0 or (D_key // heads) % 2 != 0:
        raise ValueError(f"rope_head_band_tables: the head width D_key / heads must be even (D_key {D_key}, "
                         f"heads {heads})")
    dk = D_key // heads
    half = dk // 2
    head_cos, head_sin = precompute_rope_freqs(dk, max_seq_len=int(max_seq_len), base=float(base))
    cos = torch.empty((int(max_seq_len), D_key // 2), dtype=torch.float32)
    sin = torch.empty((int(max_seq_len), D_key // 2), dtype=torch.float32)
    for h in range(heads):
        cos[:, h * half:(h + 1) * half] = head_cos
        sin[:, h * half:(h + 1) * half] = head_sin
    if device is not None:
        cos, sin = cos.to(device=device), sin.to(device=device)
    return cos, sin


def apply_rope_interleaved_ref(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Interleaved-pair rotation of the last dimension of ``x`` in ``x``'s dtype (see the module docstring)."""
    if x.shape[-1] % 2 != 0:
        raise ValueError(f"apply_rope_interleaved_ref: the last dimension must be even, got {x.shape[-1]}")
    x_even = x[..., ::2]
    x_odd = x[..., 1::2]
    y = torch.empty_like(x)
    y[..., ::2] = x_even * cos - x_odd * sin
    y[..., 1::2] = x_odd * cos + x_even * sin
    return y
