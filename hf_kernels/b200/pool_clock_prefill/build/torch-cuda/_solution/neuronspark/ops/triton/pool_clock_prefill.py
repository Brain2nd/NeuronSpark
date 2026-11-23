"""Pool clocks of a prefill call: the decay, its mean log, the gained cumulative sum and the ring write in two kernels.

A hippocampal layer keeps one clock per head: the cumulative sum of the mean log decay of its 256 decay channels times
the head's eligibility slope (:mod:`neuronspark.modules.pool_attention`).  For a call of T tokens from the carried
clocks ``run`` (B, H)::

    a[t, b, c]   = exp(-f32(rate[c]) * softplus(f32(logit[t, b, c]))) * base_decay          fp32
    m[t, b]      = mean_c log(max(a[t, b, c], floor))
    cum[b, h, t] = run[b, h] + sum_{s <= t} m[s, b] * gains[h]
    cum_ring[b, h, (ptr + t) % W] = cum[b, h, t];    run[b, h] = cum[b, h, T - 1]

The torch chain (:func:`pool_clock_prefill_ref`) runs about eighteen kernels on (T, B, C) and (T, B, H) tensors.  Here

* ``_decay_mean_kernel`` forms ``m`` for CLOCK_ROWS rows r = t * B + b per program with the chain's fp32 operations
  (ATen's softplus: x above 20 is returned as it is, else ``log1p(exp(x))``; libdevice ``expf`` / ``log1pf`` /
  ``logf``; no FMA contraction); the sum over the channels is a tree reduction (the chain's ``mean`` reduces in
  another order); the means are stored row-major, (B, T), so that the scan of a row reads consecutive values;
* ``_clock_scan_kernel`` runs one program per row b over the tokens, one lane per head: the gained increment and the
  chain's own recurrence ``acc = acc + m[t] * gains`` from the carried clock, strictly left to right (the order of
  ``torch.cumsum`` and of the whole-row and decode clocks: a clock of large magnitude rounds every small increment the
  same way in every regime), the store of the clocks in the (B, H, T) layout the ring read takes and into the ring
  slots, and the carried clocks advanced in place.  The tokens go in tiles of sixteen: the means of the next tile are
  loaded while the present one is summed, and the ring slot is a 32-bit count that wraps at W (one 64-bit remainder
  per program, none per token).

The mean's reduction order is the only difference from the chain: the per-operator rule class.  (A single clock,
B = H = 1, is a single column for ``torch.cumsum``, which then scans in another order; the kernel's order is the same
for every B and H.)
"""
from __future__ import annotations; from ....neuronspark._namespace import NAMESPACE as _KERNEL_OP_NAMESPACE, OPS as _KERNEL_OPS

import torch
import torch.nn.functional as F
import triton
import triton.language as tl
from triton.language.extra import libdevice

from ....neuronspark.ops.triton import require_cuda

__all__ = ["CLOCK_ROWS", "CLOCK_WARPS", "CLOCK_SCAN_WARPS", "pool_clock_prefill", "pool_clock_prefill_ref"]

# ================================================================================================ fixed constants
#: rows (t, b) per program and warps of the decay kernel; grid (row blocks,)
CLOCK_ROWS = 8
CLOCK_WARPS = 4
#: warps of the scan kernel (one lane per head); grid (B,)
CLOCK_SCAN_WARPS = 1


# ================================================================================================ kernels
@triton.jit
def _decay_mean_kernel(X, RATE, M, R, T, B, C, sx_t, sx_b, base, floor, inv_c, BR: tl.constexpr,
                       BC: tl.constexpr):
    r = tl.program_id(0) * BR + tl.arange(0, BR)
    mr = r < R
    t = (r // B).to(tl.int64)
    b = (r % B).to(tl.int64)
    c = tl.arange(0, BC)
    mc = c < C
    m2 = mr[:, None] & mc[None, :]
    x = tl.load(X + (t * sx_t + b * sx_b)[:, None] + c[None, :], mask=m2, other=0.0).to(tl.float32)
    rate = tl.load(RATE + c, mask=mc, other=0.0).to(tl.float32)
    sp = tl.where(x > 20.0, x, libdevice.log1p(libdevice.exp(x)))
    a = libdevice.exp(-(rate[None, :] * sp)) * base
    la = tl.where(m2, libdevice.log(tl.maximum(a, floor)), 0.0)
    tl.store(M + b * T + t, tl.sum(la, axis=1) * inv_c, mask=mr)        # (B, T): the means of a row side by side


@triton.jit
def _clock_step(acc, inc, cum, ring, t, slot, T, W, mh):
    """one token of the recurrence: the add, the store of the clock and of its ring slot (slot < 2 W)"""
    live = t < T
    acc = tl.where(live, acc + inc, acc)
    tl.store(cum + t, acc, mask=mh & live)
    tl.store(ring + tl.where(slot < W, slot, slot - W), acc, mask=mh & live)
    return acc


@triton.jit
def _clock_scan_kernel(M, GAINS, RUN, RING, CUM, P0, T, H, W, BH: tl.constexpr):
    """one program per row b, one lane per head, sixteen tokens per tile: a tile's means are loaded while the tile
    before it is summed, the adds run strictly left to right"""
    b = tl.program_id(0)
    h = tl.arange(0, BH)
    mh = h < H
    row = b * H + h
    g = tl.load(GAINS + h, mask=mh, other=0.0)
    acc = tl.load(RUN + row, mask=mh, other=0.0)
    slot = (tl.load(P0) % W).to(tl.int32)          # the one 64-bit remainder; T <= W: a slot wraps at most once
    cum = CUM + row * T
    ring = RING + row * W
    mean = M + b * T
    last = T - 1                                   # a load past the call reads the last token's mean (not used)
    n0 = tl.load(mean + tl.minimum(0, last))
    n1 = tl.load(mean + tl.minimum(1, last))
    n2 = tl.load(mean + tl.minimum(2, last))
    n3 = tl.load(mean + tl.minimum(3, last))
    n4 = tl.load(mean + tl.minimum(4, last))
    n5 = tl.load(mean + tl.minimum(5, last))
    n6 = tl.load(mean + tl.minimum(6, last))
    n7 = tl.load(mean + tl.minimum(7, last))
    n8 = tl.load(mean + tl.minimum(8, last))
    n9 = tl.load(mean + tl.minimum(9, last))
    n10 = tl.load(mean + tl.minimum(10, last))
    n11 = tl.load(mean + tl.minimum(11, last))
    n12 = tl.load(mean + tl.minimum(12, last))
    n13 = tl.load(mean + tl.minimum(13, last))
    n14 = tl.load(mean + tl.minimum(14, last))
    n15 = tl.load(mean + tl.minimum(15, last))
    for t0 in range(0, T, 16):
        m0 = n0
        m1 = n1
        m2 = n2
        m3 = n3
        m4 = n4
        m5 = n5
        m6 = n6
        m7 = n7
        m8 = n8
        m9 = n9
        m10 = n10
        m11 = n11
        m12 = n12
        m13 = n13
        m14 = n14
        m15 = n15
        n0 = tl.load(mean + tl.minimum(t0 + 16, last))
        n1 = tl.load(mean + tl.minimum(t0 + 17, last))
        n2 = tl.load(mean + tl.minimum(t0 + 18, last))
        n3 = tl.load(mean + tl.minimum(t0 + 19, last))
        n4 = tl.load(mean + tl.minimum(t0 + 20, last))
        n5 = tl.load(mean + tl.minimum(t0 + 21, last))
        n6 = tl.load(mean + tl.minimum(t0 + 22, last))
        n7 = tl.load(mean + tl.minimum(t0 + 23, last))
        n8 = tl.load(mean + tl.minimum(t0 + 24, last))
        n9 = tl.load(mean + tl.minimum(t0 + 25, last))
        n10 = tl.load(mean + tl.minimum(t0 + 26, last))
        n11 = tl.load(mean + tl.minimum(t0 + 27, last))
        n12 = tl.load(mean + tl.minimum(t0 + 28, last))
        n13 = tl.load(mean + tl.minimum(t0 + 29, last))
        n14 = tl.load(mean + tl.minimum(t0 + 30, last))
        n15 = tl.load(mean + tl.minimum(t0 + 31, last))
        acc = _clock_step(acc, g * m0, cum, ring, t0 + 0, slot + 0, T, W, mh)
        acc = _clock_step(acc, g * m1, cum, ring, t0 + 1, slot + 1, T, W, mh)
        acc = _clock_step(acc, g * m2, cum, ring, t0 + 2, slot + 2, T, W, mh)
        acc = _clock_step(acc, g * m3, cum, ring, t0 + 3, slot + 3, T, W, mh)
        acc = _clock_step(acc, g * m4, cum, ring, t0 + 4, slot + 4, T, W, mh)
        acc = _clock_step(acc, g * m5, cum, ring, t0 + 5, slot + 5, T, W, mh)
        acc = _clock_step(acc, g * m6, cum, ring, t0 + 6, slot + 6, T, W, mh)
        acc = _clock_step(acc, g * m7, cum, ring, t0 + 7, slot + 7, T, W, mh)
        acc = _clock_step(acc, g * m8, cum, ring, t0 + 8, slot + 8, T, W, mh)
        acc = _clock_step(acc, g * m9, cum, ring, t0 + 9, slot + 9, T, W, mh)
        acc = _clock_step(acc, g * m10, cum, ring, t0 + 10, slot + 10, T, W, mh)
        acc = _clock_step(acc, g * m11, cum, ring, t0 + 11, slot + 11, T, W, mh)
        acc = _clock_step(acc, g * m12, cum, ring, t0 + 12, slot + 12, T, W, mh)
        acc = _clock_step(acc, g * m13, cum, ring, t0 + 13, slot + 13, T, W, mh)
        acc = _clock_step(acc, g * m14, cum, ring, t0 + 14, slot + 14, T, W, mh)
        acc = _clock_step(acc, g * m15, cum, ring, t0 + 15, slot + 15, T, W, mh)
        slot = tl.where(slot + 16 < W, slot + 16, slot + 16 - W)
    tl.store(RUN + row, acc, mask=mh)


# ================================================================================================ launcher
def _check(logit, rate, gains, run, cum_ring, ptr):
    for name, x in (("decay_logit", logit), ("rate", rate), ("gains", gains), ("run", run), ("cum_ring", cum_ring),
                    ("ptr", ptr)):
        if not isinstance(x, torch.Tensor):
            raise RuntimeError(f"neuronspark.pool_clock_prefill: {name} must be a tensor, got {type(x).__name__}")
        require_cuda(x, f"pool_clock_prefill ({name})")
        if x.device != logit.device:
            raise RuntimeError(f"neuronspark.pool_clock_prefill: {name} is on {x.device}, decay_logit on "
                               f"{logit.device}")
    if logit.dim() != 3 or logit.dtype not in (torch.bfloat16, torch.float32) or logit.stride(2) != 1:
        raise RuntimeError(f"neuronspark.pool_clock_prefill: decay_logit must be bf16 / fp32 (T, B, C) with unit last "
                           f"stride, got {logit.dtype} {tuple(logit.shape)} strides {tuple(logit.stride())}")
    T, B, C = (int(s) for s in logit.shape)
    if T < 1 or B < 1 or C < 1 or C > 4096:
        raise RuntimeError(f"neuronspark.pool_clock_prefill: decay_logit {tuple(logit.shape)} is outside the kernel's "
                           f"shapes (T, B >= 1, 1 <= C <= 4096)")
    if rate.dtype not in (torch.bfloat16, torch.float32) or tuple(rate.shape) != (C,) or not rate.is_contiguous():
        raise RuntimeError(f"neuronspark.pool_clock_prefill: rate must be a contiguous bf16 / fp32 ({C},) tensor, got "
                           f"{rate.dtype} {tuple(rate.shape)}")
    if gains.dtype != torch.float32 or gains.dim() != 1 or not gains.is_contiguous() or not 1 <= gains.numel() <= 64:
        raise RuntimeError(f"neuronspark.pool_clock_prefill: gains must be a contiguous fp32 (H,) tensor, H <= 64, "
                           f"got {gains.dtype} {tuple(gains.shape)}")
    H = int(gains.numel())
    if run.dtype != torch.float32 or tuple(run.shape) != (B, H) or not run.is_contiguous():
        raise RuntimeError(f"neuronspark.pool_clock_prefill: run must be a contiguous fp32 ({B}, {H}) tensor, got "
                           f"{run.dtype} {tuple(run.shape)}")
    if (cum_ring.dtype != torch.float32 or cum_ring.dim() != 3 or tuple(cum_ring.shape[:2]) != (B, H)
            or not cum_ring.is_contiguous()):
        raise RuntimeError(f"neuronspark.pool_clock_prefill: cum_ring must be a contiguous fp32 ({B}, {H}, W) tensor, "
                           f"got {cum_ring.dtype} {tuple(cum_ring.shape)}")
    W = int(cum_ring.shape[2])
    if T > W:
        raise RuntimeError(f"neuronspark.pool_clock_prefill: a call of {T} tokens exceeds the ring capacity {W}")
    if ptr.dtype != torch.int64 or ptr.numel() != 1:
        raise RuntimeError(f"neuronspark.pool_clock_prefill: ptr must hold one int64 value, got {ptr.dtype} "
                           f"{tuple(ptr.shape)}")
    # 32-bit offsets: the scan's padded head lanes (at most 64) and its look-ahead of two tiles past T and W
    if (max(abs(int(s)) * int(n) for s, n in zip(logit.stride(), logit.shape)) >= 2 ** 31
            or (B * H + 64) * (W + 32) >= 2 ** 31):
        raise RuntimeError("neuronspark.pool_clock_prefill: tensor too large")
    return T, B, C, H, W


def _pool_clock_prefill_fwd(logit, rate, base_decay, floor, gains, run, cum_ring, ptr):
    T, B, C, H, W = _check(logit, rate, gains, run, cum_ring, ptr)
    dev = logit.device
    mean = torch.empty((B * T,), dtype=torch.float32, device=dev)
    cum = torch.empty((B, H, T), dtype=torch.float32, device=dev)
    _decay_mean_kernel[(triton.cdiv(T * B, CLOCK_ROWS),)](
        logit, rate, mean, T * B, T, B, C, int(logit.stride(0)), int(logit.stride(1)), float(base_decay), float(floor),
        1.0 / C, BR=CLOCK_ROWS, BC=triton.next_power_of_2(C), num_warps=CLOCK_WARPS, enable_fp_fusion=False)
    _clock_scan_kernel[(B,)](
        mean, gains, run, cum_ring, cum, ptr, T, H, W, BH=triton.next_power_of_2(H),
        num_warps=CLOCK_SCAN_WARPS, enable_fp_fusion=False)
    return cum


@torch.library.custom_op(_KERNEL_OP_NAMESPACE + "::pool_clock_prefill", mutates_args=("run", "cum_ring"))
def _pool_clock_prefill_op(decay_logit: torch.Tensor, rate: torch.Tensor, base_decay: float, floor: float,
                           gains: torch.Tensor, run: torch.Tensor, cum_ring: torch.Tensor,
                           ptr: torch.Tensor) -> torch.Tensor:
    return _pool_clock_prefill_fwd(decay_logit, rate, base_decay, floor, gains, run, cum_ring, ptr)


@_pool_clock_prefill_op.register_fake
def _pool_clock_prefill_fake(decay_logit, rate, base_decay, floor, gains, run, cum_ring, ptr):
    T, B = int(decay_logit.shape[0]), int(decay_logit.shape[1])
    return torch.empty((B, int(gains.numel()), T), dtype=torch.float32, device=decay_logit.device)


def pool_clock_prefill(decay_logit: torch.Tensor, rate: torch.Tensor, base_decay: float, floor: float,
                       gains: torch.Tensor, run: torch.Tensor, cum_ring: torch.Tensor,
                       ptr: torch.Tensor) -> torch.Tensor:
    """(B, H, T) fp32 contiguous clocks of the call's tokens (see the module docstring): ``decay_logit`` (T, B, C)
    bf16 / fp32 with unit last stride, ``rate`` (C,) bf16 / fp32 the decay rate, ``base_decay`` and ``floor`` the
    decay's factor and its lower clamp, ``gains`` (H,) fp32 the eligibility slopes, ``run`` (B, H) fp32 the carried
    clocks (advanced in place), ``cum_ring`` (B, H, W) fp32 the clock rings (the call's clocks written in place at the
    slots (ptr + t) % W), ``ptr`` one int64 value (read on the device).  No autograd."""
    return _KERNEL_OPS.pool_clock_prefill(decay_logit, rate, float(base_decay), float(floor), gains, run,
                                                    cum_ring, ptr)


# ================================================================================================ reference
def pool_clock_prefill_ref(decay_logit: torch.Tensor, rate: torch.Tensor, base_decay: float, floor: float,
                           gains: torch.Tensor, run: torch.Tensor, cum_ring: torch.Tensor,
                           ptr: torch.Tensor) -> torch.Tensor:
    """The hippocampal layer's torch chain (same arguments, in-place updates and output as
    :func:`pool_clock_prefill`)."""
    T = int(decay_logit.shape[0])
    H, W = int(gains.numel()), int(cum_ring.shape[2])
    a = torch.exp(-rate * F.softplus(decay_logit.float())) * float(base_decay)
    la = a.float().clamp_min(float(floor)).log().mean(dim=-1, keepdim=True) * gains.view(1, 1, H)
    cum = torch.cumsum(torch.cat([run.unsqueeze(0), la], dim=0), dim=0)[1:]
    slots = (ptr.reshape(()) + torch.arange(T, device=decay_logit.device, dtype=torch.int64)) % W
    out = cum.permute(1, 2, 0).contiguous()
    cum_ring.index_copy_(2, slots, out)
    run.copy_(cum[-1])
    return out
