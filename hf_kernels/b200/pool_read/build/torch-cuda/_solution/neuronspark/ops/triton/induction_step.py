"""Decode-step induction match of the model-level keys ring (T = 1): one kernel, one program per row.

Every ring slot ``w`` holds the ``n`` ids that preceded the event written there (``keys[b, w, i] = x_{s-1-i}``).  The
query of the step is the window of the row's last ids with the new token (``q_i = x_{t-i}``).  Per row::

    pos(w)   = ptr - ((ptr - w) mod W)                                  the row position held by slot w (< 0: never)
    m_k[w]   = [0 <= pos(w) < ptr] and for all i < k: keys[w, i] == q_i and q_i != pad        k = 1 .. n

then the step's own keys are written at the slot ``ptr % W`` and the window is rolled to end with the new token.
Integer comparisons only: the matches are those of the torch chain (``induction_step_ref``).
"""
from __future__ import annotations; from ....neuronspark._namespace import NAMESPACE as _KERNEL_OP_NAMESPACE, OPS as _KERNEL_OPS

import torch
import triton
import triton.language as tl

from ....neuronspark.ops.triton import require_cuda

__all__ = ["INDUCTION_STEP_BLOCK", "INDUCTION_STEP_WARPS", "induction_step", "induction_step_ref"]

# ================================================================================================ fixed constants
#: ring slots per inner step of a row's program, warps
INDUCTION_STEP_BLOCK = 256
INDUCTION_STEP_WARPS = 4


# ================================================================================================ kernel
@triton.jit
def _induction_step_kernel(IDS, WIN, KEYS, PTR, OUT, B, W, pad, N: tl.constexpr, BW: tl.constexpr):
    b = tl.program_id(0).to(tl.int64)
    ptr = tl.load(PTR)
    x = tl.load(IDS + b)
    for w0 in range(0, W, BW):
        ww = w0 + tl.arange(0, BW)
        wm = ww < W
        r = (ptr - ww) % W
        r = tl.where(r < 0, r + W, r)                                      # the non-negative remainder
        pos = ptr - r
        acc = wm & (pos >= 0) & (pos < ptr)                                # an event of the strict past
        for i in tl.static_range(N):
            if i == 0:
                q = x
            else:
                q = tl.load(WIN + b * N + (N - i))
            key = tl.load(KEYS + (b * W + ww) * N + i, mask=wm, other=0)
            acc = acc & (key == q) & (q != pad)
            tl.store(OUT + (i * B + b) * W + ww, acc.to(tl.uint8), mask=wm)
    # The state writes.  Every thread of the program runs the scalar loads and stores below, and the warps are not in
    # step: each phase reads only what no thread writes in it, and a barrier ends it.
    tl.debug_barrier()                                                    # every warp has finished its matches
    slot = ptr % W
    for i in tl.static_range(N):                                          # this token's keys: x_{t-1-i}
        tl.store(KEYS + (b * W + slot) * N + i, tl.load(WIN + b * N + (N - 1 - i)))
    tl.debug_barrier()                                                    # the slot holds the old window, reversed
    for j in tl.static_range(N - 1):                                      # the window rolled to end with x_t
        tl.store(WIN + b * N + j, tl.load(KEYS + (b * W + slot) * N + (N - 2 - j)))
    tl.store(WIN + b * N + (N - 1), x)


# ================================================================================================ launcher
def _need(cond: bool, msg: str) -> None:
    if not cond:
        raise RuntimeError("neuronspark.ops.induction_step: " + msg)


def _induction_step_fwd(ids: torch.Tensor, win: torch.Tensor, keys: torch.Tensor, ring_ptr: torch.Tensor,
                        pad: int) -> torch.Tensor:
    require_cuda(ids, "induction_step")
    _need(keys.dim() == 3 and keys.dtype == torch.int64 and keys.is_contiguous(),
          f"the keys ring must be a contiguous (B, W, n) int64 tensor, got {keys.dtype} {tuple(keys.shape)}")
    B, W, n = (int(s) for s in keys.shape)
    _need(tuple(ids.shape) == (B,) and ids.dtype == torch.int64 and ids.is_contiguous(),
          f"ids must be a contiguous ({B},) int64 tensor, got {ids.dtype} {tuple(ids.shape)}")
    _need(tuple(win.shape) == (B, n) and win.dtype == torch.int64 and win.is_contiguous(),
          f"the window must be a contiguous ({B}, {n}) int64 tensor, got {win.dtype} {tuple(win.shape)}")
    _need(ring_ptr.numel() == 1 and ring_ptr.dtype == torch.int64, "ring_ptr must be a (1,) int64 tensor")
    _need(win.device == ids.device and keys.device == ids.device and ring_ptr.device == ids.device,
          "ids, the window, the keys ring and ring_ptr must be on one device")
    out = torch.empty((n, B, W), dtype=torch.uint8, device=ids.device)
    _induction_step_kernel[(B,)](ids, win, keys, ring_ptr, out, B, W, int(pad), N=n, BW=INDUCTION_STEP_BLOCK,
                                 num_warps=INDUCTION_STEP_WARPS)
    return out


@torch.library.custom_op(_KERNEL_OP_NAMESPACE + "::induction_step", mutates_args=("win", "keys"))
def _induction_step_op(ids: torch.Tensor, win: torch.Tensor, keys: torch.Tensor, ring_ptr: torch.Tensor,
                       pad: int) -> torch.Tensor:
    return _induction_step_fwd(ids, win, keys, ring_ptr, pad)


@_induction_step_op.register_fake
def _induction_step_fake(ids, win, keys, ring_ptr, pad):
    return ids.new_empty((keys.shape[2], keys.shape[0], keys.shape[1]), dtype=torch.uint8)


# ================================================================================================ public function
def induction_step(ids: torch.Tensor, win: torch.Tensor, keys: torch.Tensor, ring_ptr: torch.Tensor,
                   pad: int) -> torch.Tensor:
    """(n, B, W) uint8 (0 / 1): entry [k - 1] is the order-k match of the step's query against every ring slot.
    ``ids`` (B,) int64: the step's tokens; ``win`` (B, n) int64: the rows' last n ids, oldest first (rolled in place);
    ``keys`` (B, W, n) int64: the keys ring (this step's keys written at ``ring_ptr % W``); ``ring_ptr`` (1,) int64, a
    count (>= 0); ``pad``: the id that never matches.  ``win`` and ``keys`` share no memory."""
    return _KERNEL_OPS.induction_step(ids, win, keys, ring_ptr, int(pad))


# ================================================================================================ pure-torch reference
@torch.no_grad()
def induction_step_ref(ids: torch.Tensor, win: torch.Tensor, keys: torch.Tensor, ring_ptr: torch.Tensor,
                       pad: int) -> torch.Tensor:
    """The torch chain of the session match for one token (keys written, window rolled, cumulative equalities over the
    strict past)."""
    B, W, n = (int(s) for s in keys.shape)
    ext = torch.cat([win, ids.view(B, 1)], dim=1)                          # (B, n + 1): ext[:, n] = x_t
    keys_new = torch.stack([ext[:, n - 1 - i] for i in range(n)], dim=-1)  # (B, n): x_{t-1-i}
    qwin = torch.stack([ext[:, n - i] for i in range(n)], dim=-1)          # (B, n): x_{t-i}
    keys.index_copy_(1, ring_ptr.reshape(1) % W, keys_new.unsqueeze(1))
    win.copy_(ext[:, 1:])
    w = torch.arange(W, device=ids.device, dtype=torch.int64)
    pos = ring_ptr.reshape(1) - torch.remainder(ring_ptr.reshape(1) - w, W)
    acc = ((pos >= 0) & (pos <= ring_ptr.reshape(1) - 1)).view(1, W).expand(B, W)
    eq = (keys == qwin.unsqueeze(1)) & (qwin != pad).unsqueeze(1)          # (B, W, n)
    out = []
    for k in range(n):
        acc = acc & eq[..., k]
        out.append(acc)
    return torch.stack(out, dim=0).to(torch.uint8)
