"""Fused readout balance + cross entropy of the training loss (forward and backward) on the dense (R, V) logits.

The readout of the training regime is ``logits = bf16(rows @ E^T)`` (R, V) followed by the readout balance: per width
``w`` of the configuration (the default (16, 32)), in order, ``logits[r, v] = bf16(fp32(logits[r, v]) +
fp32(c_w[r, v % w]))`` with the bf16 contributions ``c_w = bf16(rows @ W_w^T)`` (R, w) -- one rounding per width -- and
the per-token loss ``F.cross_entropy(balanced.float(), labels, ignore_index=-100, reduction="none")``.  The operator
``balance_ce_per_token`` computes that loss and its backward without materialising the balanced logits, for one to four
widths and any number of rows, by the kernel pair of :mod:`neuronspark.ops.triton.balance_stack_ce` (the forward keeps
the row max and logsum; the backward forms the balanced values again and writes the bf16 logits gradient and every
contribution's gradient).  The numerics are stated there.  The reference ``balance_ce_per_token_ref`` is the torch
chain itself.  Tests: the per-operator tolerance rule of :mod:`neuronspark.utils.testing` against the reference.

Layout: the logits are the dense (R, V) bf16 tensor of the readout (unit column stride, any row stride >= V).  The
gradient is a new dense bf16 tensor with the row length of the tensor it is returned for: (R, V) for dense logits; for
logits that are the leading V columns of a wider dense (R, P) tensor (the padded output of the readout GEMM,
:mod:`neuronspark.ops.linear`) the gradient is returned for that tensor, (R, P) with its columns V .. P - 1 written as
zeros by the backward kernel, so autograd has no slice to undo (no zero fill and no copy of the (R, V) gradient).
"""; from ....neuronspark._namespace import NAMESPACE as _KERNEL_OP_NAMESPACE, OPS as _KERNEL_OPS
from typing import List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F

from ....neuronspark.ops.triton.balance_add import balance_add_ref
from ....neuronspark.ops.triton.balance_stack_ce import check_stack, stack_ce_bwd, stack_ce_fwd

__all__ = ["IGNORE_INDEX", "BalanceCEFn", "balance_ce_per_token", "balance_ce_per_token_ref"]

IGNORE_INDEX = -100


def _check(logits, labels, contribs, widths) -> Tuple[int, int, Tuple[int, ...]]:
    if not isinstance(logits, torch.Tensor) or not logits.is_cuda or logits.dtype != torch.bfloat16:
        raise RuntimeError("balance_ce_per_token: logits must be a CUDA bfloat16 tensor")
    if logits.dim() != 2 or logits.stride(1) != 1 or (logits.shape[0] > 1 and logits.stride(0) < logits.shape[1]):
        raise RuntimeError("balance_ce_per_token: logits must be a 2-D (R, V) tensor with unit column stride and a "
                           f"row stride >= V, got shape {tuple(logits.shape)} strides {tuple(logits.stride())}")
    R, V = int(logits.shape[0]), int(logits.shape[1])
    if R < 1 or R >= 2**31 - 1 or R * max(V, int(logits.stride(0))) >= 2**62:
        raise RuntimeError(f"balance_ce_per_token: unsupported size R={R} V={V}")
    widths = tuple(int(w) for w in widths)
    check_stack(R, V, contribs, widths, logits.device)
    if (not isinstance(labels, torch.Tensor) or labels.device != logits.device or labels.dtype != torch.int64
            or labels.dim() != 1 or int(labels.shape[0]) != R or not labels.is_contiguous()):
        raise RuntimeError("balance_ce_per_token: labels must be a contiguous int64 (rows,) tensor on the logits' "
                           "device")
    return R, V, widths


def _balance_ce_fwd(logits: torch.Tensor, labels: torch.Tensor, contribs: Sequence[torch.Tensor],
                    widths: Sequence[int]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """-> (loss (R,) fp32, rowmax (R,) fp32, logsum (R,) fp32)."""
    _check(logits, labels, contribs, widths)
    return stack_ce_fwd(logits, labels, contribs, widths, IGNORE_INDEX)


def _balance_ce_bwd(logits: torch.Tensor, labels: torch.Tensor, mx: torch.Tensor, ls: torch.Tensor, g: torch.Tensor,
                    contribs: Sequence[torch.Tensor], widths: Sequence[int], row_len: int) -> List[torch.Tensor]:
    """-> [the bf16 logits gradient (R, row_len) dense (columns V .. row_len - 1 zero), the width sums (R, w) bf16
    per width]."""
    R, V, widths = _check(logits, labels, contribs, widths)
    if g.dim() != 1 or int(g.shape[0]) != R or g.dtype != torch.float32 or g.device != logits.device:
        raise RuntimeError(f"balance_ce_per_token: the upstream gradient must be (rows,) float32, got {g.dtype} "
                           f"{tuple(g.shape)} {g.device}")
    for name, t in (("rowmax", mx), ("logsum", ls)):
        if t.dtype != torch.float32 or tuple(t.shape) != (R,) or not t.is_contiguous():
            raise RuntimeError(f"balance_ce_per_token: the saved {name} must be a contiguous ({R},) fp32 tensor")
    return stack_ce_bwd(logits, labels, mx, ls, g, contribs, widths, int(row_len), IGNORE_INDEX)


@torch.library.custom_op(_KERNEL_OP_NAMESPACE + "::balance_ce_fwd", mutates_args=())
def _balance_ce_fwd_op(logits: torch.Tensor, labels: torch.Tensor, contribs: List[torch.Tensor],
                       widths: List[int]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return _balance_ce_fwd(logits, labels, list(contribs), list(widths))


@_balance_ce_fwd_op.register_fake
def _balance_ce_fwd_fake(logits, labels, contribs, widths):
    R = logits.shape[0]
    return (logits.new_empty((R,), dtype=torch.float32), logits.new_empty((R,), dtype=torch.float32),
            logits.new_empty((R,), dtype=torch.float32))


@torch.library.custom_op(_KERNEL_OP_NAMESPACE + "::balance_ce_bwd", mutates_args=())
def _balance_ce_bwd_op(logits: torch.Tensor, labels: torch.Tensor, mx: torch.Tensor, ls: torch.Tensor,
                       g: torch.Tensor, contribs: List[torch.Tensor], widths: List[int],
                       row_len: int) -> List[torch.Tensor]:
    return _balance_ce_bwd(logits, labels, mx, ls, g, list(contribs), list(widths), row_len)


@_balance_ce_bwd_op.register_fake
def _balance_ce_bwd_fake(logits, labels, mx, ls, g, contribs, widths, row_len):
    R = logits.shape[0]
    return ([logits.new_empty((R, int(row_len)), dtype=torch.bfloat16)]
            + [logits.new_empty((R, int(w)), dtype=torch.bfloat16) for w in widths])


class BalanceCEFn(torch.autograd.Function):
    """Per-token cross entropy of the balanced logits ``wide[:, :cols]``; backward: the gradient of ``wide`` (dense
    bf16 with its row length, the columns from ``cols`` on zero) and every contribution's gradient (the width sums,
    (R, w) bf16)."""

    @staticmethod
    def forward(ctx, wide, labels, widths, cols, *contribs):
        ctx.set_materialize_grads(False)
        loss, mx, ls = _KERNEL_OPS.balance_ce_fwd(wide[:, :cols], labels, list(contribs), list(widths))
        ctx.widths = tuple(widths)
        ctx.cols = int(cols)
        ctx.save_for_backward(wide, labels, mx, ls, *contribs)
        return loss

    @staticmethod
    def backward(ctx, g):
        n = len(ctx.widths)
        if g is None:
            return (None, None, None, None) + (None,) * n
        wide, labels, mx, ls, *contribs = ctx.saved_tensors
        grad, *outs = _KERNEL_OPS.balance_ce_bwd(wide[:, :ctx.cols], labels, mx, ls, g, list(contribs),
                                                            list(ctx.widths), int(wide.shape[1]))
        return (grad if ctx.needs_input_grad[0] else None, None, None, None,
                *(o if ctx.needs_input_grad[4 + i] else None for i, o in enumerate(outs)))


def _padded_base(logits: torch.Tensor) -> Optional[torch.Tensor]:
    """The dense (R, P) tensor whose leading V columns ``logits`` is a view of, when the gradient can be returned for
    it (the padded output of the readout GEMM: same rows, same storage offset, row stride P, in the autograd graph);
    else None.  The backward writes any row length P >= V."""
    base = logits._base
    if base is None or not base.requires_grad or base.dim() != 2 or base.dtype != logits.dtype:
        return None
    R, V, P = int(logits.shape[0]), int(logits.shape[1]), int(base.shape[1])
    if (int(base.shape[0]) != R or P <= V or tuple(base.stride()) != (P, 1) or int(logits.stride(0)) != P
            or base.storage_offset() != logits.storage_offset()):
        return None
    return base


def balance_ce_per_token(logits: torch.Tensor, labels: torch.Tensor, contribs: Sequence[torch.Tensor],
                         widths: Sequence[int]) -> torch.Tensor:
    """Per-token training loss (R,) fp32 of the balanced logits.

    logits (R, V) bf16 (unit column stride, row stride >= V, V >= 1024; any R >= 1): the raw readout; labels (R,) int64
    contiguous, -100 ignored (+0 loss), every other value in [0, V); contribs: one (R, w) bf16 tensor per width (unit
    column stride, any row stride): the balance contributions; widths: one to four powers of two
    in [4, 512], in application order (the default (16, 32)).  Differentiable in the logits and the contributions.  When
    the logits are the leading columns of the readout GEMM's padded output, the gradient goes to that tensor directly
    (its pad columns zero), which is what the slice's backward would hand it."""
    widths = tuple(int(w) for w in widths)
    contribs = tuple(contribs)
    _, V, _ = _check(logits, labels, contribs, widths)
    wide = _padded_base(logits) if torch.is_grad_enabled() and logits.requires_grad else None
    return BalanceCEFn.apply(logits if wide is None else wide, labels, widths, V, *contribs)


def balance_ce_per_token_ref(logits: torch.Tensor, labels: torch.Tensor, contribs: Sequence[torch.Tensor],
                             widths: Sequence[int]) -> torch.Tensor:
    """``F.cross_entropy(balance_add_ref(logits).float(), labels, ignore_index=-100, reduction="none")``: the torch
    chain the fused kernels replace (differentiable through autograd).  Same arguments and output as
    :func:`balance_ce_per_token`."""
    with torch.autocast("cuda", enabled=False):
        x = balance_add_ref(logits, contribs, widths)
        return F.cross_entropy(x.float(), labels, ignore_index=IGNORE_INDEX, reduction="none")
