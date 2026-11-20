"""Bias-free linear op with an fp32 weight gradient (every projection of the model, the tied readout included).

Numerics contract (training, whole-row regime):

* forward: ``y = F.linear(x.to(bf16), weight.to(bf16))``: bf16 RNE operands, fp32 accumulation in cuBLAS, bf16
  output.  These are the operands bf16 autocast gives ``F.linear`` (an fp32 operand is cast with ``.to(bf16)``, a bf16
  operand passes through), i.e. the GEMM of the model's autocast forward; the two casts are made once and the same
  tensors are saved for backward.  The op does not depend on the autocast state.
* forward of a weight with a bf16 training copy (:func:`bind_train_bf16_`: an optimizer that writes the bf16 values of
  its weights with every update): the copy is the GEMM's weight operand, forward and backward, and no cast is made.
  The copy holds ``weight.to(bf16)`` (and zero pad rows, see below) whenever the operator runs: its owner keeps it so,
  and a weight modified in place after the copy was bound raises at the next training call until it is bound again.
* weight gradient: ``dW = torch.mm(dy2.t(), x2, out_dtype=torch.float32)`` with ``dy2`` / ``x2`` the row-major
  ``(rows, N)`` / ``(rows, K)`` bf16 views of ``dy`` / the saved input: one GEMM with bf16 operands, fp32 accumulation
  and an fp32 result.  It is returned as the weight's gradient (fp32 ``.grad`` through AccumulateGrad); nothing is
  rounded to bf16.  A weight used twice in one backward (the tied embedding: readout + lookup) receives the fp32
  sum of its contributions from autograd.
* weight gradient of a weight bound with :func:`bind_grad_` (an optimizer that owns the weight's ``.grad`` buffer):
  the same GEMM accumulates into ``weight.grad`` itself, ``grad = grad + dy2^T x2`` (``torch.addmm`` with ``out=`` the
  buffer: beta 1, no temporary and no separate add), and autograd receives no gradient for the weight from this
  operator.  What accumulates in the buffer is what AccumulateGrad would have added: several backwards accumulate,
  and another use of the weight in the same backward (the tied embedding's lookup) adds its contribution through
  autograd as before.  ``torch.autograd.grad`` with respect to a bound weight and tensor hooks on it do not see this
  operator's contribution, and every backward that runs through the operator writes the buffer, also one asked for
  other tensors only (``torch.autograd.grad`` with respect to an activation): bind only weights whose gradient is read
  from ``.grad`` in a step made of ``backward()`` calls.  The bound buffer must stay the weight's ``.grad``: a
  training call or a backward that finds another tensor there, or none, raises.
* input gradient: ``dx = dy2 @ weight_bf16``, one GEMM with bf16 operands that writes the result in ``x.dtype``: bf16
  for a bf16 input, fp32 for an fp32 input (``torch.mm(dy2, weight_bf16, out_dtype=torch.float32)``: the fp32
  accumulation stored as it is, no bf16 output widened by a second kernel; a GEMM with an fp32 result exists on CUDA
  only, elsewhere the bf16 product is widened).
* first-order gradients only: the GEMMs with an fp32 result have no derivative of their own.
* activation recomputation (``torch.utils.checkpoint``, non-reentrant): the second forward of a recomputed function
  hands the saved operands to the node of the first forward.  :func:`linear_operands` saves them without the product,
  for the function's last product (the backward reads a product's operands, not its output).
* an output-feature count N that is not a multiple of 8 (the odd vocabulary of the tied readout, 151,669) runs the three
  GEMMs on a padded layout: P = N rounded up to a multiple of 128 (151,680), the bf16 weight
  ``wp = [weight.to(bf16); 0]`` (P, K) with zero pad rows; forward ``F.linear(xb, wp)`` (..., P) whose pad columns are
  exactly 0, returned as the (..., N) view (row stride P); backward on the (rows, P) gradient whose pad columns are 0
  (the slice's backward writes them): ``dW = torch.mm(dyp[:, :N].t(), x2, out_dtype=float32)`` (the stride-P view, no
  copy) and ``dx = dyp @ wp`` (K = P).  The real columns are the same dot products of the same bf16 operands; the
  aligned extent keeps cuBLAS off the unaligned fallback kernel, which on an odd leading dimension runs split-K with
  bf16 partial sums (several bf16 roundings per output instead of one).  The weight gradient is one beta-0 GEMM over
  all rows.

Inference (no autograd needed): ``F.linear(x.to(bf16), w)`` where ``w`` is the bf16 copy made by :func:`cache_bf16_`
when one is present and current, else ``weight.to(bf16)``; the two hold the same values while the cache is current.  A
cache whose weight was modified after caching (a version bump), moved or re-typed raises: call
``prepare_inference()`` again.  The input's view and strides are kept (no copy), as in the autocast call.  A weight of
at least 16 output features whose count is not a multiple of 8 (the tied readout) is multiplied on the padded layout of
the training GEMMs: ``w`` holds ``padded_extent(N)`` rows, the pad rows zero, and the (..., N) view of the (..., P)
product is returned (row stride P; the real columns are the same dot products of the same bf16 operands, and the
aligned extent keeps cuBLAS off its unaligned kernel: 0.08 ms against 0.11 / 0.35 / 0.39 ms for the readout of 1 / 8 /
32 rows on a B200).  A product of
2 to 32 rows and fewer than 16 output features (the decode step's per-head projections) runs
:func:`neuronspark.ops.triton.linear_small_n.linear_small_n` instead of cuBLAS's split-K GEMM and reduction: one kernel,
fp32 accumulation, one bf16 rounding (the same reference).

Pure-torch reference :func:`linear_ref`: the same bf16-rounded operands multiplied in float64, the output rounded
once to bf16; its gradients are float64 products delivered in the inputs' dtypes.
"""
from __future__ import annotations

from typing import Optional, Sequence, Tuple

import torch
import torch.nn.functional as F

__all__ = [
    "Fp32WgradLinearFn",
    "LinearOperandsFn",
    "linear",
    "linear_operands",
    "linear_ref",
    "cache_bf16_",
    "drop_bf16_",
    "has_bf16_cache",
    "inference_weight",
    "stacked_inference_weight",
    "stacked_bands",
    "bind_grad_",
    "unbind_grad_",
    "bind_train_bf16_",
    "unbind_train_bf16_",
    "padded_extent",
    "inference_extent",
]

_CACHE_ATTR = "_ns_bf16"
_VERSION_ATTR = "_ns_bf16_version"
_GRAD_ATTR = "_ns_grad_buffer"
_TRAIN_ATTR = "_ns_train_bf16"
_TRAIN_VERSION_ATTR = "_ns_train_bf16_version"
# training GEMMs of an output-feature count that is not a multiple of ALIGN run on the extent rounded up to PAD_MULTIPLE
# (the tied readout's vocabulary: 151,669 -> 151,680)
ALIGN = 8
PAD_MULTIPLE = 128
# inference pads the same way from this many output features on (fewer: the weight as it is; see linear_small_n)
INFERENCE_PAD_MIN = 16


def padded_extent(n: int) -> int:
    """Output-feature extent of the training GEMMs: ``n`` when it is a multiple of 8, else ``n`` rounded up to a multiple
    of 128."""
    n = int(n)
    if n % ALIGN == 0:
        return n
    return (n + PAD_MULTIPLE - 1) // PAD_MULTIPLE * PAD_MULTIPLE


def inference_extent(n: int) -> int:
    """Rows of the bf16 inference weight of ``n`` output features: ``padded_extent(n)`` from ``INFERENCE_PAD_MIN``
    features on, else ``n``."""
    n = int(n)
    return padded_extent(n) if n >= INFERENCE_PAD_MIN else n


def _inference_bf16(weight: torch.Tensor) -> torch.Tensor:
    """The bf16 inference weight: ``weight.to(bf16)``, with zero pad rows up to :func:`inference_extent`."""
    n, k = int(weight.shape[0]), int(weight.shape[1])
    p = inference_extent(n)
    if p == n:
        return weight.to(torch.bfloat16)
    wb = torch.empty((p, k), dtype=torch.bfloat16, device=weight.device)
    wb[:n].copy_(weight)                             # RNE fp32 -> bf16: the bits of weight.to(bf16)
    wb[n:].zero_()
    return wb


# ------------------------------------------------------------------------------------------------ checks
def _check(x: torch.Tensor, weight: torch.Tensor) -> None:
    if weight.dim() != 2:
        raise RuntimeError(f"neuronspark.ops.linear: weight must be 2-D (N, K), got shape {tuple(weight.shape)}")
    if x.dim() < 1 or x.shape[-1] != weight.shape[1]:
        raise RuntimeError(f"neuronspark.ops.linear: x {tuple(x.shape)} does not match weight {tuple(weight.shape)}")
    if not x.is_floating_point() or x.dtype == torch.float64:
        raise RuntimeError(f"neuronspark.ops.linear: x must be fp32, bf16 or fp16, got {x.dtype}")
    if weight.dtype not in (torch.float32, torch.bfloat16):
        raise RuntimeError(f"neuronspark.ops.linear: weight must be fp32 (or bf16 in inference), got {weight.dtype}")
    if x.device != weight.device:
        raise RuntimeError(f"neuronspark.ops.linear: x on {x.device}, weight on {weight.device}")


def _cached_bf16(weight: torch.Tensor) -> Optional[torch.Tensor]:
    """The current bf16 cache of ``weight`` or None; raises on a stale cache."""
    w = getattr(weight, _CACHE_ATTR, None)
    if w is None:
        return None
    if getattr(weight, _VERSION_ATTR, None) != weight._version:
        raise RuntimeError("neuronspark.ops.linear: the bf16 inference copy of a weight is stale (the weight was "
                           "modified after prepare_inference()); call prepare_inference() again")
    rows = inference_extent(int(weight.shape[0]))
    if (w.device != weight.device or tuple(w.shape) != (rows, int(weight.shape[1]))
            or w.dtype != torch.bfloat16):
        raise RuntimeError("neuronspark.ops.linear: the bf16 inference copy of a weight no longer matches it "
                           f"(copy {tuple(w.shape)} {w.dtype} on {w.device}, weight {tuple(weight.shape)} on "
                           f"{weight.device}); call prepare_inference() again")
    return w


# ------------------------------------------------------------------------------------------------ bf16 cache
@torch.no_grad()
def cache_bf16_(param: torch.Tensor) -> torch.Tensor:
    """Attach (or refresh) the bf16 inference copy of ``param``: ``param.detach().to(bf16)``, contiguous (with zero pad
    rows up to :func:`inference_extent`), with the parameter's version recorded.  Returns the copy."""
    if param.dim() != 2 or param.dtype not in (torch.float32, torch.bfloat16):
        raise RuntimeError(f"cache_bf16_: expected a 2-D fp32/bf16 weight, got {tuple(param.shape)} {param.dtype}")
    w = _inference_bf16(param.detach()).contiguous()
    setattr(param, _CACHE_ATTR, w)
    setattr(param, _VERSION_ATTR, param._version)
    return w


def drop_bf16_(param: torch.Tensor) -> None:
    """Remove the bf16 inference copy of ``param`` (no-op without one)."""
    for name in (_CACHE_ATTR, _VERSION_ATTR):
        if hasattr(param, name):
            delattr(param, name)


def has_bf16_cache(param: torch.Tensor) -> bool:
    return getattr(param, _CACHE_ATTR, None) is not None


def stacked_inference_weight(weights: Sequence[torch.Tensor]) -> torch.Tensor:
    """The bf16 inference weight of several projections of one input as one (P, K) weight: the rows of ``weights``
    (each (N_i, K)) in order, each the RNE bf16 rounding its own inference product reads (its current bf16 copy when it
    has one: a stale copy raises, as in :func:`linear`), and zero rows up to a multiple of ``PAD_MULTIPLE``.  The
    product of an input with it holds every projection's result as a column band (:func:`stacked_bands`): one product
    in place of ``len(weights)``."""
    rows = [int(w.shape[0]) for w in weights]
    k = int(weights[0].shape[1])
    if any(w.dim() != 2 or int(w.shape[1]) != k or w.device != weights[0].device for w in weights):
        raise RuntimeError("neuronspark.ops.linear: stacked weights must be (N_i, K) tensors of one K on one device")
    n = sum(rows)
    stack = torch.zeros(((n + PAD_MULTIPLE - 1) // PAD_MULTIPLE * PAD_MULTIPLE, k), dtype=torch.bfloat16,
                        device=weights[0].device)
    r = 0
    for w, m in zip(weights, rows):
        stack[r:r + m].copy_(inference_weight(w).detach())     # the bf16 copy, or RNE fp32 -> bf16 (w.to(bf16))
        r += m
    return stack


def stacked_bands(out: torch.Tensor, weights: Sequence[torch.Tensor]) -> Tuple[torch.Tensor, ...]:
    """The results of the projections ``weights`` inside ``out`` (..., P), the product with their
    :func:`stacked_inference_weight`: views (..., N_i) with unit inner stride."""
    bands, r = [], 0
    for w in weights:
        m = int(w.shape[0])
        bands.append(out[..., r:r + m])
        r += m
    return tuple(bands)


def inference_weight(weight: torch.Tensor) -> torch.Tensor:
    """The operand an inference kernel reads for ``weight`` (N, K): the (N, K) view of its current bf16 copy (raises on
    a stale copy, as :func:`linear` does), or ``weight`` itself when it has none (the kernel then rounds it)."""
    w = _cached_bf16(weight)
    return weight if w is None else w[:int(weight.shape[0])]


# ------------------------------------------------------------------------------------------------ bound gradient
def bind_grad_(param: torch.Tensor) -> None:
    """Make the training backward accumulate the weight gradient of ``param`` into its present ``param.grad`` (see
    the module docstring).  ``param`` is a 2-D fp32 leaf whose ``.grad`` exists: a dense contiguous fp32 tensor of the
    weight's shape on its device, kept by the caller (an optimizer's gradient buffer).  That tensor is the bound
    buffer; it must stay the weight's ``.grad`` (zeroed in place, never replaced or removed) until
    :func:`unbind_grad_` or the next ``bind_grad_``."""
    if param.dim() != 2 or param.dtype != torch.float32 or not param.is_leaf:
        raise RuntimeError(f"bind_grad_: expected a 2-D fp32 leaf weight, got {tuple(param.shape)} {param.dtype}"
                           f"{'' if param.is_leaf else ' (not a leaf)'}")
    g = param.grad
    if g is None:
        raise RuntimeError("bind_grad_: the weight has no .grad to bind")
    if g.layout != torch.strided or g.dtype != torch.float32 or g.shape != param.shape or g.device != param.device \
            or not g.is_contiguous() or g.requires_grad:
        raise RuntimeError(f"bind_grad_: the .grad of a weight {tuple(param.shape)} on {param.device} must be a dense "
                           f"contiguous fp32 tensor of that shape on that device without a graph, got {g.layout} "
                           f"{g.dtype} {tuple(g.shape)} on {g.device}")
    setattr(param, _GRAD_ATTR, g)


def unbind_grad_(param: torch.Tensor) -> None:
    """Undo :func:`bind_grad_`: the weight gradient is returned to autograd again (no-op when not bound)."""
    if hasattr(param, _GRAD_ATTR):
        delattr(param, _GRAD_ATTR)


def _bound_grad(weight: torch.Tensor) -> Optional[torch.Tensor]:
    """The gradient buffer bound to ``weight`` (:func:`bind_grad_`) or None; raises when the weight's ``.grad`` is no
    longer that buffer."""
    g = getattr(weight, _GRAD_ATTR, None)
    if g is None:
        return None
    now = weight.grad
    if now is not g and (now is None or now.data_ptr() != g.data_ptr() or now.shape != g.shape
                         or now.dtype != g.dtype or not now.is_contiguous()):
        raise RuntimeError("neuronspark.ops.linear: the gradient buffer of a weight bound with bind_grad_ was "
                           f"{'removed' if now is None else 'replaced'} (its .grad is no longer the bound tensor); "
                           "bind it again or unbind it")
    return g


# ------------------------------------------------------------------------------------------------ bf16 training copy
def bind_train_bf16_(param: torch.Tensor, copy: torch.Tensor) -> None:
    """Attach the bf16 training copy of ``param`` that its owner maintains: a contiguous bf16 tensor of shape
    ``(padded_extent(N), K)`` on the weight's device whose first N rows hold ``param.to(bf16)`` and whose other rows
    are zero.  The training branch reads it in place of a cast (see the module docstring).  The parameter's version
    is recorded: after an in-place modification of the parameter the training branch raises until the copy is
    brought up to date and bound again (a write through ``.data`` or through another tensor on the weight's memory
    leaves the parameter's version as it is: the owner follows such a write itself).  The values are the caller's
    responsibility and are not checked here."""
    if param.dim() != 2 or param.dtype != torch.float32:
        raise RuntimeError(f"bind_train_bf16_: expected a 2-D fp32 weight, got {tuple(param.shape)} {param.dtype}")
    want = (padded_extent(param.shape[0]), int(param.shape[1]))
    if copy.dtype != torch.bfloat16 or tuple(copy.shape) != want or copy.device != param.device \
            or not copy.is_contiguous() or copy.requires_grad:
        raise RuntimeError(f"bind_train_bf16_: the copy of a weight {tuple(param.shape)} on {param.device} must be a "
                           f"contiguous bf16 tensor {want} on that device, got {copy.dtype} {tuple(copy.shape)} on "
                           f"{copy.device}")
    setattr(param, _TRAIN_ATTR, copy)
    setattr(param, _TRAIN_VERSION_ATTR, param._version)


def unbind_train_bf16_(param: torch.Tensor) -> None:
    """Remove the bf16 training copy of ``param`` (no-op without one): the training branch casts the weight again."""
    for name in (_TRAIN_ATTR, _TRAIN_VERSION_ATTR):
        if hasattr(param, name):
            delattr(param, name)


def _train_bf16(weight: torch.Tensor) -> Optional[torch.Tensor]:
    """The bf16 training copy of ``weight`` or None; raises when the weight was modified after the copy was bound."""
    w = getattr(weight, _TRAIN_ATTR, None)
    if w is None:
        return None
    if getattr(weight, _TRAIN_VERSION_ATTR, None) != weight._version:
        raise RuntimeError("neuronspark.ops.linear: the bf16 training copy of a weight is stale (the weight was "
                           "modified in place after bind_train_bf16_()); its owner must refresh the copy and bind it "
                           "again")
    return w


def _operands(x: torch.Tensor, weight: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """``(xb, wb)``: the bf16 operands of the training GEMMs, the two tensors the operator saves for its backward, in
    this order: the input in bf16 and the weight's ``(padded_extent(N), K)`` bf16 operand."""
    xb = x.to(torch.bfloat16)
    n, k = int(weight.shape[0]), int(weight.shape[1])
    p = padded_extent(n)
    wb = _train_bf16(weight)                         # the (p, k) copy the weight's owner maintains, or None
    if wb is None and p == n:
        wb = weight.to(torch.bfloat16)
    elif wb is None:
        wb = torch.empty((p, k), dtype=torch.bfloat16, device=weight.device)
        wb[:n].copy_(weight)                         # RNE fp32 -> bf16: the bits of weight.to(bf16)
        wb[n:].zero_()
    return xb, wb


# ------------------------------------------------------------------------------------------------ autograd Function
class Fp32WgradLinearFn(torch.autograd.Function):
    """``y = x @ W^T`` with bf16 operands, bf16 output, fp32 weight gradient, input gradient in ``x.dtype``.

    The output has ``padded_extent(N)`` features: for an N that is not a multiple of 8 the pad columns are exactly 0 and
    :func:`linear` returns the (..., N) view; the gradient then reaches backward as the (..., P) tensor of the slice's
    backward (pad columns 0).  The weight gradient of a bound weight (:func:`bind_grad_`) is accumulated into its
    ``.grad`` by the backward, which then returns none for it."""

    @staticmethod
    def forward(ctx, x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        xb, wb = _operands(x, weight)
        n = int(weight.shape[0])
        out = F.linear(xb, wb)
        if out.dtype != torch.bfloat16:
            raise RuntimeError(f"neuronspark.ops.linear: the GEMM produced {out.dtype}, expected bf16 (an fp16 "
                               "autocast region is not supported)")
        ctx.input_dtype = x.dtype
        ctx.input_shape = x.shape
        ctx.out_features = n
        # a bound weight is kept for the backward, which accumulates into the buffer bound to it then
        ctx.bound_weight = weight if weight.requires_grad and _bound_grad(weight) is not None else None
        # the weight operand is saved as it is, the owner's copy too (no copy is made of it): a write of it by a torch
        # operation between this call and the backward is refused by autograd as for any saved tensor
        ctx.save_for_backward(xb, wb)
        return out

    @staticmethod
    def backward(ctx, dy: torch.Tensor):
        xb, wb = ctx.saved_tensors
        n = ctx.out_features
        dx = dw = None
        with torch.autocast(xb.device.type, enabled=False):
            dy2 = dy.reshape(-1, dy.shape[-1]).contiguous()
            if dy2.dtype != torch.bfloat16:
                raise RuntimeError(f"neuronspark.ops.linear: incoming gradient is {dy2.dtype}, expected bf16")
            if int(dy2.shape[1]) != int(wb.shape[0]):
                raise RuntimeError(f"neuronspark.ops.linear: incoming gradient has {int(dy2.shape[1])} features, "
                                   f"expected {int(wb.shape[0])}")
            if ctx.needs_input_grad[1]:
                x2 = xb.reshape(-1, xb.shape[-1]).contiguous()
                g = None if ctx.bound_weight is None else _bound_grad(ctx.bound_weight)
                if g is None:
                    dw = torch.mm(dy2[:, :n].t(), x2, out_dtype=torch.float32)
                else:
                    torch.addmm(g, dy2[:, :n].t(), x2, out_dtype=torch.float32, out=g)
            if ctx.needs_input_grad[0]:
                if ctx.input_dtype == torch.float32 and dy2.is_cuda:
                    dx = torch.mm(dy2, wb, out_dtype=torch.float32).reshape(ctx.input_shape)
                else:
                    dx = torch.mm(dy2, wb).reshape(ctx.input_shape).to(ctx.input_dtype)
        return dx, dw


class LinearOperandsFn(torch.autograd.Function):
    """The tensors :class:`Fp32WgradLinearFn` saves for its backward, saved in the same order, without the product
    (:func:`linear_operands`).  The output is an empty tensor; the node has no backward."""

    @staticmethod
    def forward(ctx, x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        xb, wb = _operands(x, weight)
        ctx.save_for_backward(xb, wb)
        return xb.new_empty(0)

    @staticmethod
    def backward(ctx, dy: torch.Tensor):
        raise RuntimeError("neuronspark.ops.linear: linear_operands() forms no product and has no backward (it stands "
                           "for linear() in the second forward of an activation recomputation only)")


# ------------------------------------------------------------------------------------------------ public functions
def linear_operands(x: torch.Tensor, weight: torch.Tensor) -> None:
    """``linear(x, weight)`` in the second forward of an activation recomputation (``torch.utils.checkpoint`` with
    ``use_reentrant=False``), for a product that no tensor saved by the recomputed function depends on (its last
    operator, whose output only leaves the function): the backward node of the first forward reads the two operands,
    which this call saves as :func:`linear` does (the same tensors in the same order; none when :func:`linear` would
    run without autograd), and the GEMM is not run.  Returns nothing."""
    _check(x, weight)
    if torch.is_grad_enabled() and (x.requires_grad or weight.requires_grad):
        if weight.dtype != torch.float32:
            raise RuntimeError(f"neuronspark.ops.linear: a trained weight must be fp32, got {weight.dtype}")
        LinearOperandsFn.apply(x, weight)


def linear(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """``x (..., K)`` fp32 / bf16, ``weight (N, K)`` fp32 -> ``(..., N)`` bf16.

    With autograd (grad mode on and ``x`` or ``weight`` requiring grad) the :class:`Fp32WgradLinearFn` path runs and the
    weight must be fp32; for an N that is not a multiple of 8 the result is the (..., N) view of the padded (..., P)
    output (row stride P, see the module docstring); a weight bound with :func:`bind_grad_` has its gradient
    accumulated into ``weight.grad`` by the backward.  Otherwise the inference path runs: the current bf16 cache of the
    weight (see :func:`cache_bf16_`) or ``weight.to(bf16)``.
    """
    _check(x, weight)
    if torch.is_grad_enabled() and (x.requires_grad or weight.requires_grad):
        if weight.dtype != torch.float32:
            raise RuntimeError(f"neuronspark.ops.linear: a trained weight must be fp32, got {weight.dtype}")
        n = int(weight.shape[0])
        out = Fp32WgradLinearFn.apply(x, weight)
        # the slice is taken outside the Function: an in-place update of the (..., N) view (the readout balance) is
        # then an ordinary view update for autograd, and the slice's backward hands the Function the (..., P)
        # gradient with zero pad columns
        return out if int(out.shape[-1]) == n else out[..., :n]
    n = int(weight.shape[0])
    w = _cached_bf16(weight)
    if w is None:
        w = _inference_bf16(weight)
    xb = x.to(torch.bfloat16)
    if n < INFERENCE_PAD_MIN and xb.is_cuda and xb.dim() >= 2:
        from neuronspark.ops.triton.linear_small_n import linear_small_n, takes
        x2 = xb.reshape(-1, int(xb.shape[-1]))
        if takes(x2, w):
            return linear_small_n(x2, w).view(*xb.shape[:-1], n)
    out = F.linear(xb, w)
    return out if int(out.shape[-1]) == n else out[..., :n]


def linear_ref(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """Pure-torch reference of :func:`linear`: the bf16-rounded operands multiplied in float64, the output rounded to
    bf16 once.  Differentiable: the gradients are float64 products of the bf16-rounded operands (straight-through
    rounding), delivered as ``x.dtype`` / ``weight.dtype`` by autograd."""
    _check(x, weight)
    x64 = x.double()
    w64 = weight.double()
    xr = x64 + (x.detach().to(torch.bfloat16).double() - x64.detach())
    wr = w64 + (weight.detach().to(torch.bfloat16).double() - w64.detach())
    return torch.matmul(xr, wr.t()).to(torch.bfloat16)
