"""Readout balance + cross entropy of the training loss for a balance stack of one to four widths.

The training loss of the readout is the per-token cross entropy of the balanced logits: per width ``w`` of the stack,
in order, ``logits[r, v] = bf16(fp32(logits[r, v]) + fp32(c_w[r, v % w]))`` (one rounding per width), then
``F.cross_entropy(balanced.float(), labels, ignore_index, reduction="none")``.  The operator ``balance_ce_per_token``
of :mod:`neuronspark.ops.triton.lm_tail` runs the two kernels here for every stack: the default configuration's
(16, 32) and the scales whose readout balance has more projections, e.g. (4, 8, 32) or (4, 8, 128, 256).  The
balanced logits are not materialised.

* forward (``_stack_fwd_kernel``, one program per row, any number of rows): every balanced value is formed on the fly (the fp32 add and
  the RNE to bf16 per width, in order), then a per-lane-group online softmax over tiles of ``FWD_TILE_GROUPS`` groups
  of ``FWD_TILE_COLS`` consecutive columns, each group's running sum with compensated summation (a term is
  ``2^((x - group max) log2 e)``: the difference is formed first, so a group of -inf columns adds nothing).
  Outputs: the row max, ``logsum = log(sum 2^((x - max) log2 e))`` and the
  per-token loss ``-((x_t - max) - logsum)`` (+0 for an ignored row; an out-of-range target traps, as the
  reference's device assert).
* backward (``_stack_bwd_kernel``, one program per row, any number of rows): each balanced value again, ``logp = (x - max) - logsum``,
  ``gI = gO - exp(logp) * sum(gO)`` with ``gO`` the nll gradient (``-g`` at the target), rounded to bf16 into a new
  dense gradient of row length ``P >= V`` (the columns from V on are zeros); and the gradient of every contribution,
  the sum over the columns ``v`` with ``v % w == c`` of that bf16 gradient: the row is accumulated in fp64 per lane
  of a ``BWD_TILE``-column tile (every width divides the tile), the lanes are folded to each width and rounded to
  bf16 once.  (fp64: a width sum holds the target's term and the largest column's term, of opposite signs and up to
  the upstream gradient in size, beside many small ones; what is left after they cancel is the sum of the small
  terms, which an fp32 running sum beside either large term drops.)

Numerics: the balanced values and the row max follow the reference chain's rounding points.  The softmax
denominator, the logarithm and the exponential are this kernel's (ex2.approx, Triton's log and exp); the width sums
are the reference's sums in another order, with one bf16 rounding where the reference rounds the head sum and adds
the tail.  The reference is ``balance_ce_per_token_ref`` of :mod:`neuronspark.ops.triton.lm_tail` (the torch chain);
tests: the per-operator tolerance rule of :mod:`neuronspark.utils.testing`.

Contract (anything else raises): 1 to 4 widths, each a power of two with ``4 <= w <= BWD_TILE``; the logits a CUDA
bf16 (R, V) tensor with unit column stride and at least one forward tile of columns
(``V >= FWD_TILE_GROUPS * FWD_TILE_COLS``: every lane group of the online softmax then holds a column of the first
tile); every contribution a CUDA bf16 (R, w) tensor with unit column stride and
any row stride (the output of a projection whose GEMM runs on a padded extent is a strided view); labels int64 (R,)
contiguous.  The launch configurations are the fixed constants below.
"""
from typing import List, Sequence, Tuple

import torch
import triton
import triton.language as tl

from neuronspark.ops.triton._ptx import _ex2ftz as _sc_ex2  # noqa: F401  (resolved through this module's globals)

__all__ = ["STACK_MAX_WIDTHS", "FWD_TILE_GROUPS", "FWD_TILE_COLS", "FWD_WARPS", "BWD_TILE", "BWD_WARPS", "check_stack",
           "stack_ce_fwd", "stack_ce_bwd"]

LOG2E = tl.constexpr(1.4426950408889634)

# ================================================================================================ fixed launch constants
#: the most widths a stack has
STACK_MAX_WIDTHS = 4
#: forward: tiles of FWD_TILE_GROUPS lane groups x FWD_TILE_COLS consecutive columns, FWD_WARPS warps per row
FWD_TILE_GROUPS = 128
FWD_TILE_COLS = 8
FWD_WARPS = 4
#: backward: columns per tile (= the largest width), warps per row
BWD_TILE = 512
BWD_WARPS = 8


# ================================================================================================ kernels
@triton.jit
def _sc_trap_if(flag):
    """Device-side abort (the reference's device assert on an out-of-range target) when flag != 0."""
    return tl.inline_asm_elementwise(asm="{ .reg .pred p; setp.ne.s32 p, $1, 0; @p trap; mov.b32 $0, 0; }",
                                     constraints="=r,r", args=[flag], dtype=tl.int32, is_pure=False, pack=1)


@triton.jit
def _sc_bal(x, c0, c1, c2, c3, NW: tl.constexpr):
    """The balance chain per width: fp32 add, RNE to bf16, back to fp32."""
    x = (x + c0).to(tl.bfloat16, fp_downcast_rounding="rtne").to(tl.float32)
    if NW > 1:
        x = (x + c1).to(tl.bfloat16, fp_downcast_rounding="rtne").to(tl.float32)
    if NW > 2:
        x = (x + c2).to(tl.bfloat16, fp_downcast_rounding="rtne").to(tl.float32)
    if NW > 3:
        x = (x + c3).to(tl.bfloat16, fp_downcast_rounding="rtne").to(tl.float32)
    return x


@triton.jit
def _sc_acc(s, e, f, t):
    """One tile into a lane group's running sum: ``s <- s f + t`` with the rounding error of the addition carried in
    ``e`` (compensated summation), so a tile's terms that are below half an ulp of the sum are not dropped one tile
    after the other (a row whose largest term shares its lane group with many small ones)."""
    s = s * f
    y = t - e * f
    u = s + y
    return u, (u - s) - y


@triton.jit
def _stack_fwd_kernel(X, C0, C1, C2, C3, T, MX, LS, LOSS, SX, S0, S1, S2, S3, IGNORE,
                      V: tl.constexpr, W0: tl.constexpr, W1: tl.constexpr, W2: tl.constexpr, W3: tl.constexpr,
                      NW: tl.constexpr, G: tl.constexpr, KV: tl.constexpr):
    """One row per program: balanced values on the fly, per-lane-group online softmax, row max / logsum / loss."""
    r = tl.program_id(0).to(tl.int64)
    xr = X + r * SX
    BLOCK: tl.constexpr = G * KV
    offs = tl.arange(0, G)[:, None] * KV + tl.arange(0, KV)[None, :]          # (G, KV) column offsets in a tile
    c0 = tl.load(C0 + r * S0 + (offs % W0)).to(tl.float32)                    # tile starts are multiples of BLOCK
    if NW > 1:
        c1 = tl.load(C1 + r * S1 + (offs % W1)).to(tl.float32)
    else:
        c1 = c0
    if NW > 2:
        c2 = tl.load(C2 + r * S2 + (offs % W2)).to(tl.float32)
    else:
        c2 = c0
    if NW > 3:
        c3 = tl.load(C3 + r * S3 + (offs % W3)).to(tl.float32)
    else:
        c3 = c0
    NEG_MAX: tl.constexpr = -3.4028234663852886e38
    m = tl.full([G], NEG_MAX, tl.float32)
    s = tl.zeros([G], tl.float32)
    e = tl.zeros([G], tl.float32)                                             # the running sums' rounding errors
    NFULL: tl.constexpr = V // BLOCK
    for i in range(0, NFULL):
        x = _sc_bal(tl.load(xr + i * BLOCK + offs).to(tl.float32), c0, c1, c2, c3, NW)
        mn = tl.maximum(m, tl.max(x, axis=1))
        s, e = _sc_acc(s, e, _sc_ex2((m - mn) * LOG2E), tl.sum(_sc_ex2((x - mn[:, None]) * LOG2E), axis=1))
        m = mn
    if V % BLOCK != 0:
        cols = NFULL * BLOCK + offs
        ok = cols < V
        x = _sc_bal(tl.load(xr + cols, mask=ok, other=0.0).to(tl.float32), c0, c1, c2, c3, NW)
        x = tl.where(ok, x, float("-inf"))
        mn = tl.maximum(m, tl.max(x, axis=1))
        s, e = _sc_acc(s, e, _sc_ex2((m - mn) * LOG2E), tl.sum(_sc_ex2((x - mn[:, None]) * LOG2E), axis=1))
        m = mn
    s = s - e
    mx = tl.max(m, axis=0, keep_dims=True)                                    # (1,): the row max
    ssum = tl.sum(s * _sc_ex2((m - mx) * LOG2E), axis=0, keep_dims=True)
    ls = tl.log(ssum)
    # the row's nll term on the balanced value of the target column
    one = tl.arange(0, 1)
    tg = tl.load(T + r + one)                                                 # (1,) int64
    valid = tg != IGNORE
    inb = (tg >= 0) & (tg < V)
    _sc_trap_if((valid & (inb == 0)).to(tl.int32))
    tgc = tl.where(valid & inb, tg, 0)
    xt = tl.load(xr + tgc).to(tl.float32)
    ct0 = tl.load(C0 + r * S0 + (tgc % W0)).to(tl.float32)
    if NW > 1:
        ct1 = tl.load(C1 + r * S1 + (tgc % W1)).to(tl.float32)
    else:
        ct1 = ct0
    if NW > 2:
        ct2 = tl.load(C2 + r * S2 + (tgc % W2)).to(tl.float32)
    else:
        ct2 = ct0
    if NW > 3:
        ct3 = tl.load(C3 + r * S3 + (tgc % W3)).to(tl.float32)
    else:
        ct3 = ct0
    xt = _sc_bal(xt, ct0, ct1, ct2, ct3, NW)
    loss = tl.where(valid, ls - (xt - mx), 0.0)
    tl.store(MX + r + one, mx)
    tl.store(LS + r + one, ls)
    tl.store(LOSS + r + one, loss)


@triton.jit
def _sc_grad(x, c0, c1, c2, c3, cols, tg32, mx, ls, v, NW: tl.constexpr):
    """raw bf16 logits x -> the fp32 gradient before its bf16 rounding: gO - exp(logp) * sum(gO)."""
    xf = _sc_bal(x.to(tl.float32), c0, c1, c2, c3, NW)
    e = tl.exp((xf - mx) - ls)
    return tl.where(cols == tg32, v, 0.0) - e * v


@triton.jit
def _sc_width_store(D, r, acc, W: tl.constexpr, TILE: tl.constexpr):
    """One width's gradient row: the tile's fp64 lanes folded to W columns, rounded to bf16."""
    out = tl.sum(tl.reshape(acc, (TILE // W, W)), axis=0).to(tl.float32)
    tl.store(D + r * W + tl.arange(0, W), out.to(tl.bfloat16, fp_downcast_rounding="rtne"))


@triton.jit
def _stack_bwd_kernel(X, C0, C1, C2, C3, T, MX, LS, G, DX, D0, D1, D2, D3, SX, SD, SG, S0, S1, S2, S3, IGNORE,
                      V: tl.constexpr, P: tl.constexpr, W0: tl.constexpr, W1: tl.constexpr, W2: tl.constexpr,
                      W3: tl.constexpr, NW: tl.constexpr, TILE: tl.constexpr):
    """One row per program: the bf16 logits gradient (row length P, the columns from V on zero) and every width's
    sum of it."""
    r = tl.program_id(0).to(tl.int64)
    rx = r * SX
    rd = r * SD
    ar = tl.arange(0, TILE)
    c0 = tl.load(C0 + r * S0 + (ar % W0)).to(tl.float32)                      # tile starts are multiples of TILE
    if NW > 1:
        c1 = tl.load(C1 + r * S1 + (ar % W1)).to(tl.float32)
    else:
        c1 = c0
    if NW > 2:
        c2 = tl.load(C2 + r * S2 + (ar % W2)).to(tl.float32)
    else:
        c2 = c0
    if NW > 3:
        c3 = tl.load(C3 + r * S3 + (ar % W3)).to(tl.float32)
    else:
        c3 = c0
    z = tl.zeros([TILE], tl.int32)
    mx = tl.load(MX + r + z)
    ls = tl.load(LS + r + z)
    tg = tl.load(T + r + z)
    g = tl.load(G + r * SG + z)
    valid = tg != IGNORE
    v = tl.where(valid, -g, 0.0)                                              # gO at the target = the row sum of gO
    tg32 = tl.where(valid, tg, -1).to(tl.int32)
    acc = tl.zeros([TILE], tl.float64)                                        # the lanes' sums of the bf16 gradient
    NFULL: tl.constexpr = V // TILE                                           # tiles with every column below V
    NT: tl.constexpr = (P + TILE - 1) // TILE
    for i in range(0, NFULL):
        cols = i * TILE + ar
        x = tl.load(X + rx + cols)
        gb = _sc_grad(x, c0, c1, c2, c3, cols, tg32, mx, ls, v, NW).to(tl.bfloat16, fp_downcast_rounding="rtne")
        tl.store(DX + rd + cols, gb)
        acc += gb.to(tl.float32).to(tl.float64)
    for i in tl.static_range(NFULL, NT):
        cols = i * TILE + ar
        ok = cols < V
        x = tl.load(X + rx + cols, mask=ok, other=0.0)
        gi = tl.where(ok, _sc_grad(x, c0, c1, c2, c3, cols, tg32, mx, ls, v, NW), 0.0)
        gb = gi.to(tl.bfloat16, fp_downcast_rounding="rtne")
        tl.store(DX + rd + cols, gb, mask=cols < P)
        acc += gb.to(tl.float32).to(tl.float64)
    _sc_width_store(D0, r, acc, W0, TILE)
    if NW > 1:
        _sc_width_store(D1, r, acc, W1, TILE)
    if NW > 2:
        _sc_width_store(D2, r, acc, W2, TILE)
    if NW > 3:
        _sc_width_store(D3, r, acc, W3, TILE)


# ================================================================================================ host side
def check_stack(R: int, V: int, contribs: Sequence[torch.Tensor], widths: Sequence[int], device) -> None:
    """Raise unless ``widths`` / ``contribs`` are a stack the kernels here take for (R, V) logits on ``device``."""
    widths = tuple(int(w) for w in widths)
    if not 1 <= len(widths) <= STACK_MAX_WIDTHS or len(contribs) != len(widths):
        raise RuntimeError(f"balance_ce_per_token: a stack of 1 to {STACK_MAX_WIDTHS} widths with one contribution "
                           f"each is supported, got widths {widths}, {len(contribs)} contributions")
    if V < FWD_TILE_GROUPS * FWD_TILE_COLS:
        raise RuntimeError(f"balance_ce_per_token: at least {FWD_TILE_GROUPS * FWD_TILE_COLS} columns (one forward "
                           f"tile) are needed, got V={V}")
    for w, t in zip(widths, contribs):
        if w < 4 or w & (w - 1) or w > BWD_TILE:
            raise RuntimeError(f"balance_ce_per_token: width {w} of the stack {widths} must be a power of two with "
                               f"4 <= w <= {BWD_TILE}")
        if (not isinstance(t, torch.Tensor) or t.dtype != torch.bfloat16 or t.device != device
                or tuple(t.shape) != (R, w) or t.stride(1) != 1 or (R > 1 and t.stride(0) < w)):
            raise RuntimeError(f"balance_ce_per_token: the contribution of width {w} must be a CUDA bf16 ({R}, {w}) "
                               f"tensor with unit column stride, got {getattr(t, 'dtype', None)} "
                               f"{tuple(getattr(t, 'shape', ()))} strides {tuple(t.stride()) if hasattr(t, 'stride') else ()}")


def _operands(contribs: Sequence[torch.Tensor], widths: Sequence[int]) -> Tuple[list, list, list]:
    """(tensors, widths, row strides) of the four kernel operands (an unused fourth aliases the first)."""
    n = len(widths)
    c = [contribs[i] if i < n else contribs[0] for i in range(STACK_MAX_WIDTHS)]
    w = [int(widths[i]) if i < n else int(widths[0]) for i in range(STACK_MAX_WIDTHS)]
    s = [int(t.stride(0)) for t in c]
    return c, w, s


def stack_ce_fwd(logits: torch.Tensor, labels: torch.Tensor, contribs: Sequence[torch.Tensor],
                 widths: Sequence[int], ignore_index: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """-> (loss (R,) fp32, rowmax (R,) fp32, logsum (R,) fp32) of the (R, V) bf16 logits (unit column stride)."""
    R, V = int(logits.shape[0]), int(logits.shape[1])
    check_stack(R, V, contribs, widths, logits.device)
    dev = logits.device
    loss = torch.empty(R, dtype=torch.float32, device=dev)
    mx = torch.empty(R, dtype=torch.float32, device=dev)
    ls = torch.empty(R, dtype=torch.float32, device=dev)
    c, w, s = _operands(contribs, widths)
    _stack_fwd_kernel[(R,)](logits, c[0], c[1], c[2], c[3], labels, mx, ls, loss, int(logits.stride(0)),
                            s[0], s[1], s[2], s[3], int(ignore_index), V=V, W0=w[0], W1=w[1], W2=w[2], W3=w[3],
                            NW=len(widths), G=FWD_TILE_GROUPS, KV=FWD_TILE_COLS, num_warps=FWD_WARPS)
    return loss, mx, ls


def stack_ce_bwd(logits: torch.Tensor, labels: torch.Tensor, mx: torch.Tensor, ls: torch.Tensor, g: torch.Tensor,
                 contribs: Sequence[torch.Tensor], widths: Sequence[int], row_len: int,
                 ignore_index: int) -> List[torch.Tensor]:
    """-> [the bf16 logits gradient (R, row_len) dense (the columns from V on zero), the width sums (R, w) bf16 dense
    per width]."""
    R, V, P = int(logits.shape[0]), int(logits.shape[1]), int(row_len)
    check_stack(R, V, contribs, widths, logits.device)
    if P < V:
        raise RuntimeError(f"balance_ce_per_token: a gradient row length of {P} for {V} columns")
    dev = logits.device
    grad = torch.empty((R, P), dtype=torch.bfloat16, device=dev)
    outs = [torch.empty((R, int(w)), dtype=torch.bfloat16, device=dev) for w in widths]
    c, w, s = _operands(contribs, widths)
    d = [outs[i] if i < len(outs) else outs[0] for i in range(STACK_MAX_WIDTHS)]
    _stack_bwd_kernel[(R,)](logits, c[0], c[1], c[2], c[3], labels, mx, ls, g, grad, d[0], d[1], d[2], d[3],
                            int(logits.stride(0)), P, int(g.stride(0)), s[0], s[1], s[2], s[3], int(ignore_index),
                            V=V, P=P, W0=w[0], W1=w[1], W2=w[2], W3=w[3], NW=len(widths), TILE=BWD_TILE,
                            num_warps=BWD_WARPS)
    return [grad] + outs
