"""ops.triton.copy_stats: the session copy-detector statistics of a prefill call.

``copy_stats_prefill`` (one kernel: online max / logsumexp over ring tiles) against its torch chain
``copy_stats_prefill_ref`` under ``regime("infer")`` (IEEE fp32 scores), on rings of the 1.27B shape (B in {1, 8},
W = 2048, d = 96 fp32 unit rows, zero vectors before a row's first event) and calls of T tokens at several ring
pointers (wrapped and not), with three row layouts: every row joins at this call (``fresh``), rows that joined at this
call / at clock 0 / in between (``mixed``), and every row from clock 0 (``origin``).  About a third of the queries are
copy events (a strictly-past row plus noise), so the confidence covers (0, 1).

Compared by the per-operator rule of ``neuronspark.utils.testing`` (matched ratio 1.0): the finite maxima and
log-sum-exps (the -inf entries, queries without a valid slot, must coincide), and the confidence ``pi`` the memory
layers form from them at tau in {0.3, 0.6, 0.9}.  The module path (``copy_stats_session``) is checked with the kernel
namespace against the reference namespace over a chunked session.
"""
import pytest
import torch
import torch.nn.functional as F

from neuronspark.modules.copy_detector import CopyStats, copy_pi, copy_stats_session
from neuronspark.ops import REFS
from neuronspark.ops.triton.copy_stats import copy_stats_prefill, copy_stats_prefill_ref
from neuronspark.utils.cache import CopyState
from neuronspark.utils.precision import regime
from neuronspark.utils.testing import assert_all_match, assert_match

pytestmark = pytest.mark.cuda

W, D = 2048, 96
# (B, T, ring pointer at the call's start, row layout)
CASES = [
    (1, 2, 2048, "fresh"), (1, 16, 2053, "mixed"), (8, 128, 2048, "fresh"), (8, 128, 3000, "mixed"),
    (1, 1024, 2948, "mixed"), (8, 1024, 2048, "fresh"), (8, 1024, 2500, "mixed"), (1, 2048, 2048, "fresh"),
    (8, 2048, 2048, "fresh"), (8, 2048, 4100, "mixed"), (8, 2048, 0, "origin"), (8, 128, 3000, "origin"),
    (8, 1024, 2500, "origin"),
]
SEEDS = (0, 1)
TAUS = (0.3, 0.6, 0.9)


def _ring_case(B: int, T: int, p0: int, layout: str, seed: int, device):
    """A ring after the ring write of a call of T tokens that started at pointer p0: unit rows at every slot holding a
    clock at or after the row's first event (zero vectors before it); about a third of the call's queries copy a
    strictly-past row plus noise.  Returns (ring, ring_ptr, pos) with pos the tokens each row consumed before the
    call."""
    g = torch.Generator(device="cpu").manual_seed(int(seed))
    ring = torch.zeros((B, W, D), dtype=torch.float32)
    end1 = p0 + T - 1
    w = torch.arange(W)
    clock = end1 - torch.remainder(end1 - w, W)
    lo = max(0, end1 - W + 1)
    starts = []
    for b in range(B):
        if layout == "fresh":
            st = p0
        elif layout == "mixed":
            st = [p0, 0, int(torch.randint(lo, p0 + 1, (), generator=g))][b % 3]
        else:
            st = 0
        starts.append(st)
        live = clock >= st
        ring[b, live] = F.normalize(torch.randn((int(live.sum()), D), generator=g), dim=-1)
    for tau_i in range(T):
        t = p0 + tau_i
        if float(torch.rand((), generator=g)) > 0.35:
            continue
        eps = (0.02, 0.1, 0.3, 0.6)[int(torch.randint(0, 4, (), generator=g))]
        for b in range(B):
            lob = max(lo, starts[b])
            if t - 2 < lob:
                continue
            s = int(torch.randint(lob, t - 1, (), generator=g))
            v = ring[b, s % W] + eps * torch.randn(D, generator=g) / D ** 0.5
            ring[b, t % W] = F.normalize(v, dim=-1)
    pos = torch.tensor([p0 - st for st in starts], dtype=torch.int64, device=device)
    ring_ptr = torch.tensor([p0], dtype=torch.int64, device=device)
    return ring.to(device), ring_ptr, pos


def _finite_rule(new: torch.Tensor, ref: torch.Tensor, name: str) -> None:
    """The -inf entries (no valid slot) coincide; the rule on the finite entries."""
    assert torch.equal(torch.isneginf(new), torch.isneginf(ref)), f"{name}: the queries without a valid slot differ"
    assert not bool(torch.isnan(new).any()) and not bool(torch.isposinf(new).any()), f"{name}: nan / +inf"
    fin = torch.isfinite(ref)
    assert_match(new[fin], ref[fin], name)


@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("B,T,p0,layout", CASES, ids=[f"b{c[0]}_t{c[1]}_p{c[2]}_{c[3]}" for c in CASES])
def test_copy_stats_prefill_vs_reference(device, B, T, p0, layout, seed):
    ring, ring_ptr, pos = _ring_case(B, T, p0, layout, 1000 * seed + T + p0, device)
    with torch.no_grad(), regime("infer"):
        new_max, new_lse = copy_stats_prefill(ring, ring_ptr, pos, T)
        ref_max, ref_lse = copy_stats_prefill_ref(ring, ring_ptr, pos, T)
        for t in (new_max, new_lse):
            assert tuple(t.shape) == (T, B) and t.dtype == torch.float32 and t.is_contiguous()
        _finite_rule(new_max, ref_max, "maximum")
        _finite_rule(new_lse, ref_lse, "lse")
        new_pi, ref_pi = {}, {}
        for i, tau in enumerate(TAUS):
            new_pi[f"pi{i}"] = copy_pi(CopyStats(new_max, new_lse), tau)
            ref_pi[f"pi{i}"] = copy_pi(CopyStats(ref_max, ref_lse), tau)
    assert_all_match(new_pi, ref_pi, prefix="copy_pi")


def test_copy_stats_session_kernel_vs_ref_namespace(device):
    """``copy_stats_session`` over a chunked session (prefill chunks and decode steps, a wrapped ring): the kernel
    namespace (default) against the reference namespace."""
    B, Wr, M = 3, 512, 160
    chunks = (37, 1, 200, 1, 1, 300, 64, 1)
    g = torch.Generator(device=device).manual_seed(5)
    table = torch.randn((12, M), generator=g, device=device)
    ids = torch.randint(0, 12, (sum(chunks), B), generator=g, device=device)
    lex = table[ids] + 0.3 * torch.randn((sum(chunks), B, M), generator=g, device=device)
    states = [CopyState.allocate(B, Wr, D, device) for _ in range(2)]
    ptrs = [torch.zeros((1,), dtype=torch.int64, device=device) for _ in range(2)]
    poss = [torch.zeros((B,), dtype=torch.int64, device=device) for _ in range(2)]
    new, ref, off = {}, {}, 0
    with torch.no_grad(), regime("infer"):
        for i, T in enumerate(chunks):
            a = copy_stats_session(lex[off:off + T], states[0], ptrs[0], poss[0], Wr)
            b = copy_stats_session(lex[off:off + T], states[1], ptrs[1], poss[1], Wr, ops=REFS)
            new[f"pi{i}"] = copy_pi(a, 0.6)
            ref[f"pi{i}"] = copy_pi(b, 0.6)
            for k in range(2):
                ptrs[k] += T
                poss[k] += T
            off += T
    assert_all_match(new, ref, prefix="copy_session")
