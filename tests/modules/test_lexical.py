"""LexicalMemory: the model-level hashed n-gram tables.

* the hash constants (the seed's ``random.Random`` recomputation against the documented constants of the trained
  model, and the post-load check);
* the buckets against the Python-integer reference of the uint64 hash, through the lexical vector they select;
* the session form (chunks from a carried window) against the whole row at the same positions;
* the gradient of the order-1 table against a dense ``index_add`` reference;
* the layout (keys, shapes, element count, dtypes).

The bucket indices are compared through the vectors they select (a wrong bucket reads another random table row, so
the vector differs at O(1)); every comparison of values is the per-operator rule of ``neuronspark.utils.testing``.
"""
from __future__ import annotations

import pytest
import torch

from neuronspark.models.config_neuronspark import NeuronSparkConfig
from neuronspark.modules.lexical_memory import LexicalMemory, lexical_hash_reference
from neuronspark.utils.testing import assert_match

pytestmark = pytest.mark.cuda

CFG = NeuronSparkConfig()
V = CFG.vocab_size
# hash constants of the trained model (lexical_memory_hash_seed 12345): order 2 heads 0 / 1, order 3 heads 0 / 1
HASH_MULT = [6757828935835136783, 7404969417227990939, 8594196886349444303, 4023496952251142795]
HASH_ADD = [3782388285278155333, 3943629549311600962, 1246284344877894228, 1720453321244041099]
HASH_WRAP = [128000, 128000, 170027, 170027]
CHUNKS = (5, 1, 1, 37, 1, 211, 1)
TORCH_SEED = 4321


def _ids(B: int, T: int, device, vocab: int = V) -> torch.Tensor:
    """Token ids with repeats (half the positions drawn from a small id set) so that n-grams recur."""
    g = torch.Generator(device=device).manual_seed(77)
    wide = torch.randint(0, vocab, (B, T), generator=g, device=device)
    narrow = torch.randint(0, 16, (B, T), generator=g, device=device)
    pick = torch.rand((B, T), generator=g, device=device) < 0.5
    return torch.where(pick, narrow, wide)


def _built(device) -> LexicalMemory:
    torch.manual_seed(TORCH_SEED)
    return LexicalMemory(CFG, device=device)


def test_layout(device):
    lm = LexicalMemory(CFG, device=device, init="empty")
    assert [tuple(t.shape) for t in lm.tables] == [(151669, 96), (262139, 16), (262139, 16), (344951, 16),
                                                  (344951, 16)]
    assert lm.M == CFG.lexical_dim == 160 and lm.order1_slice == CFG.lexical_order1_slice == (0, 96)
    keys = sorted(lm.state_dict())
    assert keys == sorted([f"tables.{i}" for i in range(5)] + ["hash_mult", "hash_add", "hash_wrap"])
    assert sum(t.numel() for t in lm.tables) == 33_987_104
    assert all(t.dtype == torch.float32 for t in lm.tables)
    assert all(getattr(lm, n).dtype == torch.int64 for n in ("hash_mult", "hash_add", "hash_wrap"))


def test_hash_constants(device):
    lm = LexicalMemory(CFG, device=device, init="empty")
    want = {"hash_mult": HASH_MULT, "hash_add": HASH_ADD, "hash_wrap": HASH_WRAP}
    assert lm.expected_hash_buffers() == want                   # integer constants of the checkpoint (exact)
    lm.check_hash_buffers()
    with torch.no_grad():
        lm.hash_add[1] += 1
    with pytest.raises(RuntimeError):
        lm.check_hash_buffers()


def test_buckets_vs_integer_reference(device):
    lm = _built(device)
    B, T = 2, 64
    ids = _ids(B, T, device)
    with torch.no_grad():
        m = lm(ids)
    # the integer reference: base-(V+1) keys oldest first, row start padded with V, one hash per hashed table
    x = ids.cpu().tolist()
    mult, add = lm.expected_hash_buffers()["hash_mult"], lm.expected_hash_buffers()["hash_add"]
    ref_buckets = [torch.empty((T, B), dtype=torch.int64) for _ in lm.spec]
    for b in range(B):
        for t in range(T):
            hist = [V, V] + x[b]
            for k, (order, rows, hi) in enumerate(lm.spec):
                if order == 1:
                    ref_buckets[k][t, b] = x[b][t]
                    continue
                key = 0
                for j in range(order - 1, -1, -1):
                    key = key * (V + 1) + hist[2 + t - j]
                ref_buckets[k][t, b] = lexical_hash_reference(key, mult[hi], add[hi], rows)
    with torch.no_grad():
        m_ref = lm.lookup([rb.to(device) for rb in ref_buckets])
    assert_match(m, m_ref, "lexical_vector_vs_integer_hash")


def test_session_vs_whole_row(device):
    lm = _built(device)
    B = 3
    ids = _ids(B, sum(CHUNKS), device)
    st = lm.allocate_inference_cache(B, device=device)
    with torch.no_grad():
        whole = lm(ids)
        parts, off = [], 0
        for T in CHUNKS:
            parts.append(lm.session_vector(ids[:, off:off + T], st))
            off += T
    assert_match(torch.cat(parts, dim=0), whole, "session_chunks_vs_whole_row")


def test_gradient_reaches_tables(device):
    lm = _built(device)
    ids = _ids(2, 32, device)
    m = lm(ids)
    g = torch.randn_like(m)
    (m * g).sum().backward()
    # the order-1 table's gradient is the scaled upstream gradient scattered to the ids (reference: dense index_add)
    ref = torch.zeros_like(lm.tables[0])
    ref.index_add_(0, ids.transpose(0, 1).reshape(-1), (g[..., 0:96] * lm.scale).reshape(-1, 96))
    assert_match(lm.tables[0].grad, ref, "order1_table_grad")
    assert all(t.grad is not None and t.grad.dtype == torch.float32 for t in lm.tables)
