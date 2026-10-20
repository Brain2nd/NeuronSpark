"""Model-level lexical n-gram memory: hashed n-gram tables read once per call and shared by the six hippocampal layers.

``forward(ids (B, T)) -> m (T, B, M)`` fp32 with ``M = sum_n heads_n * dim_n`` (1.27B: 96 + 2 * 16 + 2 * 16 = 160)::

    m_t = s * concat over tables k of T_k[bucket_k(x_{t-n_k+1} .. x_t)]                s = 1 / init_std = 50.0

* order 1 reads the exact id (table (vocab_size, 96), no hash); the order-1 channels [0, 96) are the copy detector's
  operand;
* order n >= 2: the key is the base-(V + 1) number of the n ids, oldest first (``V`` = vocab_size; positions before
  the row start read the pad id ``V``, outside the vocabulary); each head of the order hashes the key with its own odd
  64-bit multiplier: ``h = key * mult + add (mod 2^64); h ^= h >> 29; bucket = h mod rows`` (unsigned arithmetic,
  emulated on int64: two's-complement wrap of the product, a masked logical shift, and the unsigned residue fixed
  with ``2^64 mod rows`` for negative values).  1.27B: order 2 (262139, 16) x 2 heads, order 3 (344951, 16) x 2.

The hash constants come from ``random.Random(lexical_memory_hash_seed)`` (per hashed table, in table order:
``mult = getrandbits(62) * 2 + 1`` then ``add = getrandbits(62)``) and are int64 buffers ``hash_mult``, ``hash_add``,
``hash_wrap`` of the checkpoint; :meth:`LexicalMemory.check_hash_buffers` compares the loaded buffers with the
recomputation.  The tables (``tables.{0..4}``, fp32 parameters) launch N(0, init_std^2) on a private CPU generator
seeded with ``mechanism_init_seed(990000, 0)``, drawn table after table.

Inference (:meth:`LexicalMemory.session_vector`): the cache keeps the last ``n_max - 1`` ids of the row
(:class:`~neuronspark.utils.cache.LexicalState`, pad ``V`` at the row start); a call on T >= 1 ids hashes
``cat([window, ids])`` and keeps the last T rows, so every bucket is the integer the whole row computes at that position
(the same keys, the same integer arithmetic); the window then holds the last ``n_max - 1`` ids.

No gradient flows through the indices; the tables receive dense fp32 gradients through ``m``.
"""
from __future__ import annotations

import random
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from neuronspark.modules import mechanism_init_seed
from neuronspark.utils.cache import LexicalState

__all__ = ["LexicalMemory", "lexical_hash_reference", "LEXICAL_LAUNCH_SEED_BASE", "LEXICAL_INITS"]

LEXICAL_LAUNCH_SEED_BASE = 990_000     # private launch stream: mechanism_init_seed(990000, 0)
LEXICAL_INITS = ("launch", "empty")
_U64 = 1 << 64
_MASK35 = (1 << 35) - 1                # logical right shift by 29 of a 64-bit value keeps 35 bits


def lexical_hash_reference(key: int, mult: int, add: int, rows: int) -> int:
    """Python-integer reference of one bucket: ``h = key * mult + add mod 2^64; h ^= h >> 29; h mod rows``."""
    h = (int(key) * int(mult) + int(add)) % _U64
    h ^= h >> 29
    return h % int(rows)


def _hash_constants(rows, heads, seed: int) -> Tuple[List[int], List[int], List[int], List[Tuple[int, int, int]]]:
    """(mult, add, wrap, spec) for the table layout: spec = (order, rows, hash index or -1) per table in m order."""
    rng = random.Random(int(seed))
    mult: List[int] = []
    add: List[int] = []
    spec: List[Tuple[int, int, int]] = []
    for n, (r, H) in enumerate(zip(rows, heads)):
        for _ in range(int(H)):
            if n == 0:
                spec.append((1, int(r), -1))
            else:
                mult.append(rng.getrandbits(62) * 2 + 1)
                add.append(rng.getrandbits(62))
                spec.append((n + 1, int(r), len(mult) - 1))
    wrap = [(_U64 % r) for (_, r, i) in spec if i >= 0]
    return mult, add, wrap, spec


class LexicalMemory(nn.Module):
    """``LexicalMemory(config, device=None, init="launch")``: the hashed n-gram tables of the model (see the module
    docstring).  ``init="empty"`` allocates the tables without drawing (a build followed by a checkpoint load); the
    hash buffers are computed in both cases."""

    def __init__(self, config, device: Optional[torch.device] = None, init: str = "launch") -> None:
        super().__init__()
        if init not in LEXICAL_INITS:
            raise ValueError(f"LexicalMemory: init must be one of {LEXICAL_INITS}, got {init!r}")
        self.vocab_size = int(config.vocab_size)
        self.pad_id = self.vocab_size
        self.rows = tuple(int(r) for r in config.lexical_memory_rows)
        self.heads = tuple(int(h) for h in config.lexical_memory_heads)
        self.dims = tuple(int(d) for d in config.lexical_memory_head_dims)
        self.init_std = float(config.lexical_memory_init_std)
        self.scale = 1.0 / self.init_std
        self.hash_seed = int(config.lexical_memory_hash_seed)
        self.M = sum(h * d for h, d in zip(self.heads, self.dims))
        if not (len(self.rows) == len(self.heads) == len(self.dims) >= 1):
            raise ValueError("LexicalMemory: rows, heads and head dims need one entry per order")
        if self.rows[0] != self.vocab_size or self.heads[0] != 1:
            raise ValueError("LexicalMemory: order 1 must be the exact-id table (rows = vocab_size, one head)")
        if (self.vocab_size + 1) ** len(self.rows) >= (1 << 63):
            raise ValueError("LexicalMemory: the base-(V+1) key of the highest order does not fit in int64")
        mult, add, wrap, spec = _hash_constants(self.rows, self.heads, self.hash_seed)
        self.spec = tuple(spec)                                   # (order, rows, hash index) per table, in m order
        self.n_max = max(o for o, _, _ in self.spec)
        tables = []
        for (_, r, _), d in zip(self.spec, [d for d, H in zip(self.dims, self.heads) for _ in range(H)]):
            tables.append(nn.Parameter(torch.empty((r, d), dtype=torch.float32, device=device)))
        self.tables = nn.ParameterList(tables)
        self.register_buffer("hash_mult", torch.tensor(mult if mult else [1], dtype=torch.int64, device=device))
        self.register_buffer("hash_add", torch.tensor(add if add else [0], dtype=torch.int64, device=device))
        self.register_buffer("hash_wrap", torch.tensor(wrap if wrap else [0], dtype=torch.int64, device=device))
        if init == "launch":
            self.reset_parameters()

    # ------------------------------------------------------------------------------------------------ launch / checks
    @torch.no_grad()
    def reset_parameters(self, generator: Optional[torch.Generator] = None) -> None:
        """Draw every table N(0, init_std^2) on the CPU, table after table, from ``generator`` (default: a private
        generator seeded with ``mechanism_init_seed(990000, 0)``), and copy it to the table's device."""
        g = generator
        if g is None:
            g = torch.Generator(device="cpu").manual_seed(mechanism_init_seed(LEXICAL_LAUNCH_SEED_BASE, 0))
        if g.device.type != "cpu":
            raise ValueError("LexicalMemory.reset_parameters: the launch draws on the CPU; pass a CPU generator")
        for t in self.tables:
            v = torch.empty(tuple(t.shape), dtype=torch.float32, device="cpu").normal_(mean=0.0, std=self.init_std,
                                                                                       generator=g)
            t.copy_(v)

    def expected_hash_buffers(self) -> Dict[str, List[int]]:
        """The hash constants recomputed from the seed: ``{"hash_mult", "hash_add", "hash_wrap"}`` as int lists."""
        mult, add, wrap, _ = _hash_constants(self.rows, self.heads, self.hash_seed)
        return {"hash_mult": mult or [1], "hash_add": add or [0], "hash_wrap": wrap or [0]}

    def check_hash_buffers(self) -> None:
        """Raise unless the (loaded) hash buffers hold the constants of the seed (integer constants, compared
        exactly)."""
        want = self.expected_hash_buffers()
        for name, values in want.items():
            got = [int(v) for v in getattr(self, name).detach().cpu().tolist()]
            if got != [int(v) for v in values]:
                raise RuntimeError(f"LexicalMemory: buffer {name} {got} differs from the constants of hash seed "
                                   f"{self.hash_seed} {values}")

    @property
    def order1_slice(self) -> Tuple[int, int]:
        """(start, stop) of the order-1 channels of ``m`` (the exact-id table row of x_t)."""
        return (0, int(self.dims[0]))

    def extra_repr(self) -> str:
        return (f"vocab_size={self.vocab_size}, tables={[tuple(t.shape) for t in self.tables]}, M={self.M}, "
                f"scale={self.scale}")

    # ------------------------------------------------------------------------------------------------ hashing
    def buckets(self, ids: torch.Tensor) -> List[torch.Tensor]:
        """``ids`` (T, B) int64 -> one (T, B) int64 index tensor per table (in m order); positions before the start of
        ``ids`` read the pad id."""
        if ids.dim() != 2:
            raise ValueError(f"LexicalMemory.buckets: expected (T, B) ids, got {tuple(ids.shape)}")
        T = int(ids.shape[0])
        V1 = self.vocab_size + 1
        n_max = self.n_max
        pad = torch.full((n_max - 1, int(ids.shape[1])), self.pad_id, dtype=torch.int64, device=ids.device)
        ext = torch.cat([pad, ids.to(torch.int64)], dim=0)                 # ext[n_max - 1 + t] = x_t
        keys: Dict[int, torch.Tensor] = {}
        out: List[torch.Tensor] = []
        wrap_i = 0
        for order, r, hi in self.spec:
            if order == 1:
                out.append(ids.to(torch.int64))
                continue
            if order not in keys:
                key = torch.zeros_like(ids, dtype=torch.int64)
                for j in range(order - 1, -1, -1):                          # x_{t-j}, oldest first
                    key = key * V1 + ext[n_max - 1 - j:n_max - 1 - j + T]
                keys[order] = key
            h = keys[order] * self.hash_mult[hi] + self.hash_add[hi]       # wraps mod 2^64 (two's complement)
            h = torch.bitwise_xor(h, torch.bitwise_and(torch.bitwise_right_shift(h, 29), _MASK35))
            b = torch.remainder(h, r)
            b = torch.where(h < 0, torch.remainder(b + self.hash_wrap[wrap_i], r), b)
            wrap_i += 1
            out.append(b)
        return out

    def lookup(self, buckets: List[torch.Tensor]) -> torch.Tensor:
        """``m`` = the concatenated table rows of ``buckets`` (fp32) times ``scale``."""
        parts = [F.embedding(b, t) for b, t in zip(buckets, self.tables)]
        return torch.cat([p.float() for p in parts], dim=-1) * self.scale

    # ------------------------------------------------------------------------------------------------ regimes
    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """Whole row: ``token_ids`` (B, T) -> ``m`` (T, B, M) fp32 (row start padded)."""
        if token_ids.dim() != 2:
            raise ValueError(f"LexicalMemory: expected (B, T) token ids, got {tuple(token_ids.shape)}")
        ids = token_ids.transpose(0, 1)                                    # (T, B)
        return self.lookup(self.buckets(ids))

    def session_vector(self, token_ids: torch.Tensor, state: LexicalState) -> torch.Tensor:
        """Inference call: ``token_ids`` (B, T >= 1) -> ``m`` (T, B, M) fp32 of the call, the n-gram keys reading the
        carried window ``state.win`` (B, n_max - 1); the window is advanced in place to the last n_max - 1 ids."""
        if token_ids.dim() != 2:
            raise ValueError(f"LexicalMemory: expected (B, T) token ids, got {tuple(token_ids.shape)}")
        win = state.win                                                    # (B, k), oldest first
        k = int(win.shape[1])
        if k != self.n_max - 1 or int(win.shape[0]) != int(token_ids.shape[0]) or win.dtype != torch.int64:
            raise RuntimeError(f"LexicalMemory: the window must be ({int(token_ids.shape[0])}, {self.n_max - 1}) "
                               f"int64, got {tuple(win.shape)} {win.dtype}")
        ids = token_ids.transpose(0, 1)                                    # (T, B)
        ext = torch.cat([win.t(), ids.to(torch.int64)], dim=0)            # (k + T, B): ext[k + t] = x_t
        bk = [b[k:] for b in self.buckets(ext)]
        m = self.lookup(bk)
        win.copy_(ext[ext.shape[0] - k:].t())
        return m

    def allocate_inference_cache(self, batch_size: int, device: Optional[torch.device] = None) -> LexicalState:
        """The row-start window (pad ids) for ``batch_size`` rows."""
        dev = self.hash_mult.device if device is None else torch.device(device)
        return LexicalState.allocate(int(batch_size), self.n_max - 1, self.pad_id, dev)
