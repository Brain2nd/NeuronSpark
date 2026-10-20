"""Inference cache: the recurrent state of every layer plus the model-level session state.

One ``InferenceParams`` is allocated per (batch size, ring capacity) by ``NeuronSparkLMHeadModel.allocate_inference_cache``
and updated in place by prefill (T >= 2) and decode (T == 1) calls.  Every write is an in-place ``copy_`` /
``index_copy_`` / fill into these tensors and the token counters advance on the device, so a CUDA graph captured over a
decode step stays valid across replays and across :meth:`InferenceParams.reset`.

State per row (W = ``max_seqlen``, the ring capacity shared by every ring; 1.27B shapes):

* ``SDState`` (18 SD layers): ``conv`` (9472, 3) fp32, the last three conv inputs (lags t-3, t-2, t-1; their values
  are bf16-exact); ``ssm`` (144, 64, 128) in the cache's SD state dtype (bf16 by default, fp32 as the option).
* ``MemoryState`` (6 memory layers): ``conv_hist`` (4608, 3) fp32 qkv conv history; ``pool_k`` / ``pool_gv``
  (6, W, 256) bf16 key and gated-value rings; ``pool_cum`` (6, W) fp32 absolute write clocks (+inf = empty slot);
  ``pool_run`` (6,) fp32 carried clock; ``vr_sum`` (1536,) fp32 running sum of the value-residual gate input; the four
  PLIF membranes (``gate_neuron``, ``input_neuron2`` with its running input sum, ``ffn_gate``, ``ffn_up``).
* model level: ``LexicalState.win`` (2,) int64 last n_max - 1 ids; ``InductionState.keys`` (W, 4) int64 ring of the
  four ids preceding each written slot and ``win`` (4,) the last four ids; ``CopyState.u`` (W, 96) fp32 ring of the
  unit order-1 lexical vectors (one ring for all memory layers: the operand is model level).
* counters: ``pos`` (B,) int64 tokens consumed per row (RoPE positions, running-mean counts, live ring length) and
  ``ring_ptr`` (1,) int64: the next slot of every ring is ``ring_ptr % W``.

Reset values (:meth:`InferenceParams.reset`, in place): zeros, except ``pool_cum`` = +inf and the id windows / key ring =
the pad id ``V`` (= vocab_size).  ``pool_k`` / ``pool_gv`` are zero at allocation and are not reset: a slot is read only
where its clock is finite.

The SD state dtype is the model's one numerics option (``NeuronSparkConfig.inference_sd_state_dtype``, overridable per
allocation); the cache prints one line per process naming the dtype in use.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass, field
from typing import Dict, Iterator, List, Optional, Tuple, Union

import torch

__all__ = [
    "SDState",
    "PLIFState",
    "MemoryState",
    "LexicalState",
    "InductionState",
    "CopyState",
    "InferenceParams",
    "resolve_sd_state_dtype",
    "announce_sd_state_dtype",
    "RING_MULTIPLE",
    "DEFAULT_MAX_SEQLEN",
    "MAX_DECODE_BATCH",
]

# ring capacity W must be a multiple of this (prefill ring kernels tile W) and at most the RoPE table length
RING_MULTIPLE = 512
# default ring capacity of allocate_inference_cache
DEFAULT_MAX_SEQLEN = 2048
# largest decode batch the decode kernels' fixed launch tables cover
MAX_DECODE_BATCH = 32

_SD_STATE_DTYPES = {"bfloat16": torch.bfloat16, "float32": torch.float32}
# documented first-call cache: the dtypes already announced in this process
_ANNOUNCED: set = set()


def resolve_sd_state_dtype(value: Union[str, torch.dtype]) -> torch.dtype:
    """``"bfloat16"`` / ``"float32"`` (or the torch dtypes) -> torch dtype; anything else raises."""
    if isinstance(value, torch.dtype):
        if value not in (torch.bfloat16, torch.float32):
            raise ValueError(f"SD state dtype must be torch.bfloat16 or torch.float32, got {value}")
        return value
    if value not in _SD_STATE_DTYPES:
        raise ValueError(f"SD state dtype must be one of {tuple(_SD_STATE_DTYPES)}, got {value!r}")
    return _SD_STATE_DTYPES[value]


def announce_sd_state_dtype(dtype: torch.dtype) -> None:
    """Print, once per process and dtype, which SD state storage the inference cache uses."""
    name = "bfloat16" if dtype == torch.bfloat16 else "float32"
    if name in _ANNOUNCED:
        return
    _ANNOUNCED.add(name)
    print(f"neuronspark: inference SD state stored in {name} (default bfloat16; option float32)",
          file=sys.stderr, flush=True)


def _zeros(shape, dtype, device) -> torch.Tensor:
    return torch.zeros(shape, dtype=dtype, device=device)


# ------------------------------------------------------------------------------------------------ per-layer states
@dataclass
class SDState:
    """Recurrent state of one SD layer: ``conv`` (B, conv_dim, width - 1) fp32, ``ssm`` (B, nheads, head_dim,
    state_size) in the SD state dtype."""

    conv: torch.Tensor
    ssm: torch.Tensor

    @classmethod
    def allocate(cls, batch_size: int, conv_dim: int, conv_width: int, nheads: int, head_dim: int, state_size: int,
                 ssm_dtype: torch.dtype, device: torch.device) -> "SDState":
        return cls(conv=_zeros((batch_size, conv_dim, conv_width - 1), torch.float32, device),
                   ssm=_zeros((batch_size, nheads, head_dim, state_size), resolve_sd_state_dtype(ssm_dtype), device))

    def reset(self) -> None:
        self.conv.zero_()
        self.ssm.zero_()

    def tensors(self, prefix: str = "") -> Iterator[Tuple[str, torch.Tensor]]:
        yield prefix + "conv", self.conv
        yield prefix + "ssm", self.ssm


@dataclass
class PLIFState:
    """Membrane ``v`` (B, C) fp32 of one PLIF population; ``xsum`` (B, C) fp32 running sum of its input for the
    centred neuron (``input_neuron2``), None otherwise."""

    v: torch.Tensor
    xsum: Optional[torch.Tensor] = None

    @classmethod
    def allocate(cls, batch_size: int, channels: int, centred: bool, device: torch.device) -> "PLIFState":
        return cls(v=_zeros((batch_size, channels), torch.float32, device),
                   xsum=_zeros((batch_size, channels), torch.float32, device) if centred else None)

    def reset(self) -> None:
        self.v.zero_()
        if self.xsum is not None:
            self.xsum.zero_()

    def tensors(self, prefix: str = "") -> Iterator[Tuple[str, torch.Tensor]]:
        yield prefix + "v", self.v
        if self.xsum is not None:
            yield prefix + "xsum", self.xsum


@dataclass
class MemoryState:
    """Recurrent state of one memory layer (see the module docstring for shapes)."""

    conv_hist: torch.Tensor
    pool_k: torch.Tensor
    pool_gv: torch.Tensor
    pool_cum: torch.Tensor
    pool_run: torch.Tensor
    vr_sum: torch.Tensor
    gate_neuron: PLIFState
    input_neuron2: PLIFState
    ffn_gate: PLIFState
    ffn_up: PLIFState

    @classmethod
    def allocate(cls, batch_size: int, max_seqlen: int, *, qkv_dim: int, conv_width: int, heads: int,
                 key_head_dim: int, value_head_dim: int, value_dim: int, d_model: int, d_ff: int,
                 device: torch.device) -> "MemoryState":
        B, W = int(batch_size), int(max_seqlen)
        st = cls(
            conv_hist=_zeros((B, qkv_dim, conv_width - 1), torch.float32, device),
            pool_k=_zeros((B, heads, W, key_head_dim), torch.bfloat16, device),
            pool_gv=_zeros((B, heads, W, value_head_dim), torch.bfloat16, device),
            pool_cum=torch.full((B, heads, W), float("inf"), dtype=torch.float32, device=device),
            pool_run=_zeros((B, heads), torch.float32, device),
            vr_sum=_zeros((B, value_dim), torch.float32, device),
            gate_neuron=PLIFState.allocate(B, d_model, False, device),
            input_neuron2=PLIFState.allocate(B, d_model, True, device),
            ffn_gate=PLIFState.allocate(B, d_ff, False, device),
            ffn_up=PLIFState.allocate(B, d_ff, False, device),
        )
        return st

    def reset(self) -> None:
        self.conv_hist.zero_()
        self.pool_cum.fill_(float("inf"))
        self.pool_run.zero_()
        self.vr_sum.zero_()
        for neuron in (self.gate_neuron, self.input_neuron2, self.ffn_gate, self.ffn_up):
            neuron.reset()

    def tensors(self, prefix: str = "") -> Iterator[Tuple[str, torch.Tensor]]:
        for name in ("conv_hist", "pool_k", "pool_gv", "pool_cum", "pool_run", "vr_sum"):
            yield prefix + name, getattr(self, name)
        for name in ("gate_neuron", "input_neuron2", "ffn_gate", "ffn_up"):
            yield from getattr(self, name).tensors(f"{prefix}{name}.")


# ------------------------------------------------------------------------------------------------ model-level states
@dataclass
class LexicalState:
    """``win`` (B, n_max - 1) int64: the last ids, oldest first (pad = ``pad_id`` = vocab_size at row start)."""

    win: torch.Tensor
    pad_id: int

    @classmethod
    def allocate(cls, batch_size: int, window: int, pad_id: int, device: torch.device) -> "LexicalState":
        return cls(win=torch.full((batch_size, window), int(pad_id), dtype=torch.int64, device=device), pad_id=int(pad_id))

    def reset(self) -> None:
        self.win.fill_(self.pad_id)

    def tensors(self, prefix: str = "") -> Iterator[Tuple[str, torch.Tensor]]:
        yield prefix + "win", self.win


@dataclass
class InductionState:
    """``keys`` (B, W, n_keys) int64 ring of the ids preceding each written slot; ``win`` (B, n_keys) int64 last ids."""

    keys: torch.Tensor
    win: torch.Tensor
    pad_id: int

    @classmethod
    def allocate(cls, batch_size: int, max_seqlen: int, n_keys: int, pad_id: int,
                 device: torch.device) -> "InductionState":
        return cls(keys=torch.full((batch_size, max_seqlen, n_keys), int(pad_id), dtype=torch.int64, device=device),
                   win=torch.full((batch_size, n_keys), int(pad_id), dtype=torch.int64, device=device),
                   pad_id=int(pad_id))

    def reset(self) -> None:
        self.keys.fill_(self.pad_id)
        self.win.fill_(self.pad_id)

    def tensors(self, prefix: str = "") -> Iterator[Tuple[str, torch.Tensor]]:
        yield prefix + "keys", self.keys
        yield prefix + "win", self.win


@dataclass
class CopyState:
    """``u`` (B, W, dim) fp32 ring of the unit order-1 lexical vectors (the copy detector's operands)."""

    u: torch.Tensor

    @classmethod
    def allocate(cls, batch_size: int, max_seqlen: int, dim: int, device: torch.device) -> "CopyState":
        return cls(u=_zeros((batch_size, max_seqlen, dim), torch.float32, device))

    def reset(self) -> None:
        self.u.zero_()

    def tensors(self, prefix: str = "") -> Iterator[Tuple[str, torch.Tensor]]:
        yield prefix + "u", self.u


# ------------------------------------------------------------------------------------------------ the cache
@dataclass
class InferenceParams:
    """The whole inference cache (mamba's ``InferenceParams`` plus the NeuronSpark state).

    ``max_seqlen`` is the ring capacity W (a multiple of 512, at most the RoPE table length); ``seqlen_offset`` counts
    the tokens consumed on the host side (it only selects prefill vs decode and checks capacity; captured graphs read
    the device counters ``pos`` / ``ring_ptr``).  ``layers`` maps a layer index to its ``SDState`` / ``MemoryState``.
    """

    max_seqlen: int
    max_batch_size: int
    pos: torch.Tensor
    ring_ptr: torch.Tensor
    sd_state_dtype: torch.dtype
    layers: Dict[int, Union[SDState, MemoryState]]
    lexical: LexicalState
    induction: InductionState
    copy: CopyState
    seqlen_offset: int = 0
    batch_size_offset: int = 0
    rope_max_seq_len: int = 8192
    announce: bool = field(default=True, repr=False)

    def __post_init__(self) -> None:
        W, B = int(self.max_seqlen), int(self.max_batch_size)
        if W <= 0 or W % RING_MULTIPLE != 0 or W > int(self.rope_max_seq_len):
            raise ValueError(f"InferenceParams: max_seqlen (ring capacity) must be a positive multiple of "
                             f"{RING_MULTIPLE} and at most {self.rope_max_seq_len}, got {W}")
        if not 1 <= B <= MAX_DECODE_BATCH:
            raise ValueError(f"InferenceParams: max_batch_size must lie in [1, {MAX_DECODE_BATCH}], got {B}")
        if self.pos.shape != (B,) or self.pos.dtype != torch.int64:
            raise ValueError(f"InferenceParams: pos must be ({B},) int64, got {tuple(self.pos.shape)} {self.pos.dtype}")
        if self.ring_ptr.shape != (1,) or self.ring_ptr.dtype != torch.int64:
            raise ValueError("InferenceParams: ring_ptr must be (1,) int64")
        self.sd_state_dtype = resolve_sd_state_dtype(self.sd_state_dtype)
        for i, st in self.layers.items():
            if isinstance(st, SDState) and st.ssm.dtype != self.sd_state_dtype:
                raise ValueError(f"InferenceParams: layer {i} SD state is {st.ssm.dtype}, the cache declares "
                                 f"{self.sd_state_dtype}")
        if self.announce:
            announce_sd_state_dtype(self.sd_state_dtype)

    @staticmethod
    def allocate_counters(batch_size: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        """Zero ``pos`` (B,) and ``ring_ptr`` (1,) int64 counters."""
        return (torch.zeros((int(batch_size),), dtype=torch.int64, device=device),
                torch.zeros((1,), dtype=torch.int64, device=device))

    @property
    def batch_size(self) -> int:
        return int(self.max_batch_size)

    def check_room(self, num_tokens: int) -> None:
        """Raise unless ``num_tokens`` more tokens fit in the rings (host-side count, no device sync)."""
        if self.seqlen_offset + int(num_tokens) > int(self.max_seqlen):
            raise RuntimeError(f"InferenceParams: {self.seqlen_offset} + {int(num_tokens)} tokens exceed the ring "
                               f"capacity {self.max_seqlen} (allocate the cache with a larger max_seqlen)")

    def reset(self, max_seqlen: Optional[int] = None, max_batch_size: Optional[int] = None) -> None:
        """Row-start state in place (captured graphs stay valid).  Mamba's call form ``reset(max_length, batch_size)``
        is allowed when the length fits the ring capacity and the batch size equals the allocated one: the cache is
        never resized in place."""
        if (max_seqlen is not None and int(max_seqlen) > int(self.max_seqlen)) or \
                (max_batch_size is not None and int(max_batch_size) != int(self.max_batch_size)):
            raise ValueError(f"InferenceParams.reset: the cache holds (capacity {self.max_seqlen}, batch "
                             f"{self.max_batch_size}); got ({max_seqlen}, {max_batch_size}): allocate a new cache")
        self.seqlen_offset = 0
        self.pos.zero_()
        self.ring_ptr.zero_()
        for st in self.layers.values():
            st.reset()
        self.lexical.reset()
        self.induction.reset()
        self.copy.reset()

    def tensors(self) -> Iterator[Tuple[str, torch.Tensor]]:
        """Every state tensor with a stable name (``layers.<i>.<field>``, ``lexical.win``, ...)."""
        yield "pos", self.pos
        yield "ring_ptr", self.ring_ptr
        for i in sorted(self.layers):
            yield from self.layers[i].tensors(f"layers.{i}.")
        yield from self.lexical.tensors("lexical.")
        yield from self.induction.tensors("induction.")
        yield from self.copy.tensors("copy.")

    def snapshot(self) -> Tuple[int, List[torch.Tensor]]:
        """Copies of every state tensor and the host offset (for warm-up calls that must leave the cache unchanged)."""
        return self.seqlen_offset, [t.clone() for _, t in self.tensors()]

    def restore(self, snap: Tuple[int, List[torch.Tensor]]) -> None:
        """Write a :meth:`snapshot` back in place."""
        offset, saved = snap
        current = [t for _, t in self.tensors()]
        if len(current) != len(saved):
            raise RuntimeError("InferenceParams.restore: the snapshot does not belong to this cache")
        for dst, src in zip(current, saved):
            dst.copy_(src)
        self.seqlen_offset = int(offset)

    def nbytes(self) -> int:
        return sum(t.numel() * t.element_size() for _, t in self.tensors())
