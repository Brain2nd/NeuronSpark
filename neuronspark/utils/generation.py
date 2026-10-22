"""Generation: the prompt as one prefill call, then one token per step -- each step replayed from a captured CUDA graph
(``cg=True``, the default) or run eagerly -- with greedy decoding or sampling.

Every row of a batch advances together, so the prompts of one call have equal length (B, T).  ``max_length`` counts the
prompt and the new tokens; the hippocampal slot rings hold :func:`ring_capacity` ``(max_length)`` slots (a multiple of
512, at most ``rope_max_seq_len``).

CUDA-graph decoding keeps one :class:`DecodeGraph` on the model (``model._decode_graph``): the inference cache the step
reads and writes, the memory pool of the capture and the captured single-token step.  Before each generation the cache
is reset to the row start; the graph is captured again only when the batch size, the device, the dendritic-layer state
dtype or a weight (its storage or version) changed, or when the rings must grow.

Sampling (:func:`next_tokens`): ``top_k == 1`` is greedy (the first index on ties).  Otherwise the logits are divided by
the temperature and filtered in this order -- the ``top_k`` largest (``top_k > 1``), the tokens whose probability is at
least ``min_p`` times the largest one (``0 < min_p < 1``), the most probable tokens whose cumulative probability first
reaches ``top_p`` (``0 < top_p < 1``) -- and one token is drawn from the remaining distribution.  ``teacher_outputs``
(B, >= max_length) forces the tokens instead (teacher-forced decoding).
"""
from __future__ import annotations

import gc
from dataclasses import dataclass
from typing import Optional, Tuple, Union

import torch

from neuronspark.utils.cache import InferenceParams, RING_MULTIPLE, resolve_dendritic_state_dtype

__all__ = ["DecodeOutput", "DecodeGraph", "GenerationMixin", "decode", "next_tokens", "repetition_penalty_",
           "ring_capacity", "prepare_decode_graph"]


# ================================================================================================ sampling
def next_tokens(logits: torch.Tensor, top_k: int = 1, top_p: float = 0.0, min_p: float = 0.0,
                temperature: float = 1.0) -> torch.Tensor:
    """(B,) int64 next tokens from logits (B, V) (module docstring)."""
    if top_k == 1:
        return logits.argmax(dim=-1)
    if temperature <= 0.0:
        raise ValueError(f"temperature must be positive, got {temperature}")
    if not 0.0 <= top_p <= 1.0:
        raise ValueError(f"top_p must be in [0, 1], got {top_p}")
    scores = logits.float() / temperature
    neg = float("-inf")
    if top_k > 1:
        kth = torch.topk(scores, min(int(top_k), scores.shape[-1]), dim=-1).values[..., -1:]
        scores = scores.masked_fill(scores < kth, neg)
    if 0.0 < min_p < 1.0:
        probs = torch.softmax(scores, dim=-1)
        scores = scores.masked_fill(probs < min_p * probs.amax(dim=-1, keepdim=True), neg)
    if 0.0 < top_p < 1.0:
        probs, order = torch.softmax(scores, dim=-1).sort(dim=-1, descending=True)
        before = probs.cumsum(dim=-1) - probs                # the mass of the more probable tokens
        drop_sorted = before >= top_p                         # the most probable token is always kept
        scores = scores.masked_fill(torch.zeros_like(drop_sorted).scatter(-1, order, drop_sorted), neg)
    return torch.multinomial(torch.softmax(scores, dim=-1), num_samples=1).squeeze(-1)


def repetition_penalty_(logits: torch.Tensor, seen: torch.Tensor, penalty: float) -> torch.Tensor:
    """In place on logits (B, V): the logits of the tokens in ``seen`` (B, L) are divided by ``penalty`` when positive
    and multiplied by it when negative (each lowers the token's probability for penalty > 1)."""
    if penalty == 1.0:
        return logits
    picked = logits.gather(1, seen)
    logits.scatter_(1, seen, torch.where(picked > 0, picked / penalty, picked * penalty))
    return logits


# ================================================================================================ ring capacity
def ring_capacity(max_length: int, rope_max_seq_len: int) -> int:
    """Slots that hold ``max_length`` tokens: ``max_length`` rounded up to a multiple of 512; raises when that exceeds
    ``rope_max_seq_len`` (the RoPE table bounds the rings)."""
    n = int(max_length)
    if n <= 0:
        raise ValueError(f"max_length must be positive, got {max_length}")
    slots = -(-n // RING_MULTIPLE) * RING_MULTIPLE
    if slots > int(rope_max_seq_len):
        raise ValueError(f"max_length {n} exceeds the longest supported row ({int(rope_max_seq_len)} tokens)")
    return slots


# ================================================================================================ CUDA-graph step
class DecodeGraph:
    """One captured single-token decode step of a model over its own inference cache (module docstring)."""

    def __init__(self) -> None:
        self.key: Optional[Tuple] = None          # (device, state dtype, batch size, weight signature)
        self.capacity = 0                          # ring slots of the cache
        self.params: Optional[InferenceParams] = None
        self.graph: Optional[torch.cuda.CUDAGraph] = None
        self.pool = None
        self.tokens: Optional[torch.Tensor] = None     # static input (B, 1)
        self.logits: Optional[torch.Tensor] = None     # static output (B, 1, V)
        self.bf16_weights: list = []                    # the bf16 weight copies the graph reads, kept alive with it

    @staticmethod
    def _weights(model) -> Tuple:
        return tuple((int(p.data_ptr()), int(p._version)) for p in model.parameters())

    def ensure(self, model, batch_size: int, max_length: int, dendritic_state_dtype=None,
               warmup_steps: int = 2) -> InferenceParams:
        """The cache for this generation, reset to the row start; (re)captures the step when needed."""
        device = next(model.parameters()).device
        if device.type != "cuda":
            raise RuntimeError(f"CUDA-graph decoding needs the model on a CUDA device, got {device}")
        cfg = model.config
        dtype = resolve_dendritic_state_dtype(cfg.inference_dendritic_state_dtype if dendritic_state_dtype is None
                                              else dendritic_state_dtype)
        slots = ring_capacity(max_length, cfg.rope_max_seq_len)
        key = (device, dtype, int(batch_size), self._weights(model))
        if key != self.key or slots > self.capacity:
            self._drop()
            model.prepare_inference()
            self.params = model.allocate_inference_cache(int(batch_size), slots, dendritic_state_dtype=dtype)
            self.capacity = slots
            self.pool = torch.cuda.graphs.graph_pool_handle()
            self._capture(model, int(warmup_steps))
            self.bf16_weights = [w for w in (getattr(p, "_ns_bf16", None) for p in model.parameters()) if w is not None]
            self.key = (device, dtype, int(batch_size), self._weights(model))
        self.params.reset()
        return self.params

    def _drop(self) -> None:
        self.key = self.params = self.graph = self.pool = self.tokens = self.logits = None
        self.capacity = 0
        self.bf16_weights = []
        gc.collect()

    @torch.no_grad()
    def _capture(self, model, warmup_steps: int) -> None:
        params = self.params
        self.tokens = torch.zeros((params.max_batch_size, 1), dtype=torch.long, device=params.pos.device)
        start = params.snapshot()
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):                   # first calls (kernel compilation, caches) outside the capture
            for _ in range(warmup_steps):
                model(self.tokens, inference_params=params, num_last_tokens=1)
        torch.cuda.current_stream().wait_stream(side)
        torch.cuda.synchronize()
        params.restore(start)
        offset = params.seqlen_offset
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph, pool=self.pool):   # recorded, not executed
            self.logits = model(self.tokens, inference_params=params, num_last_tokens=1).logits
        params.seqlen_offset = offset                    # the host count the recorded call advanced

    def step(self, tokens: torch.Tensor) -> torch.Tensor:
        """Replay one step on ``tokens`` (B, 1): the cache advances in place; returns the logits (B, V)."""
        self.params.check_room(1)
        self.tokens.copy_(tokens)
        self.graph.replay()
        self.params.seqlen_offset += 1
        return self.logits[:, -1].clone()


def prepare_decode_graph(model, batch_size: int, max_length: int, dendritic_state_dtype=None) -> DecodeGraph:
    """``model._decode_graph`` ready for ``batch_size`` rows of total length ``max_length`` (its cache reset)."""
    graph = getattr(model, "_decode_graph", None)
    if graph is None:
        graph = DecodeGraph()
        model._decode_graph = graph
    graph.ensure(model, batch_size, max_length, dendritic_state_dtype)
    return graph


# ================================================================================================ decode loop
@dataclass
class DecodeOutput:
    """``sequences`` (B, L) int64: the prompts followed by the generated tokens; ``scores``: the logits (B, V) of each
    generated position when requested."""

    sequences: torch.Tensor
    scores: Optional[Tuple[torch.Tensor, ...]] = None


@torch.no_grad()
def decode(input_ids: torch.Tensor, model, max_length: int, top_k: int = 1, top_p: float = 0.0, min_p: float = 0.0,
           temperature: float = 1.0, repetition_penalty: float = 1.0, eos_token_id: Optional[int] = None,
           teacher_outputs: Optional[torch.Tensor] = None, cg: bool = True, output_scores: bool = False,
           streamer=None, dendritic_state_dtype: Optional[Union[str, torch.dtype]] = None) -> DecodeOutput:
    """Greedy decoding or sampling (module docstring).

    ``input_ids`` (B, T) int64 prompts of equal length; ``max_length`` the total length (> T); ``eos_token_id``: stop
    once every row has produced it; ``teacher_outputs`` (B, >= max_length): take position t's token from it;
    ``cg``: replay the steps from the captured graph; ``output_scores``: return the logits of every generated
    position; ``streamer``: an object with ``put(tokens)`` / ``end()``; ``dendritic_state_dtype``: the cache's
    dendritic-layer state storage (default ``config.inference_dendritic_state_dtype``).
    """
    if input_ids.dim() != 2:
        raise ValueError(f"decode: input_ids must be (B, T), got {tuple(input_ids.shape)}")
    batch, prompt_len = (int(s) for s in input_ids.shape)
    total = int(max_length)
    if total <= prompt_len:
        raise ValueError(f"decode: max_length {total} is the total length and must exceed the prompt length "
                         f"{prompt_len}")
    slots = ring_capacity(total, model.config.rope_max_seq_len)
    if cg:
        graph = prepare_decode_graph(model, batch, total, dendritic_state_dtype)
        params = graph.params
    else:
        graph = None
        params = model.allocate_inference_cache(batch, slots, dendritic_state_dtype=dendritic_state_dtype)
    if streamer is not None:
        streamer.put(input_ids.cpu())
    forced = int(teacher_outputs.shape[1]) if teacher_outputs is not None else 0
    pieces, scores = [input_ids], []
    seen = input_ids
    logits = model(input_ids, inference_params=params, num_last_tokens=1).logits[:, -1]     # prefill
    for pos in range(prompt_len, total):
        if output_scores:
            scores.append(logits.clone())
        if repetition_penalty != 1.0:
            logits = repetition_penalty_(logits.float(), seen, repetition_penalty)
        if pos < forced:
            tok = teacher_outputs[:, pos]
        else:
            tok = next_tokens(logits, top_k=top_k, top_p=top_p, min_p=min_p, temperature=temperature)
        tok = tok.to(torch.int64).unsqueeze(1)
        pieces.append(tok)
        if repetition_penalty != 1.0:
            seen = torch.cat([seen, tok], dim=1)
        if streamer is not None:
            streamer.put(tok.cpu())
        if pos == total - 1 or (eos_token_id is not None and bool((tok == eos_token_id).all())):
            break
        if graph is not None:
            logits = graph.step(tok)
        else:
            logits = model(tok, inference_params=params, num_last_tokens=1).logits[:, -1]
    if streamer is not None:
        streamer.end()
    return DecodeOutput(sequences=torch.cat(pieces, dim=1), scores=tuple(scores))


class GenerationMixin:
    """``generate`` for the language model (it provides ``config``, ``allocate_inference_cache``,
    ``prepare_inference`` and ``forward(input_ids, inference_params, num_last_tokens)``)."""

    def generate(self, input_ids: torch.Tensor, max_length: int, top_k: int = 1, top_p: float = 0.0,
                 min_p: float = 0.0, temperature: float = 1.0, return_dict_in_generate: bool = False,
                 output_scores: bool = False, **kwargs):
        """:func:`decode` (CUDA-graph steps unless ``cg=False``).  ``max_length`` is the total length: it must exceed
        the prompt length and fit ``rope_max_seq_len``.  Returns the sequences (B, L), or the :class:`DecodeOutput`
        with ``return_dict_in_generate``."""
        ring_capacity(max_length, self.config.rope_max_seq_len)
        out = decode(input_ids, self, max_length, top_k=top_k, top_p=top_p, min_p=min_p, temperature=temperature,
                     output_scores=output_scores, **kwargs)
        if not output_scores:
            out.scores = None
        return out if return_dict_in_generate else out.sequences
