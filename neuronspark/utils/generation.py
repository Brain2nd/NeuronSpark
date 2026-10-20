# Adapted from mamba_ssm 2.3.2.post1, mamba_ssm/utils/generation.py (sha256
# ae59067ba50941640e34129149daa55dd13b98c25db47d8e2cc432d876ba34f4), Copyright (c) 2023, Albert Gu, Tri Dao.
# Licensed under the Apache License, Version 2.0 (the license text is in neuronspark/ops/triton/mamba/LICENSE; see
# also NOTICE).
# Modified by NeuronSpark: the inference cache is neuronspark.utils.cache.InferenceParams (ring capacity a multiple of
# 512; device counters advanced by the model, so the decode loop passes no positions); the graph warm-up saves and
# restores the cache; the graph cache is re-captured when the batch size, the ring capacity, the SD state dtype or any
# weight changes; only the single-token step is captured; min-p filtering is applied to probabilities; sampling runs
# on fp32 logits; no transformers dependency (own output record, duck-typed streamer); torch.no_grad instead of
# inference mode, so the cache can be reused by ordinary no-grad calls.
"""Generation: prefill, single-token decode steps replayed from a CUDA graph (default) or run eagerly, and greedy
decoding or sampling.

``decode(input_ids, model, max_length, ...)``:

1. the cache (``InferenceParams``) is reset to the row start: with ``cg=True`` the cached one of
   :func:`update_graph_cache` (allocated once per batch size / ring capacity / SD state dtype, with the bf16 weight
   copies of ``model.prepare_inference()`` and one captured decode step), else a fresh one;
2. the prompt runs as one eager prefill call (``num_last_tokens=1``), the logits of its last position give the first
   token;
3. every further token is one T = 1 step: ``graph.replay()`` on the static input with ``cg=True`` (the captured step
   advances the device counters and every carried state in place), the eager step otherwise.

Every row of a batch advances together, so the prompts of one call have equal length (B, T).  ``max_length`` is the
total length (prompt + new tokens) and must fit the ring capacity: the cache holds ``ceil(max_length / 512) * 512``
slots, at most ``rope_max_seq_len`` (8192).  The default sampling arguments are greedy decoding (argmax, the first index
on ties).  ``teacher_outputs`` (B, >= max_length) forces the tokens (the teacher-forced form used by the tests).
"""
from __future__ import annotations

import gc
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional, Sequence, Tuple, Union

import torch

from neuronspark.utils.cache import InferenceParams, RING_MULTIPLE, resolve_sd_state_dtype

__all__ = [
    "DecodeOutput",
    "DecodingCGCache",
    "GenerationMixin",
    "capture_graph",
    "decode",
    "modify_logit_for_repetition_penalty",
    "modify_logits_for_min_p_filtering",
    "modify_logits_for_top_k_filtering",
    "modify_logits_for_top_p_filtering",
    "ring_capacity",
    "sample",
    "update_graph_cache",
]


# ================================================================================================ sampling
def modify_logits_for_min_p_filtering(logits: torch.Tensor, min_p: float) -> None:
    """In place: set to -inf the logits whose probability is below ``min_p`` times the row's largest probability."""
    if min_p <= 0.0 or min_p >= 1.0:
        return
    probs = torch.softmax(logits, dim=-1)
    indices_to_remove = probs < probs.amax(dim=-1, keepdim=True) * min_p
    logits.masked_fill_(indices_to_remove, float("-inf"))


def modify_logits_for_top_k_filtering(logits: torch.Tensor, top_k: int) -> None:
    """In place: set to -inf the logits below the k-th largest of their row."""
    indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
    logits.masked_fill_(indices_to_remove, float("-inf"))


def modify_logits_for_top_p_filtering(logits: torch.Tensor, top_p: float) -> None:
    """In place: set to -inf the logits outside the smallest set whose probability reaches ``top_p``."""
    if top_p <= 0.0 or top_p >= 1.0:
        return
    sorted_logits, sorted_indices = torch.sort(logits, descending=False)
    cumulative_probs = sorted_logits.softmax(dim=-1).cumsum(dim=-1)
    # remove the tokens whose cumulative probability stays within 1 - top_p (the tokens with 0 are kept)
    sorted_indices_to_remove = cumulative_probs <= (1 - top_p)
    indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
    logits.masked_fill_(indices_to_remove, float("-inf"))


def modify_logit_for_repetition_penalty(logits: torch.Tensor, prev_output_tokens: torch.Tensor,
                                        repetition_penalty: float = 1.0) -> torch.Tensor:
    """Repetition penalty (https://arxiv.org/abs/1909.05858) on ``logits`` (B, V) for the tokens ``prev_output_tokens``
    (B, L), in place."""
    if repetition_penalty == 1.0:
        return logits
    score = torch.gather(logits, 1, prev_output_tokens)
    # a negative score is multiplied by the penalty, a positive one divided, to lower the token's probability
    score = torch.where(score < 0, score * repetition_penalty, score / repetition_penalty)
    logits.scatter_(1, prev_output_tokens, score)
    return logits


def sample(logits: torch.Tensor, top_k: int = 1, top_p: float = 0.0, min_p: float = 0.0,
           temperature: float = 1.0) -> torch.Tensor:
    """Next tokens (B,) from ``logits`` (B, V): ``top_k == 1`` is greedy (argmax, first index on ties); ``top_k > 0``
    samples among the top-k (then top-p); ``top_k == 0`` samples from the whole vocabulary after min-p and top-p
    filtering."""
    if top_k == 1:                                     # greedy decoding
        return logits.argmax(dim=-1)
    if top_p > 1.0:
        raise ValueError("top-p should be in (0, 1].")
    logits = logits.float()
    if top_k > 0:
        top_k = min(top_k, logits.size(-1))
        logits_top, indices = torch.topk(logits, top_k, dim=-1)
        if temperature != 1.0:
            logits_top /= temperature
        modify_logits_for_top_p_filtering(logits_top, top_p)
        return indices[
            torch.arange(indices.shape[0], device=indices.device),
            torch.multinomial(torch.softmax(logits_top, dim=-1), num_samples=1).squeeze(dim=-1),
        ]
    logits_top = logits / temperature if temperature != 1.0 else logits.clone()
    modify_logits_for_min_p_filtering(logits_top, min_p)
    modify_logits_for_top_p_filtering(logits_top, top_p)
    return torch.multinomial(torch.softmax(logits_top, dim=-1), num_samples=1).squeeze(dim=-1)


# ================================================================================================ cache helpers
def ring_capacity(max_length: int, rope_max_seq_len: int) -> int:
    """The ring capacity that holds ``max_length`` tokens: ``max_length`` rounded up to a multiple of 512; raises
    beyond ``rope_max_seq_len``."""
    n = int(max_length)
    if n <= 0:
        raise ValueError(f"max_length must be positive, got {max_length}")
    W = -(-n // RING_MULTIPLE) * RING_MULTIPLE
    if W > int(rope_max_seq_len):
        raise ValueError(f"max_length {n} exceeds the longest supported row ({int(rope_max_seq_len)} tokens: the RoPE "
                         "table length bounds the ring capacity)")
    return W


def _weight_signature(model) -> Tuple[Tuple[int, int], ...]:
    """(storage address, version counter) of every parameter: changes when a weight is replaced or written."""
    return tuple((int(p.data_ptr()), int(p._version)) for p in model.parameters())


@dataclass
class DecodeOutput:
    """``sequences`` (B, L) int64: the prompt followed by the generated tokens; ``scores``: the per-step logits (B, V)
    when requested."""

    sequences: torch.Tensor
    scores: Optional[Tuple[torch.Tensor, ...]] = None


@dataclass
class DecodingCGCache:
    """The CUDA-graph decode state of one model: the cache the graphs read, their memory pool, one captured step per
    (batch size, decoding length) and the key that decides re-capture."""

    max_batch_size: int = 0
    max_seqlen: int = 0
    device: Optional[torch.device] = None
    sd_state_dtype: Optional[torch.dtype] = None
    callables: Dict[Tuple[int, int], Callable] = field(default_factory=dict)
    mempool: Any = None
    inference_params: Optional[InferenceParams] = None
    run: Optional[Callable] = None
    weight_signature: Optional[Tuple[Tuple[int, int], ...]] = None
    # the bf16 weight copies the captured graphs read (kept alive with the graphs)
    keepalive: list = field(default_factory=list)


# ================================================================================================ decode loop
@torch.no_grad()
def decode(
    input_ids: torch.Tensor,
    model,
    max_length: int,
    top_k: int = 1,
    top_p: float = 0.0,
    min_p: float = 0.0,
    temperature: float = 1.0,
    repetition_penalty: float = 1.0,
    eos_token_id: Optional[int] = None,
    teacher_outputs: Optional[torch.Tensor] = None,
    cg: bool = True,
    enable_timing: bool = False,
    output_scores: bool = False,
    streamer=None,
    sd_state_dtype: Optional[Union[str, torch.dtype]] = None,
) -> DecodeOutput:
    """Greedy decoding or sampling (see the module docstring).

    ``input_ids`` (B, T) int64 prompts of equal length; ``max_length`` the total length; ``eos_token_id``: stop once
    every row produced it; ``teacher_outputs`` (B, >= max_length): take the tokens from it instead of sampling;
    ``cg``: replay the decode step from a CUDA graph; ``output_scores``: return the per-step logits;
    ``streamer``: an object with ``put(tokens)`` / ``end()``; ``sd_state_dtype``: the cache's SD state storage
    (default ``config.inference_sd_state_dtype``).
    """
    if input_ids.dim() != 2:
        raise ValueError(f"decode: input_ids must be (B, T), got {tuple(input_ids.shape)}")
    batch_size, seqlen_og = (int(s) for s in input_ids.shape)
    max_length = int(max_length)
    if max_length <= seqlen_og:
        raise ValueError(f"decode: max_length {max_length} is the total length and must exceed the prompt length "
                         f"{seqlen_og}")
    W = ring_capacity(max_length, model.config.rope_max_seq_len)
    if streamer is not None:
        streamer.put(input_ids.cpu())

    teacher_output_len = int(teacher_outputs.shape[1]) if teacher_outputs is not None else 0
    if cg:
        model._decoding_cache = update_graph_cache(model, getattr(model, "_decoding_cache", None), batch_size,
                                                   seqlen_og, max_length, sd_state_dtype=sd_state_dtype)
        inference_params = model._decoding_cache.inference_params
        inference_params.reset()
    else:
        inference_params = model.allocate_inference_cache(batch_size, W, sd_state_dtype=sd_state_dtype)

    def get_logits(input_ids, inference_params):
        # the model call (or the replayed step) advances inference_params.seqlen_offset by the call's length
        decoding = inference_params.seqlen_offset > 0
        if not cg or not decoding:
            logits = model(input_ids, inference_params=inference_params, num_last_tokens=1).logits.squeeze(dim=1)
        else:
            logits = model._decoding_cache.run(input_ids).squeeze(dim=1)
        return logits

    def sample_tokens(logits, inference_params):
        if teacher_outputs is None or teacher_output_len <= inference_params.seqlen_offset:
            token = sample(logits, top_k=top_k, top_p=top_p, min_p=min_p, temperature=temperature)
        else:
            token = teacher_outputs[:, inference_params.seqlen_offset]
        return token.to(torch.int64).unsqueeze(1)

    def should_stop(current_token, inference_params):
        if inference_params.seqlen_offset == 0:
            return False
        if eos_token_id is not None and (current_token == eos_token_id).all():
            return True
        if inference_params.seqlen_offset >= max_length - 1:
            return True
        return False

    if enable_timing:
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
    scores, sequences = [], [input_ids]
    sequences_cat = input_ids
    while not should_stop(sequences[-1], inference_params):
        logits = get_logits(sequences[-1], inference_params)
        if output_scores:
            scores.append(logits.clone())
        if repetition_penalty == 1.0:
            sampled_tokens = sample_tokens(logits, inference_params)
        else:
            logits = modify_logit_for_repetition_penalty(logits.float(), sequences_cat, repetition_penalty)
            sampled_tokens = sample_tokens(logits, inference_params)
            sequences_cat = torch.cat([sequences_cat, sampled_tokens], dim=1)
        sequences.append(sampled_tokens)
        if streamer is not None:
            streamer.put(sampled_tokens.cpu())
    if streamer is not None:
        streamer.end()
    if enable_timing:
        end.record()
        torch.cuda.synchronize()
        print(f"Prompt processing + decoding time: {(start.elapsed_time(end)):.0f}ms")
    return DecodeOutput(sequences=torch.cat(sequences, dim=1), scores=tuple(scores))


class GenerationMixin:
    """``generate`` for a model with ``allocate_inference_cache`` and the mamba-style ``forward(input_ids,
    inference_params, num_last_tokens)``; the model class provides ``allocate_inference_cache`` and ``config``."""

    def generate(
        self,
        input_ids: torch.Tensor,
        max_length: int,
        top_k: int = 1,
        top_p: float = 0.0,
        min_p: float = 0.0,
        temperature: float = 1.0,
        return_dict_in_generate: bool = False,
        output_scores: bool = False,
        **kwargs,
    ):
        """:func:`decode` with ``cg=True`` by default.  ``max_length`` is the total length (prompt + new tokens): it must
        exceed the prompt length and fit the longest supported row (``rope_max_seq_len``).  Returns the sequences
        (B, L), or the :class:`DecodeOutput` with ``return_dict_in_generate``."""
        ring_capacity(max_length, self.config.rope_max_seq_len)
        output = decode(input_ids, self, max_length, top_k=top_k, top_p=top_p, min_p=min_p, temperature=temperature,
                        output_scores=output_scores, **kwargs)
        if not output_scores:
            output.scores = None
        return output if return_dict_in_generate else output.sequences


# ================================================================================================ CUDA-graph decode
@torch.no_grad()
def update_graph_cache(
    model,
    cache: Optional[DecodingCGCache],
    batch_size: int,
    seqlen_og: int,
    max_seqlen: int,
    decoding_seqlens: Sequence[int] = (1,),
    sd_state_dtype: Optional[Union[str, torch.dtype]] = None,
    n_warmups: int = 2,
) -> DecodingCGCache:
    """The graph cache for ``batch_size`` rows of total length ``max_seqlen`` (``seqlen_og``: the prompt length, kept
    for mamba's signature).  The cache, its memory pool and the graphs are (re)built when the device, the SD state
    dtype or the batch size changes, when the ring capacity must grow, or when any weight was replaced or written since
    the capture; then ``model.prepare_inference()`` makes the bf16 weight copies and one decode step is captured per
    decoding length (only the single-token step is supported)."""
    if tuple(int(s) for s in decoding_seqlens) != (1,):
        raise ValueError(f"update_graph_cache: only the single-token decode step is captured, got decoding_seqlens "
                         f"{tuple(decoding_seqlens)}")
    if cache is None:
        cache = DecodingCGCache()
    param_example = next(iter(model.parameters()))
    device = param_example.device
    if device.type != "cuda":
        raise RuntimeError(f"update_graph_cache: CUDA-graph decoding needs the model on a CUDA device, got {device}")
    dtype = resolve_sd_state_dtype(model.config.inference_sd_state_dtype if sd_state_dtype is None else sd_state_dtype)
    W = ring_capacity(max_seqlen, model.config.rope_max_seq_len)
    if (
        (device, dtype) != (cache.device, cache.sd_state_dtype)
        or batch_size != cache.max_batch_size
        or W > cache.max_seqlen
        or _weight_signature(model) != cache.weight_signature
    ):  # invalidate the cache
        cache.callables = {}
        cache.mempool = None
        cache.inference_params = None
        cache.run = None
        cache.keepalive = []
        gc.collect()
        model.prepare_inference()
        cache.device, cache.sd_state_dtype = device, dtype
        cache.max_batch_size, cache.max_seqlen = int(batch_size), W
        cache.inference_params = model.allocate_inference_cache(batch_size, W, sd_state_dtype=dtype)
        cache.mempool = torch.cuda.graphs.graph_pool_handle()
        cache.weight_signature = _weight_signature(model)
        cache.keepalive = [w for w in (getattr(p, "_ns_bf16", None) for p in model.parameters()) if w is not None]
    for decoding_seqlen in decoding_seqlens:
        if (batch_size, decoding_seqlen) not in cache.callables:
            cache.callables[batch_size, decoding_seqlen] = capture_graph(
                model,
                cache.inference_params,
                batch_size,
                cache.max_seqlen,
                decoding_seqlen=decoding_seqlen,
                mempool=cache.mempool,
                n_warmups=n_warmups,
            )

    def dispatch(input_ids):
        batch_size, decoding_seqlen = (int(s) for s in input_ids.shape[:2])
        return cache.callables[batch_size, decoding_seqlen](input_ids)

    cache.run = dispatch
    cache.inference_params.seqlen_offset = 0          # reset so it is not confusing
    return cache


@torch.no_grad()
def capture_graph(model, inference_params: InferenceParams, batch_size: int, max_seqlen: int,
                  decoding_seqlen: int = 1, mempool=None, n_warmups: int = 2) -> Callable:
    """Capture one decode step of ``model`` over ``inference_params`` and return ``run(input_ids) -> logits``.

    The warm-up steps run on a side stream (they compile the kernels and build every first-call cache outside the
    capture) and advance the cache, so the cache is saved before them and restored after them; the capture records the
    step without executing it.  ``run`` copies the token ids into the static input, replays the graph (the cache and
    the device counters advance in place), advances the host count and returns a copy of the static logits
    (B, decoding_seqlen, V)."""
    if int(decoding_seqlen) != 1:
        raise ValueError("capture_graph: only the single-token decode step is captured")
    if int(inference_params.max_batch_size) != int(batch_size):
        raise ValueError(f"capture_graph: a cache of {inference_params.max_batch_size} rows for batch {batch_size}")
    device = next(iter(model.parameters())).device
    input_ids = torch.full((batch_size, decoding_seqlen), 0, dtype=torch.long, device=device)
    snapshot = inference_params.snapshot()

    # warm-up before capture
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(n_warmups):
            logits = model(input_ids, inference_params=inference_params, num_last_tokens=decoding_seqlen).logits
        s.synchronize()
        # needed for correctness when graph launches and non-captured launches must not overlap across ranks
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.barrier()
    torch.cuda.current_stream().wait_stream(s)
    inference_params.restore(snapshot)
    del snapshot

    # capture (a side stream is made current inside the context automatically); nothing executes on the device, only
    # the host count advanced by the model call is restored afterwards
    seqlen_offset_og = inference_params.seqlen_offset
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, pool=mempool):
        logits = model(input_ids, inference_params=inference_params, num_last_tokens=decoding_seqlen).logits
    inference_params.seqlen_offset = seqlen_offset_og

    def run(new_input_ids):
        inference_params.check_room(decoding_seqlen)
        input_ids.copy_(new_input_ids)
        graph.replay()
        inference_params.seqlen_offset += decoding_seqlen
        return logits.clone()

    return run
