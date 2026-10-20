"""utils.generation and the model's inference regimes, on the NeuronSpark checkpoint (``--ckpt``, GPU).

* CUDA-graph decode against eager decode: the same tokens fed to both (teacher forcing with the eager run's tokens),
  the per-step logits by the per-operator rule (decode batch sizes 1, 2, 8, 32: the ring-kernel and torch-chain routes
  and every dendritic-layer step route); the free-running greedy agreement is reported;
* a chunked prefill (4 calls of 256 tokens) against one prefill of 1024 tokens: the cache counters and the integer state
  exactly, the logits and the floating state reported (a one-ulp difference between call lengths is amplified by the
  spikes and the top-k selection);
* teacher-forced graph decode against one whole-sequence prefill (per-step logits), and the whole-row regime against
  the prefill: reported (the regimes run different kernels: the comparison is informative);
* the generate API: lengths, the length refusals, the graph cache reuse and its re-capture after a weight write or a
  batch-size change.

Every assertion on floating values is the per-operator rule of ``neuronspark.utils.testing`` (matched ratio 1.0).
"""
from __future__ import annotations

import gc

import pytest
import torch

from neuronspark.models.neuronspark_lm import NeuronSparkLMHeadModel
from neuronspark.utils.generation import decode, ring_capacity
from neuronspark.utils.testing import assert_match, match_stats, summarize

pytestmark = [pytest.mark.cuda, pytest.mark.ckpt]

SMALL_VOCAB = 256


def _prompts(B: int, T: int, seed: int) -> torch.Tensor:
    """Token ids drawn from a few ids (repeats exercise the copy detector and the induction bias)."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    return (1000 + torch.randint(0, SMALL_VOCAB, (B, T), generator=g)).cuda()


@pytest.fixture(scope="module")
def model(request):
    path = request.config.getoption("--ckpt")
    if not path:
        pytest.skip("needs --ckpt")
    if not torch.cuda.is_available():
        pytest.skip("needs a CUDA GPU")
    m = NeuronSparkLMHeadModel.from_pretrained(path, device="cuda")
    yield m
    m.release_inference()
    del m
    gc.collect()
    torch.cuda.empty_cache()


def _rel_rms(a: torch.Tensor, b: torch.Tensor) -> float:
    a, b = a.float(), b.float()
    return float((a - b).pow(2).mean().sqrt() / b.pow(2).mean().sqrt().clamp_min(1e-30))


# ================================================================================================ graph vs eager
@pytest.mark.parametrize("b", [1, 2, 8, 32])
def test_cg_decode_matches_eager(model, b):
    prompt = _prompts(b, 64, seed=10 + b)
    max_length = 64 + 32
    eager = decode(prompt, model, max_length, cg=False, output_scores=True)
    assert tuple(eager.sequences.shape) == (b, max_length)
    graph = decode(prompt, model, max_length, cg=True, output_scores=True, teacher_outputs=eager.sequences)
    assert len(graph.scores) == len(eager.scores) == max_length - 64
    for step, (got, ref) in enumerate(zip(graph.scores, eager.scores)):
        assert_match(got, ref, f"b{b} step{step} logits")
    assert graph.sequences.tolist() == eager.sequences.tolist()
    free = decode(prompt, model, max_length, cg=True)
    agree = float((free.sequences[:, 64:] == eager.sequences[:, 64:]).float().mean())
    print(f"[report] b={b}: free-running graph decode vs eager greedy token agreement {agree:.4f}")


# ================================================================================================ chunked prefill
@pytest.mark.parametrize("b", [1, 8])
def test_chunked_prefill_matches_single(model, b):
    ids = _prompts(b, 1024, seed=20 + b)
    with torch.no_grad():
        ip1 = model.allocate_inference_cache(b, 2048)
        whole = model(ids, inference_params=ip1).logits                       # (b, 1024, V)
        ip2 = model.allocate_inference_cache(b, 2048)
        parts = [model(ids[:, i:i + 256], inference_params=ip2).logits for i in range(0, 1024, 256)]
    chunked = torch.cat(parts, dim=1)
    # reported: a call of 256 and a call of 1024 tokens round some GEMM / scan sums in a different order (one bf16 ulp in the
    # first layer), and the PLIF spikes and the PKM top-k selection amplify that through the depth -- the logits agree in
    # distribution, not element by element
    print(f"[report] b{b} chunked (4 x 256) vs one prefill (1024) logits: {match_stats(chunked, whole, f'b{b} chunked prefill logits')}")
    del parts, chunked, whole
    assert ip1.seqlen_offset == ip2.seqlen_offset == 1024
    for (n1, t1), (n2, t2) in zip(ip1.tensors(), ip2.tensors()):
        assert n1 == n2
        assert t1.dtype == t2.dtype, n1
        if t1.is_floating_point():
            # empty ring slots hold +inf clocks: the pattern of finite entries (exact), then the finite values (reported,
            # for the same reason as the logits)
            fin = torch.isfinite(t1)
            assert_match(torch.isfinite(t2).float(), fin.float(), f"{n1} (finite pattern)")
            print(f"[report] {n1}: {match_stats(torch.where(fin, t2, torch.zeros_like(t2)), torch.where(fin, t1, torch.zeros_like(t1)), n1)}")
        else:
            assert t1.tolist() == t2.tolist(), n1


# ================================================================================================ reported comparisons
def test_teacher_forced_decode_vs_prefill_reported(model):
    ids = _prompts(1, 320, seed=30)
    with torch.no_grad():
        ip = model.allocate_inference_cache(1, 2048)
        full = model(ids, inference_params=ip).logits[0]                       # (320, V)
    out = decode(ids[:, :256], model, 320, cg=True, output_scores=True, teacher_outputs=ids)
    assert len(out.scores) == 64
    results = []
    for i, sc in enumerate(out.scores):
        assert bool(torch.isfinite(sc.float()).all())
        results.append(match_stats(sc[0], full[255 + i], f"position {255 + i}"))
    s = summarize(results)
    worst = max(_rel_rms(sc[0], full[255 + i]) for i, sc in enumerate(out.scores))
    print(f"[report] teacher-forced graph decode vs one prefill: {s}, max rel_rms {worst:.3e}")


def test_whole_row_vs_prefill_reported(model):
    ids = _prompts(1, 1024, seed=40)
    with torch.no_grad():
        row = model(ids).logits[0]
        ip = model.allocate_inference_cache(1, 2048)
        pre = model(ids, inference_params=ip).logits[0]
    assert bool(torch.isfinite(row.float()).all()) and bool(torch.isfinite(pre.float()).all())
    res = match_stats(pre, row, "prefill vs whole row")
    print(f"[report] {res}; rel_rms {_rel_rms(pre, row):.3e}")


# ================================================================================================ generate API
def test_generate_api_and_graph_cache(model):
    prompt = _prompts(2, 16, seed=50)
    seq = model.generate(prompt, max_length=48)
    assert tuple(seq.shape) == (2, 48)
    assert seq[:, :16].tolist() == prompt.tolist()
    out = model.generate(prompt, max_length=48, return_dict_in_generate=True, output_scores=True)
    assert len(out.scores) == 32 and tuple(out.scores[0].shape) == (2, model.config.vocab_size)
    assert tuple(out.sequences.shape) == (2, 48) and out.sequences[:, :16].tolist() == prompt.tolist()
    with pytest.raises(ValueError):
        model.generate(prompt, max_length=16)                           # not longer than the prompt
    with pytest.raises(ValueError):
        model.generate(prompt, max_length=model.config.rope_max_seq_len + 1)
    assert ring_capacity(48, 8192) == 512 and ring_capacity(513, 8192) == 1024
    cache = model._decoding_cache
    run = cache.callables[(2, 1)]
    model.generate(prompt, max_length=40)                               # same batch, smaller length: reused
    assert model._decoding_cache is cache and cache.callables[(2, 1)] is run
    with torch.no_grad():
        model.backbone.norm.weight.mul_(1.0)                           # a weight write (same values)
    model.generate(prompt, max_length=40)
    assert cache.callables[(2, 1)] is not run                           # re-captured
    run2 = cache.callables[(2, 1)]
    model.generate(_prompts(3, 16, seed=51), max_length=40)             # another batch size: re-allocated
    assert (3, 1) in cache.callables and (2, 1) not in cache.callables and cache.max_batch_size == 3
    assert run2 is not cache.callables[(3, 1)]
    eager = decode(prompt, model, 40, cg=False)                          # eager decode after graph decode
    graph = decode(prompt, model, 40, cg=True)
    assert tuple(eager.sequences.shape) == tuple(graph.sequences.shape) == (2, 40)
