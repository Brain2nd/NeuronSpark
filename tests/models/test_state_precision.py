"""The one numerics option: the storage dtype of the SD state in the inference cache (``inference_sd_state_dtype``,
``"bfloat16"`` by default, ``"float32"`` as the alternative).

* option and cache (no GPU, meta-device build): the config values, the allocation default and override, the one-line
  announcement, the cache's dtype consistency check, the capacity / batch refusals;
* the decode step operator on the checkpoint's SD layer weights (``--ckpt``, GPU) with either state dtype, at every
  decode batch size (1, 2, 4, 8, 16, 32: every launch route of the step kernels), against its pure-torch reference: output and
  the updated states (the state stays in its storage dtype);
* model level: one prefill from the zero state gives the same logits with either storage, and the bf16 state is the
  fp32 state rounded once; decode with either dtype (eager and CUDA graph), the graph decode against the eager decode
  by the rule, the drift between the two storages reported per step; ``from_pretrained`` with the option.

Every assertion on floating values is the per-operator rule of ``neuronspark.utils.testing`` (matched ratio 1.0).
"""
from __future__ import annotations

import dataclasses
import gc

import pytest
import torch

import neuronspark.utils.cache as cache_mod
from neuronspark import ops
from neuronspark.models.config_neuronspark import NeuronSparkConfig
from neuronspark.models.neuronspark_lm import NeuronSparkLMHeadModel
from neuronspark.utils.cache import InferenceParams, SDState, resolve_sd_state_dtype
from neuronspark.utils.generation import decode
from neuronspark.utils.precision import regime
from neuronspark.utils.testing import assert_match, match_stats, summarize

CFG = NeuronSparkConfig()
SMALL_VOCAB = 256
DECODE_SIZES = (1, 2, 4, 8, 16, 32)
DTYPES = {"bfloat16": torch.bfloat16, "float32": torch.float32}


def _prompts(B: int, T: int, seed: int) -> torch.Tensor:
    g = torch.Generator(device="cpu").manual_seed(seed)
    return (1000 + torch.randint(0, SMALL_VOCAB, (B, T), generator=g)).cuda()


def _rel_rms(a: torch.Tensor, b: torch.Tensor) -> float:
    a, b = a.float(), b.float()
    return float((a - b).pow(2).mean().sqrt() / b.pow(2).mean().sqrt().clamp_min(1e-30))


# ================================================================================================ option and cache
def test_option_values():
    assert CFG.inference_sd_state_dtype == "bfloat16"
    assert NeuronSparkConfig(inference_sd_state_dtype="float32").inference_sd_state_dtype == "float32"
    with pytest.raises(ValueError):
        NeuronSparkConfig(inference_sd_state_dtype="float16")
    assert resolve_sd_state_dtype("bfloat16") == torch.bfloat16
    assert resolve_sd_state_dtype("float32") == torch.float32
    assert resolve_sd_state_dtype(torch.float32) == torch.float32
    with pytest.raises(ValueError):
        resolve_sd_state_dtype(torch.float16)


def test_cache_allocation(monkeypatch, capsys):
    monkeypatch.setattr(cache_mod, "_ANNOUNCED", set())
    model = NeuronSparkLMHeadModel(CFG, device="meta", init="empty")
    ip = model.allocate_inference_cache(2, 2048)
    assert ip.sd_state_dtype == torch.bfloat16
    sd_ids = [i for i in range(CFG.num_layers) if not CFG.is_memory_layer(i)]
    assert sorted(i for i, st in ip.layers.items() if isinstance(st, SDState)) == sd_ids
    for i in sd_ids:
        st = ip.layers[i]
        assert st.ssm.dtype == torch.bfloat16 and tuple(st.ssm.shape) == (2, 144, 64, 128)
        assert st.conv.dtype == torch.float32 and tuple(st.conv.shape) == (2, 9472, 3)
    err = capsys.readouterr().err
    assert err.count("inference SD state stored in bfloat16") == 1
    ip32 = model.allocate_inference_cache(2, 2048, sd_state_dtype="float32")
    assert ip32.sd_state_dtype == torch.float32
    assert all(ip32.layers[i].ssm.dtype == torch.float32 for i in sd_ids)
    assert capsys.readouterr().err.count("inference SD state stored in float32") == 1
    model.allocate_inference_cache(1, 512)                       # announced once per process and dtype
    assert "inference SD state stored" not in capsys.readouterr().err
    cfg32 = dataclasses.replace(CFG, inference_sd_state_dtype="float32")
    m32 = NeuronSparkLMHeadModel(cfg32, device="meta", init="empty")
    assert m32.allocate_inference_cache(1, 512).sd_state_dtype == torch.float32
    # a cache whose SD states do not have the declared dtype is refused
    layers = dict(ip.layers)
    layers[sd_ids[0]] = ip32.layers[sd_ids[0]]
    with pytest.raises(ValueError):
        InferenceParams(max_seqlen=ip.max_seqlen, max_batch_size=2, pos=ip.pos, ring_ptr=ip.ring_ptr,
                        sd_state_dtype=torch.bfloat16, layers=layers, lexical=ip.lexical, induction=ip.induction,
                        copy=ip.copy, announce=False)
    with pytest.raises(ValueError):
        model.allocate_inference_cache(2, 1000)                  # not a multiple of 512
    with pytest.raises(ValueError):
        model.allocate_inference_cache(2, 8192 + 512)            # beyond the RoPE table
    with pytest.raises(ValueError):
        model.allocate_inference_cache(33, 512)                  # beyond the decode batch tables


# ================================================================================================ checkpoint (GPU)
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


@pytest.mark.cuda
@pytest.mark.ckpt
@pytest.mark.parametrize("dtype_name", list(DTYPES))
@pytest.mark.parametrize("b", DECODE_SIZES)
def test_sd_step_routes_both_state_dtypes(model, b, dtype_name):
    dtype = DTYPES[dtype_name]
    mx = model.backbone.layers[0].mixer_a
    g = torch.Generator(device="cuda").manual_seed(100 + b)
    dev = torch.device("cuda")
    with torch.no_grad(), regime("infer"):
        hn = torch.randn((b, CFG.D), device=dev, generator=g).to(torch.bfloat16)
        proj = ops.KERNELS.linear(hn, mx.in_proj.weight)                         # (b, 18832) bf16
        conv = torch.randn((b, mx.conv_dim, 3), device=dev, generator=g).to(torch.bfloat16).float()
        ssm = (0.1 * torch.randn((b, mx.nheads, mx.head_dim, mx.state_size), device=dev, generator=g)).to(dtype)
        conv_k, ssm_k = conv.clone(), ssm.clone()
        conv_r, ssm_r = conv.clone(), ssm.clone()
        args = (mx.conv_weight, mx.conv_bias, mx.A(), mx.dt_bias, mx.D, mx.theta(), mx.gamma, mx.norm_eps)
        xs_k = ops.KERNELS.sd_step_core(proj, conv_k, ssm_k, *args)
        xs_r = ops.REFS.sd_step_core(proj, conv_r, ssm_r, *args)
    assert ssm_k.dtype == dtype and ssm_r.dtype == dtype
    assert_match(xs_k, xs_r, f"b{b} {dtype_name} xs")
    assert_match(ssm_k, ssm_r, f"b{b} {dtype_name} ssm state")
    assert_match(conv_k, conv_r, f"b{b} {dtype_name} conv state")


@pytest.mark.cuda
@pytest.mark.ckpt
@pytest.mark.parametrize("b", [1, 8])
def test_prefill_stores_the_state_once(model, b):
    """From the zero state a prefill computes the same function with either storage; the bf16 cache holds the fp32
    final state rounded once."""
    ids = _prompts(b, 300, seed=200 + b)                                         # a partial last scan chunk
    with torch.no_grad():
        ip16 = model.allocate_inference_cache(b, 2048, sd_state_dtype="bfloat16")
        ip32 = model.allocate_inference_cache(b, 2048, sd_state_dtype="float32")
        lg16 = model(ids, inference_params=ip16, num_last_tokens=1).logits
        lg32 = model(ids, inference_params=ip32, num_last_tokens=1).logits
    assert_match(lg16, lg32, "prefill logits")
    for i, st16 in ip16.layers.items():
        st32 = ip32.layers[i]
        if isinstance(st16, SDState):
            assert st16.ssm.dtype == torch.bfloat16 and st32.ssm.dtype == torch.float32
            assert_match(st16.ssm, st32.ssm.to(torch.bfloat16), f"layer {i} ssm")
            assert_match(st16.conv, st32.conv, f"layer {i} conv")


@pytest.mark.cuda
@pytest.mark.ckpt
@pytest.mark.parametrize("b", DECODE_SIZES)
def test_decode_with_either_storage(model, b):
    prompt = _prompts(b, 64, seed=300 + b)
    L = 64 + 16
    ref16 = decode(prompt, model, L, cg=False, output_scores=True, sd_state_dtype="bfloat16")
    eager32 = decode(prompt, model, L, cg=False, output_scores=True, sd_state_dtype="float32",
                     teacher_outputs=ref16.sequences)
    graph32 = decode(prompt, model, L, cg=True, output_scores=True, sd_state_dtype="float32",
                     teacher_outputs=ref16.sequences)
    assert model._decoding_cache.inference_params.sd_state_dtype == torch.float32
    for step, (g, e) in enumerate(zip(graph32.scores, eager32.scores)):
        assert_match(g, e, f"b{b} fp32 state step{step}: graph vs eager")
    drift, results = [], []
    for s32, s16 in zip(eager32.scores, ref16.scores):
        assert bool(torch.isfinite(s32.float()).all()) and bool(torch.isfinite(s16.float()).all())
        drift.append(_rel_rms(s16, s32))
        results.append(match_stats(s16, s32))
    print(f"[report] b={b}: bf16 vs fp32 SD state, decode logits rel_rms max {max(drift):.3e} mean "
          f"{sum(drift) / len(drift):.3e}; rule summary {summarize(results)}")


@pytest.mark.cuda
@pytest.mark.ckpt
def test_from_pretrained_option(ckpt_dir):
    m = NeuronSparkLMHeadModel.from_pretrained(ckpt_dir, device="cuda", inference_sd_state_dtype="float32")
    try:
        assert m.config.inference_sd_state_dtype == "float32"
        ip = m.allocate_inference_cache(1, 512)
        assert ip.sd_state_dtype == torch.float32
        assert m.allocate_inference_cache(1, 512, sd_state_dtype="bfloat16").sd_state_dtype == torch.bfloat16
    finally:
        del m
        gc.collect()
        torch.cuda.empty_cache()
