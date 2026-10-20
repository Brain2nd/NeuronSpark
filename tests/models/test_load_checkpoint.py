"""models: checkpoint layout, loading, verification and saving.

* layout (no GPU, meta-device build): the 678 state tensors of the 1MIX model (609 parameters / 1,270,342,292
  elements, 69 persistent buffers / 10,908 elements), key prefixes, shapes and dtypes; the configuration file round
  trip and the model-type check;
* checkpoint (``--ckpt``, GPU): a strict load of the 678 tensors; every loaded tensor against the file's tensor (dtype
  and shape equal, values by the per-operator rule); the post-load checks (lexical hash constants, value-residual
  anchors, calibrated product-key memories); no parameter is written by inference calls after the load (version
  counters unchanged; the PLIF ``b_eta`` values against the file); the verification refuses incomplete or altered
  checkpoints; ``save_pretrained`` round trips.

Every assertion on floating values is the per-operator rule of ``neuronspark.utils.testing`` (matched ratio 1.0);
integer buffers and version counters are compared as integers.
"""
from __future__ import annotations

import gc
import json
import math
import shutil
from pathlib import Path

import pytest
import torch

from neuronspark.models.config_neuronspark import MODEL_TYPE, NeuronSparkConfig
from neuronspark.models.neuronspark_lm import NeuronSparkLMHeadModel
from neuronspark.modules.pool_attention import value_residual_anchor
from neuronspark.utils.hf import (
    CONFIG_NAME,
    PREFIX,
    TOKENIZER_FILES,
    WEIGHTS_NAME,
    expected_state_layout,
    load_config,
    load_state_dict,
    verify_state_dict,
)
from neuronspark.utils.testing import assert_match

CFG = NeuronSparkConfig()
SMALL_VOCAB = 256


def _numel(shape) -> int:
    return int(math.prod(shape)) if shape else 1


def _prompts(B: int, T: int, seed: int, device) -> torch.Tensor:
    """Token ids drawn from a few ids (repeats exercise the copy detector and the induction bias)."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    ids = 1000 + torch.randint(0, SMALL_VOCAB, (B, T), generator=g)
    return ids.to(device)


def _file_keys(path: Path):
    from safetensors import safe_open

    with safe_open(str(path / WEIGHTS_NAME), framework="pt", device="cpu") as f:
        return list(f.keys())


def _compare_tensors(name: str, got: torch.Tensor, ref: torch.Tensor) -> None:
    assert got.dtype == ref.dtype, f"{name}: dtype {got.dtype} vs the file's {ref.dtype}"
    assert tuple(got.shape) == tuple(ref.shape), f"{name}: shape {tuple(got.shape)} vs the file's {tuple(ref.shape)}"
    if ref.is_floating_point():
        assert_match(got, ref.to(got.device), name)
    else:
        assert got.detach().cpu().tolist() == ref.detach().cpu().tolist(), f"{name}: integer values differ"


def _compare_with_file(state: dict, path: Path, device) -> int:
    """Every tensor of ``path``'s weights file against ``state`` (``backbone.*`` keys); returns the count."""
    from safetensors import safe_open

    n = 0
    with safe_open(str(path / WEIGHTS_NAME), framework="pt", device="cpu") as f:
        for key in f.keys():
            assert key in state, f"{key} not in the model"
            _compare_tensors(key, state[key], f.get_tensor(key).to(device))
            n += 1
    return n


# ================================================================================================ layout (no GPU)
def test_state_layout():
    params, buffers = expected_state_layout(CFG)
    assert len(params) == CFG.n_param_tensors_expected == 609
    assert sum(_numel(s) for s, _ in params.values()) == CFG.n_params_expected == 1_270_342_292
    assert len(buffers) == CFG.n_buffer_tensors_expected == 69
    assert sum(_numel(s) for s, _ in buffers.values()) == CFG.n_buffers_expected == 10_908
    assert len(params) + len(buffers) == CFG.n_state_tensors == 678
    assert all(k.startswith(PREFIX) for k in list(params) + list(buffers))
    assert all(dt == torch.float32 for _, dt in params.values())
    for k, (shape, dt) in buffers.items():
        want = torch.int64 if ".lexical_memory.hash_" in k else torch.float32
        assert dt == want, f"{k}: {dt}"
    # top level
    assert params["backbone.embed_tokens.weight"][0] == (151669, 1536)
    assert params["backbone.norm.weight"][0] == (1536,)
    assert params["backbone.balance_stack.projs.0.weight"][0] == (16, 1536)
    assert params["backbone.balance_stack.projs.1.weight"][0] == (32, 1536)
    assert [params[f"backbone.lexical_memory.tables.{i}"][0] for i in range(5)] == [
        (151669, 96), (262139, 16), (262139, 16), (344951, 16), (344951, 16)]
    for name in ("hash_mult", "hash_add", "hash_wrap"):
        assert buffers[f"backbone.lexical_memory.{name}"][0] == (4,)
    # per layer
    for i in range(CFG.num_layers):
        p = [k for k in params if k.startswith(f"backbone.layers.{i}.")]
        b = [k for k in buffers if k.startswith(f"backbone.layers.{i}.")]
        if CFG.is_memory_layer(i):
            assert (len(p), len(b)) == (70, 11), (i, len(p), len(b))
        else:
            assert (len(p), len(b)) == (10, 0), (i, len(p), len(b))
            assert params[f"backbone.layers.{i}.mixer_a.in_proj.weight"][0] == (18832, 1536)
            assert params[f"backbone.layers.{i}.mixer_a.W_out"][0] == (1536, 9216)
    assert params["backbone.layers.3.pkm_values"][0] == (16384, 768)
    assert buffers["backbone.layers.3.pool_value_residual_anchor"][0] == (1536,)


def test_config_round_trip(tmp_path):
    CFG.to_json(tmp_path / CONFIG_NAME)
    assert load_config(tmp_path) == CFG
    assert json.loads((tmp_path / CONFIG_NAME).read_text())["model_type"] == MODEL_TYPE
    (tmp_path / "other.json").write_text(json.dumps({"model_type": "llama"}))
    with pytest.raises(ValueError):
        load_config(tmp_path / "other.json")


# ================================================================================================ checkpoint (GPU)
@pytest.fixture(scope="module")
def loaded(request):
    path = request.config.getoption("--ckpt")
    if not path:
        pytest.skip("needs --ckpt")
    if not torch.cuda.is_available():
        pytest.skip("needs a CUDA GPU")
    model = NeuronSparkLMHeadModel.from_pretrained(path, device="cuda")
    yield model
    del model
    gc.collect()
    torch.cuda.empty_cache()


@pytest.mark.cuda
@pytest.mark.ckpt
def test_strict_load_matches_file(loaded, ckpt_dir):
    state = loaded.state_dict()
    assert len(state) == 678
    n = _compare_with_file(state, ckpt_dir, torch.device("cuda"))
    assert n == 678
    n_param = sum(1 for _ in loaded.parameters())
    assert n_param == 609
    assert sum(p.numel() for p in loaded.parameters()) == 1_270_342_292
    assert all(p.dtype == torch.float32 for p in loaded.parameters())


@pytest.mark.cuda
@pytest.mark.ckpt
def test_post_load_checks(loaded):
    bb = loaded.backbone
    bb.lexical_memory.check_hash_buffers()
    assert [int(v) for v in bb.lexical_memory.hash_wrap.tolist()] == [128000, 128000, 170027, 170027]
    mem = bb.memory_layers
    assert [l.layer_idx for l in mem] == list(CFG.memory_layer_ids)
    for l in mem:
        assert l.pkm_ready, l.layer_idx
        assert float(l.pkm_calibrated) == 1.0
        rank = CFG.memory_index(l.layer_idx)
        want = (torch.zeros(CFG.D_value) if rank == 0 else
                value_residual_anchor(CFG.memory_pool_value_residual_anchor[rank - 1], CFG.D_value, CFG.memory_heads))
        assert_match(l.pool_value_residual_anchor, want.to(l.pool_value_residual_anchor.device),
                     f"layers.{l.layer_idx}.pool_value_residual_anchor")
    frozen = [n for n, p in loaded.named_parameters() if not p.requires_grad]
    assert len(frozen) == 24 and all(n.endswith(".w") for n in frozen)
    assert loaded.config.inference_sd_state_dtype == "bfloat16"


@pytest.mark.cuda
@pytest.mark.ckpt
def test_inference_writes_no_parameter(loaded, ckpt_dir):
    """Prefill, decode steps and a whole-row call leave every parameter untouched; the 24 PLIF b_eta vectors are the
    file's."""
    model = loaded
    dev = torch.device("cuda")
    versions = {n: (int(p._version), int(p.data_ptr())) for n, p in model.named_parameters()}
    b_eta = {n: p.detach().clone() for n, p in model.named_parameters() if n.endswith(".b_eta")}
    assert len(b_eta) == 24
    with torch.no_grad():
        ids = _prompts(2, 64, seed=1, device=dev)
        ip = model.allocate_inference_cache(2, 2048)
        lg = model(ids, inference_params=ip, num_last_tokens=1).logits
        for _ in range(4):
            tok = lg[:, -1].argmax(dim=-1, keepdim=True)
            lg = model(tok, inference_params=ip).logits
        model(_prompts(1, 256, seed=2, device=dev))
    now = {n: (int(p._version), int(p.data_ptr())) for n, p in model.named_parameters()}
    changed = [n for n in versions if versions[n] != now[n]]
    assert not changed, f"inference wrote parameters: {changed[:8]}"
    from safetensors import safe_open

    with safe_open(str(ckpt_dir / WEIGHTS_NAME), framework="pt", device="cpu") as f:
        for n, saved in b_eta.items():
            ref = f.get_tensor(n).to(dev)
            assert_match(dict(model.named_parameters())[n], ref, n)
            assert_match(saved, ref, n + " (before inference)")


@pytest.mark.cuda
@pytest.mark.ckpt
def test_verification_refuses_altered_checkpoints(ckpt_dir):
    config = load_config(ckpt_dir)
    sd = load_state_dict(ckpt_dir, device="cpu")
    summary = verify_state_dict(config, sd)
    assert summary["state_tensors"] == 678 and summary["uncalibrated_memory_layers"] == []

    def refused(mutate, **kw):
        bad = dict(sd)
        mutate(bad)
        with pytest.raises(RuntimeError):
            verify_state_dict(config, bad, **kw)

    refused(lambda d: d.pop("backbone.norm.weight"))
    refused(lambda d: d.__setitem__("backbone.extra", torch.zeros(1)))
    refused(lambda d: d.__setitem__("backbone.norm.weight", d["backbone.norm.weight"].to(torch.bfloat16)))
    refused(lambda d: d.__setitem__("backbone.norm.weight", d["backbone.norm.weight"][:-1]))
    refused(lambda d: d.__setitem__("backbone.lexical_memory.hash_add", d["backbone.lexical_memory.hash_add"] + 1))
    refused(lambda d: d.__setitem__("backbone.layers.7.pool_value_residual_anchor",
                                    d["backbone.layers.7.pool_value_residual_anchor"] * 2.0))
    key = "backbone.layers.11.pkm_calibrated"
    refused(lambda d: d.__setitem__(key, torch.zeros_like(d[key])))
    unc = dict(sd)
    unc[key] = torch.zeros_like(sd[key])
    assert verify_state_dict(config, unc, allow_uncalibrated=True)["uncalibrated_memory_layers"] == [11]


@pytest.mark.cuda
@pytest.mark.ckpt
def test_save_round_trip(loaded, ckpt_dir, tmp_path):
    dev = torch.device("cuda")
    saved = tmp_path / "saved"
    summary = loaded.save_pretrained(saved, tokenizer_dir=ckpt_dir)
    assert summary["state_tensors"] == 678
    assert json.loads((saved / CONFIG_NAME).read_text())["model_type"] == MODEL_TYPE
    assert load_config(saved) == loaded.config
    present = [n for n in TOKENIZER_FILES if (ckpt_dir / n).is_file()]
    assert sorted(summary["tokenizer_files"]) == sorted(present)
    assert all((saved / n).is_file() for n in present)
    assert all(k.startswith(PREFIX) for k in _file_keys(saved))
    assert _compare_with_file(loaded.state_dict(), saved, dev) == 678
    model2 = NeuronSparkLMHeadModel.from_pretrained(saved, device="cuda")
    assert all(l.pkm_ready for l in model2.backbone.memory_layers)
    assert _compare_with_file(model2.state_dict(), saved, dev) == 678
    del model2
    shutil.rmtree(saved)
    gc.collect()
    torch.cuda.empty_cache()
