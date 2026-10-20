"""Checkpoint loading and saving (local directories; no hub access and no transformers dependency).

Checkpoint directory format (written by :func:`save_pretrained`, read by :func:`load_pretrained` /
``NeuronSparkLMHeadModel.from_pretrained``; no other format is read):

* ``config.json`` (:data:`CONFIG_NAME`): the configuration file of :mod:`neuronspark.models.config_neuronspark` --
  ``"model_type": "neuronspark"``, ``"architectures": ["NeuronSparkLMHeadModel"]`` and the
  :class:`NeuronSparkConfig` fields under their own names;
* ``model.safetensors`` (:data:`WEIGHTS_NAME`, metadata ``{"format": "pt"}``): ``NeuronSparkLMHeadModel.state_dict()``
  with its own key names, every key under the prefix ``backbone.`` (:data:`PREFIX`)::

      backbone.embed_tokens.weight                  (vocab_size, D) fp32, also the tied readout weight (stored once)
      backbone.layers.{i}.mixer_a.<name>            spiking layer i: pre_norm_gain, in_proj.weight, conv_weight,
      conv_bias,
                                                    dt_bias, A_log, D, gamma, W_out, rho (fp32)
      backbone.layers.{i}.<name>                    memory layer i: 70 fp32 parameters and 11 fp32 persistent buffers
      backbone.norm.weight                          (D,) fp32 final RMSNorm gain
      backbone.balance_stack.projs.{k}.weight       (w_k, D) fp32 readout-balance projections
      backbone.lexical_memory.tables.{k}            (rows, dim) fp32 lexical n-gram tables
      backbone.lexical_memory.hash_{mult,add,wrap}  int64 lexical hash constants

  Non-persistent buffers (RoPE tables, unit norm weights) are not stored.  For the 1.27B configuration the file holds
  678 tensors: 609 fp32 parameters (1,270,342,292 elements) and 69 persistent buffers (10,908 elements);
* optionally the tokenizer files :data:`TOKENIZER_FILES`, copied unchanged.

:func:`verify_state_dict` requires exactly the model's key set with the model's shapes and dtypes and the configured
tensor / element counts; the lexical hash constants must equal their recomputation from the hash seed (integers,
compared exactly), every memory layer's value-residual anchor must equal the configuration's anchor (per-operator rule
of :mod:`neuronspark.utils.testing`), and every memory layer must carry ``pkm_calibrated = 1`` (the launch
calibration's values) unless ``allow_uncalibrated``.

Loading never writes a parameter after ``load_state_dict``: the model is built with ``init="empty"`` (no random draw;
every PLIF ``b_eta`` comes from the file), loaded strictly, and the memory layers mark their product-key memory ready
from the loaded ``pkm_calibrated`` buffer.

Structure follows mamba_ssm ``utils/hf.py`` (``load_config_hf`` / ``load_state_dict_hf``), for local paths.
"""
from __future__ import annotations

import dataclasses
import functools
import json
import os
import shutil
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple, Type, Union

import torch
from safetensors.torch import load_file, save_file

from neuronspark.models.config_neuronspark import MODEL_TYPE, NeuronSparkConfig
from neuronspark.utils.testing import match_stats

__all__ = [
    "CONFIG_NAME",
    "WEIGHTS_NAME",
    "TOKENIZER_FILES",
    "PREFIX",
    "load_config",
    "load_state_dict",
    "expected_state_layout",
    "verify_state_dict",
    "load_pretrained",
    "save_pretrained",
    "copy_tokenizer_files",
]

CONFIG_NAME = "config.json"
WEIGHTS_NAME = "model.safetensors"
# tokenizer files copied next to a saved checkpoint when present in the tokenizer directory
TOKENIZER_FILES = (
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "vocab.json",
    "merges.txt",
    "added_tokens.json",
    "chat_template.jinja",
    "generation_config.json",
)
# prefix of every tensor key of model.safetensors (the LM head model's backbone)
PREFIX = "backbone."

_PathLike = Union[str, "os.PathLike[str]"]


# ------------------------------------------------------------------------------------------------ config
def _config_file(path: _PathLike) -> Path:
    p = Path(path)
    return p / CONFIG_NAME if p.is_dir() else p


def load_config(path: _PathLike) -> NeuronSparkConfig:
    """The :class:`NeuronSparkConfig` of a checkpoint directory (or of a ``config.json`` path).  The file must be a
    JSON object with ``"model_type": "neuronspark"``; it is read by :meth:`NeuronSparkConfig.from_dict`."""
    f = _config_file(path)
    with open(f, "r", encoding="utf-8") as fh:
        d = json.load(fh)
    if not isinstance(d, dict):
        raise ValueError(f"{f}: the configuration is not a JSON object")
    mt = d.get("model_type", None)
    if mt != MODEL_TYPE:
        raise ValueError(f"{f}: model_type {mt!r} is not {MODEL_TYPE!r}")
    return NeuronSparkConfig.from_dict(d)


# ------------------------------------------------------------------------------------------------ tensors
def load_state_dict(path: _PathLike, device: Union[str, torch.device, None] = None) -> Dict[str, torch.Tensor]:
    """``model.safetensors`` of a checkpoint directory (or the file itself), loaded onto ``device`` (default CPU), keys
    as stored (``backbone.*``)."""
    p = Path(path)
    f = p / WEIGHTS_NAME if p.is_dir() else p
    if not f.is_file():
        raise FileNotFoundError(f"checkpoint weights not found: {f}")
    dev = "cpu" if device is None else str(torch.device(device))
    return dict(load_file(str(f), device=dev))


@functools.lru_cache(maxsize=4)
def _layout(config: NeuronSparkConfig) -> Tuple[Dict[str, Tuple[Tuple[int, ...], torch.dtype]],
                                               Dict[str, Tuple[Tuple[int, ...], torch.dtype]],
                                               Dict[str, Tuple[int, ...]]]:
    # the model built on the meta device (no storage, no draw): the expected keys, shapes and dtypes
    from neuronspark.models.neuronspark_lm import NeuronSparkLMHeadModel

    model = NeuronSparkLMHeadModel(config, device="meta", init="empty")
    params = {n: (tuple(p.shape), p.dtype) for n, p in model.named_parameters()}
    buffers = {n: (tuple(b.shape), b.dtype) for n, b in model.state_dict().items() if n not in params}
    hashes = {k: tuple(int(x) for x in v) for k, v in model.backbone.lexical_memory.expected_hash_buffers().items()}
    return params, buffers, hashes


def expected_state_layout(config: NeuronSparkConfig) -> Tuple[Dict[str, Tuple[Tuple[int, ...], torch.dtype]],
                                                               Dict[str, Tuple[Tuple[int, ...], torch.dtype]]]:
    """``(parameters, buffers)``: ``{key: (shape, dtype)}`` of the model's state dict (``backbone.*`` keys) for
    ``config``."""
    params, buffers, _ = _layout(config)
    return dict(params), dict(buffers)


def _expected_anchor(config: NeuronSparkConfig, mem_index: int) -> torch.Tensor:
    from neuronspark.modules.pool_attention import value_residual_anchor

    if mem_index == 0:
        return torch.zeros((int(config.D_value),), dtype=torch.float32)
    pairs = config.memory_pool_value_residual_anchor
    return value_residual_anchor(pairs[mem_index - 1], int(config.D_value), int(config.memory_heads))


def verify_state_dict(config: NeuronSparkConfig, state_dict: Mapping[str, torch.Tensor],
                      allow_uncalibrated: bool = False) -> Dict[str, Any]:
    """Raise unless ``state_dict`` (``backbone.*`` keys) is a complete checkpoint for ``config`` (see the module
    docstring).  Returns a summary (tensor and element counts, the uncalibrated memory layers)."""
    params, buffers, hashes = _layout(config)
    want = set(params) | set(buffers)
    have = set(state_dict)
    missing = sorted(want - have)
    unexpected = sorted(have - want)
    if missing or unexpected:
        raise RuntimeError(f"checkpoint: {len(missing)} missing keys {missing[:10]}, {len(unexpected)} unexpected keys "
                           f"{unexpected[:10]}")
    bad = []
    for name, (shape, dtype) in list(params.items()) + list(buffers.items()):
        t = state_dict[name]
        if not isinstance(t, torch.Tensor):
            bad.append(f"{name}: not a tensor ({type(t).__name__})")
        elif tuple(t.shape) != shape:
            bad.append(f"{name}: shape {tuple(t.shape)}, expected {shape}")
        elif t.dtype != dtype:
            bad.append(f"{name}: dtype {t.dtype}, expected {dtype}")
    if bad:
        raise RuntimeError(f"checkpoint: {len(bad)} tensors do not match the model: " + "; ".join(bad[:10]))
    n_param = len(params)
    e_param = sum(int(state_dict[k].numel()) for k in params)
    n_buf = len(buffers)
    e_buf = sum(int(state_dict[k].numel()) for k in buffers)
    counts = ((n_param, int(config.n_param_tensors_expected)), (e_param, int(config.n_params_expected)),
              (n_buf, int(config.n_buffer_tensors_expected)), (e_buf, int(config.n_buffers_expected)),
              (n_param + n_buf, int(config.n_state_tensors)))
    if any(got != exp for got, exp in counts):
        raise RuntimeError(f"checkpoint: tensor / element counts {[g for g, _ in counts]} differ from the "
                           f"configuration's {[e for _, e in counts]} (parameter tensors, parameter elements, buffer "
                           "tensors, buffer elements, state tensors)")
    for name, values in hashes.items():
        key = f"{PREFIX}lexical_memory.{name}"
        got = tuple(int(v) for v in state_dict[key].detach().to("cpu").tolist())
        if got != tuple(values):
            raise RuntimeError(f"checkpoint: {key} {list(got)} differs from the constants of lexical_memory_hash_seed "
                               f"{config.lexical_memory_hash_seed} {list(values)}")
    for layer_idx in config.memory_layer_ids:
        key = f"{PREFIX}layers.{layer_idx}.pool_value_residual_anchor"
        got = state_dict[key].detach()
        exp = _expected_anchor(config, config.memory_index(layer_idx)).to(got.device)
        res = match_stats(got.to(torch.float32), exp, key)
        if not res.passed:
            raise RuntimeError(f"checkpoint: the value-residual anchor differs from the configuration's "
                               f"memory_pool_value_residual_anchor: {res}")
    uncalibrated = [i for i in config.memory_layer_ids
                    if not float(state_dict[f"{PREFIX}layers.{i}.pkm_calibrated"]) > 0.5]
    if uncalibrated and not allow_uncalibrated:
        raise RuntimeError(f"checkpoint: memory layers {uncalibrated} carry pkm_calibrated = 0 (the product-key memory "
                           "was never calibrated); pass allow_uncalibrated=True to load it for calibration")
    return {"state_tensors": n_param + n_buf, "parameter_tensors": n_param, "parameter_elements": e_param,
            "buffer_tensors": n_buf, "buffer_elements": e_buf, "uncalibrated_memory_layers": uncalibrated}


# ------------------------------------------------------------------------------------------------ models
def load_pretrained(path: _PathLike, device: Union[str, torch.device] = "cuda", allow_uncalibrated: bool = False,
                    impl: str = "kernel", inference_spiking_state_dtype: Optional[str] = None,
                    model_cls: Optional[Type] = None):
    """Build and load a :class:`NeuronSparkLMHeadModel` (or ``model_cls``) from a checkpoint directory.

    1. :func:`load_config`; ``inference_spiking_state_dtype`` (``"bfloat16"`` / ``"float32"``) replaces the
       configuration's spiking-layer state storage option when given;
    2. the tensors onto ``device``, :func:`verify_state_dict`;
    3. the model built with ``init="empty"`` on ``device`` (fp32, no random draw), ``load_state_dict(strict=True)``;
    4. post-load checks: the lexical hash buffers, the product-key memories marked ready, fp32 parameters.
    """
    from neuronspark.utils.precision import check_fp32_parameters

    if model_cls is None:
        from neuronspark.models.neuronspark_lm import NeuronSparkLMHeadModel as model_cls
    config = load_config(path)
    if inference_spiking_state_dtype is not None:
        config = dataclasses.replace(config, inference_spiking_state_dtype=str(inference_spiking_state_dtype))
    sd = load_state_dict(path, device=device)
    verify_state_dict(config, sd, allow_uncalibrated=allow_uncalibrated)
    model = model_cls(config, device=device, dtype=torch.float32, impl=impl, init="empty")
    model.load_state_dict(sd, strict=True)
    del sd
    backbone = model.backbone
    backbone.lexical_memory.check_hash_buffers()
    for layer in backbone.memory_layers:
        calibrated = float(layer.pkm_calibrated) > 0.5
        if calibrated and not layer.pkm_ready:
            raise RuntimeError(f"load_pretrained: memory layer {layer.layer_idx} holds pkm_calibrated = 1 but was not "
                               "marked ready by the state-dict load")
    check_fp32_parameters(model)
    return model


def copy_tokenizer_files(src: _PathLike, dst: _PathLike) -> List[str]:
    """Copy the :data:`TOKENIZER_FILES` present in ``src`` into ``dst``; returns the copied names."""
    s, d = Path(src), Path(dst)
    copied = []
    for name in TOKENIZER_FILES:
        f = s / name
        if f.is_file():
            shutil.copy2(f, d / name)
            copied.append(name)
    return copied


def save_pretrained(model, path: _PathLike, tokenizer_dir: Optional[_PathLike] = None) -> Dict[str, Any]:
    """Write the checkpoint directory format (module docstring): ``path/config.json`` and ``path/model.safetensors``
    (fp32 / int64 tensors, ``backbone.*`` keys, checked by :func:`verify_state_dict` with ``allow_uncalibrated=True``
    before a file is written); copy the tokenizer files of ``tokenizer_dir``.  Returns the verification summary."""
    out = Path(path)
    out.mkdir(parents=True, exist_ok=True)
    config = model.config
    sd = {k: v.detach().to("cpu").contiguous() for k, v in model.state_dict().items()}
    summary = verify_state_dict(config, sd, allow_uncalibrated=True)
    config.to_json(out / CONFIG_NAME)
    save_file(sd, str(out / WEIGHTS_NAME), metadata={"format": "pt"})
    summary["tokenizer_files"] = [] if tokenizer_dir is None else copy_tokenizer_files(tokenizer_dir, out)
    return summary
