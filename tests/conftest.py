"""Shared pytest configuration.

Markers: ``cuda`` (skipped without a GPU), ``sm100`` (skipped below compute capability 10.0).  Every test starts from
the fixed seed 0.

Fixture ``model_dir`` (session, GPU): a checkpoint directory written by ``save_pretrained`` from a fresh model of the
default configuration, launched from the fixed seed and calibrated on random token rows (no external checkpoint);
removed at the end of the session.
"""
from __future__ import annotations

import gc
import random
import shutil
from pathlib import Path

import pytest
import torch

SEED = 0


def pytest_configure(config):
    for line in (
        "cuda: needs a CUDA GPU",
        "sm100: needs a Blackwell GPU (compute capability >= 10.0)",
    ):
        config.addinivalue_line("markers", line)


def _capability():
    if not torch.cuda.is_available():
        return None
    return tuple(torch.cuda.get_device_capability())


def pytest_collection_modifyitems(config, items):
    cap = _capability()
    skip_cuda = pytest.mark.skip(reason="needs a CUDA GPU")
    skip_sm100 = pytest.mark.skip(reason="needs compute capability >= 10.0")
    for item in items:
        if ("cuda" in item.keywords or "sm100" in item.keywords) and cap is None:
            item.add_marker(skip_cuda)
        elif "sm100" in item.keywords and cap < (10, 0):
            item.add_marker(skip_sm100)


@pytest.fixture(autouse=True)
def _fixed_seed():
    random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
    yield


@pytest.fixture
def device():
    if not torch.cuda.is_available():
        pytest.skip("needs a CUDA GPU")
    return torch.device("cuda", torch.cuda.current_device())


@pytest.fixture(scope="session")
def model_dir(tmp_path_factory) -> Path:
    if not torch.cuda.is_available():
        pytest.skip("needs a CUDA GPU")
    from neuronspark import NeuronSparkConfig, NeuronSparkLMHeadModel
    from neuronspark.utils.calibration import CALIBRATION_ROWS, calibrate_fresh_model

    config = NeuronSparkConfig()
    torch.manual_seed(SEED)
    model = NeuronSparkLMHeadModel(config, device="cuda", init="launch")
    g = torch.Generator(device="cpu").manual_seed(SEED + 1)
    rows = [torch.randint(0, int(config.vocab_size), (8, 1024), generator=g).cuda()
            for _ in range(CALIBRATION_ROWS // 8)]
    calibrate_fresh_model(model, rows, rows=CALIBRATION_ROWS)
    path = tmp_path_factory.mktemp("fresh_model")
    model.save_pretrained(path)
    del model, rows
    gc.collect()
    torch.cuda.empty_cache()
    yield path
    shutil.rmtree(path, ignore_errors=True)
