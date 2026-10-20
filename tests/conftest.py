"""Shared pytest configuration.

Command-line options:

* ``--ckpt DIR``: a NeuronSpark checkpoint directory (``config.json`` and ``model.safetensors`` as written by
  ``save_pretrained``).

Markers: ``cuda`` (skipped without a GPU), ``sm100`` (skipped below compute capability 10.0), ``ckpt`` (skipped
without ``--ckpt``).  Every test starts from the fixed seed 0.
"""
from __future__ import annotations

import random
from pathlib import Path

import pytest
import torch

SEED = 0


def pytest_addoption(parser):
    g = parser.getgroup("neuronspark")
    g.addoption("--ckpt", default=None, help="NeuronSpark checkpoint directory")


def pytest_configure(config):
    for line in (
        "cuda: needs a CUDA GPU",
        "sm100: needs a Blackwell GPU (compute capability >= 10.0)",
        "ckpt: needs the NeuronSpark checkpoint (--ckpt)",
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
    skip_ckpt = pytest.mark.skip(reason="needs --ckpt")
    for item in items:
        if ("cuda" in item.keywords or "sm100" in item.keywords) and cap is None:
            item.add_marker(skip_cuda)
        elif "sm100" in item.keywords and cap < (10, 0):
            item.add_marker(skip_sm100)
        if "ckpt" in item.keywords and not config.getoption("--ckpt"):
            item.add_marker(skip_ckpt)


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
def ckpt_dir(request) -> Path:
    path = request.config.getoption("--ckpt")
    if not path:
        pytest.skip("needs --ckpt")
    return Path(path)
