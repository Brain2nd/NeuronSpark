"""Activation checkpointing: the loss and every parameter gradient with the dendritic neuron layers recomputed
(``gradient_checkpointing``) and with the hippocampal layers recomputed as well (``hippocampal_checkpointing``) against
the resident run (a fresh calibrated 8-layer model of the default width; GPU).  The per-operator rule of
``neuronspark.utils.testing``: recomputation runs the same kernels on the same inputs; the checkpointed dendritic layers
form their own pre-norm instead of the joined residual + pre-norm."""
from __future__ import annotations

import gc

import pytest
import torch

from neuronspark import NeuronSparkConfig
from neuronspark.models.neuronspark_lm import NeuronSparkLMHeadModel
from neuronspark.utils.calibration import calibrate_fresh_model
from neuronspark.utils.testing import assert_all_match, assert_match

pytestmark = [pytest.mark.cuda]


@pytest.fixture(scope="module")
def model():
    cfg = NeuronSparkConfig(num_layers=8, hippocampal_pool_value_residual_anchor=((0.5, 0.5),))
    torch.manual_seed(0)
    m = NeuronSparkLMHeadModel(cfg, device="cuda", init="launch")
    g = torch.Generator(device="cpu").manual_seed(1)
    calibrate_fresh_model(m, [torch.randint(0, cfg.vocab_size, (8, 1024), generator=g).cuda() for _ in range(8)],
                          rows=64)
    yield m
    del m
    gc.collect()
    torch.cuda.empty_cache()


def _run(m, x, y, dendritic, hippocampal):
    m.gradient_checkpointing = dendritic
    m.hippocampal_checkpointing = hippocampal
    m.zero_grad(set_to_none=True)
    out = m.loss(x, y)
    out.loss.backward()
    res = ({"loss": out.loss.detach().reshape(1), "aux": out.aux.detach().reshape(1)},
           {n: p.grad.detach().clone() for n, p in m.named_parameters() if p.grad is not None})
    m.zero_grad(set_to_none=True)
    m.gradient_checkpointing = m.hippocampal_checkpointing = False
    return res


@pytest.mark.parametrize("hippocampal", [False, True], ids=["dendritic", "dendritic+hippocampal"])
def test_checkpointing_matches_resident(model, hippocampal):
    g = torch.Generator(device="cpu").manual_seed(2)
    x = torch.randint(0, model.config.vocab_size, (2, 1024), generator=g).cuda()
    y = torch.roll(x, -1, 1)
    ref_l, ref_g = _run(model, x, y, False, False)
    new_l, new_g = _run(model, x, y, True, hippocampal)
    assert_match(new_l["loss"], ref_l["loss"], "loss")
    assert_match(new_l["aux"], ref_l["aux"], "aux")
    assert set(new_g) == set(ref_g)
    assert_all_match(new_g, ref_g, prefix="grad")
