"""The row-chunked readout (``READOUT_MAX_ELEMENTS``): with the bound lowered so that 4096 rows are read out in two
chunks of 2048, the logits of the forward, the loss and every parameter gradient of the loss against the one-call
readout (a fresh calibrated 8-layer model, D = 1536, the MiniCPM5 vocabulary; GPU).  The per-operator rule of
``neuronspark.utils.testing``: the chunks change only the row count of the readout GEMMs (cuBLAS may pick another
kernel) and the order in which the embedding's readout gradient is summed."""
from __future__ import annotations

import gc

import pytest
import torch

import neuronspark.models.neuronspark_lm as lm
from neuronspark import NeuronSparkConfig
from neuronspark.models.neuronspark_lm import NeuronSparkLMHeadModel
from neuronspark.utils.calibration import calibrate_fresh_model
from neuronspark.utils.testing import assert_all_match, assert_match

pytestmark = [pytest.mark.cuda]

VOCAB = 130560


@pytest.fixture(scope="module")
def model():
    cfg = NeuronSparkConfig(vocab_size=VOCAB, lexical_memory_rows=(VOCAB, 262139, 344951), num_layers=8,
                            hippocampal_pool_value_residual_anchor=((0.5, 0.5),))
    torch.manual_seed(0)
    m = NeuronSparkLMHeadModel(cfg, device="cuda", init="launch")
    g = torch.Generator(device="cpu").manual_seed(1)
    calibrate_fresh_model(m, [torch.randint(0, VOCAB, (8, 1024), generator=g).cuda() for _ in range(8)], rows=64)
    yield m
    del m
    gc.collect()
    torch.cuda.empty_cache()


def _loss_and_grads(m, x, y):
    m.zero_grad(set_to_none=True)
    out = m.loss(x, y)
    out.loss.backward()
    grads = {n: p.grad.detach().clone() for n, p in m.named_parameters() if p.grad is not None}
    m.zero_grad(set_to_none=True)
    return out.loss.detach().reshape(1), grads


def test_chunked_readout_matches_one_call(model, monkeypatch):
    g = torch.Generator(device="cpu").manual_seed(2)
    x = torch.randint(0, VOCAB, (2, 2048), generator=g).cuda()
    y = torch.roll(x, -1, 1)
    assert model._row_chunks(4096) == [(0, 4096)]
    loss1, grads1 = _loss_and_grads(model, x, y)
    with torch.no_grad():
        logits1 = model(x).logits
    monkeypatch.setattr(lm, "READOUT_MAX_ELEMENTS", 2048 * VOCAB)
    assert model._row_chunks(4096) == [(0, 2048), (2048, 4096)]
    loss2, grads2 = _loss_and_grads(model, x, y)
    with torch.no_grad():
        logits2 = model(x).logits
    assert_match(logits2, logits1, "logits")
    assert_match(loss2, loss1, "loss")
    assert set(grads2) == set(grads1)
    assert_all_match(grads2, grads1, prefix="grad")
