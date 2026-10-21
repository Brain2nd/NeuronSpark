"""The model at two model widths, D = 1536 (the default) and D = 2048, with the MiniCPM5 vocabulary (130,560) and 8
layers (6 dendritic neuron layers, 2 hippocampal layers), on fresh calibrated models (GPU).

* training: the kernel model against the reference model (``impl="ref"``) holding the same weights, B 2 x T 1024: the
  loss within a relative 1e-3, the median relative error of the parameter gradients below 0.2 (a model-level bound:
  the per-operator differences compound through the layers, and the product-key top-k selection is discontinuous; a
  kernel that is wrong at a width gives errors of order 1);
* decode: CUDA-graph decode against eager decode on the same tokens at decode batches 1 and 8 (the gated and split
  dendritic step routes), the per-step logits by the per-operator rule.

The other model tests run the default configuration only.
"""
from __future__ import annotations

import gc

import pytest
import torch

from neuronspark import NeuronSparkConfig
from neuronspark.models.neuronspark_lm import NeuronSparkLMHeadModel
from neuronspark.utils.calibration import calibrate_fresh_model
from neuronspark.utils.generation import decode
from neuronspark.utils.testing import assert_match

pytestmark = [pytest.mark.cuda]

VOCAB = 130560


def width_config(D: int) -> NeuronSparkConfig:
    return NeuronSparkConfig(vocab_size=VOCAB, lexical_memory_rows=(VOCAB, 262139, 344951), D=D, D_key=D, D_value=D,
                             D_ff=D // 2, hippocampal_heads=D // 256, hippocampal_pkm_value_dim=D // 2,
                             lexical_memory_head_dims=(D // 16, 16, 16), num_layers=8,
                             hippocampal_pool_value_residual_anchor=((0.5, 0.5),))


@pytest.fixture(scope="module", params=[1536, 2048], ids=["D1536", "D2048"])
def model(request):
    cfg = width_config(request.param)
    torch.manual_seed(0)
    m = NeuronSparkLMHeadModel(cfg, device="cuda", init="launch")
    g = torch.Generator(device="cpu").manual_seed(1)
    calibrate_fresh_model(m, [torch.randint(0, VOCAB, (8, 1024), generator=g).cuda() for _ in range(8)], rows=64)
    yield m
    m.release_inference()
    del m
    gc.collect()
    torch.cuda.empty_cache()


def test_training_kernel_vs_reference(model):
    ref = NeuronSparkLMHeadModel(model.config, device="cuda", init="empty", impl="ref")
    ref.load_state_dict(model.state_dict())
    g = torch.Generator(device="cpu").manual_seed(2)
    x = torch.randint(0, VOCAB, (2, 1024), generator=g).cuda()
    y = torch.roll(x, -1, 1)
    out = {}
    for name, m in (("kernel", model), ("ref", ref)):
        m.zero_grad(set_to_none=True)
        lo = m.loss(x, y)
        lo.loss.backward()
        out[name] = (float(lo.loss.detach()), {n: p.grad.detach().clone() for n, p in m.named_parameters()
                                      if p.grad is not None})
    model.zero_grad(set_to_none=True)
    (lk, gk), (lr, gr) = out["kernel"], out["ref"]
    assert abs(lk - lr) <= 1e-3 * abs(lr), (lk, lr)
    assert set(gk) == set(gr)
    rel = sorted(float((gk[n] - gr[n]).norm() / gr[n].norm().clamp_min(1e-30)) for n in gr)
    assert rel[len(rel) // 2] < 0.2, rel[len(rel) // 2]
    del ref
    gc.collect()
    torch.cuda.empty_cache()


@pytest.mark.parametrize("b", [1, 8])
def test_cg_decode_matches_eager(model, b):
    g = torch.Generator(device="cpu").manual_seed(10 + b)
    prompt = (1000 + torch.randint(0, 256, (b, 64), generator=g)).cuda()
    eager = decode(prompt, model, 96, cg=False, output_scores=True)
    graph = decode(prompt, model, 96, cg=True, output_scores=True, teacher_outputs=eager.sequences)
    assert len(graph.scores) == len(eager.scores) == 32
    for step, (got, ref) in enumerate(zip(graph.scores, eager.scores)):
        assert_match(got, ref, f"D{model.config.D} b{b} step{step} logits")
    assert graph.sequences.tolist() == eager.sequences.tolist()
