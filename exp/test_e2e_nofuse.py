"""End-to-end model test - compare fused vs unfused modulation."""
import os
os.environ['TRITON_PTXAS_PATH'] = '/usr/local/cuda-13.0/bin/ptxas'

import sys
sys.path.insert(0, '/home/dgxspark/Desktop/NeuronSpark')

import torch
import torch.nn.functional as F
from spikingjelly.activation_based import functional

# Monkey-patch to test without fused modulation
import atomic_ops.snn_block as snn_block_mod
_orig_fused = snn_block_mod._fused_modulation

def _unfused_modulation(raw_beta, b_beta, raw_alpha, b_alpha, raw_th, b_th, v_th_min, I_all):
    beta = torch.sigmoid(raw_beta + b_beta)
    alpha = F.softplus(raw_alpha + b_alpha)
    v_th = v_th_min + torch.abs(raw_th + b_th)
    u = alpha * I_all
    return beta, u, v_th

device = 'cuda'
dtype = torch.bfloat16

from model import SNNLanguageModel

def test_variant(label, use_fused):
    torch.manual_seed(42)
    if use_fused:
        snn_block_mod._fused_modulation = _orig_fused
    else:
        snn_block_mod._fused_modulation = _unfused_modulation

    model = SNNLanguageModel(
        vocab_size=256, D=64, N=4, K=4, num_layers=2, D_ff=128
    ).to(device).to(dtype)

    x = torch.randint(0, 256, (2, 16), device=device)
    y = torch.randint(0, 256, (2, 16), device=device)

    out = model(x, y)
    loss = out.last_loss.mean()
    loss.backward()

    total = 0
    ok = 0
    no_grad = []
    for name, p in model.named_parameters():
        total += 1
        if p.grad is not None and p.grad.abs().sum() > 0:
            ok += 1
        else:
            no_grad.append(name)

    print(f"\n{label}: {ok}/{total} have grad")
    if no_grad:
        for n in no_grad:
            print(f"  NO GRAD: {n}")

test_variant("Without fused (baseline)", use_fused=False)
test_variant("With fused (torch.compile)", use_fused=True)
print("\nDone.")
