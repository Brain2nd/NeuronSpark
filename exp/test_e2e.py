"""End-to-end model test."""
import os
os.environ['TRITON_PTXAS_PATH'] = '/usr/local/cuda-13.0/bin/ptxas'

import sys
sys.path.insert(0, '/home/dgxspark/Desktop/NeuronSpark')

import torch
import traceback

device = 'cuda'
dtype = torch.bfloat16

from model import SNNLanguageModel

model = SNNLanguageModel(
    vocab_size=256, D=64, N=4, K=4, num_layers=2, D_ff=128
).to(device).to(dtype)

x = torch.randint(0, 256, (2, 16), device=device)
y = torch.randint(0, 256, (2, 16), device=device)

try:
    print("Forward...")
    out = model(x, y)
    loss = out.last_loss.mean()  # model returns per-element loss (reduction='none')
    print(f"Loss: {loss.item():.4f}")

    print("Backward...")
    loss.backward()

    total = 0
    ok = 0
    for name, p in model.named_parameters():
        total += 1
        if p.grad is not None and p.grad.abs().sum() > 0:
            ok += 1
        else:
            print(f"  NO GRAD: {name}")
    print(f"Grad: {ok}/{total}")
    print("OK" if ok == total else "SOME MISSING")
except Exception as e:
    traceback.print_exc()
