"""
Detailed layer profiling: breakdown by operation.
"""
import os
os.environ['TRITON_PTXAS_PATH'] = '/usr/local/cuda-13.0/bin/ptxas'

import sys
sys.path.insert(0, '/home/dgxspark/Desktop/NeuronSpark')

import torch
import torch.nn.functional as F
import time

device = 'cuda'
dtype = torch.bfloat16

D, N, K, seq_len, batch = 768, 8, 16, 512, 2
TK = seq_len * K
DN = D * N
D_ff = 2304


def sync_time():
    torch.cuda.synchronize()
    return time.perf_counter()


def bench_op(label, fn, n_warmup=3, n_iter=10):
    for _ in range(n_warmup):
        fn()
    t0 = sync_time()
    for _ in range(n_iter):
        fn()
    t = (sync_time() - t0) / n_iter
    print(f"  {label:40s} {t*1000:8.2f}ms")
    return t


print("=" * 60)
print("Operation-level benchmark")
print("=" * 60)

# Setup tensors
flat = torch.randn(TK * batch, D, device=device, dtype=dtype)
W_in = torch.randn(DN, D, device=device, dtype=dtype)
W_beta = torch.randn(DN, D, device=device, dtype=dtype)
W_alpha = torch.randn(DN, D, device=device, dtype=dtype)
W_th = torch.randn(DN, D, device=device, dtype=dtype)
W_gate = torch.randn(D, D, device=device, dtype=dtype)
W_skip = torch.randn(D, D, device=device, dtype=dtype)
W_out = torch.randn(D, DN, device=device, dtype=dtype)
b_beta = torch.randn(DN, device=device, dtype=dtype)
b_alpha = torch.randn(DN, device=device, dtype=dtype)
b_th = torch.randn(DN, device=device, dtype=dtype)

print("\n--- Matmul ---")
bench_op("F.linear D→DN (single)", lambda: F.linear(flat, W_in))
bench_op("F.linear D→DN (×4 separate)", lambda: (
    F.linear(flat, W_in), F.linear(flat, W_beta),
    F.linear(flat, W_alpha), F.linear(flat, W_th),
))

W_4DN = torch.cat([W_in, W_beta, W_alpha, W_th], dim=0)
bench_op("F.linear D→4DN (merged)", lambda: F.linear(flat, W_4DN))

bench_op("F.linear D→D (single)", lambda: F.linear(flat, W_gate))
bench_op("F.linear D→D (×2 separate)", lambda: (
    F.linear(flat, W_gate), F.linear(flat, W_skip),
))

W_2D = torch.cat([W_gate, W_skip], dim=0)
bench_op("F.linear D→2D (merged)", lambda: F.linear(flat, W_2D))

bench_op("torch.cat 4×(DN,D) weights", lambda: torch.cat([W_in, W_beta, W_alpha, W_th], dim=0))

print("\n--- Activations ---")
x_DN = torch.randn(TK, batch, DN, device=device, dtype=dtype)
bench_op("sigmoid(x + bias)", lambda: torch.sigmoid(x_DN + b_beta))
bench_op("softplus(x + bias)", lambda: F.softplus(x_DN + b_alpha))
bench_op("abs(x + bias)", lambda: torch.abs(x_DN + b_th))
bench_op("All 3 activations", lambda: (
    torch.sigmoid(x_DN + b_beta),
    F.softplus(x_DN + b_alpha),
    0.1 + torch.abs(x_DN + b_th),
))

print("\n--- PLIF Scans ---")
from atomic_ops.parallel_scan import plif_parallel_forward, plif_fixed_param_forward
from spikingjelly.activation_based import surrogate
surr = surrogate.Sigmoid(alpha=4.0)

beta_DN = torch.sigmoid(torch.randn(TK, batch, DN, device=device, dtype=dtype))
u_DN = torch.randn(TK, batch, DN, device=device, dtype=dtype) * 0.1
v_th_DN = 0.1 + torch.abs(torch.randn(TK, batch, DN, device=device, dtype=dtype)) * 0.3
v_init_DN = torch.zeros(batch, DN, device=device, dtype=dtype)

bench_op("PLIF hidden (DN=6144)", lambda: plif_parallel_forward(
    beta_DN, u_DN, v_th_DN, v_init_DN, max_iter=3, surrogate_function=surr,
))

beta_D = torch.sigmoid(torch.randn(1, device=device, dtype=dtype))
u_D = torch.randn(TK, batch, D, device=device, dtype=dtype) * 0.1
v_init_D = torch.zeros(batch, D, device=device, dtype=dtype)
bench_op("PLIF output (D=768, scalar β)", lambda: plif_fixed_param_forward(
    beta_D, u_D, 0.3, v_init_D, max_iter=3, surrogate_function=surr,
))

u_2Dff = torch.randn(TK, batch, 2*D_ff, device=device, dtype=dtype) * 0.1
beta_2Dff = torch.sigmoid(torch.randn(TK, batch, 2*D_ff, device=device, dtype=dtype))
v_th_2Dff = 0.1 + torch.abs(torch.randn(TK, batch, 2*D_ff, device=device, dtype=dtype)) * 0.3
v_init_2Dff = torch.zeros(batch, 2*D_ff, device=device, dtype=dtype)
bench_op("PLIF gate+up merged (2*D_ff=4608)", lambda: plif_parallel_forward(
    beta_2Dff, u_2Dff, v_th_2Dff, v_init_2Dff, max_iter=3, surrogate_function=surr,
))

u_Dff = torch.randn(TK, batch, D_ff, device=device, dtype=dtype) * 0.1
v_init_Dff = torch.zeros(batch, D_ff, device=device, dtype=dtype)
bench_op("PLIF D_ff=2304 (scalar β) ×2 sep", lambda: (
    plif_fixed_param_forward(beta_D, u_Dff, 0.3, v_init_Dff, max_iter=3, surrogate_function=surr),
    plif_fixed_param_forward(beta_D, u_Dff, 0.3, v_init_Dff, max_iter=3, surrogate_function=surr),
))

print("\n--- Misc ---")
s_flat = torch.randn(TK * batch, DN, device=device, dtype=dtype)
bench_op("F.linear DN→D (W_out)", lambda: F.linear(s_flat, W_out))
gate = torch.randn(TK, batch, D, device=device, dtype=dtype)
I_out = torch.randn(TK, batch, D, device=device, dtype=dtype)
I_skip = torch.randn(TK, batch, D, device=device, dtype=dtype)
bench_op("gate*out + skip", lambda: I_out * gate + I_skip)

alpha = torch.randn(TK, batch, DN, device=device, dtype=dtype)
I_all = torch.randn(TK, batch, DN, device=device, dtype=dtype)
bench_op("alpha * I_all (DN)", lambda: alpha * I_all)

bench_op("expand+contiguous scalar→(TK,B,D_ff)", lambda: beta_D.expand(TK, batch, D_ff).contiguous())

print("\nDone.")
