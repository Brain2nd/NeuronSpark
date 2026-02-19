"""
Benchmark: SNNFFN merged gate+up PLIF scan vs 2 separate scans.
The previous detail benchmark showed separate is MUCH faster.
Test with realistic data patterns (constant beta, varying u).
"""
import os
os.environ['TRITON_PTXAS_PATH'] = '/usr/local/cuda-13.0/bin/ptxas'

import sys
sys.path.insert(0, '/home/dgxspark/Desktop/NeuronSpark')

import torch
import time
from atomic_ops.parallel_scan import plif_parallel_forward, plif_fixed_param_forward
from spikingjelly.activation_based import surrogate

device = 'cuda'
dtype = torch.bfloat16

D_ff = 2304
TK = 8192
batch = 2
surr = surrogate.Sigmoid(alpha=4.0)


def sync_time():
    torch.cuda.synchronize()
    return time.perf_counter()


def bench_op(label, fn, n_warmup=5, n_iter=20):
    for _ in range(n_warmup):
        fn()
    t0 = sync_time()
    for _ in range(n_iter):
        fn()
    t = (sync_time() - t0) / n_iter
    print(f"  {label:55s} {t*1000:8.2f}ms")
    return t


print("=" * 80)
print("SNNFFN: Merged vs Separate PLIF scan")
print(f"TK={TK}, batch={batch}, D_ff={D_ff}")
print("=" * 80)

# Realistic data: gate and up have DIFFERENT u values, but CONSTANT beta and v_th
# (from ParametricLIFNode scalars)
u_gate = torch.randn(TK, batch, D_ff, device=device, dtype=dtype) * 0.1
u_up = torch.randn(TK, batch, D_ff, device=device, dtype=dtype) * 0.1
v_init_gate = torch.zeros(batch, D_ff, device=device, dtype=dtype)
v_init_up = torch.zeros(batch, D_ff, device=device, dtype=dtype)

# Scalar params (from neurons)
beta_gate_scalar = torch.sigmoid(torch.tensor(0.5, device=device, dtype=dtype))
beta_up_scalar = torch.sigmoid(torch.tensor(0.3, device=device, dtype=dtype))
v_th_gate = 0.3
v_th_up = 0.3

print("\n--- Approach 1: Merged (current SNNFFN code) ---")
# Build merged tensors exactly as SNNFFN does
def merged_approach():
    # (1-beta) * I
    u_merged = torch.cat([
        (1.0 - beta_gate_scalar) * u_gate,
        (1.0 - beta_up_scalar) * u_up,
    ], dim=-1)

    # beta_merged
    beta_row = torch.cat([beta_gate_scalar.expand(D_ff), beta_up_scalar.expand(D_ff)])
    beta_merged = beta_row.expand(TK, batch, 2 * D_ff).contiguous()

    # v_th_merged
    v_th_row = u_gate.new_empty(2 * D_ff)
    v_th_row[:D_ff].fill_(v_th_gate)
    v_th_row[D_ff:].fill_(v_th_up)
    v_th_merged = v_th_row.expand(TK, batch, 2 * D_ff).contiguous()

    # v_init_merged
    v_init_merged = torch.cat([v_init_gate, v_init_up], dim=-1)

    # PLIF scan
    spike_merged, V_post_merged, _ = plif_parallel_forward(
        beta_merged, u_merged, v_th_merged, v_init_merged,
        max_iter=3, surrogate_function=surr,
    )
    return spike_merged[:, :, :D_ff], spike_merged[:, :, D_ff:]

t_merged = bench_op("Merged (expand+contiguous + 1 PLIF scan)", merged_approach)

print("\n--- Approach 2: Separate (2 × plif_fixed_param_forward) ---")
def separate_approach():
    u_gate_scaled = (1.0 - beta_gate_scalar) * u_gate
    u_up_scaled = (1.0 - beta_up_scalar) * u_up

    gate_spike, _ = plif_fixed_param_forward(
        beta_gate_scalar, u_gate_scaled, v_th_gate, v_init_gate,
        max_iter=3, surrogate_function=surr,
    )
    up_spike, _ = plif_fixed_param_forward(
        beta_up_scalar, u_up_scaled, v_th_up, v_init_up,
        max_iter=3, surrogate_function=surr,
    )
    return gate_spike, up_spike

t_separate = bench_op("Separate (2 × plif_fixed_param_forward)", separate_approach)

print(f"\n  Speedup (separate over merged): {t_merged/t_separate:.2f}x")

# Verify results match
print("\n--- Correctness check ---")
torch.manual_seed(42)
r1 = merged_approach()
torch.manual_seed(42)
r2 = separate_approach()
for i, name in enumerate(['gate_spike', 'up_spike']):
    diff = (r1[i].float() - r2[i].float()).abs().max().item()
    print(f"  {name}: max_diff={diff:.4f}")

# Also benchmark just the overhead components
print("\n--- Component breakdown ---")
bench_op("torch.cat u_gate + u_up", lambda: torch.cat([
    (1.0 - beta_gate_scalar) * u_gate,
    (1.0 - beta_up_scalar) * u_up,
], dim=-1))

bench_op("expand+contiguous beta_merged", lambda: torch.cat(
    [beta_gate_scalar.expand(D_ff), beta_up_scalar.expand(D_ff)]
).expand(TK, batch, 2*D_ff).contiguous())

bench_op("expand+contiguous v_th_merged", lambda: (
    u_gate.new_empty(2*D_ff).fill_(0.3).expand(TK, batch, 2*D_ff).contiguous()
))

# Pure PLIF scan comparison (no overhead)
beta_merged_pre = torch.cat(
    [beta_gate_scalar.expand(D_ff), beta_up_scalar.expand(D_ff)]
).expand(TK, batch, 2*D_ff).contiguous()
u_merged_pre = torch.cat([
    (1.0-beta_gate_scalar)*u_gate, (1.0-beta_up_scalar)*u_up
], dim=-1)
v_th_merged_pre = u_gate.new_empty(2*D_ff).fill_(0.3).expand(TK, batch, 2*D_ff).contiguous()
v_init_merged_pre = torch.cat([v_init_gate, v_init_up], dim=-1)

bench_op("PLIF scan merged (pre-built, 4608 cols)", lambda: plif_parallel_forward(
    beta_merged_pre, u_merged_pre, v_th_merged_pre, v_init_merged_pre,
    max_iter=3, surrogate_function=surr,
))

u_gate_scaled = (1.0 - beta_gate_scalar) * u_gate
bench_op("PLIF scan separate (2304 cols)", lambda: plif_fixed_param_forward(
    beta_gate_scalar, u_gate_scaled, v_th_gate, v_init_gate,
    max_iter=3, surrogate_function=surr,
))

print("\nDone.")
