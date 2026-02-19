"""
Benchmark: Fused PLIF kernel vs 3-phase approach (Triton linear_recurrence backend).
Kernel-level comparison, not end-to-end model.
"""
import os
os.environ['TRITON_PTXAS_PATH'] = '/usr/local/cuda-13.0/bin/ptxas'

import sys
sys.path.insert(0, '/home/dgxspark/Desktop/NeuronSpark')

import torch
import time

from atomic_ops.parallel_scan import (
    linear_recurrence,
    plif_parallel_forward,
    _HAS_TRITON,
    _TritonPLIFForward,
)
from spikingjelly.activation_based import surrogate

device = 'cuda'
assert _HAS_TRITON


def bench_plif_forward(K, batch, D, dtype, n_warmup=5, n_iter=20, label=""):
    """Benchmark plif_parallel_forward: Fused vs 3-phase."""
    torch.manual_seed(42)
    surr = surrogate.Sigmoid(alpha=4.0)
    beta = torch.sigmoid(torch.randn(K, batch, D, device=device, dtype=dtype))
    u = torch.randn(K, batch, D, device=device, dtype=dtype) * 0.1
    v_th = 0.1 + torch.abs(torch.randn(K, batch, D, device=device, dtype=dtype)) * 0.5
    v_init = torch.zeros(batch, D, device=device, dtype=dtype)

    # ---- Fused Triton kernel ----
    for _ in range(n_warmup):
        spike, V_post = _TritonPLIFForward.apply(
            beta, u, v_th, v_init, float(surr.alpha),
        )
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n_iter):
        spike, V_post = _TritonPLIFForward.apply(
            beta, u, v_th, v_init, float(surr.alpha),
        )
    torch.cuda.synchronize()
    t_fused = (time.perf_counter() - t0) / n_iter

    # ---- 3-phase approach (force fallback by passing None surrogate, then use surrogate manually) ----
    def threephase_plif(beta, u, v_th, v_init, surr):
        """3-phase approach: linear_recurrence + spike iteration + correction."""
        V_L = linear_recurrence(beta, u, v_init)
        with torch.no_grad():
            V_L_det = V_L.detach()
            beta_det = beta.detach()
            v_th_det = v_th.detach()
            spike_pattern = (V_L_det >= v_th_det).float()
            for _ in range(2):  # max_iter-1
                delta_S = linear_recurrence(
                    beta_det, v_th_det * spike_pattern,
                    torch.zeros_like(v_init),
                )
                delta_S_prev = torch.zeros_like(delta_S)
                delta_S_prev[1:] = delta_S[:-1]
                V_pre_det = V_L_det - beta_det * delta_S_prev
                spike_new = (V_pre_det >= v_th_det).float()
                if torch.equal(spike_new, spike_pattern):
                    break
                spike_pattern = spike_new
        u_eff = u - v_th * spike_pattern
        V_post = linear_recurrence(beta, u_eff, v_init)
        V_post_prev = torch.zeros_like(V_post)
        V_post_prev[0] = v_init
        V_post_prev[1:] = V_post[:-1]
        V_pre = beta * V_post_prev + u
        spike = surr(V_pre - v_th)
        return spike, V_post

    for _ in range(n_warmup):
        spike3, V3 = threephase_plif(beta, u, v_th, v_init, surr)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n_iter):
        spike3, V3 = threephase_plif(beta, u, v_th, v_init, surr)
    torch.cuda.synchronize()
    t_3phase = (time.perf_counter() - t0) / n_iter

    speedup = t_3phase / t_fused
    tensor_mb = K * batch * D * (2 if dtype == torch.bfloat16 else 4) / 1e6
    print(f"  {label:25s}  shape=({K:5d},{batch},{D:5d})  tensor={tensor_mb:6.1f}MB"
          f"  Fused={t_fused*1000:7.2f}ms  3-Phase={t_3phase*1000:7.2f}ms  speedup={speedup:.2f}x")


def bench_plif_fwd_bwd(K, batch, D, dtype, n_warmup=3, n_iter=10, label=""):
    """Benchmark forward + backward."""
    torch.manual_seed(42)
    surr = surrogate.Sigmoid(alpha=4.0)
    beta_logits = torch.randn(K, batch, D, device=device, dtype=torch.float32, requires_grad=True)
    u = torch.randn(K, batch, D, device=device, dtype=torch.float32, requires_grad=True)
    v_th_raw = torch.randn(K, batch, D, device=device, dtype=torch.float32, requires_grad=True)
    v_init = torch.zeros(batch, D, device=device, dtype=torch.float32, requires_grad=True)

    # ---- Fused ----
    for _ in range(n_warmup):
        beta = torch.sigmoid(beta_logits)
        v_th = 0.1 + torch.abs(v_th_raw) * 0.5
        spike, V_post, _ = plif_parallel_forward(beta, u, v_th, v_init, surrogate_function=surr)
        (spike.sum() + V_post.sum()).backward()
        beta_logits.grad = None; u.grad = None; v_th_raw.grad = None; v_init.grad = None
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n_iter):
        beta = torch.sigmoid(beta_logits)
        v_th = 0.1 + torch.abs(v_th_raw) * 0.5
        spike, V_post, _ = plif_parallel_forward(beta, u, v_th, v_init, surrogate_function=surr)
        (spike.sum() + V_post.sum()).backward()
        beta_logits.grad = None; u.grad = None; v_th_raw.grad = None; v_init.grad = None
    torch.cuda.synchronize()
    t_fused = (time.perf_counter() - t0) / n_iter

    # ---- 3-phase ----
    # Use a dummy surrogate class to force fallback
    class _NonSigmoid:
        """Dummy wrapper that makes plif_parallel_forward use the 3-phase fallback."""
        def __init__(self, surr):
            self._surr = surr
        def __call__(self, x):
            return self._surr(x)
    dummy_surr = _NonSigmoid(surr)

    for _ in range(n_warmup):
        beta = torch.sigmoid(beta_logits)
        v_th = 0.1 + torch.abs(v_th_raw) * 0.5
        spike, V_post, _ = plif_parallel_forward(beta, u, v_th, v_init, surrogate_function=dummy_surr)
        (spike.sum() + V_post.sum()).backward()
        beta_logits.grad = None; u.grad = None; v_th_raw.grad = None; v_init.grad = None
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n_iter):
        beta = torch.sigmoid(beta_logits)
        v_th = 0.1 + torch.abs(v_th_raw) * 0.5
        spike, V_post, _ = plif_parallel_forward(beta, u, v_th, v_init, surrogate_function=dummy_surr)
        (spike.sum() + V_post.sum()).backward()
        beta_logits.grad = None; u.grad = None; v_th_raw.grad = None; v_init.grad = None
    torch.cuda.synchronize()
    t_3phase = (time.perf_counter() - t0) / n_iter

    speedup = t_3phase / t_fused
    print(f"  {label:25s}  shape=({K:5d},{batch},{D:5d})"
          f"  Fused={t_fused*1000:7.2f}ms  3-Phase={t_3phase*1000:7.2f}ms  speedup={speedup:.2f}x")


print("=" * 100)
print("Forward only (bf16) — plif_parallel_forward: Fused vs 3-Phase")
print("=" * 100)
bench_plif_forward(8192, 2, 6144, torch.bfloat16, label="hidden (DN=6144)")
bench_plif_forward(8192, 2, 768,  torch.bfloat16, label="output (D=768)")
bench_plif_forward(8192, 2, 2304, torch.bfloat16, label="ffn_mid (D_ff=2304)")

print()
print("=" * 100)
print("Forward + Backward (fp32) — plif_parallel_forward + .backward()")
print("=" * 100)
bench_plif_fwd_bwd(8192, 2, 768,  torch.float32, label="output (D=768)")
bench_plif_fwd_bwd(8192, 2, 2304, torch.float32, label="ffn_mid (D_ff=2304)")

print()
print("Done.")
