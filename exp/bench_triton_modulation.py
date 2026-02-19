"""
Benchmark: Triton fused modulation kernel vs torch.compile vs separate ops.
Tests a hand-written Triton kernel for sigmoid + softplus + abs + alpha*I.
"""
import os
os.environ['TRITON_PTXAS_PATH'] = '/usr/local/cuda-13.0/bin/ptxas'

import sys
sys.path.insert(0, '/home/dgxspark/Desktop/NeuronSpark')

import torch
import torch.nn.functional as F
import time
import triton
import triton.language as tl

device = 'cuda'
dtype = torch.bfloat16

D, N, K, seq_len, batch = 768, 8, 16, 512, 2
TK = seq_len * K
DN = D * N


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
    print(f"  {label:50s} {t*1000:8.2f}ms")
    return t


# ====== Triton kernel ======

@triton.jit
def _fused_modulation_fwd_kernel(
    RAW_BETA_ptr, B_BETA_ptr,
    RAW_ALPHA_ptr, B_ALPHA_ptr,
    RAW_TH_ptr, B_TH_ptr,
    I_ALL_ptr,
    BETA_ptr, U_ptr, VTH_ptr,
    numel, DN, V_TH_MIN,
    BLOCK: tl.constexpr,
):
    """Single kernel: bias + sigmoid/softplus/abs + alpha*I."""
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < numel
    bias_idx = offs % DN

    # sigmoid(raw_beta + b_beta)
    raw_b = tl.load(RAW_BETA_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    bb = tl.load(B_BETA_ptr + bias_idx, mask=mask, other=0.0).to(tl.float32)
    beta = tl.sigmoid(raw_b + bb)
    tl.store(BETA_ptr + offs, beta, mask=mask)

    # softplus(raw_alpha + b_alpha)
    raw_a = tl.load(RAW_ALPHA_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    ba = tl.load(B_ALPHA_ptr + bias_idx, mask=mask, other=0.0).to(tl.float32)
    z_a = raw_a + ba
    # softplus = log(1 + exp(x)), with numerical stability
    alpha = tl.where(z_a > 20.0, z_a, tl.log(1.0 + tl.exp(z_a)))

    # u = alpha * I_all
    i_all = tl.load(I_ALL_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    u = alpha * i_all
    tl.store(U_ptr + offs, u, mask=mask)

    # v_th_min + abs(raw_th + b_th)
    raw_t = tl.load(RAW_TH_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    bt = tl.load(B_TH_ptr + bias_idx, mask=mask, other=0.0).to(tl.float32)
    vth = V_TH_MIN + tl.abs(raw_t + bt)
    tl.store(VTH_ptr + offs, vth, mask=mask)


@triton.jit
def _fused_modulation_bwd_kernel(
    RAW_BETA_ptr, B_BETA_ptr,
    RAW_ALPHA_ptr, B_ALPHA_ptr,
    RAW_TH_ptr, B_TH_ptr,
    I_ALL_ptr, BETA_ptr,
    GRAD_BETA_ptr, GRAD_U_ptr, GRAD_VTH_ptr,
    GRAD_RAW_BETA_ptr, GRAD_RAW_ALPHA_ptr, GRAD_RAW_TH_ptr,
    GRAD_I_ALL_ptr,
    numel, DN,
    BLOCK: tl.constexpr,
):
    """Backward kernel: element-wise gradients (no reduction for biases)."""
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < numel
    bias_idx = offs % DN

    # grad w.r.t. raw_beta: grad_beta * sigmoid'(z) = grad_beta * beta * (1 - beta)
    beta = tl.load(BETA_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    g_beta = tl.load(GRAD_BETA_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    grad_raw_beta = g_beta * beta * (1.0 - beta)
    tl.store(GRAD_RAW_BETA_ptr + offs, grad_raw_beta, mask=mask)

    # grad w.r.t. raw_alpha: grad_u * I_all * softplus'(z_alpha)
    # softplus'(z) = sigmoid(z)
    raw_a = tl.load(RAW_ALPHA_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    ba = tl.load(B_ALPHA_ptr + bias_idx, mask=mask, other=0.0).to(tl.float32)
    z_a = raw_a + ba
    sp_grad = tl.sigmoid(z_a)

    g_u = tl.load(GRAD_U_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    i_all = tl.load(I_ALL_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    grad_raw_alpha = g_u * i_all * sp_grad
    tl.store(GRAD_RAW_ALPHA_ptr + offs, grad_raw_alpha, mask=mask)

    # grad w.r.t. I_all: grad_u * alpha
    alpha = tl.where(z_a > 20.0, z_a, tl.log(1.0 + tl.exp(z_a)))
    grad_i_all = g_u * alpha
    tl.store(GRAD_I_ALL_ptr + offs, grad_i_all, mask=mask)

    # grad w.r.t. raw_th: grad_v_th * sign(z_th)
    raw_t = tl.load(RAW_TH_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    bt = tl.load(B_TH_ptr + bias_idx, mask=mask, other=0.0).to(tl.float32)
    z_t = raw_t + bt
    g_vth = tl.load(GRAD_VTH_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    grad_raw_th = g_vth * tl.where(z_t >= 0.0, 1.0, -1.0)
    tl.store(GRAD_RAW_TH_ptr + offs, grad_raw_th, mask=mask)


class _TritonFusedModulation(torch.autograd.Function):
    _BLOCK = 1024

    @staticmethod
    def forward(ctx, raw_beta, b_beta, raw_alpha, b_alpha, raw_th, b_th, v_th_min, I_all):
        numel = raw_beta.numel()
        DN = b_beta.shape[0]
        BLOCK = _TritonFusedModulation._BLOCK

        beta = torch.empty_like(raw_beta)
        u = torch.empty_like(raw_beta)
        v_th = torch.empty_like(raw_beta)

        grid = ((numel + BLOCK - 1) // BLOCK,)
        _fused_modulation_fwd_kernel[grid](
            raw_beta.contiguous(), b_beta.contiguous(),
            raw_alpha.contiguous(), b_alpha.contiguous(),
            raw_th.contiguous(), b_th.contiguous(),
            I_all.contiguous(),
            beta, u, v_th,
            numel, DN, float(v_th_min),
            BLOCK=BLOCK,
        )
        ctx.save_for_backward(raw_alpha, b_alpha, raw_th, b_th, I_all, beta)
        ctx.numel = numel
        ctx.DN = DN
        return beta, u, v_th

    @staticmethod
    def backward(ctx, grad_beta, grad_u, grad_v_th):
        raw_alpha, b_alpha, raw_th, b_th, I_all, beta = ctx.saved_tensors
        numel = ctx.numel
        DN = ctx.DN
        BLOCK = _TritonFusedModulation._BLOCK

        grad_raw_beta = torch.empty_like(beta)
        grad_raw_alpha = torch.empty_like(beta)
        grad_raw_th = torch.empty_like(beta)
        grad_I_all = torch.empty_like(I_all)

        grid = ((numel + BLOCK - 1) // BLOCK,)
        _fused_modulation_bwd_kernel[grid](
            # Note: raw_beta not needed, we use saved beta for sigmoid'
            beta, b_alpha,  # dummy for raw_beta_ptr since we use beta directly
            raw_alpha.contiguous(), b_alpha.contiguous(),
            raw_th.contiguous(), b_th.contiguous(),
            I_all.contiguous(), beta.contiguous(),
            grad_beta.contiguous(), grad_u.contiguous(), grad_v_th.contiguous(),
            grad_raw_beta, grad_raw_alpha, grad_raw_th, grad_I_all,
            numel, DN,
            BLOCK=BLOCK,
        )

        # Bias gradients: reduce over non-DN dims
        grad_b_beta = grad_raw_beta.reshape(-1, DN).sum(dim=0)
        grad_b_alpha = grad_raw_alpha.reshape(-1, DN).sum(dim=0)
        grad_b_th = grad_raw_th.reshape(-1, DN).sum(dim=0)

        return grad_raw_beta, grad_b_beta, grad_raw_alpha, grad_b_alpha, grad_raw_th, grad_b_th, None, grad_I_all


print("=" * 70)
print("Triton fused modulation kernel benchmark")
print(f"TK={TK}, batch={batch}, DN={DN}")
print("=" * 70)

# Setup
raw_beta = torch.randn(TK, batch, DN, device=device, dtype=dtype)
raw_alpha = torch.randn(TK, batch, DN, device=device, dtype=dtype)
raw_th = torch.randn(TK, batch, DN, device=device, dtype=dtype)
I_all = torch.randn(TK, batch, DN, device=device, dtype=dtype)
b_beta = torch.randn(DN, device=device, dtype=dtype)
b_alpha = torch.randn(DN, device=device, dtype=dtype)
b_th = torch.randn(DN, device=device, dtype=dtype)
v_th_min = 0.1

print("\n--- Forward only ---")

t_sep = bench_op("Separate (baseline)", lambda: (
    torch.sigmoid(raw_beta + b_beta),
    F.softplus(raw_alpha + b_alpha) * I_all,
    v_th_min + torch.abs(raw_th + b_th),
))

t_triton = bench_op("Triton fused kernel", lambda: _TritonFusedModulation.apply(
    raw_beta, b_beta, raw_alpha, b_alpha, raw_th, b_th, v_th_min, I_all,
))

@torch.compile(backend='inductor', fullgraph=True)
def _compile_mod(raw_beta, b_beta, raw_alpha, b_alpha, raw_th, b_th, v_th_min, I_all):
    beta = torch.sigmoid(raw_beta + b_beta)
    alpha = F.softplus(raw_alpha + b_alpha)
    v_th = v_th_min + torch.abs(raw_th + b_th)
    u = alpha * I_all
    return beta, u, v_th

# Warm up compile
print("  [Compiling torch.compile version...]")
_ = _compile_mod(raw_beta, b_beta, raw_alpha, b_alpha, raw_th, b_th, v_th_min, I_all)
torch.cuda.synchronize()

t_compile = bench_op("torch.compile fused", lambda: _compile_mod(
    raw_beta, b_beta, raw_alpha, b_alpha, raw_th, b_th, v_th_min, I_all,
))

print(f"\n  Triton vs Separate: {t_sep/t_triton:.2f}x")
print(f"  Compile vs Separate: {t_sep/t_compile:.2f}x")

# Verify correctness
print("\n--- Correctness (fp32) ---")
raw_beta_f = raw_beta.float()
raw_alpha_f = raw_alpha.float()
raw_th_f = raw_th.float()
I_all_f = I_all.float()
b_beta_f = b_beta.float()
b_alpha_f = b_alpha.float()
b_th_f = b_th.float()

ref = (
    torch.sigmoid(raw_beta_f + b_beta_f),
    F.softplus(raw_alpha_f + b_alpha_f) * I_all_f,
    v_th_min + torch.abs(raw_th_f + b_th_f),
)
triton_out = _TritonFusedModulation.apply(
    raw_beta_f, b_beta_f, raw_alpha_f, b_alpha_f, raw_th_f, b_th_f, v_th_min, I_all_f,
)
for i, name in enumerate(['beta', 'u', 'v_th']):
    diff = (ref[i] - triton_out[i]).abs().max().item()
    print(f"  {name}: max_diff={diff:.2e} {'OK' if diff < 1e-5 else 'FAIL'}")

print("\nDone.")
