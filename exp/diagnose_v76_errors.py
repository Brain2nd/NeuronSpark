"""
v7.6 误差诊断：逐个隔离每个优化引入的数值差异。

对比 v7.5 基线行为，精确定位每个误差来源。
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F
from spikingjelly.activation_based import surrogate, functional

torch.manual_seed(42)
torch.backends.cudnn.deterministic = True

device = 'cuda'
dtype = torch.bfloat16

print("=" * 70)
print("v7.6 误差诊断: 逐个隔离每个优化的数值差异")
print("=" * 70)
print()

from atomic_ops.parallel_scan import (
    plif_rowparam_forward, plif_parallel_forward,
    _HAS_TRITON, _TritonPLIFRowParamForward, _TritonPLIFRowParam1DForward,
)

surr = surrogate.Sigmoid(alpha=4.0)
alpha = float(surr.alpha)


# ============================================================
# 误差源 1: 1D row-param kernel forward 精度
# v7.5: expand(batch, D).contiguous() → expanded kernel
# v7.6: 1D (D,) → 1D kernel
# ============================================================
print("【误差源 1】1D kernel FORWARD vs expanded kernel")
print("-" * 50)

for test_dtype, label in [(torch.float32, "fp32"), (torch.bfloat16, "bf16")]:
    K, batch, D = 16, 2, 768
    torch.manual_seed(42)
    beta_1d = torch.sigmoid(torch.randn(D, device=device, dtype=test_dtype))
    v_th_1d = torch.abs(torch.randn(D, device=device, dtype=test_dtype)) + 0.1
    u = torch.randn(K, batch, D, device=device, dtype=test_dtype)
    v_init = torch.zeros(batch, D, device=device, dtype=test_dtype)

    # Expanded path (v7.5)
    beta_exp = beta_1d.unsqueeze(0).expand(batch, D).contiguous()
    v_th_exp = v_th_1d.unsqueeze(0).expand(batch, D).contiguous()
    spike_exp, vpost_exp = _TritonPLIFRowParamForward.apply(
        beta_exp, u, v_th_exp, v_init, alpha,
    )

    # 1D path (v7.6)
    spike_1d, vpost_1d = _TritonPLIFRowParam1DForward.apply(
        beta_1d, u, v_th_1d, v_init, alpha,
    )

    spike_diff = (spike_exp - spike_1d).abs().max().item()
    vpost_diff = (vpost_exp - vpost_1d).abs().max().item()
    print(f"  {label}: spike max_diff={spike_diff:.2e}, V_post max_diff={vpost_diff:.2e}")

print()


# ============================================================
# 误差源 2: 1D kernel BACKWARD 精度 (grad_beta, grad_vth)
# v7.5: per-row grad 存储为 (batch, D), 外部 .sum(dim=0) → bf16 累加
# v7.6: kernel 内 fp32 atomic_add → 转回 bf16
# ============================================================
print("【误差源 2】1D kernel BACKWARD vs expanded kernel")
print("-" * 50)

for test_dtype, label in [(torch.float32, "fp32"), (torch.bfloat16, "bf16")]:
    K, batch, D = 16, 2, 768
    torch.manual_seed(42)
    beta_1d = torch.sigmoid(torch.randn(D, device=device, dtype=test_dtype))
    v_th_1d = torch.abs(torch.randn(D, device=device, dtype=test_dtype)) + 0.1
    u = torch.randn(K, batch, D, device=device, dtype=test_dtype)
    v_init = torch.zeros(batch, D, device=device, dtype=test_dtype)

    # Expanded (v7.5)
    beta_a = beta_1d.clone().requires_grad_(True)
    v_th_a = v_th_1d.clone().requires_grad_(True)
    u_a = u.clone().requires_grad_(True)
    beta_exp = beta_a.unsqueeze(0).expand(batch, D).contiguous()
    v_th_exp = v_th_a.unsqueeze(0).expand(batch, D).contiguous()
    s_a, v_a = _TritonPLIFRowParamForward.apply(beta_exp, u_a, v_th_exp, v_init, alpha)
    (s_a.float().sum() + v_a.float().sum()).backward()

    # 1D (v7.6)
    beta_b = beta_1d.clone().requires_grad_(True)
    v_th_b = v_th_1d.clone().requires_grad_(True)
    u_b = u.clone().requires_grad_(True)
    s_b, v_b = _TritonPLIFRowParam1DForward.apply(beta_b, u_b, v_th_b, v_init, alpha)
    (s_b.float().sum() + v_b.float().sum()).backward()

    du_diff = (u_a.grad - u_b.grad).abs().max().item()
    db_abs = (beta_a.grad - beta_b.grad).abs()
    db_rel = db_abs / (beta_b.grad.abs() + 1e-8)
    dv_abs = (v_th_a.grad - v_th_b.grad).abs()
    dv_rel = dv_abs / (v_th_b.grad.abs() + 1e-8)

    print(f"  {label}:")
    print(f"    grad_u   max_abs_diff = {du_diff:.2e}")
    print(f"    grad_β   max_abs_diff = {db_abs.max().item():.2e}, max_rel_diff = {db_rel.max().item():.4f}")
    print(f"    grad_vth max_abs_diff = {dv_abs.max().item():.2e}, max_rel_diff = {dv_rel.max().item():.4f}")
    # 检查 grad_beta 大差异的具体值
    if label == "bf16":
        worst_idx = db_rel.argmax().item()
        print(f"    grad_β 最大 relerr 位置 [{worst_idx}]: expanded={beta_a.grad[worst_idx].item():.6e}, 1D={beta_b.grad[worst_idx].item():.6e}")
        # 看这些值的分布
        print(f"    grad_β relerr 分布: >{0.5}: {(db_rel>0.5).sum()}, >{0.1}: {(db_rel>0.1).sum()}, >{0.01}: {(db_rel>0.01).sum()}, total: {D}")

print()


# ============================================================
# 误差源 3: torch.addcmul vs mul+add
# v7.5: I_out_all * gate_all + I_skip_all  (两次 kernel)
# v7.6: torch.addcmul(I_skip_all, I_out_all, gate_all) (一次 FMA kernel)
# ============================================================
print("【误差源 3】addcmul vs mul+add")
print("-" * 50)

for test_dtype, label in [(torch.float32, "fp32"), (torch.bfloat16, "bf16")]:
    torch.manual_seed(42)
    a = torch.randn(16, 2, 768, device=device, dtype=test_dtype)
    b = torch.randn(16, 2, 768, device=device, dtype=test_dtype)
    c = torch.randn(16, 2, 768, device=device, dtype=test_dtype)

    old = b * c + a
    new = torch.addcmul(a, b, c)
    diff = (old - new).abs()
    print(f"  {label}: max_diff={diff.max().item():.2e}, mean_diff={diff.mean().item():.2e}, nonzero_diff={diff.nonzero().shape[0]}/{diff.numel()}")

print()


# ============================================================
# 误差源 4: 分块 PLIF pipeline vs 完整 PLIF scan
# v7.5: 一次 plif_parallel_forward(全 TK 步)
# v7.6: 分 chunk plif_parallel_forward(每 64 步) + 传递 v
# ============================================================
print("【误差源 4】分块 PLIF pipeline vs 完整 scan")
print("-" * 50)

for test_dtype, label in [(torch.float32, "fp32"), (torch.bfloat16, "bf16")]:
    TK, batch, DN = 512, 2, 256  # 用小 DN 加速
    CHUNK = 64
    torch.manual_seed(42)
    beta_all = torch.sigmoid(torch.randn(TK, batch, DN, device=device, dtype=test_dtype))
    u = torch.randn(TK, batch, DN, device=device, dtype=test_dtype)
    v_th = torch.abs(torch.randn(TK, batch, DN, device=device, dtype=test_dtype)) + 0.1
    v_init = torch.zeros(batch, DN, device=device, dtype=test_dtype)

    # v7.5: 完整 scan
    s_full, vp_full, _ = plif_parallel_forward(
        beta_all, u, v_th, v_init, max_iter=3, surrogate_function=surr)

    # v7.6: 分块 scan
    v = v_init
    s_chunks = []
    for k_start in range(0, TK, CHUNK):
        k_end = min(k_start + CHUNK, TK)
        s_chunk, vp_chunk, _ = plif_parallel_forward(
            beta_all[k_start:k_end], u[k_start:k_end],
            v_th[k_start:k_end], v, max_iter=3, surrogate_function=surr)
        v = vp_chunk[-1]
        s_chunks.append(s_chunk)
    s_chunked = torch.cat(s_chunks, dim=0)

    spike_diff = (s_full - s_chunked).abs().max().item()
    spike_mismatch = (s_full != s_chunked).sum().item()
    vpost_diff = (vp_full[-1] - v).abs().max().item()
    print(f"  {label}: spike max_diff={spike_diff:.2e}, mismatch_count={spike_mismatch}/{s_full.numel()}, V_post[-1] diff={vpost_diff:.2e}")

    # Backward 对比
    s_full_g = s_full.clone().requires_grad_(True)
    s_chunked_g = s_chunked.clone().requires_grad_(True)

print()


# ============================================================
# 误差源 5: sparse_row_gemm forward vs F.linear
# v7.5: F.linear(gated_flat, down_proj.weight)
# v7.6: sparse_row_gemm(gated_flat, down_proj.weight, mask)
# ============================================================
print("【误差源 5】sparse_row_gemm forward vs F.linear")
print("-" * 50)

from atomic_ops.sparse_gemm import sparse_row_gemm, _HAS_TRITON as _HAS_TRITON_SPARSE

if _HAS_TRITON_SPARSE:
    for test_dtype, label in [(torch.float32, "fp32"), (torch.bfloat16, "bf16")]:
        M, K_dim, N = 1024, 2304, 768
        torch.manual_seed(42)
        # 模拟 AND 门输出: ~70% 行稀疏（整行为零）
        x = torch.randn(M, K_dim, device=device, dtype=test_dtype)
        row_zero_mask = torch.rand(M, device=device) < 0.70  # 70% 行全零
        x[row_zero_mask] = 0.0
        weight = torch.randn(N, K_dim, device=device, dtype=test_dtype) * 0.01
        row_mask = x.any(dim=-1)

        # Dense (v7.5)
        y_dense = F.linear(x, weight)

        # Sparse (v7.6)
        y_sparse = sparse_row_gemm(x, weight, row_mask)

        diff = (y_dense - y_sparse).abs()
        # 分别看 active 行和 inactive 行
        active_diff = diff[row_mask]
        inactive_diff = diff[~row_mask]
        n_active = row_mask.sum().item()

        print(f"  {label} (row_sparsity={1-row_mask.float().mean().item():.1%}, {n_active}/{M} active rows):")
        if n_active > 0:
            print(f"    active rows   max_diff={active_diff.max().item():.2e}, mean_diff={active_diff.mean().item():.2e}")
        if (~row_mask).sum().item() > 0:
            print(f"    inactive rows max_diff={inactive_diff.max().item():.2e}")
else:
    print("  SKIP (no Triton)")

print()


# ============================================================
# 误差源 6: sparse_row_gemm backward (dW) vs dense
# v7.6 的 dX 已改为 dense F.linear, 只有 dW 用 Triton kernel
# ============================================================
print("【误差源 6】sparse_row_gemm backward dW vs dense")
print("-" * 50)

if _HAS_TRITON_SPARSE:
    for test_dtype, label in [(torch.float32, "fp32"), (torch.bfloat16, "bf16")]:
        M, K_dim, N = 1024, 2304, 768
        torch.manual_seed(42)
        x = torch.randn(M, K_dim, device=device, dtype=test_dtype)
        row_zero_mask = torch.rand(M, device=device) < 0.70
        x[row_zero_mask] = 0.0
        row_mask = x.any(dim=-1)

        # Dense path (使用 nn.Linear 确保 leaf grad)
        weight_data = torch.randn(N, K_dim, device=device, dtype=test_dtype) * 0.01
        x_d = x.clone().requires_grad_(True)
        w_d = weight_data.clone().requires_grad_(True)
        y_d = F.linear(x_d, w_d)
        y_d.sum().backward()

        # Sparse path
        x_s = x.clone().requires_grad_(True)
        w_s = weight_data.clone().requires_grad_(True)
        y_s = sparse_row_gemm(x_s, w_s, row_mask)
        y_s.sum().backward()

        dw_diff = (w_d.grad - w_s.grad).abs()
        dx_diff = (x_d.grad - x_s.grad).abs()
        print(f"  {label}:")
        print(f"    grad_W max_diff={dw_diff.max().item():.2e}, mean_diff={dw_diff.mean().item():.2e}")
        print(f"    grad_X max_diff={dx_diff.max().item():.2e}, mean_diff={dx_diff.mean().item():.2e}")
        # dX 应该完全一致（v7.6 的 dX 改为 dense F.linear）
        print(f"    grad_X exact? {(dx_diff == 0).all().item()}")
else:
    print("  SKIP (no Triton)")

print()


# ============================================================
# 综合测试: 完整 SNNDecoderLayer v7.5 vs v7.6
# 用 v7.5 backup 代码的逻辑手写 forward, 对比当前 v7.6 forward
# ============================================================
print("【综合】完整 SNNDecoderLayer forward 对比")
print("-" * 50)

from atomic_ops.snn_decoder_layer import SNNDecoderLayer

D, N, D_ff = 64, 4, 192
TK, batch = 32, 2

torch.manual_seed(42)
layer = SNNDecoderLayer(
    D=D, N=N, D_ff=D_ff, v_th_min=0.1,
    block_output_v_threshold=0.3,
    ffn_output_v_threshold=0.5,
    num_layers=2, layer_idx=0,
).to(device).to(dtype)

h = torch.randn(TK, batch, D, device=device, dtype=dtype)

# v7.6 forward
functional.reset_net(layer)
h_v76 = layer.forward_parallel(h.clone())

# 手写 v7.5 逻辑 forward (用同一 layer 的参数, 但走 v7.5 代码路径)
functional.reset_net(layer)
DN = D * N

# --- 子层1: input_neuron1 → snn_block → out_proj → residual ---
# input_neuron1: v7.5 用 expand+contiguous
in1 = layer.input_neuron1
beta_in1 = in1.beta  # (D,)
u_in1 = (1.0 - beta_in1) * h  # (D,) broadcast
v_init_in1 = torch.zeros(batch, D, device=device, dtype=dtype)
# v7.5 路径: expand
beta_in1_exp = beta_in1.unsqueeze(0).expand(batch, D).contiguous()
vth_in1_exp = in1.v_th.unsqueeze(0).expand(batch, D).contiguous()
spike_in1_v75, vpost_in1_v75 = plif_rowparam_forward(
    beta_in1_exp, u_in1, vth_in1_exp, v_init_in1, surrogate_function=in1.surrogate_function)
in1.v = vpost_in1_v75[-1].detach()

# snn_block forward_parallel — 手写 v7.5 逻辑
blk = layer.snn_block
flat_blk = spike_in1_v75.reshape(TK * batch, D)
I_all = F.linear(flat_blk, blk.W_in.weight).reshape(TK, batch, DN)
raw_beta = F.linear(flat_blk, blk.W_beta_x.weight).reshape(TK, batch, DN)
raw_alpha = F.linear(flat_blk, blk.W_alpha_x.weight).reshape(TK, batch, DN)
raw_th = F.linear(flat_blk, blk.W_th_x.weight).reshape(TK, batch, DN)
gate_all = torch.sigmoid(F.linear(flat_blk, blk.W_gate.weight).reshape(TK, batch, D))
I_skip_all = F.linear(flat_blk, blk.W_skip.weight).reshape(TK, batch, D)

# fused modulation (same torch.compile function)
from atomic_ops.snn_block import _fused_modulation
beta_all, u_hidden, v_th_all = _fused_modulation(
    raw_beta, blk.b_beta, raw_alpha, blk.b_alpha,
    raw_th, blk.b_th, blk.v_th_min, I_all)

v_init_hidden = torch.zeros(batch, DN, device=device, dtype=dtype)

# v7.5: 完整 PLIF scan (不分 chunk)
s_hidden, vpost_hidden, _ = plif_parallel_forward(
    beta_all, u_hidden, v_th_all, v_init_hidden, max_iter=3,
    surrogate_function=blk.hidden_neuron.surrogate_function)
blk.hidden_neuron.v = vpost_hidden[-1].detach()

s_flat = s_hidden.reshape(TK * batch, DN)
I_out_all = F.linear(s_flat, blk.W_out.weight).reshape(TK, batch, D)
# v7.5: mul + add (不是 addcmul)
I_total_all = I_out_all * gate_all + I_skip_all

# output neuron: v7.5 expand path
beta_out = blk.output_neuron.beta
u_out = (1.0 - beta_out) * I_total_all
v_init_out = torch.zeros(batch, D, device=device, dtype=dtype)
beta_out_exp = beta_out.unsqueeze(0).expand(batch, D).contiguous()
vth_out_exp = blk.output_neuron.v_th.unsqueeze(0).expand(batch, D).contiguous()
spike_blk_v75, vpost_blk_v75 = plif_rowparam_forward(
    beta_out_exp, u_out, vth_out_exp, v_init_out,
    surrogate_function=blk.output_neuron.surrogate_function)
blk.output_neuron.v = vpost_blk_v75[-1].detach()

h_after_blk_v75 = h + layer.block_out_proj(spike_blk_v75)

# --- 子层2: input_neuron2 → snn_ffn → out_proj → residual ---
in2 = layer.input_neuron2
beta_in2 = in2.beta
u_in2 = (1.0 - beta_in2) * h_after_blk_v75
v_init_in2 = torch.zeros(batch, D, device=device, dtype=dtype)
beta_in2_exp = beta_in2.unsqueeze(0).expand(batch, D).contiguous()
vth_in2_exp = in2.v_th.unsqueeze(0).expand(batch, D).contiguous()
spike_in2_v75, vpost_in2_v75 = plif_rowparam_forward(
    beta_in2_exp, u_in2, vth_in2_exp, v_init_in2, surrogate_function=in2.surrogate_function)
in2.v = vpost_in2_v75[-1].detach()

# snn_ffn forward_parallel — 手写 v7.5 逻辑
ffn = layer.snn_ffn
flat_ffn = spike_in2_v75.reshape(TK * batch, D)
D_ff_val = ffn.D_ff
W_gate_up = torch.cat([ffn.gate_proj.weight, ffn.up_proj.weight], dim=0)
I_gate_up = F.linear(flat_ffn, W_gate_up).reshape(TK, batch, 2 * D_ff_val)
I_skip_ffn = F.linear(flat_ffn, ffn.skip_proj.weight).reshape(TK, batch, D)

beta_gate = ffn.gate_neuron.beta
beta_up = ffn.up_neuron.beta
scale_row = torch.cat([1.0 - beta_gate, 1.0 - beta_up])
u_merged = I_gate_up * scale_row

# v7.5: expand path for gate+up PLIF
beta_row_ffn = torch.cat([beta_gate, beta_up])
beta_row_ffn_exp = beta_row_ffn.unsqueeze(0).expand(batch, 2 * D_ff_val).contiguous()
v_th_row_ffn = torch.cat([ffn.gate_neuron.v_th, ffn.up_neuron.v_th])
v_th_row_ffn_exp = v_th_row_ffn.unsqueeze(0).expand(batch, 2 * D_ff_val).contiguous()
v_init_gate = torch.zeros(batch, D_ff_val, device=device, dtype=dtype)
v_init_up = torch.zeros(batch, D_ff_val, device=device, dtype=dtype)
v_init_merged = torch.cat([v_init_gate, v_init_up], dim=-1)

spike_merged, vpost_merged = plif_rowparam_forward(
    beta_row_ffn_exp, u_merged, v_th_row_ffn_exp, v_init_merged,
    surrogate_function=ffn.gate_neuron.surrogate_function)
gate_spike = spike_merged[:, :, :D_ff_val]
up_spike = spike_merged[:, :, D_ff_val:]
ffn.gate_neuron.v = vpost_merged[-1, :, :D_ff_val].detach()
ffn.up_neuron.v = vpost_merged[-1, :, D_ff_val:].detach()

# v7.5: dense F.linear for down_proj
gated = gate_spike * up_spike
gated_flat = gated.reshape(TK * batch, D_ff_val)
I_out_ffn = F.linear(gated_flat, ffn.down_proj.weight).reshape(TK, batch, D) + I_skip_ffn

# v7.5: expand path for output neuron
beta_out_ffn = ffn.output_neuron.beta
u_out_ffn = (1.0 - beta_out_ffn) * I_out_ffn
v_init_out_ffn = torch.zeros(batch, D, device=device, dtype=dtype)
beta_out_ffn_exp = beta_out_ffn.unsqueeze(0).expand(batch, D).contiguous()
vth_out_ffn_exp = ffn.output_neuron.v_th.unsqueeze(0).expand(batch, D).contiguous()
spike_ffn_v75, vpost_ffn_v75 = plif_rowparam_forward(
    beta_out_ffn_exp, u_out_ffn, vth_out_ffn_exp, v_init_out_ffn,
    surrogate_function=ffn.output_neuron.surrogate_function)
ffn.output_neuron.v = vpost_ffn_v75[-1].detach()

h_v75 = h_after_blk_v75 + layer.ffn_out_proj(spike_ffn_v75)

# 对比
h_diff = (h_v75 - h_v76).abs()
print(f"  h_out max_diff = {h_diff.max().item():.2e}")
print(f"  h_out mean_diff = {h_diff.mean().item():.2e}")
print(f"  h_out rel_diff = {(h_diff / (h_v75.abs() + 1e-8)).max().item():.2e}")
if h_diff.max().item() == 0:
    print(f"  => EXACT MATCH")
else:
    print(f"  => MISMATCH detected!")

    # 分段定位
    # Step A: input_neuron1 输出对比
    functional.reset_net(layer)
    spike_in1_v76 = layer._input_neuron_parallel(layer.input_neuron1, h.clone())
    functional.reset_net(layer)
    # 重跑 v7.5
    in1 = layer.input_neuron1
    beta_in1 = in1.beta
    u_in1_2 = (1.0 - beta_in1) * h
    v_init_in1_2 = torch.zeros(batch, D, device=device, dtype=dtype)
    beta_in1_exp2 = beta_in1.unsqueeze(0).expand(batch, D).contiguous()
    vth_in1_exp2 = in1.v_th.unsqueeze(0).expand(batch, D).contiguous()
    spike_in1_v75_2, _ = plif_rowparam_forward(
        beta_in1_exp2, u_in1_2, vth_in1_exp2, v_init_in1_2,
        surrogate_function=in1.surrogate_function)

    in1_diff = (spike_in1_v76 - spike_in1_v75_2).abs().max().item()
    in1_mismatch = (spike_in1_v76 != spike_in1_v75_2).sum().item()
    print(f"\n  分段定位:")
    print(f"  [A] input_neuron1 spike diff: {in1_diff:.2e}, mismatch: {in1_mismatch}/{spike_in1_v76.numel()}")

print()
print("=" * 70)
print("诊断完成")
print("=" * 70)
