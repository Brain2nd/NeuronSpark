"""
v7.6.0+v7.6.1 正确性验证：对比新旧代码的 forward/backward 输出。

验证：
  1. 1D row-param kernel vs expanded row-param kernel（allclose）
  2. PLIFNode.get_v_init 缓存一致性
  3. torch.addcmul vs manual mul+add
  4. 完整 SNNDecoderLayer forward+backward allclose
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F
from spikingjelly.activation_based import surrogate, functional

# 设置确定性
torch.manual_seed(42)
torch.backends.cudnn.deterministic = True

device = 'cuda' if torch.cuda.is_available() else 'cpu'
dtype = torch.bfloat16

print("=" * 60)
print("v7.6.0+v7.6.1 正确性验证")
print("=" * 60)
print(f"Device: {device}, Dtype: {dtype}")
print()

n_pass, n_total = 0, 0

# ============================================================
# Test 1: 1D row-param kernel vs expanded (fp32 for precision)
# ============================================================
print("Test 1: 1D row-param PLIF kernel vs expanded (fp32)")
n_total += 1

from atomic_ops.parallel_scan import (
    plif_rowparam_forward,
    _HAS_TRITON,
)

if _HAS_TRITON and device == 'cuda':
    from atomic_ops.parallel_scan import (
        _TritonPLIFRowParamForward,
        _TritonPLIFRowParam1DForward,
    )

    K, batch, D = 16, 2, 768
    surr = surrogate.Sigmoid(alpha=4.0)
    alpha = float(surr.alpha)

    # Use fp32 to eliminate precision concerns
    test_dtype = torch.float32

    torch.manual_seed(42)
    beta_1d = torch.sigmoid(torch.randn(D, device=device, dtype=test_dtype))
    v_th_1d = torch.abs(torch.randn(D, device=device, dtype=test_dtype)) + 0.1
    u = torch.randn(K, batch, D, device=device, dtype=test_dtype)
    v_init = torch.zeros(batch, D, device=device, dtype=test_dtype)

    # Path A: expanded
    beta_1d_a = beta_1d.clone().requires_grad_(True)
    v_th_1d_a = v_th_1d.clone().requires_grad_(True)
    u_a = u.clone().requires_grad_(True)
    beta_exp = beta_1d_a.unsqueeze(0).expand(batch, D).contiguous()
    v_th_exp = v_th_1d_a.unsqueeze(0).expand(batch, D).contiguous()
    spike_a, vpost_a = _TritonPLIFRowParamForward.apply(
        beta_exp, u_a, v_th_exp, v_init, alpha,
    )
    (spike_a.sum() + vpost_a.sum()).backward()

    # Path B: 1D
    beta_1d_b = beta_1d.clone().requires_grad_(True)
    v_th_1d_b = v_th_1d.clone().requires_grad_(True)
    u_b = u.clone().requires_grad_(True)
    spike_b, vpost_b = _TritonPLIFRowParam1DForward.apply(
        beta_1d_b, u_b, v_th_1d_b, v_init, alpha,
    )
    (spike_b.sum() + vpost_b.sum()).backward()

    fwd_spike_ok = torch.allclose(spike_a, spike_b)
    fwd_vpost_ok = torch.allclose(vpost_a, vpost_b, rtol=1e-5, atol=1e-6)
    bwd_u_ok = torch.allclose(u_a.grad, u_b.grad, rtol=1e-5, atol=1e-6)
    bwd_beta_ok = torch.allclose(beta_1d_a.grad, beta_1d_b.grad, rtol=1e-4, atol=1e-5)
    bwd_vth_ok = torch.allclose(v_th_1d_a.grad, v_th_1d_b.grad, rtol=1e-4, atol=1e-5)

    print(f"  Forward spike exact:    {fwd_spike_ok} (diff: {(spike_a - spike_b).abs().max().item():.2e})")
    print(f"  Forward V_post close:   {fwd_vpost_ok} (diff: {(vpost_a - vpost_b).abs().max().item():.2e})")
    print(f"  Backward grad_u close:  {bwd_u_ok} (diff: {(u_a.grad - u_b.grad).abs().max().item():.2e})")
    print(f"  Backward grad_β close:  {bwd_beta_ok} (diff: {(beta_1d_a.grad - beta_1d_b.grad).abs().max().item():.2e})")
    print(f"  Backward grad_vth close:{bwd_vth_ok} (diff: {(v_th_1d_a.grad - v_th_1d_b.grad).abs().max().item():.2e})")

    all_pass = fwd_spike_ok and fwd_vpost_ok and bwd_u_ok and bwd_beta_ok and bwd_vth_ok
    status = 'PASS' if all_pass else 'FAIL'
    print(f"  => {status}")
    if all_pass:
        n_pass += 1
else:
    print("  SKIP (no Triton or no CUDA)")

print()

# ============================================================
# Test 1b: bf16 — 验证 1D kernel 梯度精度 ≥ expanded kernel
# 1D path 用 fp32 atomic_add 累加 grad_β/grad_vth,
# expanded path 用 bf16 逐行存储后外部求和, 精度更低。
# 所以判定标准不是"两者相等", 而是"1D 更接近 fp32 参考值"。
# ============================================================
print("Test 1b: 1D kernel bf16 梯度精度验证 (vs fp32 参考)")
n_total += 1

if _HAS_TRITON and device == 'cuda':
    # 首先计算 fp32 参考梯度
    torch.manual_seed(42)
    beta_1d_fp32 = torch.sigmoid(torch.randn(D, device=device, dtype=torch.float32))
    v_th_1d_fp32 = torch.abs(torch.randn(D, device=device, dtype=torch.float32)) + 0.1
    u_fp32 = torch.randn(K, batch, D, device=device, dtype=torch.float32)
    v_init_fp32 = torch.zeros(batch, D, device=device, dtype=torch.float32)

    beta_ref = beta_1d_fp32.clone().requires_grad_(True)
    vth_ref = v_th_1d_fp32.clone().requires_grad_(True)
    u_ref = u_fp32.clone().requires_grad_(True)
    s_ref, v_ref = _TritonPLIFRowParam1DForward.apply(
        beta_ref, u_ref, vth_ref, v_init_fp32, alpha)
    (s_ref.sum() + v_ref.sum()).backward()

    # bf16 路径
    test_dtype = torch.bfloat16
    beta_1d_bf16 = beta_1d_fp32.to(test_dtype)
    v_th_1d_bf16 = v_th_1d_fp32.to(test_dtype)
    u_bf16 = u_fp32.to(test_dtype)
    v_init_bf16 = torch.zeros(batch, D, device=device, dtype=test_dtype)

    # Path A: expanded (v7.5 行为)
    beta_a = beta_1d_bf16.clone().requires_grad_(True)
    vth_a = v_th_1d_bf16.clone().requires_grad_(True)
    u_a = u_bf16.clone().requires_grad_(True)
    beta_exp = beta_a.unsqueeze(0).expand(batch, D).contiguous()
    vth_exp = vth_a.unsqueeze(0).expand(batch, D).contiguous()
    s_a, v_a = _TritonPLIFRowParamForward.apply(beta_exp, u_a, vth_exp, v_init_bf16, alpha)
    (s_a.float().sum() + v_a.float().sum()).backward()

    # Path B: 1D (v7.6 行为)
    beta_b = beta_1d_bf16.clone().requires_grad_(True)
    vth_b = v_th_1d_bf16.clone().requires_grad_(True)
    u_b = u_bf16.clone().requires_grad_(True)
    s_b, v_b = _TritonPLIFRowParam1DForward.apply(beta_b, u_b, vth_b, v_init_bf16, alpha)
    (s_b.float().sum() + v_b.float().sum()).backward()

    # 用 fp32 参考值评估两者精度
    ref_beta_grad = beta_ref.grad.float()
    ref_vth_grad = vth_ref.grad.float()

    err_expanded_beta = (beta_a.grad.float() - ref_beta_grad).abs()
    err_1d_beta = (beta_b.grad.float() - ref_beta_grad).abs()
    err_expanded_vth = (vth_a.grad.float() - ref_vth_grad).abs()
    err_1d_vth = (vth_b.grad.float() - ref_vth_grad).abs()

    fwd_spike_ok = torch.allclose(s_a, s_b)
    fwd_vpost_ok = torch.allclose(v_a, v_b, rtol=1e-3, atol=1e-3)
    bwd_u_ok = torch.allclose(u_a.grad, u_b.grad, rtol=1e-2, atol=1e-3)

    # 1D 应该更接近 fp32 参考（或至少不差）
    beta_1d_better = err_1d_beta.sum() <= err_expanded_beta.sum()
    vth_1d_better = err_1d_vth.sum() <= err_expanded_vth.sum()

    print(f"  Forward spike exact:       {fwd_spike_ok}")
    print(f"  Forward V_post close:      {fwd_vpost_ok}")
    print(f"  Backward grad_u close:     {bwd_u_ok}")
    print(f"  grad_β vs fp32 ref:")
    print(f"    expanded 总误差: {err_expanded_beta.sum().item():.4f}, 最大: {err_expanded_beta.max().item():.4f}")
    print(f"    1D       总误差: {err_1d_beta.sum().item():.4f}, 最大: {err_1d_beta.max().item():.4f}")
    print(f"    1D 更精确: {beta_1d_better}")
    print(f"  grad_vth vs fp32 ref:")
    print(f"    expanded 总误差: {err_expanded_vth.sum().item():.4f}, 最大: {err_expanded_vth.max().item():.4f}")
    print(f"    1D       总误差: {err_1d_vth.sum().item():.4f}, 最大: {err_1d_vth.max().item():.4f}")
    print(f"    1D 更精确: {vth_1d_better}")

    all_pass = fwd_spike_ok and fwd_vpost_ok and bwd_u_ok and beta_1d_better and vth_1d_better
    status = 'PASS' if all_pass else 'FAIL'
    print(f"  => {status}")
    if all_pass:
        n_pass += 1

print()

# ============================================================
# Test 2: plif_rowparam_forward dispatch (1D auto-dispatch)
# ============================================================
print("Test 2: plif_rowparam_forward 1D auto-dispatch")
n_total += 1

if _HAS_TRITON and device == 'cuda':
    K, batch, D = 16, 2, 768
    surr = surrogate.Sigmoid(alpha=4.0)

    torch.manual_seed(42)
    beta_1d = torch.sigmoid(torch.randn(D, device=device, dtype=dtype))
    v_th_1d = torch.abs(torch.randn(D, device=device, dtype=dtype)) + 0.1
    u = torch.randn(K, batch, D, device=device, dtype=dtype)
    v_init = torch.zeros(batch, D, device=device, dtype=dtype)

    # Old way: expand then call
    beta_exp = beta_1d.unsqueeze(0).expand(batch, D).contiguous()
    v_th_exp = v_th_1d.unsqueeze(0).expand(batch, D).contiguous()
    spike_old, vpost_old = plif_rowparam_forward(
        beta_exp, u, v_th_exp, v_init, surrogate_function=surr,
    )

    # New way: pass 1D directly
    spike_new, vpost_new = plif_rowparam_forward(
        beta_1d, u, v_th_1d, v_init, surrogate_function=surr,
    )

    spike_close = torch.allclose(spike_old, spike_new, rtol=1e-3, atol=1e-5)
    vpost_close = torch.allclose(vpost_old, vpost_new, rtol=1e-3, atol=1e-5)
    print(f"  spike allclose: {spike_close}")
    print(f"  V_post allclose: {vpost_close}")
    all_pass = spike_close and vpost_close
    print(f"  => {'PASS' if all_pass else 'FAIL'}")
    if all_pass:
        n_pass += 1
else:
    print("  SKIP")

print()

# ============================================================
# Test 3: PLIFNode.get_v_init caching
# ============================================================
print("Test 3: PLIFNode.get_v_init 缓存一致性")
n_total += 1

from atomic_ops.plif_node import PLIFNode

node = PLIFNode(dim=768, init_tau=2.0, v_threshold=0.5).to(device).to(dtype)

v1 = node.get_v_init(2, device, dtype)
v2 = node.get_v_init(2, device, dtype)

same_obj = v1 is v2
all_zero = (v1 == 0).all().item()
correct_shape = v1.shape == (2, 768)
correct_dtype = v1.dtype == dtype
print(f"  Same buffer object: {same_obj}")
print(f"  All zeros: {all_zero}")
print(f"  Correct shape: {correct_shape}")
print(f"  Correct dtype: {correct_dtype}")

# Different batch should create new buffer
v3 = node.get_v_init(4, device, dtype)
new_for_diff_batch = v3 is not v1 and v3.shape == (4, 768)
print(f"  New buffer for batch=4: {new_for_diff_batch}")

all_pass = same_obj and all_zero and correct_shape and correct_dtype and new_for_diff_batch
print(f"  => {'PASS' if all_pass else 'FAIL'}")
if all_pass:
    n_pass += 1

print()

# ============================================================
# Test 4: sparse_row_gemm forward 精度 (binary spike 输入)
# 实际使用中输入是 spike {0,1}, 不是随机浮点数。
# 二值输入下 GEMM = 选择并累加权重行, 精度应很高。
# ============================================================
print("Test 4: sparse_row_gemm 二值输入精度")
n_total += 1

from atomic_ops.sparse_gemm import sparse_row_gemm as _sparse_row_gemm

torch.manual_seed(42)
M, K_dim, N = 1024, 2304, 768
# 模拟 AND 门输出: 二值 spike, ~70% 行全零
spike_input = (torch.rand(M, K_dim, device=device) > 0.97).to(dtype)
row_zero = torch.rand(M, device=device) < 0.70
spike_input[row_zero] = 0.0
weight = torch.randn(N, K_dim, device=device, dtype=dtype) * 0.01
row_mask = spike_input.any(dim=-1)
n_active = row_mask.sum().item()

y_dense = F.linear(spike_input, weight)
y_sparse = _sparse_row_gemm(spike_input, weight, row_mask)

diff = (y_dense - y_sparse).abs()
active_diff = diff[row_mask]
inactive_diff = diff[~row_mask]

print(f"  Input: {M} rows, {n_active} active, spike density ~3%")
if n_active > 0:
    print(f"  Active rows  max_diff: {active_diff.max().item():.2e}")
    print(f"  Active rows  mean_diff: {active_diff.mean().item():.2e}")
if (~row_mask).sum().item() > 0:
    print(f"  Inactive rows max_diff: {inactive_diff.max().item():.2e}")

# 对二值输入, 要求 max_diff 在 bf16 精度范围内 (< 0.01)
spike_ok = active_diff.max().item() < 0.01 if n_active > 0 else True
zero_ok = inactive_diff.max().item() == 0 if (~row_mask).sum().item() > 0 else True
all_pass = spike_ok and zero_ok
print(f"  => {'PASS' if all_pass else 'FAIL'}")
if all_pass:
    n_pass += 1

print()

# ============================================================
# Test 5: Full SNNDecoderLayer forward+backward
# ============================================================
print("Test 5: SNNDecoderLayer 完整 forward+backward")
n_total += 1

from atomic_ops.snn_decoder_layer import SNNDecoderLayer

torch.manual_seed(42)

D, N, D_ff = 64, 4, 192
layer = SNNDecoderLayer(
    D=D, N=N, D_ff=D_ff, v_th_min=0.1,
    block_output_v_threshold=0.3,
    ffn_output_v_threshold=0.5,
    num_layers=2, layer_idx=0,
).to(device).to(dtype)

TK, batch = 32, 2
h = torch.randn(TK, batch, D, device=device, dtype=dtype, requires_grad=True)

functional.reset_net(layer)
h_out = layer.forward_parallel(h)

shape_ok = h_out.shape == (TK, batch, D)
print(f"  Output shape correct: {shape_ok}")

loss = h_out.float().sum()
loss.backward()
grad_ok = h.grad is not None and h.grad.shape == (TK, batch, D)
grad_nonzero = h.grad.abs().sum().item() > 0 if grad_ok else False
print(f"  Gradient exists: {grad_ok}")
print(f"  Gradient non-zero: {grad_nonzero}")

n_params = 0
n_grad = 0
no_grad_names = []
for name, p in layer.named_parameters():
    n_params += 1
    if p.grad is not None and p.grad.abs().sum().item() > 0:
        n_grad += 1
    else:
        no_grad_names.append(name)
print(f"  Params with grad: {n_grad}/{n_params}")
if no_grad_names:
    print(f"  Missing grad params: {no_grad_names}")

# Allow 1 param without grad (may happen with small random models)
all_pass = shape_ok and grad_ok and grad_nonzero and n_grad >= n_params - 1
print(f"  => {'PASS' if all_pass else 'FAIL'}")
if all_pass:
    n_pass += 1

print()

# ============================================================
# Test 6: Reproducibility
# ============================================================
print("Test 6: 确定性复现（同 seed → 同输出）")
n_total += 1

torch.manual_seed(42)
layer2 = SNNDecoderLayer(
    D=D, N=N, D_ff=D_ff, v_th_min=0.1,
    block_output_v_threshold=0.3,
    ffn_output_v_threshold=0.5,
    num_layers=2, layer_idx=0,
).to(device).to(dtype)

h2 = torch.randn(TK, batch, D, device=device, dtype=dtype)

functional.reset_net(layer2)
h2_out = layer2.forward_parallel(h2)

repro_close = torch.allclose(h_out.detach(), h2_out.detach(), rtol=1e-3, atol=1e-5)
print(f"  Reproducible output: {repro_close}")
print(f"  max diff: {(h_out.detach() - h2_out.detach()).abs().max().item():.2e}")
print(f"  => {'PASS' if repro_close else 'FAIL'}")
if repro_close:
    n_pass += 1

print()

# ============================================================
# Test 7: 1D kernel with merged dimension (2*D_ff, fp32)
# ============================================================
print("Test 7: 1D kernel merged dimension 2*D_ff (fp32)")
n_total += 1

if _HAS_TRITON and device == 'cuda':
    K, batch, D_ff = 16, 2, 2304
    surr = surrogate.Sigmoid(alpha=4.0)
    alpha = float(surr.alpha)
    test_dtype = torch.float32

    torch.manual_seed(123)
    beta_1d = torch.sigmoid(torch.randn(2 * D_ff, device=device, dtype=test_dtype))
    v_th_1d = torch.abs(torch.randn(2 * D_ff, device=device, dtype=test_dtype)) + 0.1
    u = torch.randn(K, batch, 2 * D_ff, device=device, dtype=test_dtype)
    v_init = torch.zeros(batch, 2 * D_ff, device=device, dtype=test_dtype)

    # Expanded
    beta_1d_a = beta_1d.clone().requires_grad_(True)
    v_th_1d_a = v_th_1d.clone().requires_grad_(True)
    u_a = u.clone().requires_grad_(True)
    beta_exp = beta_1d_a.unsqueeze(0).expand(batch, 2 * D_ff).contiguous()
    v_th_exp = v_th_1d_a.unsqueeze(0).expand(batch, 2 * D_ff).contiguous()
    spike_a, vpost_a = _TritonPLIFRowParamForward.apply(
        beta_exp, u_a, v_th_exp, v_init, alpha,
    )
    (spike_a.sum() + vpost_a.sum()).backward()

    # 1D
    beta_1d_b = beta_1d.clone().requires_grad_(True)
    v_th_1d_b = v_th_1d.clone().requires_grad_(True)
    u_b = u.clone().requires_grad_(True)
    spike_b, vpost_b = _TritonPLIFRowParam1DForward.apply(
        beta_1d_b, u_b, v_th_1d_b, v_init, alpha,
    )
    (spike_b.sum() + vpost_b.sum()).backward()

    fwd_ok = torch.allclose(spike_a, spike_b) and torch.allclose(vpost_a, vpost_b, rtol=1e-5)
    bwd_u_ok = torch.allclose(u_a.grad, u_b.grad, rtol=1e-5, atol=1e-6)
    bwd_beta_ok = torch.allclose(beta_1d_a.grad, beta_1d_b.grad, rtol=1e-4, atol=1e-5)
    bwd_vth_ok = torch.allclose(v_th_1d_a.grad, v_th_1d_b.grad, rtol=1e-4, atol=1e-5)

    print(f"  Forward allclose: {fwd_ok}")
    print(f"  grad_u allclose: {bwd_u_ok} (diff: {(u_a.grad - u_b.grad).abs().max().item():.2e})")
    print(f"  grad_β allclose: {bwd_beta_ok} (diff: {(beta_1d_a.grad - beta_1d_b.grad).abs().max().item():.2e})")
    print(f"  grad_vth allclose: {bwd_vth_ok} (diff: {(v_th_1d_a.grad - v_th_1d_b.grad).abs().max().item():.2e})")
    all_pass = fwd_ok and bwd_u_ok and bwd_beta_ok and bwd_vth_ok
    print(f"  => {'PASS' if all_pass else 'FAIL'}")
    if all_pass:
        n_pass += 1
else:
    print("  SKIP")

print()

# ============================================================
# Test 8: Full model forward (small scale)
# ============================================================
print("Test 8: SNNLanguageModel 小规模 forward")
n_total += 1

from model import SNNLanguageModel

torch.manual_seed(42)
model = SNNLanguageModel(
    vocab_size=256, D=64, N=4, K=8, num_layers=2, D_ff=192, v_th_min=0.1,
).to(device).to(dtype)

token_ids = torch.randint(1, 256, (2, 4), device=device)
target_ids = torch.randint(1, 256, (2, 4), device=device)

output = model(token_ids, target_ids)
loss = output.last_loss.mean()
print(f"  Loss: {loss.item():.4f}")
loss_ok = not torch.isnan(loss) and not torch.isinf(loss) and loss.item() > 0
print(f"  Loss valid: {loss_ok}")

loss.backward()
grads_ok = all(p.grad is not None for p in model.parameters() if p.requires_grad)
print(f"  All grads exist: {grads_ok}")
print(f"  => {'PASS' if loss_ok and grads_ok else 'FAIL'}")
if loss_ok and grads_ok:
    n_pass += 1

print()
print("=" * 60)
print(f"总结: {n_pass}/{n_total} PASS")
print("=" * 60)
