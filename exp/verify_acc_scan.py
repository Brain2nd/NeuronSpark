"""
验证 accelerated-scan 替换后的正确性：新旧实现输出和梯度对比。
"""
import torch
import sys
sys.path.insert(0, '/home/dgxspark/Desktop/NeuronSpark')

from atomic_ops.parallel_scan import (
    _linear_recurrence_legacy,
    _linear_recurrence_acc,
    linear_recurrence,
    plif_parallel_forward,
    _HAS_ACC_SCAN,
)

torch.manual_seed(42)
device = 'cuda'

print(f"accelerated-scan available: {_HAS_ACC_SCAN}")
print()

# ============================================================
# Test 1: linear_recurrence 输出对比
# ============================================================
print("=" * 60)
print("Test 1: linear_recurrence 输出对比")
print("=" * 60)

K, batch, D = 8, 4, 512
beta = torch.sigmoid(torch.randn(K, batch, D, device=device))
u = torch.randn(K, batch, D, device=device)
v_init = torch.randn(batch, D, device=device)

V_old = _linear_recurrence_legacy(beta, u, v_init)
V_new = _linear_recurrence_acc(beta, u, v_init)

max_diff = (V_old - V_new).abs().max().item()
rel_diff = ((V_old - V_new).abs() / (V_old.abs() + 1e-8)).max().item()
print(f"  shape: ({K}, {batch}, {D})")
print(f"  max abs diff:  {max_diff:.2e}")
print(f"  max rel diff:  {rel_diff:.2e}")
print(f"  PASS: {max_diff < 1e-4}")
print()

# ============================================================
# Test 2: linear_recurrence v_init=0
# ============================================================
print("=" * 60)
print("Test 2: linear_recurrence v_init=0 (Phase 2 场景)")
print("=" * 60)

v_init_zero = torch.zeros(batch, D, device=device)
V_old_z = _linear_recurrence_legacy(beta, u, v_init_zero)
V_new_z = _linear_recurrence_acc(beta, u, v_init_zero)

max_diff_z = (V_old_z - V_new_z).abs().max().item()
print(f"  max abs diff:  {max_diff_z:.2e}")
print(f"  PASS: {max_diff_z < 1e-4}")
print()

# ============================================================
# Test 3: 梯度对比
# ============================================================
print("=" * 60)
print("Test 3: 梯度对比 (backward pass)")
print("=" * 60)

# 旧实现梯度（用叶子张量）
beta_logits1 = torch.randn(K, batch, D, device=device, requires_grad=True)
u1 = torch.randn(K, batch, D, device=device, requires_grad=True)
vi1 = torch.randn(batch, D, device=device, requires_grad=True)

V1 = _linear_recurrence_legacy(torch.sigmoid(beta_logits1), u1, vi1)
loss1 = V1.sum()
loss1.backward()

# 新实现梯度（相同输入）
beta_logits2 = beta_logits1.detach().clone().requires_grad_(True)
u2 = u1.detach().clone().requires_grad_(True)
vi2 = vi1.detach().clone().requires_grad_(True)

V2 = _linear_recurrence_acc(torch.sigmoid(beta_logits2), u2, vi2)
loss2 = V2.sum()
loss2.backward()

grad_beta_diff = (beta_logits1.grad - beta_logits2.grad).abs().max().item()
grad_u_diff = (u1.grad - u2.grad).abs().max().item()
grad_vi_diff = (vi1.grad - vi2.grad).abs().max().item()

print(f"  grad(beta) max diff: {grad_beta_diff:.2e}")
print(f"  grad(u)    max diff: {grad_u_diff:.2e}")
print(f"  grad(vi)   max diff: {grad_vi_diff:.2e}")
print(f"  PASS: {max(grad_beta_diff, grad_u_diff, grad_vi_diff) < 1e-3}")
print()

# ============================================================
# Test 4: plif_parallel_forward 完整对比
# ============================================================
print("=" * 60)
print("Test 4: plif_parallel_forward 完整对比")
print("=" * 60)

from spikingjelly.activation_based import surrogate

sf = surrogate.Sigmoid(alpha=4.0)

beta_p = torch.sigmoid(torch.randn(K, batch, D, device=device))
u_p = torch.randn(K, batch, D, device=device)
v_th_p = torch.abs(torch.randn(K, batch, D, device=device)) + 0.5
vi_p = torch.randn(batch, D, device=device)

# 用旧实现
import atomic_ops.parallel_scan as ps
ps._HAS_ACC_SCAN = False  # 强制用旧实现
spike_old, Vpost_old, Vpre_old = plif_parallel_forward(
    beta_p.clone(), u_p.clone(), v_th_p.clone(), vi_p.clone(),
    max_iter=3, surrogate_function=sf,
)

# 用新实现
ps._HAS_ACC_SCAN = True
spike_new, Vpost_new, Vpre_new = plif_parallel_forward(
    beta_p.clone(), u_p.clone(), v_th_p.clone(), vi_p.clone(),
    max_iter=3, surrogate_function=sf,
)

spike_diff = (spike_old - spike_new).abs().max().item()
Vpost_diff = (Vpost_old - Vpost_new).abs().max().item()
Vpre_diff = (Vpre_old - Vpre_new).abs().max().item()

print(f"  spike  max diff: {spike_diff:.2e}")
print(f"  V_post max diff: {Vpost_diff:.2e}")
print(f"  V_pre  max diff: {Vpre_diff:.2e}")
print(f"  spike pattern identical: {torch.equal(spike_old, spike_new)}")
print(f"  PASS: {max(spike_diff, Vpost_diff, Vpre_diff) < 1e-3}")
print()

# ============================================================
# Test 5: plif_parallel_forward 梯度对比
# ============================================================
print("=" * 60)
print("Test 5: plif_parallel_forward 梯度对比")
print("=" * 60)

bl_a = torch.randn(K, batch, D, device=device, requires_grad=True)
u_a = torch.randn(K, batch, D, device=device, requires_grad=True)
vth_logits_a = torch.randn(K, batch, D, device=device, requires_grad=True)
vi_a = torch.randn(batch, D, device=device, requires_grad=True)

bl_b = bl_a.detach().clone().requires_grad_(True)
u_b = u_a.detach().clone().requires_grad_(True)
vth_logits_b = vth_logits_a.detach().clone().requires_grad_(True)
vi_b = vi_a.detach().clone().requires_grad_(True)

ps._HAS_ACC_SCAN = False
s_a, vp_a, _ = plif_parallel_forward(
    torch.sigmoid(bl_a), u_a, torch.abs(vth_logits_a) + 0.5, vi_a, 3, sf)
(s_a.sum() + vp_a.sum()).backward()

ps._HAS_ACC_SCAN = True
s_b, vp_b, _ = plif_parallel_forward(
    torch.sigmoid(bl_b), u_b, torch.abs(vth_logits_b) + 0.5, vi_b, 3, sf)
(s_b.sum() + vp_b.sum()).backward()

for name, ga, gb in [
    ("beta_l", bl_a.grad, bl_b.grad),
    ("u", u_a.grad, u_b.grad),
    ("vth_l", vth_logits_a.grad, vth_logits_b.grad),
    ("v_init", vi_a.grad, vi_b.grad),
]:
    diff = (ga - gb).abs().max().item()
    print(f"  grad({name:6s}) max diff: {diff:.2e}")

all_grad_diffs = [
    (bl_a.grad - bl_b.grad).abs().max().item(),
    (u_a.grad - u_b.grad).abs().max().item(),
    (vth_logits_a.grad - vth_logits_b.grad).abs().max().item(),
    (vi_a.grad - vi_b.grad).abs().max().item(),
]
print(f"  PASS: {max(all_grad_diffs) < 1e-2}")
print()

# ============================================================
# Summary
# ============================================================
print("=" * 60)
print("SUMMARY")
print("=" * 60)
all_pass = (
    max_diff < 1e-4
    and max_diff_z < 1e-4
    and max(grad_beta_diff, grad_u_diff, grad_vi_diff) < 1e-3
    and max(spike_diff, Vpost_diff, Vpre_diff) < 1e-3
    and max(all_grad_diffs) < 1e-2
)
print(f"  ALL TESTS PASSED: {all_pass}")
