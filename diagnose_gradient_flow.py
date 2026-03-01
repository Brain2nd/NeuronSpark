"""
诊断梯度流衰减：分别测量层间梯度和参数梯度。

测量三个层面：
1. 层间梯度 (inter-layer): 每层 OUTPUT 处的梯度范数 → 验证 binary_residual skip 是否保持
2. 参数梯度 (param): 每层 W_in.weight 的梯度范数 → 当前观测到 6 数量级衰减
3. 处理路径梯度 (processing): binary_residual 的 spike_a 输入梯度 → 内部处理衰减

用法:
    python diagnose_gradient_flow.py
"""

import torch
import torch.nn.functional as F
from model import SNNLanguageModel
from spikingjelly.activation_based import functional

# 禁用 gradient checkpoint 以便安装 hooks
torch.manual_seed(42)
device = 'cuda' if torch.cuda.is_available() else 'cpu'

print("=" * 70)
print("梯度流诊断: 定位 6 数量级衰减的根因")
print("=" * 70)

# 构建模型 (小配置, 快速诊断)
model = SNNLanguageModel(
    vocab_size=6144, D=1024, N=8, K=16,
    num_layers=20, D_ff=3072, v_th_min=0.1,
).to(device).to(torch.bfloat16)

# 禁用 gradient checkpoint: 直接 forward, 不经过 checkpoint
# 方法: 替换 snn_forward 为无 checkpoint 版本
original_snn_forward = model.snn_forward

def _reset_snn(mod):
    inner = getattr(mod, '_fsdp_wrapped_module', mod)
    functional.reset_net(inner)

def snn_forward_no_ckpt(spike_seq):
    spike = spike_seq
    for layer_module in model.layers:
        _reset_snn(layer_module)
        spike = layer_module(spike)
    return spike

model.snn_forward = snn_forward_no_ckpt

# 注册 hooks 捕获层间梯度
layer_output_grads = {}
layer_input_grads = {}

def make_output_hook(layer_idx):
    def hook(module, grad_input, grad_output):
        # grad_output[0] = 该层 output 的梯度
        if grad_output[0] is not None:
            layer_output_grads[layer_idx] = grad_output[0].detach().float().norm().item()
    return hook

def make_input_hook(layer_idx):
    def hook(module, grad_input, grad_output):
        # grad_input[0] = 该层 input (spike_in) 的梯度
        if grad_input[0] is not None:
            layer_input_grads[layer_idx] = grad_input[0].detach().float().norm().item()
    return hook

hooks = []
for i, layer in enumerate(model.layers):
    hooks.append(layer.register_full_backward_hook(make_output_hook(i)))
    hooks.append(layer.register_full_backward_hook(make_input_hook(i)))

# 也在 binary_residual 上安装 hooks (通过 SNNBlock 和 SNNFFN 的内部)
block_spike_out_grads = {}  # SNNBlock 内部 spike_out (处理路径) 的梯度
ffn_spike_out_grads = {}    # SNNFFN 内部 spike_out (处理路径) 的梯度

# 用 tensor hook 捕获 binary_residual 的 spike_a 梯度
# 方法: 在 forward 时注册 tensor hook

# Forward pass
batch_size = 2
seq_len = 64  # 小序列, 快速诊断
token_ids = torch.randint(1, 6144, (batch_size, seq_len), device=device)
target_ids = torch.randint(1, 6144, (batch_size, seq_len), device=device)

print(f"\n配置: batch={batch_size}, seq_len={seq_len}, layers=20, D=1024")
print(f"无 gradient checkpoint, 直接 forward")

with torch.amp.autocast('cuda', dtype=torch.bfloat16):
    output = model(token_ids, target_ids)
    loss = output.last_loss.mean()

print(f"\nLoss: {loss.item():.4f}")

# Backward
loss.backward()

# 清理 hooks
for h in hooks:
    h.remove()

# ==== 报告 1: 层间梯度 (grad_output at each layer) ====
print("\n" + "=" * 70)
print("1. 层间梯度 (每层 OUTPUT 处的梯度范数)")
print("   如果 binary_residual STE 正确保持, 应该各层近似相等")
print("=" * 70)

if layer_output_grads:
    for i in sorted(layer_output_grads.keys()):
        g = layer_output_grads[i]
        print(f"  Layer {i:2d} output grad norm: {g:.4e}")

    g_19 = layer_output_grads.get(19, 1.0)
    g_0 = layer_output_grads.get(0, 1.0)
    if g_19 > 0 and g_0 > 0:
        ratio = g_0 / g_19
        print(f"\n  Layer 0 / Layer 19 = {ratio:.2f}x")
else:
    print("  (未捕获到 output 梯度)")

# ==== 报告 2: 层输入梯度 ====
print("\n" + "=" * 70)
print("2. 层间梯度 (每层 INPUT 处的梯度范数)")
print("   = 前一层的 output 梯度 + 处理路径反传")
print("=" * 70)

if layer_input_grads:
    for i in sorted(layer_input_grads.keys()):
        g = layer_input_grads[i]
        print(f"  Layer {i:2d} input grad norm: {g:.4e}")

    g_19 = layer_input_grads.get(19, 1.0)
    g_0 = layer_input_grads.get(0, 1.0)
    if g_19 > 0 and g_0 > 0:
        ratio = g_0 / g_19
        print(f"\n  Layer 0 / Layer 19 = {ratio:.2f}x")
else:
    print("  (未捕获到 input 梯度)")

# ==== 报告 3: 参数梯度 ====
print("\n" + "=" * 70)
print("3. 参数梯度 (每层 W_in.weight.grad 范数)")
print("   这是之前观测到 6 数量级衰减的指标")
print("=" * 70)

for i, layer in enumerate(model.layers):
    g = layer.snn_block.W_in.weight.grad
    if g is not None:
        norm = g.float().norm().item()
        print(f"  Layer {i:2d} W_in.weight.grad norm: {norm:.4e}")
    else:
        print(f"  Layer {i:2d} W_in.weight.grad: None")

# ==== 报告 4: 其他参数对比 ====
print("\n" + "=" * 70)
print("4. 各类参数梯度范数 (Layer 0 vs Layer 19)")
print("=" * 70)

param_names = [
    ('W_in', lambda l: l.snn_block.W_in.weight),
    ('W_out', lambda l: l.snn_block.W_out.weight),
    ('W_skip', lambda l: l.snn_block.W_skip.weight),
    ('W_gate', lambda l: l.snn_block.W_gate.weight),
    ('W_beta_x', lambda l: l.snn_block.W_beta_x.weight),
    ('b_beta', lambda l: l.snn_block.b_beta),
    ('b_th', lambda l: l.snn_block.b_th),
    ('block_output_v_th', lambda l: l.snn_block.output_neuron.v_th),
    ('ffn_gate_proj', lambda l: l.snn_ffn.gate_proj.weight),
    ('ffn_down_proj', lambda l: l.snn_ffn.down_proj.weight),
]

for name, getter in param_names:
    try:
        p0 = getter(model.layers[0])
        p19 = getter(model.layers[19])
        g0 = p0.grad.float().norm().item() if p0.grad is not None else 0.0
        g19 = p19.grad.float().norm().item() if p19.grad is not None else 0.0
        ratio = g0 / g19 if g19 > 0 else float('inf')
        print(f"  {name:20s}  L0={g0:.4e}  L19={g19:.4e}  ratio={ratio:.2f}x")
    except Exception as e:
        print(f"  {name:20s}  error: {e}")

# ==== 报告 5: fp16_decode 梯度分析 ====
print("\n" + "=" * 70)
print("5. fp16_decode 梯度分布 (不同位的梯度权重)")
print("   sign/exponent/mantissa 位的梯度大小差异")
print("=" * 70)

# 测试 fp16_decode 的梯度
from atomic_ops.fp16_codec import fp16_decode
test_spikes = torch.randn(seq_len * 16, batch_size, 1024, device=device, dtype=torch.bfloat16).clamp(0, 1).round()
test_spikes.requires_grad_(True)

decoded = fp16_decode(test_spikes, seq_len, K=16)
(decoded ** 2).sum().backward()

if test_spikes.grad is not None:
    # 按 K=16 分组查看梯度
    g = test_spikes.grad.float().reshape(seq_len, 16, batch_size, 1024)
    for k in range(16):
        norm = g[:, k].norm().item()
        bit_type = "sign" if k == 0 else ("exp" if k <= 5 else "mant")
        print(f"  Bit {k:2d} ({bit_type:4s}): grad norm = {norm:.4e}")

print("\n" + "=" * 70)
print("诊断完成")
print("=" * 70)
