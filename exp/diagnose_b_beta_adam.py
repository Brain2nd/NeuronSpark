"""
诊断 b_beta 在 Adam 优化器下的实际更新行为：
- 多步 forward/backward 看梯度方向一致性
- 对比 b_beta 和 W_in 的 Adam 更新幅度
- 验证 per-neuron 梯度方向是否一致（还是随机噪声）
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F
import torch.optim as optim
from spikingjelly.activation_based import functional

from model import SNNLanguageModel


def run_multi_step_diagnostic(D=256, N=8, num_layers=4, D_ff=512,
                               seq_len=32, batch_size=2, num_steps=10):
    """
    用小模型做 num_steps 步训练，详细追踪 b_beta 的梯度和更新。
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = SNNLanguageModel(
        vocab_size=6144, D=D, N=N, num_layers=num_layers, D_ff=D_ff,
    ).to(device)

    # 记录初始值
    b_beta_init = {}
    for i, layer_module in enumerate(model.layers):
        b_beta_init[i] = layer_module.snn_block.b_beta.data.clone()

    # 设置优化器（和训练一样的分组）
    _pg = model.get_param_groups()
    _neuron_keys = {'input_neurons', 'b_beta', 'b_alpha', 'b_th',
                    'block_output_neuron', 'ffn_neurons', 'output_neuron'}
    neuron_params = [p for k in _neuron_keys for p in _pg[k]]
    other_params = [p for k, ps in _pg.items() if k not in _neuron_keys for p in ps]

    neuron_lr_mult = 10.0
    base_lr = 2e-4

    optimizer = optim.Adam([
        {'params': other_params, 'lr': base_lr, 'lr_mult': 1.0},
        {'params': neuron_params, 'lr': base_lr * neuron_lr_mult,
         'lr_mult': float(neuron_lr_mult)},
    ])
    scaler = torch.amp.GradScaler('cuda', enabled=True)

    # 追踪每步的梯度
    grad_history = {i: [] for i in range(num_layers)}
    # 追踪对比参数
    w_in_grad_history = {i: [] for i in range(num_layers)}

    print(f"Model: D={D}, N={N}, Layers={num_layers}, D_ff={D_ff}")
    print(f"b_beta shape per layer: ({D*N},)")
    print(f"Base LR={base_lr}, Neuron LR mult={neuron_lr_mult}")
    print()

    for step in range(num_steps):
        model.train()
        optimizer.zero_grad()

        # 随机数据
        token_ids = torch.randint(1, model.vocab_size, (batch_size, seq_len), device=device)
        target_ids = torch.randint(1, model.vocab_size, (batch_size, seq_len), device=device)

        # Forward
        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            out = model(token_ids, target_ids)
            loss = out.last_loss.mean()

        # Backward
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)

        # 记录原始梯度
        for i, layer_module in enumerate(model.layers):
            g = layer_module.snn_block.b_beta.grad
            if g is not None:
                grad_history[i].append(g.clone())
            g_w = layer_module.snn_block.W_in.weight.grad
            if g_w is not None:
                w_in_grad_history[i].append(g_w.abs().mean().item())

        # 补偿
        model.compensate_modulation_gradients()

        # 报告
        if step == 0 or step == num_steps - 1:
            print(f"Step {step}: loss={loss.item():.4f}")
            for i in [0, num_layers - 1]:
                block = model.layers[i].snn_block
                g = block.b_beta.grad
                if g is not None:
                    print(f"  L{i} b_beta: grad_mean={g.mean().item():.8f}, "
                          f"|grad|_mean={g.abs().mean().item():.8f}, "
                          f"grad_max={g.abs().max().item():.8f}")

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()

    # ====== 分析梯度一致性 ======
    print("\n" + "=" * 80)
    print("梯度方向一致性分析（across steps）")
    print("=" * 80)
    print("如果 sign_consistency ≈ 0.5 → 梯度方向随机（无信号）")
    print("如果 sign_consistency > 0.7 → 梯度方向一致（有信号）")
    print()

    for i in [0, num_layers - 1]:
        grads = grad_history[i]
        if len(grads) < 2:
            continue

        # 堆叠所有步的梯度 (num_steps, DN)
        grad_stack = torch.stack(grads, dim=0)  # (num_steps, DN)

        # 每个神经元跨步的符号一致性
        signs = (grad_stack > 0).float()
        sign_consistency = signs.mean(dim=0)  # (DN,) — 接近 0.5 = 随机, 接近 0/1 = 一致

        # 统计
        mostly_positive = (sign_consistency > 0.7).sum().item()
        mostly_negative = (sign_consistency < 0.3).sum().item()
        random_sign = ((sign_consistency >= 0.3) & (sign_consistency <= 0.7)).sum().item()
        total = sign_consistency.numel()

        print(f"Layer {i} ({total} neurons):")
        print(f"  一致正梯度 (>70%): {mostly_positive} ({100*mostly_positive/total:.1f}%)")
        print(f"  一致负梯度 (<30%): {mostly_negative} ({100*mostly_negative/total:.1f}%)")
        print(f"  随机方向 (30-70%): {random_sign} ({100*random_sign/total:.1f}%)")
        print(f"  平均 sign_consistency: {sign_consistency.mean().item():.4f}")

        # 按 N 组分析
        DN = D * N
        grouped = sign_consistency.reshape(D, N)
        print(f"  按 N 组的 sign_consistency:")
        for n in range(N):
            sc = grouped[:, n]
            print(f"    N={n} (β≈{torch.sigmoid(b_beta_init[i][n*D:(n+1)*D] if DN > 0 else torch.tensor(0.0)).mean().item():.3f}): "
                  f"mean={sc.mean().item():.4f}, std={sc.std().item():.4f}")

    # ====== b_beta 实际变化量 ======
    print("\n" + "=" * 80)
    print(f"b_beta 实际变化量 (after {num_steps} steps)")
    print("=" * 80)

    for i in [0, num_layers - 1]:
        b_now = model.layers[i].snn_block.b_beta.data
        b_init = b_beta_init[i]
        delta = (b_now - b_init)

        print(f"Layer {i}:")
        print(f"  Δb_beta: mean={delta.mean().item():.8f}, |mean|={delta.abs().mean().item():.8f}, "
              f"max={delta.abs().max().item():.8f}")
        print(f"  相对变化: {delta.abs().mean().item() / b_init.abs().mean().item() * 100:.4f}%")

        # 按 N 组
        delta_grouped = delta.reshape(D, N)
        b_init_grouped = b_init.reshape(D, N)
        for n in range(N):
            d = delta_grouped[:, n]
            b0 = b_init_grouped[:, n]
            print(f"    N={n}: Δb mean={d.mean().item():.8f}, |Δb| mean={d.abs().mean().item():.8f}, "
                  f"β init={torch.sigmoid(b0).mean().item():.3f}")

    # ====== 对比 W_in 梯度 ======
    print("\n" + "=" * 80)
    print("对比 W_in 梯度幅值")
    print("=" * 80)
    for i in [0, num_layers - 1]:
        w_grads = w_in_grad_history[i]
        b_grads = [g.abs().mean().item() for g in grad_history[i]]
        print(f"Layer {i}:")
        print(f"  W_in |grad| mean: {sum(w_grads)/len(w_grads):.8f}")
        print(f"  b_beta |grad| mean (raw): {sum(b_grads)/len(b_grads):.8f}")
        print(f"  ratio W_in/b_beta: {sum(w_grads)/sum(b_grads):.1f}x")


if __name__ == '__main__':
    run_multi_step_diagnostic()
