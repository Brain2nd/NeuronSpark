"""
诊断 b_beta 梯度问题：
1. 加载 checkpoint，做一次 forward/backward
2. 报告所有参数组的梯度幅值
3. 特别关注 b_beta 的原始梯度 vs 补偿后梯度
4. 检查梯度流中间节点的幅值（定位衰减瓶颈）
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F
from spikingjelly.activation_based import functional

from model import SNNLanguageModel


def load_checkpoint(ckpt_path, device='cuda'):
    """从 checkpoint 加载模型"""
    ckpt = torch.load(ckpt_path, map_location=device)
    config = ckpt.get('model_config', ckpt.get('config', {}))

    model = SNNLanguageModel(
        vocab_size=config.get('vocab_size', 6144),
        D=config.get('D', 768),
        N=config.get('N', 8),
        K=config.get('K', 16),
        num_layers=config.get('num_layers', 20),
        D_ff=config.get('D_ff', 2304),
    ).to(device)

    state_dict = ckpt.get('model_state_dict', ckpt.get('model', ckpt))
    model.load_state_dict(state_dict)
    return model, config


def diagnose_gradients(model, device='cuda', seq_len=32, batch_size=2):
    """做一次 forward/backward，收集所有梯度信息"""
    model.train()
    model.zero_grad()

    # 随机输入
    token_ids = torch.randint(1, model.vocab_size, (batch_size, seq_len), device=device)
    target_ids = torch.randint(1, model.vocab_size, (batch_size, seq_len), device=device)

    # Forward
    for layer_module in model.layers:
        functional.reset_net(layer_module)
    functional.reset_net(model.output_neuron)

    spike_seq = model.encode(token_ids)
    h = spike_seq

    # 用 hook 捕获中间梯度
    intermediate_grads = {}

    def make_hook(name):
        def hook(grad):
            intermediate_grads[name] = grad.detach().clone()
        return hook

    for i, layer_module in enumerate(model.layers):
        if h.requires_grad and (i == 0 or i == len(model.layers) - 1):
            h.register_hook(make_hook(f'layer_{i}_input'))

        h = layer_module.forward_parallel(h)

        if h.requires_grad and (i == 0 or i == len(model.layers) - 1):
            h.register_hook(make_hook(f'layer_{i}_output'))

    logits = model.decode(h, seq_len)

    # Loss
    logits_flat = logits.reshape(-1, model.vocab_size)
    targets_flat = target_ids.reshape(-1)
    loss = F.cross_entropy(logits_flat, targets_flat, ignore_index=0)

    # Backward
    loss.backward()

    return loss.item(), intermediate_grads


def report_param_gradients(model):
    """报告所有参数组的梯度幅值"""
    groups = model.get_param_groups()

    print("\n" + "=" * 80)
    print("参数组梯度幅值报告")
    print("=" * 80)
    print(f"{'参数组':<25} {'参数数量':>8} {'grad mean':>12} {'grad std':>12} {'grad |mean|':>12} {'grad max':>12}")
    print("-" * 80)

    for name, params in groups.items():
        all_grads = []
        for p in params:
            if p.grad is not None:
                all_grads.append(p.grad.flatten())
        if not all_grads:
            print(f"{name:<25} {'N/A':>8} {'无梯度':>12}")
            continue

        g = torch.cat(all_grads)
        print(f"{name:<25} {g.numel():>8} {g.mean().item():>12.6f} {g.std().item():>12.6f} "
              f"{g.abs().mean().item():>12.6f} {g.abs().max().item():>12.6f}")

    # 特别报告每层 b_beta
    print("\n" + "=" * 80)
    print("逐层 b_beta 梯度详情")
    print("=" * 80)
    print(f"{'Layer':<8} {'b_beta值 mean':>14} {'β mean':>10} {'grad mean':>14} "
          f"{'grad |mean|':>14} {'grad std':>14} {'grad max':>14}")
    print("-" * 80)

    for i, layer_module in enumerate(model.layers):
        block = layer_module.snn_block
        b = block.b_beta.data
        beta = torch.sigmoid(b)
        g = block.b_beta.grad
        if g is not None:
            print(f"L{i:<6} {b.mean().item():>14.4f} {beta.mean().item():>10.4f} "
                  f"{g.mean().item():>14.8f} {g.abs().mean().item():>14.8f} "
                  f"{g.std().item():>14.8f} {g.abs().max().item():>14.8f}")
        else:
            print(f"L{i:<6} {b.mean().item():>14.4f} {beta.mean().item():>10.4f} {'无梯度':>14}")


def report_b_beta_distribution(model):
    """报告 b_beta 的分布（不只是 mean）"""
    print("\n" + "=" * 80)
    print("b_beta 值分布（per N group）")
    print("=" * 80)

    N = model.N
    D = model.D

    # 取 Layer 0 和 Layer 19 对比
    for layer_idx in [0, len(model.layers) - 1]:
        layer_module = model.layers[layer_idx]
        b = layer_module.snn_block.b_beta.data
        beta = torch.sigmoid(b)

        # b_beta 是 (D*N,) — 按 N 分组
        # 布局: [d0_n0, d0_n1, ..., d0_n7, d1_n0, ..., d1_n7, ...]
        # 即 repeat 模式: b_beta_per_n.repeat(D)
        b_grouped = b.reshape(D, N)  # (D, N)

        print(f"\nLayer {layer_idx}:")
        print(f"  {'N group':<10} {'b_beta mean':>12} {'b_beta std':>12} {'β mean':>10} {'β min':>10} {'β max':>10}")
        for n in range(N):
            bn = b_grouped[:, n]
            beta_n = torch.sigmoid(bn)
            print(f"  N={n:<8} {bn.mean().item():>12.4f} {bn.std().item():>12.4f} "
                  f"{beta_n.mean().item():>10.4f} {beta_n.min().item():>10.4f} {beta_n.max().item():>10.4f}")


def test_compensation_effect(model):
    """对比补偿前后 b_beta 梯度的变化"""
    print("\n" + "=" * 80)
    print("Natural Gradient 补偿效果")
    print("=" * 80)

    # 保存原始梯度
    raw_grads = {}
    for i, layer_module in enumerate(model.layers):
        g = layer_module.snn_block.b_beta.grad
        if g is not None:
            raw_grads[i] = g.clone()

    # 应用补偿
    model.compensate_modulation_gradients()

    print(f"{'Layer':<8} {'raw |grad| mean':>16} {'comp |grad| mean':>16} {'amplify':>10}")
    print("-" * 60)

    for i, layer_module in enumerate(model.layers):
        g_comp = layer_module.snn_block.b_beta.grad
        if i in raw_grads and g_comp is not None:
            raw_mean = raw_grads[i].abs().mean().item()
            comp_mean = g_comp.abs().mean().item()
            amplify = comp_mean / raw_mean if raw_mean > 0 else float('inf')
            print(f"L{i:<6} {raw_mean:>16.8f} {comp_mean:>16.8f} {amplify:>10.1f}x")


def compare_with_other_params(model):
    """对比 b_beta 和其他参数的梯度幅值比"""
    print("\n" + "=" * 80)
    print("b_beta vs 其他参数梯度对比（补偿后）")
    print("=" * 80)

    # 收集关键参数组的梯度
    param_stats = {}
    for i, layer_module in enumerate(model.layers):
        block = layer_module.snn_block
        for name, p in [
            ('b_beta', block.b_beta),
            ('b_alpha', block.b_alpha),
            ('b_th', block.b_th),
            ('W_in', block.W_in.weight),
            ('W_beta_x', block.W_beta_x.weight),
            ('W_out', block.W_out.weight),
            ('W_gate', block.W_gate.weight),
            ('out_proj', layer_module.block_out_proj.weight),
        ]:
            if p.grad is not None:
                if name not in param_stats:
                    param_stats[name] = []
                param_stats[name].append(p.grad.abs().mean().item())

    print(f"{'参数':<15} {'grad |mean| (avg across layers)':>30} {'ratio vs b_beta':>20}")
    print("-" * 65)

    b_beta_avg = sum(param_stats.get('b_beta', [0])) / max(len(param_stats.get('b_beta', [1])), 1)
    for name in ['b_beta', 'b_alpha', 'b_th', 'W_in', 'W_beta_x', 'W_out', 'W_gate', 'out_proj']:
        vals = param_stats.get(name, [])
        if vals:
            avg = sum(vals) / len(vals)
            ratio = avg / b_beta_avg if b_beta_avg > 0 else float('inf')
            print(f"{name:<15} {avg:>30.8f} {ratio:>20.1f}x")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', default=None, help='Checkpoint path (optional, uses fresh init if not provided)')
    parser.add_argument('--D', type=int, default=768)
    parser.add_argument('--N', type=int, default=8)
    parser.add_argument('--num_layers', type=int, default=20)
    parser.add_argument('--D_ff', type=int, default=2304)
    parser.add_argument('--seq_len', type=int, default=32)
    parser.add_argument('--batch_size', type=int, default=2)
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    if args.ckpt:
        print(f"Loading checkpoint: {args.ckpt}")
        model, config = load_checkpoint(args.ckpt, device)
        print(f"Model config: D={config.get('D')}, N={config.get('N')}, Layers={config.get('num_layers')}")
    else:
        print("Using freshly initialized model (b_beta ≈ checkpoint since it hasn't moved)")
        model = SNNLanguageModel(
            D=args.D, N=args.N, num_layers=args.num_layers, D_ff=args.D_ff,
        ).to(device)
        print(f"Model config: D={args.D}, N={args.N}, Layers={args.num_layers}")

    # 1. b_beta 分布
    report_b_beta_distribution(model)

    # 2. Forward/backward
    print(f"\nRunning forward/backward (seq_len={args.seq_len}, batch={args.batch_size})...")
    with torch.amp.autocast('cuda', dtype=torch.bfloat16):
        loss, inter_grads = diagnose_gradients(model, device, args.seq_len, args.batch_size)
    print(f"Loss: {loss:.4f}")

    # 3. 报告中间梯度
    if inter_grads:
        print("\n中间层梯度幅值:")
        for name, grad in sorted(inter_grads.items()):
            print(f"  {name}: |grad| mean={grad.abs().mean().item():.8f}, max={grad.abs().max().item():.8f}")

    # 4. 原始梯度报告
    report_param_gradients(model)

    # 5. 补偿效果
    test_compensation_effect(model)

    # 6. 对比分析
    compare_with_other_params(model)
