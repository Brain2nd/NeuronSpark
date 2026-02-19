"""
v7.5 完整模型梯度验证

验证项目：
  1. 前向传播能正常运行（无报错、无 NaN/Inf）
  2. 反向传播能正常运行（loss.backward() 无报错）
  3. 所有参数都有有效梯度（非 None、非全零、非 NaN）
  4. 梯度范数合理（无爆炸/消失）
  5. 各层梯度比率检查（L0 vs L_last）
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ['TRITON_PTXAS_PATH'] = '/usr/local/cuda-13.0/bin/ptxas'

import torch
from model import SNNLanguageModel
from spikingjelly.activation_based import functional


def main():
    device = 'cuda'
    dtype = torch.bfloat16

    # 使用训练配置
    D = 768
    N = 8
    D_ff = 2304
    num_layers = 20
    vocab_size = 6144
    K = 16
    seq_len = 512
    batch = 2

    print("=" * 70)
    print("v7.5 完整模型梯度验证")
    print(f"D={D}, N={N}, D_ff={D_ff}, layers={num_layers}, vocab={vocab_size}")
    print(f"K={K}, seq_len={seq_len}, batch={batch}")
    print("=" * 70)

    # 构建模型
    print("\n[1] 构建模型...")
    model = SNNLanguageModel(
        vocab_size=vocab_size,
        D=D,
        N=N,
        D_ff=D_ff,
        num_layers=num_layers,
        K=K,
    ).to(device).to(dtype)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  总参数: {total_params:,} ({total_params/1e6:.1f}M)")
    print(f"  可训练: {trainable_params:,} ({trainable_params/1e6:.1f}M)")

    # 前向传播
    print("\n[2] 前向传播...")
    torch.manual_seed(42)
    input_ids = torch.randint(0, vocab_size, (batch, seq_len), device=device)
    labels = torch.randint(0, vocab_size, (batch, seq_len), device=device)

    output = model(input_ids, labels)
    loss = output.last_loss.mean()

    print(f"  Loss = {loss.item():.4f}")
    assert not torch.isnan(loss), "Loss is NaN!"
    assert not torch.isinf(loss), "Loss is Inf!"
    assert loss.item() > 0, "Loss is non-positive!"
    print(f"  前向传播 PASS (loss={loss.item():.4f})")

    # 反向传播
    print("\n[3] 反向传播...")
    loss.backward()
    print("  backward() 完成，无报错")

    # 检查所有参数梯度
    print("\n[4] 参数梯度检查...")
    total_checked = 0
    none_grad = []
    zero_grad = []
    nan_grad = []
    inf_grad = []
    valid_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        total_checked += 1

        if param.grad is None:
            none_grad.append(name)
            continue

        g = param.grad
        gnorm = g.float().norm().item()

        if torch.isnan(g).any():
            nan_grad.append(name)
            continue
        if torch.isinf(g).any():
            inf_grad.append(name)
            continue
        if gnorm == 0:
            zero_grad.append(name)
            continue

        valid_params.append((name, gnorm, param.numel()))

    print(f"  总参数组: {total_checked}")
    print(f"  有效梯度: {len(valid_params)}")

    if none_grad:
        print(f"  [FAIL] grad=None ({len(none_grad)}):")
        for n in none_grad:
            print(f"    - {n}")
    if zero_grad:
        print(f"  [WARN] grad=0 ({len(zero_grad)}):")
        for n in zero_grad:
            print(f"    - {n}")
    if nan_grad:
        print(f"  [FAIL] grad=NaN ({len(nan_grad)}):")
        for n in nan_grad:
            print(f"    - {n}")
    if inf_grad:
        print(f"  [FAIL] grad=Inf ({len(inf_grad)}):")
        for n in inf_grad:
            print(f"    - {n}")

    grad_ok = len(none_grad) == 0 and len(nan_grad) == 0 and len(inf_grad) == 0
    print(f"  梯度完整性: {'PASS' if grad_ok else 'FAIL'} ({len(valid_params)}/{total_checked} 有效)")

    # 梯度范数分布
    print("\n[5] 梯度范数分布（按模块分组）...")
    module_norms = {}
    for name, gnorm, numel in valid_params:
        # 提取模块前缀
        parts = name.split('.')
        if 'layers' in parts:
            idx = parts.index('layers')
            module_key = '.'.join(parts[:idx+2])  # e.g., "layers.0"
        else:
            module_key = parts[0]

        if module_key not in module_norms:
            module_norms[module_key] = []
        module_norms[module_key].append((name, gnorm))

    # 打印各模块梯度范围
    print(f"  {'Module':40s} {'min_norm':>12s} {'max_norm':>12s} {'#params':>8s}")
    print(f"  {'-'*40} {'-'*12} {'-'*12} {'-'*8}")
    layer_avg_norms = {}
    for module_key in sorted(module_norms.keys()):
        items = module_norms[module_key]
        norms = [g for _, g in items]
        min_n = min(norms)
        max_n = max(norms)
        avg_n = sum(norms) / len(norms)
        print(f"  {module_key:40s} {min_n:>12.4e} {max_n:>12.4e} {len(items):>8d}")
        if module_key.startswith('layers.'):
            layer_idx = int(module_key.split('.')[1])
            layer_avg_norms[layer_idx] = avg_n

    # 层间梯度比率
    if layer_avg_norms:
        print("\n[6] 层间梯度比率...")
        layer_indices = sorted(layer_avg_norms.keys())
        first = layer_indices[0]
        last = layer_indices[-1]
        ratio = layer_avg_norms[first] / (layer_avg_norms[last] + 1e-30)
        print(f"  L{first} avg_norm: {layer_avg_norms[first]:.4e}")
        print(f"  L{last} avg_norm: {layer_avg_norms[last]:.4e}")
        print(f"  L{first}/L{last} ratio: {ratio:.4f}")
        if 0.01 < ratio < 100:
            print(f"  梯度流 PASS (ratio在0.01~100之间)")
        else:
            print(f"  [WARN] 梯度流比率偏大，可能有消失/爆炸风险")

    # 总结
    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)
    all_pass = grad_ok and len(zero_grad) == 0
    print(f"  前向传播: PASS (loss={loss.item():.4f})")
    print(f"  反向传播: PASS")
    print(f"  梯度完整: {len(valid_params)}/{total_checked} 有效, "
          f"{len(zero_grad)} 全零, {len(none_grad)} None")
    if layer_avg_norms:
        print(f"  梯度流:   L{first}/L{last} = {ratio:.4f}")
    print(f"  总结:     {'ALL PASS' if all_pass else 'HAS ISSUES'}")


if __name__ == '__main__':
    main()
