import torch
import numpy as np
import sys
from collections import defaultdict

def analyze_all_params(path1, path2):
    ckpt1 = torch.load(path1, map_location="cpu", weights_only=False)
    ckpt2 = torch.load(path2, map_location="cpu", weights_only=False)
    state1 = ckpt1["model_state_dict"]
    state2 = ckpt2["model_state_dict"]

    print("="*100)
    print("NeuronSpark 模型 - 全部可训练参数分析")
    print("="*100)

    # 统计所有参数
    all_keys = sorted(state1.keys())
    print(f"\n总参数组数: {len(all_keys)}")

    # 按模块分类
    modules = defaultdict(list)
    for k in all_keys:
        parts = k.split(".")
        if parts[0] == "layers":
            # layers.X.module.submodule.param
            if len(parts) >= 4:
                module = ".".join(parts[2:-1])  # 去掉 layers.X 和最后的参数名
            else:
                module = parts[2] if len(parts) > 2 else "unknown"
        else:
            module = parts[0]
        modules[module].append(k)

    print(f"模块数: {len(modules)}")
    print("\n模块列表:")
    for m in sorted(modules.keys()):
        print(f"  {m}: {len(modules[m])} 个参数组")

    # 详细分析每个模块
    print("\n" + "="*100)
    print("【按模块详细分析】")
    print("="*100)

    total_params = 0
    total_change = 0

    for module in sorted(modules.keys()):
        keys = modules[module]
        print(f"\n{'─'*100}")
        print(f"模块: {module} ({len(keys)} 个参数组)")
        print(f"{'─'*100}")

        module_params = 0
        module_change = 0

        for k in sorted(keys)[:30]:  # 每个模块最多显示30个
            if k not in state2:
                continue
            v1 = state1[k].float()
            v2 = state2[k].float()
            diff = (v2 - v1).abs()

            num_params = v1.numel()
            abs_change = diff.mean().item()
            rel_change = abs_change / (v1.abs().mean().item() + 1e-8) * 100

            module_params += num_params
            module_change += diff.sum().item()

            # 提取参数名
            param_name = k.split(".")[-1]
            short_key = k if len(k) < 60 else "..." + k[-57:]

            print(f"  {short_key}")
            print(f"    shape={tuple(v1.shape)}, numel={num_params:,}")
            print(f"    Step1000: mean={v1.mean():.4f}, std={v1.std():.4f}")
            print(f"    Step2000: mean={v2.mean():.4f}, std={v2.std():.4f}")
            print(f"    变化: abs={abs_change:.6f}, rel={rel_change:.2f}%")

        if len(keys) > 30:
            print(f"  ... 还有 {len(keys) - 30} 个参数组未显示")

        total_params += module_params
        total_change += module_change

    # 汇总表
    print("\n" + "="*100)
    print("【模块变化汇总表】")
    print("="*100)
    print(f"\n{'模块':<45} {'参数量':<15} {'平均变化':<15} {'相对变化%':<12}")
    print("-"*90)

    summary = []
    for module in sorted(modules.keys()):
        keys = modules[module]
        v1_all = []
        v2_all = []
        for k in keys:
            if k in state2:
                v1_all.append(state1[k].float().flatten())
                v2_all.append(state2[k].float().flatten())
        if v1_all:
            v1_cat = torch.cat(v1_all)
            v2_cat = torch.cat(v2_all)
            diff = (v2_cat - v1_cat).abs()
            num_params = v1_cat.numel()
            abs_change = diff.mean().item()
            rel_change = abs_change / (v1_cat.abs().mean().item() + 1e-8) * 100
            summary.append((module, num_params, abs_change, rel_change))

    # 按相对变化排序
    summary.sort(key=lambda x: -x[3])
    for module, num_params, abs_change, rel_change in summary:
        print(f"{module:<45} {num_params:<15,} {abs_change:<15.6f} {rel_change:<12.2f}")

    # 总计
    print("-"*90)
    v1_total = torch.cat([state1[k].float().flatten() for k in all_keys if k in state2])
    v2_total = torch.cat([state2[k].float().flatten() for k in all_keys if k in state2])
    diff_total = (v2_total - v1_total).abs()
    print(f"{'总计':<45} {v1_total.numel():<15,} {diff_total.mean().item():<15.6f} {diff_total.mean().item()/(v1_total.abs().mean().item()+1e-8)*100:<12.2f}")

    # 检查变化最大和最小的参数
    print("\n" + "="*100)
    print("【变化最大的 20 个参数组】")
    print("="*100)

    changes = []
    for k in all_keys:
        if k in state2:
            v1 = state1[k].float()
            v2 = state2[k].float()
            diff = (v2 - v1).abs()
            rel = diff.mean().item() / (v1.abs().mean().item() + 1e-8) * 100
            changes.append((k, rel, diff.mean().item(), v1.numel()))

    changes.sort(key=lambda x: -x[1])
    for k, rel, abs_c, n in changes[:20]:
        print(f"  {rel:>8.2f}% | abs={abs_c:.6f} | n={n:>8,} | {k}")

    print("\n" + "="*100)
    print("【变化最小的 20 个参数组】")
    print("="*100)
    for k, rel, abs_c, n in changes[-20:]:
        print(f"  {rel:>8.2f}% | abs={abs_c:.6f} | n={n:>8,} | {k}")

    # Checkpoint 信息
    print("\n" + "="*100)
    print("【Checkpoint 信息】")
    print("="*100)
    print(f"Step: {ckpt1.get('step')} → {ckpt2.get('step')}")
    print(f"Loss: {ckpt1.get('best_loss', 'N/A')} → {ckpt2.get('best_loss', 'N/A')}")
    print(f"Tokens: {ckpt1.get('tokens_seen', 0):,} → {ckpt2.get('tokens_seen', 0):,}")

if __name__ == "__main__":
    analyze_all_params(sys.argv[1], sys.argv[2])
