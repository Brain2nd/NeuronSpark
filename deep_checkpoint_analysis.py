import torch
import numpy as np
import sys

def analyze_checkpoint(ckpt_path):
    print(f"加载 {ckpt_path}...")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state = ckpt["model_state_dict"]
    
    print("="*100)
    print(f"Checkpoint: {ckpt_path}")
    print(f"Step: {ckpt.get('step')}, Loss: {ckpt.get('best_loss', N/A):.4f}")
    print("="*100)
    
    # 1. 选择性参数分析 (b_beta, b_alpha, b_th)
    print("\n" + "="*100)
    print("【1. 选择性神经元参数 (b_beta/b_alpha/b_th) 实际值分布】")
    print("="*100)
    
    for layer_idx in range(20):
        prefix = f"layers.{layer_idx}.snn_block"
        b_beta = state.get(f"{prefix}.b_beta")
        b_alpha = state.get(f"{prefix}.b_alpha")
        b_th = state.get(f"{prefix}.b_th")
        
        if b_beta is not None:
            # 计算 sigmoid/softplus 后的实际值
            beta = torch.sigmoid(b_beta).numpy()
            alpha = torch.nn.functional.softplus(b_alpha).numpy()
            th = torch.nn.functional.softplus(b_th).numpy()
            
            if layer_idx < 5 or layer_idx >= 18:  # 只打印前5层和后2层
                print(f"\nLayer {layer_idx}:")
                print(f"  b_beta  raw: mean={b_beta.mean():.4f}, std={b_beta.std():.4f}, range=[{b_beta.min():.4f}, {b_beta.max():.4f}]")
                print(f"  β=sigmoid(b_beta): mean={beta.mean():.4f}, std={beta.std():.4f}, range=[{beta.min():.4f}, {beta.max():.4f}]")
                print(f"  b_alpha raw: mean={b_alpha.mean():.4f}, std={b_alpha.std():.4f}, range=[{b_alpha.min():.4f}, {b_alpha.max():.4f}]")
                print(f"  α=softplus(b_alpha): mean={alpha.mean():.4f}, std={alpha.std():.4f}, range=[{alpha.min():.4f}, {alpha.max():.4f}]")
                print(f"  b_th raw: mean={b_th.mean():.4f}, std={b_th.std():.4f}, range=[{b_th.min():.4f}, {b_th.max():.4f}]")
                print(f"  th=softplus(b_th): mean={th.mean():.4f}, std={th.std():.4f}, range=[{th.min():.4f}, {th.max():.4f}]")
    
    # 汇总所有层
    all_beta, all_alpha, all_th = [], [], []
    for layer_idx in range(20):
        prefix = f"layers.{layer_idx}.snn_block"
        b_beta = state.get(f"{prefix}.b_beta")
        b_alpha = state.get(f"{prefix}.b_alpha")
        b_th = state.get(f"{prefix}.b_th")
        if b_beta is not None:
            all_beta.append(torch.sigmoid(b_beta))
            all_alpha.append(torch.nn.functional.softplus(b_alpha))
            all_th.append(torch.nn.functional.softplus(b_th))
    
    all_beta = torch.cat(all_beta).numpy()
    all_alpha = torch.cat(all_alpha).numpy()
    all_th = torch.cat(all_th).numpy()
    
    print(f"\n【全局统计 (20层汇总)】")
    print(f"  β (记忆保持): mean={all_beta.mean():.4f}, std={all_beta.std():.4f}, range=[{all_beta.min():.4f}, {all_beta.max():.4f}]")
    print(f"    β<0.1占比: {(all_beta<0.1).mean()*100:.1f}%, β>0.9占比: {(all_beta>0.9).mean()*100:.1f}%")
    print(f"  α (输入缩放): mean={all_alpha.mean():.4f}, std={all_alpha.std():.4f}, range=[{all_alpha.min():.4f}, {all_alpha.max():.4f}]")
    print(f"  th (阈值): mean={all_th.mean():.4f}, std={all_th.std():.4f}, range=[{all_th.min():.4f}, {all_th.max():.4f}]")
    
    # 2. 权重分布分析
    print("\n" + "="*100)
    print("【2. 各层权重分布分析】")
    print("="*100)
    
    weight_stats = []
    for key in sorted(state.keys()):
        if "weight" in key and "norm" not in key:
            w = state[key].float()
            stats = {
                "name": key,
                "shape": tuple(w.shape),
                "mean": w.mean().item(),
                "std": w.std().item(),
                "min": w.min().item(),
                "max": w.max().item(),
                "abs_mean": w.abs().mean().item()
            }
            weight_stats.append(stats)
    
    # 按模块分类统计
    modules = {}
    for s in weight_stats:
        parts = s["name"].split(".")
        if "layers" in s["name"]:
            # 提取模块名 (去掉 layers.X)
            mod = ".".join(parts[2:-1])
        else:
            mod = parts[0]
        if mod not in modules:
            modules[mod] = []
        modules[mod].append(s)
    
    print(f"\n{模块:<35} {数量:<6} {mean:<12} {std:<12} {abs_mean:<12} {范围:<20}")
    print("-"*100)
    for mod in sorted(modules.keys()):
        stats_list = modules[mod]
        means = [s["mean"] for s in stats_list]
        stds = [s["std"] for s in stats_list]
        abs_means = [s["abs_mean"] for s in stats_list]
        mins = [s["min"] for s in stats_list]
        maxs = [s["max"] for s in stats_list]
        print(f"{mod:<35} {len(stats_list):<6} {np.mean(means):<12.6f} {np.mean(stds):<12.6f} {np.mean(abs_means):<12.6f} [{np.min(mins):.4f}, {np.max(maxs):.4f}]")
    
    # 3. 层间梯度流分析 (通过权重变化估算)
    print("\n" + "="*100)
    print("【3. 层间权重幅度分析 (反映学习程度)】")
    print("="*100)
    
    layer_magnitudes = []
    for layer_idx in range(20):
        layer_weights = []
        for key in state.keys():
            if f"layers.{layer_idx}." in key and "weight" in key:
                layer_weights.append(state[key].float().abs().mean().item())
        if layer_weights:
            layer_magnitudes.append((layer_idx, np.mean(layer_weights)))
    
    print(f"\n{Layer:<8} {平均权重幅度:<15}")
    print("-"*25)
    for layer_idx, mag in layer_magnitudes:
        bar = "█" * int(mag * 100)
        print(f"{layer_idx:<8} {mag:<15.6f} {bar}")
    
    # 4. PLIFNode 神经元参数分析
    print("\n" + "="*100)
    print("【4. PLIFNode 神经元参数 (w, v_th) 分析】")
    print("="*100)
    
    neuron_types = ["input_neuron1", "input_neuron2", "output_neuron", 
                    "snn_block.output_neuron", "snn_ffn.gate_neuron", 
                    "snn_ffn.up_neuron", "snn_ffn.output_neuron"]
    
    for ntype in neuron_types:
        w_vals, th_vals = [], []
        for key in state.keys():
            if ntype in key:
                if key.endswith(".w"):
                    w_vals.append(state[key].float())
                elif key.endswith(".v_th"):
                    th_vals.append(state[key].float())
        
        if w_vals:
            w_all = torch.cat([v.flatten() for v in w_vals])
            th_all = torch.cat([v.flatten() for v in th_vals]) if th_vals else None
            
            # w 经过 sigmoid 变成实际的衰减系数
            w_actual = torch.sigmoid(w_all).numpy()
            print(f"\n{ntype}:")
            print(f"  w raw: mean={w_all.mean():.4f}, std={w_all.std():.4f}")
            print(f"  decay=sigmoid(w): mean={w_actual.mean():.4f}, std={w_actual.std():.4f}, range=[{w_actual.min():.4f}, {w_actual.max():.4f}]")
            if th_all is not None:
                print(f"  v_th: mean={th_all.mean():.4f}, std={th_all.std():.4f}, range=[{th_all.min():.4f}, {th_all.max():.4f}]")

    # 5. 检查异常值
    print("\n" + "="*100)
    print("【5. 异常值检测】")
    print("="*100)
    
    issues = []
    for key in state.keys():
        t = state[key].float()
        if torch.isnan(t).any():
            issues.append(f"NaN detected in {key}")
        if torch.isinf(t).any():
            issues.append(f"Inf detected in {key}")
        if t.abs().max() > 100:
            issues.append(f"Large value (>{t.abs().max():.1f}) in {key}")
    
    if issues:
        for issue in issues[:20]:
            print(f"  ⚠️ {issue}")
    else:
        print("  ✓ 未发现 NaN/Inf/极大值")

if __name__ == "__main__":
    analyze_checkpoint(sys.argv[1])
