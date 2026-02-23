import torch
import numpy as np
import sys

def analyze_checkpoint(ckpt_path):
    print(f"Loading {ckpt_path}...")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state = ckpt["model_state_dict"]
    
    print("="*100)
    print(f"Checkpoint: {ckpt_path}")
    step = ckpt.get("step", "N/A")
    loss = ckpt.get("best_loss", 0)
    print(f"Step: {step}, Loss: {loss:.4f}")
    print("="*100)
    
    # 1. 选择性参数分析
    print("\n" + "="*100)
    print("【1. 选择性神经元参数 b_beta/b_alpha/b_th 实际值分布】")
    print("="*100)
    
    all_beta, all_alpha, all_th = [], [], []
    for layer_idx in range(20):
        prefix = f"layers.{layer_idx}.snn_block"
        b_beta = state.get(f"{prefix}.b_beta")
        b_alpha = state.get(f"{prefix}.b_alpha")
        b_th = state.get(f"{prefix}.b_th")
        
        if b_beta is not None:
            beta = torch.sigmoid(b_beta)
            alpha = torch.nn.functional.softplus(b_alpha)
            th = torch.nn.functional.softplus(b_th)
            
            all_beta.append(beta)
            all_alpha.append(alpha)
            all_th.append(th)
            
            if layer_idx < 3 or layer_idx >= 18:
                print(f"\nLayer {layer_idx}:")
                print(f"  b_beta raw: mean={b_beta.mean():.4f}, std={b_beta.std():.4f}, range=[{b_beta.min():.4f}, {b_beta.max():.4f}]")
                print(f"  beta=sigmoid: mean={beta.mean():.4f}, std={beta.std():.4f}, range=[{beta.min():.4f}, {beta.max():.4f}]")
                print(f"  b_alpha raw: mean={b_alpha.mean():.4f}, std={b_alpha.std():.4f}")
                print(f"  alpha=softplus: mean={alpha.mean():.4f}, std={alpha.std():.4f}, range=[{alpha.min():.4f}, {alpha.max():.4f}]")
                print(f"  b_th raw: mean={b_th.mean():.4f}, std={b_th.std():.4f}")
                print(f"  th=softplus: mean={th.mean():.4f}, std={th.std():.4f}, range=[{th.min():.4f}, {th.max():.4f}]")
    
    if all_beta:
        all_beta = torch.cat(all_beta).numpy()
        all_alpha = torch.cat(all_alpha).numpy()
        all_th = torch.cat(all_th).numpy()
        
        print(f"\n【全局统计 (20层汇总)】")
        print(f"  beta (记忆保持): mean={all_beta.mean():.4f}, std={all_beta.std():.4f}, range=[{all_beta.min():.4f}, {all_beta.max():.4f}]")
        print(f"    beta<0.1: {(all_beta<0.1).mean()*100:.1f}%, beta>0.9: {(all_beta>0.9).mean()*100:.1f}%, 0.3<beta<0.7: {((all_beta>0.3)&(all_beta<0.7)).mean()*100:.1f}%")
        print(f"  alpha (输入缩放): mean={all_alpha.mean():.4f}, std={all_alpha.std():.4f}, range=[{all_alpha.min():.4f}, {all_alpha.max():.4f}]")
        print(f"  th (阈值): mean={all_th.mean():.4f}, std={all_th.std():.4f}, range=[{all_th.min():.4f}, {all_th.max():.4f}]")
    
    # 2. 权重分布
    print("\n" + "="*100)
    print("【2. 各模块权重分布】")
    print("="*100)
    
    modules = {}
    for key in sorted(state.keys()):
        if "weight" in key and "norm" not in key:
            w = state[key].float()
            parts = key.split(".")
            if "layers" in key:
                mod = ".".join(parts[2:-1])
            else:
                mod = parts[0]
            if mod not in modules:
                modules[mod] = {"mean": [], "std": [], "abs_mean": [], "min": [], "max": []}
            modules[mod]["mean"].append(w.mean().item())
            modules[mod]["std"].append(w.std().item())
            modules[mod]["abs_mean"].append(w.abs().mean().item())
            modules[mod]["min"].append(w.min().item())
            modules[mod]["max"].append(w.max().item())
    
    print(f"\n{模块:<35} {mean:<12} {std:<12} {abs_mean:<12} {范围}")
    print("-"*90)
    for mod in sorted(modules.keys()):
        m = modules[mod]
        print(f"{mod:<35} {np.mean(m[mean]):<12.6f} {np.mean(m[std]):<12.6f} {np.mean(m[abs_mean]):<12.6f} [{np.min(m[min]):.4f}, {np.max(m[max]):.4f}]")
    
    # 3. 层间权重幅度
    print("\n" + "="*100)
    print("【3. 层间权重幅度 (学习程度)】")
    print("="*100)
    
    layer_mags = []
    for layer_idx in range(20):
        weights = []
        for key in state.keys():
            if f"layers.{layer_idx}." in key and "weight" in key:
                weights.append(state[key].float().abs().mean().item())
        if weights:
            layer_mags.append(np.mean(weights))
    
    print(f"\n{Layer:<8} {幅度:<12} 可视化")
    print("-"*60)
    max_mag = max(layer_mags) if layer_mags else 1
    for i, mag in enumerate(layer_mags):
        bar = "█" * int(mag / max_mag * 40)
        print(f"{i:<8} {mag:<12.6f} {bar}")
    
    # 4. PLIFNode 参数
    print("\n" + "="*100)
    print("【4. PLIFNode 参数 (w, v_th)】")
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
            decay = torch.sigmoid(w_all).numpy()
            print(f"\n{ntype}:")
            print(f"  w raw: mean={w_all.mean():.4f}, std={w_all.std():.4f}")
            print(f"  decay=sigmoid(w): mean={decay.mean():.4f}, std={decay.std():.4f}, range=[{decay.min():.4f}, {decay.max():.4f}]")
            if th_vals:
                th_all = torch.cat([v.flatten() for v in th_vals])
                print(f"  v_th: mean={th_all.mean():.4f}, std={th_all.std():.4f}, range=[{th_all.min():.4f}, {th_all.max():.4f}]")
    
    # 5. 异常检测
    print("\n" + "="*100)
    print("【5. 异常值检测】")
    print("="*100)
    
    issues = []
    for key in state.keys():
        t = state[key].float()
        if torch.isnan(t).any():
            issues.append(f"NaN in {key}")
        if torch.isinf(t).any():
            issues.append(f"Inf in {key}")
        if t.abs().max() > 100:
            issues.append(f"Large value ({t.abs().max():.1f}) in {key}")
    
    if issues:
        for issue in issues[:20]:
            print(f"  WARNING: {issue}")
    else:
        print("  OK: No NaN/Inf/extreme values")

if __name__ == "__main__":
    analyze_checkpoint(sys.argv[1])
