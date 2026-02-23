import torch
import numpy as np
import sys

def analyze(path1, path2):
    ckpt1 = torch.load(path1, map_location="cpu", weights_only=False)
    ckpt2 = torch.load(path2, map_location="cpu", weights_only=False)
    state1 = ckpt1["model_state_dict"]
    state2 = ckpt2["model_state_dict"]

    print("="*60)
    print("Step 1000 vs Step 2000 对比")
    print("="*60)

    # 神经元 w 参数
    print("\n[神经元 w 参数]")
    for name, state in [("Step1000", state1), ("Step2000", state2)]:
        vals = []
        for k, v in state.items():
            parts = k.split(".")
            if parts[-1] == "w" and "weight" not in k:
                vals.append(v.float().numpy().flatten())
        if vals:
            arr = np.concatenate(vals)
            print(f"  {name}: mean={arr.mean():.4f}, std={arr.std():.4f}")

    # 神经元 v_th 参数
    print("\n[神经元 v_th 参数]")
    for name, state in [("Step1000", state1), ("Step2000", state2)]:
        vals = []
        for k, v in state.items():
            if "v_th" in k:
                vals.append(v.float().numpy().flatten())
        if vals:
            arr = np.concatenate(vals)
            print(f"  {name}: mean={arr.mean():.4f}, std={arr.std():.4f}")

    # block_out_proj 列和
    print("\n[block_out_proj 列和变化]")
    for layer in [0, 5, 10, 19]:
        key = f"layers.{layer}.block_out_proj.weight"
        if key in state1 and key in state2:
            col1 = state1[key].float().numpy().sum(axis=0)
            col2 = state2[key].float().numpy().sum(axis=0)
            print(f"  Layer {layer}: {col1.mean():.4f} -> {col2.mean():.4f}")

    # RMSNorm 权重
    print("\n[RMSNorm 权重]")
    for layer in [0, 5]:
        for norm in ["block_norm", "ffn_norm"]:
            key = f"layers.{layer}.{norm}.weight"
            if key in state2:
                w = state2[key].float().numpy()
                print(f"  Layer {layer} {norm}: mean={w.mean():.4f}, std={w.std():.4f}")

    # 总体变化
    print("\n[参数总体变化]")
    total_diff = 0
    total_norm = 0
    for k in state1.keys():
        if k in state2:
            diff = (state2[k].float() - state1[k].float()).abs().sum().item()
            norm = state1[k].float().abs().sum().item() + 1e-8
            total_diff += diff
            total_norm += norm
    print(f"  相对变化: {total_diff/total_norm*100:.4f}%")
    print(f"  Tokens: {ckpt1.get('tokens_seen', 0):,} -> {ckpt2.get('tokens_seen', 0):,}")

if __name__ == "__main__":
    analyze(sys.argv[1], sys.argv[2])
