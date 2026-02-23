import torch
import numpy as np
import sys

def analyze_detailed(path1, path2):
    ckpt1 = torch.load(path1, map_location="cpu", weights_only=False)
    ckpt2 = torch.load(path2, map_location="cpu", weights_only=False)
    state1 = ckpt1["model_state_dict"]
    state2 = ckpt2["model_state_dict"]

    print("="*70)
    print("深度神经元参数分析: Step 1000 vs Step 2000")
    print("="*70)

    # 所有神经元参数类型
    neuron_params = ["w", "v_th", "b_beta", "b_alpha", "b_th"]

    # 按层分析每种参数
    for param_name in neuron_params:
        print(f"\n{'='*70}")
        print(f"【{param_name} 参数详细分析】")
        print("="*70)

        # 收集每层的数据
        layer_data_1 = {}  # {layer_idx: {neuron_type: tensor}}
        layer_data_2 = {}

        for k, v in state1.items():
            if f".{param_name}" in k and "weight" not in k:
                # 解析 key: layers.X.input_neuronY.param 或 layers.X.snn_block.proj_neurons.Z.param
                parts = k.split(".")
                try:
                    layer_idx = int(parts[1])
                    neuron_type = ".".join(parts[2:-1])  # e.g., "input_neuron1" or "snn_block.proj_neurons.0"

                    if layer_idx not in layer_data_1:
                        layer_data_1[layer_idx] = {}
                    layer_data_1[layer_idx][neuron_type] = v.float().numpy()
                except:
                    pass

        for k, v in state2.items():
            if f".{param_name}" in k and "weight" not in k:
                parts = k.split(".")
                try:
                    layer_idx = int(parts[1])
                    neuron_type = ".".join(parts[2:-1])

                    if layer_idx not in layer_data_2:
                        layer_data_2[layer_idx] = {}
                    layer_data_2[layer_idx][neuron_type] = v.float().numpy()
                except:
                    pass

        # 打印每层每种神经元的统计
        print(f"\n{'Layer':<6} {'Neuron Type':<35} {'Step1000 (mean±std)':<25} {'Step2000 (mean±std)':<25} {'Change%':<10}")
        print("-"*110)

        all_layers = sorted(set(layer_data_1.keys()) | set(layer_data_2.keys()))

        for layer_idx in all_layers:
            neurons_1 = layer_data_1.get(layer_idx, {})
            neurons_2 = layer_data_2.get(layer_idx, {})
            all_types = sorted(set(neurons_1.keys()) | set(neurons_2.keys()))

            for ntype in all_types:
                v1 = neurons_1.get(ntype)
                v2 = neurons_2.get(ntype)

                if v1 is not None and v2 is not None:
                    m1, s1 = v1.mean(), v1.std()
                    m2, s2 = v2.mean(), v2.std()
                    change = (m2 - m1) / (abs(m1) + 1e-8) * 100
                    print(f"{layer_idx:<6} {ntype:<35} {m1:+.4f}±{s1:.4f}           {m2:+.4f}±{s2:.4f}           {change:+.2f}%")

        # 汇总统计
        all_v1 = []
        all_v2 = []
        for layer_idx in all_layers:
            for ntype, v in layer_data_1.get(layer_idx, {}).items():
                all_v1.append(v.flatten())
            for ntype, v in layer_data_2.get(layer_idx, {}).items():
                all_v2.append(v.flatten())

        if all_v1 and all_v2:
            arr1 = np.concatenate(all_v1)
            arr2 = np.concatenate(all_v2)
            print(f"\n  汇总: Step1000 mean={arr1.mean():.4f}, std={arr1.std():.4f}, min={arr1.min():.4f}, max={arr1.max():.4f}")
            print(f"        Step2000 mean={arr2.mean():.4f}, std={arr2.std():.4f}, min={arr2.min():.4f}, max={arr2.max():.4f}")
            print(f"        绝对变化: {np.abs(arr2 - arr1).mean():.6f}")
            print(f"        相对变化: {np.abs(arr2 - arr1).mean() / (np.abs(arr1).mean() + 1e-8) * 100:.4f}%")

    # 分析 out_proj 权重
    print(f"\n{'='*70}")
    print("【残差投影权重 (out_proj) 详细分析】")
    print("="*70)

    print(f"\n{'Layer':<6} {'Proj Type':<15} {'Step1000 ColSum':<20} {'Step2000 ColSum':<20} {'Change':<15}")
    print("-"*80)

    for layer_idx in range(20):
        for proj in ["block_out_proj", "ffn_out_proj"]:
            key = f"layers.{layer_idx}.{proj}.weight"
            if key in state1 and key in state2:
                w1 = state1[key].float().numpy()
                w2 = state2[key].float().numpy()
                col1 = w1.sum(axis=0)
                col2 = w2.sum(axis=0)
                change = col2.mean() - col1.mean()
                print(f"{layer_idx:<6} {proj:<15} {col1.mean():+.4f}±{col1.std():.4f}     {col2.mean():+.4f}±{col2.std():.4f}     {change:+.4f}")

    # 分析 RMSNorm 权重
    print(f"\n{'='*70}")
    print("【RMSNorm 权重详细分析】")
    print("="*70)

    print(f"\n{'Layer':<6} {'Norm Type':<15} {'mean':<12} {'std':<12} {'min':<12} {'max':<12}")
    print("-"*70)

    for layer_idx in range(20):
        for norm in ["block_norm", "ffn_norm"]:
            key = f"layers.{layer_idx}.{norm}.weight"
            if key in state2:
                w = state2[key].float().numpy()
                print(f"{layer_idx:<6} {norm:<15} {w.mean():.6f}     {w.std():.6f}     {w.min():.6f}     {w.max():.6f}")

    # Embedding 和 Head 分析
    print(f"\n{'='*70}")
    print("【Embedding & Head 权重分析】")
    print("="*70)

    for key in ["tok_emb.weight", "head.weight"]:
        if key in state1 and key in state2:
            w1 = state1[key].float().numpy()
            w2 = state2[key].float().numpy()
            diff = np.abs(w2 - w1)
            print(f"\n{key}:")
            print(f"  Step1000: mean={w1.mean():.6f}, std={w1.std():.4f}")
            print(f"  Step2000: mean={w2.mean():.6f}, std={w2.std():.4f}")
            print(f"  绝对变化: mean={diff.mean():.6f}, max={diff.max():.6f}")
            print(f"  相对变化: {diff.mean() / (np.abs(w1).mean() + 1e-8) * 100:.4f}%")

    # 输出 checkpoint 元信息
    print(f"\n{'='*70}")
    print("【Checkpoint 元信息】")
    print("="*70)
    print(f"Step 1000: step={ckpt1.get('step')}, tokens={ckpt1.get('tokens_seen', 0):,}, loss={ckpt1.get('best_loss', 'N/A')}")
    print(f"Step 2000: step={ckpt2.get('step')}, tokens={ckpt2.get('tokens_seen', 0):,}, loss={ckpt2.get('best_loss', 'N/A')}")

if __name__ == "__main__":
    analyze_detailed(sys.argv[1], sys.argv[2])
