import torch
import sys

ckpt = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
state = ckpt["model_state_dict"]

print("="*80)
print("模型中所有神经元参数完整列表")
print("="*80)

# 按参数类型分组
param_types = {}
for k in state.keys():
    param_name = k.split(".")[-1]
    if param_name not in param_types:
        param_types[param_name] = []
    param_types[param_name].append(k)

# 显示神经元相关参数
neuron_params = ["w", "v_th", "b_beta", "b_alpha", "b_th"]
print("\n【神经元参数类型统计】")
for p in neuron_params:
    if p in param_types:
        print(f"  {p}: {len(param_types[p])} 个参数")

# 详细列出每种参数
for p in neuron_params:
    if p in param_types:
        print(f"\n{'='*80}")
        print(f"【{p} 参数详细列表】共 {len(param_types[p])} 个")
        print("="*80)
        for k in sorted(param_types[p]):
            shape = state[k].shape
            val = state[k].float()
            print(f"  {k}")
            print(f"    shape={shape}, mean={val.mean():.4f}, std={val.std():.4f}, min={val.min():.4f}, max={val.max():.4f}")

# 检查是否有 PLIFNode 相关参数
print(f"\n{'='*80}")
print("【检查是否有遗漏的神经元参数】")
print("="*80)
for k in sorted(state.keys()):
    if "plif" in k.lower() or "lif" in k.lower() or "neuron" in k.lower():
        if k.split(".")[-1] not in neuron_params:
            print(f"  可能遗漏: {k}")
