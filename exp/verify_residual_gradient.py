"""
验证连续残差流（v7.5）是否解决了 20 层梯度消失问题。

测试内容：
1. 前向传播 + loss.backward() 无报错
2. 每一层 block_out_proj / ffn_out_proj 的梯度范数 > 1e-10
3. 第 0 层与第 19 层的梯度范数比值 > 1e-6（之前是 ~1e-20）
4. 无 NaN 梯度
"""

import sys
sys.path.insert(0, '/home/dgxspark/Desktop/NeuronSpark')

import torch
from model import SNNLanguageModel


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    # 小模型：快速验证
    model = SNNLanguageModel(
        vocab_size=64, D=32, N=4, K=4, num_layers=20, D_ff=64,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total params: {total_params:,}")

    x = torch.randint(0, 64, (2, 16), device=device)
    y = torch.randint(0, 64, (2, 16), device=device)

    # 前向传播
    out = model(x, y)
    loss = out.last_loss.mean()
    print(f"Loss: {loss.item():.4f}")

    # 反向传播
    loss.backward()

    # 检查每一层的梯度
    print("\n=== 梯度覆盖检查 ===")
    grad_norms = []
    all_ok = True

    for i, layer in enumerate(model.layers):
        block_grad = layer.block_out_proj.weight.grad
        ffn_grad = layer.ffn_out_proj.weight.grad

        if block_grad is None or ffn_grad is None:
            print(f"Layer {i:2d}: !! grad is None !!")
            all_ok = False
            continue

        block_norm = block_grad.norm().item()
        ffn_norm = ffn_grad.norm().item()

        has_nan = (
            torch.isnan(block_grad).any().item() or
            torch.isnan(ffn_grad).any().item()
        )

        status = "NaN!" if has_nan else ("OK" if block_norm > 1e-10 and ffn_norm > 1e-10 else "VANISH!")
        if status != "OK":
            all_ok = False

        grad_norms.append((block_norm, ffn_norm))
        print(f"Layer {i:2d}: block_proj={block_norm:.2e}  ffn_proj={ffn_norm:.2e}  [{status}]")

    # 层 0 vs 层 19 比值
    if len(grad_norms) == 20:
        ratio_block = grad_norms[0][0] / (grad_norms[19][0] + 1e-30)
        ratio_ffn = grad_norms[0][1] / (grad_norms[19][1] + 1e-30)
        print(f"\nLayer0/Layer19 ratio: block={ratio_block:.2e}, ffn={ratio_ffn:.2e}")
        print(f"  (之前无残差: ~1e-20, 期望: > 1e-6)")

        if ratio_block < 1e-6 or ratio_ffn < 1e-6:
            print("  WARNING: 梯度比值仍然很小，可能还有问题")
            all_ok = False

    # 也检查 SNNBlock 内部参数（W_in 等）是否有梯度
    print("\n=== SNNBlock 内部参数梯度检查 ===")
    for i in [0, 9, 19]:
        layer = model.layers[i]
        w_in_grad = layer.snn_block.W_in.weight.grad
        if w_in_grad is not None:
            print(f"Layer {i:2d} W_in grad norm: {w_in_grad.norm().item():.2e}")
        else:
            print(f"Layer {i:2d} W_in grad: None")

    # 检查输入 PLIF 神经元参数梯度
    print("\n=== 输入 PLIF 神经元梯度检查 ===")
    for i in [0, 9, 19]:
        layer = model.layers[i]
        in1_grad = layer.input_neuron1.w.grad
        in2_grad = layer.input_neuron2.w.grad
        in1_norm = in1_grad.norm().item() if in1_grad is not None else 0.0
        in2_norm = in2_grad.norm().item() if in2_grad is not None else 0.0
        print(f"Layer {i:2d} input_neuron1.w={in1_norm:.2e}  input_neuron2.w={in2_norm:.2e}")

    print(f"\n{'=' * 40}")
    if all_ok:
        print("ALL PASSED: 所有 20 层梯度正常，残差流工作正确！")
    else:
        print("FAILED: 仍有梯度问题，需进一步调查")

    return all_ok


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
