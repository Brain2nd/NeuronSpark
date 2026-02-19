"""
端到端梯度回传监测：检查 v7.5 全模型每一个可学习参数的梯度状态。

检查项：
1. 每层所有组件的梯度范数（SelectivePLIF 调制路径、PLIFNode、突触、残差流）
2. 编码/解码/Embedding 的梯度
3. 无 NaN、无零梯度
4. 层间梯度衰减比（第 0 层 vs 最后层）
"""

import sys
sys.path.insert(0, '/home/dgxspark/Desktop/NeuronSpark')

import torch
from model import SNNLanguageModel


def check_grad(name, param):
    """返回 (grad_norm, status) 元组"""
    if param.grad is None:
        return 0.0, "NONE"
    if torch.isnan(param.grad).any():
        return float('nan'), "NaN!"
    norm = param.grad.norm().item()
    if norm == 0.0:
        return 0.0, "ZERO"
    if norm < 1e-15:
        return norm, "VANISH"
    return norm, "OK"


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}\n")

    model = SNNLanguageModel(
        vocab_size=64, D=32, N=4, K=4, num_layers=20, D_ff=64,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total params: {total_params:,}  Trainable: {trainable:,}\n")

    x = torch.randint(0, 64, (2, 16), device=device)
    y = torch.randint(0, 64, (2, 16), device=device)

    out = model(x, y)
    loss = out.last_loss.mean()
    print(f"Loss: {loss.item():.4f}\n")
    loss.backward()

    all_ok = True
    failed = []

    # ====== 1. 全局组件 ======
    print("=" * 70)
    print("全局组件")
    print("=" * 70)
    for name, param in [
        ("embed_tokens", model.embed_tokens.weight),
        ("encode_proj.weight", model.encode_proj.weight),
        ("encode_proj.bias", model.encode_proj.bias),
        ("decode_proj.weight", model.decode_proj.weight),
        ("decode_proj.bias", model.decode_proj.bias),
        ("output_norm", model.norm.gain),
    ]:
        norm, status = check_grad(name, param)
        print(f"  {name:25s}  shape={str(list(param.shape)):15s}  grad={norm:.2e}  [{status}]")
        if status != "OK":
            all_ok = False
            failed.append(name)

    # ====== 2. 逐层检查 ======
    for i, layer in enumerate(model.layers):
        print(f"\n{'=' * 70}")
        print(f"Layer {i}")
        print(f"{'=' * 70}")
        block = layer.snn_block
        ffn = layer.snn_ffn

        components = []

        # 残差流: 输入 PLIFNode + out_proj（无层内归一化）
        components += [
            ("input_neuron1.w", layer.input_neuron1.w),
            ("input_neuron1.v_th", layer.input_neuron1.v_th),
            ("input_neuron2.w", layer.input_neuron2.w),
            ("input_neuron2.v_th", layer.input_neuron2.v_th),
            ("block_out_proj", layer.block_out_proj.weight),
            ("ffn_out_proj", layer.ffn_out_proj.weight),
        ]

        # SNNBlock: 6 路突触 + 调制偏置 + SelectivePLIF(无自有参数) + 输出 PLIFNode
        components += [
            ("block.W_in", block.W_in.weight),
            ("block.W_beta_x", block.W_beta_x.weight),
            ("block.W_alpha_x", block.W_alpha_x.weight),
            ("block.W_th_x", block.W_th_x.weight),
            ("block.W_gate", block.W_gate.weight),
            ("block.W_skip", block.W_skip.weight),
            ("block.W_out", block.W_out.weight),
            ("block.b_beta", block.b_beta),
            ("block.b_alpha", block.b_alpha),
            ("block.b_th", block.b_th),
            ("block.out_neuron.w", block.output_neuron.w),
            ("block.out_neuron.v_th", block.output_neuron.v_th),
        ]

        # SNNFFN: 投影 + 3 个 PLIFNode
        components += [
            ("ffn.gate_proj", ffn.gate_proj.weight),
            ("ffn.up_proj", ffn.up_proj.weight),
            ("ffn.down_proj", ffn.down_proj.weight),
            ("ffn.skip_proj", ffn.skip_proj.weight),
            ("ffn.gate_neuron.w", ffn.gate_neuron.w),
            ("ffn.gate_neuron.v_th", ffn.gate_neuron.v_th),
            ("ffn.up_neuron.w", ffn.up_neuron.w),
            ("ffn.up_neuron.v_th", ffn.up_neuron.v_th),
            ("ffn.out_neuron.w", ffn.output_neuron.w),
            ("ffn.out_neuron.v_th", ffn.output_neuron.v_th),
        ]

        for name, param in components:
            norm, status = check_grad(name, param)
            tag = f"L{i:02d}.{name}"
            print(f"  {name:25s}  shape={str(list(param.shape)):15s}  grad={norm:.2e}  [{status}]")
            if status != "OK":
                all_ok = False
                failed.append(tag)

    # ====== 3. 层间梯度衰减汇总 ======
    print(f"\n{'=' * 70}")
    print("层间梯度衰减汇总（第 0 层 / 最后层 比值）")
    print(f"{'=' * 70}")

    probe_params = [
        ("block_out_proj", lambda l: l.block_out_proj.weight),
        ("ffn_out_proj", lambda l: l.ffn_out_proj.weight),
        ("block.W_in", lambda l: l.snn_block.W_in.weight),
        ("block.b_beta", lambda l: l.snn_block.b_beta),
        ("ffn.gate_proj", lambda l: l.snn_ffn.gate_proj.weight),
        ("input_neuron1.w", lambda l: l.input_neuron1.w),
        ("block.out_neuron.w", lambda l: l.snn_block.output_neuron.w),
    ]

    for pname, getter in probe_params:
        g0 = getter(model.layers[0]).grad
        gN = getter(model.layers[-1]).grad
        if g0 is not None and gN is not None:
            n0 = g0.norm().item()
            nN = gN.norm().item()
            ratio = n0 / (nN + 1e-30)
            status = "OK" if ratio > 1e-3 else "WARN"
            print(f"  {pname:25s}  L0={n0:.2e}  L19={nN:.2e}  ratio={ratio:.2e}  [{status}]")
            if status != "OK":
                all_ok = False

    # ====== 4. 总结 ======
    print(f"\n{'=' * 70}")
    total_checked = sum(1 for p in model.parameters() if p.requires_grad)
    grad_ok = sum(1 for p in model.parameters() if p.requires_grad and p.grad is not None and p.grad.norm().item() > 1e-15)
    print(f"可训练参数: {total_checked}  有效梯度: {grad_ok}  覆盖率: {grad_ok}/{total_checked}")

    if failed:
        print(f"\n失败参数: {failed}")

    if all_ok:
        print("\nALL PASSED: 端到端梯度回传正常，所有参数都有有效梯度。")
    else:
        print("\nFAILED: 存在梯度问题。")

    return all_ok


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
