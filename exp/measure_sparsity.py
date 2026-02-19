"""
测量 v7.5 模型各层 spike 的实际稀疏度分布。

通过 forward hook 在各关键位置捕获 spike 张量，
测量元素级稀疏度和行级稀疏度。
"""

import sys
sys.path.insert(0, '/home/dgxspark/Desktop/NeuronSpark')

import torch
from model import SNNLanguageModel
from spikingjelly.activation_based import functional


def sparsity_stats(t, name=""):
    """测量张量的元素级和行级稀疏度。"""
    total = t.numel()
    zeros = (t == 0).sum().item()
    elem_sp = zeros / total

    flat = t.reshape(-1, t.shape[-1])
    row_all_zero = (~(flat != 0).any(dim=-1)).sum().item()
    total_rows = flat.shape[0]
    row_sp = row_all_zero / total_rows

    nnz_per_row = (flat != 0).sum(dim=-1).float()
    avg_nnz = nnz_per_row.mean().item()
    min_nnz = nnz_per_row.min().item()
    max_nnz = nnz_per_row.max().item()

    print(f"  {name:40s} {str(list(t.shape)):22s}  "
          f"elem={elem_sp:5.1%}  row_zero={row_sp:5.1%}  "
          f"nnz/row: {avg_nnz:.0f} [{min_nnz:.0f}-{max_nnz:.0f}]/{t.shape[-1]}")
    return elem_sp, row_sp


def main():
    device = 'cuda'
    print(f"Device: {device}\n")

    model = SNNLanguageModel(
        vocab_size=6144, D=768, N=8, K=16, num_layers=4, D_ff=2304,
    ).to(device).to(torch.bfloat16)
    model.eval()

    x = torch.randint(1, 6144, (2, 32), device=device)
    y = torch.randint(1, 6144, (2, 32), device=device)

    # ====== 方法：monkey-patch forward_parallel 来捕获中间 spike ======
    captured = {}

    # Patch SNNDecoderLayer._input_neuron_parallel
    for i, layer in enumerate(model.layers):
        orig_inp = layer._input_neuron_parallel

        def make_patched_inp(layer_idx, orig_fn):
            call_count = [0]
            def patched(neuron, x):
                spike = orig_fn(neuron, x)
                sublayer = "block" if call_count[0] % 2 == 0 else "ffn"
                captured[f"L{layer_idx}.input_neuron_{sublayer}"] = spike.detach()
                call_count[0] += 1
                return spike
            return patched

        layer._input_neuron_parallel = make_patched_inp(i, orig_inp)

    # Patch SNNBlock.forward_parallel 内部
    for i, layer in enumerate(model.layers):
        block = layer.snn_block
        orig_block_fwd = block.forward_parallel

        def make_patched_block(layer_idx, orig_fn, blk):
            def patched(spike_in_seq):
                result = orig_fn(spike_in_seq)
                # 同时也 patch output_neuron 的最后一次输出
                # 通过 hook 更难，直接记录 result
                captured[f"L{layer_idx}.block_output"] = result.detach()
                return result
            return patched

        block.forward_parallel = make_patched_block(i, orig_block_fwd, block)

    # Patch SNNFFN.forward_parallel 来捕获 gate/up/gated
    for i, layer in enumerate(model.layers):
        ffn = layer.snn_ffn
        orig_ffn_fwd = ffn.forward_parallel

        def make_patched_ffn(layer_idx, orig_fn, ffn_mod):
            def patched(spike_in):
                # 手动重现 forward_parallel 但捕获中间值
                from atomic_ops.parallel_scan import plif_rowparam_forward
                TK, batch, D = spike_in.shape
                D_ff = ffn_mod.D_ff

                flat = spike_in.reshape(TK * batch, D)
                gate_raw = torch.nn.functional.linear(flat, ffn_mod.gate_proj.weight)
                up_raw = torch.nn.functional.linear(flat, ffn_mod.up_proj.weight)
                skip_raw = torch.nn.functional.linear(flat, ffn_mod.skip_proj.weight)

                merged = torch.cat([gate_raw, up_raw], dim=-1).reshape(TK, batch, 2 * D_ff)

                beta_gate = ffn_mod.gate_neuron.beta.unsqueeze(0).expand(batch, D_ff).contiguous()
                beta_up = ffn_mod.up_neuron.beta.unsqueeze(0).expand(batch, D_ff).contiguous()
                beta_merged = torch.cat([beta_gate, beta_up], dim=-1)

                u_merged = (1.0 - beta_merged) * merged

                vth_gate = ffn_mod.gate_neuron.v_th.unsqueeze(0).expand(batch, D_ff).contiguous()
                vth_up = ffn_mod.up_neuron.v_th.unsqueeze(0).expand(batch, D_ff).contiguous()
                vth_merged = torch.cat([vth_gate, vth_up], dim=-1)

                v_init = torch.zeros(batch, 2 * D_ff, device=spike_in.device, dtype=spike_in.dtype)

                spike_merged, V_post = plif_rowparam_forward(
                    beta_merged, u_merged, vth_merged, v_init,
                    ffn_mod.gate_neuron.surrogate_function,
                )

                gate_spike = spike_merged[:, :, :D_ff]
                up_spike = spike_merged[:, :, D_ff:]
                gated = gate_spike * up_spike

                captured[f"L{layer_idx}.ffn_gate_spike"] = gate_spike.detach()
                captured[f"L{layer_idx}.ffn_up_spike"] = up_spike.detach()
                captured[f"L{layer_idx}.ffn_gated_AND"] = gated.detach()

                # down_proj + skip
                gated_flat = gated.reshape(TK * batch, D_ff)
                I_out = torch.nn.functional.linear(gated_flat, ffn_mod.down_proj.weight)
                I_out = I_out.reshape(TK, batch, D)
                I_skip = skip_raw.reshape(TK, batch, D)
                combined = I_out + I_skip

                # output neuron
                beta_out = ffn_mod.output_neuron.beta.unsqueeze(0).expand(batch, D).contiguous()
                vth_out = ffn_mod.output_neuron.v_th.unsqueeze(0).expand(batch, D).contiguous()
                u_out = (1.0 - beta_out) * combined
                v_init_out = torch.zeros(batch, D, device=spike_in.device, dtype=spike_in.dtype)

                spike_out, V_post_out = plif_rowparam_forward(
                    beta_out, u_out, vth_out, v_init_out,
                    ffn_mod.output_neuron.surrogate_function,
                )
                captured[f"L{layer_idx}.ffn_output"] = spike_out.detach()

                ffn_mod.gate_neuron.v = V_post[:, :D_ff][-1].detach()
                ffn_mod.up_neuron.v = V_post[:, D_ff:][-1].detach()
                ffn_mod.output_neuron.v = V_post_out[-1].detach()

                return spike_out

            return patched

        ffn.forward_parallel = make_patched_ffn(i, orig_ffn_fwd, ffn)

    # ====== 运行 forward ======
    functional.reset_net(model)
    with torch.no_grad():
        out = model(x, y)

    # ====== 输出结果 ======
    print("=" * 120)
    print("Spike 稀疏度分析")
    print("=" * 120)

    # 编码层
    print("\n--- 编码 ---")
    # 重新编码来测量
    with torch.no_grad():
        emb = model.embed_tokens(x)
        h_enc = torch.sigmoid(model.encode_proj(emb))
        scaled = h_enc.unsqueeze(2) * model.bit_scales.view(1, 1, model.K, 1)
        bit_hard = torch.floor(scaled) % 2
        spike_enc = bit_hard.reshape(2, 32 * 16, 768).permute(1, 0, 2)
    sparsity_stats(spike_enc, "encode K-bit spike")

    # 逐层
    for i in range(4):
        print(f"\n--- Layer {i} ---")
        for key in sorted(captured.keys()):
            if key.startswith(f"L{i}."):
                name = key[len(f"L{i}."):]
                sparsity_stats(captured[key], name)

    # ====== 汇总 ======
    print(f"\n{'=' * 120}")
    print("稀疏 matmul 潜力分析")
    print("=" * 120)

    for i in range(4):
        key_in = f"L{i}.input_neuron_block"
        key_gated = f"L{i}.ffn_gated_AND"
        if key_in in captured and key_gated in captured:
            in_sp = (captured[key_in] == 0).sum().item() / captured[key_in].numel()
            gated_sp = (captured[key_gated] == 0).sum().item() / captured[key_gated].numel()
            print(f"  Layer {i}: input_spike 稀疏度={in_sp:.1%} (6个matmul共享)  "
                  f"gated(AND) 稀疏度={gated_sp:.1%} (down_proj)")


if __name__ == '__main__':
    main()
