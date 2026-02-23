"""
MoE 优化 Benchmark: 100 次 fwd + bwd + optimizer.step()
测试 1: 隔离 MoE _batched_expert_forward
测试 2: 全模型端到端（含 gradient checkpointing）
"""

import time
import types
import torch
from spikingjelly.activation_based import functional

from model import SNNLanguageModel
from atomic_ops.parallel_scan import plif_rowparam_forward_alpha


# ============================================================
# 旧版 _batched_expert_forward (TK=8192 顺序扫描)
# ============================================================
def _batched_expert_forward_OLD(self, spike_in_seq):
    E = self.num_experts
    TK, B, D = spike_in_seq.shape
    D_ff = self.D_ff
    flat = spike_in_seq.reshape(TK * B, D)

    proj = torch.bmm(
        flat.unsqueeze(0).expand(E, -1, -1),
        self.expert_W_gus.transpose(1, 2),
    )
    I_gate_up = proj[:, :, :2 * D_ff].reshape(E, TK, B, 2 * D_ff)
    I_skip = proj[:, :, 2 * D_ff:].reshape(E, TK, B, D)

    beta_gu = torch.sigmoid(self.expert_gu_w)
    scale_gu = 1.0 - beta_gu
    u_merged = I_gate_up * scale_gu.reshape(E, 1, 1, 2 * D_ff)

    u_flat = u_merged.permute(1, 0, 2, 3).reshape(TK, E * B, 2 * D_ff)
    beta_row = beta_gu.unsqueeze(1).expand(E, B, 2 * D_ff).reshape(E * B, 2 * D_ff).contiguous()
    v_th_row = self.expert_gu_v_th.unsqueeze(1).expand(E, B, 2 * D_ff).reshape(E * B, 2 * D_ff).contiguous()

    if isinstance(self.expert_gu_v, float):
        v_init = self.expert_gu_v_init.to(dtype=flat.dtype).unsqueeze(1).expand(E, B, 2 * D_ff).reshape(E * B, 2 * D_ff)
    else:
        v_init = self.expert_gu_v

    alpha_gu = self._compute_adaptive_alpha(u_flat)
    spike_flat, V_post_flat = plif_rowparam_forward_alpha(
        beta_row, u_flat, v_th_row, v_init, alpha_gu,
    )
    self.expert_gu_v = V_post_flat[-1].detach()
    spike_all = spike_flat.reshape(TK, E, B, 2 * D_ff)

    gate_spike = spike_all[:, :, :, :D_ff]
    up_spike = spike_all[:, :, :, D_ff:]
    gated = gate_spike * up_spike

    gated_flat = gated.permute(1, 0, 2, 3).reshape(E, TK * B, D_ff)
    I_out = torch.bmm(gated_flat, self.expert_W_down.transpose(1, 2))
    I_out = I_out.reshape(E, TK, B, D) + I_skip

    beta_out = torch.sigmoid(self.expert_out_w)
    scale_out = 1.0 - beta_out
    u_out = I_out * scale_out.reshape(E, 1, 1, D)

    u_out_flat = u_out.permute(1, 0, 2, 3).reshape(TK, E * B, D)
    beta_out_row = beta_out.unsqueeze(1).expand(E, B, D).reshape(E * B, D).contiguous()
    v_th_out_row = self.expert_out_v_th.unsqueeze(1).expand(E, B, D).reshape(E * B, D).contiguous()

    if isinstance(self.expert_out_v, float):
        v_init_out = self.expert_out_v_init.to(dtype=flat.dtype).unsqueeze(1).expand(E, B, D).reshape(E * B, D)
    else:
        v_init_out = self.expert_out_v

    alpha_out = self._compute_adaptive_alpha(u_out_flat)
    spike_out_flat, V_post_out_flat = plif_rowparam_forward_alpha(
        beta_out_row, u_out_flat, v_th_out_row, v_init_out, alpha_out,
    )
    self.expert_out_v = V_post_out_flat[-1].detach()
    expert_outs = spike_out_flat.reshape(TK, E, B, D).permute(1, 0, 2, 3).contiguous()
    return expert_outs


def trimmed_mean(vals, pct=0.1):
    n = len(vals); k = int(n * pct); s = sorted(vals)
    return sum(s[k:-k]) / (n - 2 * k) * 1000


def bench_moe_isolated(label, moe, spike_in, n_iters=100, warmup=20):
    """隔离 MoE _batched_expert_forward"""
    optimizer = torch.optim.AdamW(moe.parameters(), lr=3e-4)

    for _ in range(warmup):
        functional.reset_net(moe)
        out = moe._batched_expert_forward(spike_in)
        out.sum().backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    fwd_t, bwd_t, step_t = [], [], []

    for _ in range(n_iters):
        functional.reset_net(moe)
        torch.cuda.synchronize(); t0 = time.perf_counter()
        out = moe._batched_expert_forward(spike_in)
        torch.cuda.synchronize(); t1 = time.perf_counter()
        out.sum().backward()
        torch.cuda.synchronize(); t2 = time.perf_counter()
        optimizer.step(); optimizer.zero_grad(set_to_none=True)
        torch.cuda.synchronize(); t3 = time.perf_counter()
        fwd_t.append(t1 - t0); bwd_t.append(t2 - t1); step_t.append(t3 - t2)

    peak = torch.cuda.max_memory_allocated()
    f, b, s = trimmed_mean(fwd_t), trimmed_mean(bwd_t), trimmed_mean(step_t)
    print(f"  {label:<40} fwd:{f:7.2f}  bwd:{b:7.2f}  step:{s:5.2f}  tot:{f+b+s:7.2f} ms  mem:{peak/1024**2:.0f}MB")
    return {'fwd': f, 'bwd': b, 'step': s, 'peak': peak}


def bench_model_e2e(label, model, x, y, n_iters=100, warmup=10):
    """全模型 fwd + bwd + optimizer.step"""
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

    for _ in range(warmup):
        functional.reset_net(model)
        out = model(x, y); loss = out.last_loss.mean()
        loss.backward(); optimizer.step(); optimizer.zero_grad(set_to_none=True)

    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    fwd_t, bwd_t, step_t = [], [], []

    for _ in range(n_iters):
        functional.reset_net(model)
        torch.cuda.synchronize(); t0 = time.perf_counter()
        out = model(x, y); loss = out.last_loss.mean()
        torch.cuda.synchronize(); t1 = time.perf_counter()
        loss.backward()
        torch.cuda.synchronize(); t2 = time.perf_counter()
        optimizer.step(); optimizer.zero_grad(set_to_none=True)
        torch.cuda.synchronize(); t3 = time.perf_counter()
        fwd_t.append(t1 - t0); bwd_t.append(t2 - t1); step_t.append(t3 - t2)

    peak = torch.cuda.max_memory_allocated()
    f, b, s = trimmed_mean(fwd_t), trimmed_mean(bwd_t), trimmed_mean(step_t)
    print(f"  {label:<40} fwd:{f:7.1f}  bwd:{b:7.1f}  step:{s:5.1f}  tot:{f+b+s:7.1f} ms  mem:{peak/1024**2:.0f}MB")
    return {'fwd': f, 'bwd': b, 'step': s, 'peak': peak}


def print_cmp(r_old, r_new):
    for name, key in [('Forward', 'fwd'), ('Backward', 'bwd'), ('Optim', 'step')]:
        o, n = r_old[key], r_new[key]
        print(f"    {name:<10} {o:8.2f} → {n:8.2f} ms  ({o/n:.2f}x)")
    to = r_old['fwd'] + r_old['bwd'] + r_old['step']
    tn = r_new['fwd'] + r_new['bwd'] + r_new['step']
    print(f"    {'Total':<10} {to:8.2f} → {tn:8.2f} ms  ({to/tn:.2f}x)")
    mo, mn = r_old['peak'] / 1024**2, r_new['peak'] / 1024**2
    print(f"    {'Memory':<10} {mo:8.0f} → {mn:8.0f} MB  ({mn-mo:+.0f})")


def patch_old(model):
    """Monkey-patch 所有 MoE 层为旧版"""
    for layer in model.layers:
        if hasattr(layer, 'snn_ffn') and hasattr(layer.snn_ffn, '_batched_expert_forward'):
            moe = layer.snn_ffn
            moe._batched_expert_forward = types.MethodType(_batched_expert_forward_OLD, moe)


def main():
    torch.manual_seed(42)
    N_ITERS = 100

    cfg = dict(
        vocab_size=6144, D=256, N=4, K=16, num_layers=4, D_ff=768,
        use_moe=True, num_experts=4, top_k=2,
        D_ff_shared=256, D_ff_expert=128,
    )
    seq_len, batch_size = 512, 4
    TK = seq_len * cfg['K']

    print(f"D={cfg['D']}, layers={cfg['num_layers']}, E={cfg['num_experts']}, top_k={cfg['top_k']}")
    print(f"batch={batch_size}×{seq_len}, K={cfg['K']}, TK={TK}, iters={N_ITERS}")

    x = torch.randint(0, cfg['vocab_size'], (batch_size, seq_len)).cuda()
    y = torch.randint(0, cfg['vocab_size'], (batch_size, seq_len)).cuda()
    spike_in = torch.randint(0, 2, (TK, batch_size, cfg['D']), device='cuda', dtype=torch.bfloat16).float()

    # ======== 测试 1: 隔离 MoE _batched_expert_forward ========
    print(f"\n{'='*90}")
    print(f"  测试 1: 隔离 MoE _batched_expert_forward ({N_ITERS} iters)")
    print(f"{'='*90}")

    torch.manual_seed(42)
    m = SNNLanguageModel(**cfg).cuda()
    moe = m.layers[0].snn_ffn
    moe._batched_expert_forward = types.MethodType(_batched_expert_forward_OLD, moe)
    r1_old = bench_moe_isolated("OLD: TK=8192 seq scan", moe, spike_in, N_ITERS)
    del m, moe; torch.cuda.empty_cache()

    torch.manual_seed(42)
    m = SNNLanguageModel(**cfg).cuda()
    moe = m.layers[0].snn_ffn
    r1_new = bench_moe_isolated("NEW: K=16 per-token scan", moe, spike_in, N_ITERS)
    del m, moe; torch.cuda.empty_cache()

    print(f"\n  对比:")
    print_cmp(r1_old, r1_new)

    # ======== 测试 2: 全模型端到端 ========
    print(f"\n{'='*90}")
    print(f"  测试 2: 全模型端到端 ({N_ITERS} iters, 含 gradient checkpointing)")
    print(f"{'='*90}")

    torch.manual_seed(42)
    model_old = SNNLanguageModel(**cfg).cuda()
    patch_old(model_old)
    r2_old = bench_model_e2e("OLD: TK=8192 seq scan", model_old, x, y, N_ITERS)
    del model_old; torch.cuda.empty_cache()

    torch.manual_seed(42)
    model_new = SNNLanguageModel(**cfg).cuda()
    r2_new = bench_model_e2e("NEW: K=16 per-token scan", model_new, x, y, N_ITERS)
    del model_new; torch.cuda.empty_cache()

    print(f"\n  对比:")
    print_cmp(r2_old, r2_new)

    print(f"\n{'='*90}")


if __name__ == '__main__':
    main()
