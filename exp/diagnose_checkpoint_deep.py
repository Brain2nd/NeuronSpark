"""
v7.5 Checkpoint 深层诊断：表征收敛分析

检查项目：
  1. Embedding 多样性（token 间余弦相似度、有效秩）
  2. encode_proj 输出分布（sigmoid 前后、饱和度）
  3. K-bit spike 编码多样性（逐 bit 发放率、pattern 唯一性）
  4. 逐层 token 间余弦相似度 + 有效秩
  5. 解码层输出区分度
  6. 对比随机初始化模型

使用真实训练数据。
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ['TRITON_PTXAS_PATH'] = '/usr/local/cuda-13.0/bin/ptxas'

import torch
import torch.nn.functional as F
from model import SNNLanguageModel
from spikingjelly.activation_based import functional
from transformers import AutoTokenizer
from dataset import PretrainDataset
from torch.utils.data import DataLoader
import math

device = 'cuda'
dtype = torch.bfloat16

D = 768; N = 8; D_ff = 2304; num_layers = 20
vocab_size = 6144; K = 16; seq_len = 512


def cosine_sim_matrix(x):
    """x: (N, D) → (N, N) 余弦相似度矩阵"""
    x_norm = F.normalize(x, dim=-1)
    return x_norm @ x_norm.T


def off_diag_mean(sim):
    """相似度矩阵非对角线均值"""
    n = sim.shape[0]
    mask = ~torch.eye(n, dtype=torch.bool, device=sim.device)
    return sim[mask].mean().item()


def effective_rank(x):
    """信息熵有效秩。x: (N, D)"""
    q = min(x.shape[0], x.shape[1], 128)
    _, s, _ = torch.svd_lowrank(x.float(), q=q)
    p = s / s.sum()
    p = p[p > 1e-10]
    ent = -(p * torch.log(p)).sum()
    return torch.exp(ent).item()


def aggregate_tokens(h_TK, seq_len, K):
    """(TK, batch, D) → (seq_len, batch, D)，K 步均值"""
    return h_TK.reshape(seq_len, K, h_TK.shape[1], h_TK.shape[2]).mean(dim=1)


def main():
    ckpt_path = 'checkpoints/pretrain_768_20_6144.pth'

    print("=" * 80)
    print("Checkpoint 深层诊断：表征收敛分析")
    print("=" * 80)

    # ---- 加载模型 ----
    model = SNNLanguageModel(
        vocab_size=vocab_size, D=D, N=N, D_ff=D_ff,
        num_layers=num_layers, K=K,
    ).to(device).to(dtype)

    ckpt = torch.load(ckpt_path, map_location=device)
    state_dict = ckpt.get('model_state_dict', ckpt)
    model.load_state_dict(state_dict)
    model.eval()
    print(f"  step={ckpt.get('step')}, best_loss={ckpt.get('best_loss')}")

    # ---- 加载真实训练数据 ----
    print("  加载真实训练数据...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained("./tokenizer_snn/")
    dataset = PretrainDataset(
        "data/seq-monkey/seq_monkey_datawhale.jsonl",
        tokenizer, max_length=seq_len,
    )
    loader = DataLoader(dataset, batch_size=2, num_workers=0, shuffle=True)
    X, Y, loss_mask = next(iter(loader))
    input_ids = X.to(device)
    batch_sz = input_ids.shape[0]
    actual_seq = input_ids.shape[1]  # 511
    n_real = (input_ids != 0).sum(dim=1)
    print(f"  input_ids shape: {list(input_ids.shape)}, 非 pad token: {n_real.tolist()}")

    # ==================================================================
    # [1] Embedding 分析
    # ==================================================================
    print("\n" + "=" * 80)
    print("[1] Embedding 多样性")
    print("=" * 80)

    with torch.no_grad():
        emb = model.embed_tokens(input_ids).float()  # (batch, seq, D)

    print(f"  shape: {list(emb.shape)}")
    print(f"  mean={emb.mean():.4e}  std={emb.std():.4e}")

    for b in range(batch_sz):
        n = n_real[b].item()
        emb_real = emb[b, :n]  # 只取非 pad 部分
        sim = cosine_sim_matrix(emb_real)
        er = effective_rank(emb_real)
        print(f"  batch {b} (n={n}): cos_sim_offdiag={off_diag_mean(sim):.4f}  eff_rank={er:.1f}")

    # ==================================================================
    # [2] encode_proj + sigmoid
    # ==================================================================
    print("\n" + "=" * 80)
    print("[2] encode_proj 分析")
    print("=" * 80)

    with torch.no_grad():
        pre_sig = model.encode_proj(emb.to(dtype)).float()
        post_sig = torch.sigmoid(pre_sig)

    print(f"\n  Pre-sigmoid (encode_proj 线性输出):")
    print(f"    mean={pre_sig.mean():.4e}  std={pre_sig.std():.4e}")
    print(f"    min={pre_sig.min():.4e}  max={pre_sig.max():.4e}")
    print(f"    |x|>3 (饱和): {(pre_sig.abs() > 3).float().mean():.4f}")
    print(f"    |x|>5 (深度饱和): {(pre_sig.abs() > 5).float().mean():.4f}")
    print(f"    |x|<0.5 (线性区): {(pre_sig.abs() < 0.5).float().mean():.4f}")

    # bias 分析
    bias = model.encode_proj.bias.float()
    print(f"\n  encode_proj.bias:")
    print(f"    mean={bias.mean():.4e}  std={bias.std():.4e}  abs_mean={bias.abs().mean():.4e}")

    print(f"\n  Post-sigmoid [0,1]:")
    print(f"    mean={post_sig.mean():.4e}  std={post_sig.std():.4e}")

    # 分区段统计
    bins = [(0, 0.05), (0.05, 0.1), (0.1, 0.2), (0.2, 0.3), (0.3, 0.4),
            (0.4, 0.5), (0.5, 0.6), (0.6, 0.7), (0.7, 0.8), (0.8, 0.9),
            (0.9, 0.95), (0.95, 1.0)]
    print(f"\n    sigmoid 输出直方图:")
    for lo, hi in bins:
        frac = ((post_sig >= lo) & (post_sig < hi)).float().mean().item()
        bar = "#" * int(frac * 100)
        print(f"    [{lo:.2f}, {hi:.2f}): {frac:6.3f}  {bar}")

    # Per-dimension variance (across tokens)
    dim_std = post_sig.std(dim=(0, 1))  # (D,)
    dim_mean = post_sig.mean(dim=(0, 1))  # (D,)
    print(f"\n  Per-dimension analysis:")
    print(f"    dim_std: mean={dim_std.mean():.4e}  min={dim_std.min():.4e}  max={dim_std.max():.4e}")
    print(f"    dim_mean: min={dim_mean.min():.4f}  max={dim_mean.max():.4f}")
    print(f"    死维度 (std<0.01): {(dim_std < 0.01).sum().item()}/{D}")
    print(f"    低变异 (std<0.05): {(dim_std < 0.05).sum().item()}/{D}")
    print(f"    高变异 (std>0.15): {(dim_std > 0.15).sum().item()}/{D}")

    # 固定在 0 或 1 附近的维度
    near0 = (dim_mean < 0.1).sum().item()
    near1 = (dim_mean > 0.9).sum().item()
    near05 = ((dim_mean > 0.4) & (dim_mean < 0.6)).sum().item()
    print(f"    维度 mean<0.1: {near0}  mean>0.9: {near1}  mean∈[0.4,0.6]: {near05}")

    # Token-level cosine similarity
    for b in range(batch_sz):
        n = n_real[b].item()
        sim = cosine_sim_matrix(post_sig[b, :n])
        er = effective_rank(post_sig[b, :n])
        print(f"  batch {b}: sigmoid cos_sim={off_diag_mean(sim):.4f}  eff_rank={er:.1f}")

    # ==================================================================
    # [3] K-bit 编码分析
    # ==================================================================
    print("\n" + "=" * 80)
    print("[3] K-bit 编码分析")
    print("=" * 80)

    with torch.no_grad():
        spike_seq = model._encode_all_tokens(input_ids)  # (TK, batch, D)

    spike_f = spike_seq.float()
    TK_actual = spike_seq.shape[0]
    print(f"  spike_seq shape: {list(spike_seq.shape)}, TK={TK_actual}")
    print(f"  overall firing rate: {spike_f.mean():.4f}")

    # Reshape to (seq, K, batch, D)
    spike_by_token = spike_f.reshape(actual_seq, K, batch_sz, D)

    # Per-bit firing rate
    per_bit = spike_by_token.mean(dim=(0, 2, 3))  # (K,)
    print(f"\n  Per-bit (k) firing rate:")
    for k in range(K):
        bar = "#" * int(per_bit[k].item() * 100)
        print(f"    bit {k:2d} (2^{-(k+1):2d}): {per_bit[k]:.4f}  {bar}")

    # K-bit pattern diversity analysis
    # 对 h ≈ 0.5 的情况: bit_0=1, bit_1=0, ..., 所有 token 相同
    print(f"\n  K-bit pattern 多样性:")
    for b in range(batch_sz):
        n = n_real[b].item()
        # (n, K*D) binary pattern per token
        patterns = spike_by_token[:n, :, b, :].reshape(n, K * D)

        # Pairwise Hamming distance (sample 128 pairs to save compute)
        n_pairs = min(n, 128)
        idx = torch.randperm(n)[:n_pairs]
        pat_sample = patterns[idx]  # (n_pairs, K*D)
        hamming = (pat_sample.unsqueeze(0) != pat_sample.unsqueeze(1)).float().mean(-1)
        avg_hamming = hamming[~torch.eye(n_pairs, dtype=torch.bool, device=device)].mean().item()

        # Unique patterns per dimension (sample 32 dims)
        n_dims = min(D, 32)
        unique_per_dim = []
        for d in range(n_dims):
            col = spike_by_token[:n, :, b, d]  # (n, K) — K-bit code for dim d
            unique_codes = col.unique(dim=0).shape[0]
            unique_per_dim.append(unique_codes)
        avg_unique = sum(unique_per_dim) / len(unique_per_dim)
        max_possible = min(n, 2 ** K)

        print(f"  batch {b} (n={n}):")
        print(f"    avg Hamming distance: {avg_hamming:.4f} (0=全同, 0.5=完全不同)")
        print(f"    avg unique K-bit codes per dim: {avg_unique:.1f}/{max_possible} possible")

    # Token cosine similarity after K-step averaging
    token_avg = aggregate_tokens(spike_seq, actual_seq, K)  # (seq, batch, D)
    print(f"\n  K-avg token cos_sim:")
    for b in range(batch_sz):
        n = n_real[b].item()
        sim = cosine_sim_matrix(token_avg[:n, b].float())
        er = effective_rank(token_avg[:n, b].float())
        print(f"    batch {b}: cos_sim={off_diag_mean(sim):.4f}  eff_rank={er:.1f}")

    # ==================================================================
    # [4] 逐层 token 间余弦相似度 + 有效秩
    # ==================================================================
    print("\n" + "=" * 80)
    print("[4] 逐层表征收敛分析")
    print("=" * 80)

    layer_repr_stats = {}
    saved = []

    for i, lm in enumerate(model.layers):
        st = {}
        layer_repr_stats[i] = st
        orig = lm.forward_parallel
        saved.append((lm, orig))

        def _make(_orig, _st, _seq=actual_seq, _K=K, _n_real=n_real):
            def wrapper(h):
                h_out = _orig(h)
                h_tok = aggregate_tokens(h_out.detach(), _seq, _K).float()  # (seq, batch, D)
                sims, ers = [], []
                for b in range(h_tok.shape[1]):
                    n = _n_real[b].item()
                    sim = cosine_sim_matrix(h_tok[:n, b])
                    sims.append(off_diag_mean(sim))
                    ers.append(effective_rank(h_tok[:n, b]))
                _st['cos_sim'] = sum(sims) / len(sims)
                _st['eff_rank'] = sum(ers) / len(ers)
                _st['h_std'] = h_out.float().std().item()
                # 检查幅度方差的 per-token 分布
                h_tok_norms = h_tok.norm(dim=-1)  # (seq, batch)
                _st['norm_cv'] = (h_tok_norms.std() / (h_tok_norms.mean() + 1e-10)).item()  # 变异系数
                return h_out
            return wrapper

        lm.forward_parallel = _make(orig, st)

    with torch.no_grad():
        output = model(input_ids)

    for lm, orig in saved:
        lm.forward_parallel = orig

    print(f"\n{'L':>3s} {'h_std':>8s} {'cos_sim':>9s} {'eff_rank':>10s} {'norm_CV':>9s}")
    print("-" * 42)
    for i in range(num_layers):
        s = layer_repr_stats[i]
        print(f" {i:2d} {s['h_std']:>8.4f} {s['cos_sim']:>9.4f} {s['eff_rank']:>10.1f} {s['norm_cv']:>9.4f}")

    # 入口处的编码
    print(f"\n  编码层→L0: cos_sim = {'%.4f' % off_diag_mean(cosine_sim_matrix(token_avg[:n_real[0], 0].float()))}")
    print(f"  L0 出口:    cos_sim = {layer_repr_stats[0]['cos_sim']:.4f}")
    print(f"  L19 出口:   cos_sim = {layer_repr_stats[num_layers-1]['cos_sim']:.4f}")

    # ==================================================================
    # [5] 解码输出区分度
    # ==================================================================
    print("\n" + "=" * 80)
    print("[5] 解码层区分度")
    print("=" * 80)

    logits = output.logits.float()
    for b in range(batch_sz):
        n = n_real[b].item()
        sim = cosine_sim_matrix(logits[b, :n])
        er = effective_rank(logits[b, :n])
        print(f"  batch {b}: logits cos_sim={off_diag_mean(sim):.4f}  eff_rank={er:.1f}")

    # ==================================================================
    # [6] 对比随机初始化
    # ==================================================================
    print("\n" + "=" * 80)
    print("[6] 对比：随机初始化模型")
    print("=" * 80)

    torch.manual_seed(0)
    model_init = SNNLanguageModel(
        vocab_size=vocab_size, D=D, N=N, D_ff=D_ff,
        num_layers=num_layers, K=K,
    ).to(device).to(dtype)
    model_init.eval()

    with torch.no_grad():
        emb_i = model_init.embed_tokens(input_ids).float()
        pre_i = model_init.encode_proj(emb_i.to(dtype)).float()
        post_i = torch.sigmoid(pre_i)

    print(f"\n  Init encode_proj pre-sigmoid: mean={pre_i.mean():.4e} std={pre_i.std():.4e}")
    print(f"  Init sigmoid: mean={post_i.mean():.4e} std={post_i.std():.4e}")
    print(f"  Init sigmoid [0.4,0.6] 占比: {((post_i > 0.4) & (post_i < 0.6)).float().mean():.4f}")

    for b in range(batch_sz):
        n = n_real[b].item()
        sim_t = cosine_sim_matrix(post_i[b, :n])
        sim_e = cosine_sim_matrix(emb_i[b, :n])
        print(f"  batch {b}: init_emb cos_sim={off_diag_mean(sim_e):.4f}  "
              f"init_sigmoid cos_sim={off_diag_mean(sim_t):.4f}")

    # Init model K-bit encoding
    with torch.no_grad():
        spike_init = model_init._encode_all_tokens(input_ids)

    spike_init_by_tok = spike_init.float().reshape(actual_seq, K, batch_sz, D)
    per_bit_init = spike_init_by_tok.mean(dim=(0, 2, 3))
    print(f"\n  Init per-bit firing rate:")
    for k in range(K):
        print(f"    bit {k:2d}: init={per_bit_init[k]:.4f}  trained={per_bit[k]:.4f}")

    tok_avg_init = aggregate_tokens(spike_init, actual_seq, K)
    for b in range(batch_sz):
        n = n_real[b].item()
        sim = cosine_sim_matrix(tok_avg_init[:n, b].float())
        er = effective_rank(tok_avg_init[:n, b].float())
        print(f"  batch {b}: init K-avg cos_sim={off_diag_mean(sim):.4f}  eff_rank={er:.1f}")

    # Init model full forward (first 3 layers only to save time)
    layer_stats_init = {}
    saved_init = []
    for i, lm in enumerate(model_init.layers):
        st = {}
        layer_stats_init[i] = st
        orig = lm.forward_parallel
        saved_init.append((lm, orig))

        def _make(_orig, _st, _seq=actual_seq, _K=K, _n=n_real):
            def wrapper(h):
                h_out = _orig(h)
                h_tok = aggregate_tokens(h_out.detach(), _seq, _K).float()
                sims, ers = [], []
                for b in range(h_tok.shape[1]):
                    nb = _n[b].item()
                    sim = cosine_sim_matrix(h_tok[:nb, b])
                    sims.append(off_diag_mean(sim))
                    ers.append(effective_rank(h_tok[:nb, b]))
                _st['cos_sim'] = sum(sims) / len(sims)
                _st['eff_rank'] = sum(ers) / len(ers)
                return h_out
            return wrapper

        lm.forward_parallel = _make(orig, st)

    with torch.no_grad():
        model_init(input_ids)

    for lm, orig in saved_init:
        lm.forward_parallel = orig

    print(f"\n  逐层 cos_sim 对比 (trained vs init):")
    print(f"  {'L':>3s} {'trained':>9s} {'init':>9s} {'Δ':>9s} {'trained_rank':>13s} {'init_rank':>11s}")
    print(f"  " + "-" * 55)
    for i in range(num_layers):
        cs_t = layer_repr_stats[i]['cos_sim']
        cs_i = layer_stats_init[i]['cos_sim']
        er_t = layer_repr_stats[i]['eff_rank']
        er_i = layer_stats_init[i]['eff_rank']
        print(f"  {i:2d}  {cs_t:>9.4f} {cs_i:>9.4f} {cs_t - cs_i:>+9.4f} "
              f"{er_t:>13.1f} {er_i:>11.1f}")

    del model_init
    torch.cuda.empty_cache()

    print("\n" + "=" * 80)
    print("深层诊断完成")
    print("=" * 80)


if __name__ == '__main__':
    main()
