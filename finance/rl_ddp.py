"""
分布式金融模型 RL 脚本：阶段 3 PPO（DDP 多卡并行）

每张 GPU 独立收集不同窗口的 rollout，PPO 更新通过 DDP 自动同步梯度。

用法：
    torchrun --nproc_per_node=4 finance/rl_ddp.py \
        --pretrained_ckpt checkpoints_finance_sft/ckpt_step65.pth \
        --features_path data/finance/features.npy \
        --prices_path data/finance/raw/BTC_USDT-1h.feather

    # 单卡也能跑
    torchrun --nproc_per_node=1 finance/rl_ddp.py \
        --pretrained_ckpt checkpoints_finance_sft/ckpt_step65.pth \
        --features_path data/finance/features.npy \
        --prices_path data/finance/raw/BTC_USDT-1h.feather
"""

import os
import sys
import glob
import time
import math
import argparse
import warnings

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from contextlib import nullcontext
from spikingjelly.activation_based import functional

from finance.model import SNNFinanceModel, POS, LEV, SL, TP
from finance.rl import TradingEnv, PPOPolicy, compute_gae

warnings.filterwarnings('ignore')


# ============================================================
# 分布式工具
# ============================================================

def setup_distributed():
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    torch.cuda.set_device(local_rank)
    return local_rank, rank, world_size


def cleanup_distributed():
    dist.destroy_process_group()


def is_main_process(rank):
    return rank == 0


def Logger(content, rank=0):
    if is_main_process(rank):
        print(content)


# ============================================================
# Checkpoint
# ============================================================

def save_checkpoint(save_dir, policy, optimizer, step, epoch, metrics, max_keep=5):
    os.makedirs(save_dir, exist_ok=True)
    raw_model = policy.module if isinstance(policy, DDP) else policy
    path = os.path.join(save_dir, f'rl_ckpt_step{step}.pth')
    torch.save({
        'model_state_dict': raw_model.model.state_dict(),
        'value_head_state': raw_model.value_head.state_dict(),
        'log_std': raw_model.log_std.data,
        'optimizer_state': optimizer.state_dict(),
        'step': step,
        'epoch': epoch,
        'metrics': metrics,
        'stage': 'rl',
    }, path)
    print(f"  → Checkpoint saved: {path}")

    ckpts = sorted(glob.glob(os.path.join(save_dir, 'rl_ckpt_step*.pth')))
    while len(ckpts) > max_keep:
        old = ckpts.pop(0)
        os.remove(old)


def load_pretrained(path, policy, device, rank):
    Logger(f"Loading pretrained weights from {path}...", rank)
    ckpt = torch.load(path, map_location=device, weights_only=False)
    raw_policy = policy.module if isinstance(policy, DDP) else policy
    if 'model_state_dict' in ckpt:
        raw_policy.model.load_state_dict(ckpt['model_state_dict'], strict=False)
    Logger(f"  Loaded (step={ckpt.get('step', '?')}, stage={ckpt.get('stage', 'bc')})", rank)


# ============================================================
# PPO 训练循环 (DDP)
# ============================================================

def ppo_train(args):
    local_rank, rank, world_size = setup_distributed()
    device = torch.device(f"cuda:{local_rank}")

    ctx = torch.amp.autocast('cuda',
                             dtype=torch.bfloat16 if args.dtype == 'bfloat16' else torch.float16)

    # --- 加载数据 ---
    features = np.load(args.features_path, mmap_mode='r')
    total_steps, n_features = features.shape
    Logger(f"Features: {features.shape}", rank)

    prices_df = pd.read_feather(args.prices_path)
    prices_df = prices_df.sort_values('date').reset_index(drop=True)
    if prices_df['date'].dt.tz is not None:
        prices_df['date'] = prices_df['date'].dt.tz_localize(None)
    Logger(f"Prices: {len(prices_df)} bars", rank)

    start_idx = len(prices_df) - total_steps
    Logger(f"Price offset: features[0] → prices[{start_idx}]", rank)

    env = TradingEnv(prices_df, start_idx=start_idx,
                     fee_rate=args.fee_rate, slippage=args.slippage)

    # --- 模型 ---
    model = SNNFinanceModel(
        n_features=n_features,
        n_assets=args.n_assets,
        D=args.D, N=args.N, K=args.K,
        num_layers=args.num_layers,
        D_ff=args.D_ff,
        v_th_min=args.v_th_min,
        max_leverage=args.max_leverage,
        sl_range=(args.sl_min, args.sl_max),
        tp_range=(args.tp_min, args.tp_max),
    )

    raw_policy = PPOPolicy(model).to(device)

    if args.pretrained_ckpt:
        load_pretrained(args.pretrained_ckpt, raw_policy, device, rank)

    total_params = sum(p.numel() for p in raw_policy.parameters() if p.requires_grad)
    Logger(f"PPO Policy 总参数量：{total_params / 1e6:.3f}M", rank)

    # DDP 包装
    policy = DDP(raw_policy, device_ids=[local_rank])

    # --- Optimizer ---
    optimizer = torch.optim.AdamW([
        {'params': raw_policy.model.parameters(), 'lr': args.policy_lr},
        {'params': raw_policy.value_head.parameters(), 'lr': args.value_lr},
        {'params': [raw_policy.log_std], 'lr': args.policy_lr},
    ], weight_decay=0.01)

    # --- 序列窗口（按 rank 分配）---
    seq_len = args.seq_len
    stride = args.stride or seq_len // 2
    all_starts = list(range(0, total_steps - seq_len + 1, stride))

    # 每个 GPU 分到不同的窗口
    my_starts = [s for i, s in enumerate(all_starts) if i % world_size == rank]
    Logger(f"Rollout windows: {len(all_starts)} total, {len(my_starts)} per GPU", rank)

    # --- 训练信息 ---
    Logger(f"\n{'='*60}", rank)
    Logger(f"PPO RL Training (Stage 3, DDP, {world_size} GPUs)", rank)
    Logger(f"  Model:       D={args.D}, N={args.N}, K={args.K}, Layers={args.num_layers}", rank)
    Logger(f"  Params:      {total_params / 1e6:.3f}M", rank)
    Logger(f"  Windows:     {len(all_starts)} total, {len(my_starts)}/gpu", rank)
    Logger(f"  Seq length:  {seq_len}", rank)
    Logger(f"  PPO epochs:  {args.ppo_epochs}", rank)
    Logger(f"  Clip ratio:  {args.clip_ratio}", rank)
    Logger(f"  Policy LR:   {args.policy_lr}", rank)
    Logger(f"  Value LR:    {args.value_lr}", rank)
    Logger(f"  Gamma:       {args.gamma}, Lambda: {args.gae_lambda}", rank)
    Logger(f"  Fee rate:    {args.fee_rate}", rank)
    mem_baseline = torch.cuda.memory_allocated() / 1e9
    Logger(f"  CUDA memory: {mem_baseline:.2f} GB baseline (GPU {local_rank})", rank)
    Logger(f"{'='*60}\n", rank)

    # --- 主训练循环 ---
    global_step = 0

    for epoch in range(args.epochs):
        epoch_rewards = []
        epoch_policy_loss = []
        epoch_value_loss = []
        epoch_start = time.time()

        # 每个 epoch 随机打乱本 GPU 分到的窗口
        window_order = np.random.permutation(len(my_starts))

        for wi, window_idx in enumerate(window_order):
            seq_start = my_starts[window_idx]

            feat_np = features[seq_start:seq_start + seq_len].copy()
            feat_t = torch.from_numpy(feat_np).float().unsqueeze(0).to(device)

            # Rollout
            policy.eval()
            with torch.no_grad(), ctx:
                sampled_dec, values, log_probs, det_dec = raw_policy.forward_rollout(feat_t)

            sampled_np = sampled_dec[0].cpu().numpy()
            values_np = values[0].cpu().float().numpy()
            log_probs_t = log_probs[0].detach()

            actions_norm = raw_policy.model._normalize_decisions(sampled_dec).detach()

            rewards_np = env.compute_rewards(sampled_np[:, 0, :], seq_start, seq_len)
            epoch_rewards.append(rewards_np.sum())

            advantages_np, returns_np = compute_gae(
                rewards_np, values_np, gamma=args.gamma, lam=args.gae_lambda
            )
            advantages_t = torch.from_numpy(advantages_np).float().to(device)
            returns_t = torch.from_numpy(returns_np).float().to(device)

            if advantages_t.std() > 1e-8:
                advantages_t = (advantages_t - advantages_t.mean()) / (advantages_t.std() + 1e-8)

            # PPO 更新（DDP 自动同步梯度）
            policy.train()
            for ppo_ep in range(args.ppo_epochs):
                with ctx:
                    new_log_probs, new_values, entropy = raw_policy.evaluate_actions(
                        feat_t, actions_norm
                    )

                ratio = (new_log_probs[0] - log_probs_t).exp()
                clipped_ratio = ratio.clamp(1.0 - args.clip_ratio, 1.0 + args.clip_ratio)

                surr1 = ratio * advantages_t
                surr2 = clipped_ratio * advantages_t
                policy_loss = -torch.min(surr1, surr2).mean()

                value_loss = F.mse_loss(new_values[0], returns_t)

                loss = policy_loss + args.value_coef * value_loss - args.entropy_coef * entropy

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(policy.parameters(), args.grad_clip)
                optimizer.step()

            epoch_policy_loss.append(policy_loss.item())
            epoch_value_loss.append(value_loss.item())
            global_step += 1

            if (wi + 1) % args.log_interval == 0 and is_main_process(rank):
                avg_rew = np.mean(epoch_rewards[-args.log_interval:])
                avg_ploss = np.mean(epoch_policy_loss[-args.log_interval:])
                avg_vloss = np.mean(epoch_value_loss[-args.log_interval:])
                std_val = raw_policy.log_std.exp().detach().cpu().numpy()
                mem_str = f" | Mem {torch.cuda.memory_allocated()/1e9:.1f}/{torch.cuda.max_memory_allocated()/1e9:.1f}GB"
                print(
                    f"RL Epoch:{epoch+1}/{args.epochs} Win:{wi+1}/{len(my_starts)} "
                    f"R:{avg_rew:+.4f} PL:{avg_ploss:.4f} VL:{avg_vloss:.4f} "
                    f"std:[{std_val[0]:.3f},{std_val[1]:.3f},{std_val[2]:.3f},{std_val[3]:.3f}]"
                    f" | GPUs:{world_size}{mem_str}"
                )

        # Epoch 统计 — 汇总所有 GPU 的奖励
        local_reward = torch.tensor(sum(epoch_rewards), device=device)
        local_count = torch.tensor(len(epoch_rewards), device=device, dtype=torch.float32)
        if world_size > 1:
            dist.all_reduce(local_reward, op=dist.ReduceOp.SUM)
            dist.all_reduce(local_count, op=dist.ReduceOp.SUM)

        total_reward = local_reward.item()
        mean_reward = total_reward / max(local_count.item(), 1)
        elapsed = time.time() - epoch_start

        if is_main_process(rank):
            print(
                f"\n--- Epoch {epoch+1}/{args.epochs} done in {elapsed:.0f}s ---\n"
                f"  Total reward:  {total_reward:+.4f} (across {int(local_count.item())} windows)\n"
                f"  Mean reward:   {mean_reward:+.6f}\n"
                f"  Policy loss:   {np.mean(epoch_policy_loss):.4f}\n"
                f"  Value loss:    {np.mean(epoch_value_loss):.4f}\n"
            )

        # 保存
        if (epoch + 1) % args.save_interval == 0 or epoch == args.epochs - 1:
            if is_main_process(rank):
                save_checkpoint(
                    args.out_dir, policy, optimizer, global_step, epoch,
                    {'total_reward': total_reward, 'mean_reward': mean_reward},
                )
            if world_size > 1:
                dist.barrier()

    if is_main_process(rank):
        print(f"\nRL training finished. Global steps: {global_step}")
        print(f"Peak CUDA memory: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")

    cleanup_distributed()


# ============================================================
# 入口
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SNN Finance Model RL PPO (DDP)")

    # 模型参数
    parser.add_argument('--n_assets', type=int, default=1)
    parser.add_argument('--D', type=int, default=256)
    parser.add_argument('--N', type=int, default=8)
    parser.add_argument('--K', type=int, default=16)
    parser.add_argument('--num_layers', type=int, default=12)
    parser.add_argument('--D_ff', type=int, default=768)
    parser.add_argument('--v_th_min', type=float, default=0.1)
    parser.add_argument('--max_leverage', type=float, default=10.0)
    parser.add_argument('--sl_min', type=float, default=0.005)
    parser.add_argument('--sl_max', type=float, default=0.10)
    parser.add_argument('--tp_min', type=float, default=0.01)
    parser.add_argument('--tp_max', type=float, default=0.20)

    # PPO 参数
    parser.add_argument('--clip_ratio', type=float, default=0.2)
    parser.add_argument('--ppo_epochs', type=int, default=4)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--gae_lambda', type=float, default=0.95)
    parser.add_argument('--value_coef', type=float, default=0.5)
    parser.add_argument('--entropy_coef', type=float, default=0.01)

    # 训练参数
    parser.add_argument('--pretrained_ckpt', type=str, default=None)
    parser.add_argument('--out_dir', type=str, default='checkpoints_finance_rl')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--dtype', type=str, default='bfloat16')
    parser.add_argument('--seq_len', type=int, default=2048)
    parser.add_argument('--stride', type=int, default=None)
    parser.add_argument('--grad_clip', type=float, default=0.5)

    # 学习率
    parser.add_argument('--policy_lr', type=float, default=3e-5)
    parser.add_argument('--value_lr', type=float, default=1e-4)

    # 交易环境参数
    parser.add_argument('--fee_rate', type=float, default=0.0006)
    parser.add_argument('--slippage', type=float, default=0.0001)

    # 数据
    parser.add_argument('--features_path', type=str, required=True)
    parser.add_argument('--prices_path', type=str, required=True)

    # 日志和保存
    parser.add_argument('--log_interval', type=int, default=5)
    parser.add_argument('--save_interval', type=int, default=10)

    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    torch.manual_seed(42)
    np.random.seed(42)

    ppo_train(args)
