"""
金融模型 RL 脚本：阶段 3 — PPO 强化学习

在模拟交易环境中通过 PnL 奖励优化策略。

核心设计:
  - Policy: SNNFinanceModel (BC/SFT 预训练)
  - Value:  单独的 value_head (Linear D→1)
  - Action: 连续 4 维决策 (position, leverage, stop_loss, take_profit)
  - 探索:   对 raw logits 加高斯噪声 (learnable log_std)
  - Reward: 逐步 PnL = position × leverage × return − |Δposition| × fee
  - SL/TP:  使用 intra-bar high/low 模拟触发

训练流程:
  1. Rollout: 在历史数据上跑模型，收集 (state, action, reward, value, log_prob)
  2. GAE:    计算广义优势估计
  3. PPO:    多轮 mini-batch 更新 (clip ratio + value loss + entropy bonus)

用法:
    conda activate SNN

    python finance/rl.py \
        --pretrained_ckpt checkpoints_finance_sft/ckpt_step65.pth \
        --features_path data/finance/features.npy \
        --prices_path data/finance/raw/BTC_USDT-1h.feather \
        --epochs 100
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
from contextlib import nullcontext
from spikingjelly.activation_based import functional

from finance.model import SNNFinanceModel, POS, LEV, SL, TP

warnings.filterwarnings('ignore')


def Logger(content):
    print(content)


# ============================================================
# 交易环境模拟
# ============================================================

class TradingEnv:
    """模拟交易环境：根据模型决策和真实价格计算逐步 PnL 奖励。

    Reward 设计（每步）:
      pnl = position × leverage × bar_return
      fee = |position_change| × fee_rate
      sl/tp 触发: 使用 intra-bar high/low 检测

    Args:
        prices_df: DataFrame with [date, open, high, low, close, volume]
        start_idx: features 中第 0 行对应 prices 中的起始索引
        fee_rate: 单边手续费率（默认 0.06% = OKX maker fee）
        slippage: 滑点（默认 0.01%）
    """

    def __init__(self, prices_df, start_idx: int = 0,
                 fee_rate: float = 0.0006, slippage: float = 0.0001):
        self.close = prices_df['close'].values.astype(np.float64)
        self.high = prices_df['high'].values.astype(np.float64)
        self.low = prices_df['low'].values.astype(np.float64)
        self.start_idx = start_idx
        self.fee_rate = fee_rate
        self.slippage = slippage

    def compute_rewards(self, decisions: np.ndarray, seq_start: int, seq_len: int) -> np.ndarray:
        """计算一段序列的逐步奖励。

        Args:
            decisions: (seq_len, 4) — [position, leverage, stop_loss, take_profit]
            seq_start: 在 features 空间的起始索引
            seq_len: 序列长度

        Returns:
            rewards: (seq_len,) float32
        """
        rewards = np.zeros(seq_len, dtype=np.float32)
        prev_position = 0.0

        for t in range(seq_len - 1):
            price_idx = self.start_idx + seq_start + t

            if price_idx + 1 >= len(self.close):
                break

            pos = float(decisions[t, POS])
            lev = float(decisions[t, LEV])
            sl = float(decisions[t, SL])
            tp = float(decisions[t, TP])

            close_t = self.close[price_idx]
            close_t1 = self.close[price_idx + 1]
            high_t1 = self.high[price_idx + 1]
            low_t1 = self.low[price_idx + 1]

            bar_return = (close_t1 - close_t) / close_t

            # --- SL/TP 触发检测（使用 intra-bar high/low）---
            effective_return = bar_return
            if abs(pos) > 0.01:
                if pos > 0:  # Long
                    worst = (low_t1 - close_t) / close_t
                    best = (high_t1 - close_t) / close_t
                    if worst < -sl:
                        effective_return = -sl - self.slippage
                    elif best > tp:
                        effective_return = tp - self.slippage
                else:  # Short
                    worst = (high_t1 - close_t) / close_t
                    best = (low_t1 - close_t) / close_t
                    if worst > sl:
                        effective_return = sl + self.slippage  # loss for short
                        effective_return = -effective_return    # neg for short
                    elif best < -tp:
                        effective_return = tp - self.slippage
                        effective_return = effective_return     # profit for short (neg price move)

                # 对 short 而言 PnL 方向相反
                if pos > 0:
                    pnl = pos * lev * effective_return
                else:
                    pnl = pos * lev * effective_return  # pos is negative, so sign is correct

            else:
                pnl = 0.0

            # --- 手续费 ---
            position_change = abs(pos - prev_position)
            fee = position_change * self.fee_rate
            prev_position = pos

            rewards[t] = pnl - fee

        return rewards


# ============================================================
# PPO 策略包装
# ============================================================

class PPOPolicy(nn.Module):
    """在 SNNFinanceModel 基础上添加 value head + 探索噪声。

    - value_head:  Linear(D, 1) — 状态价值估计
    - log_std:     可学习参数 (4,) — 各决策维度的探索标准差
    """

    def __init__(self, model: SNNFinanceModel):
        super().__init__()
        self.model = model
        self.value_head = nn.Linear(model.D, 1)
        nn.init.zeros_(self.value_head.weight)
        nn.init.zeros_(self.value_head.bias)

        # 探索噪声: 在 normalized [0,1] 空间的 std
        # 初始 std ≈ 0.3 (exp(-1.2) ≈ 0.3)
        self.log_std = nn.Parameter(torch.full((4,), -1.2))

    def forward_rollout(self, features: torch.Tensor):
        """Rollout 前向: 返回采样的决策 + value + log_prob。

        Args:
            features: (batch, seq_len, n_features)

        Returns:
            sampled_decisions: (batch, seq_len, n_assets, 4)
            values:            (batch, seq_len)
            log_probs:         (batch, seq_len, n_assets)
            det_decisions:     (batch, seq_len, n_assets, 4) 确定性决策（均值）
        """
        batch, seq_len, _ = features.shape

        for layer_module in self.model.layers:
            functional.reset_net(layer_module)
        functional.reset_net(self.model.output_neuron)

        spike_seq = self.model.encode(features)
        h_out = self.model.snn_forward(spike_seq)

        # 输出边界
        h_out_normed = self.model.output_norm(h_out)
        spikes = self.model._output_neuron_parallel(h_out_normed)

        from atomic_ops.fp16_codec import fp16_decode
        decoded = fp16_decode(spikes, seq_len, K=self.model.K)
        h = self.model.decode_proj(decoded)
        h = self.model.norm(h)

        # Value
        values = self.value_head(h).squeeze(-1)  # (batch, seq_len)

        # 确定性决策（均值）
        raw = self.model.decision_head(h)
        raw = raw.view(batch, seq_len, self.model.n_assets, 4)
        det_decisions = self.model._activate_decisions(raw)

        # 归一化到 [0,1] 空间加高斯噪声
        det_norm = self.model._normalize_decisions(det_decisions)
        std = self.log_std.exp().clamp(min=0.01, max=1.0)  # (4,)

        # 采样
        noise = torch.randn_like(det_norm) * std
        sampled_norm = (det_norm + noise).clamp(0.0, 1.0)

        # 反归一化回物理空间
        sampled_decisions = self._denormalize(sampled_norm)

        # Log probability (对角高斯)
        var = std.pow(2)
        log_probs_per_dim = -0.5 * ((det_norm - sampled_norm).pow(2) / var + torch.log(var) + math.log(2 * math.pi))
        log_probs = log_probs_per_dim.sum(dim=-1).sum(dim=-1)  # sum over 4 dims + n_assets → (batch, seq_len)

        return sampled_decisions, values, log_probs, det_decisions

    def evaluate_actions(self, features: torch.Tensor, old_actions_norm: torch.Tensor):
        """PPO update 阶段: 重新计算 log_prob 和 value。

        Args:
            features:          (batch, seq_len, n_features)
            old_actions_norm:  (batch, seq_len, n_assets, 4) 归一化后的旧 action

        Returns:
            log_probs: (batch, seq_len)
            values:    (batch, seq_len)
            entropy:   scalar
        """
        batch, seq_len, _ = features.shape

        for layer_module in self.model.layers:
            functional.reset_net(layer_module)
        functional.reset_net(self.model.output_neuron)

        spike_seq = self.model.encode(features)
        h_out = self.model.snn_forward(spike_seq)

        h_out_normed = self.model.output_norm(h_out)
        spikes = self.model._output_neuron_parallel(h_out_normed)

        from atomic_ops.fp16_codec import fp16_decode
        decoded = fp16_decode(spikes, seq_len, K=self.model.K)
        h = self.model.decode_proj(decoded)
        h = self.model.norm(h)

        values = self.value_head(h).squeeze(-1)

        raw = self.model.decision_head(h)
        raw = raw.view(batch, seq_len, self.model.n_assets, 4)
        det_decisions = self.model._activate_decisions(raw)
        det_norm = self.model._normalize_decisions(det_decisions)

        std = self.log_std.exp().clamp(min=0.01, max=1.0)
        var = std.pow(2)

        log_probs_per_dim = -0.5 * ((det_norm - old_actions_norm).pow(2) / var + torch.log(var) + math.log(2 * math.pi))
        log_probs = log_probs_per_dim.sum(dim=-1).sum(dim=-1)

        # Entropy
        entropy = 0.5 * (1.0 + math.log(2 * math.pi)) + self.log_std.sum()

        return log_probs, values, entropy

    def _denormalize(self, norm: torch.Tensor) -> torch.Tensor:
        """[0,1] → 物理空间。"""
        m = self.model
        pos = norm[..., POS] * 2.0 - 1.0
        lev = 1.0 + (m.max_leverage - 1.0) * norm[..., LEV]
        sl = m.sl_min + (m.sl_max - m.sl_min) * norm[..., SL]
        tp = m.tp_min + (m.tp_max - m.tp_min) * norm[..., TP]
        return torch.stack([pos, lev, sl, tp], dim=-1)


# ============================================================
# GAE 计算
# ============================================================

def compute_gae(rewards, values, gamma=0.99, lam=0.95):
    """广义优势估计 (Generalized Advantage Estimation)。

    Args:
        rewards: (seq_len,) np.ndarray
        values:  (seq_len,) np.ndarray (含 bootstrap)
        gamma:   折扣因子
        lam:     GAE lambda

    Returns:
        advantages: (seq_len,) np.ndarray
        returns:    (seq_len,) np.ndarray
    """
    seq_len = len(rewards)
    advantages = np.zeros(seq_len, dtype=np.float32)
    gae = 0.0

    for t in reversed(range(seq_len - 1)):
        delta = rewards[t] + gamma * values[t + 1] - values[t]
        gae = delta + gamma * lam * gae
        advantages[t] = gae

    returns = advantages + values[:seq_len]
    return advantages, returns


# ============================================================
# Checkpoint
# ============================================================

def save_checkpoint(save_dir, policy, optimizer, step, epoch, metrics, max_keep=5):
    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, f'rl_ckpt_step{step}.pth')
    torch.save({
        'model_state_dict': policy.model.state_dict(),
        'value_head_state': policy.value_head.state_dict(),
        'log_std': policy.log_std.data,
        'optimizer_state': optimizer.state_dict(),
        'step': step,
        'epoch': epoch,
        'metrics': metrics,
        'stage': 'rl',
    }, path)
    Logger(f"  → Checkpoint saved: {path}")

    ckpts = sorted(glob.glob(os.path.join(save_dir, 'rl_ckpt_step*.pth')))
    while len(ckpts) > max_keep:
        old = ckpts.pop(0)
        os.remove(old)


def load_pretrained(path, policy, device):
    """加载 BC/SFT 预训练权重到 policy.model。"""
    Logger(f"Loading pretrained weights from {path}...")
    ckpt = torch.load(path, map_location=device, weights_only=False)
    if 'model_state_dict' in ckpt:
        policy.model.load_state_dict(ckpt['model_state_dict'], strict=False)
    Logger(f"  Loaded (step={ckpt.get('step', '?')}, stage={ckpt.get('stage', 'bc')})")


# ============================================================
# PPO 训练循环
# ============================================================

def ppo_train(args):
    device = args.device
    device_type = "cuda" if "cuda" in device else "cpu"
    ctx = nullcontext() if device_type == "cpu" else torch.amp.autocast(
        'cuda', dtype=torch.bfloat16 if args.dtype == 'bfloat16' else torch.float16)

    # --- 加载数据 ---
    features = np.load(args.features_path, mmap_mode='r')
    total_steps, n_features = features.shape
    Logger(f"Features: {features.shape}")

    # 加载价格数据（用于 reward 计算）
    prices_df = pd.read_feather(args.prices_path)
    prices_df = prices_df.sort_values('date').reset_index(drop=True)
    if prices_df['date'].dt.tz is not None:
        prices_df['date'] = prices_df['date'].dt.tz_localize(None)
    Logger(f"Prices: {len(prices_df)} bars")

    # features 的第 0 行对应 prices 的哪一行（warm-up 行被去掉了）
    # preprocess.py 去掉了前 ~199 行 NaN，所以 start_idx ≈ 199
    start_idx = len(prices_df) - total_steps
    Logger(f"Price offset: features[0] → prices[{start_idx}]")

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

    policy = PPOPolicy(model).to(device)

    if args.pretrained_ckpt:
        load_pretrained(args.pretrained_ckpt, policy, device)

    total_params = sum(p.numel() for p in policy.parameters() if p.requires_grad)
    Logger(f"PPO Policy 总参数量：{total_params / 1e6:.3f}M")

    # --- Optimizer ---
    optimizer = torch.optim.AdamW([
        {'params': policy.model.parameters(), 'lr': args.policy_lr},
        {'params': policy.value_head.parameters(), 'lr': args.value_lr},
        {'params': [policy.log_std], 'lr': args.policy_lr},
    ], weight_decay=0.01)

    # --- 序列窗口 ---
    seq_len = args.seq_len
    stride = args.stride or seq_len // 2
    starts = list(range(0, total_steps - seq_len + 1, stride))
    Logger(f"Rollout windows: {len(starts)} (seq_len={seq_len}, stride={stride})")

    # --- 训练信息 ---
    Logger(f"\n{'='*60}")
    Logger(f"PPO RL Training (Stage 3)")
    Logger(f"  Model:       D={args.D}, N={args.N}, K={args.K}, Layers={args.num_layers}")
    Logger(f"  Params:      {total_params / 1e6:.3f}M")
    Logger(f"  Windows:     {len(starts)}")
    Logger(f"  Seq length:  {seq_len}")
    Logger(f"  PPO epochs:  {args.ppo_epochs}")
    Logger(f"  Clip ratio:  {args.clip_ratio}")
    Logger(f"  Policy LR:   {args.policy_lr}")
    Logger(f"  Value LR:    {args.value_lr}")
    Logger(f"  Gamma:       {args.gamma}, Lambda: {args.gae_lambda}")
    Logger(f"  Fee rate:    {args.fee_rate}")
    Logger(f"  Entropy coef: {args.entropy_coef}")
    if device != 'cpu':
        Logger(f"  CUDA memory: {torch.cuda.memory_allocated() / 1e9:.2f} GB baseline")
    Logger(f"{'='*60}\n")

    # --- 主训练循环 ---
    global_step = 0

    for epoch in range(args.epochs):
        epoch_rewards = []
        epoch_policy_loss = []
        epoch_value_loss = []
        epoch_start = time.time()

        # 随机打乱窗口顺序
        window_order = np.random.permutation(len(starts))

        for wi, window_idx in enumerate(window_order):
            seq_start = starts[window_idx]

            # 1) 加载特征
            feat_np = features[seq_start:seq_start + seq_len].copy()
            feat_t = torch.from_numpy(feat_np).float().unsqueeze(0).to(device)  # (1, seq_len, F)

            # 2) Rollout: 收集轨迹
            policy.eval()
            with torch.no_grad(), ctx:
                sampled_dec, values, log_probs, det_dec = policy.forward_rollout(feat_t)

            sampled_np = sampled_dec[0].cpu().numpy()  # (seq_len, n_assets, 4)
            values_np = values[0].cpu().float().numpy()        # (seq_len,)
            log_probs_t = log_probs[0].detach()                # (seq_len,) on device

            # 归一化 action 存储（用于 evaluate_actions）
            actions_norm = policy.model._normalize_decisions(sampled_dec).detach()  # (1, S, A, 4)

            # 3) 计算奖励
            rewards_np = env.compute_rewards(sampled_np[:, 0, :], seq_start, seq_len)
            epoch_rewards.append(rewards_np.sum())

            # 4) GAE
            advantages_np, returns_np = compute_gae(
                rewards_np, values_np, gamma=args.gamma, lam=args.gae_lambda
            )
            advantages_t = torch.from_numpy(advantages_np).float().to(device)
            returns_t = torch.from_numpy(returns_np).float().to(device)

            # 标准化 advantage
            if advantages_t.std() > 1e-8:
                advantages_t = (advantages_t - advantages_t.mean()) / (advantages_t.std() + 1e-8)

            # 5) PPO 更新
            policy.train()
            for ppo_ep in range(args.ppo_epochs):
                with ctx:
                    new_log_probs, new_values, entropy = policy.evaluate_actions(
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

            if (wi + 1) % args.log_interval == 0:
                avg_rew = np.mean(epoch_rewards[-args.log_interval:])
                avg_ploss = np.mean(epoch_policy_loss[-args.log_interval:])
                avg_vloss = np.mean(epoch_value_loss[-args.log_interval:])
                std_val = policy.log_std.exp().detach().cpu().numpy()
                mem_str = ""
                if device != 'cpu':
                    mem_str = f" | Mem {torch.cuda.memory_allocated()/1e9:.1f}/{torch.cuda.max_memory_allocated()/1e9:.1f}GB"
                Logger(
                    f"RL Epoch:{epoch+1}/{args.epochs} Win:{wi+1}/{len(starts)} "
                    f"R:{avg_rew:+.4f} PL:{avg_ploss:.4f} VL:{avg_vloss:.4f} "
                    f"std:[{std_val[0]:.3f},{std_val[1]:.3f},{std_val[2]:.3f},{std_val[3]:.3f}]"
                    f"{mem_str}"
                )

        # Epoch 统计
        elapsed = time.time() - epoch_start
        total_reward = sum(epoch_rewards)
        mean_reward = np.mean(epoch_rewards)
        Logger(
            f"\n--- Epoch {epoch+1}/{args.epochs} done in {elapsed:.0f}s ---\n"
            f"  Total reward:  {total_reward:+.4f}\n"
            f"  Mean reward:   {mean_reward:+.6f}\n"
            f"  Policy loss:   {np.mean(epoch_policy_loss):.4f}\n"
            f"  Value loss:    {np.mean(epoch_value_loss):.4f}\n"
        )

        # 保存
        if (epoch + 1) % args.save_interval == 0 or epoch == args.epochs - 1:
            save_checkpoint(
                args.out_dir, policy, optimizer, global_step, epoch,
                {'total_reward': total_reward, 'mean_reward': mean_reward},
            )

    Logger(f"\nRL training finished. Global steps: {global_step}")
    if device != 'cpu':
        Logger(f"Peak CUDA memory: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")


# ============================================================
# 入口
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SNN Finance Model RL (Stage 3: PPO)")

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
    parser.add_argument('--ppo_epochs', type=int, default=4,
                        help='每个 rollout 窗口的 PPO 更新轮数')
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--gae_lambda', type=float, default=0.95)
    parser.add_argument('--value_coef', type=float, default=0.5)
    parser.add_argument('--entropy_coef', type=float, default=0.01)

    # 训练参数
    parser.add_argument('--pretrained_ckpt', type=str, default=None,
                        help='BC/SFT 预训练 checkpoint')
    parser.add_argument('--out_dir', type=str, default='checkpoints_finance_rl')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--device', type=str,
                        default='cuda:0' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--dtype', type=str, default='bfloat16')
    parser.add_argument('--seq_len', type=int, default=2048)
    parser.add_argument('--stride', type=int, default=None)
    parser.add_argument('--grad_clip', type=float, default=0.5)

    # 学习率
    parser.add_argument('--policy_lr', type=float, default=3e-5,
                        help='策略网络学习率')
    parser.add_argument('--value_lr', type=float, default=1e-4,
                        help='价值网络学习率')

    # 交易环境参数
    parser.add_argument('--fee_rate', type=float, default=0.0006,
                        help='单边手续费率 (OKX maker: 0.06%)')
    parser.add_argument('--slippage', type=float, default=0.0001,
                        help='滑点 (0.01%)')

    # 数据
    parser.add_argument('--features_path', type=str, required=True)
    parser.add_argument('--prices_path', type=str, required=True,
                        help='价格 feather 文件（需包含 OHLCV，用于 reward 计算）')

    # 日志和保存
    parser.add_argument('--log_interval', type=int, default=5)
    parser.add_argument('--save_interval', type=int, default=10,
                        help='每 N 个 epoch 保存一次')

    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    torch.manual_seed(42)
    np.random.seed(42)

    ppo_train(args)
