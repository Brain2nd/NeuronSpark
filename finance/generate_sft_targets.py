"""
SFT 标签生成：基于未来实际价格的回测最优决策

与 BC 规则标签的区别:
  BC:  基于当前技术指标（EMA 交叉 + RSI + ADX）→ 只用已知信息
  SFT: 基于未来 H 小时的实际价格走势 → "事后诸葛亮"最优决策

输出 targets_sft.npy (n_steps, 4):
  [0] position:    未来收益方向+幅度, clamp [-1, +1]
  [1] leverage:    基于收益/回撤比, clamp [1, max_leverage]
  [2] stop_loss:   最大不利偏移 (MAE) + buffer, clamp [sl_min, sl_max]
  [3] take_profit: 最大有利偏移 (MFE) × discount, clamp [tp_min, tp_max]

用法:
    python finance/generate_sft_targets.py \
        --raw_dir data/finance/raw \
        --out_dir data/finance \
        --horizon 24 \
        --target_pair BTC_USDT
"""

import os
import argparse
import numpy as np
import pandas as pd


def generate_sft_targets(
    close: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    horizon: int = 24,
    max_leverage: float = 10.0,
    sl_min: float = 0.005,
    sl_max: float = 0.10,
    tp_min: float = 0.01,
    tp_max: float = 0.20,
) -> np.ndarray:
    """基于未来 H 步的实际价格生成最优决策标签。

    Args:
        close: (n,) 收盘价序列
        high:  (n,) 最高价序列
        low:   (n,) 最低价序列
        horizon: 前瞻窗口 (小时数)
        max_leverage: 杠杆上限

    Returns:
        targets: (n, 4) float32 — [position, leverage, stop_loss, take_profit]
    """
    n = len(close)
    targets = np.zeros((n, 4), dtype=np.float32)

    for i in range(n - horizon):
        entry = close[i]
        if entry <= 0:
            continue

        # 未来 H 步价格
        future_close = close[i + 1 : i + 1 + horizon]
        future_high  = high[i + 1 : i + 1 + horizon]
        future_low   = low[i + 1 : i + 1 + horizon]

        # ---- Position: 未来累积收益方向+幅度 ----
        # 使用 horizon 末尾收益
        final_return = (future_close[-1] - entry) / entry
        # 也看中间最大收益（加权）
        max_up   = (future_high.max() - entry) / entry
        max_down = (entry - future_low.min()) / entry

        # 方向：正收益 → 做多，负收益 → 做空
        # 幅度：用 final_return 缩放，但不超过 ±1
        # 放大系数让小幅变动也有非零信号
        position = np.clip(final_return * 20.0, -1.0, 1.0)

        # ---- Stop Loss: 最大不利偏移 (MAE) ----
        if position >= 0:
            # 做多：MAE = entry 到 future_low 最低点
            mae = max_down
        else:
            # 做空：MAE = future_high 最高点到 entry
            mae = max_up

        # 加 20% buffer 防止刚好触发
        stop_loss = np.clip(mae * 1.2, sl_min, sl_max)

        # ---- Take Profit: 最大有利偏移 (MFE) ----
        if position >= 0:
            mfe = max_up
        else:
            mfe = max_down

        # 打 8 折，保守一点确保能成交
        take_profit = np.clip(mfe * 0.8, tp_min, tp_max)

        # ---- Leverage: 基于收益/回撤比 ----
        if mae > 1e-6:
            # 简单 R/R ratio
            rr = abs(final_return) / mae
            # rr=0 → 1x, rr=2 → max_leverage
            lev = 1.0 + (max_leverage - 1.0) * np.clip(rr / 2.0, 0, 1)
        else:
            # 几乎无回撤 → 高杠杆
            lev = max_leverage if abs(final_return) > 0.001 else 1.0
        leverage = np.clip(lev, 1.0, max_leverage)

        # 如果最终收益接近零，减小杠杆
        if abs(final_return) < 0.001:
            leverage = 1.0
            position = 0.0

        targets[i] = [position, leverage, stop_loss, take_profit]

    return targets


def main():
    parser = argparse.ArgumentParser(description='SFT 标签生成: 回测最优决策')
    parser.add_argument('--raw_dir', type=str, default='data/finance/raw')
    parser.add_argument('--out_dir', type=str, default='data/finance')
    parser.add_argument('--target_pair', type=str, default='BTC_USDT')
    parser.add_argument('--timeframe', type=str, default='1h')
    parser.add_argument('--horizon', type=int, default=24,
                        help='前瞻窗口 (小时)')
    parser.add_argument('--max_leverage', type=float, default=10.0)
    parser.add_argument('--sl_min', type=float, default=0.005)
    parser.add_argument('--sl_max', type=float, default=0.10)
    parser.add_argument('--tp_min', type=float, default=0.01)
    parser.add_argument('--tp_max', type=float, default=0.20)
    args = parser.parse_args()

    # 加载原始数据（只需 target pair 的 close/high/low）
    path = os.path.join(args.raw_dir, f'{args.target_pair}-{args.timeframe}.feather')
    df = pd.read_feather(path)
    df = df.sort_values('date').reset_index(drop=True)
    df['date'] = df['date'].dt.tz_localize(None)
    print(f'Loaded {args.target_pair}: {len(df)} rows')
    print(f'  Range: {df["date"].min()} → {df["date"].max()}')

    # 需要与 features.npy 对齐 — 加载所有 9 对的日期取交集
    all_pairs = ['BTC_USDT', 'ETH_USDT', 'SOL_USDT', 'DOGE_USDT', 'XRP_USDT',
                 'ADA_USDT', 'AVAX_USDT', 'LINK_USDT', 'DOT_USDT']
    date_sets = []
    for pair in all_pairs:
        p = os.path.join(args.raw_dir, f'{pair}-{args.timeframe}.feather')
        if os.path.exists(p):
            tmp = pd.read_feather(p)
            tmp['date'] = tmp['date'].dt.tz_localize(None)
            date_sets.append(set(tmp['date']))
    common_dates = sorted(set.intersection(*date_sets))
    print(f'Common dates across {len(date_sets)} pairs: {len(common_dates)}')

    # 对齐
    mask = df['date'].isin(common_dates)
    df = df[mask].reset_index(drop=True)
    print(f'Aligned rows: {len(df)}')

    # 还需要去掉 features.npy 中因 NaN 被删除的行
    # 加载 features.npy 获取有效行数
    feat_path = os.path.join(args.out_dir, 'features.npy')
    features = np.load(feat_path)
    n_valid = features.shape[0]
    print(f'Features rows: {n_valid}')

    # features.npy 是从 common_dates 中去掉前 N 行（技术指标 warm-up）得到的
    # 去掉的行数 = len(common_dates) - n_valid，都在序列头部
    n_skip = len(df) - n_valid
    print(f'Skipping first {n_skip} warm-up rows')
    df = df.iloc[n_skip:].reset_index(drop=True)
    assert len(df) == n_valid, f"Row mismatch: {len(df)} vs {n_valid}"

    # 生成 SFT 标签
    close = df['close'].values.astype(np.float64)
    high = df['high'].values.astype(np.float64)
    low = df['low'].values.astype(np.float64)

    print(f'\nGenerating SFT targets (horizon={args.horizon}h)...')
    targets = generate_sft_targets(
        close, high, low,
        horizon=args.horizon,
        max_leverage=args.max_leverage,
        sl_min=args.sl_min, sl_max=args.sl_max,
        tp_min=args.tp_min, tp_max=args.tp_max,
    )

    # 统计
    print(f'\nSFT Target statistics:')
    labels = ['position', 'leverage', 'stop_loss', 'take_profit']
    for i, name in enumerate(labels):
        col = targets[:, i]
        nonzero = col[col != 0]
        print(f'  {name:12s}: mean={col.mean():.4f}, std={col.std():.4f}, '
              f'min={col.min():.4f}, max={col.max():.4f}, '
              f'nonzero={len(nonzero)}/{len(col)} ({100*len(nonzero)/len(col):.1f}%)')

    # Position 方向分布
    pos = targets[:, 0]
    n_long = (pos > 0).sum()
    n_short = (pos < 0).sum()
    n_flat = (pos == 0).sum()
    print(f'\n  Direction: long={n_long} ({100*n_long/len(pos):.1f}%), '
          f'short={n_short} ({100*n_short/len(pos):.1f}%), '
          f'flat={n_flat} ({100*n_flat/len(pos):.1f}%)')

    # 保存
    out_path = os.path.join(args.out_dir, 'targets_sft.npy')
    np.save(out_path, targets)
    print(f'\nSaved: {out_path} ({targets.nbytes / 1e6:.1f} MB)')


if __name__ == '__main__':
    main()
