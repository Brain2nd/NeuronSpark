"""
金融数据预处理：OHLCV → features.npy + targets.npy

Pipeline:
  1. 加载 Freqtrade 下载的 feather 文件（BTC/ETH/SOL 1h K线）
  2. 计算技术指标特征（每品种 ~95 维，跨品种+时间 ~309 维总计）
  3. 生成规则策略 BC 标签（4 维决策：position, leverage, stop_loss, take_profit）
  4. 对齐时间轴、去除 NaN 行、保存 .npy

用法:
    python finance/preprocess.py \
        --raw_dir data/finance/raw \
        --out_dir data/finance \
        --pairs BTC_USDT ETH_USDT SOL_USDT \
        --max_leverage 10.0
"""

import os
import argparse
import numpy as np
import pandas as pd
import talib


# ====== 单品种技术指标 ======

def compute_single_asset_features(df: pd.DataFrame, prefix: str) -> pd.DataFrame:
    """计算单品种技术指标特征，所有特征归一化到合理范围。

    Args:
        df: DataFrame with columns [date, open, high, low, close, volume]
        prefix: 品种前缀（如 'btc_'），用于列名区分

    Returns:
        DataFrame with ~95 列技术指标特征
    """
    o = df['open'].values.astype(np.float64)
    h = df['high'].values.astype(np.float64)
    l = df['low'].values.astype(np.float64)
    c = df['close'].values.astype(np.float64)
    v = df['volume'].values.astype(np.float64)

    feats = {}

    # --- 1. Price ratios (6) ---
    feats['open_close_ratio'] = o / c - 1
    feats['high_close_ratio'] = h / c - 1
    feats['low_close_ratio'] = l / c - 1
    feats['high_low_range'] = (h - l) / c
    feats['upper_shadow'] = (h - np.maximum(o, c)) / c
    feats['lower_shadow'] = (np.minimum(o, c) - l) / c

    # --- 2. Returns at multiple periods (12) ---
    cs = pd.Series(c)
    for p in [1, 2, 3, 5, 10, 20]:
        feats[f'return_{p}'] = cs.pct_change(p).values
        feats[f'log_return_{p}'] = np.log(c / np.roll(c, p))
        feats[f'log_return_{p}'][:p] = 0

    # --- 3. SMA relative to close (6) ---
    for w in [5, 10, 20, 50, 100, 200]:
        sma = talib.SMA(c, timeperiod=w)
        feats[f'sma_{w}_rel'] = c / sma - 1

    # --- 4. EMA relative to close (6) ---
    for w in [5, 10, 20, 50, 100, 200]:
        ema = talib.EMA(c, timeperiod=w)
        feats[f'ema_{w}_rel'] = c / ema - 1

    # --- 5. MA crossovers (8) ---
    for (fast, slow) in [(5, 20), (10, 50), (20, 100), (50, 200)]:
        feats[f'sma_{fast}_{slow}_cross'] = talib.SMA(c, fast) / talib.SMA(c, slow) - 1
        feats[f'ema_{fast}_{slow}_cross'] = talib.EMA(c, fast) / talib.EMA(c, slow) - 1

    # --- 6. RSI (3) ---
    for p in [7, 14, 21]:
        feats[f'rsi_{p}'] = talib.RSI(c, timeperiod=p) / 50.0 - 1  # [-1, 1]

    # --- 7. MACD (3) ---
    macd, signal, hist = talib.MACD(c)
    feats['macd'] = macd / c
    feats['macd_signal'] = signal / c
    feats['macd_hist'] = hist / c

    # --- 8. Bollinger Bands (3) ---
    upper, middle, lower = talib.BBANDS(c, timeperiod=20)
    bw = upper - lower
    feats['bb_pctb'] = np.where(bw > 0, (c - lower) / bw, 0.5)
    feats['bb_width'] = np.where(middle > 0, bw / middle, 0)
    feats['bb_pos'] = np.where(middle > 0, (c - middle) / middle, 0)

    # --- 9. ATR relative (3) ---
    for p in [7, 14, 21]:
        atr = talib.ATR(h, l, c, timeperiod=p)
        feats[f'atr_{p}_rel'] = atr / c

    # --- 10. Stochastic (2) ---
    slowk, slowd = talib.STOCH(h, l, c)
    feats['stoch_k'] = slowk / 50.0 - 1
    feats['stoch_d'] = slowd / 50.0 - 1

    # --- 11. ADX / DI (3) ---
    feats['adx'] = talib.ADX(h, l, c, timeperiod=14) / 50.0 - 1
    feats['plus_di'] = talib.PLUS_DI(h, l, c, timeperiod=14) / 50.0 - 1
    feats['minus_di'] = talib.MINUS_DI(h, l, c, timeperiod=14) / 50.0 - 1

    # --- 12. CCI (2) ---
    feats['cci_14'] = talib.CCI(h, l, c, timeperiod=14) / 200.0
    feats['cci_20'] = talib.CCI(h, l, c, timeperiod=20) / 200.0

    # --- 13. Williams %R (1) ---
    feats['willr'] = talib.WILLR(h, l, c, timeperiod=14) / 50.0 + 1  # [-1, 1]

    # --- 14. MFI (1) ---
    feats['mfi'] = talib.MFI(h, l, c, v, timeperiod=14) / 50.0 - 1

    # --- 15. OBV (1) ---
    obv = talib.OBV(c, v)
    obv_s = pd.Series(obv)
    feats['obv_roc_5'] = obv_s.pct_change(5).values

    # --- 16. Volume features (5) ---
    vs = pd.Series(v)
    for w in [5, 10, 20]:
        sma_v = talib.SMA(v, timeperiod=w)
        feats[f'vol_sma_{w}_ratio'] = np.where(sma_v > 0, v / sma_v, 1.0)
    feats['vol_change_1'] = vs.pct_change(1).values
    feats['vol_change_5'] = vs.pct_change(5).values

    # --- 17. Historical volatility (4) ---
    log_ret_arr = np.log(c / np.roll(c, 1))
    log_ret_arr[0] = 0
    log_ret = pd.Series(log_ret_arr)
    for w in [5, 10, 20, 60]:
        feats[f'hvol_{w}'] = (log_ret.rolling(w).std() * np.sqrt(252 * 24)).values

    # --- 18. Momentum (3) ---
    for p in [5, 10, 20]:
        feats[f'mom_{p}'] = talib.MOM(c, timeperiod=p) / c

    # --- 19. ROC (3) ---
    for p in [5, 10, 20]:
        feats[f'roc_{p}'] = talib.ROC(c, timeperiod=p) / 100.0

    # --- 20. Aroon (2) ---
    aroon_down, aroon_up = talib.AROON(h, l, timeperiod=14)
    feats['aroon_up'] = aroon_up / 50.0 - 1
    feats['aroon_down'] = aroon_down / 50.0 - 1

    # --- 21. TRIX (1) ---
    feats['trix'] = talib.TRIX(c, timeperiod=15)

    # --- 22. PPO (1) ---
    feats['ppo'] = talib.PPO(c) / 100.0

    # --- 23. Z-score of close (3) ---
    for w in [10, 20, 50]:
        roll_mean = cs.rolling(w).mean()
        roll_std = cs.rolling(w).std()
        feats[f'zscore_{w}'] = np.where(roll_std > 0, (c - roll_mean) / roll_std, 0)

    # --- 24. Rolling skew/kurtosis (4) ---
    for w in [10, 20]:
        feats[f'skew_{w}'] = log_ret.rolling(w).skew().values
        feats[f'kurt_{w}'] = log_ret.rolling(w).kurt().values

    # --- 25. Ichimoku components (3) ---
    tenkan = (talib.MAX(h, 9) + talib.MIN(l, 9)) / 2
    kijun = (talib.MAX(h, 26) + talib.MIN(l, 26)) / 2
    feats['ichimoku_tenkan_rel'] = np.where(c > 0, tenkan / c - 1, 0)
    feats['ichimoku_kijun_rel'] = np.where(c > 0, kijun / c - 1, 0)
    feats['ichimoku_tk_cross'] = np.where(kijun > 0, tenkan / kijun - 1, 0)

    # Add prefix
    result = pd.DataFrame(feats)
    result.columns = [f'{prefix}{col}' for col in result.columns]
    return result


def compute_cross_asset_features(dfs: dict[str, pd.DataFrame], window: int = 20) -> pd.DataFrame:
    """计算跨品种特征（相关性、价差、波动率比等）。

    Args:
        dfs: {pair_name: DataFrame} 各品种原始 OHLCV
        window: 滚动窗口

    Returns:
        DataFrame with cross-asset features
    """
    pairs = list(dfs.keys())
    feats = {}

    # Close returns for correlation
    returns = {}
    for pair in pairs:
        cs = pd.Series(dfs[pair]['close'].values)
        returns[pair] = cs.pct_change()

    # Pairwise rolling correlation
    for i in range(len(pairs)):
        for j in range(i + 1, len(pairs)):
            p1, p2 = pairs[i], pairs[j]
            for w in [10, 20, 60]:
                corr = returns[p1].rolling(w).corr(returns[p2])
                feats[f'corr_{p1}_{p2}_{w}'] = corr.values

    # Return spread
    for i in range(len(pairs)):
        for j in range(i + 1, len(pairs)):
            p1, p2 = pairs[i], pairs[j]
            feats[f'ret_spread_{p1}_{p2}'] = (returns[p1] - returns[p2]).values

    # Volume ratio
    for i in range(len(pairs)):
        for j in range(i + 1, len(pairs)):
            p1, p2 = pairs[i], pairs[j]
            v1 = pd.Series(dfs[p1]['volume'].values)
            v2 = pd.Series(dfs[p2]['volume'].values)
            ratio = v1 / (v1 + v2)
            feats[f'vol_ratio_{p1}_{p2}'] = ratio.rolling(window).mean().values

    return pd.DataFrame(feats)


def compute_time_features(dates: pd.DatetimeIndex) -> pd.DataFrame:
    """时间编码特征（正弦/余弦周期编码）。"""
    feats = {}
    hours = dates.hour
    dow = dates.dayofweek
    month = dates.month

    feats['hour_sin'] = np.sin(2 * np.pi * hours / 24)
    feats['hour_cos'] = np.cos(2 * np.pi * hours / 24)
    feats['dow_sin'] = np.sin(2 * np.pi * dow / 7)
    feats['dow_cos'] = np.cos(2 * np.pi * dow / 7)
    feats['month_sin'] = np.sin(2 * np.pi * (month - 1) / 12)
    feats['month_cos'] = np.cos(2 * np.pi * (month - 1) / 12)

    return pd.DataFrame(feats, index=dates)


# ====== BC 规则策略标签 ======

def generate_rule_targets(
    df: pd.DataFrame,
    max_leverage: float = 10.0,
    sl_min: float = 0.005,
    sl_max: float = 0.10,
    tp_min: float = 0.01,
    tp_max: float = 0.20,
) -> np.ndarray:
    """为单品种生成规则策略 BC 标签（4 维决策向量）。

    规则:
      position: EMA(10)/EMA(50) 交叉 + RSI 过滤 + ADX 强度缩放
      leverage: ADX → [1, max_leverage] 映射
      stop_loss: 2×ATR(14)/close, clamp [sl_min, sl_max]
      take_profit: 3×ATR(14)/close, clamp [tp_min, tp_max]

    Args:
        df: 原始 OHLCV DataFrame
        max_leverage: 杠杆上限

    Returns:
        targets: (n_steps, 4) float32
    """
    c = df['close'].values.astype(np.float64)
    h = df['high'].values.astype(np.float64)
    l = df['low'].values.astype(np.float64)

    ema_fast = talib.EMA(c, timeperiod=10)
    ema_slow = talib.EMA(c, timeperiod=50)
    rsi = talib.RSI(c, timeperiod=14)
    adx = talib.ADX(h, l, c, timeperiod=14)
    atr = talib.ATR(h, l, c, timeperiod=14)

    n = len(c)
    targets = np.zeros((n, 4), dtype=np.float32)

    for i in range(n):
        # Skip if any indicator is NaN
        if np.isnan(ema_fast[i]) or np.isnan(ema_slow[i]) or \
           np.isnan(rsi[i]) or np.isnan(adx[i]) or np.isnan(atr[i]):
            continue

        # --- Position: direction + magnitude ---
        # Base direction from MA crossover
        ma_diff = (ema_fast[i] - ema_slow[i]) / c[i]
        direction = np.sign(ma_diff)

        # RSI override: extreme RSI → reduce/flip
        if rsi[i] > 75:
            direction = min(direction, -0.3)  # overbought → lean short
        elif rsi[i] < 25:
            direction = max(direction, 0.3)   # oversold → lean long

        # Magnitude from ADX (trend strength) and MA separation
        strength = np.clip(adx[i] / 50.0, 0.1, 1.0)
        ma_magnitude = np.clip(abs(ma_diff) * 100, 0.1, 1.0)
        position = np.clip(direction * strength * ma_magnitude, -1.0, 1.0)

        # --- Leverage: based on ADX ---
        # ADX < 15 → 1x, ADX > 40 → max_leverage
        lev_raw = 1.0 + (max_leverage - 1.0) * np.clip((adx[i] - 15) / 25, 0, 1)
        leverage = np.clip(lev_raw, 1.0, max_leverage)

        # --- Stop loss: 2 × ATR / close ---
        sl_raw = 2.0 * atr[i] / c[i]
        stop_loss = np.clip(sl_raw, sl_min, sl_max)

        # --- Take profit: 3 × ATR / close (1.5:1 R/R) ---
        tp_raw = 3.0 * atr[i] / c[i]
        take_profit = np.clip(tp_raw, tp_min, tp_max)

        targets[i] = [position, leverage, stop_loss, take_profit]

    return targets


# ====== Main ======

def main():
    parser = argparse.ArgumentParser(description='金融数据预处理: OHLCV → features.npy + targets.npy')
    parser.add_argument('--raw_dir', type=str, default='data/finance/raw',
                        help='Freqtrade 下载的原始数据目录')
    parser.add_argument('--out_dir', type=str, default='data/finance',
                        help='输出目录')
    parser.add_argument('--pairs', nargs='+', default=['BTC_USDT', 'ETH_USDT', 'SOL_USDT'],
                        help='交易对（文件名格式）')
    parser.add_argument('--timeframe', type=str, default='1h')
    parser.add_argument('--max_leverage', type=float, default=10.0)
    parser.add_argument('--sl_min', type=float, default=0.005)
    parser.add_argument('--sl_max', type=float, default=0.10)
    parser.add_argument('--tp_min', type=float, default=0.01)
    parser.add_argument('--tp_max', type=float, default=0.20)
    parser.add_argument('--target_pair', type=str, default=None,
                        help='标签品种（默认第一个）。只对该品种生成交易决策标签')
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # --- 加载原始数据 ---
    dfs = {}
    for pair in args.pairs:
        path = os.path.join(args.raw_dir, f'{pair}-{args.timeframe}.feather')
        df = pd.read_feather(path)
        df = df.sort_values('date').reset_index(drop=True)
        print(f'Loaded {pair}: {len(df)} rows, {df["date"].min()} → {df["date"].max()}')
        dfs[pair] = df

    # 对齐时间轴（取交集）— 统一去除时区再比较
    for pair in args.pairs:
        dfs[pair]['date'] = dfs[pair]['date'].dt.tz_localize(None)
    date_sets = [set(df['date']) for df in dfs.values()]
    common_dates = sorted(set.intersection(*date_sets))
    print(f'\nCommon dates: {len(common_dates)} (from {common_dates[0]} to {common_dates[-1]})')

    aligned = {}
    for pair in args.pairs:
        mask = dfs[pair]['date'].isin(common_dates)
        aligned[pair] = dfs[pair][mask].reset_index(drop=True)

    # --- 计算特征 ---
    print('\nComputing technical indicators...')
    feature_dfs = []

    # Per-asset features
    short_names = {p: p.split('_')[0].lower() for p in args.pairs}
    for pair in args.pairs:
        prefix = f'{short_names[pair]}_'
        feat_df = compute_single_asset_features(aligned[pair], prefix)
        feature_dfs.append(feat_df)
        print(f'  {pair}: {feat_df.shape[1]} features')

    # Cross-asset features
    cross_df = compute_cross_asset_features(
        {short_names[p]: aligned[p] for p in args.pairs}
    )
    feature_dfs.append(cross_df)
    print(f'  Cross-asset: {cross_df.shape[1]} features')

    # Time features
    dates = pd.DatetimeIndex(aligned[args.pairs[0]]['date'])
    time_df = compute_time_features(dates)
    time_df = time_df.reset_index(drop=True)
    feature_dfs.append(time_df)
    print(f'  Time: {time_df.shape[1]} features')

    # Concatenate all features
    all_features = pd.concat(feature_dfs, axis=1)
    print(f'\nTotal features: {all_features.shape[1]}')

    # --- 生成 BC 标签 ---
    target_pair = args.target_pair or args.pairs[0]
    print(f'\nGenerating rule-based targets for {target_pair}...')
    targets = generate_rule_targets(
        aligned[target_pair],
        max_leverage=args.max_leverage,
        sl_min=args.sl_min, sl_max=args.sl_max,
        tp_min=args.tp_min, tp_max=args.tp_max,
    )

    # --- 去除 NaN 行 ---
    # Features 中有 NaN（指标 warm-up 期间）
    feature_arr = all_features.values.astype(np.float32)
    valid_mask = ~np.any(np.isnan(feature_arr), axis=1)
    # Targets 中全零行也排除（指标 warm-up 期间）
    valid_mask &= ~np.all(targets == 0, axis=1)

    feature_arr = feature_arr[valid_mask]
    targets = targets[valid_mask]

    # Replace any remaining inf/nan with 0
    feature_arr = np.nan_to_num(feature_arr, nan=0.0, posinf=0.0, neginf=0.0)

    print(f'\nAfter removing NaN/warm-up rows:')
    print(f'  Features: {feature_arr.shape}  (total_steps, n_features)')
    print(f'  Targets:  {targets.shape}  (total_steps, 4)')

    # --- 统计摘要 ---
    print(f'\nTarget statistics:')
    labels = ['position', 'leverage', 'stop_loss', 'take_profit']
    for i, name in enumerate(labels):
        col = targets[:, i]
        print(f'  {name:12s}: mean={col.mean():.4f}, std={col.std():.4f}, '
              f'min={col.min():.4f}, max={col.max():.4f}')

    print(f'\nFeature statistics (first 10):')
    for i in range(min(10, feature_arr.shape[1])):
        col = feature_arr[:, i]
        name = all_features.columns[i] if i < len(all_features.columns) else f'feat_{i}'
        print(f'  {name:30s}: mean={col.mean():.4f}, std={col.std():.4f}')

    # --- 保存 ---
    feat_path = os.path.join(args.out_dir, 'features.npy')
    tgt_path = os.path.join(args.out_dir, 'targets.npy')
    np.save(feat_path, feature_arr)
    np.save(tgt_path, targets)
    print(f'\nSaved: {feat_path} ({feature_arr.nbytes / 1e6:.1f} MB)')
    print(f'Saved: {tgt_path} ({targets.nbytes / 1e6:.1f} MB)')

    # Save feature names for reference
    names_path = os.path.join(args.out_dir, 'feature_names.txt')
    valid_cols = all_features.columns[~np.any(np.isnan(all_features.values), axis=0)]
    with open(names_path, 'w') as f:
        for col in all_features.columns:
            f.write(col + '\n')
    print(f'Saved: {names_path} ({len(all_features.columns)} names)')


if __name__ == '__main__':
    main()
