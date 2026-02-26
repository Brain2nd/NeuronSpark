"""
金融数据集: 加载预计算特征 + 交易决策目标，滑动窗口切割时间序列。

数据格式:
  features.npy: (total_steps, n_features) float32 — 预计算的技术指标特征
  targets.npy:  (total_steps, n_assets * 4) float32 — 交易决策目标向量
    每资产 4 维: [position, leverage, stop_loss, take_profit]
      position:    [-1, +1]     方向+仓位比例
      leverage:    [1, max_lev] 杠杆倍率
      stop_loss:   [sl_min, sl_max] 止损距离（相对当前价的百分比）
      take_profit: [tp_min, tp_max] 止盈距离（相对当前价的百分比）

    单品种 (n_assets=1): shape = (total_steps, 4)
    三品种 (n_assets=3): shape = (total_steps, 12)

  BC 标签来源: 规则策略自动生成（MA 交叉、RSI 阈值、ATR 波动率 → position/leverage/sl/tp）
  SFT 标签来源: 回测筛选最优决策参数

滑动窗口:
  seq_len=2048 个连续时间步作为一个样本（≈85天），步长 stride（默认 seq_len//2，50% 重叠）。
"""

import numpy as np
import torch
from torch.utils.data import Dataset


class FinanceDataset(Dataset):
    """滑动窗口金融时间序列数据集。

    Args:
        features_path: 特征文件路径 (.npy, shape=(total_steps, n_features))
        targets_path: 目标文件路径 (.npy, shape=(total_steps, n_assets * 4))
        seq_len: 每个样本的时间步长度
        stride: 滑动窗口步长（默认 seq_len//2）
        n_assets: 资产数（用于 reshape targets）
    """

    def __init__(
        self,
        features_path: str,
        targets_path: str,
        seq_len: int = 2048,
        stride: int = None,
        n_assets: int = 1,
    ):
        super().__init__()
        self.seq_len = seq_len
        self.stride = stride or seq_len // 4
        self.n_assets = n_assets

        self.features = np.load(features_path, mmap_mode='r')  # (total_steps, n_features)
        self.targets = np.load(targets_path, mmap_mode='r')    # (total_steps, n_assets * 4)

        assert len(self.features) == len(self.targets), \
            f"Features ({len(self.features)}) and targets ({len(self.targets)}) length mismatch"
        assert self.targets.shape[1] == n_assets * 4, \
            f"Targets dim ({self.targets.shape[1]}) != n_assets * 4 ({n_assets * 4})"

        total_steps = len(self.features)
        self.n_features = self.features.shape[1]

        self.starts = list(range(0, total_steps - seq_len + 1, self.stride))

    def __len__(self):
        return len(self.starts)

    def __getitem__(self, index: int):
        start = self.starts[index]
        end = start + self.seq_len

        features = torch.from_numpy(
            self.features[start:end].copy()
        ).float()   # (seq_len, n_features)

        targets = torch.from_numpy(
            self.targets[start:end].copy()
        ).float()    # (seq_len, n_assets * 4)

        # reshape → (seq_len, n_assets, 4)
        targets = targets.view(self.seq_len, self.n_assets, 4)

        return features, targets
