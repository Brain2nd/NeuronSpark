# SNN 文生图模型计划（待启动）

> 当前方案，持续迭代。优先级低于自回归语言模型。

## 目标

构建全 SNN 的文生图 diffusion 模型，无 attention，无 ANN 组件。

## 参考模型

PixArt-α (600M, DiT): https://github.com/PixArt-alpha/PixArt-alpha

## 三阶段路线

### 阶段 1: SNN 去噪骨干（核心）

- 替代 DiT 的去噪网络
- VAE、Text Encoder 暂用 ANN 占位（冻结），后续替换
- 训练数据: 图文对（COCO 2017 验证 → LAION-Aesthetics 正式）
- 架构: Patchify → N × (SNNBlock + SNNFFN) → Unpatchify
- 条件注入: 文本/时间步信息与 patch token 拼接后作为 SNNBlock 输入，通过 W_β, W_α, W_th, W_gate 调制神经元动力学（非 cross-attention）
- 参数规模: ~600M (D=1152, N=8, K=16, 28层, D_ff=4608)
- 训练方式: 蒸馏 (PixArt-α teacher) + 标准 diffusion loss 混合
- Loss: λ₁·||ε_student - ε_teacher||² + λ₂·||ε_student - ε_true||²

### 阶段 2: SNN 图像自编码器

- 替代 VAE
- 训练数据: 纯图像（ImageNet / OpenImages）
- 架构: 图像 patch → spike 编码 → SNN Encoder（降维）→ latent → SNN Decoder（升维）→ spike 解码 → 图像
- Loss: L_recon + λ·L_KL

### 阶段 3: SNN 文本编码器

- 替代 CLIP / T5
- 训练数据: 图文对（对比学习）
- 架构: 与自回归语言模型结构相同，但训练目标为对比学习（文本-图像对齐），非 next token prediction
- 注意: 与 NeuronSpark 语言模型是独立项目，共享 atomic_ops 代码但训练数据/目标完全不同

## 与自回归语言模型的关系

| | 自回归语言模型（当前） | SNN Diffusion（待启动） |
|---|---|---|
| 任务 | Next token prediction | 噪声预测 ε_θ |
| 数据 | 纯文本 (seq-monkey) | 图文对 (COCO/LAION) |
| 输入 | Token 序列 | Noisy image patches + text + timestep |
| 输出 | 下一个 token logits | 预测噪声 ε |
| 共享代码 | atomic_ops/ (SNNBlock, SNNFFN, parallel_scan) | 同左 |
| 独立代码 | model.py, train.py, dataset.py | 需新建: diffusion_model.py, diffusion_train.py, image_dataset.py |

## 前置条件

- 自回归语言模型训练收敛，验证 SNN 架构有效
- atomic_ops/ 经过充分测试和优化

## 数学基础

见 exp/superposition_verify.py — 非线性叠加分析实验，验证了:
1. SNN 的"先累加再阈值"范式天然计算 f(Σx_k) = f(x)
2. 门控机制提供二次非线性能力
3. K-bit binary coding 比 rate coding 效率高 16×
4. Boolean 多线性展开中 0-1 阶占 >99% 能量

## 关键技术风险

1. 图像 patch 序列长度 1024，比语言模型的 512 更长，显存/速度待评估
2. Diffusion 需要反复推理（~20-50步采样），SNN 的 K 步 parallel scan 在每步去噪中都要执行
3. 文本条件通过参数调制注入的效果是否足够（vs cross-attention），需实验验证
