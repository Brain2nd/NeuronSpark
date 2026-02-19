# NeuronSpark — SNN Hidden State Space Language Model

一个**完全基于脉冲神经网络 (SNN)** 构建的语言模型，灵感来自 Mamba 的选择性状态空间模型 (SSM)。隐层神经元的动态参数 β(t), α(t), V_th(t) 作为输入依赖的调制信号，实现选择性信息过滤。**整个网络是纯 SNN —— 不包含任何标准 ANN 组件**。

**当前版本: v7.5** (连续残差流 + Triton Fused PLIF Kernel + Surrogate Gradient)

## 致谢与参考

本项目的训练基础设施（数据处理、Tokenizer 训练、预训练/SFT 流程）**严格参照 [happy-llm](https://github.com/datawhalechina/happy-llm) 教学项目**（Datawhale 开源社区）。happy-llm 是一个从零搭建大模型的教程，我们在其基础上将模型架构替换为 SNN，训练流程保持对齐。

具体借鉴关系见下方 [与 happy-llm 的关系](#与-happy-llm-的关系) 章节。

## 核心架构

```
token → Embedding(D) → encode_proj → sigmoid → K-bit 二进制编码 → K 帧 spike
  → L × SNNDecoderLayer:
      连续 h → PLIFNode → spike → SNNBlock → out_proj → 残差连接
      连续 h → PLIFNode → spike → SNNFFN  → out_proj → 残差连接
  → K 帧加权求和 → decode_proj → LateralInhibition → Embedding^T (tied) → logits
```

### 架构要点

- **基础神经元**: PLIF (Parametric LIF)，动态 β(t), α(t), V_th(t) 由调制网络生成
- **连续残差流 (v7.5)**: 层间传递连续值 h，仅 SNN 子层内部使用 spike，解决 20 层梯度消失
- **6 条并行输入路径**: W_in, W_β, W_α, W_th, W_gate, W_skip
- **并行化**: Triton Fused PLIF Kernel，单 kernel 完成扫描 + spike + 软重置 + 替代梯度
- **训练**: Surrogate Gradient + 标准反向传播 (v7.1+，替代 v5 的 SPSA 零阶优化)
- **框架**: SpikingJelly (conda env `SNN`)

## 与 happy-llm 的关系

本项目模型架构为原创 SNN 设计，但**训练基础设施严格对齐 happy-llm 教程**（[第五章：动手搭建大模型](https://github.com/datawhalechina/happy-llm/blob/main/docs/chapter5/第五章%20动手搭建大模型.md)），确保训练流程经过验证，集中精力在 SNN 架构创新上。

### 直接借鉴的部分（训练基础设施）

| 组件 | happy-llm 源文件 | NeuronSpark 对应文件 | 对齐程度 |
|------|-----------------|---------------------|---------|
| **数据预处理** | `deal_dataset.py` | `scripts/deal_dataset.py` | 完全对齐：512 字符切块 + JSONL 输出 |
| **Tokenizer 训练** | `train_tokenizer.py` | `scripts/train_tokenizer.py` | 完全对齐：BPE, vocab=6144, NFKC 正则化, ByteLevel, chat_template 与 Qwen2.5 一致 |
| **数据集加载** | `dataset.py` | `dataset.py` | 完全对齐：PretrainDataset (byte-offset JSONL 随机访问, bos 前缀, loss_mask), SFTDataset (chat_template + assistant-only loss_mask) |
| **预训练循环** | `ddp_pretrain.py` | `train.py` | 高度对齐：Adam 优化器, Warmup+Cosine LR, GradScaler+autocast 混合精度, 梯度累积, 梯度裁剪, checkpoint 保存 |
| **SFT 训练** | `ddp_sft.py` | `train.py` (计划复用) | 高度对齐：加载预训练权重 + SFTDataset 替换 |
| **推理生成** | `model_sample.py` | (计划实现) | — |
| **训练数据** | Seq-Monkey 10B tokens | 同（29M 样本 JSONL） | 完全相同 |
| **SFT 数据** | BelleGroup 350 万条 | 同 (计划使用) | 完全相同 |

### NeuronSpark 独创部分（SNN 架构）

| 组件 | 说明 |
|------|------|
| `model.py` | SNNLanguageModel：SNN 编码/解码 pipeline，K-bit 二进制编码 (STE)，连续残差流 |
| `atomic_ops/selective_plif.py` | SelectivePLIFNode：动态参数 PLIF 神经元 |
| `atomic_ops/plif_node.py` | PLIFNode：D 维固定参数 PLIF 神经元 |
| `atomic_ops/snn_block.py` | SNNBlock：SNN 注意力等价层（6 条并行路径 + 门控 + 跳跃连接） |
| `atomic_ops/snn_ffn.py` | SNNFFN：SNN 前馈网络（SwiGLU 风格三分支 spike 门控） |
| `atomic_ops/snn_decoder_layer.py` | SNNDecoderLayer：Block + FFN + 残差连接 |
| `atomic_ops/parallel_scan.py` | Triton Fused PLIF Kernel (v7.3+)，Row-param Kernel (v7.4+) |
| `atomic_ops/lateral_inhibition.py` | Triton 实现的侧抑制归一化 (divisive normalization) |
| `docs/SNN_SELECTIVE_STATE_SPACE.md` | 完整架构设计文档（2200+ 行，v1→v7.5 全部演进） |

### 与 happy-llm 的配置对比

| 参数 | happy-llm (LLaMA2) | NeuronSpark (SNN) |
|------|-------------------|-------------------|
| 架构 | Transformer (Attention + MLP) | SNN (SNNBlock + SNNFFN) |
| 参数量 | 215M | 643M |
| 隐藏维度 | 1024 | 768 |
| 层数 | 18 | 20 |
| 词表 | 6144 | 6144 |
| 序列长度 | 512 | 512 |
| SNN 时间步 K | — | 16 |
| 优化器 | Adam | Adam |
| 学习率 | 2e-4 | 2e-4 |
| LR 调度 | Warmup + Cosine | Warmup + Cosine |
| 精度 | bfloat16 | bfloat16 |
| 预训练数据 | Seq-Monkey | Seq-Monkey |
| SFT 数据 | BelleGroup 3.5M | BelleGroup 3.5M (计划) |
| 硬件 | 8 × RTX 4090 | 1 × NVIDIA GB10 (DGX Spark, 128GB 统一内存) |

## 快速开始

### 环境准备

```bash
# 1. 创建 conda 环境
conda create -n SNN python=3.10
conda activate SNN

# 2. 安装依赖
pip install torch torchvision torchaudio
pip install spikingjelly transformers tokenizers pandas numpy tqdm

# 3. (DGX Spark / Blackwell GPU) Triton ptxas 配置
export TRITON_PTXAS_PATH=/usr/local/cuda-13.0/bin/ptxas
```

### 数据准备

参照 [happy-llm 第五章 5.3.1 节](https://github.com/datawhalechina/happy-llm/blob/main/docs/chapter5/第五章%20动手搭建大模型.md#531-数据下载) 下载数据集：

```bash
# 1. 下载 Seq-Monkey 预训练数据集
mkdir -p data/seq-monkey
# 从 ModelScope 下载:
# modelscope download --dataset ddzhu123/seq-monkey mobvoi_seq_monkey_general_open_corpus.jsonl.tar.bz2 --local_dir data/seq-monkey/
# 解压:
# tar -xvf data/seq-monkey/mobvoi_seq_monkey_general_open_corpus.jsonl.tar.bz2 -C data/seq-monkey/

# 2. 预处理：将长文本按 512 字符切分
python scripts/deal_dataset.py
# 输出: data/seq-monkey/seq_monkey_datawhale.jsonl (约 29M 样本)

# 3. (可选) 下载 SFT 数据集
# huggingface-cli download --repo-type dataset --resume-download BelleGroup/train_3.5M_CN --local-dir data/BelleGroup/
```

### 训练 Tokenizer

参照 [happy-llm 第五章 5.3.2 节](https://github.com/datawhalechina/happy-llm/blob/main/docs/chapter5/第五章%20动手搭建大模型.md#532-训练-tokenizer)：

```bash
conda activate SNN

# 训练 6144 词表的 BPE tokenizer
python scripts/train_tokenizer.py \
    --data_path data/seq-monkey/seq_monkey_datawhale.jsonl \
    --save_dir tokenizer_snn \
    --vocab_size 6144

# 输出: tokenizer_snn/ 目录 (tokenizer.json, tokenizer_config.json, special_tokens_map.json)
```

> 项目已包含训练好的 tokenizer (`tokenizer_snn/`)，可跳过此步直接使用。

### 预训练

参照 [happy-llm 第五章 5.3.4 节](https://github.com/datawhalechina/happy-llm/blob/main/docs/chapter5/第五章%20动手搭建大模型.md#534-预训练)：

```bash
conda activate SNN

# 启动预训练 (DGX Spark 推荐配置)
TRITON_PTXAS_PATH=/usr/local/cuda-13.0/bin/ptxas \
python -u train.py \
    --D 768 --D_ff 2304 --num_layers 20 \
    --batch_size 8 --accumulation_steps 8 \
    --warmup_iters 1000 --log_interval 10 \
    --data_path data/seq-monkey/seq_monkey_datawhale.jsonl

# 使用 tmux 后台运行 (推荐)
tmux new-session -d -s snn_train
tmux send-keys -t snn_train 'PYTHONUNBUFFERED=1 TRITON_PTXAS_PATH=/usr/local/cuda-13.0/bin/ptxas conda run --no-capture-output -n SNN python -u train.py --D 768 --D_ff 2304 --batch_size 8 --accumulation_steps 8 --warmup_iters 1000 --log_interval 10 2>&1 | tee train_v75.log' Enter

# 查看训练日志
tail -f train_v75.log

# 断续训练
python train.py --resume checkpoints/latest.pt
```

#### 训练参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--D` | 768 | 隐藏维度 |
| `--N` | 8 | 神经元分组数（状态扩展因子） |
| `--K` | 16 | 每 token SNN 时间步数 / 二进制编码位数 |
| `--num_layers` | 20 | SNN 解码层数 |
| `--D_ff` | 2304 | FFN 中间维度 (通常 3×D) |
| `--batch_size` | 8 | 每 GPU micro-batch 大小 |
| `--accumulation_steps` | 8 | 梯度累积步数 (effective batch = batch_size × accumulation_steps) |
| `--learning_rate` | 2e-4 | 峰值学习率 |
| `--warmup_iters` | 1000 | Warmup 步数 (micro-batch 计) |
| `--epochs` | 1 | 训练轮数 |
| `--grad_clip` | 1.0 | 梯度裁剪阈值 |
| `--log_interval` | 10 | 日志打印间隔 (micro-batch 计) |
| `--save_interval` | 1000 | Checkpoint 保存间隔 |
| `--data_path` | `data/seq-monkey/seq_monkey_datawhale.jsonl` | 训练数据路径 |
| `--resume` | — | 断续训练 checkpoint 路径 |

#### 显存与 batch_size 参考

| batch_size | 约显存占用 | 适用硬件 |
|:----------:|:---------:|:--------:|
| 2 | ~29 GB | RTX 3090/4090 (24GB) + 梯度检查点 |
| 4 | ~55 GB | A100 40GB / A6000 48GB |
| 8 | ~103 GB | DGX Spark (128GB 统一内存) / A100 80GB |

### SFT 微调 (计划中)

参照 [happy-llm 第五章 5.3.5 节](https://github.com/datawhalechina/happy-llm/blob/main/docs/chapter5/第五章%20动手搭建大模型.md#535-sft-训练)：

```bash
# SFT 训练 (加载预训练权重，使用 SFTDataset)
python train.py \
    --resume checkpoints/pretrain_best.pt \
    --data_path data/BelleGroup/BelleGroup_sft.jsonl \
    --sft  # (待实现)
```

## 项目结构

```
NeuronSpark/
├── model.py                        # 顶层模型 SNNLanguageModel (v7.5 连续残差流)
├── train.py                        # 训练脚本 (对齐 happy-llm ddp_pretrain.py)
├── dataset.py                      # 数据集加载 (对齐 happy-llm dataset.py)
├── atomic_ops/                     # SNN 核心算子
│   ├── __init__.py
│   ├── selective_plif.py           # SelectivePLIFNode: 动态参数 PLIF 神经元
│   ├── plif_node.py                # PLIFNode: D 维固定参数 PLIF 神经元
│   ├── snn_block.py                # SNNBlock: SNN 注意力等价层 (6 路并行 + 门控)
│   ├── snn_ffn.py                  # SNNFFN: SNN 前馈网络 (SwiGLU 风格三分支)
│   ├── snn_decoder_layer.py        # SNNDecoderLayer: Block + FFN + 残差
│   ├── parallel_scan.py            # Triton Fused PLIF Kernel + Row-param Kernel
│   └── lateral_inhibition.py       # Triton 侧抑制归一化 (输出层使用)
├── scripts/                        # 数据处理脚本 (对齐 happy-llm)
│   ├── deal_dataset.py             # Seq-Monkey JSONL 预处理 (512 字符切块)
│   ├── train_tokenizer.py          # BPE Tokenizer 训练 (6144 词表)
│   └── prepare_data.py             # SkyPile-150B 数据集处理 (二进制 token 格式)
├── docs/                           # 设计文档
│   ├── SNN_SELECTIVE_STATE_SPACE.md  # 主设计文档 (2200+ 行, v1→v7.5 全部演进)
│   ├── PARALLEL_SCAN_OPTIMIZATION.md # Triton Kernel 优化文档
│   ├── OPEN_ISSUES.md              # 已解决的设计问题记录
│   ├── Q5.md                       # 技术验证方法论
│   └── SNN_DIFFUSION_PLAN.md       # 未来方向: SNN 扩散模型
├── exp/                            # 实验脚本 (验证 + 基准测试)
│   ├── verify_*.py                 # 各版本正确性验证 (梯度、融合算子、端到端)
│   ├── bench_*.py                  # 性能基准测试 (Triton kernel, 编译, 层级)
│   └── measure_sparsity.py         # SNN 脉冲稀疏度测量
├── notebooks/                      # 实验 Notebook
│   ├── linear_layer_analysis.ipynb # SpikingJelly 线性层分析
│   └── neuron_comparison.ipynb     # 7 种神经元模型对比 → PLIF 最优
├── tokenizer_snn/                  # 已训练的 BPE tokenizer (6144 词表)
├── data/                           # 训练数据 (不纳入 git)
│   └── seq-monkey/                 # Seq-Monkey 预训练数据 (29M 样本 JSONL)
├── checkpoints/                    # 模型 checkpoint (不纳入 git)
├── archive/                        # 历史版本归档
│   ├── v5_pre_ffn/                 # v5 代码快照 (SNNFFN 之前)
│   ├── v6_pre_parallel/            # v6 代码快照 (parallel scan 之前)
│   ├── atomic_ops_v75_pre/         # v7.5 纯净备份
│   ├── logs/                       # 历史训练日志
│   └── checkpoints/                # 历史 checkpoint
└── train_v75.log                   # 当前训练日志
```

## 版本演进

| 版本 | 关键变更 |
|------|---------|
| v5 | SelectivePLIFNode + SNNBlock，SPSA 零阶优化 |
| v6 | 新增 SNNFFN（三分支门控 FFN），SNNDecoderLayer 组合层 |
| v7 | 移除 W^(V)，引入 Hillis-Steele parallel scan，K=16 二进制编码 |
| v7.1 | SPSA → Surrogate Gradient + 反向传播，K-bit 并行编码，梯度检查点 |
| v7.2 | Triton fused linear recurrence kernel 替换 Hillis-Steele (CUDA) |
| v7.3 | Fused PLIF Kernel：单 kernel 完成 scan + spike + soft reset + surrogate |
| v7.4 | Row-param PLIF Kernel + torch.compile fused modulation，1.74× 单层加速 |
| **v7.5** | **连续残差流**：层间传递连续值，层内归一化移除，梯度流 L0/L19=0.41 |

## 当前训练配置 (v7.5)

| 参数 | 值 |
|------|-----|
| 参数量 | 643M (619M base + 24M 残差流) |
| D (隐藏维度) | 768 |
| N (神经元分组) | 8 |
| K (SNN 步数/编码位数) | 16 |
| Layers | 20 |
| D_ff (FFN 维度) | 2304 |
| Vocab | 6144 |
| 序列长度 | 512 |
| Batch size | 8 × accum 8 = 64 effective |
| 学习率 | 2e-4 (warmup 1000 → cosine → 2e-5) |
| 精度 | bfloat16 |
| 数据集 | Seq-Monkey (29M 样本) |
| 硬件 | NVIDIA GB10 (DGX Spark, 128GB 统一内存) |
| TPS | ~95 tokens/sec |
| 显存占用 | ~103 GB |

## 训练流程总览

参照 happy-llm 教程的完整训练流程：

```
Step 1: 数据准备
  下载 Seq-Monkey 数据集 → deal_dataset.py 预处理 → 29M 样本 JSONL

Step 2: Tokenizer 训练
  train_tokenizer.py → 6144 词表 BPE tokenizer

Step 3: 预训练 (当前阶段)
  train.py + PretrainDataset → next token prediction
  目标: loss 收敛到 3~4，模型能续写通顺中文

Step 4: SFT 微调 (计划中)
  train.py + SFTDataset → 加载预训练权重，BelleGroup 350 万条指令数据
  目标: 模型能进行基本对话

Step 5: 推理生成 (计划中)
  model_sample.py → 文本续写 / 对话生成
```

## 环境

```bash
conda activate SNN
# 核心依赖: PyTorch, SpikingJelly, Triton, transformers, pandas, numpy
# 硬件: NVIDIA GPU (推荐 Blackwell/Ampere 架构, 40GB+ 显存)
# DGX Spark 需配置: export TRITON_PTXAS_PATH=/usr/local/cuda-13.0/bin/ptxas
```

## License

Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0)
