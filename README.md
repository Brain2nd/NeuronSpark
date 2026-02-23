# NeuronSpark — SNN Hidden State Space Language Model

一个**完全基于脉冲神经网络 (SNN)** 构建的语言模型，灵感来自 Mamba 的选择性状态空间模型 (SSM)。隐层神经元的动态参数 β(t), α(t), V_th(t) 作为输入依赖的调制信号，实现选择性信息过滤。**整个网络是纯 SNN —— 不包含任何标准 ANN 组件**。

**当前版本: v8.0** (Full V_post + 874M 参数 + 完整预训练 & SFT)

## 模型下载

预训练权重托管在 HuggingFace: [LumenscopeAI/NeuronSpark-V8.0Pre](https://huggingface.co/LumenscopeAI/NeuronSpark-V8.0Pre)

| Checkpoint | 说明 | 大小 |
|------------|------|------|
| `checkpoints/ckpt_step85000.pth` | 预训练 85000 步，loss ~3.6 | 9.8 GB |
| `checkpoints_sft/ckpt_step6500.pth` | SFT 6500 步，基础对话能力 | 9.8 GB |

### 快速下载

```bash
# 方法1: 使用 huggingface_hub (推荐)
pip install huggingface_hub
python -c "
from huggingface_hub import hf_hub_download
repo_id = 'LumenscopeAI/NeuronSpark-V8.0Pre'

# 下载预训练 checkpoint
hf_hub_download(repo_id=repo_id, filename='checkpoints/ckpt_step85000.pth', local_dir='.')

# 下载 SFT checkpoint
hf_hub_download(repo_id=repo_id, filename='checkpoints_sft/ckpt_step6500.pth', local_dir='.')
"

# 方法2: 使用 curl 直接下载
mkdir -p checkpoints checkpoints_sft
curl -L -C - -o checkpoints/ckpt_step85000.pth \
    "https://huggingface.co/LumenscopeAI/NeuronSpark-V8.0Pre/resolve/main/checkpoints/ckpt_step85000.pth"
curl -L -C - -o checkpoints_sft/ckpt_step6500.pth \
    "https://huggingface.co/LumenscopeAI/NeuronSpark-V8.0Pre/resolve/main/checkpoints_sft/ckpt_step6500.pth"
```

### 快速推理

```bash
# SFT 对话模式 (推荐 temperature=0.1~0.3)
python generate_sample.py \
    --checkpoint checkpoints_sft/ckpt_step6500.pth \
    --mode sft \
    --prompt "中国的首都是哪里？" \
    --temperature 0.1 --top_k 10

# 预训练续写模式
python generate_sample.py \
    --checkpoint checkpoints/ckpt_step85000.pth \
    --mode pretrain \
    --prompt "人工智能的发展"
```

**示例输出 (SFT, temp=0.1):**
```
Q: 中国的首都是哪里？
A: 中国的首都在北京。
```

## 致谢与参考

本项目的训练基础设施（数据处理、Tokenizer 训练、预训练/SFT 流程）**严格参照 [happy-llm](https://github.com/datawhalechina/happy-llm) 教学项目**（Datawhale 开源社区）。happy-llm 是一个从零搭建大模型的教程，我们在其基础上将模型架构替换为 SNN，训练流程保持对齐。

具体借鉴关系见下方 [与 happy-llm 的关系](#与-happy-llm-的关系) 章节。

## 核心架构

```
token → Embedding(D) → FP16 位提取 → K=16 帧 binary spike {0,1} (detach)
  → L × SNNDecoderLayer:
      RMSNorm(h) → PLIFNode → spike → SNNBlock → out_proj → 残差(中心化)
      RMSNorm(h) → PLIFNode → spike → SNNFFN  → out_proj → 残差(中心化)
  → RMSNorm → PLIFNode → spike → FP16 位重建 → decode_proj → LateralInhibition → Embedding^T (tied) → logits
```

### 架构要点

- **基础神经元**: PLIF (Parametric LIF)，动态 β(t), α(t), V_th(t) 由调制网络生成
- **连续残差流 (v7.5)**: 层间传递连续值 h，仅 SNN 子层内部使用 spike，解决 20 层梯度消失
- **Pre-LN RMSNorm (v7.5c)**: 分支归一化控制 PLIFNode 输入 scale + 残差中心化消除 DC 漂移
- **FP16 边界编解码 (v7.5b)**: IEEE 754 float16 位提取/重建，encode detach + decode 可微分
- **6 条并行输入路径**: W_in, W_β, W_α, W_th, W_gate, W_skip
- **并行化**: Triton Fused PLIF Kernel，单 kernel 完成扫描 + spike + 软重置 + 替代梯度
- **训练**: Surrogate Gradient + 标准反向传播 (v7.1+，替代 v5 的 SPSA 零阶优化)
- **框架**: SpikingJelly (conda env `SNN`)

## 与 happy-llm 的关系

本项目模型架构为原创 SNN 设计，但**训练基础设施严格对齐 happy-llm 教程**（[第五章：动手搭建大模型](https://github.com/datawhalechina/happy-llm/blob/main/docs/chapter5/第五章%20动手搭建大模型.md)），确保训练流程经过验证，集中精力在 SNN 架构创新上。

### 直接借鉴的部分（训练基础设施）

| 组件 | happy-llm 源文件 | NeuronSpark 对应文件 | 对齐程度 |
|------|-----------------|---------------------|---------|
| **数据下载** | `download_dataset.sh` | `scripts/download_dataset.sh` | 完全对齐：ModelScope + HuggingFace 镜像 |
| **数据预处理** | `deal_dataset.py` | `scripts/deal_dataset.py` | 完全对齐：预训练 512 字符切块 + SFT ChatML 格式转换 |
| **Tokenizer 训练** | `train_tokenizer.py` | `scripts/train_tokenizer.py` | 完全对齐：BPE, vocab=6144, NFKC 正则化, ChatML chat_template |
| **数据集加载** | `dataset.py` | `dataset.py` | 完全对齐：PretrainDataset (byte-offset JSONL), SFTDataset (assistant-only loss_mask) |
| **预训练循环** | `ddp_pretrain.py` | `train.py` / `train_ddp.py` | 高度对齐：Adam, Warmup+Cosine LR, autocast bf16, 梯度累积/裁剪 |
| **SFT 训练** | `ddp_sft_full.py` | `sft.py` / `sft_ddp.py` | 高度对齐：加载预训练权重 + SFTDataset + 相同训练循环 |
| **推理生成** | `model_sample.py` | `generate_sample.py` | 对齐：pretrain 续写 + SFT 对话，temperature/top_k 采样 |
| **预训练数据** | Seq-Monkey 10B tokens | 同（29M 样本 JSONL） | 完全相同 |
| **SFT 数据** | BelleGroup 350 万条 | 同 | 完全相同 |

### NeuronSpark 独创部分（SNN 架构）

| 组件 | 说明 |
|------|------|
| `model.py` | SNNLanguageModel：三段式 encode/snn_forward/decode，FP16 边界编解码，自回归 generate() |
| `atomic_ops/selective_plif.py` | SelectivePLIFNode：动态参数 PLIF 神经元 |
| `atomic_ops/plif_node.py` | PLIFNode：D 维固定参数 PLIF 神经元 |
| `atomic_ops/snn_block.py` | SNNBlock：SNN 注意力等价层（6 条并行路径 + 门控 + 跳跃连接） |
| `atomic_ops/snn_ffn.py` | SNNFFN：SNN 前馈网络（SwiGLU 风格三分支 spike 门控） |
| `atomic_ops/snn_decoder_layer.py` | SNNDecoderLayer：Pre-LN RMSNorm + Block + FFN + 残差中心化 |
| `atomic_ops/parallel_scan.py` | Triton Fused PLIF Kernel (v7.3+)，Row-param Kernel (v7.4+) |
| `atomic_ops/lateral_inhibition.py` | Triton 实现的侧抑制归一化 (divisive normalization) |
| `atomic_ops/rms_norm.py` | Pre-LN 分支归一化 (v7.5c) |
| `atomic_ops/fp16_codec.py` | IEEE 754 float16 位提取/重建编解码器 (v7.5b) |
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
| SFT 数据 | BelleGroup 3.5M | BelleGroup 3.5M |
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
pip install modelscope huggingface_hub  # 数据集下载

# 3. (DGX Spark / Blackwell GPU) Triton ptxas 配置
export TRITON_PTXAS_PATH=/usr/local/cuda-13.0/bin/ptxas
```

### 数据下载

参照 [happy-llm 第五章 5.3.1 节](https://github.com/datawhalechina/happy-llm/blob/main/docs/chapter5/第五章%20动手搭建大模型.md#531-数据下载)。

一键下载预训练数据（Seq-Monkey）和 SFT 数据（BelleGroup 350 万条）：

```bash
bash scripts/download_dataset.sh
```

**手动下载**（如脚本不可用）：

```bash
# 预训练数据：Seq-Monkey（出门问问序列猴子通用语料，~10B tokens）
mkdir -p data/seq-monkey
modelscope download --dataset ddzhu123/seq-monkey \
    mobvoi_seq_monkey_general_open_corpus.jsonl.tar.bz2 \
    --local_dir data/seq-monkey/
tar -xvf data/seq-monkey/mobvoi_seq_monkey_general_open_corpus.jsonl.tar.bz2 \
    -C data/seq-monkey/

# SFT 数据：BelleGroup 350 万条中文指令数据
mkdir -p data/sft
export HF_ENDPOINT=https://hf-mirror.com  # 国内镜像加速
huggingface-cli download --repo-type dataset --resume-download \
    BelleGroup/train_3.5M_CN --local-dir data/sft/BelleGroup/
```

### 数据预处理

参照 [happy-llm 第五章 5.3.1 节](https://github.com/datawhalechina/happy-llm/blob/main/docs/chapter5/第五章%20动手搭建大模型.md#531-数据下载)。

将原始数据转换为训练所需格式：

```bash
# 处理全部数据（预训练 + SFT）
python scripts/deal_dataset.py

# 仅处理预训练数据
python scripts/deal_dataset.py --pretrain_only

# 仅处理 SFT 数据
python scripts/deal_dataset.py --sft_only
```

**预训练数据处理**：将 Seq-Monkey 长文本按 512 字符切分成块，输出 JSONL 格式。

```
输入: data/seq-monkey/mobvoi_seq_monkey_general_open_corpus.jsonl
      {"text": "一段很长的中文文本...（可能数千字）"}

输出: data/seq-monkey/seq_monkey_datawhale.jsonl（约 29M 样本）
      {"text": "切分后的512字符文本块1"}
      {"text": "切分后的512字符文本块2"}
      ...
```

**SFT 数据处理**：将 BelleGroup 原始对话格式转换为 ChatML 标准格式。

```
输入: data/sft/BelleGroup/train_3.5M_CN.json
      {"conversations": [{"from": "human", "value": "..."}, {"from": "assistant", "value": "..."}]}

输出: data/sft/sft_data.jsonl（约 350 万条对话）
      [{"role": "system", "content": "你是一个AI助手"}, {"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]
```

### 训练 Tokenizer

参照 [happy-llm 第五章 5.3.2 节](https://github.com/datawhalechina/happy-llm/blob/main/docs/chapter5/第五章%20动手搭建大模型.md#532-训练-tokenizer)：

```bash
python scripts/train_tokenizer.py \
    --data_path data/seq-monkey/seq_monkey_datawhale.jsonl \
    --save_dir tokenizer_snn \
    --vocab_size 6144
```

输出 `tokenizer_snn/` 目录，包含：
- `tokenizer.json` — BPE 模型（6144 词表，NFKC 正则化，ByteLevel 编码）
- `tokenizer_config.json` — ChatML chat_template（兼容 Qwen2.5 格式）
- `special_tokens_map.json` — 特殊 token 映射

> 项目已包含训练好的 tokenizer (`tokenizer_snn/`)，可跳过此步直接使用。

### 预训练

参照 [happy-llm 第五章 5.3.4 节](https://github.com/datawhalechina/happy-llm/blob/main/docs/chapter5/第五章%20动手搭建大模型.md#534-预训练)。

```bash
# 单卡预训练
TRITON_PTXAS_PATH=/usr/local/cuda-13.0/bin/ptxas \
python -u train.py \
    --D 768 --D_ff 2304 --num_layers 20 \
    --batch_size 2 --accumulation_steps 32 \
    --warmup_iters 1000 --log_interval 10 \
    --save_interval 1000

# 多卡预训练 (DDP)
TRITON_PTXAS_PATH=/usr/local/cuda-13.0/bin/ptxas \
torchrun --nproc_per_node=4 train_ddp.py \
    --D 768 --D_ff 2304 \
    --batch_size 2 --accumulation_steps 8

# 后台运行（推荐）
tmux new-session -d -s train
tmux send-keys -t train 'PYTHONUNBUFFERED=1 TRITON_PTXAS_PATH=/usr/local/cuda-13.0/bin/ptxas conda run --no-capture-output -n SNN python -u train.py --D 768 --D_ff 2304 --batch_size 2 --accumulation_steps 32 --warmup_iters 1000 --log_interval 10 2>&1 | tee train.log' Enter

# 断续训练
python train.py --resume checkpoints/ckpt_step5000.pth

# 查看训练日志
tail -f train.log
```

Checkpoint 自动管理：每 `--save_interval` 步保存为 `ckpt_step{N}.pth`，自动清理只保留最新 5 个。

### SFT 微调

参照 [happy-llm 第五章 5.3.5 节](https://github.com/datawhalechina/happy-llm/blob/main/docs/chapter5/第五章%20动手搭建大模型.md#535-sft-训练)。

加载预训练权重，用 SFTDataset 进行监督微调。Loss 仅在 assistant 回复 token 上计算。

```bash
# 单卡 SFT
python sft.py \
    --pretrained_ckpt checkpoints/ckpt_step10000.pth \
    --sft_data_path data/sft/sft_data.jsonl \
    --D 768 --D_ff 2304 \
    --batch_size 4 --accumulation_steps 16 \
    --learning_rate 5e-5 --epochs 3 --warmup_iters 100

# 多卡 SFT (DDP)
torchrun --nproc_per_node=4 sft_ddp.py \
    --pretrained_ckpt checkpoints/ckpt_step10000.pth \
    --sft_data_path data/sft/sft_data.jsonl \
    --D 768 --D_ff 2304 \
    --batch_size 2 --accumulation_steps 8 \
    --learning_rate 5e-5 --epochs 3

# 断续训练
python sft.py --resume checkpoints_sft/ckpt_step500.pth
```

**SFT 与预训练的关键区别：**

| | 预训练 | SFT |
|---|---|---|
| 数据 | PretrainDataset (纯文本) | SFTDataset (ChatML 对话) |
| Loss 范围 | 全部非 padding token | 仅 assistant 回复 token |
| 初始权重 | 随机初始化 | 加载预训练 checkpoint |
| 学习率 | 2e-4 | 5e-5 (更低) |
| Epochs | 1 | 3 |
| 输出目录 | `checkpoints/` | `checkpoints_sft/` |

### 推理生成

参照 [happy-llm model_sample.py](https://github.com/datawhalechina/happy-llm/blob/main/docs/chapter5/code/model_sample.py)。

SNN 生成机制：Prefill 阶段用 `forward_parallel` 并行处理 prompt 建立神经元状态，自回归阶段逐 token 用 `forward_parallel` 处理 K=16 帧（复用 Triton parallel scan），神经元膜电位 V 跨 token 连续传递。

```bash
# 文本续写（预训练模型）
python generate_sample.py \
    --checkpoint checkpoints/ckpt_step10000.pth \
    --mode pretrain \
    --prompt "人工智能的发展" \
    --max_new_tokens 256 --temperature 0.8 --top_k 50

# 对话生成（SFT 模型）
python generate_sample.py \
    --checkpoint checkpoints_sft/ckpt_step3000.pth \
    --mode sft \
    --prompt "什么是脉冲神经网络？"

# 交互模式
python generate_sample.py \
    --checkpoint checkpoints/ckpt_step10000.pth \
    --interactive

# 交互模式中可切换模式：
#   [pretrain] > 今天天气
#   [pretrain] > mode sft
#   [sft] > 请介绍一下深度学习
#   [sft] > quit
```

**生成参数说明：**

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--checkpoint` | (必填) | 模型 checkpoint 路径 |
| `--mode` | `pretrain` | 生成模式：`pretrain` (续写) / `sft` (对话) |
| `--prompt` | — | 输入文本（非交互模式） |
| `--max_new_tokens` | 256 | 最大生成 token 数 |
| `--temperature` | 0.8 | 采样温度（0 = greedy，越高越随机） |
| `--top_k` | 50 | Top-k 采样（限制候选 token 数） |
| `--interactive` | — | 交互式生成模式 |

## 训练参数参考

### 模型参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--D` | 768 | 隐藏维度 |
| `--N` | 8 | 神经元分组数（状态扩展因子） |
| `--K` | 16 | 每 token SNN 时间步数 / FP16 编码位数 |
| `--num_layers` | 20 | SNN 解码层数 |
| `--D_ff` | 2304 | FFN 中间维度 (通常 3×D) |
| `--vocab_size` | 6144 | 词表大小 |

### 训练参数

| 参数 | 预训练默认 | SFT 默认 | 说明 |
|------|-----------|---------|------|
| `--batch_size` | 8 | 4 | 每 GPU micro-batch 大小 |
| `--accumulation_steps` | 8 | 8 | 梯度累积步数 |
| `--learning_rate` | 2e-4 | 5e-5 | 峰值学习率 |
| `--warmup_iters` | 1000 | 100 | Warmup 步数 |
| `--epochs` | 1 | 3 | 训练轮数 |
| `--grad_clip` | 1.0 | 1.0 | 梯度裁剪阈值 |
| `--neuron_lr_mult` | 10.0 | 10.0 | 神经元参数学习率倍率 |
| `--log_interval` | 10 | 100 | 日志打印间隔 |
| `--save_interval` | 1000 | 1000 | Checkpoint 保存间隔 |

### 显存与 batch_size 参考

| batch_size | 约显存占用 | 适用硬件 |
|:----------:|:---------:|:--------:|
| 2 | ~29 GB | RTX 3090/4090 (24GB) + 梯度检查点 |
| 4 | ~55 GB | A100 40GB / A6000 48GB |
| 8 | ~103 GB | DGX Spark (128GB 统一内存) / A100 80GB |

## 训练流程总览

参照 happy-llm 教程的完整训练流程：

```
Step 1: 数据下载
  bash scripts/download_dataset.sh
  → Seq-Monkey 预训练数据 + BelleGroup SFT 数据

Step 2: 数据预处理
  python scripts/deal_dataset.py
  → 预训练: 29M 样本 JSONL (512 字符切块)
  → SFT: 350 万条 ChatML 格式对话 JSONL

Step 3: Tokenizer 训练 (首次)
  python scripts/train_tokenizer.py
  → tokenizer_snn/ (6144 词表 BPE + ChatML chat_template)

Step 4: 预训练
  python train.py (单卡) / torchrun train_ddp.py (多卡)
  → next token prediction on Seq-Monkey
  → 目标: loss 收敛到 3~4

Step 5: SFT 微调
  python sft.py --pretrained_ckpt checkpoints/ckpt_step{N}.pth
  → 加载预训练权重 + BelleGroup 指令数据
  → 目标: 模型能进行基本对话

Step 6: 推理生成
  python generate_sample.py --checkpoint checkpoints_sft/ckpt_step{N}.pth
  → 文本续写 / 对话生成
```

## 项目结构

```
NeuronSpark/
├── model.py                        # SNNLanguageModel (v7.5c: 三段式 encode/snn_forward/decode + generate)
├── train.py                        # 预训练脚本-单卡 (对齐 happy-llm ddp_pretrain.py)
├── train_ddp.py                    # 预训练脚本-多卡 DDP (torchrun)
├── sft.py                          # SFT 微调脚本-单卡 (对齐 happy-llm ddp_sft_full.py)
├── sft_ddp.py                      # SFT 微调脚本-多卡 DDP
├── generate_sample.py              # 推理生成脚本 (对齐 happy-llm model_sample.py)
├── dataset.py                      # 数据集加载 (对齐 happy-llm dataset.py)
├── atomic_ops/                     # SNN 核心算子（独创）
│   ├── __init__.py
│   ├── selective_plif.py           # SelectivePLIFNode: 动态参数 PLIF 神经元
│   ├── plif_node.py                # PLIFNode: D 维固定参数 PLIF 神经元
│   ├── snn_block.py                # SNNBlock: SNN 注意力等价层 (6 路并行 + 门控)
│   ├── snn_ffn.py                  # SNNFFN: SNN 前馈网络 (SwiGLU 风格三分支)
│   ├── snn_decoder_layer.py        # SNNDecoderLayer: Pre-LN + Block + FFN + 残差中心化
│   ├── parallel_scan.py            # Triton Fused PLIF Kernel + Row-param Kernel
│   ├── lateral_inhibition.py       # Triton 侧抑制归一化 (输出层使用)
│   ├── rms_norm.py                 # Pre-LN 分支归一化 (v7.5c)
│   └── fp16_codec.py              # IEEE 754 float16 位编解码器 (v7.5b)
├── scripts/                        # 数据处理脚本 (对齐 happy-llm)
│   ├── download_dataset.sh         # 一键下载预训练 + SFT 数据集
│   ├── deal_dataset.py             # 数据预处理 (预训练切块 + SFT ChatML 转换)
│   ├── train_tokenizer.py          # BPE Tokenizer 训练 (6144 词表)
│   └── prepare_data.py             # SkyPile-150B 数据集处理 (备选)
├── docs/                           # 设计文档
│   ├── SNN_SELECTIVE_STATE_SPACE.md  # 主设计文档 (2200+ 行, v1→v7.5 全部演进)
│   ├── PARALLEL_SCAN_OPTIMIZATION.md # Triton Kernel 优化文档
│   ├── OPEN_ISSUES.md              # 已解决的设计问题记录
│   ├── Q5.md                       # 技术验证方法论
│   └── SNN_DIFFUSION_PLAN.md       # 未来方向: SNN 扩散模型
├── exp/                            # 实验脚本 (验证 + 基准测试)
│   ├── verify_*.py                 # 各版本正确性验证 (梯度、融合算子、端到端)
│   ├── bench_*.py                  # 性能基准测试 (Triton kernel, 编译, 层级)
│   └── diagnose_*.py              # 训练诊断工具
├── notebooks/                      # 实验 Notebook
│   ├── linear_layer_analysis.ipynb # SpikingJelly 线性层分析
│   └── neuron_comparison.ipynb     # 7 种神经元模型对比 → PLIF 最优
├── tokenizer_snn/                  # 已训练的 BPE tokenizer (6144 词表)
├── data/                           # 训练数据 (不纳入 git)
│   ├── seq-monkey/                 # Seq-Monkey 预训练数据 (29M 样本 JSONL)
│   └── sft/                        # BelleGroup SFT 数据 (350 万条对话)
├── checkpoints/                    # 预训练 checkpoint (不纳入 git)
├── checkpoints_sft/                # SFT checkpoint (不纳入 git)
└── archive/                        # 历史版本归档
    ├── v5_pre_ffn/                 # v5 代码快照
    ├── v6_pre_parallel/            # v6 代码快照
    ├── logs/                       # 历史训练日志
    └── checkpoints/                # 历史 checkpoint
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
| v7.5 | 连续残差流：层间传递连续值，层内归一化移除 |
| v7.5b | FP16 边界编解码 + 输出 PLIFNode |
| **v7.5c** | **Pre-LN RMSNorm + 残差中心化，解决 h 均值漂负问题** |

## 当前训练配置 (v7.5c)

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
| Batch size | 2 × accum 32 = 64 effective |
| 学习率 | 2e-4 (warmup 1000 → cosine → 2e-5) |
| 精度 | bfloat16 |
| 数据集 | Seq-Monkey (29M 样本) |
| 硬件 | NVIDIA GB10 (DGX Spark, 128GB 统一内存) |
| TPS | ~95 tokens/sec |
| 显存占用 | ~29 GB |

## 环境

```bash
conda activate SNN
# 核心依赖: PyTorch, SpikingJelly, Triton, transformers, tokenizers, pandas, numpy, tqdm
# 数据下载: modelscope, huggingface_hub
# 硬件: NVIDIA GPU (推荐 Blackwell/Ampere 架构, 24GB+ 显存)
# DGX Spark 需配置: export TRITON_PTXAS_PATH=/usr/local/cuda-13.0/bin/ptxas
```

## License

Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0)
