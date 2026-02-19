#!/bin/bash
#
# 数据集下载脚本（参照 happy-llm download_dataset.sh）
#
# 下载预训练数据集（Seq-Monkey）和 SFT 数据集（BelleGroup 350万条）。
#
# 前置依赖：
#   pip install modelscope huggingface_hub
#
# 用法：
#   bash scripts/download_dataset.sh
#

set -e

# HuggingFace 镜像加速（国内环境）
export HF_ENDPOINT=https://hf-mirror.com

# ============================================================
# 1. 预训练数据：Seq-Monkey（出门问问序列猴子通用语料）
#    来源: ModelScope ddzhu123/seq-monkey
#    大小: ~10B tokens，压缩包约 6GB
# ============================================================

PRETRAIN_DIR="data/seq-monkey"
mkdir -p "${PRETRAIN_DIR}"

echo "=========================================="
echo "  Step 1: 下载 Seq-Monkey 预训练数据集"
echo "=========================================="

if [ ! -f "${PRETRAIN_DIR}/mobvoi_seq_monkey_general_open_corpus.jsonl" ]; then
    echo "从 ModelScope 下载（需要 pip install modelscope）..."
    modelscope download \
        --dataset ddzhu123/seq-monkey \
        mobvoi_seq_monkey_general_open_corpus.jsonl.tar.bz2 \
        --local_dir "${PRETRAIN_DIR}"

    echo "解压..."
    tar -xvf "${PRETRAIN_DIR}/mobvoi_seq_monkey_general_open_corpus.jsonl.tar.bz2" \
        -C "${PRETRAIN_DIR}"

    echo "Seq-Monkey 下载完成: ${PRETRAIN_DIR}/mobvoi_seq_monkey_general_open_corpus.jsonl"
else
    echo "Seq-Monkey 已存在，跳过下载"
fi

# ============================================================
# 2. SFT 数据：BelleGroup 350万条中文指令数据
#    来源: HuggingFace BelleGroup/train_3.5M_CN
#    大小: 约 3.5GB
# ============================================================

SFT_DIR="data/sft"
mkdir -p "${SFT_DIR}"

echo ""
echo "=========================================="
echo "  Step 2: 下载 BelleGroup SFT 数据集"
echo "=========================================="

if [ ! -d "${SFT_DIR}/BelleGroup" ] || [ -z "$(ls -A ${SFT_DIR}/BelleGroup 2>/dev/null)" ]; then
    echo "从 HuggingFace 下载（使用镜像加速）..."
    huggingface-cli download \
        --repo-type dataset \
        --resume-download \
        BelleGroup/train_3.5M_CN \
        --local-dir "${SFT_DIR}/BelleGroup"

    echo "BelleGroup 下载完成: ${SFT_DIR}/BelleGroup/"
else
    echo "BelleGroup 已存在，跳过下载"
fi

echo ""
echo "=========================================="
echo "  下载完成！后续步骤："
echo "=========================================="
echo ""
echo "  1. 预处理数据:"
echo "     python scripts/deal_dataset.py"
echo ""
echo "  2. (首次) 训练 Tokenizer:"
echo "     python scripts/train_tokenizer.py"
echo ""
echo "  3. 开始预训练:"
echo "     python train.py --D 768 --D_ff 2304"
echo ""
