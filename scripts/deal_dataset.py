"""
数据预处理脚本（参照 happy-llm deal_dataset.py）

处理两类数据：
  1. 预训练数据：Seq-Monkey 长文本 → 512 字符切块 → JSONL
  2. SFT 数据：BelleGroup 原始格式 → ChatML 标准对话格式 → JSONL

前置步骤：
  先运行 scripts/download_dataset.sh 下载原始数据。

用法：
    # 处理全部
    python scripts/deal_dataset.py

    # 仅预训练数据
    python scripts/deal_dataset.py --pretrain_only

    # 仅 SFT 数据
    python scripts/deal_dataset.py --sft_only
"""

import os
import json
import argparse
from tqdm import tqdm


# ============================================================
# 1. 预训练数据预处理
# ============================================================

def process_pretrain_data(input_path, output_path, chunk_size=512):
    """
    将 Seq-Monkey 原始 JSONL 中的长文本按固定长度切分。

    输入格式: {"text": "一段很长的中文文本..."}
    输出格式: {"text": "切分后的512字符文本块"}（每块一行）

    Args:
        input_path: 原始 JSONL 路径
        output_path: 输出 JSONL 路径
        chunk_size: 切分长度（字符数，默认 512）
    """
    def split_text(text, size=512):
        return [text[i:i+size] for i in range(0, len(text), size)]

    print(f"处理预训练数据: {input_path}")
    print(f"  切分长度: {chunk_size} 字符")

    total_chunks = 0
    with open(output_path, 'w', encoding='utf-8') as out:
        with open(input_path, 'r', encoding='utf-8') as f:
            for line in tqdm(f, desc="预训练数据"):
                data = json.loads(line)
                text = data['text']
                chunks = split_text(text, chunk_size)
                for chunk in chunks:
                    if len(chunk.strip()) > 0:
                        out.write(json.dumps({'text': chunk}, ensure_ascii=False) + '\n')
                        total_chunks += 1

    print(f"  完成: {total_chunks:,} 条样本 → {output_path}")
    return total_chunks


# ============================================================
# 2. SFT 数据预处理
# ============================================================

def convert_message(data):
    """
    将 BelleGroup 原始对话格式转换为 ChatML 标准格式。

    BelleGroup 原始: {"conversations": [{"from": "human", "value": "..."}, {"from": "assistant", "value": "..."}]}
    ChatML 标准:     [{"role": "system", "content": "..."}, {"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]
    """
    messages = [
        {"role": "system", "content": "你是一个AI助手"},
    ]
    for item in data:
        if item['from'] == 'human':
            messages.append({'role': 'user', 'content': item['value']})
        elif item['from'] == 'assistant':
            messages.append({'role': 'assistant', 'content': item['value']})
    return messages


def process_sft_data(input_path, output_path):
    """
    将 BelleGroup 原始 JSON 转换为 ChatML 格式 JSONL。

    输入格式: {"conversations": [{"from": "human", "value": "..."}, ...]}
    输出格式: [{"role": "system", ...}, {"role": "user", ...}, {"role": "assistant", ...}]（每条一行）

    Args:
        input_path: BelleGroup 原始 JSON 路径
        output_path: 输出 JSONL 路径
    """
    print(f"处理 SFT 数据: {input_path}")

    total = 0
    skipped = 0
    with open(output_path, 'w', encoding='utf-8') as out:
        with open(input_path, 'r', encoding='utf-8') as f:
            for line in tqdm(f, desc="SFT 数据"):
                try:
                    item = json.loads(line)
                    messages = convert_message(item['conversations'])
                    # 至少要有 system + user + assistant 三条
                    if len(messages) >= 3:
                        out.write(json.dumps(messages, ensure_ascii=False) + '\n')
                        total += 1
                    else:
                        skipped += 1
                except (json.JSONDecodeError, KeyError):
                    skipped += 1
                    continue

    print(f"  完成: {total:,} 条对话 → {output_path}")
    if skipped > 0:
        print(f"  跳过: {skipped:,} 条（格式不合规）")
    return total


# ============================================================
# 入口
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="数据预处理（预训练 + SFT）")
    parser.add_argument('--pretrain_only', action='store_true', help='仅处理预训练数据')
    parser.add_argument('--sft_only', action='store_true', help='仅处理 SFT 数据')

    # 预训练数据路径
    parser.add_argument('--pretrain_input', type=str,
                        default='data/seq-monkey/mobvoi_seq_monkey_general_open_corpus.jsonl')
    parser.add_argument('--pretrain_output', type=str,
                        default='data/seq-monkey/seq_monkey_datawhale.jsonl')
    parser.add_argument('--chunk_size', type=int, default=512)

    # SFT 数据路径
    parser.add_argument('--sft_input', type=str,
                        default='data/sft/BelleGroup/train_3.5M_CN.json')
    parser.add_argument('--sft_output', type=str,
                        default='data/sft/sft_data.jsonl')

    args = parser.parse_args()

    do_pretrain = not args.sft_only
    do_sft = not args.pretrain_only

    if do_pretrain:
        if os.path.exists(args.pretrain_input):
            process_pretrain_data(args.pretrain_input, args.pretrain_output, args.chunk_size)
        else:
            print(f"预训练原始数据不存在: {args.pretrain_input}")
            print("  请先运行: bash scripts/download_dataset.sh")

    if do_sft:
        if os.path.exists(args.sft_input):
            os.makedirs(os.path.dirname(args.sft_output), exist_ok=True)
            process_sft_data(args.sft_input, args.sft_output)
        else:
            print(f"SFT 原始数据不存在: {args.sft_input}")
            print("  请先运行: bash scripts/download_dataset.sh")
