"""
预处理脚本：SkyPile-150B JSONL → flat binary token 文件

功能：
- 遍历所有 JSONL 文件，逐行读取 text 字段
- 用 Qwen3-0.6B tokenizer encode（add_special_tokens=False）
- Token IDs 写入 flat binary 文件（np.uint32，vocab=151936 > 65535）
- 输出：train_tokens.bin + val_tokens.bin
- 多进程并行 tokenize
- 断点续传（记录已处理文件列表）

用法：
    conda activate SNN
    python prepare_data.py \
        --data_dir data/SkyPile-150B \
        --output_dir data/processed \
        --val_tokens 2000000 \
        --num_workers 8
"""

import os
import json
import glob
import argparse
import numpy as np
from pathlib import Path
from multiprocessing import Pool
from functools import partial

from transformers import AutoTokenizer


def tokenize_file(jsonl_path: str, tokenizer_name: str) -> np.ndarray:
    """对单个 JSONL 文件进行 tokenize，返回 uint32 数组。"""
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    all_ids = []
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            text = obj.get('text', '')
            if not text.strip():
                continue
            ids = tokenizer.encode(text, add_special_tokens=False)
            all_ids.extend(ids)
    return np.array(all_ids, dtype=np.uint32)


def get_jsonl_files(data_dir: str) -> list[str]:
    """获取所有 JSONL 文件，按名称排序确保确定性。"""
    patterns = ['**/*.jsonl', '**/*.jsonl.gz']
    files = []
    for pat in patterns:
        files.extend(glob.glob(os.path.join(data_dir, pat), recursive=True))
    # 排除 .gz（需要额外处理），只保留纯 jsonl
    files = [f for f in files if not f.endswith('.gz')]
    files.sort()
    return files


def load_progress(progress_file: str) -> set[str]:
    """加载已处理文件列表（断点续传）。"""
    if not os.path.exists(progress_file):
        return set()
    with open(progress_file, 'r') as f:
        return set(line.strip() for line in f if line.strip())


def save_progress(progress_file: str, filepath: str):
    """追加记录已处理文件。"""
    with open(progress_file, 'a') as f:
        f.write(filepath + '\n')


def main():
    parser = argparse.ArgumentParser(description='预处理 SkyPile-150B → binary tokens')
    parser.add_argument('--data_dir', type=str, default='data/SkyPile-150B',
                        help='JSONL 数据目录')
    parser.add_argument('--output_dir', type=str, default='data/processed',
                        help='输出目录')
    parser.add_argument('--tokenizer', type=str, default='Qwen/Qwen3-0.6B',
                        help='Tokenizer 名称')
    parser.add_argument('--val_tokens', type=int, default=2_000_000,
                        help='验证集 token 数（从头部切分）')
    parser.add_argument('--num_workers', type=int, default=8,
                        help='并行 tokenize 进程数')
    parser.add_argument('--batch_files', type=int, default=4,
                        help='每批并行处理的文件数')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    train_bin = os.path.join(args.output_dir, 'train_tokens.bin')
    val_bin = os.path.join(args.output_dir, 'val_tokens.bin')
    progress_file = os.path.join(args.output_dir, 'progress.txt')

    # 获取所有 JSONL 文件
    jsonl_files = get_jsonl_files(args.data_dir)
    if not jsonl_files:
        print(f"ERROR: No JSONL files found in {args.data_dir}")
        print("请确认数据集已下载到指定目录。")
        return

    print(f"Found {len(jsonl_files)} JSONL files in {args.data_dir}")

    # 断点续传：跳过已处理文件
    done_files = load_progress(progress_file)
    remaining = [f for f in jsonl_files if f not in done_files]
    print(f"Already processed: {len(done_files)}, remaining: {len(remaining)}")

    # 如果全部处理完，直接做 val split
    if not remaining and os.path.exists(train_bin):
        print("All files already processed. Checking val split...")
        _do_val_split(train_bin, val_bin, args.val_tokens)
        return

    # 打开输出文件（追加模式）
    total_tokens = 0
    if os.path.exists(train_bin):
        total_tokens = os.path.getsize(train_bin) // 4  # uint32 = 4 bytes
        print(f"Existing train_tokens.bin: {total_tokens:,} tokens")

    tokenize_fn = partial(tokenize_file, tokenizer_name=args.tokenizer)

    # 分批并行处理
    for batch_start in range(0, len(remaining), args.batch_files):
        batch = remaining[batch_start:batch_start + args.batch_files]
        print(f"\nProcessing batch {batch_start // args.batch_files + 1}: "
              f"{len(batch)} files...")

        if args.num_workers > 1 and len(batch) > 1:
            with Pool(min(args.num_workers, len(batch))) as pool:
                results = pool.map(tokenize_fn, batch)
        else:
            results = [tokenize_fn(f) for f in batch]

        # 写入 binary 文件
        with open(train_bin, 'ab') as fout:
            for filepath, tokens in zip(batch, results):
                if len(tokens) > 0:
                    tokens.tofile(fout)
                    total_tokens += len(tokens)
                save_progress(progress_file, filepath)
                print(f"  {os.path.basename(filepath)}: "
                      f"{len(tokens):,} tokens (total: {total_tokens:,})")

    print(f"\nTokenization complete. Total: {total_tokens:,} tokens")

    # Val/Train 切分
    _do_val_split(train_bin, val_bin, args.val_tokens)


def _do_val_split(train_bin: str, val_bin: str, val_tokens: int):
    """从 train_tokens.bin 头部切出 val_tokens.bin。"""
    total_tokens = os.path.getsize(train_bin) // 4

    if total_tokens <= val_tokens:
        print(f"WARNING: Total tokens ({total_tokens:,}) <= val_tokens ({val_tokens:,})")
        print("Skipping val split.")
        return

    if os.path.exists(val_bin):
        existing_val = os.path.getsize(val_bin) // 4
        print(f"val_tokens.bin already exists ({existing_val:,} tokens). Skipping split.")
        return

    print(f"\nSplitting: {val_tokens:,} val tokens from {total_tokens:,} total...")

    # 读取前 val_tokens 到 val_bin
    data = np.memmap(train_bin, dtype=np.uint32, mode='r')
    val_data = np.array(data[:val_tokens])  # copy to memory
    val_data.tofile(val_bin)
    print(f"  Wrote {val_bin}: {val_tokens:,} tokens")

    # 重写 train_bin（去掉前 val_tokens 个）
    train_data = np.array(data[val_tokens:])  # copy to memory
    del data  # release mmap
    train_data.tofile(train_bin)
    print(f"  Rewrote {train_bin}: {len(train_data):,} tokens")
    print("Val split complete.")


if __name__ == '__main__':
    main()
