"""
预处理数据集（参考 happy-llm deal_dataset.py）

将 Seq-Monkey 原始 JSONL 中的长文本按 512 字符切分成块，
输出标准 JSONL 格式供 tokenizer 训练和模型预训练使用。

用法：
    python deal_dataset.py
"""

import json
from tqdm import tqdm

# Seq-Monkey 原始数据路径
pretrain_data = 'data/seq-monkey/mobvoi_seq_monkey_general_open_corpus.jsonl'
output_pretrain_data = 'data/seq-monkey/seq_monkey_datawhale.jsonl'


# 处理预训练数据
def split_text(text, chunk_size=512):
    """将文本按指定长度切分成块"""
    return [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]


with open(output_pretrain_data, 'a', encoding='utf-8') as pretrain:
    with open(pretrain_data, 'r', encoding='utf-8') as f:
        data = f.readlines()
        for line in tqdm(data, desc=f"Processing lines in {pretrain_data}", leave=False):
            line = json.loads(line)
            text = line['text']
            chunks = split_text(text)
            for chunk in chunks:
                pretrain.write(json.dumps({'text': chunk}, ensure_ascii=False) + '\n')
