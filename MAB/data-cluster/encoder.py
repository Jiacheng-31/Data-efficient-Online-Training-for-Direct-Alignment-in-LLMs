#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Batch‑encode an ULTRA‑style JSONL file with bge-large-en-v1.5.
Outputs a numpy file (embeddings) + optional line‑id mapping.
"""

import json
import os
import numpy as np
from pathlib import Path
from tqdm import tqdm

import torch
from transformers import AutoTokenizer, AutoModel

# ---------- 可按需修改 ----------
MODEL_NAME   = "bge-large-en-v1.5"   # 嵌入模型
INPUT_FILE   = "hh-prompt.jsonl"  # 原始数据
OUTPUT_NPY   = "embedding_data_hh.npy" # 保存嵌入
BATCH_SIZE   = 16                   # 每 GPU 可显存调节
MAX_LENGTH   = 512                  # tokenizer truncation
# --------------------------------

# 1) 载入模型 & 多 GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME)
if torch.cuda.device_count() > 1:
    model = torch.nn.DataParallel(model)        # 利用多 GPU
model.to(device).eval()

@torch.no_grad()
def encode_batch(text_list):
    """给定若干文本，返回 (N, H) numpy embedding。"""
    tok = tokenizer(
        text_list,
        padding=True,
        truncation=True,
        max_length=MAX_LENGTH,
        return_tensors="pt"
    ).to(device)

    out = model(**tok).last_hidden_state.mean(dim=1)   # mean‑pool
    return out.cpu().numpy()                           # -> numpy

# 2) 统计样本总数（为了进度条总长度）
with open(INPUT_FILE, "r", encoding="utf-8") as f:
    total_lines = sum(1 for _ in f)

# 3) 主循环：批量嵌入
all_embs = []
batch_prompt = []

with open(INPUT_FILE, "r", encoding="utf-8") as f, tqdm(total=total_lines, desc="Encoding") as pbar:
    for line in f:
        item = json.loads(line)
        batch_prompt.append(item["prompt"])  # 使用 "prompt" 作为输入

        # 批量已满 -> 计算嵌入
        if len(batch_prompt) == BATCH_SIZE:
            emb_prompt = encode_batch(batch_prompt)  # 对 batch 进行嵌入
            all_embs.extend(emb_prompt)  # 将嵌入结果添加到结果列表
            batch_prompt = []  # 清空当前 batch
            pbar.update(BATCH_SIZE)

    # 处理最后不足 batch 的残留
    if batch_prompt:
        emb_prompt = encode_batch(batch_prompt)  # 对剩余文本进行嵌入
        all_embs.extend(emb_prompt)
        pbar.update(len(batch_prompt))

all_embs = np.vstack(all_embs)  # 将嵌入结果堆叠成一个 numpy 数组

# 4) 保存嵌入
np.save(OUTPUT_NPY, all_embs)
print(f"[✓] embeddings saved to {OUTPUT_NPY} — shape {all_embs.shape}")
