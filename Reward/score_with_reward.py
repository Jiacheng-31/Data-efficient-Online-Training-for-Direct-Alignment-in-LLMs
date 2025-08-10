#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
多 GPU reward 打分：
  1. torchrun --nproc_per_node=N score_jsonl.py --model_path ... --input_file ... --output_file ...
  2. 支持 Qwen3 Reward / 其它单列 logits = 1 的奖励模型
  3. 每行 JSON 必须含 {"prompt": str, "responses": [str, ...]}
  4. 结果新增 "scores": [float, ...] 后写回
"""

import argparse, json, os, sys, torch
import torch.distributed as dist                 # ← 新增
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from tqdm import tqdm


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", required=True, help="本地或 HF Hub 模型路径")
    p.add_argument("--input_file", required=True, help="输入 .jsonl")
    p.add_argument("--output_file", required=True, help="输出 .jsonl")
    p.add_argument("--dtype", default="bfloat16",
                   choices=["bfloat16", "float16", "float32"])
    p.add_argument("--max_length", type=int, default=4096,
                   help="截断长度（tokens）")
    return p.parse_args()


def main():
    args = parse_args()

    # ---------- 分布式信息 ----------
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    device = f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu"

    # *重要*：多卡时初始化进程组（单卡跳过）
    if world_size > 1 and not dist.is_initialized():
        dist.init_process_group(backend="nccl", init_method="env://")

    # ---------- 加载模型 / tokenizer ----------
    dtype_map = {
        "bfloat16": torch.bfloat16,
        "float16":  torch.float16,
        "float32":  torch.float32,
    }
    rm = AutoModelForSequenceClassification.from_pretrained(
        args.model_path,
        torch_dtype=dtype_map[args.dtype],
        low_cpu_mem_usage=True,
        device_map=None,          # 显式放到当前卡
    ).to(device).eval()

    tokenizer = AutoTokenizer.from_pretrained(args.model_path)

    # ---------- 打分循环 ----------
    part_out = f"{args.output_file}.part{local_rank}"
    fout = open(part_out, "w", encoding="utf-8")

    with open(args.input_file, "r", encoding="utf-8") as fin:
        for idx, line in enumerate(fin):
            if idx % world_size != local_rank:
                continue  # 留给其它进程
            record = json.loads(line)
            prompt = record["prompt"]
            responses = record["responses"]

            scores = []
            for rsp in responses:
                conv = [
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": rsp},
                ]
                text = tokenizer.apply_chat_template(conv, tokenize=False)
                # 去掉可能重复的 <bos>
                if tokenizer.bos_token and text.startswith(tokenizer.bos_token):
                    text = text[len(tokenizer.bos_token):]

                inputs = tokenizer(
                    text,
                    return_tensors="pt",
                    truncation=True,
                    max_length=args.max_length,
                ).to(device)

                with torch.no_grad():
                    score = rm(**inputs).logits[0][0].item()
                scores.append(score)

            record["scores"] = scores
            fout.write(json.dumps(record, ensure_ascii=False) + "\n")

    fout.close()

    # ---------- 主进程合并分片 ----------
    if dist.is_initialized():
        dist.barrier()  # 等所有分片写完

    if local_rank == 0:
        with open(args.output_file, "w", encoding="utf-8") as final_out:
            for r in range(world_size):
                part_path = f"{args.output_file}.part{r}"
                with open(part_path, "r", encoding="utf-8") as pf:
                    for l in pf:
                        final_out.write(l)
                os.remove(part_path)
        print(f"[Done] merged into {args.output_file}")

    # ---------- 清理进程组 ----------
    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
