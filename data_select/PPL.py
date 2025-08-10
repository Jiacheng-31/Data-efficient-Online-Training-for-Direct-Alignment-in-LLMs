#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
from pathlib import Path

import torch
import torch.distributed as dist
from transformers import AutoTokenizer, AutoModelForCausalLM

def parse_args():
    parser = argparse.ArgumentParser(
        description="Distributed batch PPL + TokenCount computation for JSONL of (prompt, responses)"
    )
    parser.add_argument("--model_name", type=str, required=True,
                        help="预训练模型名或路径，如 meta-llama/Llama-3-7b 或 Qwen/Qwen-7b")
    parser.add_argument("--input_path", type=Path, required=True,
                        help="输入 JSONL 文件路径")
    parser.add_argument("--output_prefix", type=str, required=True,
                        help="输出文件前缀，例如 predictions")
    parser.add_argument("--batch_size", type=int, default=8,
                        help="每个进程累积多少行后调用一次模型计算")
    parser.add_argument("--max_length", type=int, default=1024,
                        help="tokenizer 的最大截断长度")
    return parser.parse_args()

def init_distributed():
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    return rank, world_size

def compute_metrics(prompts, responses, model, tokenizer, device, max_length):
    """
    对一批 prompts + 每个 prompt 下的多个 responses，返回：
      - ppls_batch: List[List[float]]
      - tokencounts_batch: List[List[int]]
    """
    ppls_batch = []
    tokencounts_batch = []

    for prompt, resp_list in zip(prompts, responses):
        # 将 prompt+response 拼成小批次
        texts = [prompt + r for r in resp_list]
        enc = tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length
        ).to(device)

        input_ids      = enc["input_ids"]       # [M, L]
        attention_mask = enc["attention_mask"]  # [M, L]

        # 前向计算 logits
        with torch.no_grad():
            logits = model(input_ids, attention_mask=attention_mask).logits  # [M, L, V]

        # 计算 log-probs
        log_probs = torch.log_softmax(logits, dim=-1)  # [M, L, V]

        # 计算 prompt 长度（token 数）
        prompt_ids = tokenizer(prompt, add_special_tokens=False).input_ids
        p_len = len(prompt_ids)

        # 构造 labels：prompt 部分置 -100，不计入 loss
        labels = input_ids.clone()
        labels[:, :p_len] = -100

        # 取每个位置的 log-prob
        seq_logp = log_probs.gather(2, input_ids.unsqueeze(-1)).squeeze(-1)  # [M, L]

        # mask：只保留 response 部分且不是 pad 的位置
        mask = (attention_mask == 1) & (labels != -100)  # [M, L]

        # 累计 log-prob 及 token 数
        sum_logp = (seq_logp * mask).sum(dim=1)  # [M]
        n_tokens = mask.sum(dim=1)               # [M]

        # 计算 PPL
        avg_nll = - sum_logp / n_tokens
        ppls = torch.exp(avg_nll).tolist()       # List[float]
        ppls_batch.append(ppls)

        # 记录每条 response 的 token 数
        tokencounts_batch.append(n_tokens.tolist())

    return ppls_batch, tokencounts_batch

if __name__ == "__main__":
    args = parse_args()
    rank, world_size = init_distributed()
    device = f"cuda:{rank}"

    # 加载 tokenizer & 模型
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    if tokenizer.pad_token is None:
        if tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    model = AutoModelForCausalLM.from_pretrained(args.model_name)
    if len(tokenizer) != model.config.vocab_size:
        model.resize_token_embeddings(len(tokenizer))
    model.to(device).eval()

    out_path = Path(f"{args.output_prefix}_rank{rank}.jsonl")

    buffer_prompts, buffer_responses, buffer_entries = [], [], []
    with args.input_path.open("r", encoding="utf-8") as fin, \
         out_path.open("w", encoding="utf-8")   as fout:

        for idx, raw in enumerate(fin):
            line = raw.strip()
            if not line or idx % world_size != rank:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue

            buffer_prompts.append(entry.get("prompt", ""))
            buffer_responses.append(entry.get("responses", []))
            buffer_entries.append(entry)

            if len(buffer_prompts) >= args.batch_size:
                ppls_batch, tokencounts_batch = compute_metrics(
                    buffer_prompts, buffer_responses,
                    model, tokenizer, device,
                    args.max_length
                )
                # 写回原 entry，新增 "PPL" 与 "TokenCount"
                for orig_entry, ppls, tcounts in zip(buffer_entries, ppls_batch, tokencounts_batch):
                    orig_entry["PPL"] = ppls
                    orig_entry["TokenCount"] = tcounts
                    fout.write(json.dumps(orig_entry, ensure_ascii=False) + "\n")
                buffer_prompts.clear()
                buffer_responses.clear()
                buffer_entries.clear()

        # 处理剩余
        if buffer_prompts:
            ppls_batch, tokencounts_batch = compute_metrics(
                buffer_prompts, buffer_responses,
                model, tokenizer, device,
                args.max_length
            )
            for orig_entry, ppls, tcounts in zip(buffer_entries, ppls_batch, tokencounts_batch):
                orig_entry["PPL"] = ppls
                orig_entry["TokenCount"] = tcounts
                fout.write(json.dumps(orig_entry, ensure_ascii=False) + "\n")

    dist.barrier()
    dist.destroy_process_group()
    if rank == 0:
        print(f"All done. Outputs in {args.output_prefix}_rank*.jsonl")

