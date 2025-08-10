#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
批量对 prompt 生成多条 response，并以 JSONL 格式保存结果

增加参数：
  --max-prompt-tokens  限制 prompt 的最大 token 数，超出时截断到前面部分
  --batch-size         每次大批量生成的 prompt 数量

依赖：
  pip install transformers torch tqdm protobuf

用法示例（run.sh 同理调用）：
  bash run.sh

也可以直接调用：
  python test_responses.py \
    --input extracted_prompts.jsonl \
    --output results_3.0.jsonl \
    --model /fs-computility/llmit_d/shared/zhangchi/wjc/Llama-3-8B-Instruct \
    --num-responses 2 \
    --temperature 0.7 \
    --top-p 0.9 \
    --max-prompt-tokens 200 \
    --batch-size 8
"""

import argparse
import json
import os
from tqdm import tqdm
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    logging as hf_logging
)

# 抑制 HF 警告
hf_logging.set_verbosity_error()

def load_prompts(input_path: str) -> list[str]:
    prompts = []
    with open(input_path, 'r', encoding='utf-8') as f:
        for line in f:
            obj = json.loads(line)
            prompts.append(obj['prompt'])
    return prompts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input',  '-i', required=True,
                        help='输入 JSONL，每行 {"prompt": "..."}')
    parser.add_argument('--output', '-o', required=True,
                        help='输出 JSONL，每行 {"prompt": "...", "responses": [...]}')
    parser.add_argument('--model',  '-m', required=True,
                        help='本地模型目录，如 /fs/.../Llama-3-8B-Instruct')
    parser.add_argument('--num-responses', type=int, default=2,
                        help='为每个 prompt 生成多少条 response')
    parser.add_argument('--max-length', type=int, default=None,
                        help='最大生成长度（含 prompt）；不传则 eos 停止')
    parser.add_argument('--temperature', type=float, default=0.7,
                        help='采样温度')
    parser.add_argument('--top-p', type=float, default=0.9,
                        help='nucleus 采样 top-p')
    parser.add_argument('--max-prompt-tokens', type=int, default=None,
                        help='限制 prompt 最大 token 数，超出时截断')
    parser.add_argument('--batch-size', type=int, default=8,
                        help='每次 batch 生成多少条 prompt')
    args = parser.parse_args()

    if not os.path.isdir(args.model):
        raise FileNotFoundError(f"找不到模型目录：{args.model}")

    print(f'加载 tokenizer 和模型：{args.model}')
    tokenizer = AutoTokenizer.from_pretrained(
        args.model, use_fast=True, trust_remote_code=False
    )
    # 如果 tokenizer 没有 pad_token，就将 eos_token 设为 pad_token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        device_map='auto',
        torch_dtype=torch.float16,
        trust_remote_code=True
    )
    model.eval()

    prompts = load_prompts(args.input)
    print(f'共 {len(prompts)} 条 prompt。')

    with open(args.output, 'w', encoding='utf-8') as fout:
        for start in tqdm(range(0, len(prompts), args.batch_size), desc='生成中'):
            batch_prompts = prompts[start:start + args.batch_size]

            tokenize_kwargs = dict(
                return_tensors='pt',
                padding=True,
                truncation=True if args.max_prompt_tokens is not None else False
            )
            if args.max_prompt_tokens is not None:
                tokenize_kwargs['max_length'] = args.max_prompt_tokens
            enc = tokenizer(batch_prompts, **tokenize_kwargs)
            enc = {k: v.to(model.device) for k, v in enc.items()}

            gen_kwargs = {
                "do_sample": True,
                "temperature": args.temperature,
                "top_p": args.top_p,
                "num_return_sequences": args.num_responses,
                "eos_token_id": tokenizer.eos_token_id,
                "pad_token_id": tokenizer.pad_token_id,
            }
            if args.max_length is not None:
                gen_kwargs["max_length"] = args.max_length

            with torch.no_grad():
                outputs = model.generate(**enc, **gen_kwargs)

            for idx in range(0, outputs.size(0), args.num_responses):
                seqs = outputs[idx: idx + args.num_responses]
                raw_prompt = batch_prompts[idx // args.num_responses]
                responses = []
                for seq in seqs:
                    text = tokenizer.decode(seq, skip_special_tokens=True)
                    if text.startswith(raw_prompt):
                        text = text[len(raw_prompt):].lstrip()
                    responses.append(text)

                fout.write(json.dumps({
                    "prompt": raw_prompt,
                    "responses": responses
                }, ensure_ascii=False) + "\n")

    print(f'完成，结果保存在：{args.output}')

if __name__ == '__main__':
    main()


