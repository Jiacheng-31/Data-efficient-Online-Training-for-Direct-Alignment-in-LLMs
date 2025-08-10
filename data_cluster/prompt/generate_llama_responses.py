#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
批量对 prompt 生成多条 response，并以 JSONL 格式保存结果

增加参数：
  --max-prompt-tokens  限制 prompt 的最大 token 数，超出时截断到前面部分

依赖：
  pip install transformers torch tqdm protobuf

用法示例（run.sh 同理调用）：
  python generate_llama_responses.py \
    --input prompts.jsonl \
    --output results.jsonl \
    --model /fs/.../Llama-3-8B-Instruct \
    --num-responses 2 \
    --temperature 0.7 \
    --top-p 0.9 \
    --max-prompt-tokens 200
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


def generate_responses(
    model,
    tokenizer,
    prompt: str,
    num_responses: int,
    max_length: int | None,
    temperature: float,
    top_p: float
) -> list[str]:
    inputs = tokenizer(prompt, return_tensors='pt').to(model.device)
    gen_kwargs = {
        "do_sample": True,
        "temperature": temperature,
        "top_p": top_p,
        "num_return_sequences": num_responses,
        "eos_token_id": tokenizer.eos_token_id,
        "pad_token_id": tokenizer.eos_token_id,
    }
    if max_length is not None:
        gen_kwargs["max_length"] = max_length

    outputs = model.generate(**inputs, **gen_kwargs)
    results = []
    for seq in outputs:
        text = tokenizer.decode(seq, skip_special_tokens=True)
        if text.startswith(prompt):
            text = text[len(prompt):].lstrip()
        results.append(text)
    return results


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
    args = parser.parse_args()

    # 检查模型目录
    if not os.path.isdir(args.model):
        raise FileNotFoundError(f"找不到模型目录：{args.model}")

    # 加载 tokenizer 和模型
    print(f'加载 tokenizer 和模型：{args.model}')
    tokenizer = AutoTokenizer.from_pretrained(
        args.model, use_fast=True, trust_remote_code=False
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        device_map='auto',
        torch_dtype=torch.float16,
        trust_remote_code=True
    )
    model.eval()

    # 读取 prompts
    prompts = load_prompts(args.input)
    print(f'共 {len(prompts)} 条 prompt。')

    with open(args.output, 'w', encoding='utf-8') as fout:
        for raw_prompt in tqdm(prompts, desc='生成中'):
            prompt = raw_prompt
            # 限制 prompt token 长度
            if args.max_prompt_tokens is not None:
                enc = tokenizer(prompt, return_tensors='pt')
                tokens = enc['input_ids'][0]
                if tokens.size(0) > args.max_prompt_tokens:
                    truncated = tokens[:args.max_prompt_tokens]
                    prompt = tokenizer.decode(truncated, skip_special_tokens=True)

            responses = generate_responses(
                model, tokenizer, prompt,
                num_responses=args.num_responses,
                max_length=args.max_length,
                temperature=args.temperature,
                top_p=args.top_p
            )
            fout.write(json.dumps({
                "prompt": prompt,
                "responses": responses
            }, ensure_ascii=False))
            fout.write('\n')

    print(f'完成，结果保存在：{args.output}')

if __name__ == '__main__':
    main()
