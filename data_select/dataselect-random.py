#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
随机抽取 top_n 条偏好对（不使用 PPL / TokenCount）
---------------------------------------------------
• 仅当 chosen_score - rejected_score ≥ score_gap 时保留该组合
• 在所有满足条件的组合中随机抽取 top_n 条
用法示例：
python random_pairs_no_ppl.py --input data.jsonl --output pref.jsonl --top_n 20 --score_gap 2
"""
import json
import argparse
import random
from itertools import combinations

def read_jsonl(path):
    with open(path, 'r', encoding='utf-8') as f:
        return [json.loads(line) for line in f]

def write_jsonl(data, path):
    with open(path, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

def build_pairs(entries, score_gap):
    """生成所有满足分差阈值的 (chosen, rejected) 组合"""
    results = []
    for entry in entries:
        prompt, resps, scores = entry['prompt'], entry['responses'], entry['scores']
        for i, j in combinations(range(len(resps)), 2):
            # 令 hi 为得分高、lo 为得分低
            hi, lo = (i, j) if scores[i] >= scores[j] else (j, i)
            if scores[hi] - scores[lo] < score_gap:
                continue
            results.append({
                "chosen":   f"{prompt} {resps[hi]}",
                "rejected": f"{prompt} {resps[lo]}"
            })
    return results

def main():
    ap = argparse.ArgumentParser(description="随机抽取偏好对（不依赖 PPL）")
    ap.add_argument('--input',  required=True, help='输入 JSONL 文件')
    ap.add_argument('--output', required=True, help='输出 JSONL 文件')
    ap.add_argument('--top_n',  type=int, default=5, help='随机保留条数 (默认 5)')
    ap.add_argument('--score_gap', type=float, default=2.0,
                    help='chosen_score 必须比 rejected_score 至少高多少分 (默认 2)')
    args = ap.parse_args()

    data   = read_jsonl(args.input)
    pairs  = build_pairs(data, args.score_gap)

    # 随机抽样；不足 top_n 时全部保留
    sampled = pairs if len(pairs) <= args.top_n else random.sample(pairs, k=args.top_n)
    write_jsonl(sampled, args.output)

if __name__ == '__main__':
    main()
