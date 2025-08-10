#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
处理 JSONL 数据：
1. 仅当 chosen_score - rejected_score >= score_gap 才视为有效偏好对
2. 计算基于 PPL 与 TokenCount 的 score
3. 输出得分最高的 top_n 条数据（移除中间 score 字段）
用法示例：
python script.py --input data.jsonl --output result.jsonl --top_n 5 --score_gap 2
"""

import json
import argparse
from itertools import combinations

def calculate_score(yw_ppl, yl_ppl, yw_token_count, yl_token_count):
    """根据公式计算得分"""
    return yw_token_count * yw_ppl - yl_token_count * yl_ppl

def read_jsonl(file_path):
    """读取 JSONL 文件"""
    with open(file_path, 'r', encoding='utf-8') as f:
        return [json.loads(line) for line in f]

def write_jsonl(data, file_path):
    """写入 JSONL 文件"""
    with open(file_path, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

def process_data(input_data, top_n, score_gap):
    """核心处理函数"""
    pairs = []

    for entry in input_data:
        prompt        = entry['prompt']
        responses     = entry['responses']
        scores        = entry['scores']
        ppl_values    = entry['PPL']
        token_counts  = entry['TokenCount']

        # 枚举两两组合
        for i, j in combinations(range(len(responses)), 2):
            # 默认 i 为 chosen，j 为 rejected；如分数更低则交换
            if scores[i] >= scores[j]:
                ch_idx, re_idx = i, j
            else:
                ch_idx, re_idx = j, i

            # 仅当分数差 ≥ score_gap 时才纳入
            if scores[ch_idx] - scores[re_idx] < score_gap:
                continue

            # 计算自定义得分
            pair_score = calculate_score(
                ppl_values[ch_idx], ppl_values[re_idx],
                token_counts[ch_idx], token_counts[re_idx]
            )

            # 组织输出格式
            pairs.append({
                "chosen":   f"{prompt}{responses[ch_idx]}",
                "rejected": f"{prompt}{responses[re_idx]}",
                "score": pair_score  # 仅用于后续排序
            })

    # 选取得分最高的 top_n
    sorted_pairs = sorted(pairs, key=lambda x: x['score'], reverse=True)[:top_n]
    # 去掉 score 字段
    for item in sorted_pairs:
        item.pop('score', None)

    return sorted_pairs

def main():
    parser = argparse.ArgumentParser(
        description='根据分数差筛选偏好对并输出得分最高的前 N 条记录'
    )
    parser.add_argument('--input',  required=True, help='输入 JSONL 路径')
    parser.add_argument('--output', required=True, help='输出 JSONL 路径')
    parser.add_argument('--top_n',  type=int, default=5, help='保留前 N 条记录 (默认 5)')
    parser.add_argument('--score_gap', type=float, default=2.0,
                        help='chosen_score 必须至少高于 rejected_score 的差值 (默认 2)')
    args = parser.parse_args()

    raw_data = read_jsonl(args.input)
    result   = process_data(raw_data, args.top_n, args.score_gap)
    write_jsonl(result, args.output)

if __name__ == '__main__':
    main()
