#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json

def jsonl_to_json(input_path: str, output_path: str):
    data = []
    # 1. 逐行读取 JSONL，并解析成 Python 对象
    with open(input_path, 'r', encoding='utf-8') as fin:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            data.append(json.loads(line))
    # 2. 将全部对象写成一个 JSON 数组
    with open(output_path, 'w', encoding='utf-8') as fout:
        json.dump(data, fout, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Convert JSONL to JSON array")
    parser.add_argument("input_jsonl", help="输入 JSONL 文件路径")
    parser.add_argument("output_json",  help="输出 JSON 文件路径")
    args = parser.parse_args()
    jsonl_to_json(args.input_jsonl, args.output_json)
    print(f"已转换 → {args.output_json}")
