import json
import argparse
from collections import OrderedDict

def extract_anthropic_prompt(prompt_and_response):
    """Extract the anthropic prompt from a prompt and response pair."""
    search_term = '\n\nAssistant:'
    search_term_idx = prompt_and_response.rfind(search_term)
    assert search_term_idx != -1, f"Prompt and response does not contain '{search_term}'"
    return prompt_and_response[:search_term_idx + len(search_term)]

def extract_prompts_to_jsonl(input_file: str, output_file: str):
    """
    从 JSONL 文件中提取 prompt，按首次出现顺序去重，
    并将它们以 JSONL 格式保存到 output_file。
    """
    # OrderedDict 用于记住插入顺序并去重
    prompts = OrderedDict()

    # 1. 读取原始 JSONL，每行取 row['chosen'] 提取 prompt
    with open(input_file, 'r', encoding='utf-8') as fin:
        for line in fin:
            row = json.loads(line)
            prompt = extract_anthropic_prompt(row['chosen'])
            prompts[prompt] = None

    # 2. 将所有唯一 prompt 写入新的 JSONL 文件
    with open(output_file, 'w', encoding='utf-8') as fout:
        for prompt in prompts:
            fout.write(json.dumps({"prompt": prompt}, ensure_ascii=False))
            fout.write('\n')
    print(f'已提取并保存 {len(prompts)} 条 prompt 到 {output_file}')

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="Extract prompts from a JSONL file and save to another JSONL file")
    parser.add_argument('--input', '-i', required=True, help='输入的 JSONL 文件路径')
    parser.add_argument('--output', '-o', required=True, help='输出的 JSONL 文件路径')
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()

    extract_prompts_to_jsonl(args.input, args.output)
