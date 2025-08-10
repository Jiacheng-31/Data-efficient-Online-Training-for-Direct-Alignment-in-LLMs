import json
from itertools import combinations
import argparse

def process_data(input_data, top_n):
    """
    处理数据，将数据按照 chosen_score - rejected_score 排序，返回前 top_n 条
    """
    data = []
    for entry in input_data:
        prompt = entry['prompt']
        responses = entry['responses']
        scores = entry['scores']

        # 遍历所有响应对
        for (i, j) in combinations(range(len(responses)), 2):
            chosen = responses[i]
            rejected = responses[j]
            chosen_score = scores[i]
            rejected_score = scores[j]

            # 高分为 chosen，低分为 rejected
            if chosen_score >= rejected_score:
                final_chosen = chosen
                final_rejected = rejected
                score_diff = chosen_score - rejected_score
            else:
                final_chosen = rejected
                final_rejected = chosen
                score_diff = rejected_score - chosen_score

            data.append({
                "chosen": f"{prompt}{final_chosen}",
                "rejected": f"{prompt}{final_rejected}",
                "score_diff": score_diff
            })

    # 根据分差排序
    sorted_data = sorted(data, key=lambda x: x['score_diff'], reverse=True)[:top_n]

    # 移除 score_diff 字段
    for item in sorted_data:
        item.pop('score_diff')

    return sorted_data

def read_jsonl(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))
    return data

def write_jsonl(data, file_path):
    with open(file_path, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

def main():
    parser = argparse.ArgumentParser(description='根据分数差计算 chosen 与 rejected 的对，并排序输出')
    parser.add_argument('--input', required=True, help='输入 JSONL 文件路径')
    parser.add_argument('--output', required=True, help='输出 JSONL 文件路径')
    parser.add_argument('--top_n', type=int, default=5, help='输出前 top_n 条')
    args = parser.parse_args()

    input_data = read_jsonl(args.input)
    results = process_data(input_data, args.top_n)
    write_jsonl(results, args.output)

if __name__ == '__main__':
    main()
