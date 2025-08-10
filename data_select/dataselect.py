import json
from itertools import combinations
import argparse

def calculate_score(yw_ppl, yl_ppl, yw_token_count, yl_token_count):
    """
    计算每对响应的得分
    :param yw_ppl: 选择的响应的 PPL
    :param yl_ppl: 被拒绝的响应的 PPL
    :param yw_token_count: 选择的响应的 TokenCount
    :param yl_token_count: 被拒绝的响应的 TokenCount
    :return: 计算出的得分
    """
    return yw_token_count * yw_ppl - yl_token_count * yl_ppl

def process_data(input_data, top_n):
    """
    处理数据，将数据按照要求进行组合，计算分数并返回排序后的前n条数据
    :param input_data: 输入的原始数据
    :param top_n: 最终需要的top n条数据
    :return: 排序后的前n条数据
    """
    # 解析输入数据
    data = []
    for entry in input_data:
        prompt = entry['prompt']
        responses = entry['responses']
        scores = entry['scores']
        ppl = entry['PPL']
        token_count = entry['TokenCount']

        # 生成响应组合（根据responses的数量进行组合）
        for (i, j) in combinations(range(len(responses)), 2):
            # 计算每个组合的分数
            chosen = responses[i]
            rejected = responses[j]

            chosen_score = scores[i]
            rejected_score = scores[j]

            chosen_ppl = ppl[i]
            rejected_ppl = ppl[j]

            chosen_token_count = token_count[i]
            rejected_token_count = token_count[j]

            # 按照分数高的作为chosen，分数低的作为rejected
            if chosen_score >= rejected_score:
                final_chosen = chosen
                final_rejected = rejected
                final_chosen_ppl = chosen_ppl
                final_rejected_ppl = rejected_ppl
                final_chosen_token_count = chosen_token_count
                final_rejected_token_count = rejected_token_count
            else:
                final_chosen = rejected
                final_rejected = chosen
                final_chosen_ppl = rejected_ppl
                final_rejected_ppl = chosen_ppl
                final_chosen_token_count = rejected_token_count
                final_rejected_token_count = chosen_token_count

            # 计算分数
            score = calculate_score(final_chosen_ppl, final_rejected_ppl, 
                                  final_chosen_token_count, final_rejected_token_count)

            # 保存计算好的数据，按照你要求的格式，只保留chosen和rejected，且每个包含prompt和response
            data.append({
                "chosen": f"{prompt}{final_chosen}",
                "rejected": f"{prompt}{final_rejected}",
                "score": score  # 添加得分用于排序，写文件前再移除
            })

    # 根据得分排序
    sorted_data = sorted(data, key=lambda x: x['score'], reverse=True)[:top_n]

    # 移除 score 字段，仅保留指定格式
    for item in sorted_data:
        item.pop('score')
    
    # 返回前top_n条数据
    return sorted_data

def read_jsonl(file_path):
    """读取JSONL文件并返回解析后的数据列表"""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))
    return data

def write_jsonl(data, file_path):
    """将数据写入JSONL文件"""
    with open(file_path, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

def main():
    # 设置命令行参数解析
    parser = argparse.ArgumentParser(description='处理 JSONL 数据，计算响应对的得分，并按得分排序输出')
    parser.add_argument('--input', required=True, help='输入的 JSONL 文件路径')
    parser.add_argument('--output', required=True, help='输出的 JSONL 文件路径')
    parser.add_argument('--top_n', type=int, default=5, help='最终需要的 top n 条数据，默认为 5')
    args = parser.parse_args()

    # 读取输入文件
    input_data = read_jsonl(args.input)

    # 处理数据
    sorted_results = process_data(input_data, args.top_n)

    # 保存结果到输出文件
    write_jsonl(sorted_results, args.output)

if __name__ == '__main__':
    main()