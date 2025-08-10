import json
import random
import argparse

def sample_jsonl(input_file, output_file, sample_size, seed=42):
    # 读取全部数据
    with open(input_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    print(f"共读取 {len(lines)} 条样本")

    # 如果请求样本数量大于数据总量，进行裁剪提示
    if sample_size > len(lines):
        print(f"⚠️ 请求样本数 {sample_size} 超过总数据量 {len(lines)}，将仅返回全部数据")
        sample_size = len(lines)

    # 设置随机种子，确保可复现
    random.seed(seed)

    # 随机采样
    sampled_lines = random.sample(lines, sample_size)

    # 写入到新文件
    with open(output_file, 'w', encoding='utf-8') as f:
        for line in sampled_lines:
            f.write(line)

    print(f"✅ 已保存 {sample_size} 条样本到 {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", "-i", type=str, required=True, help="原始 JSONL 文件路径")
    parser.add_argument("--output", "-o", type=str, required=True, help="输出 JSONL 文件路径")
    parser.add_argument("--num", "-n", type=int, default=10000, help="采样条数")
    parser.add_argument("--seed", type=int, default=42, help="随机种子（可选）")
    args = parser.parse_args()

    sample_jsonl(args.input, args.output, args.num, args.seed)
