import json
import argparse
from pathlib import Path
from itertools import combinations
from heapq import nlargest

def parse_args():
    parser = argparse.ArgumentParser(description="Select top-K preference pairs based on TokenCount * PPL difference")
    parser.add_argument("--input_path", type=Path, required=True, help="输入 JSONL 文件，包含 prompt, responses, PPL, TokenCount")
    parser.add_argument("--output_path", type=Path, required=True, help="输出偏好对 JSONL 文件路径")
    parser.add_argument("--top_k", type=int, default=1000, help="选择分差最大的前 k 个偏好对")
    return parser.parse_args()

def main():
    args = parse_args()
    pair_candidates = []

    with open(args.input_path, "r", encoding="utf-8") as fin:
        for entry in fin:
            obj = json.loads(entry)
            prompt = obj["prompt"]
            responses = obj["responses"]
            ppls = obj["PPL"]
            token_counts = obj["TokenCount"]

            n = len(responses)
            if n < 2:
                continue

            for i, j in combinations(range(n), 2):
                score_i = token_counts[i] * ppls[i]
                score_j = token_counts[j] * ppls[j]
                diff = abs(score_i - score_j)

                # 确定高分低分的偏好方向
                if score_i > score_j:
                    chosen, rejected = i, j
                else:
                    chosen, rejected = j, i

                pair_candidates.append({
                    "score_diff": diff,
                    "chosen_text": f"{prompt}{responses[chosen]}",
                    "rejected_text": f"{prompt}{responses[rejected]}"
                })

    # 选出 top-k
    top_k_pairs = nlargest(args.top_k, pair_candidates, key=lambda x: x["score_diff"])

    with open(args.output_path, "w", encoding="utf-8") as fout:
        for pair in top_k_pairs:
            fout.write(json.dumps({
                "chosen": pair["chosen_text"],
                "rejected": pair["rejected_text"]
            }, ensure_ascii=False) + "\n")

    print(f"✅ Saved {len(top_k_pairs)} preference pairs to {args.output_path}")

if __name__ == "__main__":
    main()
