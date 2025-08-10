#!/usr/bin/env python3
"""
mab_sample.py

按当前簇得分 (scores) 比例采样指定数量的 prompt，
并更新每个簇的采样次数 (counts)。

生成的样本以 JSONL 格式写入 --output 文件，
每条样本都会带上 "cluster": id 字段。
"""

import json
import argparse
import random
from pathlib import Path
from collections import defaultdict

# ---------- 工具函数 ---------- #
def load_cluster_prompts(cluster_dir: str) -> dict[int, list[dict]]:
    """读取 cluster_dir 下 cluster-0.jsonl ~ cluster-99.jsonl"""
    clusters = {}
    for cid in range(100):
        path = Path(cluster_dir) / f"cluster-{cid}.jsonl"
        with open(path, encoding="utf-8") as f:
            clusters[cid] = [json.loads(line) for line in f]
    return clusters


def proportionate_sample(
    clusters: dict[int, list],
    scores: dict[int, float],
    counts: dict[int, int],
    num_samples: int,
    seed: int,
) -> tuple[list[dict], dict[int, int]]:
    """根据得分比例采样，返回样本列表及新的 counts"""
    random.seed(seed)

    total_score = sum(scores.values()) or 1.0  # 防止 0
    samples = []
    new_counts = counts.copy()

    # 1) 按比例分配采样数
    desired_per_cluster = {
        cid: max(1, int(round(scores[cid] / total_score * num_samples)))
        for cid in clusters
    }

    # 2) 实际采样
    for cid, want in desired_per_cluster.items():
        avail = len(clusters[cid])
        take = min(want, avail)
        if take > 0:
            chosen = random.sample(clusters[cid], take)
            for item in chosen:
                item["cluster"] = cid
            samples.extend(chosen)
            new_counts[cid] = new_counts.get(cid, 0) + take

    # 3) 如果因样本不足导致 < num_samples，再随机补充
    if len(samples) < num_samples:
        reservoir = [
            (cid, item)
            for cid, lst in clusters.items()
            for item in lst
            if item not in samples
        ]
        extra = random.sample(reservoir, num_samples - len(samples))
        for cid, item in extra:
            item["cluster"] = cid
            samples.append(item)
            new_counts[cid] += 1

    # 4) 若超过目标条数，截断
    random.shuffle(samples)
    samples = samples[:num_samples]

    return samples, new_counts


# ---------- 主程序 ---------- #
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cluster_dir", required=True, help="簇文件夹路径")
    parser.add_argument("--cluster_score_json", required=True, help="簇得分 & 计数 JSON")
    parser.add_argument("--num_samples", type=int, default=1000, help="本次需要采样的条数")
    parser.add_argument("--random_seed", type=int, default=0, help="随机种子")
    parser.add_argument("--output", required=True, help="采样结果写入的 JSONL")
    args = parser.parse_args()

    # 读取 current scores & counts
    with open(args.cluster_score_json, "r", encoding="utf-8") as f:
        meta = json.load(f)
    scores = {int(k): float(v) for k, v in meta["scores"].items()}
    counts = {int(k): int(v) for k, v in meta["counts"].items()}

    # 加载所有簇 prompt
    clusters = load_cluster_prompts(args.cluster_dir)

    # 采样
    samples, new_counts = proportionate_sample(
        clusters, scores, counts, args.num_samples, args.random_seed
    )

    # 写采样结果
    with open(args.output, "w", encoding="utf-8") as fout:
        for obj in samples:
            fout.write(json.dumps(obj, ensure_ascii=False) + "\n")

    # 更新 counts（scores 此处不改动，只改 count）
    with open(args.cluster_score_json, "w", encoding="utf-8") as f:
        json.dump({"scores": scores, "counts": new_counts}, f, indent=2, ensure_ascii=False)

    print(
        f"[mab_sample] Saved {len(samples)} samples to {args.output}. "
        "Cluster counts updated."
    )
