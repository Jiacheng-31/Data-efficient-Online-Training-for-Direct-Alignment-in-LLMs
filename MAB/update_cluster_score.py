#!/usr/bin/env python3
import json
import argparse
import math
from collections import defaultdict

def calc_hpp(yw_ppl, yl_ppl, yw_tk, yl_tk):
    """计算 Human Preference Perception 分数"""
    return yw_tk * yw_ppl - yl_tk * yl_ppl

def min_max_norm(xlist):
    """对 HPP 列表进行 Min-Max 归一化"""
    mn, mx = min(xlist), max(xlist)
    if mx == mn:
        return [0.0] * len(xlist)
    return [(x - mn) / (mx - mn) for x in xlist]

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", required=True, help="输入的 JSONL 文件，包含 prompt 和 response 信息")
    parser.add_argument("--cluster_score_json_in", required=True, help="输入：当前簇的得分和采样次数")
    parser.add_argument("--cluster_score_json_out", required=True, help="输出：更新后的簇得分文件")
    parser.add_argument("--selected_out", required=True, help="追加写入的完整样本保存路径")
    args = parser.parse_args()

    # Step 1: 读取 response 数据（包含 prompt、responses、PPL、TokenCount、scores 等）
    with open(args.data_path, "r", encoding="utf-8") as f:
        data = [json.loads(l) for l in f]

    hpps = []
    clusters = []
    records = []

    for d in data:
        scores = d.get("scores", [])
        if len(scores) < 2:
            continue  # 需要至少两个 response 才能构造偏好对

        # 找出得分最高和最低的 response
        yw = scores.index(max(scores))
        yl = scores.index(min(scores))

        # 计算 HPP 值
        h = calc_hpp(d["PPL"][yw], d["PPL"][yl], d["TokenCount"][yw], d["TokenCount"][yl])
        hpps.append(h)

        # 记录 cluster（如不存在，设为 -1）
        cid = d.get("cluster", -1)
        clusters.append(cid)

        # 添加 cluster 字段并保存原始数据
        d["cluster"] = cid
        records.append(d)

    # Step 2: 对所有 HPP 分数归一化
    hnorm = min_max_norm(hpps)

    # Step 3: 聚簇计算平均归一化 HPP
    per_cluster_hpp = defaultdict(list)
    for hpp, cid in zip(hnorm, clusters):
        per_cluster_hpp[cid].append(hpp)

    # Step 4: 读取当前簇得分和采样次数
    with open(args.cluster_score_json_in, "r", encoding="utf-8") as f:
        meta = json.load(f)
    scores = {int(k): float(v) for k, v in meta["scores"].items()}
    counts = {int(k): int(v) for k, v in meta["counts"].items()}

    # Step 5: 更新簇得分（UCB）
    total_count = sum(counts.values()) if sum(counts.values()) > 0 else 1
    alpha = 1.0  # UCB 探索因子
    updated_scores = {}

    for cid in range(100):  # 假设簇编号为 0~99
        cluster_hpps = per_cluster_hpp.get(cid, [])
        mean_hpp = sum(cluster_hpps) / len(cluster_hpps) if cluster_hpps else 0.0
        n = counts.get(cid, 1)  # 防止除零
        bonus = alpha * math.sqrt(2 * math.log(total_count + 1) / n)
        updated_scores[cid] = mean_hpp + bonus

    # Step 6: 保存新的簇得分和采样次数
    with open(args.cluster_score_json_out, "w", encoding="utf-8") as f:
        json.dump({"scores": updated_scores, "counts": counts}, f, indent=2, ensure_ascii=False)

    # Step 7: 将完整样本写入偏好对保存文件
    with open(args.selected_out, "a", encoding="utf-8") as fout:
        for r in records:
            fout.write(json.dumps(r, ensure_ascii=False) + "\n")
