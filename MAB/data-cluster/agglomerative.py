import json, numpy as np, os
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import pairwise_distances
from sklearn.preprocessing import normalize

def save_clusters(texts, n_clusters, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    for i in range(n_clusters):
        group = [d for d in texts if d['cluster'] == i]
        with open(os.path.join(out_dir, f'cluster-{i}.jsonl'), 'w', encoding='utf-8') as f:
            for d in group:
                f.write(json.dumps(d) + '\n')

emb = np.load("embedding_data.npy")
# 建议先 L2 归一化，可提升余弦/欧氏效果
emb_norm = normalize(emb)

# 方法 1：指定类别数
n_clusters = 200
agg = AgglomerativeClustering(n_clusters=n_clusters,
                              metric="euclidean",    # 也可 "cosine"
                              linkage="ward")        # ward 需欧氏
labels = agg.fit_predict(emb_norm)

# 方法 2：指定距离阈值（不确定类别数）
# agg = AgglomerativeClustering(distance_threshold=1.5,
#                               n_clusters=None,
#                               metric="cosine", linkage="average")
# labels = agg.fit_predict(emb_norm)
# n_clusters = labels.max() + 1

texts = []
with open("ultra-train-prompt.jsonl", "r", encoding="utf-8") as f:
    for i, line in enumerate(f):
        obj = json.loads(line)
        obj["cluster"] = int(labels[i])
        texts.append(obj)

save_clusters(texts, n_clusters, "agglo-clusters")
print("层次聚类完成")
