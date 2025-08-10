import json, numpy as np, os
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import normalize

def save_clusters(texts, n_clusters, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    for i in range(n_clusters):
        group = [d for d in texts if d['cluster'] == i]
        with open(os.path.join(out_dir, f'cluster-{i}.jsonl'), 'w', encoding='utf-8') as f:
            for d in group:
                f.write(json.dumps(d) + '\n')

emb = np.load("embedding_data.npy")
emb_norm = normalize(emb)

# eps: 距离阈 ; min_samples: 同一稠密区最少点数
db = DBSCAN(eps=0., min_samples=5, metric="cosine").fit(emb_norm)
labels = db.labels_            # -1 标记噪声点
n_clusters = len(set(labels)) - (1 if -1 in labels else 0)

texts = []
with open("ultra-train-prompt.jsonl", "r", encoding="utf-8") as f:
    for i, line in enumerate(f):
        obj = json.loads(line)
        obj["cluster"] = int(labels[i])
        texts.append(obj)

save_clusters(texts, n_clusters, "dbscan-clusters")
print(f"DBSCAN 完成，共 {n_clusters} 个簇，噪声 {np.sum(labels==-1)} 条")
