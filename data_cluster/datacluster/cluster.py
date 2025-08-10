import json
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss

# 1. 加载原始记录
with open("/fs-computility/llmit_d/shared/zhangchi/wjc/DPO/data_cluster/pre_data/merged.jsonl", "r", encoding="utf-8") as f:
    records = [json.loads(line) for line in f]

# 2. 拼接 chosen 与 rejected
texts = [
    rec["chosen"].strip() + " [SEP] " + rec["rejected"].strip()
    for rec in records
]

# 3. 文本向量化
model = SentenceTransformer("all-MiniLM-L6-v2")
emb = model.encode(texts, batch_size=64, show_progress_bar=True)
emb = np.array(emb, dtype="float32")

# 4. 构建索引（CPU/GPU 兼容）
dim = emb.shape[1]
if hasattr(faiss, 'StandardGpuResources'):
    # GPU 环境
    res = faiss.StandardGpuResources()
    cfg = faiss.GpuIndexFlatConfig()
    cfg.device = 0
    index = faiss.GpuIndexFlatL2(res, dim, cfg)
else:
    # CPU 环境
    index = faiss.IndexFlatL2(dim)

index.add(emb)

# 5. K-Means 聚类
n_clusters = 500
clus = faiss.Clustering(dim, n_clusters)
clus.niter = 50
clus.verbose = True
clus.train(emb, index)
centroids = faiss.vector_to_array(clus.centroids).reshape(n_clusters, dim)

# 6. 检索聚类中心代表记录
_, I = index.search(centroids, 1)
cluster_centers = [records[i] for i in I.flatten()]

# 7. 保存结果为 JSONL
output_path = "cluster_centers.jsonl"
with open(output_path, "w", encoding="utf-8") as fout:
    for rec in cluster_centers:
        fout.write(json.dumps(rec, ensure_ascii=False) + "\n")

print(f"已将 {len(cluster_centers)} 条聚类中心结果保存到 `{output_path}`")
