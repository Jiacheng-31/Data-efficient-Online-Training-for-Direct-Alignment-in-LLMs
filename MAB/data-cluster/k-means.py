import json
import numpy as np
from sklearn.cluster import KMeans
import os

# 1. 加载嵌入数据
embedding_file = 'embedding_data_hh.npy'  # 读取嵌入数据的文件路径
embeddings = np.load(embedding_file)  # 加载嵌入向量

# 2. 聚类方法
n_clusters = 100  # 聚类数量，可以根据需要调整
kmeans = KMeans(n_clusters=n_clusters, random_state=42)

# 3. 进行K-means聚类
kmeans.fit(embeddings)

# 4. 读取原始数据，并将聚类标签添加到数据中
input_file = 'hh-prompt.jsonl'  # 输入文件路径
texts = []

with open(input_file, 'r', encoding='utf-8') as f:
    for i, line in enumerate(f):
        data = json.loads(line)
        data['cluster'] = int(kmeans.labels_[i])  # 使用聚类标签
        texts.append(data)

# 5. 将每个聚类的数据保存为不同的文件
output_dir = 'k-means-clusters-hh'  # 输出目录
os.makedirs(output_dir, exist_ok=True)

for i in range(n_clusters):
    cluster_data = [data for data in texts if data['cluster'] == i]
    cluster_file = os.path.join(output_dir, f'cluster-{i}.jsonl')
    with open(cluster_file, 'w', encoding='utf-8') as f:
        for data in cluster_data:
            f.write(json.dumps(data) + '\n')

print(f"聚类结果已保存至 {output_dir} 目录下。")
