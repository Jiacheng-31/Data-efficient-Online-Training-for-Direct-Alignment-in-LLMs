# #!/usr/bin/env bash
# set -euo pipefail

# # 日志文件（同目录下）
# LOG_FILE="cluster.log"

# # 清空旧日志
# : > "$LOG_FILE"

# # --------------- 参数配置 ---------------
export HTTPS_PROXY="http://100.68.170.107:3128"
export HTTP_PROXY="http://100.68.170.107:3128"
# N_CLUSTERS=800
# BATCH_SIZE=50000
# INPUT_JSONL="/fs-computility/llmit_d/shared/zhangchi/wjc/DPO/data_cluster/pre_data/merged.jsonl"
# # 注意：cluster.py 输出单个 JSON 对象，推荐用 .json 后缀，但如果你想要 JSONL 格式可保持 .jsonl
# OUTPUT_JSON="/fs-computility/llmit_d/shared/zhangchi/wjc/DPO/data_cluster/datacluster/cluster_${N_CLUSTERS}.json"

# # --------------- 开始 ---------------
# {
#   echo "[$(date '+%Y-%m-%d %H:%M:%S')] 开始聚类"
#   echo "  INPUT_JSONL : $INPUT_JSONL"
#   echo "  OUTPUT_JSON : $OUTPUT_JSON"
#   echo "  N_CLUSTERS  : $N_CLUSTERS"
#   echo "  BATCH_SIZE  : $BATCH_SIZE"
#   echo

#   # 执行 Python 聚类脚本
#   python cluster.py \
#     "$INPUT_JSONL" \
#     "$OUTPUT_JSON" \
#     --n_clusters "$N_CLUSTERS" \
#     --batch_size "$BATCH_SIZE"

#   echo
#   echo "[$(date '+%Y-%m-%d %H:%M:%S')] 聚类完成，结果已写入 $OUTPUT_JSON"
# } 2>&1 | tee -a "$LOG_FILE"

python cluster.py > cluster1.log 2>&1 &