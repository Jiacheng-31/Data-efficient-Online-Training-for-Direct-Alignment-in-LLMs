export CUDA_VISIBLE_DEVICES=1,2,3

#!/usr/bin/env bash
set -euo pipefail

NGPU=3
MODEL="/fs-computility/llmit_d/shared/zhangchi/wjc/qwen3-1.7-reward"
IN="/fs-computility/llmit_d/shared/zhangchi/wjc/DPO/data_cluster/prompt/results_4.0.jsonl"
OUT="/fs-computility/llmit_d/shared/zhangchi/wjc/DPO/Reward/scored.jsonl"

# 若机器没装 flash-attn，可继续关闭
export FLASH_ATTENTION_FORCE_DISABLED=1

bash run_reward_scoring.sh $NGPU $MODEL $IN $OUT
