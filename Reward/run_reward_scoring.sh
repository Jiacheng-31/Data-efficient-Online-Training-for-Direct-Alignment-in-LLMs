#!/usr/bin/env bash
# 用法：
#   bash run_score.sh 4 /path/to/model /path/to/input.jsonl /path/to/output.jsonl

set -e

if [[ $# -lt 4 ]]; then
  echo "Usage: bash $0 <NUM_GPUS> <MODEL_PATH> <INPUT_JSONL> <OUTPUT_JSONL>"
  exit 1
fi

NUM_GPUS=$1
MODEL=$2
INPUT=$3
OUTPUT=$4

torchrun --standalone --nproc_per_node=$NUM_GPUS \
  /fs-computility/llmit_d/shared/zhangchi/wjc/DPO/Reward/score_with_reward.py \
  --model_path "$MODEL" \
  --input_file "$INPUT" \
  --output_file "$OUTPUT"
