#!/bin/bash
set -euo pipefail
export NCCL_BLOCKING_WAIT=1
export NCCL_TIMEOUT=10000  # 超时时间设置为1000秒，适应慢进程
export FLASH_ATTENTION_FORCE_DISABLED=1


export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
NGPU=8

BASE_FILE="/fs-computility/llmit_d/shared/zhangchi/wjc/DPO/dataset/hh-train.jsonl"


DecoderLayer=Qwen3DecoderLayer
MODEL_NAME=qwen3-4b
EXP_NAME="$MODEL_NAME-dataset-hh-MAB-dpo-start-t0.7-0.5-ref-sft"
SAMPLE_NUM=1000
SAMPLE_TIMES=12
TOP_N=10000
MODEL_LOR="/fs-computility/llmit_d/shared/zhangchi/wjc/Qwen3-4B"
NUM_RESPONSE=3
REWARDMODEL="/fs-computility/llmit_d/shared/zhangchi/wjc/llama3-8b-reward"

BASE_WORKSPACE="/fs-computility/llmit_d/shared/zhangchi/wjc/DPO/MAB/$EXP_NAME"
mkdir -p $BASE_WORKSPACE
cp -r /fs-computility/llmit_d/shared/zhangchi/wjc/DPO/turn/qwen3-4b-dpo-test1 $BASE_WORKSPACE
REF_MODEL_LOR="/fs-computility/llmit_d/shared/zhangchi/wjc/DPO/turn/qwen3-4b-sft-test1"
EXP_MODEL_LOR="$BASE_WORKSPACE/qwen3-4b-dpo-test1"


CLUSTER_DIR="/fs-computility/llmit_d/shared/zhangchi/wjc/DPO/MAB/k-means-clusters-hh"
CLUSTER_SCORE_JSON="$BASE_WORKSPACE/cluster_score.json"


# 初始化簇分数（所有 100 个簇初始 score=1, count=1）
python <<EOF
import json
scores = {i: 1.0 for i in range(100)}
counts = {i: 0 for i in range(100)}
json.dump({"scores": scores, "counts": counts}, open("$CLUSTER_SCORE_JSON", "w"))
EOF


for round in {1..3}; do
  echo "🏁 Round $round: 开始 MAB 采样 + 12 次迭代踩点"

  WORK_DIR="$BASE_WORKSPACE/round-${round}"
  mkdir -p "$WORK_DIR"
  SELECTED_PRE_PATH="$WORK_DIR/selected-preferences.jsonl"
  > "$SELECTED_PRE_PATH"

  # 每轮采样 12 次，每次生成 response、PPL、更新分数
  for iter in {1..12}; do
    SEED=$((RANDOM))
    echo "[$(date +'%F %T')] [Round $round][$iter/12] Step‑0: cluster_score 初始化"
    cp "$CLUSTER_SCORE_JSON" "$WORK_DIR/cluster_score_iter-${iter}-in.json"

    echo "[$(date +'%F %T')] [Step 1] 采样 1000 个 prompt 输入"
    python /fs-computility/llmit_d/shared/zhangchi/wjc/DPO/MAB/mab_sample.py \
      --cluster_dir "$CLUSTER_DIR" \
      --cluster_score_json "$CLUSTER_SCORE_JSON" \
      --num_samples $SAMPLE_NUM \
      --random_seed $SEED \
      --output "$WORK_DIR/prompt_r${round}_i${iter}.jsonl"

    echo "[$(date +'%F %T')] [Step 2] 生成 response"
    torchrun --nproc_per_node=$NGPU /fs-computility/llmit_d/shared/zhangchi/wjc/DPO/data_cluster/prompt/torch_run_model.py \
      --input "$WORK_DIR/prompt_r${round}_i${iter}.jsonl" \
      --output "$WORK_DIR/resp_r${round}_i${iter}.jsonl" \
      --model "$EXP_MODEL_LOR" \
      --num-responses $NUM_RESPONSE \
      --temperature 0.7 \
      --top-p 0.9 \
      --max-prompt-tokens 256 \
      --max-length 1024 \
      --batch-size 8
    echo "[$(date +'%F %T')] [Step 2] 生成response完成"

    echo "[$(date +'%F %T')] [Step 3] 计算 reward 得分"
    bash /fs-computility/llmit_d/shared/zhangchi/wjc/DPO/Reward/run_reward_scoring.sh \
      $NGPU $REWARDMODEL "$WORK_DIR/resp_r${round}_i${iter}.jsonl" \
      "$WORK_DIR/reward_r${round}_i${iter}.jsonl"

    echo "[$(date +'%F %T')] [Step 4] 计算 PPL 和 TokenCount"
    torchrun --nproc_per_node=$NGPU /fs-computility/llmit_d/shared/zhangchi/wjc/DPO/data_select/PPL-NEW.py \
      --model_name "$EXP_MODEL_LOR" \
      --input_path "$WORK_DIR/reward_r${round}_i${iter}.jsonl" \
      --output_path "$WORK_DIR/data_pre_r${round}_i${iter}.jsonl" \
      --batch_size 64 \
      --max_length 1200
    echo "[$(date +'%F %T')] [Step 4] 计算PPL和Token完成"

    echo "[$(date +'%F %T')] [Step 5] 构建偏好对 & 更新簇得分"
    python /fs-computility/llmit_d/shared/zhangchi/wjc/DPO/MAB/update_cluster_score.py \
      --data_path "$WORK_DIR/data_pre_r${round}_i${iter}.jsonl" \
      --cluster_score_json_in "$CLUSTER_SCORE_JSON" \
      --cluster_score_json_out "$CLUSTER_SCORE_JSON" \
      --selected_out "$SELECTED_PRE_PATH"

    echo "[$(date +'%F %T')] [Step 6] 完成第 ${iter} 次采样"

  done  # iter 12 次

  echo "✅ Round $round 完成共 12 000 条偏好对采样，开始训练模型"

  # （选项1）训练前还可以做一次随机数据选择或清洗
  # 这里我们假设 SELECTED_PRE_PATH 已经累积了 12 000 条偏好对
  echo "[$(date +'%F %T')] [Step 7] 数据选择: 输入=$SELECTED_PRE_PATH 输出=$WORK_DIR/train.jsonl"
    python /fs-computility/llmit_d/shared/zhangchi/wjc/DPO/data_select/dataselect-random.py \
    --input "$SELECTED_PRE_PATH" \
    --output "$WORK_DIR/train.jsonl" \
    --top_n $TOP_N \
    --score_gap 3
    echo "[$(date +'%F %T')] [Step 7] 数据选择完成"

  # 进入DPO主目录
  DPO_LOG=$WORK_DIR/$EXP_NAME-dpo.log

  cd /fs-computility/llmit_d/shared/zhangchi/wjc/DPO/direct-preference-optimization-main
  echo "[$(date +'%F %T')] [Step ] 进入DPO主目录:"
  mkdir -p $WORK_DIR/.cache


  # DPO训练
  echo "[$(date +'%F %T')] [Step 10] 开始DPO训练: 日志输出到 $DPO_LOG"
  python ref-train.py \
  local_dirs="['$WORK_DIR/.cache']" \
  model=$MODEL_NAME \
  model.name_or_path=$EXP_MODEL_LOR \
  model.reference_name_or_path=$REF_MODEL_LOR \
  model.tokenizer_name_or_path=$EXP_MODEL_LOR \
  model.block_name=$DecoderLayer \
  model.policy_dtype=bfloat16 \
  model.reference_dtype=bfloat16 \
  trainer=FSDPTrainer \
  datasets=["$WORK_DIR/train.jsonl"] \
  loss=dpo \
  loss.beta=0.1 \
  exp_name=$EXP_NAME-$round-dpo \
  gradient_accumulation_steps=16 \
  lr=5e-6 \
  batch_size=128 \
  n_epochs=1 \
  eval_batch_size=8 \
  sample_during_eval=false \
  model.fsdp_policy_mp=bfloat16 \
  > $DPO_LOG 2>&1
  echo "[$(date +'%F %T')] 🎉 DPO训练完成"

  DPO_PT=$WORK_DIR/.cache/root/$EXP_NAME-$round-dpo/LATEST/policy.pt
  DPO_SAVE=$WORK_DIR/$EXP_NAME-$round-dpo

  # 权重转换
  echo "[$(date +'%F %T')] [Step 11/$round] DPO权重转换: 输出到 $DPO_SAVE"
  python /fs-computility/llmit_d/shared/zhangchi/wjc/DPO/turn/turn_to.py \
    --base_model $MODEL_LOR \
    --policy_ckpt $DPO_PT \
    --out_dir $DPO_SAVE
  echo "[$(date +'%F %T')] [Step 11/$round] DPO权重转换完成"

  # 权重拷贝（先删再拷，保证完全覆盖）
  echo "[$(date +'%F %T')] [Step 12/$round] 拷贝权重到 EXP_MODEL_LOR: $EXP_MODEL_LOR"
  rm -rf $EXP_MODEL_LOR
  cp -r $DPO_SAVE $EXP_MODEL_LOR
  echo "[$(date +'%F %T')] [Step 12/$round] 权重拷贝完成"

  echo "[$(date +'%F %T')] [Step END/$round] 第 $round 轮实验全部完成"

done

echo "🏁 全部三轮完成，总共生成 ${EXP_NAME}‐round[1‑3]"
echo "[$(date +'%F %T')]"
