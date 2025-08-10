CHECKPOINT=/fs-computility/llmit_d/shared/zhangchi/wjc/DPO/direct-preference-optimization-main/.cache/root/qwen3-1.7b-sft_2025-06-25_12-55-19_420106/LATEST/policy.pt


# export CUDA_VISIBLE_DEVICES=1

# ( \
#   python -u train.py \
#     model=qwen2.5-0.5B \
#     datasets=[helpful-base] \
#     loss=dpo \
#     loss.beta=0.1 \
#     model.archive=$CHECKPOINT \
#     exp_name=anthropic_dpo_qwen2.5-0.5B \
#     gradient_accumulation_steps=2 \
#     batch_size=32 \
#     eval_batch_size=16 \
#     trainer=FSDPTrainer \
#     sample_during_eval=false \
#     model.fsdp_policy_mp=bfloat16 \
#     > dpo_train.log 2>&1 \
#   && echo "[$(date +'%Y-%m-%d\ %H:%M:%S')] 🎉 DPO 训练完成！" \
# ) &


export CUDA_VISIBLE_DEVICES=4,5,6,7

torchrun \
  --nproc_per_node=4 \
  --master_addr=127.0.0.1 \
  --master_port=29500 \
  train.py \
    model=qwen3-1.7b \
    datasets=[helpful-base] \
    loss=dpo \
    loss.beta=0.1 \
    model.archive=$CHECKPOINT \
    exp_name=qwen3-1.7b-dpo \
    gradient_accumulation_steps=4 \
    batch_size=16 \
    eval_batch_size=8 \
    trainer=FSDPTrainer \
    sample_during_eval=false \
    model.fsdp_policy_mp=bfloat16 \
    > dpo-train-qwen3-1.7b.log 2>&1 \
    && echo "[$(date +'%Y-%m-%d\ %H:%M:%S')] 🎉 训练完成！看这里 11111111111111111111111111111111111111111111111111111111111111111111111111111111111111111"  &
