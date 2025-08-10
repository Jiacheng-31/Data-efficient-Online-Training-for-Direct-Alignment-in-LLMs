export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

EXP_NAME=qwen3-4b-test-1


python train.py \
    local_dirs="['/fs-computility/llmit_d/shared/zhangchi/wjc/DPO/direct-preference-optimization-main/.cache']" \
    model=qwen3-4b \
    model.name_or_path=/fs-computility/llmit_d/shared/zhangchi/wjc/Qwen3-4B \
    model.tokenizer_name_or_path=/fs-computility/llmit_d/shared/zhangchi/wjc/Qwen3-4B \
    model.block_name=Qwen3DecoderLayer \
    model.policy_dtype=float32 \
    model.reference_dtype=float16 \
    trainer=FSDPTrainer \
    datasets=["/fs-computility/llmit_d/shared/zhangchi/wjc/DPO/dataset/ultra-train-dataset.jsonl"] \
    loss=sft \
    exp_name=$EXP_NAME-sft \
    lr=5e-6 \
    gradient_accumulation_steps=8 \
    batch_size=128 \
    n_epochs=1 \
    eval_batch_size=8 \
    sample_during_eval=false \
    model.fsdp_policy_mp=bfloat16 \
    > qwen3-4b-sft-test-1.log 2>&1
    echo "[$(date +'%F %T')] 🎉 SFT训练完成"





    CHECKPOINT=/fs-computility/llmit_d/shared/zhangchi/wjc/DPO/direct-preference-optimization-main/.cache/root/$EXP_NAME-sft/LATEST/policy.pt

    # DPO训练
    python ref-train.py \
    local_dirs="['/fs-computility/llmit_d/shared/zhangchi/wjc/DPO/direct-preference-optimization-main/.cache']" \
    model=qwen3-4b \
    model.name_or_path=/fs-computility/llmit_d/shared/zhangchi/wjc/Qwen3-4B \
    model.tokenizer_name_or_path=/fs-computility/llmit_d/shared/zhangchi/wjc/Qwen3-4B \
    model.block_name=Qwen3DecoderLayer \
    model.policy_dtype=bfloat16 \
    model.reference_dtype=bfloat16 \
    trainer=FSDPTrainer \
    datasets=["/fs-computility/llmit_d/shared/zhangchi/wjc/DPO/dataset/ultra-train-dataset.jsonl"] \
    loss=dpo \
    loss.beta=0.1 \
    model.archive=$CHECKPOINT \
    exp_name=$EXP_NAME-dpo \
    gradient_accumulation_steps=16 \
    lr=5e-6 \
    batch_size=128 \
    n_epochs=1 \
    eval_batch_size=16 \
    sample_during_eval=false \
    model.fsdp_policy_mp=bfloat16 \
    >  qwen3-4b-dpo-test-1.log 2>&1
    echo "[$(date +'%F %T')] 🎉 DPO训练完成"