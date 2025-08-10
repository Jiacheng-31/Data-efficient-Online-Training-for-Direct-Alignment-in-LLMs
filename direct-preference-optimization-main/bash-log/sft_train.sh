# export CUDA_VISIBLE_DEVICES=1

# # 用 (…​) 开启子 shell，训练跑完之后打印一条提示  
# ( \
#   python -u train.py \
#     model=qwen2.5-7B \
#     datasets=[helpful-base] \
#     loss=sft \
#     exp_name=anthropic_sft_qwen2.5-7B \
#     gradient_accumulation_steps=2 \
#     batch_size=8 \
#     eval_batch_size=4 \
#     trainer=FSDPTrainer \
#     sample_during_eval=false \
#     model.fsdp_policy_mp=bfloat16 \
#     > sft_train_qwen_7B.log 2>&1 \
#   && echo "[$(date +'%Y-%m-%d\ %H:%M:%S')] 🎉 训练完成！看这里 11111111111111111111111111111111111111111111111111111111111111111111111111111111111111111" \
# ) &

# # 这时你会立即看到：
# # [1] 12345
# #   ↑ job number   ↑ PID

export CUDA_VISIBLE_DEVICES=0,1,2,3

# torchrun \
#   --nproc_per_node=4 \               # 2 个进程 → 2 张卡
#   --master_addr=127.0.0.1 \          # rendezvous 地址
#   --master_port=29500 \              # rendezvous 端口，必须所有进程一致
#   python train.py \
#     model=qwen3-1.7b \
#     datasets=[helpful-base] \
#     loss=sft \
#     exp_name=qwen3-1.7b-sft \
#     gradient_accumulation_steps=4 \
#     batch_size=16 \
#     eval_batch_size=8 \
#     trainer=FSDPTrainer \
#     sample_during_eval=false \
#     model.fsdp_policy_mp=bfloat16 \
#     > sft_train_qwen3-1.7b.log 2>&1 \
#     && echo "[$(date +'%Y-%m-%d\ %H:%M:%S')] 🎉 训练完成！看这里 11111111111111111111111111111111111111111111111111111111111111111111111111111111111111111"  &



# torchrun \
#     --nproc_per_node=4 \
#     --master_addr=127.0.0.1 \
#     --master_port=29501 \



    # python train.py \
    # model=qwen3-1.7b-base \
    # datasets=["/fs-computility/llmit_d/shared/zhangchi/wjc/DPO/dataset/temp/train.jsonl"] \
    # loss=sft \
    # exp_name=qwen3-1.7b-sft-test \
    # gradient_accumulation_steps=4 \
    # batch_size=16 \
    # eval_batch_size=8 \
    # trainer=FSDPTrainer \
    # sample_during_eval=false \
    # model.fsdp_policy_mp=bfloat16 \
    # > sft_train_qwen3-1.7b.log 2>&1 \


    python train.py \
  exp_name=dpo_qwen3_1.7b_test \
  local_dirs="['/fs-computility/llmit_d/shared/zhangchi/wjc/DPO/direct-preference-optimization-main/.cache']" \
  model=qwen3-1.7b-base \
  model.name_or_path=/fs-computility/llmit_d/shared/zhangchi/wjc/DPO/workspace/dpo-qwen3-1.7B \
  model.tokenizer_name_or_path=/fs-computility/llmit_d/shared/zhangchi/wjc/DPO/workspace/dpo-qwen3-1.7B \
  model.block_name=Qwen3DecoderLayer \
  model.policy_dtype=float32 \
  model.reference_dtype=float16 \
  trainer=FSDPTrainer \
  loss.name=sft \
  datasets="['/fs-computility/llmit_d/shared/zhangchi/wjc/DPO/dataset/temp/train.jsonl']" \
  batch_size=16 \
  eval_batch_size=8 \
  lr=1e-5 \
  n_epochs=3 \
  gradient_accumulation_steps=4 \
