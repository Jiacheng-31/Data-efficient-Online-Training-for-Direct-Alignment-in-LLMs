# export CUDA_VISIBLE_DEVICES=0,1,2,3


# python generate_llama_responses.py \
#     --input extracted_prompts.jsonl \
#     --output results_2.0.jsonl \
#     --model /fs-computility/llmit_d/shared/zhangchi/wjc/Llama-3-8B-Instruct \
#     --num-responses 2 \
#     --temperature 0.7 \
#     --top-p 0.9 \
#     --max-prompt-tokens 200 \
#     --max-length 600 \
#     > responses_2.0.log 2>&1 &

# run.sh 示例：
# ----------------
#!/usr/bin/env bash
# 批量生成 prompts 的响应
# python test_responses.py \
#   --input extracted_prompts.jsonl \
#   --output results_3.0.jsonl \
#   --model /fs-computility/llmit_d/shared/zhangchi/wjc/Llama-3-8B-Instruct \
#   --num-responses 5 \
#   --temperature 0.7 \
#   --top-p 0.9 \
#   --max-prompt-tokens 200 \
#   --max-length 600 \
#   --batch-size 8


export CUDA_VISIBLE_DEVICES=0,1,2,3 
torchrun --nproc_per_node=4 torch_run_model.py \
    --input extracted_prompts.jsonl \
    --output results_4.0.jsonl \
    --model /fs-computility/llmit_d/shared/zhangchi/wjc/Llama-3-8B-Instruct \
    --num-responses 2 \
    --temperature 0.7 \
    --top-p 0.9 \
    --max-prompt-tokens 200 \
    --max-length 600 \
    --batch-size 8