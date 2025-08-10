

export CUDA_VISIBLE_DEVICES=0,1,2,3
# torchrun \
#   --nproc_per_node=4 \
#   --master_addr=127.0.0.1 \
#   --master_port=29500 \
#   PPL.py \
#     --model_name /fs-computility/llmit_d/shared/zhangchi/wjc/Llama-3-8B-Instruct \
#     --input_path /fs-computility/llmit_d/shared/zhangchi/wjc/DPO/reward-model/predictions-1.jsonl \
#     --output_prefix predictions \
#     --batch_size 8 \
#     --max_length 1024

# torchrun --nproc_per_node=4 PPL.py  \
#     --model_name /fs-computility/llmit_d/shared/zhangchi/wjc/Llama-3-8B-Instruct \
#     --input_path /fs-computility/llmit_d/shared/zhangchi/wjc/DPO/reward-model/predictions.jsonl \
#     --output_path output_file.jsonl \
#     --batch_size 4 \
#     --max_length 1024

torchrun --nproc_per_node=4 PPL-NEW.py \
    --model_name "/fs-computility/llmit_d/shared/zhangchi/wjc/Llama-3-8B-Instruct" \
    --input_path "/fs-computility/llmit_d/shared/zhangchi/wjc/DPO/reward-model/predictions-1.jsonl" \
    --output_path "output.jsonl" \
    --batch_size 64 \
    --max_length 1024
