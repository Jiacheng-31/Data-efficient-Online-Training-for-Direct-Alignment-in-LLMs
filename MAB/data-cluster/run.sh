export CUDA_VISIBLE_DEVICES=0,1

# python encoder.py > log1.log 2>&1 &

python k-means.py > log2.log 2>&1 &

# python /fs-computility/llmit_d/shared/zhangchi/wjc/DPO/data_cluster/prompt/exact_prompt.py --input /fs-computility/llmit_d/shared/zhangchi/wjc/DPO/MAB/data-cluster/hh-train.jsonl --output hh-prompt.jsonl