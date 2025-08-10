# python /fs-computility/llmit_d/shared/zhangchi/wjc/DPO/turn/turn_to.py

# python - << 'PYCODE'
# import torch
# ckpt = torch.load(
#     "/fs-computility/llmit_d/shared/zhangchi/wjc/DPO/direct-preference-optimization-main/.cache/root/anthropic_dpo_qwen2.5-0.5B_2025-04-18_03-53-18_964004/step-40000/policy.pt",
#     map_location="cpu"
# )
# print("Keys in policy.pt:", ckpt.keys() if isinstance(ckpt, dict) else type(ckpt))
# PYCODE

# python - << 'PYCODE'
# import torch
# ckpt = torch.load(
#     "/fs-computility/llmit_d/shared/zhangchi/wjc/DPO/direct-preference-optimization-main/.cache/root/anthropic_dpo_qwen2.5-0.5B_2025-04-18_03-53-18_964004/step-40000/policy.pt",
#     map_location="cpu"
# )
# print("Outer keys:", ckpt.keys())
# print("State keys:", ckpt["state"].keys() if isinstance(ckpt["state"], dict) else type(ckpt["state"]))
# PYCODE


# python turn_to.py \
#   --base_model /fs-computility/llmit_d/shared/zhangchi/wjc/Qwen3-1.7B \
#   --policy_ckpt /fs-computility/llmit_d/shared/zhangchi/wjc/DPO/direct-preference-optimization-main/.cache/root/qwen3-1.7b-test-1-sft/LATEST/policy.pt \
#   --out_dir /fs-computility/llmit_d/shared/zhangchi/wjc/DPO/turn/qwen3-1.7b-sft-test1

# python turn_to.py \
#   --base_model /fs-computility/llmit_d/shared/zhangchi/wjc/Qwen3-1.7B \
#   --policy_ckpt /fs-computility/llmit_d/shared/zhangchi/wjc/DPO/direct-preference-optimization-main/.cache/root/qwen3-1.7b-test-1-dpo/LATEST/policy.pt \
#   --out_dir /fs-computility/llmit_d/shared/zhangchi/wjc/DPO/turn/qwen3-1.7b-dpo-test1

# python turn_to.py \
#   --base_model /fs-computility/llmit_d/shared/zhangchi/wjc/Qwen3-4B \
#   --policy_ckpt /fs-computility/llmit_d/shared/zhangchi/wjc/DPO/direct-preference-optimization-main/.cache/root/qwen3-4b-test-1-sft/LATEST/policy.pt \
#   --out_dir /fs-computility/llmit_d/shared/zhangchi/wjc/DPO/turn/qwen3-4b-sft-test1

python turn_to.py \
  --base_model /fs-computility/llmit_d/shared/zhangchi/wjc/Qwen3-4B \
  --policy_ckpt /fs-computility/llmit_d/shared/zhangchi/wjc/DPO/direct-preference-optimization-main/.cache/root/qwen3-4b-test-1-dpo/LATEST/policy.pt \
  --out_dir /fs-computility/llmit_d/shared/zhangchi/wjc/DPO/turn/qwen3-4b-dpo-test1