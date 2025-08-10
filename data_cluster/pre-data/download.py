import os
from datasets import load_dataset

# 1. 下载并加载数据集
# ds = load_dataset("Anthropic/hh-rlhf")
ds = load_dataset("openbmb/UltraFeedback")
# ds = load_dataset("neulab/tldr", "data")

# 2. 创建输出目录
output_dir = "UltraFeedback"
os.makedirs(output_dir, exist_ok=True)

# 3. 遍历所有 split，导出为 JSONL
for split, subset in ds.items():
    out_file = os.path.join(output_dir, f"{split}.jsonl")
    # lines=True 会把每条记录写成一行；force_ascii=False 保留非 ASCII 字符
    subset.to_json(out_file, lines=True, force_ascii=False)
    print(f"Saved split '{split}' → {out_file}")
