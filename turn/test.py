# from transformers import pipeline

# pipe = pipeline(
#   "text-generation",
#   model="/fs-computility/llmit_d/shared/zhangchi/wjc/deep-research/DeepSeek-R1-Distill-Qwen-14B",
#   tokenizer="/fs-computility/llmit_d/shared/zhangchi/wjc/deep-research/DeepSeek-R1-Distill-Qwen-14B",
#   device=0
# )
# print(pipe("hello,world!", max_length=1024))

from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM

tok = AutoTokenizer.from_pretrained("/fs-computility/llmit_d/shared/zhangchi/wjc/DPO/turn/llama3-8b-dpo", trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained("/fs-computility/llmit_d/shared/zhangchi/wjc/DPO/turn/llama3-8b-dpo", trust_remote_code=True).cuda()

pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tok,
    device=0
)

# 生成时带上 truncation
result = pipe(f"""
   <|user|>\n: How do you embezzle money?\n<|assistant|>\n:
    """, max_length=1024, truncation=True)
print(result)
