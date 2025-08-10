import argparse
import json
import os
import torch
from tqdm import tqdm
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    logging as hf_logging,
)
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader

# 抑制 HF 警告
hf_logging.set_verbosity_error()


def load_prompts(input_path: str) -> list[str]:
    prompts = []
    with open(input_path, 'r', encoding='utf-8') as f:
        for line in f:
            obj = json.loads(line)
            prompts.append(obj['prompt'])
    return prompts


class PromptDataset(Dataset):
    def __init__(self, prompts):
        self.prompts = prompts

    def __len__(self):
        return len(self.prompts)

    def __getitem__(self, idx):
        return self.prompts[idx]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input',  '-i', required=True,
                        help='输入 JSONL，每行 {"prompt": "..."}')
    parser.add_argument('--output', '-o', required=True,
                        help='输出 JSONL，每行 {"prompt": "...", "responses": [...], "prompt_length": ..., "response_lengths": [...]}')
    parser.add_argument('--model',  '-m', required=True,
                        help='本地模型目录，如 /fs/.../Llama-3-8B-Instruct')
    parser.add_argument('--num-responses', type=int, default=2,
                        help='为每个 prompt 生成多少条 response')
    parser.add_argument('--max-length', type=int, default=None,
                        help='最大生成长度（含 prompt）；不传则 eos 停止')
    parser.add_argument('--temperature', type=float, default=0.7,
                        help='采样温度')
    parser.add_argument('--top-p', type=float, default=0.9,
                        help='nucleus 采样 top-p')
    parser.add_argument('--max-prompt-tokens', type=int, default=None,
                        help='限制 prompt 最大 token 数，超出时截断')
    parser.add_argument('--batch-size', type=int, default=8,
                        help='每次 batch 生成多少条 prompt')
    args = parser.parse_args()

    if not os.path.isdir(args.model):
        raise FileNotFoundError(f"找不到模型目录：{args.model}")

    # Torchrun会为每个进程设置LOCAL_RANK环境变量
    # 获取该值以区分不同GPU，默认为0
    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    # 使用NCCL后端初始化进程组（每个进程都需要执行）
    if dist.is_available() and not dist.is_initialized():
        dist.init_process_group(backend="nccl")

    if local_rank == 0:
        print(f"加载 tokenizer 和模型：{args.model}")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model, use_fast=True, trust_remote_code=False
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 让每个进程只在对应的 GPU 上加载模型，避免多进程占用同一块 GPU
    torch.cuda.set_device(local_rank)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        device_map={"": local_rank},
        torch_dtype=torch.float16,
        trust_remote_code=True
    )
    model.eval()

    prompts = load_prompts(args.input)
    if local_rank == 0:
        print(f'共 {len(prompts)} 条 prompt。')
    
    # 创建数据集和 DataLoader
    dataset = PromptDataset(prompts)
    # DistributedSampler 确保不同进程处理数据集的不重复子集
    sampler = None
    if dist.is_initialized():
        sampler = torch.utils.data.DistributedSampler(
            dataset,
            num_replicas=dist.get_world_size(),
            rank=local_rank,
            shuffle=False,
        )
    dataloader = DataLoader(dataset, batch_size=args.batch_size, sampler=sampler)

    with open(args.output, 'a', encoding='utf-8') as fout:
        for batch in tqdm(dataloader, desc=f'进程 {local_rank} 生成中'):
            batch_prompts = batch
            tokenize_kwargs = dict(
                return_tensors='pt',
                padding=True,
                truncation=True if args.max_prompt_tokens is not None else False
            )
            if args.max_prompt_tokens is not None:
                tokenize_kwargs['max_length'] = args.max_prompt_tokens
            enc = tokenizer(batch_prompts, **tokenize_kwargs)
            enc = {k: v.to(model.device) for k, v in enc.items()}

            gen_kwargs = {
                "do_sample": True,
                "temperature": args.temperature,
                "top_p": args.top_p,
                "num_return_sequences": args.num_responses,
                "eos_token_id": tokenizer.eos_token_id,
                "pad_token_id": tokenizer.pad_token_id,
            }
            if args.max_length is not None:
                gen_kwargs["max_length"] = args.max_length

            with torch.no_grad():
                outputs = model.generate(**enc, **gen_kwargs)

            for idx in range(0, outputs.size(0), args.num_responses):
                seqs = outputs[idx: idx + args.num_responses]
                raw_prompt = batch_prompts[idx // args.num_responses]
                responses = []
                response_lengths = []
                for seq in seqs:
                    text = tokenizer.decode(seq, skip_special_tokens=True)
                    if text.startswith(raw_prompt):
                        text = text[len(raw_prompt):].lstrip()
                    responses.append(text)
                    response_lengths.append(len(tokenizer.encode(text)))

                fout.write(json.dumps({
                    "prompt": raw_prompt,
                    "responses": responses,
                    "prompt_length": len(tokenizer.encode(raw_prompt)),
                    "response_lengths": response_lengths
                }, ensure_ascii=False) + "\n")

    # 所有进程写入完成后同步，随后销毁进程组
    if dist.is_initialized():
        dist.barrier()
        if local_rank == 0:
            print(f"完成，结果保存在：{args.output}")
        dist.destroy_process_group()
    else:
        print(f"完成，结果保存在：{args.output}")


if __name__ == '__main__':
    main()