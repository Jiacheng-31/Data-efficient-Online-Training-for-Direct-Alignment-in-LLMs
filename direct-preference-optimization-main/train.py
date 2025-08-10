import torch
torch.backends.cuda.matmul.allow_tf32 = True
import torch.nn as nn
import transformers
from utils import get_local_dir, get_local_run_dir, disable_dropout, init_distributed, get_open_port
import os
import hydra
import torch.multiprocessing as mp
from omegaconf import OmegaConf, DictConfig
import trainers
import json
import socket
from typing import Optional, Set
import resource
import time
from threading import Thread

# 禁用 wandb：覆盖其所有关键函数，确保不进行初始化和日志记录
try:
    import wandb
    wandb.init   = lambda *args, **kwargs: None
    wandb.log    = lambda *args, **kwargs: None
    wandb.watch  = lambda *args, **kwargs: None
except ImportError:
    pass

# 注册 hydra 的自定义解析器，便于获取本地运行目录
OmegaConf.register_new_resolver("get_local_run_dir", lambda exp_name, local_dirs: get_local_run_dir(exp_name, local_dirs))

def monitor_gpu_usage(interval=10):
    """
    每隔 interval 秒打印一次所有 GPU 的利用率和显存使用情况。
    需要安装 pynvml 库：pip install nvidia-ml-py3
    """
    try:
        import pynvml
        pynvml.nvmlInit()
        device_count = pynvml.nvmlDeviceGetCount()
        while True:
            for i in range(device_count):
                handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                print(f"[GPU Monitor] GPU {i}: 利用率 {util.gpu}% | 内存使用 {mem_info.used/1024**2:.1f} MB / {mem_info.total/1024**2:.1f} MB")
            time.sleep(interval)
    except Exception as e:
        print(f"[GPU Monitor] 监控异常: {e}")

def worker_main(rank: int, world_size: int, config: DictConfig, policy: nn.Module, reference_model: Optional[nn.Module] = None):
    """每个训练进程的主函数（对 BasicTrainer/TensorParallelTrainer 可能仅有一个进程）。"""
    if 'FSDP' in config.trainer:
        init_distributed(rank, world_size, port=config.fsdp_port)
    
    # 在主进程中启动 GPU 利用率监控线程，避免多进程输出混乱
    if rank == 0:
        monitor_thread = Thread(target=monitor_gpu_usage, kwargs={'interval': 10}, daemon=True)
        monitor_thread.start()

    # 如果需要禁用wandb，这里不做任何 wandb 初始化调用

    TrainerClass = getattr(trainers, config.trainer)
    print(f'Creating trainer on process {rank} with world size {world_size}')
    trainer = TrainerClass(policy, config, config.seed, config.local_run_dir, reference_model=reference_model, rank=rank, world_size=world_size)
    trainer.train()
    trainer.save()

@hydra.main(version_base=None, config_path="config", config_name="config")
def main(config: DictConfig):
    """训练的主入口。验证配置、创建模型、初始化并启动工作进程。"""

    # 解析 hydra 配置中的引用
    OmegaConf.resolve(config)

    missing_keys: Set[str] = OmegaConf.missing_keys(config)
    if missing_keys:
        raise ValueError(f"Got missing keys in config:\n{missing_keys}")

    if config.eval_every % config.batch_size != 0:
        print('WARNING: eval_every must be divisible by batch_size')
        print('Setting eval_every to', config.eval_every - config.eval_every % config.batch_size)
        config.eval_every = config.eval_every - config.eval_every % config.batch_size

    if 'FSDP' in config.trainer and config.fsdp_port is None:
        free_port = get_open_port()
        print('no FSDP port specified; using open port for FSDP:', free_port)
        config.fsdp_port = free_port

    print(OmegaConf.to_yaml(config))

    config_path = os.path.join(config.local_run_dir, 'config.yaml')
    with open(config_path, 'w') as f:
        OmegaConf.save(config, f)

    print('=' * 80)
    print(f'Writing to {socket.gethostname()}:{config.local_run_dir}')
    print('=' * 80)
 
    os.environ['XDG_CACHE_HOME'] = get_local_dir(config.local_dirs)
    print('building policy')
    model_kwargs = {'device_map': 'balanced'} if config.trainer == 'BasicTrainer' else {}
    policy_dtype = getattr(torch, config.model.policy_dtype)
    policy = transformers.AutoModelForCausalLM.from_pretrained(
        config.model.name_or_path,
        cache_dir=get_local_dir(config.local_dirs),
        low_cpu_mem_usage=True,
        torch_dtype=policy_dtype,
        **model_kwargs)
    disable_dropout(policy)

    if config.loss.name in {'dpo', 'ipo'}:
        print('building reference model')
        reference_model_dtype = getattr(torch, config.model.reference_dtype)
        reference_model = transformers.AutoModelForCausalLM.from_pretrained(
            config.model.name_or_path,
            cache_dir=get_local_dir(config.local_dirs),
            low_cpu_mem_usage=True,
            torch_dtype=reference_model_dtype,
            **model_kwargs)
        disable_dropout(reference_model)
    else:
        reference_model = None

    if config.model.archive is not None:
        state_dict = torch.load(config.model.archive, map_location='cpu')
        step, metrics = state_dict['step_idx'], state_dict['metrics']
        print(f'loading pre-trained weights at step {step} from {config.model.archive} with metrics {json.dumps(metrics, indent=2)}')
        policy.load_state_dict(state_dict['state'])
        if config.loss.name in {'dpo', 'ipo'}:
            reference_model.load_state_dict(state_dict['state'])
        print('loaded pre-trained weights')
    
    if 'FSDP' in config.trainer:
        world_size = torch.cuda.device_count()
        print('starting', world_size, 'processes for FSDP training')
        soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
        resource.setrlimit(resource.RLIMIT_NOFILE, (hard, hard))
        print(f'setting RLIMIT_NOFILE soft limit to {hard} from {soft}')
        mp.spawn(worker_main, nprocs=world_size, args=(world_size, config, policy, reference_model), join=True)
    else:
        print('starting single-process worker')
        worker_main(0, 1, config, policy, reference_model)

if __name__ == '__main__':
    main()
