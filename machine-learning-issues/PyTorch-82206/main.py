# main.py

import os
import torch
import torch.distributed as dist
from unet import UNetModel
from torch.distributed.fsdp.fully_sharded_data_parallel import (
    FullyShardedDataParallel as FSDP,
    CPUOffload
)
from torch.cuda.amp import autocast
from torch.optim import AdamW
from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy
from torch.distributed.fsdp.sharded_grad_scaler import ShardedGradScaler
from torch.nn.parallel import DistributedDataParallel as DDP

os.environ['RANK'] = '0'
os.environ['LOCAL_RANK'] = '0'
os.environ['WORLD_SIZE'] = '1'
os.environ['MASTER_ADDR'] = 'localhost'
os.environ['MASTER_PORT'] = '12355'
dist.init_process_group(backend="nccl")
dev = int(os.environ['LOCAL_RANK'])
torch.cuda.set_device(dev)

model = UNetModel(
    in_channels=3,
    model_channels=128,
    out_channels=3,
    num_res_blocks=3,
    attention_resolutions=[8]
)
img = torch.rand((2, 3, 64, 64), dtype=torch.float32).to(f'cuda')
t = torch.rand(2, dtype=torch.float32).to(f'cuda')

fsdp_model = FSDP(
    model,
    device_id=dev,
    auto_wrap_policy=size_based_auto_wrap_policy,
    cpu_offload=CPUOffload(offload_params=True),
)
fsdp_model.train()
with autocast():
    r = fsdp_model(img, t)
    loss = r.sum()

scaler = ShardedGradScaler()
optim = AdamW(fsdp_model.parameters(), lr=1e-4)
scaler.scale(loss).backward()
scaler.step(optim)

print("Success")