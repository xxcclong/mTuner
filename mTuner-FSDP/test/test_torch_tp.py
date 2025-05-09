import torch
import torch.nn as nn
from torch.distributed._tensor import DeviceMesh
from torch.distributed.tensor.parallel import (
    PairwiseParallel,
    parallelize_module,
)
import os
import torch.distributed as dist

class LinearWrap(nn.Module):
    def __init__(self, in_features, out_features, bias=True, device=None):
        super().__init__()
        self.linear = nn.Linear(in_features, in_features, bias=bias).to(device)
        self.linear2 = nn.Linear(in_features, out_features, bias=bias).to(device)


    def forward(self, x):
        return self.linear2(self.linear(x))
 
def test_tp(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"

    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)
    mesh = DeviceMesh("cuda", torch.arange(world_size))
    model = LinearWrap(in_features=1600, out_features=800, device=rank)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    parallelize_module(model, mesh, PairwiseParallel())

    import time
    avg_time = 0
    epochs = 50
    for _ in range(epochs):
        torch.cuda.synchronize()
        t0 = time.time()
        src = torch.rand((100, 1600)).to(rank)
        tgt = torch.rand((100, 800)).to(rank)

        output = model(src)

        loss = nn.MSELoss()(output, tgt)
        loss.backward()
        optimizer.step()

        torch.cuda.synchronize()
        t1 = time.time()
        avg_time += t1-t0

    if rank == 0:
        print(f'mp WORLD_SIZE:{world_size}, average time:{avg_time/epochs}')


def test_single_gpu():
    model = LinearWrap(in_features=1600, out_features=800, device='cuda:0')
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    import time
    avg_time = 0
    epochs = 50
    for _ in range(epochs):
        torch.cuda.synchronize()
        t0 = time.time()
        src = torch.rand((100, 1600)).to('cuda:0')
        tgt = torch.rand((100, 800)).to('cuda:0')

        output = model(src)

        loss = nn.MSELoss()(output, tgt)
        loss.backward()
        optimizer.step()
        torch.cuda.synchronize()
        t1 = time.time()
        avg_time += t1-t0
    print(f'WORLD_SIZE:{1}, average time:{avg_time/epochs}')

if __name__ == "__main__":
    WORLD_SIZE = torch.cuda.device_count()
    print(f'{WORLD_SIZE=}')
    if WORLD_SIZE == 1:
        test_single_gpu()
        # exit
    import torch.multiprocessing as mp
    mp.spawn(test_tp, args=(WORLD_SIZE,), nprocs=WORLD_SIZE, join=True)