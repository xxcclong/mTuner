import torch
from torch.distributed._tensor import DTensor, Shard, Replicate, distribute_tensor, distribute_module, init_device_mesh
from torch.distributed.tensor.parallel import make_input_reshard_replicate, make_output_reshard_tensor
import time
import torch.distributed as dist
import os
import torch.multiprocessing as mp

def run(rank, world_size):
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "12454"

    # initialize the process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

    def bw_bench():
        bs = 16
        seq_len = 1024
        hidden = 8192
        hidden_out = 28672
        num_trial = 10
        num_device = world_size
        local_weight = torch.randn((hidden * hidden_out), requires_grad=True, device=rank, dtype=torch.float16)
        full_weight = torch.empty((hidden * hidden_out * num_device), device=rank, dtype=torch.float16)
        dist.all_gather_into_tensor(full_weight, local_weight)

        workload2 = hidden_out * hidden * (num_device - 1) * 2
        torch.cuda.synchronize(rank)
        t0 = time.time()
        for _ in range(num_trial):
            dist.all_gather_into_tensor(full_weight, local_weight)
        torch.cuda.synchronize(rank)
        t1 = time.time()
        print("time: ", (t1 - t0) / num_trial, "bandwidth: ", workload2 * num_trial / (t1 - t0) / 1e9, "GB/s")

    bw_bench()

if __name__ == "__main__":
    WORLD_SIZE = torch.cuda.device_count()
    mp.spawn(run, args=(WORLD_SIZE,), nprocs=WORLD_SIZE)