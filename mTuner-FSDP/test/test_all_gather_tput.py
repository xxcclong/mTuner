import torch
import torch.distributed as dist
import time, os
import torch.multiprocessing as mp
import numpy as np
import math


def setup(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12357"

    # initialize the process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)


def get_replica_size(size, rate, world_size):
    replica_size = math.ceil(size * rate)
    replica_size = (replica_size + world_size - 1) // world_size * world_size
    per_shard_size = (size - replica_size) // world_size
    per_shard_size = (per_shard_size + 8 - 1) // 8 * 8
    replica_size = size - per_shard_size * world_size
    return replica_size


def run(rank, world_size):
    setup(rank, world_size)
    dtype = torch.float32

    filename = "memory_traces/per_param_facebook_opt_30b_1_1024_True_False_-1.txt"
    trace = np.loadtxt(filename, dtype=np.float32)[:, 1]

    # trace = [51200000]
    iters = 4
    for rate in [0, 0.1]:
        for _ in range(iters):
            t = 0
            s = 0
            ts = []
            sizes = []
            bws = []
            for size in trace:
                replica_size = get_replica_size(size, rate, world_size)
                size -= replica_size
                tensor = torch.randn(int(size // world_size), dtype=dtype, device=rank)
                recv_tensor = torch.empty(int(size), dtype=dtype, device=rank)

                dist.barrier()
                torch.cuda.synchronize(rank)
                t0 = time.time()
                for trial in range(4):
                    dist.all_gather_into_tensor(
                        recv_tensor,
                        tensor,
                    )
                torch.cuda.synchronize(rank)
                exec_time = time.time() - t0
                t += exec_time
                s += size
                ts.append(exec_time)
                sizes.append(size)
                bws.append(size / exec_time / 1024 / 1024 / 1024)
            if rank == 0:
                print(
                    f"rank {rank} world_size {world_size} size {s} takes {t} seconds bw {s/t/1024/1024/1024} GB/s"
                )
            if _ == iters - 1 and rank == 0:
                with open(f"all_gather_{rate}.txt", "w") as f:
                    for i in range(len(ts)):
                        f.write(f"{ts[i]} {sizes[i]} {bws[i]}\n")
            # if rank == 0 and _ == iters - 1:
            #     for i in range(len(ts)):
            #         print(f"{ts[i]} {sizes[i]} {bws[i]}\n")


if __name__ == "__main__":
    WORLD_SIZE = torch.cuda.device_count()
    mp.spawn(run, args=(WORLD_SIZE,), nprocs=WORLD_SIZE)
