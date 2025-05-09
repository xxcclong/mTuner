# import subprocess

# def check_all_gpus_on_same_node():
#     try:
#         # 使用 subprocess 运行 nvidia-smi 命令并获取输出
#         result = subprocess.run(['nvidia-smi', '-L'], check=True, text=True, stdout=subprocess.PIPE)
#         output = result.stdout

#         # 解析输出来获取 GPU 列表
#         gpus = output.splitlines()

#         if len(gpus) <= 1:
#             return True

#         # 由于 nvidia-smi 在同一台机器上运行，所以所有 GPU 都应该在同一台机器上
#         return True

#     except subprocess.CalledProcessError:
#         print("Error while running nvidia-smi.")
#         return False

# if __name__ == "__main__":
#     if check_all_gpus_on_same_node():
#         print("All GPUs are on the same machine.")
#     else:
#         print("GPUs are not on the same machine.")


# ----------------------------------------------------------


# import torch
# import torch.distributed as dist
# import torch.multiprocessing as mp
# import argparse
# import os

# def setup_distributed(backend="nccl", port=None):
#     """Initialize distributed training environment.
#     support both slurm and torch.distributed.launch
#     see torch.distributed.init_process_group() for more details
#     """
#     num_gpus = torch.cuda.device_count()

#     if "SLURM_JOB_ID" in os.environ:
#         rank = int(os.environ["SLURM_PROCID"])
#         world_size = int(os.environ["SLURM_NTASKS"])
#         node_list = os.environ["SLURM_NODELIST"]
#         addr = subprocess.getoutput(f"scontrol show hostname {node_list} | head -n1")
#         # specify master port
#         if port is not None:
#             os.environ["MASTER_PORT"] = str(port)
#         elif "MASTER_PORT" not in os.environ:
#             os.environ["MASTER_PORT"] = "29500"
#         if "MASTER_ADDR" not in os.environ:
#             os.environ["MASTER_ADDR"] = addr
#         os.environ["WORLD_SIZE"] = str(world_size)
#         os.environ["LOCAL_RANK"] = str(rank % num_gpus)
#         os.environ["RANK"] = str(rank)
#     else:
#         rank = int(os.environ["RANK"])
#         world_size = int(os.environ["WORLD_SIZE"])

#     torch.cuda.set_device(rank % num_gpus)

#     dist.init_process_group(
#         backend=backend,
#         world_size=world_size,
#         rank=rank,
#     )


# def setup(rank, world_size):
#     os.environ['MASTER_ADDR'] = 'localhost'
#     os.environ['MASTER_PORT'] = '12356'
#     # initialize the process group
#     dist.init_process_group("nccl", rank=rank, world_size=world_size)


# def run(rank,world_size,args):
#     setup(rank, world_size)
#     print(rank,world_size)

# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
#     args = parser.parse_args()

#     WORLD_SIZE = torch.cuda.device_count()
#     print(WORLD_SIZE)
#     mp.spawn(run, args=(WORLD_SIZE,args), nprocs=WORLD_SIZE, join=True)


"""
(MNMC) Multiple Nodes Multi-GPU Cards Training
Minimal ImageNet training code powered by DDP
"""

import os
import subprocess

import torch
import torch.distributed as dist

BATCH_SIZE = 256
EPOCHS = 1


def setup_distributed(backend="nccl", port=None):
    """
    Initialize distributed training environment.
    support both slurm and torch.distributed.launch
    """
    num_gpus = torch.cuda.device_count()

    if "SLURM_JOB_ID" in os.environ:
        rank = int(os.environ["SLURM_PROCID"])
        world_size = int(os.environ["SLURM_NTASKS"])
        node_list = os.environ["SLURM_NODELIST"]
        addr = subprocess.getoutput(f"scontrol show hostname {node_list} | head -n1")
        # specify master port
        if port is not None:
            os.environ["MASTER_PORT"] = str(port)
        elif "MASTER_PORT" in os.environ:
            pass  # use MASTER_PORT in the environment variable
        else:
            os.environ["MASTER_PORT"] = "29500"
        if "MASTER_ADDR" not in os.environ:
            os.environ["MASTER_ADDR"] = addr
        os.environ["WORLD_SIZE"] = str(world_size)
        os.environ["LOCAL_RANK"] = str(rank % num_gpus)
        os.environ["RANK"] = str(rank)
    else:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])

    torch.cuda.set_device(rank % num_gpus)

    dist.init_process_group(
        backend=backend,
        world_size=world_size,
        rank=rank,
    )


if __name__ == "__main__":

    # 0. set up distributed device
    setup_distributed()

    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    device = torch.device("cuda", local_rank)

    print(f"[init] == local rank: {local_rank}, global rank: {rank} ==")

