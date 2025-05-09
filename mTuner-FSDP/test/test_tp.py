from transformers import AutoModelForCausalLM

import os
import time

import subprocess
import torch
import torch.multiprocessing as mp
import torch.distributed as dist
from torch import nn
import torch.nn.functional as F

from torch.distributed._tensor.device_mesh import DeviceMesh
import torch.optim as optim

from torch.distributed.tensor.parallel import (
    RowwiseParallel,
    ColwiseParallel,
    parallelize_module,
)

import argparse


is_slurm = "SLURM_JOB_ID" in os.environ and int(os.environ.get("SLURM_NTASKS", -1)) > 1

# print("SLURM_JOB_ID" in os.environ, os.environ.get("SLURM_NTASKS", -1))
# exit()


def setup_distributed(is_slurm, backend="nccl", port=None):
    num_gpus = torch.cuda.device_count()
    if is_slurm:
        rank = int(os.environ["SLURM_PROCID"])
        world_size = int(os.environ["SLURM_NTASKS"])
        os.environ["WORLD_SIZE"] = str(world_size)
        os.environ["LOCAL_RANK"] = str(rank % num_gpus)
        os.environ["RANK"] = str(rank)
        # addr
        node_list = os.environ["SLURM_NODELIST"]
        addr = subprocess.getoutput(f"scontrol show hostname {node_list} | head -n1")
        if "MASTER_ADDR" not in os.environ:
            os.environ["MASTER_ADDR"] = addr
    else:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        # addr
        if "MASTER_ADDR" not in os.environ:
            os.environ["MASTER_ADDR"] = "127.0.0.1"
    # port
    if port is not None:
        os.environ["MASTER_PORT"] = str(port)
    elif "MASTER_PORT" in os.environ:
        pass  # use MASTER_PORT in the environment variable
    else:
        os.environ["MASTER_PORT"] = "29500"

    torch.cuda.set_device(rank % num_gpus)
    dist.init_process_group(
        backend=backend,
        world_size=world_size,
        rank=rank,
    )


class LlamaMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)

    def forward(self, x):
        down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))

        return down_proj


def run(rank, world_size, args):
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    if rank == 0:
        print(args)
    batch_size = args.batch_size
    seq_len = args.seq_len
    model_name = args.model

    setup_distributed(is_slurm)
    root_mesh = DeviceMesh("cuda", torch.arange(world_size))

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
    )

    parallelize_plan = {}
    num_layers = len(model.model.layers)
    for i in range(num_layers):
        parallelize_plan[f"model.layers.{i}.mlp.gate_proj"] = ColwiseParallel()
        parallelize_plan[f"model.layers.{i}.mlp.up_proj"] = ColwiseParallel()
        parallelize_plan[f"model.layers.{i}.mlp.down_proj"] = RowwiseParallel()

        parallelize_plan[f"model.layers.{i}.self_attn.q_proj"] = ColwiseParallel()
        parallelize_plan[f"model.layers.{i}.self_attn.k_proj"] = ColwiseParallel()
        parallelize_plan[f"model.layers.{i}.self_attn.v_proj"] = ColwiseParallel()
        parallelize_plan[f"model.layers.{i}.self_attn.o_proj"] = RowwiseParallel()

    model = parallelize_module(
        model, device_mesh=root_mesh, parallelize_plan=parallelize_plan
    )
    model = model.cuda(rank)

    for layer in model.model.layers:
        assert model.model.config.num_attention_heads % world_size == 0
        layer.self_attn.num_heads = model.model.config.num_attention_heads // world_size
        layer.self_attn.num_key_value_heads = (
            model.model.config.num_key_value_heads // world_size
        )
        layer.self_attn.hidden_size = model.model.config.hidden_size // world_size

    optimizer = optim.AdamW(model.parameters(), lr=0.1)
    loss_func = torch.nn.CrossEntropyLoss()

    for i in range(5):
        input_ids = torch.randint(0, 10000, (batch_size, seq_len), device=rank)
        valley_mem_usage = torch.cuda.memory_allocated(rank) / 1024**3
        t0 = time.time()
        output = model(
            input_ids,
            labels=input_ids,
            use_cache=False,
        )
        loss = loss_func(
            output.logits.view(-1, output.logits.size(-1)),
            input_ids.view(-1),
        )
        peak_mem_usage = torch.cuda.memory_allocated(rank) / 1024**3
        loss.backward()
        peak_mem_usage2 = torch.cuda.memory_allocated(rank) / 1024**3
        optimizer.step()
        t1 = time.time()
        if rank == 0:
            print(
                f"Iter {i} Time {(t1 - t0):3f} Throughput {(batch_size * seq_len / (t1 - t0)):3f} Mem-usage {valley_mem_usage:3f} {peak_mem_usage:3f} {peak_mem_usage2:3f}"
            )

    dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", type=str, default="meta-llama/Llama-2-7b-hf"
    )  # model path
    parser.add_argument("--batch-size", type=int, default=1)  # batch_size
    parser.add_argument("--seq-len", type=int, default=1024)  # batch_size
    args = parser.parse_args()

    if not is_slurm:
        torch.multiprocessing.set_start_method("spawn", force=True)
        world_size = torch.cuda.device_count()
        assert world_size > 0, f"No GPU device detected, world_size={world_size}"
        mp.spawn(
            run,
            args=(
                world_size,
                args,
            ),
            nprocs=world_size,
        )
    else:
        rank = int(os.environ["SLURM_PROCID"])
        world_size = int(os.environ["SLURM_NTASKS"])
        run(rank, world_size, args)
