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

from torch.profiler import profile, record_function, ProfilerActivity


is_slurm = "SLURM_JOB_ID" in os.environ and int(os.environ.get("SLURM_NTASKS", -1)) > 1


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


def run_comm(dst, src, cnt):
    for i in range(cnt):
        dist.all_gather_into_tensor(dst, src)


def run_comm_given_size(sizes):
    dtype = torch.float16
    start_events = []
    stop_events = []
    for size in sizes:
        tensor = torch.empty(int(size), dtype=dtype, device=rank)
        recv_tensor = torch.empty(int(size * world_size), dtype=dtype, device=rank)

        # e1 = torch.cuda.Event(enable_timing=True)
        # e2 = torch.cuda.Event(enable_timing=True)
        # e1.record()
        dist.all_gather_into_tensor(
            recv_tensor,
            tensor,
        )
        # e2.record()
        # start_events.append(e1)
        # stop_events.append(e2)

    return start_events, stop_events


def run_computation(mat1, mat2, cnt):
    for i in range(cnt):
        torch.add(mat1, mat2)


def run_model(model, input_ids, cnt):
    for i in range(cnt):
        model(input_ids)


def run(rank, world_size, args):
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.float16,
    ).to(rank)
    if rank == 0:
        print(args)
    setup_distributed(is_slurm)
    size = 256 * 1024**2
    cnt = 10
    model_cnt = 30
    add_cnt = 16000
    src_data = torch.ones(size, dtype=torch.float16, device=rank)
    dst_data = torch.empty(size * world_size, dtype=torch.float16, device=rank)
    mat1 = torch.randn(4096, 4096, device=rank)
    mat2 = torch.randn(4096, 4096, device=rank)

    import numpy as np

    stream1 = torch.cuda.default_stream()  # torch.cuda.Stream()
    stream2 = torch.cuda.Stream()

    filename = "size2.log"
    trace = np.loadtxt(filename, dtype=np.float32)

    input_ids = torch.randint(0, 10000, (args.batch_size, args.seq_len), device=rank)
    run_model(model, input_ids, model_cnt)

    torch.cuda.synchronize()
    t0 = time.time()
    run_model(model, input_ids, model_cnt)
    # run_computation(mat1, mat2, add_cnt)
    torch.cuda.synchronize()
    t1 = time.time()
    if rank == 0:
        print(t1 - t0)

    dist.barrier()
    t0 = time.time()
    with torch.cuda.stream(stream2):
        # run_comm(dst_data, src_data, cnt)
        start, stop = run_comm_given_size(trace)
    torch.cuda.synchronize()
    t1 = time.time()
    if rank == 0:
        t = 0
        for i in range(len(start)):
            t += start[i].elapsed_time(stop[i])
        print(f"Time {t1 - t0} Event Time {t}")

    if rank == 0:
        # with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        if 1:
            t0 = time.time()
            with torch.cuda.stream(stream2):
                # run_comm(dst_data, src_data, cnt)
                start, stop = run_comm_given_size(trace)
            with torch.cuda.stream(stream1):
                run_model(model, input_ids, model_cnt)
                # run_computation(mat1, mat2, add_cnt)
            torch.cuda.synchronize()
            t1 = time.time()
            print(f"Time {t1 - t0}")
        # prof.export_chrome_trace("trace.json")
        t = 0
        for i in range(len(start)):
            t += start[i].elapsed_time(stop[i])
        print(f"Event Time {t}")
    else:
        with torch.cuda.stream(stream2):
            run_comm_given_size(trace)
            # run_comm(dst_data, src_data, cnt)
        with torch.cuda.stream(stream1):
            run_model(model, input_ids, model_cnt)
            # run_computation(mat1, mat2, add_cnt)
        torch.cuda.synchronize()

    dist.destroy_process_group()


def tput(rank, world_size, args):
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    if rank == 0:
        print(args)
    setup_distributed(is_slurm)

    import numpy as np

    filename = "size2.log"

    trace = np.loadtxt(filename, dtype=np.float32)
    iters = 3

    dtype = torch.float16
    for _ in range(iters):
        t = 0
        s = 0
        ts = []
        sizes = []
        bws = []
        for size in trace:
            tensor = torch.randn(int(size), dtype=dtype, device=rank)
            recv_tensor = torch.empty(int(size * world_size), dtype=dtype, device=rank)
            dist.barrier()
            torch.cuda.synchronize(rank)
            t0 = time.time()
            for trial in range(4):
                dist.all_gather_into_tensor(
                    recv_tensor,
                    tensor,
                )
            torch.cuda.synchronize(rank)
            exec_time = (time.time() - t0) / 4
            t += exec_time
            s += size
            ts.append(exec_time)
            sizes.append(size)
            bws.append(size / exec_time / 1024 / 1024 / 1024)
        if rank == 0:
            print(
                f"rank {rank} world_size {world_size} size {s} takes {t} seconds bw {s/t/1024/1024/1024} GB/s"
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
            # tput,
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
        # tput(rank, world_size, args)
