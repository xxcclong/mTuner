import torch
import torch.nn as nn
from torch.distributed.tensor.parallel import (
    RowwiseParallel,
    ColwiseParallel,
    make_input_reshard_replicate,
    make_output_reshard_tensor,

)
import numpy
import os
import torch.distributed as dist
import transformers
import os
import hydra
from omegaconf import DictConfig
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp
from fthub.fsdp import FullyShardedDataParallel as FSDP
import fthub
import numpy as np

def setup(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12357"

    # initialize the process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

def test_tp(rank, world_size, args):

    setup(rank, world_size)
    if args.validation:
        np.random.seed(0)
        torch.manual_seed(0)
        torch.cuda.manual_seed(0)
        torch.backends.cudnn.deterministic = True

    num_iter = args.train.iter
    max_batch = args.train.batch_size
    seq_len = args.train.seq_len
    model_name = args.model

    if not args.validation:
        # Skip model initilization
        transformers.PreTrainedModel._initialize_weights = lambda x, *args, **kwargs: x
        torch.nn.init.normal_ = lambda x, *args, **kwargs: x
        torch.nn.init.uniform_ = lambda x, *args, **kwargs: x
        torch.nn.init.xavier_normal_ = lambda x, *args, **kwargs: x
        torch.nn.init.xavier_uniform_ = lambda x, *args, **kwargs: x
        torch.nn.init.kaiming_normal_ = lambda x, *args, **kwargs: x
        torch.nn.init.kaiming_uniform_ = lambda x, *args, **kwargs: x

    torch.cuda.set_device(rank)
    torch.distributed.barrier()

    assert model_name in [
        "facebook_opt_125m",
        "facebook_opt_6.7b",
        "facebook_opt_30b",
        "simple",
        "/mnt/data/zhongrx/Llama-2-13b-hf",
        "/mnt/data/zhongrx/Llama-2-7b-hf",
        "/data/dataset/Llama-2-70b-hf-trans",
    ]

    if "opt" in model_name:
        model = fthub.FTOPTForCausalLM.from_pretrained(
            model_name, torch_dtype=torch.float16, groups=args.group_size
        )
    elif "Llama" in model_name:
        model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=torch.float16
        )
    else:
        assert False, f"{model_name} not supported"
    # model = fthub.change_model_with_config(model, args, rank, world_size)
    model = fthub.to_bettertransformer(model)
    model = fthub.to_peft(model, args)
    model = fthub.to_FSDP(model, args, rank, world_size)
    model = fthub.to_ac(model, args)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001)

    a = torch.cuda.memory_allocated(rank) / (1024 * 1024 * 1024)
    print("模型transform后使用显存为{}".format(a))
    import time

    avg_time = 0
    avg_activation_per_sample = 0
    tput = 0
    for i in range(num_iter):
        torch.cuda.synchronize(rank)
        dist.barrier()
        t0 = time.time()

        input_ids = torch.randint(0, 1024, (max_batch, seq_len)).to(rank)

        before = torch.cuda.memory_allocated(rank) / (1024 * 1024 * 1024)
        print("模型前向传播前使用显存为{}".format(before))

        output = model(
            input_ids,
            labels=input_ids,
            use_cache=False,  # reduce
        )

        after = torch.cuda.memory_allocated(rank) / (1024 * 1024 * 1024)
        print("模型前向传播后使用显存为{}，差值（中间激活）为{}".format(after, after - before))

        loss_func = nn.CrossEntropyLoss()
        loss = loss_func(
            output.logits.view(-1, output.logits.size(-1)),
            input_ids.view(-1),
        )

        del output
        torch.cuda.synchronize(rank)
        dist.barrier()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        del loss
        torch.cuda.synchronize(rank)
        dist.barrier()
        t1 = time.time()
        if i > 0:
            avg_time += t1 - t0
            avg_activation_per_sample += (after - before)/ max_batch 
    if rank == 0:
        avg_time = avg_time / (num_iter - 1)
        tput = max_batch * seq_len / avg_time
        avg_activation_per_sample = avg_activation_per_sample / (num_iter - 1)

        numbers = numpy.array([max_batch, avg_time, tput, avg_activation_per_sample])
        txt_path = f"13b_tp_{seq_len}_dec20_gpu4_mergedmain_baseline.txt"
        with open(txt_path, "a") as file:
            file.write(" ".join(map(str, numbers)) + "\n")

@hydra.main(version_base=None, config_path="./configs", config_name="config")
def main(config: DictConfig):
    torch.multiprocessing.set_start_method("spawn", force=True)
    WORLD_SIZE = torch.cuda.device_count()
    mp.spawn(test_tp, args=(WORLD_SIZE, config), nprocs=WORLD_SIZE)

if __name__ == "__main__":
    main()
