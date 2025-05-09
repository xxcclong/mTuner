import torch
from torch import nn
import copy
import os
from torch.distributed._tensor import DTensor, Shard, Replicate, distribute_tensor, distribute_module, init_device_mesh
from torch.distributed._tensor.device_mesh import DeviceMesh, mesh_resources
from torch.distributed.tensor.parallel import (
    PairwiseParallel,
    RowwiseParallel,
    ColwiseParallel,
    make_output_tensor,
    parallelize_module,
    make_sharded_output_tensor,
    make_input_reshard_replicate,
    make_output_shard_1d,
    make_output_reshard_tensor,
    SequenceParallel,
    make_input_shard_1d,
)
import torch.distributed._functional_collectives
import time

class LlamaConfig():
    def __init__(self, hidden_size, intermediate_size):
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size

class LlamaMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = nn.GELU()

    def forward(self, x):
        down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))

        return down_proj

def llama_forward2(self, x):
    # make x replicated on dim 0
    current_mesh = mesh_resources.root_mesh
    shard_spec = [Shard(0)]
    replicate = [Replicate()]
    x = DTensor.from_local(x, current_mesh, shard_spec, run_check=False).redistribute(current_mesh, replicate)
    down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
    return down_proj

def run():
    num_trial = 10
    config = LlamaConfig(8192, 28672)
    bs = 6
    seq_len = 1024
    rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    # device_mesh = init_device_mesh("cuda", (world_size,))
    device_mesh = init_device_mesh("cuda", (2,2))
    root_mesh = DeviceMesh("cuda", torch.arange(world_size))
    mesh_resources.root_mesh = root_mesh
    input_tensor = torch.randn((bs, seq_len, config.hidden_size), requires_grad=True, device=rank, dtype=torch.float16)
    torch.manual_seed(0)
    model = LlamaMLP(config).to(torch.float16)
    model_gpu = copy.deepcopy(model).to(rank)
    output_correct = model_gpu(input_tensor)
    torch.cuda.synchronize(rank)
    t2 = time.time()
    for _ in range(num_trial):
        output = model_gpu(input_tensor)
    torch.cuda.synchronize(rank)
    t3 = time.time()
    skip_input = 0
    if skip_input == 1: # skip all
        input_tensor_replicated = make_input_reshard_replicate(input_tensor, device_mesh)
        print(input_tensor_replicated.shape, input_tensor_replicated._local_tensor.shape)

    # TP
    parallelize_plan = {}

    if skip_input > 0:
        parallelize_plan["up_proj"] = ColwiseParallel()
        parallelize_plan["gate_proj"] = ColwiseParallel()
    else:
        parallelize_plan["up_proj"] = ColwiseParallel(_prepare_input=make_input_reshard_replicate)
        parallelize_plan["gate_proj"] = ColwiseParallel(_prepare_input=make_input_reshard_replicate)
    parallelize_plan["down_proj"] = RowwiseParallel(_prepare_output=make_output_reshard_tensor)
    parallelize_module(module=model, device_mesh=device_mesh, parallelize_plan=parallelize_plan)
    if skip_input == 2:
        LlamaMLP.forward = llama_forward2
    # print(model)
    if skip_input == 1:
        output = model(input_tensor_replicated)
    else:
        output = model(input_tensor)
    # print(output.shape, type(output))
    # print(output_correct.shape, type(output_correct))
    print(torch.allclose(output, output_correct, atol=1e-3))
    # if rank == 0:
    #     print(output)


    torch.cuda.synchronize(rank)
    t0 = time.time()
    for _ in range(num_trial):
        if skip_input == 1:
            output = model(input_tensor_replicated)
        else:
            output = model(input_tensor)
    torch.cuda.synchronize(rank)
    t1 = time.time()


    if rank == 0:
        print(f"TP: time: {(t1 - t0)/num_trial}")
        print(f"GPU: time: {(t3 - t2)/num_trial}")




run()