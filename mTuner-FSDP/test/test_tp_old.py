import torch
import torch.nn as nn
import functools
import torch.optim as optim
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel

from fthub.fsdp import FullyShardedDataParallel as FSDP
from fthub.fsdp.api import ShardingStrategy

# from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
# from torch.distributed.fsdp import ShardingStrategy

from fthub.fsdp.wrap import size_based_auto_wrap_policy

import logging

import os
from torch.distributed.tensor.parallel import (
    parallelize_module,
    PairwiseParallel,
    RowwiseParallel,
    ColwiseParallel,
    fsdp,
    make_output_tensor,
)
from torch.distributed._tensor import DeviceMesh, Shard, DTensor


# Define a simple model
class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.fc = nn.Linear(10, 20, bias=False)
        self.fc.weight.data = torch.arange(200).reshape(20, 10).float()
        self.fc2 = nn.Linear(20, 40, bias=False)
        self.fc2.weight.data = torch.arange(800).reshape(40, 20).float()
        # self.fc2.weight.data = torch.ones_like(self.fc2.weight.data)

    def forward(self, x):
        # print(
        #     x.device,
        #     self.fc.weight.device,
        #     x.shape,
        #     self.fc.weight.shape,
        #     self.fc.weight.data.shape,
        #     type(self.fc.weight),
        # )
        # return self.fc(x)
        return self.fc2(self.fc(x))


def train(rank, world_size):
    # Initialize distributed training
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12357"
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    # os.environ["CUDA_VISIBLE_DEVICES"] = str(rank)
    # dist.init_process_group("nccl", rank=rank, world_size=world_size)

    # Create a model and move it to GPU
    model = SimpleModel()
    # model = DistributedDataParallel(model)
    device_mesh = DeviceMesh("cuda", [0, 1])
    batch_size = 12
    output_cpu = model(torch.ones(batch_size * 2, 10))
    print(output_cpu.shape)

    # def wfn(module):
    #     print(module, module.weight.numel())
    #     return isinstance(module, nn.Linear) and module.weight.numel() >= 1
    rowwise_placement = [Shard(0)]

    my_auto_wrap_policy = functools.partial(
        size_based_auto_wrap_policy, min_num_params=1
    )
    # my_auto_wrap_policy = functools.partial(lambda_auto_wrap_policy, lambda_fn=wfn)
    # fsdp.enable_2d_with_fsdp()

    model = parallelize_module(
        model,
        device_mesh=device_mesh,
        parallelize_plan={"fc": ColwiseParallel(_prepare_output=make_output_tensor)},
    )
    model = FSDP(
        model,
        device_id=rank,
        sharding_strategy=ShardingStrategy.HYBRID_SHARD,
        auto_wrap_policy=my_auto_wrap_policy,
        ignored_parameters=[model.fc.weight],
    )
    for name, param in model.named_parameters():
        print(rank, name, param.shape, param.device)

    # Dummy input data
    input_data = torch.ones(batch_size, 10)
    rowwise_data = DTensor.from_local(input_data, device_mesh, rowwise_placement)

    # Dummy target data
    target_data = torch.randint(0, 2, (batch_size * 2,)).cuda()

    # Define a loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01)

    for epoch in range(5):
        # Forward pass
        output = model(rowwise_data)
        if epoch == 0:
            print(output.shape, output)
            print(torch.allclose(output.cpu(), output_cpu))
        # break
        loss = criterion(output, target_data)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # print(f"Rank {rank}, Epoch {epoch}, Loss: {loss.item()}")

    # Cleanup
    dist.destroy_process_group()


def main():
    world_size = 2  # Number of GPUs
    mp.spawn(train, args=(world_size,), nprocs=world_size, join=True)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
