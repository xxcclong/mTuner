import torch
from torch.distributed.pipeline.sync import Pipe
import tempfile
from torch.distributed import rpc
import torch.distributed as dist
import os

# from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from fthub.fsdp import FullyShardedDataParallel as FSDP
from fthub.fsdp.api import ShardingStrategy


def init_pg(rank, world_size):
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "12444"
    # initialize the process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    group_all = torch.distributed.new_group(ranks=[i for i in range(world_size)])
    sub_groups = []
    for k in range(world_size):
        ids = [i for i in range(world_size) if i != k]
        if rank == 0:
            print(ids)
        tmp = torch.distributed.new_group(ranks=ids)
        sub_groups.append(tmp)
    return group_all, sub_groups


def run(rank, world_size):
    model = torch.nn.Sequential(*[torch.nn.Linear(10, 10) for i in range(4)])
    all_group, sub_groups = init_pg(rank, world_size)
    if rank == 0:
        model_pipe = torch.nn.Sequential(*[model[i].cuda(i) for i in range(4)])

        tmpfile = tempfile.NamedTemporaryFile()
        rpc.init_rpc(
            name="worker",
            rank=0,
            world_size=1,
            rpc_backend_options=rpc.TensorPipeRpcBackendOptions(
                init_method="file://{}".format(tmpfile.name)
            ),
        )
        model_pipe = Pipe(model_pipe, chunks=1)

    t = torch.ones(13, device=rank)
    src = torch.ones(37, device=rank)
    dst = torch.zeros(111, device=rank)
    # if rank != 0:
    #     dist.all_reduce(t, group=sub_groups[0])
    #     print(t)
    #     dist.all_gather_into_tensor(
    #         dst,
    #         src,
    #         sub_groups[0],
    #     )
    #     # print(dst)
    #     print(rank, sub_groups[0])
    #     torch.cuda.set_device(rank)
    #     dst._typed_storage()._resize_(0)
    #     size = dst.numel()
    #     print(size)
    #     dst._typed_storage()._resize_(size)
    #     print(
    #         dst.shape,
    #         dst.numel(),
    #         dst.data_ptr(),
    #         dst.storage_offset(),
    #         dst.device,
    #         dst.dtype,
    #         dst._typed_storage().get_device()
    #         # dst.memory_format,
    #     )
    #     dist.all_reduce(dst, group=sub_groups[0])
    #     # dist.all_gather_into_tensor(
    #     #     dst,
    #     #     src,
    #     #     sub_groups[0],
    #     # )

    # exit()

    fsdp_modules = []
    for it, item in enumerate(model):
        if it != rank:  # perform fsdp
            fsdp_module = FSDP(
                item,
                process_group=sub_groups[it],
                sharding_strategy=ShardingStrategy.FULL_SHARD,
                device_id=rank,
            )
            fsdp_modules.append(fsdp_module)
        else:
            fsdp_modules.append(None)
        break
    input_data = torch.randn(13, 10, device=rank)
    # if rank != 0:
    #     fsdp_modules[0](input_data)

    # run fsdp modules
    for it, item in enumerate(fsdp_modules):
        if it != rank:
            input_data = item(input_data)
        if rank == it + 1:
            dist.send(input_data, rank - 1)
        elif rank == it and it != world_size - 1:
            dist.recv(input_data, rank + 1)
    # run pipeline
    if rank == 0:
        model_pipe(input_data)
        # rpc.shutdown()


if __name__ == "__main__":
    num_process = 4
    import torch.multiprocessing as mp

    mp.spawn(run, args=(num_process,), nprocs=num_process, join=True)
