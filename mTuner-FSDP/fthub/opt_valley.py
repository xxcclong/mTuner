import torch.distributed as dist


def init_partial_process_group(args, handles):
    world_size = args.world_size
    rank = args.rank
    if args.time_dim.layer_id != -1:
        if args.time_dim.group_scaler > 1:
            pgs = world_size // args.time_dim.group_scaler
        for gid in range(int(world_size // pgs)):
            tmp = dist.new_group(ranks=[i + gid * pgs for i in range(pgs)])
            if gid == int(rank // pgs):
                group_shard = tmp
        for handle in handles:
            handle.partial_process_group = group_shard
