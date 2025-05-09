import torch.distributed as dist
import torch.multiprocessing as mp
import torch
import argparse
import os
import time

# import allgather

# def setup(rank, world_size):
#     os.environ['MASTER_ADDR'] = 'localhost'
#     os.environ['MASTER_PORT'] = str(12356 + os.getpid()%40000)
#     # initialize the process group
#     dist.init_process_group("nccl", rank=rank, world_size=world_size)


def get_init_method(addr=None, port=None):
    if port is None:
        port = 13531

    if addr is None:
        slurm_ip = os.getenv("SLURM_STEP_NODELIST")
        name = slurm_ip.split("[")[0]
        num = slurm_ip.split("[")[1].split("-")[0]
        addr = f"{name}{num}"

    return f"tcp://{name}{num}:{port}"


def init_multiprocessing(
    rank,
    world_size,
    master_addr=None,
    master_port=None,
    backend="nccl",
    init_method=None,
):
    if init_method is None:
        init_method = get_init_method(master_addr, master_port)

    torch.distributed.init_process_group(
        rank=rank,
        world_size=world_size,
        backend=backend,
        init_method=init_method,
    )


def setup():
    # WORLD_SIZE = torch.cuda.device_count()
    local_rank = int(os.environ["SLURM_LOCALID"])
    global_rank = int(os.environ["SLURM_PROCID"])
    world_size = int(os.environ["SLURM_NPROCS"])
    node_id = os.environ["SLURM_NODEID"]
    torch.cuda.set_device(local_rank)
    # print(WORLD_SIZE,world_size)
    # assert WORLD_SIZE == world_size
    # initialize the process group
    print(
        "local_rank: ",
        local_rank,
        "global_rank: ",
        global_rank,
        "world_size: ",
        world_size,
        "node_id: ",
        node_id,
    )
    init_multiprocessing(global_rank, world_size)
    return world_size, local_rank, global_rank


def setup_mgpu(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12373"

    # initialize the process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)


def cleanup():
    dist.destroy_process_group()


def all_gather_surrogate_v1(dst_tensor, src_tensor, rank, world_size):
    """
    dist.all_gather_into_tensor简化版
    """
    # rank = int(os.environ["SLURM_PROCID"])
    # world_size = int(os.environ["SLURM_NPROCS"])

    # 存储所有的发送和接收句柄
    requests = []

    # 为每个进程发送数据
    for i in range(world_size):
        if i == rank:
            continue  # 当前进程不需要发送数据给自己

        if i > rank:
            # 发送数据到进程i
            requests.append(dist.isend(src_tensor, i))

            # 从进程i接收数据
            start_idx = i * src_tensor.numel()
            end_idx = start_idx + src_tensor.numel()
            requests.append(dist.irecv(dst_tensor[start_idx:end_idx], i))
        else:
            # 从进程i接收数据
            start_idx = i * src_tensor.numel()
            end_idx = start_idx + src_tensor.numel()
            requests.append(dist.irecv(dst_tensor[start_idx:end_idx], i))

            # 发送数据到进程i
            requests.append(dist.isend(src_tensor, i))

    # 对于当前进程，直接将src_tensor的数据拷贝到dst_tensor的相应位置
    start_idx = rank * src_tensor.numel()
    end_idx = start_idx + src_tensor.numel()
    dst_tensor[start_idx:end_idx] = src_tensor

    # 等待所有非阻塞操作完成
    # for req in reqs:
    #     req.wait()
    for req in requests:
        req.wait()


def all_gather_with_replica(dst_tensor, src_tensor, replica):
    expected_numel = dst_tensor.numel()
    replica_size = replica.numel()
    dist.all_gather_into_tensor(
        dst_tensor[: expected_numel - replica_size],
        src_tensor,
    )
    dst_tensor[expected_numel - replica_size :] = replica


def all_gather_with_replica_2(dst_tensor, src_tensor, replica, rank, world_size):
    expected_numel = dst_tensor.numel()
    replica_size = replica.numel()
    all_gather_surrogate_v1(
        dst_tensor[: expected_numel - replica_size], src_tensor, rank, world_size
    )
    dst_tensor[expected_numel - replica_size :] = replica


def all_gather_with_replica_diff_devices_2(dst_tensor, replica, rank, world_size):
    # local_rank = rank % 2
    global_rank = rank
    new_replica_size = replica.numel()
    size = dst_tensor.numel()
    comm_size = size - new_replica_size
    requests = []

    """先处理2,4,8卡的情况"""
    # assert world_size % 2 == 0
    if comm_size <= new_replica_size:
        if global_rank % 2 == 0:
            """local_rank == 0"""
            dst_tensor[:new_replica_size] = replica
            send_op = dist.P2POp(dist.isend, replica[:comm_size], global_rank + 1)
            requests.append(send_op)
            recv_op = dist.P2POp(dist.irecv, dst_tensor[-comm_size:], global_rank + 1)
            requests.append(recv_op)
        else:
            """local_rank == 1"""
            dst_tensor[-new_replica_size:] = replica
            recv_op = dist.P2POp(dist.irecv, dst_tensor[:comm_size], global_rank - 1)
            requests.append(recv_op)
            send_op = dist.P2POp(dist.isend, replica[-comm_size:], global_rank - 1)
            requests.append(send_op)
    elif comm_size <= 3 * new_replica_size:
        """
        4卡, replica_size > 1/3
        8卡, replica_size > 3/7
        """
        l3 = (size - 2 * new_replica_size) // 2
        l4 = size - 2 * new_replica_size - l3

        if global_rank % 4 == 0:
            dst_tensor[:new_replica_size] = replica

            "和1通信"
            send_op = dist.P2POp(dist.isend, replica, global_rank + 1)
            requests.append(send_op)
            recv_op = dist.P2POp(
                dist.irecv,
                dst_tensor[-new_replica_size:],
                global_rank + 1,
            )
            requests.append(recv_op)

            "和2通信"
            send_op = dist.P2POp(dist.isend, replica[:l3], global_rank + 2)
            requests.append(send_op)
            recv_op = dist.P2POp(
                dist.irecv,
                dst_tensor[new_replica_size : new_replica_size + l3],
                global_rank + 2,
            )
            requests.append(recv_op)

            "和3通信"
            send_op = dist.P2POp(dist.isend, replica[:l3], global_rank + 3)
            requests.append(send_op)

            recv_op = dist.P2POp(
                dist.irecv,
                dst_tensor[new_replica_size + l3 : size - new_replica_size],
                global_rank + 3,
            )
            requests.append(recv_op)

        elif global_rank % 4 == 1:
            dst_tensor[-new_replica_size:] = replica

            "和0通信"
            send_op = dist.P2POp(dist.isend, replica, global_rank - 1)
            requests.append(send_op)
            recv_op = dist.P2POp(
                dist.irecv, dst_tensor[:new_replica_size], global_rank - 1
            )
            requests.append(recv_op)

            "和2通信"
            send_op = dist.P2POp(dist.isend, replica[-l4:], global_rank + 1)
            requests.append(send_op)
            recv_op = dist.P2POp(
                dist.irecv,
                dst_tensor[new_replica_size : new_replica_size + l3],
                global_rank + 1,
            )
            requests.append(recv_op)

            "和3通信"
            send_op = dist.P2POp(dist.isend, replica[-l4:], global_rank + 2)
            requests.append(send_op)

            recv_op = dist.P2POp(
                dist.irecv,
                dst_tensor[new_replica_size + l3 : size - new_replica_size],
                global_rank + 2,
            )
            requests.append(recv_op)

        elif global_rank % 4 == 2:
            dst_tensor[l3 : new_replica_size + l3] = replica

            "和0通信"
            recv_op = dist.P2POp(dist.irecv, dst_tensor[:l3], global_rank - 2)
            requests.append(recv_op)

            send_op = dist.P2POp(dist.isend, replica[-l3:], global_rank - 2)
            requests.append(send_op)

            "和1通信"
            send_op = dist.P2POp(dist.isend, replica[-l3:], global_rank - 1)
            requests.append(send_op)
            recv_op = dist.P2POp(
                dist.irecv,
                dst_tensor[-l4:],
                global_rank - 1,
            )
            requests.append(recv_op)

            "和3通信"
            send_op = dist.P2POp(
                dist.isend,
                replica,
                global_rank + 1,
            )
            requests.append(send_op)
            recv_op = dist.P2POp(
                dist.irecv,
                dst_tensor[new_replica_size + l3 : size - l4],
                global_rank + 1,
            )
            requests.append(recv_op)

        elif global_rank % 4 == 3:
            dst_tensor[new_replica_size + l3 : 2 * new_replica_size + l3] = replica

            "和0通信"
            recv_op = dist.P2POp(
                dist.irecv,
                dst_tensor[:l3],
                global_rank - 3,
            )
            requests.append(recv_op)

            send_op = dist.P2POp(dist.isend, replica[:l4], global_rank - 3)
            requests.append(send_op)

            "和1通信"
            send_op = dist.P2POp(dist.isend, replica[:l4], global_rank - 2)
            requests.append(send_op)
            recv_op = dist.P2POp(
                dist.irecv,
                dst_tensor[-l4:],
                global_rank - 2,
            )
            requests.append(recv_op)

            "和2通信"
            recv_op = dist.P2POp(
                dist.irecv, dst_tensor[l3 : new_replica_size + l3], global_rank - 1
            )
            requests.append(recv_op)
            send_op = dist.P2POp(dist.isend, replica, global_rank - 1)
            requests.append(send_op)

    if len(requests) > 0:
        reqs = dist.batch_isend_irecv(requests)
        for req in reqs:
            req.wait()


def all_gather_with_replica_diff_devices(dst_tensor, replica, rank, world_size):
    local_rank = rank
    global_rank = rank
    node_id = rank // 2
    num_nodes = world_size // 2

    # world_size = int(os.environ["SLURM_NPROCS"])
    # local_rank = int(os.environ["SLURM_LOCALID"])
    # global_rank = int(os.environ["SLURM_PROCID"])
    # node_id = int(os.environ["SLURM_NODEID"])
    # num_nodes = int(os.environ["SLURM_NNODES"])
    tasks_per_node = 2

    new_replica_size = replica.numel()
    size = dst_tensor.numel()
    if tasks_per_node * new_replica_size >= size:
        "case2"
        k = size // new_replica_size
        gap = size % new_replica_size
        # machine = global_rank // tasks_per_node
        local_rank1 = k - 1
        requests = []

        if gap != 0:
            "大多数情况, size不是new_replica_size的整数倍"
            if local_rank >= k:
                dst_tensor[-new_replica_size:] = replica
                if local_rank == k:
                    "第一个达到顶端的block"
                    for i in range(
                        node_id * tasks_per_node, node_id * tasks_per_node + local_rank
                    ):
                        send_op = dist.P2POp(dist.isend, replica[-gap:], i)
                        requests.append(send_op)

                        local_rank_of_i = i % tasks_per_node
                        if local_rank_of_i == k - 1:
                            start = local_rank_of_i * new_replica_size
                            end = start + gap
                            recv_op = dist.P2POp(dist.irecv, dst_tensor[start:end], i)
                            requests.append(recv_op)
                        else:
                            start = local_rank_of_i * new_replica_size
                            end = start + new_replica_size
                            recv_op = dist.P2POp(dist.irecv, dst_tensor[start:end], i)
                            requests.append(recv_op)

                else:
                    "第i>1个达到顶端的block"
                    for i in range(
                        node_id * tasks_per_node, node_id * tasks_per_node + k
                    ):
                        local_rank_of_i = i % tasks_per_node
                        if local_rank_of_i == k - 1:
                            start = local_rank_of_i * new_replica_size
                            end = start + gap
                            recv_op = dist.P2POp(dist.irecv, dst_tensor[start:end], i)
                            requests.append(recv_op)
                        else:
                            start = local_rank_of_i * new_replica_size
                            end = start + new_replica_size
                            recv_op = dist.P2POp(dist.irecv, dst_tensor[start:end], i)
                            requests.append(recv_op)

            else:
                "这些block没到顶"
                dst_tensor[
                    local_rank * new_replica_size : (local_rank + 1) * new_replica_size
                ] = replica
                for i in range(
                    node_id * tasks_per_node, (node_id + 1) * tasks_per_node
                ):
                    if i == global_rank:
                        """自己和自己没有通信"""
                        continue
                    if local_rank == k - 1:
                        if i > global_rank:
                            "给后面的device传部分参数"
                            send_op = dist.P2POp(dist.isend, replica[:gap], i)
                            requests.append(send_op)
                            if i == global_rank + 1:
                                recv_op = dist.P2POp(
                                    dist.irecv, dst_tensor[k * new_replica_size :], i
                                )
                                requests.append(recv_op)
                        else:
                            "i < global_rank"
                            send_op = dist.P2POp(dist.isend, replica, i)
                            requests.append(send_op)
                            recv_op = dist.P2POp(
                                dist.irecv,
                                dst_tensor[
                                    i * new_replica_size : (i + 1) * new_replica_size
                                ],
                                i,
                            )
                            requests.append(recv_op)
                    else:
                        "local_rank < k - 1"
                        if i >= node_id * tasks_per_node + k:
                            "device i的参数到顶了"
                            if i == node_id * tasks_per_node + k:
                                recv_op = dist.P2POp(
                                    dist.irecv, dst_tensor[k * new_replica_size :], i
                                )
                                requests.append(recv_op)
                        else:
                            "device i的参数没到顶"
                            recv_op = dist.P2POp(
                                dist.irecv,
                                dst_tensor[
                                    i * new_replica_size : (i + 1) * new_replica_size
                                ],
                                i,
                            )
                            requests.append(recv_op)
                        send_op = dist.P2POp(dist.isend, replica, i)
                        requests.append(send_op)
        else:
            "size是new_replica_size的整数倍"
            if local_rank == k - 1:
                "第一个到顶的block"
                dst_tensor[local_rank * new_replica_size :] = replica
                for i in range(
                    node_id * tasks_per_node, node_id * tasks_per_node + k - 1
                ):
                    local_rank_of_i = i % tasks_per_node
                    "i < global_rank = node_id * tasks_per_node + k - 1"
                    send_op = dist.P2POp(dist.isend, replica, i)
                    requests.append(send_op)
                    recv_op = dist.P2POp(
                        dist.irecv,
                        dst_tensor[
                            local_rank_of_i
                            * new_replica_size : (local_rank_of_i + 1)
                            * new_replica_size
                        ],
                        i,
                    )
                    requests.append(recv_op)
            elif local_rank > k - 1:
                "第i>1个到顶的block"
                dst_tensor[-new_replica_size:] = replica
                for i in range(
                    node_id * tasks_per_node, node_id * tasks_per_node + k - 1
                ):
                    local_rank_of_i = i % tasks_per_node
                    recv_op = dist.P2POp(
                        dist.irecv,
                        dst_tensor[
                            local_rank_of_i
                            * new_replica_size : (local_rank_of_i + 1)
                            * new_replica_size
                        ],
                        i,
                    )
                    requests.append(recv_op)
            else:
                """local_rank < k - 1"""
                dst_tensor[
                    local_rank * new_replica_size : (local_rank + 1) * new_replica_size
                ] = replica
                for i in range(
                    node_id * tasks_per_node, (node_id + 1) * tasks_per_node
                ):
                    local_rank_of_i = i % tasks_per_node
                    if local_rank_of_i == local_rank:
                        continue
                    send_op = dist.P2POp(dist.isend, replica, i)
                    requests.append(send_op)
                    if local_rank_of_i <= k - 1:
                        recv_op = dist.P2POp(
                            dist.irecv,
                            dst_tensor[
                                local_rank_of_i
                                * new_replica_size : (local_rank_of_i + 1)
                                * new_replica_size
                            ],
                            i,
                        )
                        requests.append(recv_op)

        if len(requests) > 0:
            reqs = dist.batch_isend_irecv(requests)
            for req in reqs:
                req.wait()
    else:
        "case1"
        k = size // new_replica_size
        gap = size % new_replica_size
        # machine = global_rank // tasks_per_node
        # local_rank1 = k - 1
        "到顶的第一个device的local rank,在此处应该没用"
        requests = []
        if size % (tasks_per_node * new_replica_size) != 0:
            "对不齐"
            machine_x = size // (tasks_per_node * new_replica_size)
            "machine_x和以后的nodes会到顶, 并且和machine_x-k 有交叉的地方"

            if node_id < machine_x:
                "这些node没到顶"
                dst_tensor[
                    global_rank
                    * new_replica_size : (global_rank + 1)
                    * new_replica_size
                ] = replica
                if node_id == machine_x - 1:
                    for i in range(world_size):
                        if i == global_rank:
                            continue

                        "发送"
                        if gap == 0:
                            "size是new_replica_size的整数倍"
                            L1 = machine_x * tasks_per_node * new_replica_size
                            L2 = tasks_per_node * new_replica_size
                            leftover = L1 + L2 - size
                            a = leftover // new_replica_size
                            rank = machine_x * tasks_per_node - a - 1
                            if global_rank <= rank:
                                send_op = dist.P2POp(dist.isend, replica, i)
                                requests.append(send_op)
                            else:
                                "global_rank > rank"
                                if (i // tasks_per_node) < machine_x:
                                    send_op = dist.P2POp(dist.isend, replica, i)
                                    requests.append(send_op)
                        else:
                            "size不是new_replica_size的整数倍"
                            L1 = machine_x * tasks_per_node * new_replica_size
                            L2 = tasks_per_node * new_replica_size
                            leftover = L1 + L2 - size
                            a = leftover // new_replica_size
                            rank = machine_x * tasks_per_node - a - 1
                            if global_rank < rank:
                                send_op = dist.P2POp(dist.isend, replica, i)
                                requests.append(send_op)
                            elif global_rank == rank:
                                if (i // tasks_per_node) >= machine_x:
                                    send_op = dist.P2POp(dist.isend, replica[:gap], i)
                                    requests.append(send_op)
                                else:
                                    send_op = dist.P2POp(dist.isend, replica, i)
                                    requests.append(send_op)
                            else:
                                "global_rank > rank"
                                if (i // tasks_per_node) < machine_x:
                                    send_op = dist.P2POp(dist.isend, replica, i)
                                    requests.append(send_op)

                        "接收"
                        if (i // tasks_per_node) < machine_x:
                            recv_op = dist.P2POp(
                                dist.irecv,
                                dst_tensor[
                                    i * new_replica_size : (i + 1) * new_replica_size
                                ],
                                i,
                            )
                            requests.append(recv_op)
                        elif (i // tasks_per_node) == machine_x:
                            if gap == 0:
                                "size是new_replica_size的整数倍"
                                number = k - machine_x * tasks_per_node
                                upper_rank = (machine_x + 1) * tasks_per_node - 1
                                lower_rank = upper_rank - number + 1
                                if i >= lower_rank and i <= upper_rank:
                                    recv_op = dist.P2POp(
                                        dist.irecv,
                                        dst_tensor[
                                            i
                                            * new_replica_size : (i + 1)
                                            * new_replica_size
                                        ],
                                        i,
                                    )
                                    requests.append(recv_op)
                                continue
                            else:
                                "size不是new_replica_size的整数倍"
                                number = k - machine_x * tasks_per_node
                                upper_rank = (machine_x + 1) * tasks_per_node - 1
                                lower_rank = upper_rank - number + 1
                                if i >= lower_rank and i <= upper_rank:
                                    local_rank_of_i = i % tasks_per_node
                                    start = (
                                        size
                                        - tasks_per_node * new_replica_size
                                        + local_rank_of_i * new_replica_size
                                    )
                                    end = start + new_replica_size
                                    recv_op = dist.P2POp(
                                        dist.irecv, dst_tensor[start:end], i
                                    )
                                    requests.append(recv_op)
                                elif i == lower_rank - 1:
                                    local_rank_of_i = i % tasks_per_node
                                    start = (
                                        size
                                        - tasks_per_node * new_replica_size
                                        + local_rank_of_i * new_replica_size
                                        + new_replica_size
                                        - gap
                                    )
                                    end = start + gap
                                    recv_op = dist.P2POp(
                                        dist.irecv, dst_tensor[start:end], i
                                    )
                                    # recv_op = dist.P2POp(dist.irecv, dst_tensor[ -gap: ], i)
                                    requests.append(recv_op)

                else:
                    "node_id < machine_x - 1"
                    for i in range(world_size):
                        if i == global_rank:
                            continue
                        "发送"
                        send_op = dist.P2POp(dist.isend, replica, i)
                        requests.append(send_op)
                        "接收"
                        if (i // tasks_per_node) < machine_x:
                            recv_op = dist.P2POp(
                                dist.irecv,
                                dst_tensor[
                                    i * new_replica_size : (i + 1) * new_replica_size
                                ],
                                i,
                            )
                            requests.append(recv_op)
                        elif (i // tasks_per_node) == machine_x:
                            if gap == 0:
                                "aligned,存疑"
                                number = k - machine_x * tasks_per_node
                                upper_rank = (machine_x + 1) * tasks_per_node - 1
                                lower_rank = upper_rank - number + 1
                                if i >= lower_rank and i <= upper_rank:
                                    recv_op = dist.P2POp(
                                        dist.irecv,
                                        dst_tensor[
                                            i
                                            * new_replica_size : (i + 1)
                                            * new_replica_size
                                        ],
                                        i,
                                    )
                                    requests.append(recv_op)
                            else:
                                "not aligned"
                                number = k - machine_x * tasks_per_node
                                upper_rank = (machine_x + 1) * tasks_per_node - 1
                                lower_rank = upper_rank - number + 1
                                if i >= lower_rank and i <= upper_rank:
                                    local_rank_of_i = i % tasks_per_node
                                    start = (
                                        size
                                        - tasks_per_node * new_replica_size
                                        + local_rank_of_i * new_replica_size
                                    )
                                    end = start + new_replica_size
                                    recv_op = dist.P2POp(
                                        dist.irecv, dst_tensor[start:end], i
                                    )
                                    requests.append(recv_op)
                                if i == lower_rank - 1:
                                    local_rank_of_i = i % tasks_per_node
                                    start = (
                                        size
                                        - tasks_per_node * new_replica_size
                                        + (local_rank_of_i + 1) * new_replica_size
                                        - gap
                                    )
                                    end = start + gap
                                    # recv_op = dist.P2POp(dist.irecv, dst_tensor[ - gap : ], i)
                                    recv_op = dist.P2POp(
                                        dist.irecv, dst_tensor[start:end], i
                                    )
                                    requests.append(recv_op)

            else:
                "这些node到顶了"
                dst_tensor[
                    size
                    - tasks_per_node * new_replica_size
                    + local_rank * new_replica_size : size
                    - tasks_per_node * new_replica_size
                    + local_rank * new_replica_size
                    + new_replica_size
                ] = replica

                if node_id == machine_x:
                    "第一个到顶的node"
                    for i in range(world_size):
                        if i == global_rank:
                            continue

                        "发送给自己和之前的nodes"
                        if (
                            machine_x * tasks_per_node
                            <= i
                            < (machine_x + 1) * tasks_per_node
                        ):
                            "给同一个node的其他device发送"
                            send_op = dist.P2POp(dist.isend, replica, i)
                            requests.append(send_op)

                        elif i < machine_x * tasks_per_node:
                            "给之前的nodes发送"
                            "对global_rank进行判断,有三种类型"
                            xxx = (
                                tasks_per_node
                                - 1
                                - (
                                    size // new_replica_size
                                    - machine_x * tasks_per_node
                                )
                            )
                            if local_rank > xxx:
                                send_op = dist.P2POp(dist.isend, replica, i)
                                requests.append(send_op)
                            elif local_rank == xxx:
                                send_op = dist.P2POp(dist.isend, replica[-gap:], i)
                                requests.append(send_op)

                        "从自己和之前的nodes接收"
                        if (
                            machine_x * tasks_per_node
                            <= i
                            < (machine_x + 1) * tasks_per_node
                        ):
                            "给同一个node的其他device接收"
                            local_rank_of_i = i % tasks_per_node
                            start = (
                                size
                                - tasks_per_node * new_replica_size
                                + local_rank_of_i * new_replica_size
                            )
                            end = start + new_replica_size
                            recv_op = dist.P2POp(dist.irecv, dst_tensor[start:end], i)
                            requests.append(recv_op)
                        elif i < machine_x * tasks_per_node:
                            "有两种情况"
                            xxx = size // new_replica_size - tasks_per_node
                            "global rank of the corner device"
                            if i < xxx:
                                recv_op = dist.P2POp(
                                    dist.irecv,
                                    dst_tensor[
                                        i
                                        * new_replica_size : (i + 1)
                                        * new_replica_size
                                    ],
                                    i,
                                )
                                requests.append(recv_op)
                            elif i == xxx:
                                recv_op = dist.P2POp(
                                    dist.irecv,
                                    dst_tensor[
                                        i * new_replica_size : i * new_replica_size
                                        + gap
                                    ],
                                    i,
                                )
                                requests.append(recv_op)
                else:
                    "第i>1个到顶的node"
                    for i in range(world_size):
                        if i == global_rank:
                            continue
                        "发送给自己"
                        if (
                            node_id * tasks_per_node
                            <= i
                            < (node_id + 1) * tasks_per_node
                        ):
                            "给同一个node的其他device发送"
                            send_op = dist.P2POp(dist.isend, replica, i)
                            requests.append(send_op)

                        "从自己和machine_x之前的nodes接收"
                        if (
                            node_id * tasks_per_node
                            <= i
                            < (node_id + 1) * tasks_per_node
                        ):
                            "给同一个node的其他device接收"
                            local_rank_of_i = i % tasks_per_node
                            start = (
                                size
                                - tasks_per_node * new_replica_size
                                + local_rank_of_i * new_replica_size
                            )
                            end = start + new_replica_size
                            recv_op = dist.P2POp(dist.irecv, dst_tensor[start:end], i)
                            requests.append(recv_op)
                        elif i < machine_x * tasks_per_node:
                            "有两种情况"
                            xxx = size // new_replica_size - tasks_per_node
                            "global rank of the corner device"
                            if i < xxx:
                                recv_op = dist.P2POp(
                                    dist.irecv,
                                    dst_tensor[
                                        i
                                        * new_replica_size : (i + 1)
                                        * new_replica_size
                                    ],
                                    i,
                                )
                                requests.append(recv_op)
                            elif i == xxx:
                                recv_op = dist.P2POp(
                                    dist.irecv,
                                    dst_tensor[
                                        i * new_replica_size : i * new_replica_size
                                        + gap
                                    ],
                                    i,
                                )
                                requests.append(recv_op)

        else:
            "size % (tasks_per_node * new_replica_size) == 0"
            machine_x = size // (tasks_per_node * new_replica_size)
            "machine_x和以后的nodes会到顶, 并且和machine_x-k 没有交叉的地方"
            if node_id <= machine_x:
                dst_tensor[
                    global_rank
                    * new_replica_size : (global_rank + 1)
                    * new_replica_size
                ] = replica
                for i in range(world_size):
                    if i == global_rank:
                        continue
                    "发送"
                    if node_id < machine_x:
                        send_op = dist.P2POp(dist.isend, replica, i)
                        requests.append(send_op)
                    else:
                        "node_id == machine_x"
                        if i < (machine_x + 1) * tasks_per_node:
                            send_op = dist.P2POp(dist.isend, replica, i)
                            requests.append(send_op)
                    "接收"
                    if i < (machine_x + 1) * tasks_per_node:
                        recv_op = dist.P2POp(
                            dist.irecv,
                            dst_tensor[
                                i * new_replica_size : (i + 1) * new_replica_size
                            ],
                            i,
                        )
                        requests.append(recv_op)

            else:
                "machine>machine_x"
                dst_tensor[size - tasks_per_node * new_replica_size :] = replica
                for i in range(world_size):
                    "给同一个node里的其他device发送"
                    if (
                        i >= node_id * tasks_per_node
                        and i < (node_id + 1) * tasks_per_node
                    ):
                        send_op = dist.P2POp(dist.isend, replica, i)
                        requests.append(send_op)

                    "接收"
                    if (
                        i >= node_id * tasks_per_node
                        and i < (node_id + 1) * tasks_per_node
                    ):
                        local_rank_of_i = i % tasks_per_node
                        start = (
                            size
                            - tasks_per_node * new_replica_size
                            + local_rank_of_i * new_replica_size
                        )
                        end = start + new_replica_size
                        recv_op = dist.P2POp(dist.irecv, dst_tensor[start:end], i)
                        requests.append(recv_op)
                    elif i < (machine_x - 1) * tasks_per_node:
                        recv_op = dist.P2POp(
                            dist.irecv,
                            dst_tensor[
                                i * new_replica_size : (i + 1) * new_replica_size
                            ],
                            i,
                        )
                        requests.append(recv_op)

        if len(requests) > 0:
            reqs = dist.batch_isend_irecv(requests)
            for req in reqs:
                req.wait()


def get_replica_size(size, world_size, replica_rate):
    replica_size = int(size * replica_rate)
    replica_size = (replica_size - (replica_size % world_size)) + (size % world_size)
    return replica_size


def run(local_rank, world_size, args):
    setup_mgpu(local_rank, world_size)
    torch.cuda.set_device(local_rank)
    global_rank = local_rank
    size = (args.size + world_size - 1) // world_size * world_size

    replica_size = get_replica_size(size, world_size, args.replica_rate)
    assert (size - replica_size) % world_size == 0
    replica = torch.ones(replica_size, device=local_rank)
    # sharded tensor (no replica)
    in_tensor = torch.ones(size // world_size, device=local_rank) * (global_rank + 1)
    # sharded tensor (with replica)
    in_tensor_small = torch.ones(
        (size - replica_size) // world_size, device=local_rank
    ) * (global_rank + 1)
    if args.validate:
        overall_buffer = torch.ones(size, device=local_rank) * -1
        size_per_device = size // world_size
        size_per_device_with_replica = (size - replica_size) // world_size
        for i in range(world_size):
            overall_buffer[i * size_per_device : (i + 1) * size_per_device] = i + 1
        if replica_size > 0:
            replica = overall_buffer[-replica_size:]
        else:
            replica = torch.empty(0)
        in_tensor_small = overall_buffer[
            global_rank
            * size_per_device_with_replica : (global_rank + 1)
            * size_per_device_with_replica
        ]
        # in_tensor = overall_buffer[rank * size_per_device:(rank + 1) * size_per_device]
    out_tensor = torch.zeros(size, device=local_rank)
    transfered_bytes = size * 4 * ((world_size - 1) / world_size)
    transfered_bytes_with_replica = (
        (size - replica_size) * 4 * ((world_size - 1) / world_size)
    )

    storage_rate = 0.25 + 0.75 * args.replica_rate
    storage_rate_surrogate = ((size - replica_size) // world_size + replica_size) / size
    vanilla_comm = 0
    replica_comm = 0
    replica_comm_2 = 0
    replica_comm_diff_device = 0

    for i in range(args.iter):
        t0 = time.time()
        dist.all_gather_into_tensor(
            out_tensor,
            in_tensor,
        )
        # '当size不是world_size的整数倍时直接使用dist.all_gather_into_tensor会出错'

        torch.cuda.synchronize(local_rank)
        if args.validate and i == 0:
            assert torch.allclose(
                out_tensor, overall_buffer
            ), f"allgather failed {out_tensor} {overall_buffer}"
            print("allgather passed")
        if global_rank == 0 and i == args.iter - 1:
            t1 = time.time()
            print(
                "No replica: time",
                t1 - t0,
                "bandwidth(GB/s)",
                transfered_bytes / (t1 - t0) / 1024**3,
            )
            vanilla_comm = t1 - t0
    for i in range(args.iter):
        t0 = time.time()
        all_gather_with_replica(
            out_tensor,
            in_tensor_small,
            replica,
        )
        torch.cuda.synchronize(local_rank)
        if args.validate and i == 0:
            assert torch.allclose(
                out_tensor, overall_buffer
            ), f"allgather with replica failed {out_tensor} {overall_buffer}"
            print("allgather with replica passed")
        if global_rank == 0 and i == args.iter - 1:
            t1 = time.time()
            print(
                f"replica {args.replica_rate}: time",
                t1 - t0,
                "bandwidth(GB/s)",
                transfered_bytes_with_replica / (t1 - t0) / 1024**3,
            )
            replica_comm = t1 - t0
    for i in range(args.iter):
        t0 = time.time()
        all_gather_with_replica_2(
            out_tensor, in_tensor_small, replica, local_rank, world_size
        )
        torch.cuda.synchronize(local_rank)
        if args.validate and i == 0:
            assert torch.allclose(
                out_tensor, overall_buffer
            ), f"allgather with replica (not using dist.all_gather_into_tensor) failed {out_tensor} {overall_buffer}"
            print(
                "allgather with replica (not using dist.all_gather_into_tensor) passed"
            )
        if global_rank == 0 and i == args.iter - 1:
            t1 = time.time()
            print(
                f"dist.allgather surrogate replica {args.replica_rate}: time",
                t1 - t0,
                "bandwidth(GB/s)",
                transfered_bytes_with_replica / (t1 - t0) / 1024**3,
            )

    new_replica_size = (size - replica_size) // world_size + replica_size

    replica_small_diff_devices = torch.ones(new_replica_size, device=local_rank)
    l3 = (size - 2 * new_replica_size) // 2
    if args.validate:
        overall_buffer = torch.ones(size, device=local_rank) * -1
        size_per_device = size // world_size
        for i in range(world_size):
            overall_buffer[i * size_per_device : (i + 1) * size_per_device] = i + 1
        if size <= 2 * new_replica_size:
            if global_rank % 2 == 0:
                replica_small_diff_devices = overall_buffer[:new_replica_size]
            if global_rank % 2 == 1:
                replica_small_diff_devices = overall_buffer[-new_replica_size:]
        else:
            if global_rank == 0:
                replica_small_diff_devices = overall_buffer[:new_replica_size]
            if global_rank == 1:
                replica_small_diff_devices = overall_buffer[-new_replica_size:]
            if global_rank == 2:
                replica_small_diff_devices = overall_buffer[l3 : l3 + new_replica_size]
            if global_rank == 3:
                replica_small_diff_devices = overall_buffer[
                    l3 + new_replica_size : l3 + 2 * new_replica_size
                ]
            if global_rank == 4:
                replica_small_diff_devices = overall_buffer[:new_replica_size]
            if global_rank == 5:
                replica_small_diff_devices = overall_buffer[-new_replica_size:]
            if global_rank == 6:
                replica_small_diff_devices = overall_buffer[l3 : l3 + new_replica_size]
            if global_rank == 7:
                replica_small_diff_devices = overall_buffer[
                    l3 + new_replica_size : l3 + 2 * new_replica_size
                ]

    for i in range(args.iter):
        t0 = time.time()
        all_gather_with_replica_diff_devices_2(
            out_tensor, replica_small_diff_devices, local_rank, world_size
        )
        torch.cuda.synchronize(local_rank)
        if args.validate and i == 0:
            assert torch.allclose(
                out_tensor, overall_buffer
            ), f"device aware allgather with replica failed {out_tensor} {overall_buffer}"
            print("device aware allgather with replica passed")
        if global_rank == 0 and i == args.iter - 1:
            t1 = time.time()
            print(
                f"device aware replica {args.replica_rate}: time",
                t1 - t0,
                "bandwidth(GB/s)",
                transfered_bytes_with_replica / (t1 - t0) / 1024**3,
            )
            replica_comm_diff_device = t1 - t0

            "保存数据的代码"
            # import numpy
            # numbers = numpy.array([storage_rate,storage_rate_surrogate,vanilla_comm,replica_comm,replica_comm_2,replica_comm_diff_device])
            # txt_path = "data2.txt"
            # with open(txt_path, "a") as file:
            #     file.write(" ".join(map(str, numbers)) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--iter", type=int, default=3)
    parser.add_argument("--replica_rate", type=float, default=0)
    parser.add_argument("--size", type=int, default=100000000)
    parser.add_argument("--validate", type=int, default=1)
    args = parser.parse_args()

    # world_size, local_rank, global_rank = setup()
    # run(local_rank, global_rank, world_size, args)

    WORLD_SIZE = torch.cuda.device_count()
    mp.spawn(run, args=(WORLD_SIZE, args), nprocs=WORLD_SIZE, join=True)
