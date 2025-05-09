"""
This file includes communication utilities for FSDP.
"""

import torch.distributed as dist


def get_start_end(length, new_replica_size, rank, world_size):
    comm_size = length - new_replica_size

    if comm_size <= new_replica_size:
        if rank % 2 == 0:
            start = 0
            end = new_replica_size - 1
        else:
            start = comm_size
            end = length - 1
    elif comm_size <= 3 * new_replica_size and (world_size == 4 or world_size == 8):
        l3 = (length - 2 * new_replica_size) // 2
        l4 = length - 2 * new_replica_size - l3
        if rank % 4 == 0:
            start = 0
            end = new_replica_size - 1
        elif rank % 4 == 1:
            start = length - new_replica_size
            end = length - 1
        elif rank % 4 == 2:
            start = l3
            end = l3 + new_replica_size - 1
        elif rank % 4 == 3:
            start = l3 + new_replica_size
            end = l3 + 2 * new_replica_size - 1
    elif comm_size <= 3 * new_replica_size and world_size == 6:
        l3 = (length - 2 * new_replica_size) // 2
        l4 = length - 2 * new_replica_size - l3
        l5 = l3 // 2
        l6 = l4 // 2
        if rank == 0:
            start = 0
            end = new_replica_size - 1
        elif rank == 1:
            start = length - new_replica_size
            end = length - 1
        elif rank == 2:
            start = l3
            end = l3 + new_replica_size - 1
        elif rank == 3:
            start = l3 + new_replica_size
            end = l3 + 2 * new_replica_size - 1
        elif rank == 4:
            start = l5
            end = l5 + new_replica_size - 1
        elif rank == 5:
            start = length - l6 - new_replica_size
            end = length - l6 - 1
    elif world_size == 6:
        l3 = (length - 4 * new_replica_size) // 2
        l4 = length - 4 * new_replica_size - l3
        l5 = (new_replica_size - l3) // 2
        l6 = (new_replica_size - l4) // 2

        if rank == 0:
            start = 0
            end = new_replica_size - 1
        elif rank == 1:
            start = l3 + new_replica_size
            end = l3 + 2 * new_replica_size - 1
        elif rank == 2:
            start = l3 + 2 * new_replica_size
            end = l3 + 3 * new_replica_size - 1
        elif rank == 3:
            start = length - new_replica_size
            end = length - 1
        elif rank == 4:
            start = new_replica_size - l5
            end = start + new_replica_size - 1
        elif rank == 5:
            start = 3 * new_replica_size + l3 - l6
            end = start + new_replica_size - 1

    elif world_size == 8:
        l3 = (length - 4 * new_replica_size) // 2
        l4 = length - 4 * new_replica_size - l3
        l5 = l3 // 2
        l6 = l4 // 2

        if rank == 0:
            start = 0
            end = new_replica_size - 1
        elif rank == 1:
            start = l3 + new_replica_size
            end = l3 + 2 * new_replica_size - 1
        elif rank == 2:
            start = l3 + 2 * new_replica_size
            end = l3 + 3 * new_replica_size - 1
        elif rank == 3:
            start = length - new_replica_size
            end = length - 1
        if rank == 4:
            start = l5
            end = l5 + new_replica_size - 1
        elif rank == 5:
            start = l5 + new_replica_size
            end = l5 + 2 * new_replica_size - 1
        elif rank == 6:
            start = l3 + l6 + 2 * new_replica_size
            end = l3 + l6 + 3 * new_replica_size - 1
        elif rank == 7:
            start = l3 + l6 + 3 * new_replica_size
            end = l3 + l6 + 4 * new_replica_size - 1
    return start, end


def device_aware_communication(
    size,
    new_replica_size,
    padded_unsharded_flat_param,
    sharded_flat_param,
    global_rank,
    world_size,
    requests,
):
    comm_size = size - new_replica_size

    if comm_size <= new_replica_size:
        if global_rank % 2 == 0:
            """local_rank == 0"""
            send_op = dist.P2POp(
                dist.isend, sharded_flat_param[:comm_size], global_rank + 1
            )
            requests.append(send_op)
            recv_op = dist.P2POp(
                dist.irecv,
                padded_unsharded_flat_param[-comm_size:],
                global_rank + 1,
            )
            requests.append(recv_op)
        else:
            """local_rank == 1"""
            send_op = dist.P2POp(
                dist.isend, sharded_flat_param[-comm_size:], global_rank - 1
            )
            requests.append(send_op)
            recv_op = dist.P2POp(
                dist.irecv,
                padded_unsharded_flat_param[:comm_size],
                global_rank - 1,
            )
            requests.append(recv_op)
    elif comm_size <= 3 * new_replica_size and (world_size == 4 or world_size == 8):
        """
        4卡的时候必然满足该条件, 8卡有可能不满足
        """
        l3 = (size - 2 * new_replica_size) // 2
        l3_hat = (l3 + 8 - 1) // 8 * 8
        l4 = size - 2 * new_replica_size - l3
        l4_hat = (l4 + 8 - 1) // 8 * 8
        # print("l,l hat", l3, l3_hat, l4, l4_hat)

        if global_rank % 4 == 0:
            padded_unsharded_flat_param[:new_replica_size] = sharded_flat_param

            "和1通信"
            send_op = dist.P2POp(dist.isend, sharded_flat_param, global_rank + 1)
            requests.append(send_op)
            recv_op = dist.P2POp(
                dist.irecv,
                padded_unsharded_flat_param[-new_replica_size:],
                global_rank + 1,
            )
            requests.append(recv_op)

            "和2通信"
            send_op = dist.P2POp(
                dist.isend, sharded_flat_param[:l3_hat], global_rank + 2
            )
            requests.append(send_op)
            recv_op = dist.P2POp(
                dist.irecv,
                padded_unsharded_flat_param[
                    new_replica_size + l3 - l3_hat : new_replica_size + l3
                ],
                global_rank + 2,
            )
            requests.append(recv_op)

            "和3通信"
            send_op = dist.P2POp(
                dist.isend, sharded_flat_param[:l3_hat], global_rank + 3
            )
            requests.append(send_op)
            recv_op = dist.P2POp(
                dist.irecv,
                padded_unsharded_flat_param[
                    new_replica_size + l3 : new_replica_size + l3 + l4_hat
                ],
                global_rank + 3,
            )
            requests.append(recv_op)

        elif global_rank % 4 == 1:
            padded_unsharded_flat_param[-new_replica_size:] = sharded_flat_param

            "和0通信"
            send_op = dist.P2POp(dist.isend, sharded_flat_param, global_rank - 1)
            requests.append(send_op)
            recv_op = dist.P2POp(
                dist.irecv,
                padded_unsharded_flat_param[:new_replica_size],
                global_rank - 1,
            )
            requests.append(recv_op)

            "和2通信"
            send_op = dist.P2POp(
                dist.isend, sharded_flat_param[-l4_hat:], global_rank + 1
            )
            requests.append(send_op)
            recv_op = dist.P2POp(
                dist.irecv,
                padded_unsharded_flat_param[
                    new_replica_size + l3 - l3_hat : new_replica_size + l3
                ],
                global_rank + 1,
            )
            requests.append(recv_op)

            "和3通信"
            send_op = dist.P2POp(
                dist.isend, sharded_flat_param[-l4_hat:], global_rank + 2
            )
            requests.append(send_op)

            recv_op = dist.P2POp(
                dist.irecv,
                padded_unsharded_flat_param[
                    new_replica_size + l3 : new_replica_size + l3 + l4_hat
                ],
                global_rank + 2,
            )
            requests.append(recv_op)

        elif global_rank % 4 == 2:
            padded_unsharded_flat_param[l3 : new_replica_size + l3] = sharded_flat_param

            "和0通信"
            send_op = dist.P2POp(
                dist.isend, sharded_flat_param[-l3_hat:], global_rank - 2
            )
            requests.append(send_op)
            recv_op = dist.P2POp(
                dist.irecv,
                padded_unsharded_flat_param[:l3_hat],
                global_rank - 2,
            )
            requests.append(recv_op)

            "和1通信"
            send_op = dist.P2POp(
                dist.isend, sharded_flat_param[-l3_hat:], global_rank - 1
            )
            requests.append(send_op)
            recv_op = dist.P2POp(
                dist.irecv,
                padded_unsharded_flat_param[-l4_hat:],
                global_rank - 1,
            )
            requests.append(recv_op)

            "和3通信"
            send_op = dist.P2POp(
                dist.isend,
                sharded_flat_param,
                global_rank + 1,
            )
            requests.append(send_op)
            recv_op = dist.P2POp(
                dist.irecv,
                padded_unsharded_flat_param[new_replica_size + l3 : size - l4],
                global_rank + 1,
            )
            requests.append(recv_op)

        elif global_rank % 4 == 3:
            padded_unsharded_flat_param[
                new_replica_size + l3 : 2 * new_replica_size + l3
            ] = sharded_flat_param
            "和0通信"
            send_op = dist.P2POp(
                dist.isend, sharded_flat_param[:l4_hat], global_rank - 3
            )
            requests.append(send_op)
            recv_op = dist.P2POp(
                dist.irecv,
                padded_unsharded_flat_param[:l3_hat],
                global_rank - 3,
            )
            requests.append(recv_op)

            "和1通信"
            send_op = dist.P2POp(
                dist.isend, sharded_flat_param[:l4_hat], global_rank - 2
            )
            requests.append(send_op)
            recv_op = dist.P2POp(
                dist.irecv,
                padded_unsharded_flat_param[-l4_hat:],
                global_rank - 2,
            )
            requests.append(recv_op)

            "和2通信"
            recv_op = dist.P2POp(
                dist.irecv,
                padded_unsharded_flat_param[l3 : new_replica_size + l3],
                global_rank - 1,
            )
            requests.append(recv_op)
            send_op = dist.P2POp(dist.isend, sharded_flat_param, global_rank - 1)
            requests.append(send_op)
    elif comm_size <= 3 * new_replica_size and world_size == 6:
        l3 = (size - 2 * new_replica_size) // 2
        l3_hat = (l3 + 8 - 1) // 8 * 8
        l4 = size - 2 * new_replica_size - l3
        l4_hat = (l4 + 8 - 1) // 8 * 8

        if global_rank == 0:
            padded_unsharded_flat_param[:new_replica_size] = sharded_flat_param

            "和1通信"
            send_op = dist.P2POp(dist.isend, sharded_flat_param, global_rank + 1)
            requests.append(send_op)
            recv_op = dist.P2POp(
                dist.irecv,
                padded_unsharded_flat_param[-new_replica_size:],
                global_rank + 1,
            )
            requests.append(recv_op)

            "和2通信"
            send_op = dist.P2POp(
                dist.isend, sharded_flat_param[:l3_hat], global_rank + 2
            )
            requests.append(send_op)
            recv_op = dist.P2POp(
                dist.irecv,
                padded_unsharded_flat_param[
                    new_replica_size + l3 - l3_hat : new_replica_size + l3
                ],
                global_rank + 2,
            )
            requests.append(recv_op)

            "和3通信"
            send_op = dist.P2POp(
                dist.isend, sharded_flat_param[:l3_hat], global_rank + 3
            )
            requests.append(send_op)
            recv_op = dist.P2POp(
                dist.irecv,
                padded_unsharded_flat_param[
                    new_replica_size + l3 : new_replica_size + l3 + l4_hat
                ],
                global_rank + 3,
            )
            requests.append(recv_op)

            "和4通信"
            send_op = dist.P2POp(
                dist.isend, sharded_flat_param[:l5_hat], global_rank + 4
            )
            requests.append(send_op)

            "和5通信"
            send_op = dist.P2POp(
                dist.isend, sharded_flat_param[:l5_hat], global_rank + 5
            )
            requests.append(send_op)

        elif global_rank == 1:
            padded_unsharded_flat_param[
                new_replica_size : 2 * new_replica_size
            ] = sharded_flat_param

            "和0通信"
            send_op = dist.P2POp(dist.isend, sharded_flat_param, global_rank - 1)
            requests.append(send_op)
            recv_op = dist.P2POp(
                dist.irecv,
                padded_unsharded_flat_param[:new_replica_size],
                global_rank - 1,
            )
            requests.append(recv_op)

            "和2通信"
            send_op = dist.P2POp(
                dist.isend, sharded_flat_param[-l4_hat:], global_rank + 1
            )
            requests.append(send_op)
            recv_op = dist.P2POp(
                dist.irecv,
                padded_unsharded_flat_param[
                    new_replica_size + l3 - l3_hat : new_replica_size + l3
                ],
                global_rank + 1,
            )
            requests.append(recv_op)

            "和3通信"
            send_op = dist.P2POp(
                dist.isend, sharded_flat_param[-l4_hat:], global_rank + 2
            )
            requests.append(send_op)

            recv_op = dist.P2POp(
                dist.irecv,
                padded_unsharded_flat_param[
                    new_replica_size + l3 : new_replica_size + l3 + l4_hat
                ],
                global_rank + 2,
            )
            requests.append(recv_op)

            "和4通信"
            send_op = dist.P2POp(
                dist.isend, sharded_flat_param[-l6_hat:], global_rank + 3
            )
            requests.append(send_op)

            "和5通信"
            send_op = dist.P2POp(
                dist.isend, sharded_flat_param[-l6_hat:], global_rank + 4
            )
            requests.append(send_op)

        elif global_rank == 2:
            padded_unsharded_flat_param[
                -2 * new_replica_size : -new_replica_size
            ] = sharded_flat_param

            "和0通信"
            send_op = dist.P2POp(
                dist.isend, sharded_flat_param[-l3_hat:], global_rank - 2
            )
            requests.append(send_op)
            recv_op = dist.P2POp(
                dist.irecv,
                padded_unsharded_flat_param[:l3_hat],
                global_rank - 2,
            )
            requests.append(recv_op)

            "和1通信"
            send_op = dist.P2POp(
                dist.isend, sharded_flat_param[-l3_hat:], global_rank - 1
            )
            requests.append(send_op)
            recv_op = dist.P2POp(
                dist.irecv,
                padded_unsharded_flat_param[-l4_hat:],
                global_rank - 1,
            )
            requests.append(recv_op)

            "和3通信"
            send_op = dist.P2POp(
                dist.isend,
                sharded_flat_param,
                global_rank + 1,
            )
            requests.append(send_op)
            recv_op = dist.P2POp(
                dist.irecv,
                padded_unsharded_flat_param[new_replica_size + l3 : size - l4],
                global_rank + 1,
            )
            requests.append(recv_op)

            "和4通信"
            send_op = dist.P2POp(
                dist.isend,
                sharded_flat_param[-((l3 - l5 + 7) // 8 * 8) :],
                global_rank + 2,
            )
            requests.append(send_op)

            "和5通信"
            send_op = dist.P2POp(
                dist.isend,
                sharded_flat_param[-((l3 - l5 + 7) // 8 * 8) :],
                global_rank + 3,
            )
            requests.append(send_op)

        elif global_rank == 3:
            padded_unsharded_flat_param[-new_replica_size:] = sharded_flat_param
            "和0通信"
            send_op = dist.P2POp(
                dist.isend, sharded_flat_param[:l4_hat], global_rank - 3
            )
            requests.append(send_op)
            recv_op = dist.P2POp(
                dist.irecv,
                padded_unsharded_flat_param[:l3_hat],
                global_rank - 3,
            )
            requests.append(recv_op)

            "和1通信"
            send_op = dist.P2POp(
                dist.isend, sharded_flat_param[:l4_hat], global_rank - 2
            )
            requests.append(send_op)
            recv_op = dist.P2POp(
                dist.irecv,
                padded_unsharded_flat_param[-l4_hat:],
                global_rank - 2,
            )
            requests.append(recv_op)

            "和2通信"
            recv_op = dist.P2POp(
                dist.irecv,
                padded_unsharded_flat_param[l3 : new_replica_size + l3],
                global_rank - 1,
            )
            requests.append(recv_op)
            send_op = dist.P2POp(dist.isend, sharded_flat_param, global_rank - 1)
            requests.append(send_op)

            "和4通信"
            send_op = dist.P2POp(
                dist.isend,
                sharded_flat_param[: ((l4 - l6 + 7) // 8 * 8)],
                global_rank + 1,
            )
            requests.append(send_op)

            "和5通信"
            send_op = dist.P2POp(
                dist.isend,
                sharded_flat_param[: ((l4 - l6 + 7) // 8 * 8)],
                global_rank + 2,
            )
            requests.append(send_op)

        elif global_rank == 4:
            padded_unsharded_flat_param[
                new_replica_size + l3 : 2 * new_replica_size + l3
            ] = sharded_flat_param
            "和0通信"
            recv_op = dist.P2POp(
                dist.irecv,
                padded_unsharded_flat_param[:l5_hat],
                global_rank - 4,
            )
            requests.append(recv_op)

            "和1通信"
            recv_op = dist.P2POp(
                dist.irecv,
                padded_unsharded_flat_param[-l6_hat:],
                global_rank - 3,
            )
            requests.append(recv_op)

            "和2通信"
            recv_op = dist.P2POp(
                dist.irecv,
                padded_unsharded_flat_param[
                    new_replica_size
                    + l3
                    - ((l3 - l5 + 7) // 8 * 8) : new_replica_size
                    + l3
                ],
                global_rank - 2,
            )
            requests.append(recv_op)

            "和3通信"
            recv_op = dist.P2POp(
                dist.irecv,
                padded_unsharded_flat_param[
                    new_replica_size
                    + l3 : new_replica_size
                    + l3
                    + ((l4 - l6 + 7) // 8 * 8)
                ],
                global_rank - 1,
            )
            requests.append(recv_op)

            "和5通信"
            send_op = dist.P2POp(
                dist.isend,
                sharded_flat_param,
                global_rank + 1,
            )
            requests.append(send_op)

            recv_op = dist.P2POp(
                dist.irecv,
                padded_unsharded_flat_param[size - l6 - new_replica_size : size - l6],
                global_rank + 1,
            )
            requests.append(recv_op)

        elif global_rank == 5:
            padded_unsharded_flat_param[
                2 * new_replica_size + l3 : 3 * new_replica_size + l3
            ] = sharded_flat_param
            "和0通信"
            recv_op = dist.P2POp(
                dist.irecv,
                padded_unsharded_flat_param[:l5_hat],
                global_rank - 5,
            )
            requests.append(recv_op)

            "和1通信"
            recv_op = dist.P2POp(
                dist.irecv,
                padded_unsharded_flat_param[-l6_hat:],
                global_rank - 4,
            )
            requests.append(recv_op)

            "和2通信"
            recv_op = dist.P2POp(
                dist.irecv,
                padded_unsharded_flat_param[
                    new_replica_size
                    + l3
                    - ((l3 - l5 + 7) // 8 * 8) : new_replica_size
                    + l3
                ],
                global_rank - 3,
            )
            requests.append(recv_op)

            "和3通信"
            recv_op = dist.P2POp(
                dist.irecv,
                padded_unsharded_flat_param[
                    new_replica_size
                    + l3 : new_replica_size
                    + l3
                    + ((l4 - l6 + 7) // 8 * 8)
                ],
                global_rank - 2,
            )
            requests.append(recv_op)

            "和4通信"
            send_op = dist.P2POp(
                dist.isend,
                sharded_flat_param,
                global_rank - 1,
            )
            requests.append(send_op)

            recv_op = dist.P2POp(
                dist.irecv,
                padded_unsharded_flat_param[l5 : new_replica_size + l5],
                global_rank - 1,
            )
            requests.append(recv_op)
    elif world_size == 6:
        l3 = (size - 4 * new_replica_size) // 2
        l3_hat = (l3 + 7) // 8 * 8
        l4 = size - 4 * new_replica_size - l3
        l4_hat = (l4 + 7) // 8 * 8
        l5 = (new_replica_size - l3) // 2
        l5_hat = (l5 + 7) // 8 * 8
        l6 = (new_replica_size - l4) // 2
        l6_hat = (l6 + 7) // 8 * 8

        if global_rank == 0:
            padded_unsharded_flat_param[:new_replica_size] = sharded_flat_param

            "和1通信"
            send_op = dist.P2POp(dist.isend, sharded_flat_param, global_rank + 1)
            requests.append(send_op)
            recv_op = dist.P2POp(
                dist.irecv,
                padded_unsharded_flat_param[
                    l3 + new_replica_size : l3 + 2 * new_replica_size
                ],
                global_rank + 1,
            )
            requests.append(recv_op)

            "和2通信"
            send_op = dist.P2POp(dist.isend, sharded_flat_param, global_rank + 2)
            requests.append(send_op)
            recv_op = dist.P2POp(
                dist.irecv,
                padded_unsharded_flat_param[
                    l3 + 2 * new_replica_size : l3 + 3 * new_replica_size
                ],
                global_rank + 2,
            )
            requests.append(recv_op)

            "和3通信"
            send_op = dist.P2POp(dist.isend, sharded_flat_param, global_rank + 3)
            requests.append(send_op)
            recv_op = dist.P2POp(
                dist.irecv,
                padded_unsharded_flat_param[-new_replica_size:],
                global_rank + 3,
            )
            requests.append(recv_op)

            "和4通信"
            send_op = dist.P2POp(
                dist.isend,
                sharded_flat_param[: (new_replica_size - l5 + 7) // 8 * 8],
                global_rank + 4,
            )
            requests.append(send_op)
            recv_op = dist.P2POp(
                dist.irecv,
                padded_unsharded_flat_param[
                    new_replica_size : new_replica_size + l3_hat
                ],
                global_rank + 4,
            )
            requests.append(recv_op)

            "和5通信"
            send_op = dist.P2POp(
                dist.isend,
                sharded_flat_param[: (new_replica_size - l5 + 7) // 8 * 8],
                global_rank + 5,
            )
            requests.append(send_op)
            recv_op = dist.P2POp(
                dist.irecv,
                padded_unsharded_flat_param[
                    l3 + 3 * new_replica_size : l3 + 3 * new_replica_size + l4_hat
                ],
                global_rank + 5,
            )
            requests.append(recv_op)

        elif global_rank == 1:
            padded_unsharded_flat_param[
                l3 + new_replica_size : l3 + 2 * new_replica_size
            ] = sharded_flat_param

            "和0通信"
            send_op = dist.P2POp(dist.isend, sharded_flat_param, global_rank - 1)
            requests.append(send_op)
            recv_op = dist.P2POp(
                dist.irecv,
                padded_unsharded_flat_param[:new_replica_size],
                global_rank - 1,
            )
            requests.append(recv_op)

            "和2通信"
            send_op = dist.P2POp(dist.isend, sharded_flat_param, global_rank + 1)
            requests.append(send_op)
            recv_op = dist.P2POp(
                dist.irecv,
                padded_unsharded_flat_param[
                    2 * new_replica_size + l3 : 3 * new_replica_size + l3
                ],
                global_rank + 1,
            )
            requests.append(recv_op)

            "和3通信"
            send_op = dist.P2POp(dist.isend, sharded_flat_param, global_rank + 2)
            requests.append(send_op)

            recv_op = dist.P2POp(
                dist.irecv,
                padded_unsharded_flat_param[-new_replica_size:],
                global_rank + 2,
            )
            requests.append(recv_op)

            "和4通信"
            send_op = dist.P2POp(
                dist.isend,
                sharded_flat_param[(new_replica_size - l3 - l5) // 8 * 8 :],
                global_rank + 3,
            )
            requests.append(send_op)
            recv_op = dist.P2POp(
                dist.irecv,
                padded_unsharded_flat_param[
                    new_replica_size : new_replica_size + l3_hat
                ],
                global_rank + 3,
            )
            requests.append(recv_op)

            "和5通信"
            send_op = dist.P2POp(
                dist.isend,
                sharded_flat_param[(new_replica_size - l3 - l5) // 8 * 8 :],
                global_rank + 4,
            )
            requests.append(send_op)
            recv_op = dist.P2POp(
                dist.irecv,
                padded_unsharded_flat_param[
                    l3 + 3 * new_replica_size : l3 + 3 * new_replica_size + l4_hat
                ],
                global_rank + 4,
            )
            requests.append(recv_op)

        elif global_rank == 2:
            padded_unsharded_flat_param[
                2 * new_replica_size + l3 : 3 * new_replica_size + l3
            ] = sharded_flat_param

            "和0通信"
            send_op = dist.P2POp(dist.isend, sharded_flat_param, global_rank - 2)
            requests.append(send_op)
            recv_op = dist.P2POp(
                dist.irecv,
                padded_unsharded_flat_param[:new_replica_size],
                global_rank - 2,
            )
            requests.append(recv_op)

            "和1通信"
            send_op = dist.P2POp(dist.isend, sharded_flat_param, global_rank - 1)
            requests.append(send_op)
            recv_op = dist.P2POp(
                dist.irecv,
                padded_unsharded_flat_param[
                    l3 + new_replica_size : l3 + 2 * new_replica_size
                ],
                global_rank - 1,
            )
            requests.append(recv_op)

            "和3通信"
            send_op = dist.P2POp(
                dist.isend,
                sharded_flat_param,
                global_rank + 1,
            )
            requests.append(send_op)
            recv_op = dist.P2POp(
                dist.irecv,
                padded_unsharded_flat_param[-new_replica_size:],
                global_rank + 1,
            )
            requests.append(recv_op)

            "和4通信"
            send_op = dist.P2POp(
                dist.isend,
                sharded_flat_param[: (new_replica_size - l6 + 7) // 8 * 8],
                global_rank + 2,
            )
            requests.append(send_op)
            recv_op = dist.P2POp(
                dist.irecv,
                padded_unsharded_flat_param[
                    new_replica_size : new_replica_size + l3_hat
                ],
                global_rank + 2,
            )
            requests.append(recv_op)

            "和5通信"
            send_op = dist.P2POp(
                dist.isend,
                sharded_flat_param[: (new_replica_size - l6 + 7) // 8 * 8],
                global_rank + 3,
            )
            requests.append(send_op)
            recv_op = dist.P2POp(
                dist.irecv,
                padded_unsharded_flat_param[
                    l3 + 3 * new_replica_size : l3 + 3 * new_replica_size + l4_hat
                ],
                global_rank + 3,
            )
            requests.append(recv_op)

        elif global_rank == 3:
            padded_unsharded_flat_param[-new_replica_size:] = sharded_flat_param
            "和0通信"
            send_op = dist.P2POp(dist.isend, sharded_flat_param, global_rank - 3)
            requests.append(send_op)
            recv_op = dist.P2POp(
                dist.irecv,
                padded_unsharded_flat_param[:new_replica_size],
                global_rank - 3,
            )
            requests.append(recv_op)

            "和1通信"
            send_op = dist.P2POp(dist.isend, sharded_flat_param, global_rank - 2)
            requests.append(send_op)
            recv_op = dist.P2POp(
                dist.irecv,
                padded_unsharded_flat_param[
                    l3 + new_replica_size : l3 + 2 * new_replica_size
                ],
                global_rank - 2,
            )
            requests.append(recv_op)

            "和2通信"
            recv_op = dist.P2POp(
                dist.irecv,
                padded_unsharded_flat_param[
                    l3 + 2 * new_replica_size : l3 + 3 * new_replica_size
                ],
                global_rank - 1,
            )
            requests.append(recv_op)
            send_op = dist.P2POp(dist.isend, sharded_flat_param, global_rank - 1)
            requests.append(send_op)

            "和4通信"
            send_op = dist.P2POp(
                dist.isend,
                sharded_flat_param[(new_replica_size - l6) // 8 * 8 :],
                global_rank + 1,
            )
            requests.append(send_op)
            recv_op = dist.P2POp(
                dist.irecv,
                padded_unsharded_flat_param[
                    new_replica_size : new_replica_size + l3_hat
                ],
                global_rank + 1,
            )
            requests.append(recv_op)

            "和5通信"
            send_op = dist.P2POp(
                dist.isend,
                sharded_flat_param[(new_replica_size - l6) // 8 * 8 :],
                global_rank + 2,
            )
            requests.append(send_op)
            recv_op = dist.P2POp(
                dist.irecv,
                padded_unsharded_flat_param[
                    l3 + 3 * new_replica_size : l3 + 3 * new_replica_size + l4_hat
                ],
                global_rank + 2,
            )
            requests.append(recv_op)

        elif global_rank == 4:
            padded_unsharded_flat_param[
                new_replica_size - l5 : 2 * new_replica_size - l5
            ] = sharded_flat_param

            "和0通信"
            send_op = dist.P2POp(
                dist.isend,
                sharded_flat_param[l5 : l5 + l3_hat],
                global_rank - 4,
            )
            requests.append(send_op)
            recv_op = dist.P2POp(
                dist.irecv,
                padded_unsharded_flat_param[: (new_replica_size - l5 + 7) // 8 * 8],
                global_rank - 4,
            )
            requests.append(recv_op)

            "和1通信"
            send_op = dist.P2POp(
                dist.isend,
                sharded_flat_param[l5 : l5 + l3_hat],
                global_rank - 3,
            )
            requests.append(send_op)
            recv_op = dist.P2POp(
                dist.irecv,
                padded_unsharded_flat_param[
                    l3
                    + new_replica_size
                    + (new_replica_size - l3 - l5) // 8 * 8 : l3
                    + 2 * new_replica_size
                ],
                global_rank - 3,
            )
            requests.append(recv_op)

            "和2通信"
            send_op = dist.P2POp(
                dist.isend,
                sharded_flat_param[l5 : l5 + l3_hat],
                global_rank - 2,
            )
            requests.append(send_op)
            recv_op = dist.P2POp(
                dist.irecv,
                padded_unsharded_flat_param[
                    l3
                    + 2 * new_replica_size : l3
                    + 2 * new_replica_size
                    + (new_replica_size - l6 + 7) // 8 * 8
                ],
                global_rank - 2,
            )
            requests.append(recv_op)

            "和3通信"
            send_op = dist.P2POp(
                dist.isend,
                sharded_flat_param[l5 : l5 + l3_hat],
                global_rank - 1,
            )
            requests.append(send_op)

            recv_op = dist.P2POp(
                dist.irecv,
                padded_unsharded_flat_param[
                    (new_replica_size - l6) // 8 * 8 - new_replica_size :
                ],
                global_rank - 1,
            )
            requests.append(recv_op)

            "和5通信"
            send_op = dist.P2POp(
                dist.isend,
                sharded_flat_param,
                global_rank + 1,
            )
            requests.append(send_op)

            recv_op = dist.P2POp(
                dist.irecv,
                padded_unsharded_flat_param[size - l6 - new_replica_size : size - l6],
                global_rank + 1,
            )
            requests.append(recv_op)

        elif global_rank == 5:
            padded_unsharded_flat_param[
                l3 + 3 * new_replica_size - l6 : l3 + 4 * new_replica_size - l6
            ] = sharded_flat_param

            "和0通信"
            send_op = dist.P2POp(
                dist.isend,
                sharded_flat_param[l6 : l6 + l4_hat],
                global_rank - 5,
            )
            requests.append(send_op)
            recv_op = dist.P2POp(
                dist.irecv,
                padded_unsharded_flat_param[: (new_replica_size - l5 + 7) // 8 * 8],
                global_rank - 5,
            )
            requests.append(recv_op)

            "和1通信"
            send_op = dist.P2POp(
                dist.isend,
                sharded_flat_param[l6 : l6 + l4_hat],
                global_rank - 4,
            )
            requests.append(send_op)
            recv_op = dist.P2POp(
                dist.irecv,
                padded_unsharded_flat_param[
                    l3
                    + new_replica_size
                    + (new_replica_size - l3 - l5) // 8 * 8 : l3
                    + 2 * new_replica_size
                ],
                global_rank - 4,
            )
            requests.append(recv_op)

            "和2通信"
            send_op = dist.P2POp(
                dist.isend,
                sharded_flat_param[l6 : l6 + l4_hat],
                global_rank - 3,
            )
            requests.append(send_op)
            recv_op = dist.P2POp(
                dist.irecv,
                padded_unsharded_flat_param[
                    l3
                    + 2 * new_replica_size : l3
                    + 2 * new_replica_size
                    + (new_replica_size - l6 + 7) // 8 * 8
                ],
                global_rank - 3,
            )
            requests.append(recv_op)

            "和3通信"
            send_op = dist.P2POp(
                dist.isend,
                sharded_flat_param[l6 : l6 + l4_hat],
                global_rank - 2,
            )
            requests.append(send_op)

            recv_op = dist.P2POp(
                dist.irecv,
                padded_unsharded_flat_param[
                    (new_replica_size - l6) // 8 * 8 - new_replica_size :
                ],
                global_rank - 2,
            )
            requests.append(recv_op)

            "和4通信"
            send_op = dist.P2POp(
                dist.isend,
                sharded_flat_param,
                global_rank - 1,
            )
            requests.append(send_op)

            recv_op = dist.P2POp(
                dist.irecv,
                padded_unsharded_flat_param[
                    new_replica_size - l5 : 2 * new_replica_size - l5
                ],
                global_rank - 1,
            )
            requests.append(recv_op)

    elif world_size == 8:
        l3 = (size - 4 * new_replica_size) // 2
        # l3_hat = (l3 + 7) // 8 * 8
        l4 = size - 4 * new_replica_size - l3
        # l4_hat = (l4 + 7) // 8 * 8
        l5 = l3 // 2
        l5_hat = (l5 + 7) // 8 * 8
        l6 = l4 // 2
        l6_hat = (l6 + 7) // 8 * 8

        if global_rank == 0:
            padded_unsharded_flat_param[:new_replica_size] = sharded_flat_param

            "和1通信, checked"
            send_op = dist.P2POp(dist.isend, sharded_flat_param, global_rank + 1)
            requests.append(send_op)
            recv_op = dist.P2POp(
                dist.irecv,
                padded_unsharded_flat_param[
                    l3 + new_replica_size : l3 + 2 * new_replica_size
                ],
                global_rank + 1,
            )
            requests.append(recv_op)

            "和2通信, checked"
            send_op = dist.P2POp(dist.isend, sharded_flat_param, global_rank + 2)
            requests.append(send_op)
            recv_op = dist.P2POp(
                dist.irecv,
                padded_unsharded_flat_param[
                    l3 + 2 * new_replica_size : l3 + 3 * new_replica_size
                ],
                global_rank + 2,
            )
            requests.append(recv_op)

            "和3通信, checked"
            send_op = dist.P2POp(dist.isend, sharded_flat_param, global_rank + 3)
            requests.append(send_op)
            recv_op = dist.P2POp(
                dist.irecv,
                padded_unsharded_flat_param[-new_replica_size:],
                global_rank + 3,
            )
            requests.append(recv_op)

            "和4通信"
            send_op = dist.P2POp(
                dist.isend,
                sharded_flat_param[:l5_hat],
                global_rank + 4,
            )
            requests.append(send_op)
            recv_op = dist.P2POp(
                dist.irecv,
                padded_unsharded_flat_param[
                    l5 - l5_hat + new_replica_size : l5 + new_replica_size
                ],
                global_rank + 4,
            )
            requests.append(recv_op)

            "和5通信"
            send_op = dist.P2POp(
                dist.isend,
                sharded_flat_param[:l5_hat],
                global_rank + 5,
            )
            requests.append(send_op)
            recv_op = dist.P2POp(
                dist.irecv,
                padded_unsharded_flat_param[
                    l5 + new_replica_size : l5 + l5_hat + new_replica_size
                ],
                global_rank + 5,
            )
            requests.append(recv_op)

            "和6通信"
            send_op = dist.P2POp(
                dist.isend,
                sharded_flat_param[:l5_hat],
                global_rank + 6,
            )
            requests.append(send_op)
            recv_op = dist.P2POp(
                dist.irecv,
                padded_unsharded_flat_param[
                    l3
                    + l6
                    + 3 * new_replica_size
                    - l6_hat : l3
                    + l6
                    + 3 * new_replica_size
                ],
                global_rank + 6,
            )
            requests.append(recv_op)

            "和7通信"
            send_op = dist.P2POp(
                dist.isend,
                sharded_flat_param[:l5_hat],
                global_rank + 7,
            )
            requests.append(send_op)
            recv_op = dist.P2POp(
                dist.irecv,
                padded_unsharded_flat_param[
                    l3
                    + l6
                    + 3 * new_replica_size : l3
                    + l6
                    + 3 * new_replica_size
                    + l6_hat
                ],
                global_rank + 7,
            )
            requests.append(recv_op)

        elif global_rank == 1:
            padded_unsharded_flat_param[
                l3 + new_replica_size : l3 + 2 * new_replica_size
            ] = sharded_flat_param

            "和0通信,checked"
            send_op = dist.P2POp(dist.isend, sharded_flat_param, global_rank - 1)
            requests.append(send_op)
            recv_op = dist.P2POp(
                dist.irecv,
                padded_unsharded_flat_param[:new_replica_size],
                global_rank - 1,
            )
            requests.append(recv_op)

            "和2通信, checked"
            send_op = dist.P2POp(dist.isend, sharded_flat_param, global_rank + 1)
            requests.append(send_op)
            recv_op = dist.P2POp(
                dist.irecv,
                padded_unsharded_flat_param[
                    2 * new_replica_size + l3 : 3 * new_replica_size + l3
                ],
                global_rank + 1,
            )
            requests.append(recv_op)

            "和3通信,checked"
            send_op = dist.P2POp(dist.isend, sharded_flat_param, global_rank + 2)
            requests.append(send_op)

            recv_op = dist.P2POp(
                dist.irecv,
                padded_unsharded_flat_param[-new_replica_size:],
                global_rank + 2,
            )
            requests.append(recv_op)

            "和4通信"
            send_op = dist.P2POp(
                dist.isend,
                sharded_flat_param[-l5_hat:],
                global_rank + 3,
            )
            requests.append(send_op)
            recv_op = dist.P2POp(
                dist.irecv,
                padded_unsharded_flat_param[
                    l5 - l5_hat + new_replica_size : l5 + new_replica_size
                ],
                global_rank + 3,
            )
            requests.append(recv_op)

            "和5通信"
            send_op = dist.P2POp(
                dist.isend,
                sharded_flat_param[-l5_hat:],
                global_rank + 4,
            )
            requests.append(send_op)
            recv_op = dist.P2POp(
                dist.irecv,
                padded_unsharded_flat_param[
                    l5 + new_replica_size : l5 + l5_hat + new_replica_size
                ],
                global_rank + 4,
            )
            requests.append(recv_op)

            "和6通信"
            send_op = dist.P2POp(
                dist.isend,
                sharded_flat_param[-l5_hat:],
                global_rank + 5,
            )
            requests.append(send_op)
            recv_op = dist.P2POp(
                dist.irecv,
                padded_unsharded_flat_param[
                    l3
                    + l6
                    + 3 * new_replica_size
                    - l6_hat : l3
                    + l6
                    + 3 * new_replica_size
                ],
                global_rank + 5,
            )
            requests.append(recv_op)

            "和7通信"
            send_op = dist.P2POp(
                dist.isend,
                sharded_flat_param[-l5_hat:],
                global_rank + 6,
            )
            requests.append(send_op)
            recv_op = dist.P2POp(
                dist.irecv,
                padded_unsharded_flat_param[
                    l3
                    + l6
                    + 3 * new_replica_size : l3
                    + l6
                    + 3 * new_replica_size
                    + l6_hat
                ],
                global_rank + 6,
            )
            requests.append(recv_op)

        elif global_rank == 2:
            padded_unsharded_flat_param[
                2 * new_replica_size + l3 : 3 * new_replica_size + l3
            ] = sharded_flat_param

            "和0通信,check"
            send_op = dist.P2POp(dist.isend, sharded_flat_param, global_rank - 2)
            requests.append(send_op)
            recv_op = dist.P2POp(
                dist.irecv,
                padded_unsharded_flat_param[:new_replica_size],
                global_rank - 2,
            )
            requests.append(recv_op)

            "和1通信,check"
            send_op = dist.P2POp(dist.isend, sharded_flat_param, global_rank - 1)
            requests.append(send_op)
            recv_op = dist.P2POp(
                dist.irecv,
                padded_unsharded_flat_param[
                    l3 + new_replica_size : l3 + 2 * new_replica_size
                ],
                global_rank - 1,
            )
            requests.append(recv_op)

            "和3通信,check"
            send_op = dist.P2POp(
                dist.isend,
                sharded_flat_param,
                global_rank + 1,
            )
            requests.append(send_op)
            recv_op = dist.P2POp(
                dist.irecv,
                padded_unsharded_flat_param[-new_replica_size:],
                global_rank + 1,
            )
            requests.append(recv_op)

            "和4通信"
            send_op = dist.P2POp(
                dist.isend,
                sharded_flat_param[:l6_hat],
                global_rank + 2,
            )
            requests.append(send_op)
            recv_op = dist.P2POp(
                dist.irecv,
                padded_unsharded_flat_param[
                    l5 - l5_hat + new_replica_size : l5 + new_replica_size
                ],
                global_rank + 2,
            )
            requests.append(recv_op)

            "和5通信"
            send_op = dist.P2POp(
                dist.isend,
                sharded_flat_param[:l6_hat],
                global_rank + 3,
            )
            requests.append(send_op)
            recv_op = dist.P2POp(
                dist.irecv,
                padded_unsharded_flat_param[
                    l5 + new_replica_size : l5 + l5_hat + new_replica_size
                ],
                global_rank + 3,
            )
            requests.append(recv_op)

            "和6通信"
            send_op = dist.P2POp(
                dist.isend,
                sharded_flat_param[:l6_hat],
                global_rank + 4,
            )
            requests.append(send_op)
            recv_op = dist.P2POp(
                dist.irecv,
                padded_unsharded_flat_param[
                    l3
                    + l6
                    + 3 * new_replica_size
                    - l6_hat : l3
                    + l6
                    + 3 * new_replica_size
                ],
                global_rank + 4,
            )
            requests.append(recv_op)

            "和7通信"
            send_op = dist.P2POp(
                dist.isend,
                sharded_flat_param[:l6_hat],
                global_rank + 5,
            )
            requests.append(send_op)
            recv_op = dist.P2POp(
                dist.irecv,
                padded_unsharded_flat_param[
                    l3
                    + l6
                    + 3 * new_replica_size : l3
                    + l6
                    + 3 * new_replica_size
                    + l6_hat
                ],
                global_rank + 5,
            )
            requests.append(recv_op)

        elif global_rank == 3:
            padded_unsharded_flat_param[-new_replica_size:] = sharded_flat_param
            "和0通信"
            send_op = dist.P2POp(dist.isend, sharded_flat_param, global_rank - 3)
            requests.append(send_op)
            recv_op = dist.P2POp(
                dist.irecv,
                padded_unsharded_flat_param[:new_replica_size],
                global_rank - 3,
            )
            requests.append(recv_op)

            "和1通信"
            send_op = dist.P2POp(dist.isend, sharded_flat_param, global_rank - 2)
            requests.append(send_op)
            recv_op = dist.P2POp(
                dist.irecv,
                padded_unsharded_flat_param[
                    l3 + new_replica_size : l3 + 2 * new_replica_size
                ],
                global_rank - 2,
            )
            requests.append(recv_op)

            "和2通信"
            recv_op = dist.P2POp(
                dist.irecv,
                padded_unsharded_flat_param[
                    l3 + 2 * new_replica_size : l3 + 3 * new_replica_size
                ],
                global_rank - 1,
            )
            requests.append(recv_op)
            send_op = dist.P2POp(dist.isend, sharded_flat_param, global_rank - 1)
            requests.append(send_op)

            "和4通信"
            send_op = dist.P2POp(
                dist.isend,
                sharded_flat_param[-l6_hat:],
                global_rank + 1,
            )
            requests.append(send_op)
            recv_op = dist.P2POp(
                dist.irecv,
                padded_unsharded_flat_param[
                    l5 - l5_hat + new_replica_size : l5 + new_replica_size
                ],
                global_rank + 1,
            )
            requests.append(recv_op)

            "和5通信"
            send_op = dist.P2POp(
                dist.isend,
                sharded_flat_param[-l6_hat:],
                global_rank + 2,
            )
            requests.append(send_op)
            recv_op = dist.P2POp(
                dist.irecv,
                padded_unsharded_flat_param[
                    l5 + new_replica_size : l5 + l5_hat + new_replica_size
                ],
                global_rank + 2,
            )
            requests.append(recv_op)

            "和6通信"
            send_op = dist.P2POp(
                dist.isend,
                sharded_flat_param[-l6_hat:],
                global_rank + 3,
            )
            requests.append(send_op)
            recv_op = dist.P2POp(
                dist.irecv,
                padded_unsharded_flat_param[
                    l3
                    + l6
                    + 3 * new_replica_size
                    - l6_hat : l3
                    + l6
                    + 3 * new_replica_size
                ],
                global_rank + 3,
            )
            requests.append(recv_op)

            "和7通信"
            send_op = dist.P2POp(
                dist.isend,
                sharded_flat_param[-l6_hat:],
                global_rank + 4,
            )
            requests.append(send_op)
            recv_op = dist.P2POp(
                dist.irecv,
                padded_unsharded_flat_param[
                    l3
                    + l6
                    + 3 * new_replica_size : l3
                    + l6
                    + 3 * new_replica_size
                    + l6_hat
                ],
                global_rank + 4,
            )
            requests.append(recv_op)

        elif global_rank == 4:
            padded_unsharded_flat_param[l5 : new_replica_size + l5] = sharded_flat_param

            "和0通信"
            send_op = dist.P2POp(
                dist.isend,
                sharded_flat_param[-l5_hat:],
                global_rank - 4,
            )
            requests.append(send_op)
            recv_op = dist.P2POp(
                dist.irecv,
                padded_unsharded_flat_param[:l5_hat],
                global_rank - 4,
            )
            requests.append(recv_op)

            "和1通信"
            send_op = dist.P2POp(
                dist.isend,
                sharded_flat_param[-l5_hat:],
                global_rank - 3,
            )
            requests.append(send_op)
            recv_op = dist.P2POp(
                dist.irecv,
                padded_unsharded_flat_param[
                    l3 + 2 * new_replica_size - l5_hat : l3 + 2 * new_replica_size
                ],
                global_rank - 3,
            )
            requests.append(recv_op)

            "和2通信"
            send_op = dist.P2POp(
                dist.isend,
                sharded_flat_param[-l5_hat:],
                global_rank - 2,
            )
            requests.append(send_op)
            recv_op = dist.P2POp(
                dist.irecv,
                padded_unsharded_flat_param[
                    l3 + 2 * new_replica_size : l3 + 2 * new_replica_size + l6_hat
                ],
                global_rank - 2,
            )
            requests.append(recv_op)

            "和3通信"
            send_op = dist.P2POp(
                dist.isend,
                sharded_flat_param[-l5_hat:],
                global_rank - 1,
            )
            requests.append(send_op)

            recv_op = dist.P2POp(
                dist.irecv,
                padded_unsharded_flat_param[-l6_hat:],
                global_rank - 1,
            )
            requests.append(recv_op)

            "和5通信"
            send_op = dist.P2POp(
                dist.isend,
                sharded_flat_param,
                global_rank + 1,
            )
            requests.append(send_op)

            recv_op = dist.P2POp(
                dist.irecv,
                padded_unsharded_flat_param[
                    l5 + new_replica_size : l5 + 2 * new_replica_size
                ],
                global_rank + 1,
            )
            requests.append(recv_op)

            "和6通信"
            send_op = dist.P2POp(
                dist.isend,
                sharded_flat_param,
                global_rank + 2,
            )
            requests.append(send_op)

            recv_op = dist.P2POp(
                dist.irecv,
                padded_unsharded_flat_param[
                    l3 + l6 + 2 * new_replica_size : l3 + l6 + 3 * new_replica_size
                ],
                global_rank + 2,
            )
            requests.append(recv_op)

            "和7通信"
            send_op = dist.P2POp(
                dist.isend,
                sharded_flat_param,
                global_rank + 3,
            )
            requests.append(send_op)

            recv_op = dist.P2POp(
                dist.irecv,
                padded_unsharded_flat_param[
                    l3 + l6 + 3 * new_replica_size : l3 + l6 + 4 * new_replica_size
                ],
                global_rank + 3,
            )
            requests.append(recv_op)

        elif global_rank == 5:
            padded_unsharded_flat_param[
                new_replica_size + l5 : 2 * new_replica_size + l5
            ] = sharded_flat_param

            "和0通信"
            send_op = dist.P2POp(
                dist.isend,
                sharded_flat_param[:l5_hat],
                global_rank - 5,
            )
            requests.append(send_op)
            recv_op = dist.P2POp(
                dist.irecv,
                padded_unsharded_flat_param[:l5_hat],
                global_rank - 5,
            )
            requests.append(recv_op)

            "和1通信"
            send_op = dist.P2POp(
                dist.isend,
                sharded_flat_param[:l5_hat],
                global_rank - 4,
            )
            requests.append(send_op)
            recv_op = dist.P2POp(
                dist.irecv,
                padded_unsharded_flat_param[
                    l3 + 2 * new_replica_size - l5_hat : l3 + 2 * new_replica_size
                ],
                global_rank - 4,
            )
            requests.append(recv_op)

            "和2通信"
            send_op = dist.P2POp(
                dist.isend,
                sharded_flat_param[:l5_hat],
                global_rank - 3,
            )
            requests.append(send_op)
            recv_op = dist.P2POp(
                dist.irecv,
                padded_unsharded_flat_param[
                    l3 + 2 * new_replica_size : l3 + 2 * new_replica_size + l6_hat
                ],
                global_rank - 3,
            )
            requests.append(recv_op)

            "和3通信"
            send_op = dist.P2POp(
                dist.isend,
                sharded_flat_param[:l5_hat],
                global_rank - 2,
            )
            requests.append(send_op)

            recv_op = dist.P2POp(
                dist.irecv,
                padded_unsharded_flat_param[-l6_hat:],
                global_rank - 2,
            )
            requests.append(recv_op)

            "和4通信"
            send_op = dist.P2POp(
                dist.isend,
                sharded_flat_param,
                global_rank - 1,
            )
            requests.append(send_op)

            recv_op = dist.P2POp(
                dist.irecv,
                padded_unsharded_flat_param[l5 : l5 + new_replica_size],
                global_rank - 1,
            )
            requests.append(recv_op)

            "和6通信"
            send_op = dist.P2POp(
                dist.isend,
                sharded_flat_param,
                global_rank + 1,
            )
            requests.append(send_op)

            recv_op = dist.P2POp(
                dist.irecv,
                padded_unsharded_flat_param[
                    l3 + l6 + 2 * new_replica_size : l3 + l6 + 3 * new_replica_size
                ],
                global_rank + 1,
            )
            requests.append(recv_op)

            "和7通信"
            send_op = dist.P2POp(
                dist.isend,
                sharded_flat_param,
                global_rank + 2,
            )
            requests.append(send_op)

            recv_op = dist.P2POp(
                dist.irecv,
                padded_unsharded_flat_param[
                    l3 + l6 + 3 * new_replica_size : l3 + l6 + 4 * new_replica_size
                ],
                global_rank + 2,
            )
            requests.append(recv_op)

        elif global_rank == 6:
            padded_unsharded_flat_param[
                2 * new_replica_size + l3 + l6 : 3 * new_replica_size + l3 + l6
            ] = sharded_flat_param

            "和0通信"
            send_op = dist.P2POp(
                dist.isend,
                sharded_flat_param[-l6_hat:],
                global_rank - 6,
            )
            requests.append(send_op)
            recv_op = dist.P2POp(
                dist.irecv,
                padded_unsharded_flat_param[:l5_hat],
                global_rank - 6,
            )
            requests.append(recv_op)

            "和1通信"
            send_op = dist.P2POp(
                dist.isend,
                sharded_flat_param[-l6_hat:],
                global_rank - 5,
            )
            requests.append(send_op)
            recv_op = dist.P2POp(
                dist.irecv,
                padded_unsharded_flat_param[
                    l3 + 2 * new_replica_size - l5_hat : l3 + 2 * new_replica_size
                ],
                global_rank - 5,
            )
            requests.append(recv_op)

            "和2通信"
            send_op = dist.P2POp(
                dist.isend,
                sharded_flat_param[-l6_hat:],
                global_rank - 4,
            )
            requests.append(send_op)
            recv_op = dist.P2POp(
                dist.irecv,
                padded_unsharded_flat_param[
                    l3 + 2 * new_replica_size : l3 + 2 * new_replica_size + l6_hat
                ],
                global_rank - 4,
            )
            requests.append(recv_op)

            "和3通信"
            send_op = dist.P2POp(
                dist.isend,
                sharded_flat_param[-l6_hat:],
                global_rank - 3,
            )
            requests.append(send_op)

            recv_op = dist.P2POp(
                dist.irecv,
                padded_unsharded_flat_param[-l6_hat:],
                global_rank - 3,
            )
            requests.append(recv_op)

            "和4通信"
            send_op = dist.P2POp(
                dist.isend,
                sharded_flat_param,
                global_rank - 2,
            )
            requests.append(send_op)

            recv_op = dist.P2POp(
                dist.irecv,
                padded_unsharded_flat_param[l5 : l5 + new_replica_size],
                global_rank - 2,
            )
            requests.append(recv_op)

            "和5通信"
            send_op = dist.P2POp(
                dist.isend,
                sharded_flat_param,
                global_rank - 1,
            )
            requests.append(send_op)

            recv_op = dist.P2POp(
                dist.irecv,
                padded_unsharded_flat_param[
                    l5 + new_replica_size : l5 + 2 * new_replica_size
                ],
                global_rank - 1,
            )
            requests.append(recv_op)

            "和7通信"
            send_op = dist.P2POp(
                dist.isend,
                sharded_flat_param,
                global_rank + 1,
            )
            requests.append(send_op)

            recv_op = dist.P2POp(
                dist.irecv,
                padded_unsharded_flat_param[
                    l3 + l6 + 3 * new_replica_size : l3 + l6 + 4 * new_replica_size
                ],
                global_rank + 1,
            )
            requests.append(recv_op)

        elif global_rank == 7:
            padded_unsharded_flat_param[
                3 * new_replica_size + l3 + l6 : 4 * new_replica_size + l3 + l6
            ] = sharded_flat_param

            "和0通信"
            send_op = dist.P2POp(
                dist.isend,
                sharded_flat_param[:l6_hat],
                global_rank - 7,
            )
            requests.append(send_op)
            recv_op = dist.P2POp(
                dist.irecv,
                padded_unsharded_flat_param[:l5_hat],
                global_rank - 7,
            )
            requests.append(recv_op)

            "和1通信"
            send_op = dist.P2POp(
                dist.isend,
                sharded_flat_param[:l6_hat],
                global_rank - 6,
            )
            requests.append(send_op)
            recv_op = dist.P2POp(
                dist.irecv,
                padded_unsharded_flat_param[
                    l3 + 2 * new_replica_size - l5_hat : l3 + 2 * new_replica_size
                ],
                global_rank - 6,
            )
            requests.append(recv_op)

            "和2通信"
            send_op = dist.P2POp(
                dist.isend,
                sharded_flat_param[:l6_hat],
                global_rank - 5,
            )
            requests.append(send_op)
            recv_op = dist.P2POp(
                dist.irecv,
                padded_unsharded_flat_param[
                    l3 + 2 * new_replica_size : l3 + 2 * new_replica_size + l6_hat
                ],
                global_rank - 5,
            )
            requests.append(recv_op)

            "和3通信"
            send_op = dist.P2POp(
                dist.isend,
                sharded_flat_param[:l6_hat],
                global_rank - 4,
            )
            requests.append(send_op)

            recv_op = dist.P2POp(
                dist.irecv,
                padded_unsharded_flat_param[-l6_hat:],
                global_rank - 4,
            )
            requests.append(recv_op)

            "和4通信"
            send_op = dist.P2POp(
                dist.isend,
                sharded_flat_param,
                global_rank - 3,
            )
            requests.append(send_op)

            recv_op = dist.P2POp(
                dist.irecv,
                padded_unsharded_flat_param[l5 : l5 + new_replica_size],
                global_rank - 3,
            )
            requests.append(recv_op)

            "和5通信"
            send_op = dist.P2POp(
                dist.isend,
                sharded_flat_param,
                global_rank - 2,
            )
            requests.append(send_op)

            recv_op = dist.P2POp(
                dist.irecv,
                padded_unsharded_flat_param[
                    l5 + new_replica_size : l5 + 2 * new_replica_size
                ],
                global_rank - 2,
            )
            requests.append(recv_op)

            "和6通信"
            send_op = dist.P2POp(
                dist.isend,
                sharded_flat_param,
                global_rank - 1,
            )
            requests.append(send_op)

            recv_op = dist.P2POp(
                dist.irecv,
                padded_unsharded_flat_param[
                    l3 + l6 + 2 * new_replica_size : l3 + l6 + 3 * new_replica_size
                ],
                global_rank - 1,
            )
            requests.append(recv_op)

    if len(requests) > 0:
        reqs = dist.batch_isend_irecv(requests)
        for req in reqs:
            req.wait()
