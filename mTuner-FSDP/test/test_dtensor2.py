# torchrun --standalone --nnodes=1 --nproc-per-node=4 xx.py
import torch
from torch.distributed._tensor import DTensor, Shard, Replicate, distribute_tensor, distribute_module, init_device_mesh
from torch.distributed.tensor.parallel import make_input_reshard_replicate, make_output_reshard_tensor
import time
import torch.distributed as dist
import os

device_mesh = init_device_mesh("cuda", (2, 2))
rowwise_placement=[Shard(0)]
replicate=[Replicate()]


local_rank = int(os.environ["LOCAL_RANK"])


local_tensor = torch.randn((16, 4), requires_grad=True)

t1 = DTensor.from_local(local_tensor, device_mesh, rowwise_placement)
if local_rank == 0:
    print(t1.shape, t1._local_tensor.shape)
t2 = t1.redistribute(device_mesh, replicate)
if local_rank == 0:
    print(t2.shape, t2._local_tensor.shape)

# tmp1 = DTensor.from_local(local_tensor, device_mesh, replicate) # total tensor : [16, 4] 
# tmp2 = make_input_reshard_replicate(local_tensor, device_mesh) # total tensor: [16 * 4, 4]

# tmp3 = make_output_reshard_tensor(tmp1, device_mesh)

# print(tmp1.shape, tmp2.shape, tmp3.shape)

exit()

def bw_bench():
    bs = 16
    seq_len = 1024
    hidden = 4096
    hidden_out = 11008
    num_device = world_size
    local_tensor = torch.randn((bs, seq_len, hidden), requires_grad=True)
    place = [Shard(0)]
    dtensor = DTensor.from_local(local_tensor, device_mesh, place)
    print(dtensor.shape, dtensor._local_tensor.shape)
    full_dtensor = make_input_reshard_replicate(dtensor, device_mesh)
    print(full_dtensor.shape, full_dtensor._local_tensor.shape)
    workload = bs * seq_len * hidden * (num_device - 1) / num_device * 4
    num_trial = 10
    t0 = time.time()
    for _ in range(num_trial):
        full_dtensor = make_input_reshard_replicate(dtensor, device_mesh)
    t1 = time.time()
    print("time: ", t1 - t0, "bandwidth: ", workload * num_trial / (t1 - t0) / 1e9, "GB/s")

bw_bench()