import numpy as np
import argparse
import pickle

from megatron.impl import Implementation, Result, ModelConfig

import time

t0 = time.time()

parser = argparse.ArgumentParser(description='Process model parameters.')

# Add arguments
parser.add_argument('--model_size', type=int, required=True,
                    choices=[7, 13, 30, 70],
                    help='Size of the model (small, medium, large, xl)')

parser.add_argument('--seq_len', type=int, default=8192,
                    help='Sequence length (default: 8192)')

# Parse the arguments
args = parser.parse_args()

# Access the values
print(f"Model size: {args.model_size}")
print(f"Sequence length: {args.seq_len}")

seq_len = args.seq_len
model_size = args.model_size
batch_size = 8192 // seq_len
# key: modelsize, 
# divider: tp size, fsdp size # 就是简单的除法
# value: weight mem, activation mem


def get_model_config(model_size):
    if model_size == 7:
        return ModelConfig(seq_len=seq_len, hidden_size=4096, num_heads=32, num_query_groups=32, num_layers=32, ffn_hidden_size=11008, norm_eps=1e-5, model_size=model_size)
    elif model_size == 13:
        return ModelConfig(seq_len=seq_len, hidden_size=5120, num_heads=40, num_query_groups=40, num_layers=40, ffn_hidden_size=13824, norm_eps=1e-5, model_size=model_size)
    elif model_size == 30:
        return ModelConfig(seq_len=seq_len, hidden_size=6656, num_heads=64, num_query_groups=64, num_layers=60, ffn_hidden_size=17920, norm_eps=1e-5, model_size=model_size)
    elif model_size == 70:
        return ModelConfig(seq_len=seq_len, hidden_size=8192, num_heads=64, num_query_groups=8, num_layers=80, ffn_hidden_size=28672, norm_eps=1e-5, model_size=model_size)
    else:
        raise ValueError(f"Unsupported model size: {model_size}")
    
def gen_impls(model_size, batch_size, set_gather_to=-1):
    model_config = get_model_config(model_size)
    impls = []
    if set_gather_to == -1:
        gather_list = [8, 4, 2, 1]
    else:
        gather_list = [set_gather_to]
    for gather_to in gather_list:
        for discard_to in [8, 4, 2, 1]:
            for bwd_discard_to in [8, 4, 2, 1]:
                if discard_to > gather_to:
                    continue
                if bwd_discard_to > gather_to:
                    continue
                if bwd_discard_to < discard_to:
                    continue
                impl = Implementation(model_config, gather_to, discard_to, bwd_discard_to, recompute=False, batch_size=batch_size)
                if impl.valid:
                    impls.append(impl) # H100 no recompute
                    print(len(impls), impl)
    return impls


def elastic_tensor_schedule_search(L, M_total, implementations_per_layer, resolution):
    """
    Parameters:
        L (int): Number of layers
        M_total (int): Total GPU memory
        implementations_per_layer (List[List[Implementation]]): 
            A list of L elements, each containing a list of Implementation objects for that layer.

    Returns:
        (list, float): Selected implementations for each layer and total execution time
    """
    # Initialize dp table and choice tracking
    M_total *= resolution
    dp = np.full((L + 1, M_total + 1, M_total + 1), np.inf)
    choice = [[[-1] * (M_total + 1) for _ in range(M_total + 1)] for _ in range(L + 1)]

    dp[0, :, :] = 0  # Base case

    for i in range(1, L + 1):
        for mp in range(M_total + 1):
            for mv in range(M_total + 1):
                for j, impl in enumerate(implementations_per_layer[i - 1]):
                    if impl.peak_memory * resolution <= mp and impl.valley_memory * resolution <= mv:
                        prev_mp = int(mp - impl.peak_memory * resolution)
                        prev_mv = int(mv - impl.valley_memory * resolution)
                        # print(f"{mp=} {impl.peak_memory=} {mv=} {impl.valley_memory=}")
                        prev_cost = dp[i - 1, prev_mp, prev_mv]
                        new_cost = prev_cost + impl.exec_time
                        if new_cost < dp[i, mp, mv]:
                            dp[i, mp, mv] = new_cost
                            choice[i][mp][mv] = j

    # Find best result
    min_time = np.inf
    end_mp, end_mv = 0, 0
    for mp in range(M_total + 1):
        for mv in range(M_total + 1):
            if dp[L, mp, mv] < min_time:
                min_time = dp[L, mp, mv]
                end_mp, end_mv = mp, mv

    # Backtrack
    selected = [0] * L
    mp, mv = end_mp, end_mv
    for i in range(L, 0, -1):
        print(f"{i=} {mp=} {mv=}")
        j = choice[i][int(mp)][int(mv)]
        selected[i - 1] = j
        impl = implementations_per_layer[i - 1][j]
        mp -= impl.peak_memory * resolution
        mv -= impl.valley_memory * resolution

    return selected, min_time

# Define implementation list for each layer

if model_size > 30:
    resolution = 4
else:
    resolution = 2
mconfig = get_model_config(model_size)
L = mconfig.num_layers
single_layer = gen_impls(model_size=model_size, batch_size=batch_size)
layers = [single_layer for i in range(L)]
M_total = 80

selected, total_time = elastic_tensor_schedule_search(L, M_total, layers, resolution)
print("Selected implementations:", selected)
set_selected = list(set(selected))
for item in set_selected:
    print(item, single_layer[item])

# gather_tos = [single_layer[item].gather_to for item in selected]
# key = min(gather_tos)

counts = {}
for item in selected:
    impl = single_layer[item]
    gather_to = impl.gather_to
    if gather_to in counts:
        counts[gather_to] += 1
    else:
        counts[gather_to] = 1
m = 0
key = None
for k in counts:
    if m < counts[k]:
        key = k
        m = counts[k] 
print(f"{key=}")


single_layer = gen_impls(model_size=model_size, batch_size=batch_size, set_gather_to=key)
layers = [single_layer for i in range(L)]
selected, total_time = elastic_tensor_schedule_search(L, M_total, layers, resolution)
print("Selected implementations:", selected)

# print("Total execution time:", total_time)
res = Result(batch_size=batch_size, tp_size=8 // key, impls=[single_layer[item] for item in selected])
file_path = f"impls-{model_size}-{seq_len}.pkl"
with open(file_path, "wb") as f:
    pickle.dump(res, f)

set_selected = list(set(selected))
for item in set_selected:
    print(item, single_layer[item])
print(f"strategy dump to {file_path}, time cost {time.time() - t0}")