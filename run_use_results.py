import pickle
import subprocess

from megatron.impl import Implementation, Result, ModelConfig

import argparse

parser = argparse.ArgumentParser(description='Process model parameters.')

parser.add_argument('--model_size', type=int, required=True,
                    choices=[7, 13, 30, 70],
                    help='Size of the model (small, medium, large, xl)')

parser.add_argument('--seq_len', type=int, default=8192,
                    help='Sequence length (default: 8192)')

args = parser.parse_args()

res_path = f"results/impls-{args.model_size}-{args.seq_len}.pkl"

try:
    with open(res_path, "rb") as f:
        result = pickle.load(f)
except:
    print(f"Error: Fail load result from {res_path=}")
    exit()

# num_offset = 0
# for item in result.impls:
#     if item.discard_to == 8 // 4:
#     # if item.tp_size == config.tensor_model_parallel_size:
#         num_offset += 1
# print(num_offset)
# # exit()
env_vars = (
    f"PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True "
    f"SEQ={args.seq_len} "
    f"TP={result.tp_size} "
    f"BS={result.batch_size} "
    f"MODEL_SIZE={args.model_size} "
    f"RES_PATH={res_path} "
    f"MTUNER=1 "
)

# Optional: use impls to modify the command or log
# Example: print("Selected implementations:", result.impls)

# Step 3: Build the full command
cmd = (
    f"{env_vars} "
    "srun --exclusive --partition=debug --nodes=1 --gres=gpu:8 "
    "bash examples/gpt3/train_gpt3_175b_distributed.sh"
)

# Step 4: Run the command
subprocess.run(cmd, shell=True, check=True)
