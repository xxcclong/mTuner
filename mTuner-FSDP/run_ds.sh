#!/bin/bash

seq_lens=(1024 2048 4096 8192)
models=("7" "13" "30" "70")

mkdir -p ../ds_output
mkdir -p memory_traces

for seq in "${seq_lens[@]}"; do
  for model in "${models[@]}"; do
    echo "Running training for seq_len=$seq, model=$model"
    echo "log saving to ../fsdp_output/result_modelsize${model}_seq${seq}.txt"
    srun --exclusive --partition=debug --nodes=1 --gres=gpu:8 -n 8 python3 example/fsdp_train.py \
      train.seq_len=$seq \
      train.batch_size=max \
      ac=checkpoint profile.memory=False debug.skip_comm=False shard_group_size=8 \
      model="/public/home/hkz/repo/hkz/data/llama/${model}b" | tee ../ds_output/result_modelsize${model}_seq${seq}.txt
  done
done
