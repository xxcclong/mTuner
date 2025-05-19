#!/bin/bash

seq_lens=(1024 2048 4096 8192)
models=("7" "13" "30" "70")

for seq in "${seq_lens[@]}"; do
  for model in "${models[@]}"; do
    echo "Running training for seq_len=$seq, model=$model"
    python3 example/fsdp_train.py \
      train.seq_len=$seq \
      train.batch_size=max \
      ac=checkpoint \
      model="/data/dataset/Llama-2-${model}b-hf" | tee ds_output/result_modelsize${model}_seq${seq}.txt
  done
done
