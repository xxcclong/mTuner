#!/bin/bash

seqs=(1024 2048 4096 8192)
mkdir -p mtuner_output
model_sizes=(7 13 30 70)

for seq in "${seqs[@]}"; do
    for model_size in "${model_sizes[@]}"; do
        python run_use_results.py --model_size ${model_size} --seq_len ${seq} | tee mtuner_output/result_modelsize${model_size}_seq${seq}.txt
    done
done
