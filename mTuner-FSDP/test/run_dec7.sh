#!/usr/bin/env bash

set -x

# 定义 seq_len 和 batch 数组
seq_lens=(1024 512 256 128)
batches=(6 12 27 58)

# 使用 zip 结合两个数组
for i in "${!seq_lens[@]}"; do
    seq_len=${seq_lens[$i]}
    batch=${batches[$i]}
    for j in $(seq 1 1 $batch);
    do
        echo "[seq_len $seq_len, batch $batch, max_batch $j]"
        CUDA_VISIBLE_DEVICES=4,5,6,7 python test_tp.py \
            --seq_len $seq_len \
            --max_batch $j
    done
done
