mkdir -p flux_output

#!/bin/bash

# Define model sizes and sequence lengths
model_sizes=(7)
seq_lens=(8192)

# model_sizes=(7 13 30 70)
# seq_lens=(1024 2048 4096 8192)

# Define TP (tensor parallelism) and MTUNER flag
TP=8
MTUNER=0

# Loop over all combinations
for model_size in "${model_sizes[@]}"; do
  for seq in "${seq_lens[@]}"; do
    # Determine batch size according to rules
    if [ "$seq" -eq 8192 ]; then
      case "$model_size" in
        70) BS=2 ;;
        *)  BS=4 ;;
      esac
    elif [ "$seq" -eq 4096 ]; then
      case "$model_size" in
        70) BS=4 ;;
        *)  BS=8 ;;
      esac
    elif [ "$seq" -eq 2048 ]; then
      case "$model_size" in
        70) BS=8 ;;
        *)  BS=16 ;;
      esac
    elif [ "$seq" -eq 1024 ]; then
      case "$model_size" in
        70) BS=16 ;;
        *)  BS=32 ;;
      esac
    fi

    echo "Running model_size=$model_size, seq_len=$seq, batch_size=$BS"
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    SEQ=$seq TP=$TP BS=$BS MODEL_SIZE=$model_size MTUNER=$MTUNER \
    srun --exclusive --partition=debug --nodes=1 --gres=gpu:8 \
    bash examples/gpt3/train_gpt3_175b_distributed.sh \
    | tee flux_output/result_modelsize${model_size}_seq${seq}.txt
  done
done