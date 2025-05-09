#!/bin/bash

seqs=(1024 2048 4096 8192)
tps=(1)
bss=(1)
model_sizes=(7 13)
# model_sizes=(7 13 30 70)

for seq in "${seqs[@]}"; do
    for tp in "${tps[@]}"; do
        for bs in "${bss[@]}"; do
            for model_size in "${model_sizes[@]}"; do
                sbatch <<EOF
#!/bin/bash
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --gres=gpu:8
#SBATCH --job-name=gpt3_train_${seq}_${tp}_${bs}_${model_size}
#SBATCH --output=logs/gpt3_train_${seq}_${tp}_${bs}_${model_size}.log

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export SEQ=$seq
export TP=$tp
export BS=$bs
export MODEL_SIZE=$model_size

bash examples/gpt3/train_gpt3_175b_distributed.sh
EOF
            done
        done
    done
done

echo "All jobs submitted via sbatch."