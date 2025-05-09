filename=megatron_01-12-2025_04-56-58.txt
for bs in 1 2 4 8 16
do
for ms in 7 13 30 70
do
for seq in 1024 2048 4096 8192
do
	SEQ=$seq MODEL_SIZE=$ms BS=$bs srun --exclusive --partition=debug --nodes=1 --gres=gpu:8 bash examples/gpt3/train_gpt3_175b_distributed.sh 2>&1 | tee -a $filename
done
done

done