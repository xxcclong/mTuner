for seq in 2048 4096 8192
do
# for model in /data/dataset/Llama-2-70b-hf-trans
# for model in /data/dataset/Llama-2-30b-hf-trans
# for model in /data/dataset/Llama-2-30b-hf-trans /data/dataset/Llama-2-70b-hf-trans
# for model in /mnt/data/zhongrx/Llama-2-7b-hf /mnt/data/zhongrx/Llama-2-13b-hf /data/dataset/Llama-2-30b-hf-trans /data/dataset/Llama-2-70b-hf-trans
for model in /mnt/data/zhongrx/Llama-2-7b-hf /mnt/data/zhongrx/Llama-2-13b-hf /data/dataset/Llama-2-30b-hf-trans
# for model in /mnt/data/zhongrx/Llama-2-7b-hf
do
python3 example/fsdp_train.py train.seq_len=$seq train.batch_size=max ac=None model=$model
# python3 example/fsdp_train.py train.seq_len=$seq train.batch_size=max ac=checkpoint model=/data/dataset/Llama-2-30b-hf-trans
done
done