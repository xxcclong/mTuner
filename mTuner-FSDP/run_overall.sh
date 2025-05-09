for seq in 512 2048
do
python3 example/fsdp_train.py train.seq_len=$seq train.batch_size=max ac=None model=/mnt/data/zhongrx/Llama-2-7b-hf shard_group_size=1
python3 example/fsdp_train.py train.seq_len=$seq train.batch_size=max ac=checkpoint model=/mnt/data/zhongrx/Llama-2-7b-hf shard_group_size=1
python3 example/fsdp_train.py train.seq_len=$seq train.batch_size=max ac=None model=/mnt/data/zhongrx/Llama-2-13b-hf shard_group_size=1
python3 example/fsdp_train.py train.seq_len=$seq train.batch_size=max ac=checkpoint model=/mnt/data/zhongrx/Llama-2-13b-hf shard_group_size=1
# python3 example/fsdp_train.py train.seq_len=$seq train.batch_size=max ac=checkpoint model=/data/dataset/Llama-2-30b-hf-trans
done