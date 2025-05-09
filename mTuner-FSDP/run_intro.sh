for bs in {1..30}
do
python3 example/fsdp_train.py train.batch_size=$bs ac=checkpoint model=/data/dataset/Llama-2-30b-hf-trans
done

for bs in {1..30}
do
python3 example/fsdp_train.py train.batch_size=$bs ac=checkpoint model=/data/dataset/Llama-2-30b-hf-trans  shard_group_size=4
done

for bs in {1..30}
do
python3 example/fsdp_train.py train.batch_size=$bs ac=checkpoint model=/data/dataset/Llama-2-30b-hf-trans  shard_group_size=2
done

for bs in {1..10}
do
python3 example/fsdp_train.py train.batch_size=$bs ac=checkpoint model=/mnt/data/zhongrx/Llama-2-70b-hf
done

for bs in {1..10}
do
python3 example/fsdp_train.py train.batch_size=$bs ac=checkpoint model=/mnt/data/zhongrx/Llama-2-70b-hf  shard_group_size=4
done

