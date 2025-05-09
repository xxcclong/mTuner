# for batch_size in 1 2 3 4 5 6 7 8 9 10
# do
#     nvidia-smi
#     python3 example/fsdp_train.py  train.batch_size=$batch_size ac=checkpoint model=/data/dataset/Llama-2-70b-hf-trans
#     nvidia-smi
# done
# for lid in 75 70 65 60 55 50
# do
# python3 example/fsdp_train.py train.batch_size=8 ac=checkpoint model=/mnt/data/zhongrx/Llama-2-70b-hf dual_model.num_layer=$lid dual_model.batch_size=1
# done

# for lid in 75 70 65 60 55 50
# do
# python3 example/fsdp_train.py train.batch_size=9 ac=checkpoint model=/mnt/data/zhongrx/Llama-2-70b-hf dual_model.num_layer=$lid dual_model.batch_size=1
# done

# for lid in 60 55 50
# do
# python3 example/fsdp_train.py train.batch_size=10 ac=checkpoint model=/mnt/data/zhongrx/Llama-2-70b-hf dual_model.num_layer=$lid dual_model.batch_size=1
# done

for lid in 45 40 35 30
do
python3 example/fsdp_train.py train.batch_size=8 ac=checkpoint model=/mnt/data/zhongrx/Llama-2-70b-hf dual_model.num_layer=$lid dual_model.batch_size=1
done

for lid in 45 40 35 30
do
python3 example/fsdp_train.py train.batch_size=7 ac=checkpoint model=/mnt/data/zhongrx/Llama-2-70b-hf dual_model.num_layer=$lid dual_model.batch_size=1
done

for lid in 45 40 35 30
do
python3 example/fsdp_train.py train.batch_size=6 ac=checkpoint model=/mnt/data/zhongrx/Llama-2-70b-hf dual_model.num_layer=$lid dual_model.batch_size=1
done

for lid in 45 40 35 30
do
python3 example/fsdp_train.py train.batch_size=5 ac=checkpoint model=/mnt/data/zhongrx/Llama-2-70b-hf dual_model.num_layer=$lid dual_model.batch_size=1
done