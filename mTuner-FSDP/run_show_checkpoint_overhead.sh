for bs in {1..10}
do
python3 example/fsdp_train.py train.batch_size=$bs ac=None model=/data/dataset/Llama-2-30b-hf-trans
done