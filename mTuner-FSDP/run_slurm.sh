srun --gres=gpu:4 -p octave -w octave -n 4 python example/fsdp_train.py model=/data/dataset/llama/llama-2-7b-chat-hf/ train.batch_size=1 with_data=True validation=True lr=1e-5
