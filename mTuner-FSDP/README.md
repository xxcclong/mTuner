# FineTuneHub
Efficient finetune via partial replication

## How FineTuneHub works?
Parameter Efficient Fine-tuning (PEFT) is a technique that can reduce the number of parameters to be updated during fine-tuning. It has lower memory usage and frozen parameters.
FineTubeHub utilizes these PEFT characteristics to replicate more frozen parameter to reduce communication cost.

## How to run

### Install

```bash
pip install -r requirements.txt
python setup.py build develop
```

### Prepare models 

Use huggingface models

### Run

We use [hydra](https://github.com/facebookresearch/hydra) to manage input arguments

```bash
# other default values of arguments are in example/configs/config.yaml
CUDA_VISIBLE_DEVICES=4,5,6,7 python3 example/fsdp_train.py model=/data/dataset/llama/llama-2-7b-chat-hf/ train.batch_size=1 intra_weight.replica_rate=0.5
```

### Finetune

```bash
# this example puts data in data.json, you can replace it with your own data
CUDA_VISIBLE_DEVICES=0,1,2,3 python example/fsdp_train.py model=/data/dataset/llama/llama-2-7b-chat-hf/ train.batch_size=1 with_data=True validation=True lr=1e-5
```

### Inference with tuned model
```bash
# using original model and input from input.txt
python example/inference.py 

# using outputs/2024-02-27/00-56-00/105_1.7923864126205444.pt as the peft checkpoint
python example/inference.py --checkpoint outputs/2024-02-27/00-56-00/105_1.7923864126205444.pt 

# iterate through all .pt under checkpoint dir and all .txt under input dir
python example/inference.py --checkpoint outputs/2024-02-27/00-56-00/ --input dataset/ps/ 
```

### Evaluation

```bash
# this example evaluates fine-tuned models, you can modify the commands with your own settings
cd ./evaluation
conda activate vllm2
python merge.py --checkpoint /data/siqizhu/ftcode/FineTuneHub/outputs/2024-03-05/17-55-03/ --model /mnt/data/zhongrx/Llama-2-7b-hf/
python merge_lora_with_llama.py \
    --base_model /data/dataset/llama/llama-2-7b-chat-hf/ \
    --lora_model /data/siqizhu/merged_model_llama2_mar05/ \
    --output_type huggingface \
    --output_dir /data/siqizhu/llama-2-7b-chat-lora-math-mar05/
bash run_eval.sh
```
