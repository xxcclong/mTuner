'''python merge.py --checkpoint /data/siqizhu/ftcode/FineTuneHub/outputs/2024-03-05/17-55-03/ --model /mnt/data/zhongrx/Llama-2-7b-hf/'''

from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import GenerationConfig
import torch
from peft import LoraConfig, TaskType, get_peft_model
import argparse
import os
import time


parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", type=str, default="")
parser.add_argument(
    "--model", type=str, default="/data/dataset/llama/llama-2-7b-chat-hf/"
)
args = parser.parse_args()

if len(args.checkpoint) > 0 and os.path.isdir(args.checkpoint):
    # print("is directory")
    print("checkpoints", os.listdir(args.checkpoint))
    checkpoints = [
        os.path.join(args.checkpoint, f)
        for f in os.listdir(args.checkpoint)
        if f.endswith(".pt")
    ]
elif "," in args.checkpoint:
    checkpoints = args.checkpoint.split(",")
else:
    checkpoints = []

model_name_or_path = args.model
device = "cuda"  # or "cuda" if you have a GPU

model = AutoModelForCausalLM.from_pretrained(
    model_name_or_path, torch_dtype=torch.float16
).to(device)
tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
# model.config.pad_token_id = model.config.eos_token_id
# model.config.max_position_embeddings = 8192
tokenizer.pad_token = tokenizer.eos_token


def to_peft(model):
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=64,
        lora_alpha=16,
        lora_dropout=0.05,
        # target_modules=[
        #     "k_proj",
        #     "q_proj",
        #     "down_proj",
        #     "up_proj",
        #     "gate_proj",
        #     "o_proj",
        #     "v_proj",
        # ]
    )
    model = get_peft_model(model, peft_config)
    return model


def load_params(model, dir):
    model.load_state_dict(torch.load(dir), strict=False)
    return model

print("original model")


model = to_peft(model)
for checkpoint in checkpoints:
    print(f"load {checkpoint}")
    model = load_params(model, checkpoint)
    model.save_pretrained(f"/data/siqizhu/merged_model_llama2_vanilla_mar05/{checkpoint.split('/')[-1]}_lora",from_pt=True)
