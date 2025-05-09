from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import GenerationConfig
import torch
from peft import LoraConfig, TaskType, get_peft_model
import argparse
import os
import time


parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", type=str, default="")
parser.add_argument("--input", type=str, default="")
parser.add_argument("--tofile", action="store_true")
parser.add_argument(
    "--model", type=str, default="/data/dataset/llama/llama-2-7b-chat-hf/"
)
args = parser.parse_args()

if len(args.input) > 0 and os.path.isdir(args.input):
    print("inputs", os.listdir(args.input))
    input_files = [
        os.path.join(args.input, f)
        for f in os.listdir(args.input)
        if f.endswith(".txt")
    ]
    # sort by name
    input_files.sort(key=lambda x: int(x.split("/")[-1].split(".")[0]))
elif os.path.isfile(args.input):
    input_files = [args.input]
else:
    input_files = ["input.txt"]

if len(args.checkpoint) > 0 and os.path.isdir(args.checkpoint):
    # print("is directory")
    print("checkpoints", os.listdir(args.checkpoint))
    checkpoints = [
        os.path.join(args.checkpoint, f)
        for f in os.listdir(args.checkpoint)
        if f.endswith(".pt")
    ]
elif os.path.isfile(args.checkpoint):
    checkpoints = [args.checkpoint]
elif "," in args.checkpoint:
    checkpoints = args.checkpoint.split(",")
else:
    checkpoints = []

model_name_or_path = args.model
device = "cuda"  # or "cuda" if you have a GPU

model = AutoModelForCausalLM.from_pretrained(
    model_name_or_path, torch_dtype=torch.float16, trust_remote_code=True
).to(device)
tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
# model.config.pad_token_id = model.config.eos_token_id
# model.config.max_position_embeddings = 8192
tokenizer.pad_token = tokenizer.eos_token


def to_peft(model):
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=8,
        lora_alpha=32,
        lora_dropout=0.1,
    )
    model = get_peft_model(model, peft_config)
    return model


def infer(input_str, model, tokenizer, device):
    # print("input:", input_str)
    inputs = tokenizer(input_str, return_tensors="pt", padding=True, truncation=True)
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)
    # inputs = tokenizer.encode(input_str, return_tensors="pt").to(device)
    # gen_config = GenerationConfig(min_length=100, max_length=500)
    # gen_config = GenerationConfig(max_length=2048, eos_token_id=tokenizer.eos_token_id)
    gen_config = GenerationConfig(max_length=1024, eos_token_id=tokenizer.eos_token_id)
    # print(inputs)
    t0 = time.time()
    outputs = model.generate(
        inputs=input_ids, attention_mask=attention_mask, generation_config=gen_config
    )
    print("generate time", time.time() - t0)
    # print(outputs[0])
    out_str = tokenizer.decode(outputs[0])
    # print(out_str)
    return out_str


def load_params(model, dir):
    model.load_state_dict(torch.load(dir), strict=False)
    return model


# system_prompt = "<<SYS>>\nAlways answer with Chinese\n<</SYS>>\n"
# system_prompt = "Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n{instruction}\n\n### Response: Let's think step by step."

system_prompt = "根据下面的文档内容用中文回答问题，.\n\n### 文档内容及问题:\n{instruction}\n\n### 回答: 根据文档的内容，"

output_file = "output1.txt"

print("original model")
for input_file in input_files:
    input_str = open(input_file).read()
    if len(input_str) > 2500:
        continue
    format = {"instruction": input_str}

    output = infer(system_prompt.format_map(format), model, tokenizer, device)
    output = output + "\n"

    if args.tofile:
        with open(output_file, "a") as f:
            f.write("\noriginal model\n")
            f.write(output + '\n')
    else:
        print(output)

    model = to_peft(model)
    for checkpoint in checkpoints:
        print(f"load {checkpoint}")
        model = load_params(model, checkpoint)

        output = infer(system_prompt.format_map(format), model, tokenizer, device)
        output = output + "\n"

        if args.tofile:
            with open(output_file, "a") as f:
                f.write(f"\nload {checkpoint}\n")
                f.write(output + '\n')
        else:
            print(output)
