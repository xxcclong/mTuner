from peft import LoraConfig, TaskType, get_peft_model
import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import GenerationConfig
import time, torch
import json


def to_chinese_peft(model):
    path_to_lora_config = "/data/huangkz/chinese_lora/adapter_config.json"
    peft_config = LoraConfig(json.load(open(path_to_lora_config)))
    model = get_peft_model(model, peft_config)
    print(
        model.base_model.model.lm_head.weight.shape,
        model.base_model.model.model.embed_tokens.weight.shape,
    )
    model.base_model.model.lm_head = torch.nn.Linear(
        4096,
        49953,
        bias=False,
        device=model.device,
    )
    model.base_model.model.model.embed_tokens = torch.nn.Embedding(
        49953,
        4096,
        device=model.device,
    )
    model = model.half()
    adapter_path = "/data/huangkz/chinese_lora/adapter_model.bin"
    loaded = torch.load(adapter_path)
    model.load_state_dict(loaded, strict=False)
    tokenizer = AutoTokenizer.from_pretrained(
        "/data/huangkz/chinese_lora", trust_remote_code=True
    )
    return model, tokenizer


def infer(input_str, model, tokenizer, device):
    inputs = tokenizer(input_str, return_tensors="pt", padding=True, truncation=True)
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)
    gen_config = GenerationConfig(max_length=1024, eos_token_id=tokenizer.eos_token_id)
    t0 = time.time()
    outputs = model.generate(
        inputs=input_ids, attention_mask=attention_mask, generation_config=gen_config
    )
    print("generate time", time.time() - t0)
    out_str = tokenizer.decode(outputs[0])
    return out_str


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", type=str, default="/data/dataset/llama/llama-2-7b-chat-hf/"
    )
    args = parser.parse_args()
    print(args)
    device = "cuda"  # or "cuda" if you have a GPU

    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.float16, trust_remote_code=True
    ).to(device)
    tokenizer = AutoTokenizer.from_pretrained(
        "/data/huangkz/chinese_lora", trust_remote_code=True
    )
    tokenizer.pad_token = tokenizer.eos_token
    model = to_chinese_peft(model)
