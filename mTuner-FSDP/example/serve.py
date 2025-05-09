import argparse, json

from openai import OpenAI
from fastapi import FastAPI, Request
import uvicorn
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import GenerationConfig
import time, torch

app = FastAPI()


def to_peft(model):
    # peft_config = LoraConfig(
    #     task_type=TaskType.CAUSAL_LM,
    #     inference_mode=True,
    #     r=8,
    #     lora_alpha=32,
    #     lora_dropout=0.1,
    # )
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=64,
        lora_alpha=16,
        lora_dropout=0.05,
        target_modules=[
            "k_proj",
            "q_proj",
            "down_proj",
            "up_proj",
            "gate_proj",
            "o_proj",
            "v_proj",
        ],
    )
    model = get_peft_model(model, peft_config)
    return model


def infer(input_str, model, tokenizer, device):
    inputs = tokenizer(input_str, return_tensors="pt", padding=True, truncation=True)
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)
    gen_config = GenerationConfig(
        max_new_tokens=1024, eos_token_id=tokenizer.eos_token_id
    )
    t0 = time.time()
    outputs = model.generate(
        inputs=input_ids, attention_mask=attention_mask, generation_config=gen_config
    )
    print("generate time", time.time() - t0)
    out_str = tokenizer.decode(outputs[0])
    return out_str


def load_params(model, dir):
    model.load_state_dict(torch.load(dir), strict=False)


@app.post("/chat")
async def completion(request: Request):
    params = await request.json()
    prompt = params["input"]
    try:
        output = infer(prompt, model, tokenizer, device)
        print(output)
        return {"output": output}
    except Exception as e:
        return {"output": str(e)}


@app.post("/ckpt")
async def change_ckpt(request: Request):
    params = await request.json()
    ckpt_dir = params["ckpt"]
    try:
        load_params(model, ckpt_dir)
        print("load ckpt", ckpt_dir)
        return {"output": "Success!"}
    except Exception as e:
        return {"output": str(e)}


@app.post("/clear")
async def clear(request: Request):
    # clear peft weights of model
    try:
        for name, param in model.named_parameters():
            if "lora" in name.lower():
                param.data.zero_()
        return {"output": "Success!"}
    except Exception as e:
        return {"output": str(e)}


# @app.post("/clear")
# def get_embedding(request: Request):
#     params = await request.json()
#     text = params["input"]
#     text = text.replace("\n", " ")
#     model_name = "text-embedding-3-small"
#     emb = client.embeddings.create(input=[text], model=model).data[0].embedding
#     return emb


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=21002)
    parser.add_argument(
        "--model",
        type=str,
        default="/data/dataset/llama/Llama-2-13b-chat-hf/",
        # "--model", type=str, default="/data/dataset/llama/llama-2-7b-chat-hf/"
    )
    args = parser.parse_args()
    print(args)
    device = "cuda"  # or "cuda" if you have a GPU

    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.float16, trust_remote_code=True
    ).to(device)
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = to_peft(model)
    # model, tokenizer = to_chinese_peft(model)
    tokenizer.pad_token = tokenizer.eos_token
    query_cnt = 0
    client = OpenAI()
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
