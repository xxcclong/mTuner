# FineTuneHub (Disaggregated Finetune)
Centralized hub for fine-tuning machine learning models.

## How FineTuneHub works?

FineTuneHub is a centralized hub for fine-tuning machine learning models. 
Its core idea is to decouple the large model into base model and adapter models. The base model (e.g., LLM, difussion model) has frozen parameters and is shared across all tasks. The adapter models are task-/user-specific and get updated to finetune according to new user data and tasks. 
FineTuneHub server (FTServer) processes the request of **base model computation**, while FineTuneHub clients (FTClient) train the parameters of the adapter models.
After deploying the FTServer, users can easily use the FTClients finetune large models on their own datasets with various adapter models with cheap devices such as laptops.

The user interface of FineTuneHub is just like Huggingface's [transformers](https://huggingface.co/docs/transformers/index), as we use [torchFX](https://pytorch.org/docs/stable/fx.html) to **automatically** partition the user defined model into base model and adapter models.
The communication between FineTuneHub servers and clients are based on [gRPC](https://grpc.io/) for efficiency. 

## Installation

```bash
pip install -r requirements.txt
python setup.py build develop
```

## How To Use

### Partition the model

```python
# define the base model the same as server's, can also use other model defined in pytorch and transformers
base_model = AutoModelForCausalLM.from_pretrained(base_model_name)
# add the adapters or change the model structure as you want
model_with_adapter = modify(base_model) 
# extract the shared computation by comparing the two models, keep the local model on the client
local_model = compare_models(base_model, model_with_adapter)
# register the base model on the server
register(base_model, server_addr)
# Then you can train with the local model, any shared computation with base model will be processed by the server
train(local_model, server_addr)
```

### Server

```bash
python examples/server.py --model [MODEL_NAME] --port [PORT]
```

### Client

```bash
python examples/client.py --port [PORT] --addr [SERVER_ADDR] --model [MODEL_NAME] --dataset [DATASET_NAME]
```



## Supported Models

The implementation does not require any modification to the original model. The partitioning is based on whether a module requires gradient update.
Meanwhile, FineTuneHub also supports [PEFT](https://github.com/huggingface/peft).