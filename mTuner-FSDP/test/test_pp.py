from copy import deepcopy
import torch
import tempfile
import torch.nn as nn
import time


class LlamaLayers(torch.nn.Module):
    def __init__(self, layers):
        super().__init__()
        self.layers = layers

    def forward(self, x):
        for idx, layer in enumerate(self.layers):
            x = layer(x)[0]
        return x

def split_llama(model, num_layers):
    model1 = torch.nn.Sequential(
        deepcopy(model.base_model.model.model.embed_tokens),
        LlamaLayers(deepcopy(model.base_model.model.model.layers[:num_layers])),
    )
    model2 = torch.nn.Sequential(
        LlamaLayers(deepcopy(model.base_model.model.model.layers[num_layers:])),
        deepcopy(model.base_model.model.model.norm),
        deepcopy(model.base_model.model.lm_head),
    )
    return model1, model2

def run_worker(rank, world_size):
    from torch.distributed import rpc
    tmpfile = tempfile.NamedTemporaryFile()
    rpc.init_rpc(
        name="worker",
        rank=0,
        world_size=1,
        rpc_backend_options=rpc.TensorPipeRpcBackendOptions(
            init_method="file://{}".format(tmpfile.name)
        )
    )
    # Number of GPUs for model parallelism.
    num_gpus = 2
    partition_len = 16

    from transformers import AutoModelForCausalLM, AutoConfig
    from peft import LoraConfig, TaskType, get_peft_model
    import transformers

    # Skip model initilization
    transformers.PreTrainedModel._initialize_weights = lambda x, *args, **kwargs: x
    torch.nn.init.normal_ = lambda x, *args, **kwargs: x
    torch.nn.init.uniform_ = lambda x, *args, **kwargs: x
    torch.nn.init.xavier_normal_ = lambda x, *args, **kwargs: x
    torch.nn.init.xavier_uniform_ = lambda x, *args, **kwargs: x
    torch.nn.init.kaiming_normal_ = lambda x, *args, **kwargs: x
    torch.nn.init.kaiming_uniform_ = lambda x, *args, **kwargs: x

    model = AutoModelForCausalLM.from_pretrained(
        "/mnt/data/zhongrx/Llama-2-7b-hf", torch_dtype=torch.float16
    )
    model.to_bettertransformer()
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=8,
        lora_alpha=32,
        lora_dropout=0.1,
    )
    model = get_peft_model(model, peft_config)
    model1, model2 = split_llama(model, 16)


    module_list =[model1.cuda(2 * rank), model2.cuda(2 * rank + 1)]

    from torch.distributed.pipeline.sync import Pipe
    # Build the pipeline.
    chunks = 1
    model3 = Pipe(torch.nn.Sequential(*module_list), chunks = chunks, checkpoint="never")


    def get_total_params(module: torch.nn.Module):
        total_params = 0
        for param in module.parameters():
            total_params += param.numel()
        return total_params

    print ('Total parameters in model: {:,}'.format(get_total_params(model3)))

    # Initialize process group and wrap model in DDP.
    from torch.nn.parallel import DistributedDataParallel
    import torch.distributed as dist
    import os
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12398'
    dist.init_process_group(
                backend="nccl", rank=rank, world_size=world_size)
    model3 = DistributedDataParallel(model3)

    def get_total_params(module: torch.nn.Module):
        total_params = 0
        for param in module.parameters():
            total_params += param.numel()
        return total_params
    
    def print_with_rank(msg):
        print('[RANK {}]: {}'.format(rank, msg))

    print_with_rank('Total parameters in model: {:,}'.format(get_total_params(model3)))
    
    criterion = nn.CrossEntropyLoss()
    lr = 5.0 # learning rate
    optimizer = torch.optim.SGD(model3.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.95)

    def train():
        model3.train() # Turn on the train mode
        total_loss = 0.
        start_time = time.time()
        num_iter = 3
        batch_size = 2
        seq_len = 1024
        vocab_size = 32000

        for _ in range(num_iter):
            optimizer.zero_grad()
            input_ids = torch.randint(0, 100, (batch_size, seq_len)).cuda(2 * rank)
            # Since the Pipe is only within a single host and process the ``RRef``
            # returned by forward method is local to this node and can simply
            # retrieved via ``RRef.local_value()``.
            output = model3(input_ids).local_value()
            # Need to move targets to the device where the output of the
            # pipeline resides.
            print("output", output.shape)
            loss = criterion(
            output.view(-1, vocab_size),
            input_ids.view(-1).cuda(2 * rank + 1),
            )
            del output
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model3.parameters(), 0.5)
            optimizer.step()
            del loss        

        end_time = time.time()
        print("Throughput: {} samples/s".format(num_iter * batch_size * seq_len / (end_time - start_time)))
        print("Latency: {} s".format((end_time - start_time) / num_iter))
    train()

if __name__ == "__main__":
    # 两个进程，每个进程运行一个2级pipeline
    # ddp + pipeline
    # num_process = num_devices / pipeline_depth, cuda_visible_devices = num_devices
    num_process = 2
    import torch.multiprocessing as mp
    mp.spawn(run_worker, args=(num_process,), nprocs=num_process, join=True)


    

    
    