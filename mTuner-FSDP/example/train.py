import torch
import hydra
from omegaconf import DictConfig
import logging
import numpy as np
import time
import torch.optim as optim


def run_train(num_iter, batch_size, seq_len, model, loss_func, optimizer):
    logging.info(torch.cuda.memory_allocated())
    for i in range(num_iter):
        base_mem = torch.cuda.memory_allocated()
        logging.info(f"1, {base_mem}")
        torch.cuda.synchronize()
        t0 = time.time()
        input_ids = torch.randint(0, 1024, (batch_size, seq_len), device="cuda")
        output = model(
            input_ids,
            labels=input_ids,
            use_cache=False,  # reduce
        )
        loss = loss_func(
            output.logits.view(-1, output.logits.size(-1)),
            input_ids.view(-1),
        )
        del output
        after_mem = torch.cuda.memory_allocated()
        logging.info(f"2, {(after_mem - base_mem) / 1e9}")
        torch.cuda.synchronize()
        tb0 = time.time()
        tmid = time.time()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        del loss
        logging.info(f"3, {torch.cuda.memory_allocated()}")
        torch.cuda.synchronize()
        tb1 = time.time()
        t1 = time.time()
        logging.info(f"iter {i} fwd {tmid - t0} bwd {t1 - tmid} time {t1 - t0}")


def run_eval(num_iter, batch_size, seq_len, model):
    logging.info(torch.cuda.memory_allocated())
    for i in range(num_iter):
        with torch.no_grad():
            logging.info(torch.cuda.memory_allocated())
            torch.cuda.synchronize()
            t0 = time.time()
            input_ids = torch.randint(0, 1024, (batch_size, seq_len), device="cuda")
            output = model(
                input_ids,
                labels=input_ids,
                use_cache=False,  # reduce
            )
            logging.info(torch.cuda.memory_allocated())
            del output
            logging.info(torch.cuda.memory_allocated())
            t1 = time.time()
            logging.info(f"iter {i} time {t1 - t0}")


def show_module_structure(module, indent=0, verbose=True, parent_name=""):
    indent_str = " " * indent
    # next_indent_str = " " * (indent + 2)
    has_module = False
    for module_name, submodule in module.named_children():
        if verbose:
            grad_str = (
                submodule.requires_grad if hasattr(submodule, "requires_grad") else None
            )
            logging.info(f"{indent_str + parent_name + module_name} {grad_str}")
        show_module_structure(submodule, indent + 2, verbose, parent_name + module_name)
        has_module = True
    if not has_module:
        if verbose:
            for param_name, param in module.named_parameters():
                grad_str = (
                    param.requires_grad if hasattr(param, "requires_grad") else None
                )
                if grad_str != None:
                    if param.requires_grad:  # and param.shape == torch.Size([8, 768]):
                        pass
                        # param.requires_grad = False
                        # if (
                        #     "layers1s"
                        #     in parent_name
                        #     # and "attnq" in parent_name
                        #     # and "lora_B" in parent_name
                        # ):
                        #     logging.info("setting to true")
                        #     param.requires_grad = True
                    logging.info(
                        f"{indent_str + param_name} {param.requires_grad} {param.shape}"
                    )
                else:
                    logging.info(f"{indent_str + param_name} {grad_str}")


def run(args):
    if args.validation:
        np.random.seed(0)
        torch.manual_seed(0)
        torch.cuda.manual_seed(0)
        torch.backends.cudnn.deterministic = True

    num_iter = args.train.iter
    batch_size = args.train.batch_size
    seq_len = args.train.seq_len
    model_name = args.model
    assert model_name in [
        "facebook_opt_125m",
        "facebook_opt_6.7b",
        "facebook_opt_30b",
    ]

    from transformers import AutoModelForCausalLM
    from peft import LoraConfig, TaskType, get_peft_model
    import transformers

    if not args.validation:
        # Skip model initilization
        transformers.PreTrainedModel._initialize_weights = lambda x, *args, **kwargs: x
        torch.nn.init.normal_ = lambda x, *args, **kwargs: x
        torch.nn.init.uniform_ = lambda x, *args, **kwargs: x
        torch.nn.init.xavier_normal_ = lambda x, *args, **kwargs: x
        torch.nn.init.xavier_uniform_ = lambda x, *args, **kwargs: x
        torch.nn.init.kaiming_normal_ = lambda x, *args, **kwargs: x
        torch.nn.init.kaiming_uniform_ = lambda x, *args, **kwargs: x

    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float16
    ).to("cuda")
    # use flash attention
    model.to_bettertransformer()
    # convert to PEFT model
    if args.is_peft:
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=8,
            lora_alpha=32,
            lora_dropout=0.1,
        )
        model = get_peft_model(model, peft_config)

    # from transformers.utils.fx import symbolic_trace
    # model = symbolic_trace(model)
    # print(model)
    # fthub.draw_graph_normal_nodes(model.graph.nodes, "normal")
    # exit()
    # if args.is_peft:
    if args.show:
        show_module_structure(model, verbose=True)
    optimizer = optim.AdamW(model.parameters(), lr=1e-3)
    loss_func = torch.nn.CrossEntropyLoss()
    model.train()

    def hook(module, grad_input, grad_output):
        print(module)
        print("in:")
        for item in grad_input:
            if isinstance(item, torch.Tensor):
                print(item.shape, item.grad_fn)
            else:
                print(item)
        # logging.info(grad_input.shape)
        print("out:")
        for item in grad_output:
            if isinstance(item, torch.Tensor):
                print(item.shape, item.grad_fn)
            else:
                print(item)
        # logging.info(grad_output.shape)

    # torch.nn.modules.module.register_module_full_backward_hook(hook)
    run_train(2, batch_size, seq_len, model, loss_func, optimizer)
    # model.eval()
    # run_eval(2, batch_size, seq_len, model)


@hydra.main(
    version_base=None, config_path="./configs", config_name="single-train-config"
)
def main(config: DictConfig):
    filename = (
        hydra.core.hydra_config.HydraConfig.get().runtime.output_dir + "/train.log"
    )
    logging.info(config)
    run(config)
    logging.info(filename)


if __name__ == "__main__":
    main()
