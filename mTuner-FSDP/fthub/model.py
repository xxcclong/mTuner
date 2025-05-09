import torch
from .ft_modeling_opt import FTOPTForCausalLM
from .change_model import change_model_with_config, show_module_structure
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
import os


def get_model_and_tokenizer(args):
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    model_name = args.model
    tokenizer = None
    if "opt" in model_name:
        model = FTOPTForCausalLM.from_pretrained(
            model_name, torch_dtype=torch.float16, groups=args.group_size
        )
    elif "llama" in model_name:
        from transformers import AutoModelForCausalLM

        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            attn_implementation="flash_attention_2",
        )
        if args.with_data:
            from transformers import AutoTokenizer

            tokenizer = AutoTokenizer.from_pretrained(model_name)
    else:
        assert False, f"{model_name} not supported"
    return model, tokenizer


def init_model(model, args):
    # model initialization
    model.train()
    run_dual_model = args.dual_model.num_layer > 0 and args.dual_model.batch_size > 0
    model, optimizer, scheduler = change_model_with_config(model, args)
    if optimizer is None:
        optimizer = optim.AdamW(model.parameters(), lr=args.lr, fused=True)
    if scheduler is None:
        scheduler = CosineAnnealingLR(optimizer, T_max=1000)
    if run_dual_model:
        model1, model2 = model[0], model[1]
        handles1, modules1 = show_module_structure(model1, verbose=False)
        handles2, modules2 = show_module_structure(model2, verbose=False)
        handles = []
        handles.extend(handles1)
    else:
        handles, modules = show_module_structure(model, verbose=False)
    if args.debug.skip_comm:
        for handle in handles:
            handle.skip_communication()
    return model, optimizer, scheduler, handles
