import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, TaskType, get_peft_model
import transformers

transformers.PreTrainedModel._initialize_weights = lambda x, *args, **kwargs: x
torch.nn.init.normal_ = lambda x, *args, **kwargs: x
torch.nn.init.uniform_ = lambda x, *args, **kwargs: x
torch.nn.init.xavier_normal_ = lambda x, *args, **kwargs: x
torch.nn.init.xavier_uniform_ = lambda x, *args, **kwargs: x
torch.nn.init.kaiming_normal_ = lambda x, *args, **kwargs: x
torch.nn.init.kaiming_uniform_ = lambda x, *args, **kwargs: x


class LlamaLayers(torch.nn.Module):
    def __init__(self, layers):
        super().__init__()
        self.layers = layers

    def forward(self, x):
        for idx, layer in enumerate(self.layers):
            x = layer(x)[0]
        return x


def show_module_structure(module, indent=0, handles=[], modules=[], verbose=False):
    indent_str = " " * indent
    # next_indent_str = " " * (indent + 2)
    has_module = False
    for module_name, submodule in module.named_children():
        if verbose:
            grad_str = (
                submodule.requires_grad if hasattr(submodule, "requires_grad") else None
            )
            print(f"{indent_str + module_name} {grad_str}")
        handles, modules = show_module_structure(
            submodule, indent + 2, handles, modules, verbose
        )
        has_module = True
    if not has_module:
        if verbose:
            for param_name, param in module.named_parameters():
                # logging.info(indent_str + param_name, param.requires_grad)
                grad_str = (
                    param.requires_grad if hasattr(param, "requires_grad") else None
                )
                print(f"{indent_str + param_name} {grad_str}")
    return handles, modules


# model_name = "/mnt/data/zhongrx/Llama-2-13b-hf"
model_name = "/data/dataset/Llama-2-70b-hf-trans"
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float32)
model.to_bettertransformer()
print(model)
exit()
# print(model.model.layers[5:])
# print(model.children())
# for it, c in enumerate(model.children()):
#     print(it, c)

# show_module_structure(model, verbose=True)
model1 = torch.nn.Sequential(
    model.model.embed_tokens, LlamaLayers(model.model.layers[:5])
)
model2 = torch.nn.Sequential(
    LlamaLayers(model.model.layers[5:]), model.model.norm, model.lm_head
)


peft_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    inference_mode=False,
    r=8,
    lora_alpha=32,
    lora_dropout=0.1,
)
model = get_peft_model(model, peft_config)


# input_ids = torch.randint(0, 1024, (2, 32))
# output1 = model(input_ids)
# tmp = model1(input_ids)
# print(tmp.shape)
# output2 = model2(tmp)
# print(output2.shape)
# # model1(input_ids)
