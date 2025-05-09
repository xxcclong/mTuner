import torch
from copy import deepcopy


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

def split_llama_to_devices(model, num_device):
    models = []
    num_layer = len(model.base_model.model.model.layers)
    num_layer_per_device = num_layer // num_device
    # assert num_layer_per_device * num_device == num_layer
    model0 = torch.nn.Sequential(
        deepcopy(model.base_model.model.model.embed_tokens),
        LlamaLayers(deepcopy(model.base_model.model.model.layers[:num_layer_per_device])),
    )
    models.append(model0)
    for i in range(1, num_device - 1):
        model1 = torch.nn.Sequential(
            LlamaLayers(deepcopy(model.base_model.model.model.layers[i * num_layer_per_device : (i + 1) * num_layer_per_device])),
        )
        models.append(model1)
    model2 = torch.nn.Sequential(
        LlamaLayers(deepcopy(model.base_model.model.model.layers[(num_device - 1) * num_layer_per_device:])),
        deepcopy(model.base_model.model.model.norm),
        deepcopy(model.base_model.model.lm_head),
    ) 
    models.append(model2)
    return models



def split_llama_fsdp(model, num_layers):
    model1 = torch.nn.Sequential(
        model._fsdp_wrapped_module.base_model.model.model.embed_tokens,
        LlamaLayers(
            model._fsdp_wrapped_module.base_model.model.model.layers[:num_layers]
        ),
    )
    model2 = torch.nn.Sequential(
        LlamaLayers(
            model._fsdp_wrapped_module.base_model.model.model.layers[num_layers:]
        ),
        model._fsdp_wrapped_module.base_model.model.model.norm,
        model._fsdp_wrapped_module.base_model.model.lm_head,
    )
    return model1, model2
