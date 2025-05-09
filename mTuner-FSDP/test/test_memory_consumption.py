import torch

from transformers import AutoModelForCausalLM
from peft import LoraConfig, TaskType, get_peft_model
from transformers.utils.fx import symbolic_trace
from torch.fx.passes.shape_prop import ShapeProp

name = "facebook_opt_125m"
batch_size = 128
seq_len = 128

model = AutoModelForCausalLM.from_pretrained(name)
gm = symbolic_trace(model)

input_ids = torch.randint(0, 1024, (batch_size, seq_len))

shape_prop = ShapeProp(gm)

shape_prop.propagate(input_ids)

overall_size = 0
dtypes = []

output_node = None

for node in gm.graph.nodes:
    # print(node.meta, node.name)
    if 'tensor_meta' in node.meta:
        if hasattr(node.meta['tensor_meta'], 'shape'):
            print(node.name, node.meta['tensor_meta'].shape.numel())
            if node.meta['tensor_meta'].dtype == torch.float32:
                overall_size += node.meta['tensor_meta'].shape.numel()
            dtypes.append(node.meta['tensor_meta'].dtype)
        else:
            print(node.name, node.meta['tensor_meta'])
            output_node = node
    else:
        print(node.name, "no meta")
output_size = 0

print(overall_size)
print(set(dtypes))
