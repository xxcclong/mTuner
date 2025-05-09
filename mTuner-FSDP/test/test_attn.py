import torch
from torch import nn
from typing import List, Optional, Tuple, Union
import copy
import os
from torch.distributed._tensor import DTensor, Shard, Replicate, distribute_tensor, distribute_module, init_device_mesh
from torch.distributed._tensor.device_mesh import DeviceMesh, mesh_resources
from torch.distributed.tensor.parallel import (
    PairwiseParallel,
    RowwiseParallel,
    ColwiseParallel,
    make_output_tensor,
    parallelize_module,
    make_sharded_output_tensor,
    make_input_reshard_replicate,
    make_output_shard_1d,
    make_output_reshard_tensor,
    SequenceParallel,
    make_input_shard_1d,
)
import torch.distributed._functional_collectives
import time
from transformers.models.llama.modeling_llama import LlamaFlashAttention2, apply_rotary_pos_emb

class LlamaConfig():
    def __init__(self):
        self.hidden_size = 8192
        self.intermediate_size = 28672 
        self.num_attention_heads = 64
        self.num_key_value_heads = 8
        self.max_position_embeddings = 2048
        self.rope_theta = 0.1
        self.attention_bias = False
        self.attention_dropout = 0
        self.rope_scaling = None


# class LlamaMLP(nn.Module):
#     def __init__(self, config):
#         super().__init__()
#         self.config = config
#         self.hidden_size = config.hidden_size
#         self.intermediate_size = config.intermediate_size
#         self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
#         self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
#         self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
#         self.act_fn = nn.GELU()

#     def forward(self, x):
#         down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))

#         return down_proj

# def llama_forward2(self, x):
#     # make x replicated on dim 0
#     current_mesh = mesh_resources.root_mesh
#     shard_spec = [Shard(0)]
#     replicate = [Replicate()]
#     x = DTensor.from_local(x, current_mesh, shard_spec, run_check=False).redistribute(current_mesh, replicate)
#     down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
#     return down_proj

def forward2(
    self,
    hidden_states: torch.Tensor,
    attention_mask = None,
    position_ids = None,
    past_key_value = None,
    output_attentions: bool = False,
    use_cache: bool = False,
    **kwargs,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
    # LlamaFlashAttention2 attention does not support output_attentions
    if "padding_mask" in kwargs:
        warnings.warn(
            "Passing `padding_mask` is deprecated and will be removed in v4.37. Please make sure use `attention_mask` instead.`"
        )

        # overwrite attention_mask with padding_mask
        attention_mask = kwargs.pop("padding_mask")

    output_attentions = False

    bsz, q_len, _ = hidden_states.size()
    num_device = self.num_device
    # print(self.num_device, bsz)
    bsz = bsz * num_device
    tmp_num_heads = self.num_heads // num_device
    tmp_num_key_value_heads = self.num_key_value_heads // num_device

    current_mesh = mesh_resources.root_mesh
    shard_spec = [Shard(0)]
    replicate = [Replicate()]
    hidden_states = DTensor.from_local(hidden_states, current_mesh, shard_spec, run_check=False).redistribute(current_mesh, replicate)
    # print(hidden_states.shape, type(hidden_states))
    query_states = self.q_proj(hidden_states)
    # print(query_states.shape, type(query_states))
    key_states = self.k_proj(hidden_states)
    value_states = self.v_proj(hidden_states)

    # Flash attention requires the input to have the shape
    # batch_size x seq_length x head_dim x hidden_dim
    # therefore we just need to keep the original shape
    query_states = query_states.view(bsz, q_len, -1, self.head_dim).transpose(1, 2)
    key_states = key_states.view(bsz, q_len, -1, self.head_dim).transpose(1, 2)
    value_states = value_states.view(bsz, q_len, -1, self.head_dim).transpose(1, 2)

    kv_seq_len = key_states.shape[-2]
    if past_key_value is not None:
        kv_seq_len += past_key_value.get_usable_length(kv_seq_len, self.layer_idx)
    cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)

    if past_key_value is not None:
        cache_kwargs = {"sin": sin, "cos": cos}  # Specific to RoPE models
        key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

    # TODO: These transpose are quite inefficient but Flash Attention requires the layout [batch_size, sequence_length, num_heads, head_dim]. We would need to refactor the KV cache
    # to be able to avoid many of these transpose/reshape/view.
    query_states = query_states.transpose(1, 2)
    key_states = key_states.transpose(1, 2)
    value_states = value_states.transpose(1, 2)
    # print(query_states.shape, type(query_states))

    dropout_rate = self.attention_dropout if self.training else 0.0

    # In PEFT, usually we cast the layer norms in float32 for training stability reasons
    # therefore the input hidden states gets silently casted in float32. Hence, we need
    # cast them back in the correct dtype just to be sure everything works as expected.
    # This might slowdown training & inference so it is recommended to not cast the LayerNorms
    # in fp32. (LlamaRMSNorm handles it correctly)

    input_dtype = query_states.dtype
    if input_dtype == torch.float32:
        # Handle the case where the model is quantized
        if hasattr(self.config, "_pre_quantization_dtype"):
            target_dtype = self.config._pre_quantization_dtype
        else:
            target_dtype = self.q_proj.weight.dtype

        logger.warning_once(
            f"The input hidden states seems to be silently casted in float32, this might be related to"
            f" the fact you have upcasted embedding or layer norm layers in float32. We will cast back the input in"
            f" {target_dtype}."
        )

        query_states = query_states.to(target_dtype)
        key_states = key_states.to(target_dtype)
        value_states = value_states.to(target_dtype)

    attn_output = self._flash_attention_forward(
        query_states, key_states, value_states, attention_mask, q_len, dropout=dropout_rate
    )

    attn_output = attn_output.reshape(bsz, q_len, -1).contiguous()
    # attn_output = attn_output.reshape(bsz, q_len, self.hidden_size).contiguous()
    # print(attn_output.shape, self.o_proj.weight.shape, type(attn_output), type(self.o_proj.weight), self.hidden_size)
    attn_output = self.o_proj(attn_output)

    if not output_attentions:
        attn_weights = None

    return attn_output, attn_weights, past_key_value

def run():
    num_trial = 10
    config = LlamaConfig()
    bs = 4
    seq_len = 1024
    rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    device_mesh = init_device_mesh("cuda", (world_size,))
    root_mesh = device_mesh 
    mesh_resources.root_mesh = root_mesh
    input_tensor = torch.randn((bs, seq_len, config.hidden_size), requires_grad=True, device=rank, dtype=torch.float16)
    torch.manual_seed(0)
    # model = LlamaMLP(config).to(torch.float16)
    model = LlamaFlashAttention2(config).to(torch.float16).to(rank)
    model_gpu = copy.deepcopy(model).to(rank)
    output_correct = model_gpu(input_tensor)[0]
    # exit()
    torch.cuda.synchronize(rank)
    t2 = time.time()
    for _ in range(num_trial):
        output = model_gpu(input_tensor)
    torch.cuda.synchronize(rank)
    t3 = time.time()
    skip_input = 2
    if skip_input == 1: # skip all
        input_tensor_replicated = make_input_reshard_replicate(input_tensor, device_mesh)
        print(input_tensor_replicated.shape, input_tensor_replicated._local_tensor.shape)

    # TP
    parallelize_plan = {}

    if skip_input > 0:
        parallelize_plan["q_proj"] = ColwiseParallel()
        parallelize_plan["k_proj"] = ColwiseParallel()
        parallelize_plan["v_proj"] = ColwiseParallel()
    else:
        parallelize_plan["q_proj"] = ColwiseParallel(_prepare_input=make_input_reshard_replicate)
        parallelize_plan["k_proj"] = ColwiseParallel(_prepare_input=make_input_reshard_replicate)
        parallelize_plan["v_proj"] = ColwiseParallel(_prepare_input=make_input_reshard_replicate)
    parallelize_plan["o_proj"] = RowwiseParallel(_prepare_output=make_output_reshard_tensor)
    parallelize_module(module=model, device_mesh=device_mesh, parallelize_plan=parallelize_plan)
    LlamaFlashAttention2.forward = forward2
    model.num_device = world_size
    # if skip_input == 2:
    #     LlamaMLP.forward = llama_forward2
    # print(model)
    if skip_input == 1:
        output = model(input_tensor_replicated)[0]
    else:
        output = model(input_tensor)[0]
    # print(output.shape, type(output))
    # print(output_correct.shape, type(output_correct))
    print(torch.allclose(output, output_correct, atol=1e-3))
    # if rank == 0:
    #     print(output)


    torch.cuda.synchronize(rank)
    t0 = time.time()
    for _ in range(num_trial):
        if skip_input == 1:
            output = model(input_tensor_replicated)
        else:
            output = model(input_tensor)
    torch.cuda.synchronize(rank)
    t1 = time.time()


    if rank == 0:
        print(f"TP: time: {(t1 - t0)/num_trial}")
        print(f"GPU: time: {(t3 - t2)/num_trial}")




run()