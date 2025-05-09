import torch
import logging
from peft import LoraConfig, TaskType, get_peft_model
from .fsdp.fully_sharded_data_parallel import (
    CPUOffload,
    BackwardPrefetch,
)
from .fsdp import FullyShardedDataParallel as FSDP
from .fsdp.wrap import size_based_auto_wrap_policy
from .fsdp.api import ShardingStrategy
from .fsdp import ReplicaConfig
from .ft_wrapper import ft_checkpoint_wrapper
from .model_split import split_llama_to_devices, split_llama
import torch.distributed as dist
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    checkpoint_wrapper,
    offload_wrapper,
    CheckpointImpl,
    apply_activation_checkpointing,
)
import functools

from transformers.models.llama.modeling_llama import LlamaMLP
from transformers.models.llama.modeling_llama import (
    LlamaFlashAttention2,
    apply_rotary_pos_emb,
)
from torch.distributed._tensor import DTensor, Shard, Replicate

from torch.distributed._tensor.device_mesh import DeviceMesh, _mesh_resources
import numpy as np


import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR


def to_bettertransformer(model):
    try:
        model.to_bettertransformer()
    except ValueError:
        pass
    return model


def to_peft(model, args):
    if args.is_peft:
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=64,
            lora_alpha=16,
            lora_dropout=0.05,
            # target_modules=[
            # "k_proj",
            # "q_proj",
            # "down_proj",
            # "up_proj",
            # "gate_proj",
            # "o_proj",
            # "v_proj",
            # ]
        )
        model = get_peft_model(model, peft_config)
    return model


def llama_forward2(self, x):
    # make x replicated on dim 0
    if hasattr(self, "tp"):
        current_mesh = _mesh_resources.root_mesh
        shard_spec = [Shard(0)]
        replicate = [Replicate()]
        x = DTensor.from_local(
            x, current_mesh, shard_spec, run_check=False
        ).redistribute(current_mesh, replicate)
    down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
    return down_proj


def llama_attn_forward2(
    self,
    hidden_states: torch.Tensor,
    attention_mask=None,
    position_ids=None,
    past_key_value=None,
    output_attentions: bool = False,
    use_cache: bool = False,
    **kwargs,
):
    # LlamaFlashAttention2 attention does not support output_attentions
    if "padding_mask" in kwargs:
        # overwrite attention_mask with padding_mask
        attention_mask = kwargs.pop("padding_mask")

    output_attentions = False

    bsz, q_len, _ = hidden_states.size()
    if hasattr(self, "tp"):
        num_device = self.num_device
        bsz = bsz * num_device

        # current_mesh = get_global_device_mesh()
        current_mesh = _mesh_resources.root_mesh
        shard_spec = [Shard(0)]
        replicate = [Replicate()]
        hidden_states = DTensor.from_local(
            hidden_states, current_mesh, shard_spec, run_check=False
        ).redistribute(current_mesh, replicate)
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
    query_states, key_states = apply_rotary_pos_emb(
        query_states, key_states, cos, sin, position_ids
    )

    if past_key_value is not None:
        cache_kwargs = {"sin": sin, "cos": cos}  # Specific to RoPE models
        key_states, value_states = past_key_value.update(
            key_states, value_states, self.layer_idx, cache_kwargs
        )

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

        query_states = query_states.to(target_dtype)
        key_states = key_states.to(target_dtype)
        value_states = value_states.to(target_dtype)

    attn_output = self._flash_attention_forward(
        query_states,
        key_states,
        value_states,
        attention_mask,
        q_len,
        dropout=dropout_rate,
    )

    attn_output = attn_output.reshape(bsz, q_len, -1).contiguous()
    # attn_output = attn_output.reshape(bsz, q_len, self.hidden_size).contiguous()
    # print(attn_output.shape, self.o_proj.weight.shape, type(attn_output), type(self.o_proj.weight), self.hidden_size)
    attn_output = self.o_proj(attn_output)

    if not output_attentions:
        attn_weights = None

    return attn_output, attn_weights, past_key_value


def to_TP_dual_model(model, args, rank, world_size, root_mesh):
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

    assert isinstance(model, torch.nn.Sequential)
    parallelize_plan = {}
    num_layer = len(model[0].layers)
    logging.info(f"num_layer: {num_layer}")
    # 4 gpu in a group
    assert world_size == 8
    # root_mesh = DeviceMesh("cuda", torch.arange(world_size))
    # set_global_device_mesh(root_mesh)
    _mesh_resources.root_mesh = root_mesh

    model_name = args.model.lower()
    # logging.info(f"qproj: {model[0].layers[0].self_attn.q_proj} {type(model[0].layers[0].self_attn.q_proj)}")
    if "llama" in model_name:
        merge_reshard = True
        if merge_reshard:
            LlamaMLP.forward = llama_forward2
            # LlamaFlashAttention2.forward = llama_attn_forward2

        for i in range(num_layer):
            if merge_reshard:
                parallelize_plan[f"0.layers.{i}.mlp.gate_proj"] = ColwiseParallel()
                parallelize_plan[f"0.layers.{i}.mlp.up_proj"] = ColwiseParallel()

                # parallelize_plan[
                #     f"0.layers.{i}.self_attn.q_proj"
                # ] = ColwiseParallel()
                # parallelize_plan[
                #     f"0.layers.{i}.self_attn.k_proj"
                # ] = ColwiseParallel()
                # parallelize_plan[
                #     f"0.layers.{i}.self_attn.v_proj"
                # ] = ColwiseParallel()
                model[0].layers[i].mlp.tp = True
                # model[0].layers[i].mlp.num_device = world_size
            else:
                parallelize_plan[f"0.layers.{i}.mlp.gate_proj"] = ColwiseParallel(
                    _prepare_input=make_input_reshard_replicate
                )
                parallelize_plan[f"0.layers.{i}.mlp.up_proj"] = ColwiseParallel(
                    _prepare_input=make_input_reshard_replicate
                )
            parallelize_plan[f"0.layers.{i}.mlp.down_proj"] = RowwiseParallel(
                _prepare_output=make_output_reshard_tensor
            )
            # parallelize_plan[
            #     f"0.layers.{i}.self_attn.o_proj"
            # ] = RowwiseParallel(_prepare_output=make_output_reshard_tensor)
    # logging.info(parallelize_plan)
    # if rank == 0:
    #     logging.info(model)
    parallelize_module(model, device_mesh=root_mesh, parallelize_plan=parallelize_plan)
    ignored_parameters = []
    for name, module in model.named_modules():
        if "gate_proj" in name:
            ignored_parameters.append(module)
        if "up_proj" in name:
            ignored_parameters.append(module)
        if "down_proj" in name:
            ignored_parameters.append(module)

    if args.is_peft:
        for param in model.parameters():
            if isinstance(param, torch.distributed._tensor.api.DTensor):
                param.requires_grad_(False)
    return model, ignored_parameters


def to_TP(model, args, rank, world_size):
    if not args.tp:
        return model, None
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

    model_name = args.model.lower()
    parallelize_plan = {}
    num_layer = len(model.base_model.model.model.layers)
    # 4 gpu in a group
    mesh1 = DeviceMesh("cuda", torch.arange(4))
    mesh2 = DeviceMesh("cuda", torch.arange(4, 8))
    root_mesh = mesh1 if rank < 4 else mesh2
    # root_mesh = DeviceMesh("cuda", torch.arange(world_size))
    _mesh_resources.root_mesh = root_mesh
    # set_global_device_mesh(root_mesh)
    if "llama" in model_name:
        merge_reshard = True
        if merge_reshard:
            LlamaMLP.forward = llama_forward2
        for i in range(num_layer):
            if merge_reshard:
                parallelize_plan[f"base_model.model.model.layers.{i}.mlp.gate_proj"] = (
                    ColwiseParallel()
                )
                parallelize_plan[f"base_model.model.model.layers.{i}.mlp.up_proj"] = (
                    ColwiseParallel()
                )
                model.base_model.model.model.layers[i].mlp.tp = True
            else:
                parallelize_plan[f"base_model.model.model.layers.{i}.mlp.gate_proj"] = (
                    ColwiseParallel(_prepare_input=make_input_reshard_replicate)
                )
                parallelize_plan[f"base_model.model.model.layers.{i}.mlp.up_proj"] = (
                    ColwiseParallel(_prepare_input=make_input_reshard_replicate)
                )
            parallelize_plan[f"base_model.model.model.layers.{i}.mlp.down_proj"] = (
                RowwiseParallel(_prepare_output=make_output_reshard_tensor)
            )
    parallelize_module(model, device_mesh=root_mesh, parallelize_plan=parallelize_plan)
    ignored_parameters = []
    for name, module in model.named_modules():
        if "gate_proj" in name:
            ignored_parameters.append(module)
        if "up_proj" in name:
            ignored_parameters.append(module)
        if "down_proj" in name:
            ignored_parameters.append(module)

    if args.is_peft:
        for param in model.parameters():
            if isinstance(param, torch.distributed._tensor.api.DTensor):
                param.requires_grad_(False)

    return model, ignored_parameters


def to_FSDP(model, args, rank, world_size, ignored_parameters=None):
    # convert to FSDP model
    cf = CPUOffload()
    cf.offload_params = args.offload_param

    my_auto_wrap_policy = functools.partial(
        size_based_auto_wrap_policy, min_num_params=args.min_param
    )
    replica_config = ReplicaConfig(args)
    if args.shard_group_size <= 0 or args.shard_group_size == world_size:
        logging.info("fully shard")
        model = FSDP(
            model,
            auto_wrap_policy=my_auto_wrap_policy,
            device_id=rank,
            sharding_strategy=ShardingStrategy.FULL_SHARD,
            backward_prefetch=BackwardPrefetch.BACKWARD_PRE,
            replica_config=replica_config,
            ignored_modules=ignored_parameters,
            cpu_offload=cf,
        )
    elif args.shard_group_size == 1:
        logging.info("fully replicate")
        model = FSDP(
            model,
            auto_wrap_policy=my_auto_wrap_policy,
            device_id=rank,
            sharding_strategy=ShardingStrategy.NO_SHARD,
            backward_prefetch=BackwardPrefetch.BACKWARD_PRE,
            replica_config=replica_config,
            ignored_modules=ignored_parameters,
            cpu_offload=cf,
        )
    else:
        logging.info("partially shard")
        group_all = torch.distributed.new_group(ranks=[i for i in range(world_size)])
        for gid in range(int(world_size // args.shard_group_size)):
            tmp = torch.distributed.new_group(
                ranks=[
                    i + gid * args.shard_group_size
                    for i in range(args.shard_group_size)
                ]
            )
            if gid == int(rank // args.shard_group_size):
                group_shard = tmp
        # logging.info((group_shard, group_all))
        model = FSDP(
            model,
            process_group=(group_shard, group_all),
            auto_wrap_policy=my_auto_wrap_policy,
            device_id=rank,
            sharding_strategy=ShardingStrategy.HYBRID_SHARD,
            backward_prefetch=BackwardPrefetch.BACKWARD_PRE,
            replica_config=replica_config,
            ignored_modules=ignored_parameters,
            cpu_offload=cf,
            # use_orig_params=True,
        )
    return model


def to_ac(model, args):
    if args.ac.lower() in ["offload", "checkpoint"]:
        model_name = args.model
        if "opt" in model_name.lower():
            from fthub.ft_modeling_opt import FTOPTDecoderLayers

            wrapper = ft_checkpoint_wrapper

            def check_fn(submodule):
                return (
                    isinstance(submodule, FTOPTDecoderLayers)
                    and submodule.group_id < submodule.num_groups - args.peak_num_groups
                )

        elif "llama" in model_name.lower():
            from transformers.models.llama.modeling_llama import LlamaDecoderLayer

            wrapper = checkpoint_wrapper

            def check_fn(submodule):
                return isinstance(submodule, LlamaDecoderLayer)

        else:
            assert False, f"{model_name} not support checkpoint/offload"

        if args.ac == "offload":
            non_reentrant_wrapper = functools.partial(
                offload_wrapper,
            )
        elif args.ac == "checkpoint":
            non_reentrant_wrapper = functools.partial(
                wrapper,
                checkpoint_impl=CheckpointImpl.NO_REENTRANT,
                checkpoint_config=args.checkpoint,
            )

        apply_activation_checkpointing(
            model, checkpoint_wrapper_fn=non_reentrant_wrapper, check_fn=check_fn
        )
    return model


def to_pipeline(model, args):
    world_size = args.world_size
    assert world_size == 1

    from torch.distributed.pipeline.sync import Pipe
    from torch.distributed import rpc
    import tempfile

    tmpfile = tempfile.NamedTemporaryFile()
    rpc.init_rpc(
        name="worker",
        rank=0,
        world_size=1,
        rpc_backend_options=rpc.TensorPipeRpcBackendOptions(
            init_method="file://{}".format(tmpfile.name)
        ),
    )

    models = split_llama_to_devices(model, world_size)
    # PP can also use AC
    models = [to_ac(m, args) for m in models]
    models = [models[i].cuda(i) for i in range(world_size)]
    pipe_model = Pipe(
        torch.nn.Sequential(*models), chunks=args.pipeline.chunks, checkpoint="never"
    )
    return pipe_model


def get_mesh_for_TP(args, num_device=4):
    part_id = args.rank // num_device
    dev_range = torch.arange(part_id * num_device, (part_id + 1) * num_device)
    return DeviceMesh("cuda", dev_range)


def change_model_with_config(model, args):
    run_dual_model = args.dual_model.num_layer > 0 and args.dual_model.batch_size > 0
    world_size = args.world_size
    rank = args.rank
    model = to_peft(model, args)
    model = to_bettertransformer(model)
    optimizer = None
    scheduler = None

    if args.pipeline.use_pipeline:
        return to_pipeline(model, args), optimizer, scheduler
    elif not run_dual_model:
        model, ignored_parameters = to_TP(model, args, rank, world_size)
        model = to_FSDP(model, args, rank, world_size, ignored_parameters)
        model = to_ac(model, args)
        return model, optimizer, scheduler
    else:
        # dual model
        assert args.dual_model.batch_size <= args.train.batch_size
        optimizer = optim.AdamW(model.parameters(), lr=args.lr)
        scheduler = CosineAnnealingLR(optimizer, T_max=1000)
        num_device_per_part = args.world_size // 2
        root_mesh = get_mesh_for_TP(args, num_device_per_part)
        # init model1 sequentially
        for i in range(world_size):
            if i == rank:
                model1, model2 = split_llama(model, args.dual_model.num_layer)
                del model
                model1 = to_FSDP(model1, args, rank, world_size)
                model1 = to_ac(model1, args)
            dist.barrier()
        # init model2
        model2, ignored_parameters = to_TP_dual_model(
            model2, args, rank, world_size, root_mesh
        )
        new_args = args
        # new_args.shard_group_size = 4
        new_args.repeated = True
        model2 = to_FSDP(model2, new_args, rank, world_size, ignored_parameters)
        if args.dual_model.peak_ac:
            model2 = to_ac(model2, args)
        return [model1, model2], optimizer, scheduler


def show_module_structure(module, indent=0, handles=[], modules=[], verbose=False):
    indent_str = " " * indent
    # next_indent_str = " " * (indent + 2)
    has_module = False
    for module_name, submodule in module.named_children():
        if verbose:
            grad_str = (
                submodule.requires_grad if hasattr(submodule, "requires_grad") else None
            )
            logging.info(f"{indent_str + module_name} {grad_str}")
        if isinstance(submodule, FSDP):
            handles.extend(submodule._handles)
            modules.append(submodule)
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
                logging.info(f"{indent_str + param_name} {grad_str}")
    return handles, modules


def set_validation(validation):
    if validation:
        np.random.seed(0)
        torch.manual_seed(0)
        torch.cuda.manual_seed(0)
        torch.backends.cudnn.deterministic = True
    if not validation:
        import transformers

        # Skip model initilization
        transformers.PreTrainedModel._initialize_weights = lambda x, *args, **kwargs: x
        torch.nn.init.normal_ = lambda x, *args, **kwargs: x
        torch.nn.init.uniform_ = lambda x, *args, **kwargs: x
        torch.nn.init.xavier_normal_ = lambda x, *args, **kwargs: x
        torch.nn.init.xavier_uniform_ = lambda x, *args, **kwargs: x
        torch.nn.init.kaiming_normal_ = lambda x, *args, **kwargs: x
        torch.nn.init.kaiming_uniform_ = lambda x, *args, **kwargs: x
