import transformers
from transformers import OPTModel, OPTConfig, OPTForCausalLM
from transformers.models.opt.modeling_opt import OPTDecoder, OPTDecoderLayer
import torch
from torch import nn
import logging
from typing import Optional, Tuple


class FTOPTDecoderLayers(nn.Module):
    def __init__(self, config: OPTConfig, num: int, group_id, num_groups):
        super().__init__()
        self.layers = nn.ModuleList([OPTDecoderLayer(config) for _ in range(num)])
        self.group_id = group_id
        self.num_groups = num_groups

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        layer_head_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
    ) -> Tuple[
        torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]
    ]:
        for layer in self.layers:
            hidden_states = layer(
                hidden_states,
                attention_mask,
                layer_head_mask,
                past_key_value,
                output_attentions,
                use_cache,
            )[0]
        return (hidden_states,)


class FTOPTDecoder(OPTDecoder):
    def __init__(self, config: OPTConfig, groups=None):
        super().__init__(config)
        logging.info(config.num_hidden_layers)
        if groups is None:
            groups = [1 for i in range(48)]
        elif isinstance(groups, int):
            s = groups
            groups = [s for i in range(config.num_hidden_layers // s)]
        elif isinstance(groups, list):
            pass
        else:
            raise ValueError("groups must be None, int or list")
        logging.info(groups)
        assert sum(groups) == config.num_hidden_layers
        self.layers = nn.ModuleList(
            [
                FTOPTDecoderLayers(config, item, it, len(groups))
                for it, item in enumerate(groups)
            ]
        )
        self.num_layers = config.num_hidden_layers
        self.gradient_checkpointing = False
        self.post_init()


class FTOPTModel(OPTModel):
    def __init__(self, config: OPTConfig, groups=None):
        super().__init__(config)
        self.decoder = FTOPTDecoder(config, groups)
        self.post_init()


class FTOPTForCausalLM(OPTForCausalLM):
    def __init__(self, config, groups=None):
        super().__init__(config)
        self.model = FTOPTModel(config, groups)

        # the lm_head weight is automatically tied to the embed tokens weight
        self.lm_head = nn.Linear(
            config.word_embed_proj_dim, config.vocab_size, bias=False
        )

        # Initialize weights and apply final processing
        self.post_init()


if __name__ == "__main__":
    transformers.PreTrainedModel._initialize_weights = lambda x, *args, **kwargs: x
    OPTModel._initialize_weights = lambda x, *args, **kwargs: x
    torch.nn.init.normal_ = lambda x, *args, **kwargs: x
    torch.nn.init.uniform_ = lambda x, *args, **kwargs: x
    torch.nn.init.xavier_normal_ = lambda x, *args, **kwargs: x
    torch.nn.init.xavier_uniform_ = lambda x, *args, **kwargs: x
    torch.nn.init.kaiming_normal_ = lambda x, *args, **kwargs: x
    torch.nn.init.kaiming_uniform_ = lambda x, *args, **kwargs: x
    models_kwargs = {"groups": [2 for _ in range(24)]}
    model = FTOPTModel.from_pretrained(
        "facebook_opt_30b", torch_dtype=torch.float16, **models_kwargs
    )
