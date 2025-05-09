import transformers
from transformers import LlamaModel, LlamaForCausalLM, LlamaConfig
from transformers.models.llama.modeling_llama import LlamaDecoderLayer
import torch
from torch import nn
import logging
from typing import Optional, Tuple


class FTLlamaDecoderLayers(nn.Module):
    def __init__(self, config: LlamaConfig, num: int, group_id: int, num_groups: int):
        super().__init__()
        self.layers = nn.ModuleList([LlamaDecoderLayer(config) for _ in range(num)])
        self.group_id = group_id
        self.num_groups = num_groups

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        **kwargs,
    ) -> Tuple[
        torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]
    ]:
        for layer in self.layers:
            hidden_states = layer(
                hidden_states,
                attention_mask,
                position_ids,
                past_key_value,
                output_attentions,
                use_cache,
            )[0]
        return (hidden_states,)


class FTLlamaModel(LlamaModel):
    def __init__(self, config: LlamaConfig, groups=None):
        super().__init__(config)
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
                FTLlamaDecoderLayers(config, item, it, len(groups))
                for it, item in enumerate(groups)
            ]
        )
        self.num_layers = config.num_hidden_layers

        # Initialize weights and apply final processing
        self.post_init()
