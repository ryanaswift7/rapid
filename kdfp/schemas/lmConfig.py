from typing import Optional, Dict, Any, List
from pydantic import model_validator
from transformer_lens import HookedTransformer
from transformers import AutoModelForCausalLM
import torch
from kdfp.utils.shallow_model_loading import get_indices_and_num_layers_to_keep
import kdfp.utils.weight_conversions.pythia as wc
from kdfp.schemas.configBase import ConfigBase
from lightning.pytorch import LightningModule


class LMConfig(ConfigBase):
    model_name: str
    d_model: int
    n_layers: int
    layer_reduction_factor: Optional[float] = None
    hook_list: Optional[List[str]] = None
    kd_layer_filter_factor: Optional[float] = None

    @model_validator(mode='after')
    def filter_hook_list(self) -> 'LMConfig':
        assert not (self.layer_reduction_factor and self.kd_layer_filter_factor), (
            "You cannot select both filtering and reduction")
        if self.kd_layer_filter_factor:
            # model = HookedTransformer.from_pretrained_no_processing(
            #     self.model_name,
            #     dtype=torch.bfloat16)
            # n_layers = model.cfg.n_layers
            # del model

            kept_layer_indices, _ = get_indices_and_num_layers_to_keep(
                total_layers=self.n_layers,
                layer_reduction_factor=self.kd_layer_filter_factor,
                selection_method='linspace',)
            
            # hook names are formatter like:
            # "blocks.0.attn.hook_attn_scores"
            get_layer_idx = lambda x: int(x.split('.')[1])
            filter_constraint = lambda x: "blocks" not in x or get_layer_idx(x) in kept_layer_indices

            self.hook_list = [hook for hook in self.hook_list if filter_constraint(hook)]
        return self



