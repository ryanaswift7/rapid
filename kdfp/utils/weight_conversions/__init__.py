# Use explicit relative imports to pull in the specific functions from your files
from .gpt2 import convert_gpt2_to_hooked_weights, convert_hooked_to_gpt2_weights
from .gemma import convert_gemma_to_hooked_weights, convert_hooked_to_gemma_weights
from .llama import convert_llama_to_hooked_weights, convert_hooked_to_llama_weights
from .pythia import convert_neox_to_hooked_weights, convert_hooked_to_neox_weights
from .qwen2 import convert_qwen2_to_hooked_weights, convert_hooked_to_qwen2_weights
from .qwen3 import convert_qwen3_to_hooked_weights, convert_hooked_to_qwen3_weights

from typing import Any, Dict, List, Tuple
import torch
from transformer_lens import HookedTransformer
from transformer_lens.HookedTransformerConfig import HookedTransformerConfig

# Define your dispatchers right here
def convert_hf_to_hooked_weights(hf_model: Any, cfg: HookedTransformerConfig) -> Dict[str, torch.Tensor]:
    arch = getattr(cfg, "original_architecture", getattr(cfg, "architecture", ""))

    if arch == "GPTNeoXForCausalLM":
        return convert_neox_to_hooked_weights(hf_model, cfg)
    elif arch == "GPT2LMHeadModel":
        return convert_gpt2_to_hooked_weights(hf_model, cfg)
    elif arch == "LlamaForCausalLM" or arch == "MistralForCausalLM":
        return convert_llama_to_hooked_weights(hf_model, cfg)
    elif "Gemma" in arch:
        return convert_gemma_to_hooked_weights(hf_model, cfg)
    elif "Qwen3" in arch:
        return convert_qwen3_to_hooked_weights(hf_model, cfg)
    elif "Qwen2" in arch:
        if getattr(cfg, "use_qk_norm", False):
            return convert_qwen3_to_hooked_weights(hf_model, cfg)
        return convert_qwen2_to_hooked_weights(hf_model, cfg)
    else:
        raise ValueError(f"Unsupported architecture for HF -> Hooked conversion: '{arch}'")


def convert_hooked_to_hf_weights(
    tl_model: HookedTransformer, 
    cfg: HookedTransformerConfig, 
    verbose: bool = False
) -> Tuple[Dict[str, torch.Tensor], List[str]]:
    arch = getattr(cfg, "original_architecture", getattr(cfg, "architecture", ""))
    
    if arch == "GPTNeoXForCausalLM":
        return convert_hooked_to_neox_weights(tl_model, cfg, verbose)
    elif arch == "GPT2LMHeadModel":
        return convert_hooked_to_gpt2_weights(tl_model, cfg, verbose)
    elif arch == "LlamaForCausalLM" or arch == "MistralForCausalLM":
        return convert_hooked_to_llama_weights(tl_model, cfg, verbose)
    elif "Gemma" in arch:
        return convert_hooked_to_gemma_weights(tl_model, cfg, verbose)
    elif "Qwen3" in arch:
        return convert_hooked_to_qwen3_weights(tl_model, cfg, verbose)
    elif "Qwen2" in arch:
        if getattr(cfg, "use_qk_norm", False):
            return convert_hooked_to_qwen3_weights(tl_model, cfg, verbose)
        return convert_hooked_to_qwen2_weights(tl_model, cfg, verbose)
    else:
        raise ValueError(f"Unsupported architecture for Hooked -> HF conversion: '{arch}'")

# Optional but recommended: explicitly declare what this package exports
__all__ = [
    "convert_hf_to_hooked_weights",
    "convert_hooked_to_hf_weights"
]