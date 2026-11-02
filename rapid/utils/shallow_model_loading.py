from transformer_lens import HookedTransformer
import torch
from typing import Literal, Tuple, List
from transformers import AutoModelForCausalLM, AutoConfig, AutoTokenizer
import copy

def get_indices_and_num_layers_to_keep(total_layers: int,
                                   layer_reduction_factor: float,
                                   selection_method: Literal['linspace', 'first_n'] = 'linspace') -> Tuple[List[int], int]:
    # ensure at least 1 and maximum all are kept
    n_layers_to_keep = max(1, round(total_layers * layer_reduction_factor))
    n_layers_to_keep = min(n_layers_to_keep, total_layers)
    # Select layers evenly spaced
    if selection_method == 'linspace':
        keep_indices = torch.linspace(0, total_layers-1, n_layers_to_keep).round().int().tolist()
    # take the first N layers
    elif selection_method == 'first_n':
        keep_indices = list(range(n_layers_to_keep))
    else:
        raise ValueError(f"Invalid selection method: {selection_method}. Must be one of: 'linspace' or 'first_n'")
    return keep_indices, n_layers_to_keep

def load_shallow(model_name: str,
                 mode: Literal["tl", "hf"], 
                 layer_reduction_factor: float,
                 selection_method: Literal['linspace', 'first_n'],
                 torch_dtype: torch.dtype,
                 device: str = "cpu"):
    if mode == "tl":
        return _load_shallow_tl(model_name=model_name,
                                 layer_reduction_factor=layer_reduction_factor,
                                 selection_method=selection_method,
                                 torch_dtype=torch_dtype,
                                 device=device)
    elif mode == "hf":
        return _load_shallow_hf(model_name=model_name,
                                layer_reduction_factor=layer_reduction_factor,
                                selection_method=selection_method,
                                torch_dtype=torch_dtype)
    else:
        raise ValueError("I don't know how you did it, but you provided an invalid mode. Must be either 'tl' or 'hf'.")
    

def _load_shallow_tl(model_name: str,
                     layer_reduction_factor: float,
                     selection_method: Literal['linspace', 'first_n'],
                     torch_dtype: torch.dtype,
                     device: str = "cpu"):
    # Load original model
    original_model = HookedTransformer.from_pretrained_no_processing(
        model_name, dtype=torch_dtype, device=device
    )

    tokenizer = original_model.tokenizer
    
    # Calculate which layers to keep
    total_layers = original_model.cfg.n_layers

    keep_indices, layers_to_keep = get_indices_and_num_layers_to_keep(total_layers=total_layers,
                                                                  layer_reduction_factor=layer_reduction_factor,
                                                                  selection_method=selection_method)
    
    # Create new config with fewer layers
    new_cfg = copy.deepcopy(original_model.cfg)
    new_cfg.n_layers = layers_to_keep
    
    # Create new model with modified config
    shallow_model = HookedTransformer(new_cfg)
    
    # Copy weights for selected layers
    for i, layer_idx in enumerate(keep_indices):
        # Copy block parameters
        shallow_model.blocks[i].load_state_dict(
            original_model.blocks[layer_idx].state_dict()
        )
    
    # Copy weights outside of blocks
    shallow_model.embed.load_state_dict(original_model.embed.state_dict())
    shallow_model.unembed.load_state_dict(original_model.unembed.state_dict())
    shallow_model.ln_final.load_state_dict(original_model.ln_final.state_dict())
    # shallow_model.pos_embed.load_state_dict(original_model.pos_embed.state_dict())
    
    return shallow_model, tokenizer, keep_indices



# Registry of supported causal LM model families
MODEL_FAMILY_CONFIGS = {
    'gpt2': {
        'layer_count_attr': 'n_layer',
        'layer_path': 'transformer.h',
        'copy_components': ['transformer.wte', 'transformer.wpe', 'transformer.ln_f', 'lm_head'],
    },
    'pythia': {
        'layer_count_attr': 'num_hidden_layers',
        'layer_path': 'gpt_neox.layers', 
        'copy_components': ['gpt_neox.embed_in', 'gpt_neox.final_layer_norm', 'embed_out'],
    }
}

# Model name patterns to family mapping
MODEL_NAME_TO_FAMILY = {
    'gpt2': 'gpt2',
    'gpt2-medium': 'gpt2',
    'gpt2-large': 'gpt2',
    'gpt2-xl': 'gpt2',
    'EleutherAI/pythia-70m': 'pythia',
    'EleutherAI/pythia-160m': 'pythia',
    'EleutherAI/pythia-410m': 'pythia',
    'EleutherAI/pythia-1b': 'pythia',
    'EleutherAI/pythia-1.4b': 'pythia',
    'EleutherAI/pythia-2.8b': 'pythia',
    'EleutherAI/pythia-6.9b': 'pythia',
    'EleutherAI/pythia-12b': 'pythia',
    'EleutherAI/pythia-70m-deduped': 'pythia',
    'EleutherAI/pythia-160m-deduped': 'pythia',
    'EleutherAI/pythia-410m-deduped': 'pythia',
    'EleutherAI/pythia-1b-deduped': 'pythia',
    'EleutherAI/pythia-1.4b-deduped': 'pythia',
    'EleutherAI/pythia-2.8b-deduped': 'pythia',
    'EleutherAI/pythia-6.9b-deduped': 'pythia',
    'EleutherAI/pythia-12b-deduped': 'pythia',
}

def detect_model_family(model_name: str) -> str:
    """Detect model family or raise error if unsupported."""
    if model_name in MODEL_NAME_TO_FAMILY:
        return MODEL_NAME_TO_FAMILY[model_name]
    
    # Pattern matching
    model_lower = model_name.lower()
    if 'pythia' in model_lower:
        return 'pythia'
    elif 'gpt2' in model_lower:
        return 'gpt2'
    
    # Error message for unsupported models
    supported_models = list(MODEL_NAME_TO_FAMILY.keys())
    raise ValueError(
        f"Model '{model_name}' not supported. "
        f"Supported models: {', '.join(supported_models[:5])}... "
        f"(total: {len(supported_models)})"
    )

def _load_shallow_hf(model_name: str,
                     layer_reduction_factor: float,
                     selection_method: Literal['linspace', 'first_n'],
                     torch_dtype: torch.dtype):
    """
    Create shallow causal LM by keeping evenly spaced layers.
    
    Args:
        model_name: HuggingFace model name
        layer_reduction_factor: Factor to reduce layers by
    
    Returns:
        (shallow_model, tokenizer): Tuple of model and tokenizer
    """
    # Get model family config
    family = detect_model_family(model_name)
    config = MODEL_FAMILY_CONFIGS[family]
    
    # Load original model
    original_model = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch_dtype)
    original_config = AutoConfig.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Calculate layers to keep
    total_layers = getattr(original_config, config['layer_count_attr'])

    keep_indices, layers_to_keep = get_indices_and_num_layers_to_keep(total_layers=total_layers,
                                                                  layer_reduction_factor=layer_reduction_factor,
                                                                  selection_method=selection_method)
    
    # Create shallow config and model
    new_config = copy.deepcopy(original_config)
    setattr(new_config, config['layer_count_attr'], layers_to_keep)
    shallow_model = AutoModelForCausalLM.from_config(new_config)
    
    # Copy selected layers
    def get_nested_attr(obj, path):
        """Get nested attribute like 'transformer.h' from object"""
        for attr in path.split('.'):
            obj = getattr(obj, attr)
        return obj
    
    original_layers = get_nested_attr(original_model, config['layer_path'])
    shallow_layers = get_nested_attr(shallow_model, config['layer_path'])
    
    for i, layer_idx in enumerate(keep_indices):
        shallow_layers[i].load_state_dict(original_layers[layer_idx].state_dict())
    
    # Copy other components (embeddings, head, etc.)
    original_state = original_model.state_dict()
    shallow_state = shallow_model.state_dict()
    
    for component_path in config['copy_components']:
        # Find all parameters that start with this component path
        component_keys = [k for k in original_state.keys() if k.startswith(component_path)]
        for key in component_keys:
            if key in shallow_state:
                shallow_state[key] = original_state[key].clone()
    
    shallow_model.load_state_dict(shallow_state)
    return shallow_model, tokenizer, keep_indices
