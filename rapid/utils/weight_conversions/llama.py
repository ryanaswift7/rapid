from typing import cast, Dict, Any

import einops
import torch

from transformer_lens.HookedTransformerConfig import HookedTransformerConfig

from typing import Dict, List, Tuple
import einops
import torch
from transformer_lens import HookedTransformer
from transformer_lens.HookedTransformerConfig import HookedTransformerConfig

def convert_hooked_to_llama_weights(
    tl_model: HookedTransformer, 
    cfg: HookedTransformerConfig, 
    verbose: bool = False
) -> Tuple[Dict[str, torch.Tensor], List[str]]:
    """
    Converts weights from HookedTransformer format back to Hugging Face LLaMA format.
    This is the inverse of the convert_llama_weights function.
    
    Args:
        tl_model: The HookedTransformer model
        cfg: HookedTransformerConfig with the model's configuration
        verbose: Whether to print missing/unexpected parameters
        
    Returns:
        tuple: (llama_state_dict, missing_params)
            - llama_state_dict: Dictionary with weights in the HF LLaMA format
            - missing_params: List of expected parameters that weren't found in state_dict
    """
    tl_state_dict = tl_model.state_dict()
    llama_state_dict = {}
    missing_params = []
    
    # Handle Grouped Query Attention (GQA) formatting
    using_gqa = cfg.n_key_value_heads is not None
    gqa_uscore = "_" if using_gqa else ""

    # Define expected parameters in the HookedTransformer state dict
    expected_params = [
        "embed.W_E",
        "ln_final.w",
        "unembed.W_U",
        "unembed.b_U"  # TL generates this as zeros even though LLaMA doesn't use it
    ]
    
    # Add layer-specific parameters to expected_params
    for l in range(cfg.n_layers):
        expected_params.extend([
            f"blocks.{l}.ln1.w",
            f"blocks.{l}.attn.W_Q",
            f"blocks.{l}.attn.{gqa_uscore}W_K",
            f"blocks.{l}.attn.{gqa_uscore}W_V",
            f"blocks.{l}.attn.b_Q",
            f"blocks.{l}.attn.{gqa_uscore}b_K",
            f"blocks.{l}.attn.{gqa_uscore}b_V",
            f"blocks.{l}.attn.W_O",
            f"blocks.{l}.attn.b_O",
            f"blocks.{l}.ln2.w",
            f"blocks.{l}.mlp.W_in",
            f"blocks.{l}.mlp.W_gate",
            f"blocks.{l}.mlp.W_out",
            f"blocks.{l}.mlp.b_in",
            f"blocks.{l}.mlp.b_out"
        ])
    
    # Check for missing parameters
    for param in expected_params:
        if param not in tl_state_dict:
            missing_params.append(param)
    
    # Embedding layer
    if "embed.W_E" in tl_state_dict:
        llama_state_dict["model.embed_tokens.weight"] = tl_state_dict["embed.W_E"]
    
    # Process all transformer layers that might be in the state dict
    max_layer = -1
    for key in tl_state_dict.keys():
        if key.startswith("blocks."):
            parts = key.split(".")
            if len(parts) >= 2:
                try:
                    layer_num = int(parts[1])
                    max_layer = max(max_layer, layer_num)
                except ValueError:
                    continue
    
    for l in range(max_layer + 1):
        # Layer norms (LLaMA uses RMSNorm, so no biases)
        if f"blocks.{l}.ln1.w" in tl_state_dict:
            llama_state_dict[f"model.layers.{l}.input_layernorm.weight"] = tl_state_dict[f"blocks.{l}.ln1.w"]
        if f"blocks.{l}.ln2.w" in tl_state_dict:
            llama_state_dict[f"model.layers.{l}.post_attention_layernorm.weight"] = tl_state_dict[f"blocks.{l}.ln2.w"]
        
        # Attention
        q_key = f"blocks.{l}.attn.W_Q"
        k_key = f"blocks.{l}.attn.{gqa_uscore}W_K"
        v_key = f"blocks.{l}.attn.{gqa_uscore}W_V"
        o_key = f"blocks.{l}.attn.W_O"

        if q_key in tl_state_dict:
            W_Q = tl_state_dict[q_key]
            if not cfg.load_in_4bit:
                W_Q = einops.rearrange(W_Q, "n m h -> (n h) m")
            llama_state_dict[f"model.layers.{l}.self_attn.q_proj.weight"] = W_Q

        if k_key in tl_state_dict:
            W_K = tl_state_dict[k_key]
            if not cfg.load_in_4bit:
                W_K = einops.rearrange(W_K, "n m h -> (n h) m")
            llama_state_dict[f"model.layers.{l}.self_attn.k_proj.weight"] = W_K

        if v_key in tl_state_dict:
            W_V = tl_state_dict[v_key]
            if not cfg.load_in_4bit:
                W_V = einops.rearrange(W_V, "n m h -> (n h) m")
            llama_state_dict[f"model.layers.{l}.self_attn.v_proj.weight"] = W_V

        if o_key in tl_state_dict:
            W_O = tl_state_dict[o_key]
            if not cfg.load_in_4bit:
                W_O = einops.rearrange(W_O, "n h m -> m (n h)")
            llama_state_dict[f"model.layers.{l}.self_attn.o_proj.weight"] = W_O
            
        # Note: We intentionally skip b_Q, b_K, b_V, and b_O here because 
        # standard Hugging Face LLaMA architectures do not use attention biases.

        # MLP
        in_key = f"blocks.{l}.mlp.W_in"
        gate_key = f"blocks.{l}.mlp.W_gate"
        out_key = f"blocks.{l}.mlp.W_out"

        if in_key in tl_state_dict:
            llama_state_dict[f"model.layers.{l}.mlp.up_proj.weight"] = (
                tl_state_dict[in_key] if cfg.load_in_4bit else tl_state_dict[in_key].T
            )
            
        if gate_key in tl_state_dict:
            llama_state_dict[f"model.layers.{l}.mlp.gate_proj.weight"] = (
                tl_state_dict[gate_key] if cfg.load_in_4bit else tl_state_dict[gate_key].T
            )

        if out_key in tl_state_dict:
            llama_state_dict[f"model.layers.{l}.mlp.down_proj.weight"] = (
                tl_state_dict[out_key] if cfg.load_in_4bit else tl_state_dict[out_key].T
            )

        # Note: We intentionally skip b_in and b_out here because 
        # standard Hugging Face LLaMA architectures do not use MLP biases.
    
    # Final layer norm
    if "ln_final.w" in tl_state_dict:
        llama_state_dict["model.norm.weight"] = tl_state_dict["ln_final.w"]
    
    # Unembedding (output) layer
    if "unembed.W_U" in tl_state_dict:
        llama_state_dict["lm_head.weight"] = tl_state_dict["unembed.W_U"].T
    
    # Check for unexpected parameters
    unexpected_params = [param for param in tl_state_dict.keys() if param not in expected_params]

    if verbose:
        # Print unexpected parameters
        if unexpected_params:
            print("Unexpected parameters:")
            for param in unexpected_params:
                print(f"  - {param}")
                
        # Print missing parameters
        if missing_params:
            print("Missing parameters:")
            for param in missing_params:
                print(f"  - {param}")
    
    return llama_state_dict, missing_params


# taken from https://github.com/TransformerLensOrg/TransformerLens/blob/main/transformer_lens/pretrained/weight_conversions/llama.py
def convert_llama_to_hooked_weights(llama, cfg: HookedTransformerConfig):
    state_dict = {}

    state_dict["embed.W_E"] = llama.model.embed_tokens.weight

    # Some models with the Llama architecture use Grouped Query Attention, and so for these we need to modify
    # the state dict keys for the K/V attention weight/biases, prepending "_" to the key names.
    using_gqa = cfg.n_key_value_heads is not None
    gqa_uscore = "_" if using_gqa else ""
    # need a cast since MyPy isn't smart enough to realize that using_gqa implies n_key_value_heads is not None
    n_kv_heads = cast(int, cfg.n_key_value_heads if using_gqa else cfg.n_heads)

    # llama has no biases anywhere and deals with everything else roughly like
    # GPTNeoX with different names

    assert cfg.d_mlp is not None  # keep mypy happy

    for l in range(cfg.n_layers):
        state_dict[f"blocks.{l}.ln1.w"] = llama.model.layers[l].input_layernorm.weight

        W_Q = llama.model.layers[l].self_attn.q_proj.weight
        W_K = llama.model.layers[l].self_attn.k_proj.weight
        W_V = llama.model.layers[l].self_attn.v_proj.weight

        # in case of quantization,
        # parameters should stay as bitsandbytes.nn.modules.Params4bit
        if not cfg.load_in_4bit:
            W_Q = einops.rearrange(W_Q, "(n h) m->n m h", n=cfg.n_heads)
            W_K = einops.rearrange(W_K, "(n h) m->n m h", n=n_kv_heads)
            W_V = einops.rearrange(W_V, "(n h) m->n m h", n=n_kv_heads)

        state_dict[f"blocks.{l}.attn.W_Q"] = W_Q
        state_dict[f"blocks.{l}.attn.{gqa_uscore}W_K"] = W_K
        state_dict[f"blocks.{l}.attn.{gqa_uscore}W_V"] = W_V

        state_dict[f"blocks.{l}.attn.b_Q"] = torch.zeros(
            cfg.n_heads, cfg.d_head, dtype=cfg.dtype, device=cfg.device
        )
        state_dict[f"blocks.{l}.attn.{gqa_uscore}b_K"] = torch.zeros(
            n_kv_heads,
            cfg.d_head,
            dtype=cfg.dtype,
            device=cfg.device,
        )
        state_dict[f"blocks.{l}.attn.{gqa_uscore}b_V"] = torch.zeros(
            n_kv_heads,
            cfg.d_head,
            dtype=cfg.dtype,
            device=cfg.device,
        )

        W_O = llama.model.layers[l].self_attn.o_proj.weight

        if not cfg.load_in_4bit:
            W_O = einops.rearrange(W_O, "m (n h)->n h m", n=cfg.n_heads)

        state_dict[f"blocks.{l}.attn.W_O"] = W_O.to(device=cfg.device)

        state_dict[f"blocks.{l}.attn.b_O"] = torch.zeros(
            cfg.d_model, dtype=cfg.dtype, device=cfg.device
        )

        state_dict[f"blocks.{l}.ln2.w"] = llama.model.layers[l].post_attention_layernorm.weight

        # in case of quantization,
        # parameters should stay as bitsandbytes.nn.modules.Params4bit
        if not cfg.load_in_4bit:
            state_dict[f"blocks.{l}.mlp.W_in"] = llama.model.layers[l].mlp.up_proj.weight.T
            state_dict[f"blocks.{l}.mlp.W_gate"] = llama.model.layers[l].mlp.gate_proj.weight.T
            state_dict[f"blocks.{l}.mlp.W_out"] = llama.model.layers[l].mlp.down_proj.weight.T
        else:
            state_dict[f"blocks.{l}.mlp.W_in"] = llama.model.layers[l].mlp.up_proj.weight
            state_dict[f"blocks.{l}.mlp.W_gate"] = llama.model.layers[l].mlp.gate_proj.weight
            state_dict[f"blocks.{l}.mlp.W_out"] = llama.model.layers[l].mlp.down_proj.weight

        state_dict[f"blocks.{l}.mlp.b_in"] = torch.zeros(
            cfg.d_mlp, dtype=cfg.dtype, device=cfg.device
        )
        state_dict[f"blocks.{l}.mlp.b_out"] = torch.zeros(
            cfg.d_model, dtype=cfg.dtype, device=cfg.device
        )

    state_dict["ln_final.w"] = llama.model.norm.weight

    state_dict["unembed.W_U"] = llama.lm_head.weight.T
    state_dict["unembed.b_U"] = torch.zeros(cfg.d_vocab, dtype=cfg.dtype, device=cfg.device)

    return state_dict