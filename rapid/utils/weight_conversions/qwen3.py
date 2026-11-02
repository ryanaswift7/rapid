from typing import Dict, List, Tuple, Any
import einops
import torch
from transformer_lens import HookedTransformer
from transformer_lens.HookedTransformerConfig import HookedTransformerConfig

def convert_hooked_to_qwen3_weights(
    tl_model: HookedTransformer, 
    cfg: HookedTransformerConfig, 
    verbose: bool = False
) -> Tuple[Dict[str, torch.Tensor], List[str]]:
    """
    Converts weights from HookedTransformer format back to Hugging Face Qwen3 format.
    This is the inverse of the convert_qwen3_weights function.
    
    Args:
        tl_model: The HookedTransformer model
        cfg: HookedTransformerConfig with the model's configuration
        verbose: Whether to print missing/unexpected parameters
        
    Returns:
        tuple: (qwen3_state_dict, missing_params)
            - qwen3_state_dict: Dictionary with weights in the HF Qwen3 format
            - missing_params: List of expected parameters that weren't found in state_dict
    """
    tl_state_dict = tl_model.state_dict()
    qwen3_state_dict = {}
    missing_params = []

    # Handle GQA formatting based on config
    if cfg.n_key_value_heads is None:
        gqa_uscore = ""
    else:
        gqa_uscore = "_"

    # Define base expected parameters in the HookedTransformer state dict
    expected_params = [
        "embed.W_E",
        "ln_final.w",
        "unembed.W_U",
        "unembed.b_U"  # TL generates this as zeros
    ]
    
    # Add layer-specific parameters to expected_params
    for l in range(cfg.n_layers):
        expected_params.extend([
            f"blocks.{l}.ln1.w",
            f"blocks.{l}.attn.W_Q",
            f"blocks.{l}.attn.{gqa_uscore}W_K",
            f"blocks.{l}.attn.{gqa_uscore}W_V",
            f"blocks.{l}.attn.b_Q",            # TL generates as zeros
            f"blocks.{l}.attn.{gqa_uscore}b_K",  # TL generates as zeros
            f"blocks.{l}.attn.{gqa_uscore}b_V",  # TL generates as zeros
            f"blocks.{l}.attn.W_O",
            f"blocks.{l}.attn.b_O",            # TL generates as zeros
            f"blocks.{l}.attn.q_norm.w",       # Qwen3 specific Q-norm
            f"blocks.{l}.attn.k_norm.w",       # Qwen3 specific K-norm
            f"blocks.{l}.ln2.w",
            f"blocks.{l}.mlp.W_in",
            f"blocks.{l}.mlp.W_gate",
            f"blocks.{l}.mlp.W_out",
            f"blocks.{l}.mlp.b_in",            # TL generates as zeros
            f"blocks.{l}.mlp.b_out"            # TL generates as zeros
        ])
    
    # Check for missing parameters
    for param in expected_params:
        if param not in tl_state_dict:
            missing_params.append(param)
    
    # Embedding layer
    if "embed.W_E" in tl_state_dict:
        qwen3_state_dict["model.embed_tokens.weight"] = tl_state_dict["embed.W_E"]
    
    # Process all transformer layers
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
        # Layer norms (Qwen3 uses RMSNorm)
        if f"blocks.{l}.ln1.w" in tl_state_dict:
            qwen3_state_dict[f"model.layers.{l}.input_layernorm.weight"] = tl_state_dict[f"blocks.{l}.ln1.w"]
        if f"blocks.{l}.ln2.w" in tl_state_dict:
            qwen3_state_dict[f"model.layers.{l}.post_attention_layernorm.weight"] = tl_state_dict[f"blocks.{l}.ln2.w"]
        
        # QK Norms (Qwen3 specific feature)
        if f"blocks.{l}.attn.q_norm.w" in tl_state_dict:
            qwen3_state_dict[f"model.layers.{l}.self_attn.q_norm.weight"] = tl_state_dict[f"blocks.{l}.attn.q_norm.w"]
        if f"blocks.{l}.attn.k_norm.w" in tl_state_dict:
            qwen3_state_dict[f"model.layers.{l}.self_attn.k_norm.weight"] = tl_state_dict[f"blocks.{l}.attn.k_norm.w"]

        # Attention Weight Projections
        if f"blocks.{l}.attn.W_Q" in tl_state_dict:
            W_Q = einops.rearrange(tl_state_dict[f"blocks.{l}.attn.W_Q"], "n m h -> (n h) m")
            qwen3_state_dict[f"model.layers.{l}.self_attn.q_proj.weight"] = W_Q

        if f"blocks.{l}.attn.{gqa_uscore}W_K" in tl_state_dict:
            W_K = einops.rearrange(tl_state_dict[f"blocks.{l}.attn.{gqa_uscore}W_K"], "n m h -> (n h) m")
            qwen3_state_dict[f"model.layers.{l}.self_attn.k_proj.weight"] = W_K

        if f"blocks.{l}.attn.{gqa_uscore}W_V" in tl_state_dict:
            W_V = einops.rearrange(tl_state_dict[f"blocks.{l}.attn.{gqa_uscore}W_V"], "n m h -> (n h) m")
            qwen3_state_dict[f"model.layers.{l}.self_attn.v_proj.weight"] = W_V

        if f"blocks.{l}.attn.W_O" in tl_state_dict:
            W_O = einops.rearrange(tl_state_dict[f"blocks.{l}.attn.W_O"], "n h m -> m (n h)")
            qwen3_state_dict[f"model.layers.{l}.self_attn.o_proj.weight"] = W_O
            
        # Note: Intentionally skipping b_Q, b_K, b_V, b_O. HF Qwen3 does not use attention biases.

        # MLP
        if f"blocks.{l}.mlp.W_in" in tl_state_dict:
            qwen3_state_dict[f"model.layers.{l}.mlp.up_proj.weight"] = tl_state_dict[f"blocks.{l}.mlp.W_in"].T
            
        if f"blocks.{l}.mlp.W_gate" in tl_state_dict:
            qwen3_state_dict[f"model.layers.{l}.mlp.gate_proj.weight"] = tl_state_dict[f"blocks.{l}.mlp.W_gate"].T

        if f"blocks.{l}.mlp.W_out" in tl_state_dict:
            qwen3_state_dict[f"model.layers.{l}.mlp.down_proj.weight"] = tl_state_dict[f"blocks.{l}.mlp.W_out"].T

        # Note: Intentionally skipping b_in and b_out. HF Qwen3 does not use MLP biases.
    
    # Final layer norm
    if "ln_final.w" in tl_state_dict:
        qwen3_state_dict["model.norm.weight"] = tl_state_dict["ln_final.w"]
    
    # Unembedding (output) layer
    if "unembed.W_U" in tl_state_dict:
        qwen3_state_dict["lm_head.weight"] = tl_state_dict["unembed.W_U"].T
    
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
    
    return qwen3_state_dict, missing_params


def convert_qwen3_to_hooked_weights(qwen: Any, cfg: HookedTransformerConfig):
    """Convert Qwen3 weights to TransformerLens format."""
    state_dict = {}

    state_dict["embed.W_E"] = qwen.model.embed_tokens.weight

    if cfg.n_key_value_heads is None:
        gqa_uscore = ""
        n_kv_heads = cfg.n_heads
    else:
        gqa_uscore = "_"
        n_kv_heads = cfg.n_key_value_heads

    assert cfg.d_mlp is not None  # keep mypy happy

    for l in range(cfg.n_layers):
        state_dict[f"blocks.{l}.ln1.w"] = qwen.model.layers[l].input_layernorm.weight

        W_Q = qwen.model.layers[l].self_attn.q_proj.weight
        W_K = qwen.model.layers[l].self_attn.k_proj.weight
        W_V = qwen.model.layers[l].self_attn.v_proj.weight
        W_Q = einops.rearrange(W_Q, "(n h) m->n m h", n=cfg.n_heads)
        W_K = einops.rearrange(W_K, "(n h) m->n m h", n=n_kv_heads)
        W_V = einops.rearrange(W_V, "(n h) m->n m h", n=n_kv_heads)

        state_dict[f"blocks.{l}.attn.W_Q"] = W_Q
        state_dict[f"blocks.{l}.attn.{gqa_uscore}W_K"] = W_K
        state_dict[f"blocks.{l}.attn.{gqa_uscore}W_V"] = W_V

        # Load weights into RMSNorm modules
        state_dict[f"blocks.{l}.attn.q_norm.w"] = qwen.model.layers[l].self_attn.q_norm.weight
        state_dict[f"blocks.{l}.attn.k_norm.w"] = qwen.model.layers[l].self_attn.k_norm.weight

        state_dict[f"blocks.{l}.attn.b_Q"] = torch.zeros(cfg.n_heads, cfg.d_head, dtype=cfg.dtype)
        state_dict[f"blocks.{l}.attn.{gqa_uscore}b_K"] = torch.zeros(
            n_kv_heads, cfg.d_head, dtype=cfg.dtype
        )
        state_dict[f"blocks.{l}.attn.{gqa_uscore}b_V"] = torch.zeros(
            n_kv_heads, cfg.d_head, dtype=cfg.dtype
        )

        W_O = qwen.model.layers[l].self_attn.o_proj.weight
        W_O = einops.rearrange(W_O, "m (n h)->n h m", n=cfg.n_heads)
        state_dict[f"blocks.{l}.attn.W_O"] = W_O

        state_dict[f"blocks.{l}.attn.b_O"] = torch.zeros(cfg.d_model, dtype=cfg.dtype)

        state_dict[f"blocks.{l}.ln2.w"] = qwen.model.layers[l].post_attention_layernorm.weight

        state_dict[f"blocks.{l}.mlp.W_in"] = qwen.model.layers[l].mlp.up_proj.weight.T
        state_dict[f"blocks.{l}.mlp.W_gate"] = qwen.model.layers[l].mlp.gate_proj.weight.T
        state_dict[f"blocks.{l}.mlp.b_in"] = torch.zeros(cfg.d_mlp, dtype=cfg.dtype)

        state_dict[f"blocks.{l}.mlp.W_out"] = qwen.model.layers[l].mlp.down_proj.weight.T
        state_dict[f"blocks.{l}.mlp.b_out"] = torch.zeros(cfg.d_model, dtype=cfg.dtype)

    state_dict["ln_final.w"] = qwen.model.norm.weight

    state_dict["unembed.W_U"] = qwen.lm_head.weight.T
    state_dict["unembed.b_U"] = torch.zeros(cfg.d_vocab, dtype=cfg.dtype)

    return state_dict