# Qwen 2.5 is a Qwen2ForCausalLM HF model

from typing import Dict, List, Tuple
import einops
import torch
from transformer_lens import HookedTransformer
from transformer_lens.HookedTransformerConfig import HookedTransformerConfig

def convert_hooked_to_qwen2_weights(
    tl_model: HookedTransformer, 
    cfg: HookedTransformerConfig, 
    verbose: bool = False
) -> Tuple[Dict[str, torch.Tensor], List[str]]:
    """
    Converts weights from HookedTransformer format back to Hugging Face Qwen2 format.
    This is the inverse of the convert_qwen2_weights function (also applies to Qwen1.5).
    
    Args:
        tl_model: The HookedTransformer model
        cfg: HookedTransformerConfig with the model's configuration
        verbose: Whether to print missing/unexpected parameters
        
    Returns:
        tuple: (qwen_state_dict, missing_params)
            - qwen_state_dict: Dictionary with weights in the HF Qwen2 format
            - missing_params: List of expected parameters that weren't found in state_dict
    """
    tl_state_dict = tl_model.state_dict()
    qwen_state_dict = {}
    missing_params = []

    # Define base expected parameters
    expected_params = [
        "embed.W_E",
        "ln_final.w",
        "unembed.W_U",
        "unembed.b_U" # TL generates this as zeros
    ]
    
    # Add layer-specific parameters for Qwen2
    for l in range(cfg.n_layers):
        expected_params.extend([
            f"blocks.{l}.ln1.w",
            f"blocks.{l}.attn.W_Q",
            f"blocks.{l}.attn._W_K", # Qwen2 uses GQA natively
            f"blocks.{l}.attn._W_V",
            f"blocks.{l}.attn.b_Q",
            f"blocks.{l}.attn._b_K",
            f"blocks.{l}.attn._b_V",
            f"blocks.{l}.attn.W_O",
            f"blocks.{l}.attn.b_O",  # TL generates this as zeros
            f"blocks.{l}.ln2.w",
            f"blocks.{l}.mlp.W_in",
            f"blocks.{l}.mlp.W_gate",
            f"blocks.{l}.mlp.W_out",
            f"blocks.{l}.mlp.b_in",  # TL generates this as zeros
            f"blocks.{l}.mlp.b_out"  # TL generates this as zeros
        ])
            
    # Check for missing parameters
    for param in expected_params:
        if param not in tl_state_dict:
            missing_params.append(param)
            
    # Embedding layer
    if "embed.W_E" in tl_state_dict:
        qwen_state_dict["model.embed_tokens.weight"] = tl_state_dict["embed.W_E"]

    # Process transformer layers
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
        # --- Layer Norms ---
        if f"blocks.{l}.ln1.w" in tl_state_dict:
            qwen_state_dict[f"model.layers.{l}.input_layernorm.weight"] = tl_state_dict[f"blocks.{l}.ln1.w"]
        if f"blocks.{l}.ln2.w" in tl_state_dict:
            qwen_state_dict[f"model.layers.{l}.post_attention_layernorm.weight"] = tl_state_dict[f"blocks.{l}.ln2.w"]

        # --- Attention Weights ---
        if f"blocks.{l}.attn.W_Q" in tl_state_dict:
            W_Q = einops.rearrange(tl_state_dict[f"blocks.{l}.attn.W_Q"], "n m h -> (n h) m")
            qwen_state_dict[f"model.layers.{l}.self_attn.q_proj.weight"] = W_Q
            
        if f"blocks.{l}.attn._W_K" in tl_state_dict:
            W_K = einops.rearrange(tl_state_dict[f"blocks.{l}.attn._W_K"], "n m h -> (n h) m")
            qwen_state_dict[f"model.layers.{l}.self_attn.k_proj.weight"] = W_K
            
        if f"blocks.{l}.attn._W_V" in tl_state_dict:
            W_V = einops.rearrange(tl_state_dict[f"blocks.{l}.attn._W_V"], "n m h -> (n h) m")
            qwen_state_dict[f"model.layers.{l}.self_attn.v_proj.weight"] = W_V
            
        if f"blocks.{l}.attn.W_O" in tl_state_dict:
            W_O = einops.rearrange(tl_state_dict[f"blocks.{l}.attn.W_O"], "n h m -> m (n h)")
            qwen_state_dict[f"model.layers.{l}.self_attn.o_proj.weight"] = W_O

        # --- Attention Biases ---
        if f"blocks.{l}.attn.b_Q" in tl_state_dict:
            b_Q = einops.rearrange(tl_state_dict[f"blocks.{l}.attn.b_Q"], "n h -> (n h)")
            qwen_state_dict[f"model.layers.{l}.self_attn.q_proj.bias"] = b_Q

        if f"blocks.{l}.attn._b_K" in tl_state_dict:
            b_K = einops.rearrange(tl_state_dict[f"blocks.{l}.attn._b_K"], "n h -> (n h)")
            qwen_state_dict[f"model.layers.{l}.self_attn.k_proj.bias"] = b_K

        if f"blocks.{l}.attn._b_V" in tl_state_dict:
            b_V = einops.rearrange(tl_state_dict[f"blocks.{l}.attn._b_V"], "n h -> (n h)")
            qwen_state_dict[f"model.layers.{l}.self_attn.v_proj.bias"] = b_V

        # Note: We skip b_O here because standard HF Qwen architectures do not use an output bias.

        # --- MLP ---
        if f"blocks.{l}.mlp.W_in" in tl_state_dict:
            qwen_state_dict[f"model.layers.{l}.mlp.up_proj.weight"] = tl_state_dict[f"blocks.{l}.mlp.W_in"].T
        if f"blocks.{l}.mlp.W_gate" in tl_state_dict:
            qwen_state_dict[f"model.layers.{l}.mlp.gate_proj.weight"] = tl_state_dict[f"blocks.{l}.mlp.W_gate"].T
        if f"blocks.{l}.mlp.W_out" in tl_state_dict:
            qwen_state_dict[f"model.layers.{l}.mlp.down_proj.weight"] = tl_state_dict[f"blocks.{l}.mlp.W_out"].T

        # Note: We skip b_in and b_out here because HF Qwen does not use MLP biases.

    # --- Final Layer Norm ---
    if "ln_final.w" in tl_state_dict:
        qwen_state_dict["model.norm.weight"] = tl_state_dict["ln_final.w"]

    # --- Unembedding ---
    if "unembed.W_U" in tl_state_dict:
        qwen_state_dict["lm_head.weight"] = tl_state_dict["unembed.W_U"].T

    # Check for unexpected parameters
    unexpected_params = [param for param in tl_state_dict.keys() if param not in expected_params]

    if verbose:
        if unexpected_params:
            print("Unexpected parameters:")
            for param in unexpected_params:
                print(f"  - {param}")
                
        if missing_params:
            print("Missing parameters:")
            for param in missing_params:
                print(f"  - {param}")

    return qwen_state_dict, missing_params


def convert_qwen2_to_hooked_weights(qwen, cfg: HookedTransformerConfig):
    # Note that this method is also applied for Qwen1.5 models, since they
    # have architecture type Qwen2ForCausalLM.

    state_dict = {}

    state_dict["embed.W_E"] = qwen.model.embed_tokens.weight

    assert cfg.d_mlp is not None  # keep mypy happy

    for l in range(cfg.n_layers):
        state_dict[f"blocks.{l}.ln1.w"] = qwen.model.layers[l].input_layernorm.weight

        W_Q = qwen.model.layers[l].self_attn.q_proj.weight
        W_K = qwen.model.layers[l].self_attn.k_proj.weight
        W_V = qwen.model.layers[l].self_attn.v_proj.weight
        W_Q = einops.rearrange(W_Q, "(n h) m->n m h", n=cfg.n_heads)
        W_K = einops.rearrange(W_K, "(n h) m->n m h", n=cfg.n_key_value_heads)
        W_V = einops.rearrange(W_V, "(n h) m->n m h", n=cfg.n_key_value_heads)

        state_dict[f"blocks.{l}.attn.W_Q"] = W_Q
        state_dict[f"blocks.{l}.attn._W_K"] = W_K
        state_dict[f"blocks.{l}.attn._W_V"] = W_V

        b_Q = qwen.model.layers[l].self_attn.q_proj.bias
        b_Q = einops.rearrange(
            b_Q,
            "(n_head d_head) -> n_head d_head",
            n_head=cfg.n_heads,
        )

        b_K = qwen.model.layers[l].self_attn.k_proj.bias
        b_K = einops.rearrange(
            b_K,
            "(n_head d_head) -> n_head d_head",
            n_head=cfg.n_key_value_heads,
        )

        b_V = qwen.model.layers[l].self_attn.v_proj.bias
        b_V = einops.rearrange(
            b_V,
            "(n_head d_head) -> n_head d_head",
            n_head=cfg.n_key_value_heads,
        )

        state_dict[f"blocks.{l}.attn.b_Q"] = b_Q
        state_dict[f"blocks.{l}.attn._b_K"] = b_K
        state_dict[f"blocks.{l}.attn._b_V"] = b_V

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