import einops
import torch
from transformer_lens.HookedTransformerConfig import HookedTransformerConfig


def convert_hooked_to_gpt2_weights(tl_state_dict, cfg: HookedTransformerConfig, verbose: bool = False):
    """
    Converts weights from HookedTransformer format back to GPT-2 format.
    This is the exact inverse of the convert_gpt2_weights function.
    
    Args:
        state_dict: The HookedTransformer state dictionary
        cfg: HookedTransformerConfig with the model's configuration
        
    Returns:
        tuple: (gpt2_state_dict, missing_params)
            - gpt2_state_dict: Dictionary with weights in the GPT-2 format
            - missing_params: List of expected parameters that weren't found in state_dict
    """
    gpt2_state_dict = {}
    missing_params = []
    
    # Define expected parameters in the HookedTransformer state dict
    expected_params = [
        "embed.W_E",
        "pos_embed.W_pos",
        "ln_final.w",
        "ln_final.b",
        "unembed.W_U"
    ]
    
    # Add layer-specific parameters to expected_params
    for l in range(cfg.n_layers):
        expected_params.extend([
            f"blocks.{l}.ln1.w",
            f"blocks.{l}.ln1.b",
            f"blocks.{l}.attn.W_Q",
            f"blocks.{l}.attn.W_K",
            f"blocks.{l}.attn.W_V",
            f"blocks.{l}.attn.b_Q",
            f"blocks.{l}.attn.b_K",
            f"blocks.{l}.attn.b_V",
            f"blocks.{l}.attn.W_O",
            f"blocks.{l}.attn.b_O",
            f"blocks.{l}.ln2.w",
            f"blocks.{l}.ln2.b",
            f"blocks.{l}.mlp.W_in",
            f"blocks.{l}.mlp.b_in",
            f"blocks.{l}.mlp.W_out",
            f"blocks.{l}.mlp.b_out"
        ])
    
    # Check for missing parameters
    for param in expected_params:
        if param not in tl_state_dict:
            missing_params.append(param)
    
    # Embedding layers - Direct copy (no transformation in forward direction)
    if "embed.W_E" in tl_state_dict:
        gpt2_state_dict["transformer.wte.weight"] = tl_state_dict["embed.W_E"]
    if "pos_embed.W_pos" in tl_state_dict:
        gpt2_state_dict["transformer.wpe.weight"] = tl_state_dict["pos_embed.W_pos"]
    
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
        # Layer norms - Direct copy (no transformation in forward direction)
        if f"blocks.{l}.ln1.w" in tl_state_dict:
            gpt2_state_dict[f"transformer.h.{l}.ln_1.weight"] = tl_state_dict[f"blocks.{l}.ln1.w"]
        if f"blocks.{l}.ln1.b" in tl_state_dict:
            gpt2_state_dict[f"transformer.h.{l}.ln_1.bias"] = tl_state_dict[f"blocks.{l}.ln1.b"]
        if f"blocks.{l}.ln2.w" in tl_state_dict:
            gpt2_state_dict[f"transformer.h.{l}.ln_2.weight"] = tl_state_dict[f"blocks.{l}.ln2.w"]
        if f"blocks.{l}.ln2.b" in tl_state_dict:
            gpt2_state_dict[f"transformer.h.{l}.ln_2.bias"] = tl_state_dict[f"blocks.{l}.ln2.b"]
        
        # Attention - Convert separate Q, K, V matrices back to the concatenated form
        # Forward: W_Q, W_K, W_V = torch.tensor_split(W, 3, dim=1)
        # Forward: W_Q = einops.rearrange(W_Q, "m (i h)->i m h", i=cfg.n_heads)
        # Reverse: W_Q = einops.rearrange(W_Q, "i m h->m (i h)")
        if all(f"blocks.{l}.attn.{p}" in tl_state_dict for p in ["W_Q", "W_K", "W_V"]):
            W_Q = einops.rearrange(tl_state_dict[f"blocks.{l}.attn.W_Q"], "i m h->m (i h)")
            W_K = einops.rearrange(tl_state_dict[f"blocks.{l}.attn.W_K"], "i m h->m (i h)")
            W_V = einops.rearrange(tl_state_dict[f"blocks.{l}.attn.W_V"], "i m h->m (i h)")
            W_QKV = torch.cat([W_Q, W_K, W_V], dim=1)
            gpt2_state_dict[f"transformer.h.{l}.attn.c_attn.weight"] = W_QKV
            
            # Attention biases
            # Forward: qkv_bias = einops.rearrange(qkv_bias, "(qkv index head)->qkv index head", qkv=3, index=cfg.n_heads, head=cfg.d_head)
            # Reverse: qkv_bias = einops.rearrange(qkv_bias, "qkv index head->(qkv index head)")
            if all(f"blocks.{l}.attn.{p}" in tl_state_dict for p in ["b_Q", "b_K", "b_V"]):
                b_Q = tl_state_dict[f"blocks.{l}.attn.b_Q"]  # [n_heads, d_head]
                b_K = tl_state_dict[f"blocks.{l}.attn.b_K"]  # [n_heads, d_head]
                b_V = tl_state_dict[f"blocks.{l}.attn.b_V"]  # [n_heads, d_head]
                
                # Stack them back into the original format: [3, n_heads, d_head]
                qkv_bias = torch.stack([b_Q, b_K, b_V], dim=0)
                # Reshape back to the original flat format
                qkv_bias_flat = einops.rearrange(qkv_bias, "qkv index head->(qkv index head)")
                gpt2_state_dict[f"transformer.h.{l}.attn.c_attn.bias"] = qkv_bias_flat
        
        # Output projection
        # Forward: W_O = einops.rearrange(W_O, "(i h) m->i h m", i=cfg.n_heads)
        # Reverse: W_O = einops.rearrange(W_O, "i h m->(i h) m")
        if f"blocks.{l}.attn.W_O" in tl_state_dict:
            W_O = einops.rearrange(tl_state_dict[f"blocks.{l}.attn.W_O"], "i h m->(i h) m")
            gpt2_state_dict[f"transformer.h.{l}.attn.c_proj.weight"] = W_O
        if f"blocks.{l}.attn.b_O" in tl_state_dict:
            gpt2_state_dict[f"transformer.h.{l}.attn.c_proj.bias"] = tl_state_dict[f"blocks.{l}.attn.b_O"]
        
        # MLP - Direct copy (no transformation in forward direction)
        if f"blocks.{l}.mlp.W_in" in tl_state_dict:
            gpt2_state_dict[f"transformer.h.{l}.mlp.c_fc.weight"] = tl_state_dict[f"blocks.{l}.mlp.W_in"]
        if f"blocks.{l}.mlp.b_in" in tl_state_dict:
            gpt2_state_dict[f"transformer.h.{l}.mlp.c_fc.bias"] = tl_state_dict[f"blocks.{l}.mlp.b_in"]
        if f"blocks.{l}.mlp.W_out" in tl_state_dict:
            gpt2_state_dict[f"transformer.h.{l}.mlp.c_proj.weight"] = tl_state_dict[f"blocks.{l}.mlp.W_out"]
        if f"blocks.{l}.mlp.b_out" in tl_state_dict:
            gpt2_state_dict[f"transformer.h.{l}.mlp.c_proj.bias"] = tl_state_dict[f"blocks.{l}.mlp.b_out"]
    
    # Final layer norm - Direct copy (no transformation in forward direction)
    if "ln_final.w" in tl_state_dict:
        gpt2_state_dict["transformer.ln_f.weight"] = tl_state_dict["ln_final.w"]
    if "ln_final.b" in tl_state_dict:
        gpt2_state_dict["transformer.ln_f.bias"] = tl_state_dict["ln_final.b"]
    
    # Unembedding (output) layer
    # Forward: state_dict["unembed.W_U"] = gpt2.lm_head.weight.T
    # Reverse: gpt2.lm_head.weight = state_dict["unembed.W_U"].T
    if "unembed.W_U" in tl_state_dict:
        gpt2_state_dict["lm_head.weight"] = tl_state_dict["unembed.W_U"].T
    
    # Check for unexpected parameters (not in our expected list but present in state_dict)
    unexpected_params = [param for param in tl_state_dict.keys() if param not in expected_params]

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
    
    return gpt2_state_dict, missing_params

# taken from 
# https://github.com/TransformerLensOrg/TransformerLens/blob/main/transformer_lens/pretrained/weight_conversions/gpt2.py
def convert_gpt2_to_hooked_weights(gpt2_state_dict, cfg):
    """Your original forward conversion function"""
    tl_state_dict = {}
    tl_state_dict["embed.W_E"] = gpt2_state_dict.transformer.wte.weight
    tl_state_dict["pos_embed.W_pos"] = gpt2_state_dict.transformer.wpe.weight
    for l in range(cfg.n_layers):
        tl_state_dict[f"blocks.{l}.ln1.w"] = gpt2_state_dict.transformer.h[l].ln_1.weight
        tl_state_dict[f"blocks.{l}.ln1.b"] = gpt2_state_dict.transformer.h[l].ln_1.bias
        # In GPT-2, q,k,v are produced by one big linear map, whose output is
        # concat([q, k, v])
        W = gpt2_state_dict.transformer.h[l].attn.c_attn.weight
        W_Q, W_K, W_V = torch.tensor_split(W, 3, dim=1)
        W_Q = einops.rearrange(W_Q, "m (i h)->i m h", i=cfg.n_heads)
        W_K = einops.rearrange(W_K, "m (i h)->i m h", i=cfg.n_heads)
        W_V = einops.rearrange(W_V, "m (i h)->i m h", i=cfg.n_heads)
        tl_state_dict[f"blocks.{l}.attn.W_Q"] = W_Q
        tl_state_dict[f"blocks.{l}.attn.W_K"] = W_K
        tl_state_dict[f"blocks.{l}.attn.W_V"] = W_V
        qkv_bias = gpt2_state_dict.transformer.h[l].attn.c_attn.bias
        qkv_bias = einops.rearrange(
            qkv_bias,
            "(qkv index head)->qkv index head",
            qkv=3,
            index=cfg.n_heads,
            head=cfg.d_head,
        )
        tl_state_dict[f"blocks.{l}.attn.b_Q"] = qkv_bias[0]
        tl_state_dict[f"blocks.{l}.attn.b_K"] = qkv_bias[1]
        tl_state_dict[f"blocks.{l}.attn.b_V"] = qkv_bias[2]
        W_O = gpt2_state_dict.transformer.h[l].attn.c_proj.weight
        W_O = einops.rearrange(W_O, "(i h) m->i h m", i=cfg.n_heads)
        tl_state_dict[f"blocks.{l}.attn.W_O"] = W_O
        tl_state_dict[f"blocks.{l}.attn.b_O"] = gpt2_state_dict.transformer.h[l].attn.c_proj.bias
        tl_state_dict[f"blocks.{l}.ln2.w"] = gpt2_state_dict.transformer.h[l].ln_2.weight
        tl_state_dict[f"blocks.{l}.ln2.b"] = gpt2_state_dict.transformer.h[l].ln_2.bias
        W_in = gpt2_state_dict.transformer.h[l].mlp.c_fc.weight
        tl_state_dict[f"blocks.{l}.mlp.W_in"] = W_in
        tl_state_dict[f"blocks.{l}.mlp.b_in"] = gpt2_state_dict.transformer.h[l].mlp.c_fc.bias
        W_out = gpt2_state_dict.transformer.h[l].mlp.c_proj.weight
        tl_state_dict[f"blocks.{l}.mlp.W_out"] = W_out
        tl_state_dict[f"blocks.{l}.mlp.b_out"] = gpt2_state_dict.transformer.h[l].mlp.c_proj.bias
    tl_state_dict["unembed.W_U"] = gpt2_state_dict.lm_head.weight.T
    tl_state_dict["ln_final.w"] = gpt2_state_dict.transformer.ln_f.weight
    tl_state_dict["ln_final.b"] = gpt2_state_dict.transformer.ln_f.bias
    return tl_state_dict