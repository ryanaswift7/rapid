import einops
import torch
from transformer_lens.HookedTransformerConfig import HookedTransformerConfig

def convert_hooked_to_neox_weights(tl_model, cfg: HookedTransformerConfig, verbose: bool = False):
    """
    Converts weights from HookedTransformer format back to GPT-NeoX format.
    This is the inverse of the convert_neox_weights function.
    
    Args:
        tl_model: The HookedTransformer model
        cfg: HookedTransformerConfig with the model's configuration
        
    Returns:
        tuple: (neox_state_dict, missing_params)
            - neox_state_dict: Dictionary with weights in the GPT-NeoX format
            - missing_params: List of expected parameters that weren't found in state_dict
    """
    tl_state_dict = tl_model.state_dict()
    neox_state_dict = {}
    missing_params = []
    
    # Define expected parameters in the HookedTransformer state dict
    expected_params = [
        "embed.W_E",
        "ln_final.w",
        "ln_final.b",
        "unembed.W_U",
        "unembed.b_U"
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
    
    # Embedding layer
    if "embed.W_E" in tl_state_dict:
        neox_state_dict["gpt_neox.embed_in.weight"] = tl_state_dict["embed.W_E"]
    
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
        # Layer norms
        if f"blocks.{l}.ln1.w" in tl_state_dict:
            neox_state_dict[f"gpt_neox.layers.{l}.input_layernorm.weight"] = tl_state_dict[f"blocks.{l}.ln1.w"]
        if f"blocks.{l}.ln1.b" in tl_state_dict:
            neox_state_dict[f"gpt_neox.layers.{l}.input_layernorm.bias"] = tl_state_dict[f"blocks.{l}.ln1.b"]
        if f"blocks.{l}.ln2.w" in tl_state_dict:
            neox_state_dict[f"gpt_neox.layers.{l}.post_attention_layernorm.weight"] = tl_state_dict[f"blocks.{l}.ln2.w"]
        if f"blocks.{l}.ln2.b" in tl_state_dict:
            neox_state_dict[f"gpt_neox.layers.{l}.post_attention_layernorm.bias"] = tl_state_dict[f"blocks.{l}.ln2.b"]
        
        # Check if all attention parameters are present for this layer
        attn_params_present = all(
            f"blocks.{l}.attn.{p}" in tl_state_dict 
            for p in ["W_Q", "W_K", "W_V", "b_Q", "b_K", "b_V"]
        )
        
        if attn_params_present:
            # Attention - Convert separate Q, K, V matrices back to the concatenated form
            W_Q = tl_state_dict[f"blocks.{l}.attn.W_Q"]
            W_K = tl_state_dict[f"blocks.{l}.attn.W_K"]
            W_V = tl_state_dict[f"blocks.{l}.attn.W_V"]
            
            # Stack them along the QKV dimension
            W_QKV = torch.stack([W_Q, W_K, W_V], dim=0)
            
            # Rearrange to NeoX's format
            W_QKV = einops.rearrange(W_QKV, "qkv i m h -> (i qkv h) m")
            neox_state_dict[f"gpt_neox.layers.{l}.attention.query_key_value.weight"] = W_QKV
            
            # Biases
            b_Q = tl_state_dict[f"blocks.{l}.attn.b_Q"]
            b_K = tl_state_dict[f"blocks.{l}.attn.b_K"]
            b_V = tl_state_dict[f"blocks.{l}.attn.b_V"]
            
            # Stack biases along QKV dimension
            b_QKV = torch.stack([b_Q, b_K, b_V], dim=0)
            
            # Rearrange to NeoX's format
            b_QKV = einops.rearrange(b_QKV, "qkv index head -> (index qkv head)")
            neox_state_dict[f"gpt_neox.layers.{l}.attention.query_key_value.bias"] = b_QKV
        
        # Output projection
        if f"blocks.{l}.attn.W_O" in tl_state_dict:
            W_O = tl_state_dict[f"blocks.{l}.attn.W_O"]
            W_O = einops.rearrange(W_O, "i h m -> m (i h)")
            neox_state_dict[f"gpt_neox.layers.{l}.attention.dense.weight"] = W_O
        
        if f"blocks.{l}.attn.b_O" in tl_state_dict:
            neox_state_dict[f"gpt_neox.layers.{l}.attention.dense.bias"] = tl_state_dict[f"blocks.{l}.attn.b_O"]
        
        # MLP
        if f"blocks.{l}.mlp.W_in" in tl_state_dict:
            neox_state_dict[f"gpt_neox.layers.{l}.mlp.dense_h_to_4h.weight"] = tl_state_dict[f"blocks.{l}.mlp.W_in"].T
        if f"blocks.{l}.mlp.b_in" in tl_state_dict:
            neox_state_dict[f"gpt_neox.layers.{l}.mlp.dense_h_to_4h.bias"] = tl_state_dict[f"blocks.{l}.mlp.b_in"]
        if f"blocks.{l}.mlp.W_out" in tl_state_dict:
            neox_state_dict[f"gpt_neox.layers.{l}.mlp.dense_4h_to_h.weight"] = tl_state_dict[f"blocks.{l}.mlp.W_out"].T
        if f"blocks.{l}.mlp.b_out" in tl_state_dict:
            neox_state_dict[f"gpt_neox.layers.{l}.mlp.dense_4h_to_h.bias"] = tl_state_dict[f"blocks.{l}.mlp.b_out"]
    
    # Final layer norm
    if "ln_final.w" in tl_state_dict:
        neox_state_dict["gpt_neox.final_layer_norm.weight"] = tl_state_dict["ln_final.w"]
    if "ln_final.b" in tl_state_dict:
        neox_state_dict["gpt_neox.final_layer_norm.bias"] = tl_state_dict["ln_final.b"]
    
    # Unembedding (output) layer
    if "unembed.W_U" in tl_state_dict:
        neox_state_dict["embed_out.weight"] = tl_state_dict["unembed.W_U"].T
    
    # Check for unexpected parameters (not in our expected list but present in state_dict)
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
    
    return neox_state_dict, missing_params

# taken from TransformerLens GitHub:
# https://github.com/TransformerLensOrg/TransformerLens/blob/main/transformer_lens/pretrained/weight_conversions/neox.py
def convert_neox_to_hooked_weights(neox_model, cfg: HookedTransformerConfig):
    tl_state_dict = {}

    tl_state_dict["embed.W_E"] = neox_model.gpt_neox.embed_in.weight

    for l in range(cfg.n_layers):
        tl_state_dict[f"blocks.{l}.ln1.w"] = neox_model.gpt_neox.layers[l].input_layernorm.weight
        tl_state_dict[f"blocks.{l}.ln1.b"] = neox_model.gpt_neox.layers[l].input_layernorm.bias

        # For some inexplicable reason, NeoX both uses the concatenated QKV
        # matmul of GPT-2 (afaict this has a neglible performance impact) AND
        # has the flattened axis in the DIFFERENT order of (head_index qkv
        # d_head) - this took me an hour to debug...
        W = neox_model.gpt_neox.layers[l].attention.query_key_value.weight
        W = einops.rearrange(W, "(i qkv h) m->qkv i m h", i=cfg.n_heads, qkv=3)

        # Fold in layer norm weights
        tl_state_dict[f"blocks.{l}.attn.W_Q"] = W[0]
        tl_state_dict[f"blocks.{l}.attn.W_K"] = W[1]
        tl_state_dict[f"blocks.{l}.attn.W_V"] = W[2]

        qkv_bias = neox_model.gpt_neox.layers[l].attention.query_key_value.bias
        qkv_bias = einops.rearrange(
            qkv_bias,
            "(index qkv head)->qkv index head",
            qkv=3,
            index=cfg.n_heads,
            head=cfg.d_head,
        )
        # Fold in layer norm biases
        tl_state_dict[f"blocks.{l}.attn.b_Q"] = qkv_bias[0]
        tl_state_dict[f"blocks.{l}.attn.b_K"] = qkv_bias[1]
        tl_state_dict[f"blocks.{l}.attn.b_V"] = qkv_bias[2]

        W_O = neox_model.gpt_neox.layers[l].attention.dense.weight
        W_O = einops.rearrange(W_O, "m (i h)->i h m", i=cfg.n_heads)
        tl_state_dict[f"blocks.{l}.attn.W_O"] = W_O
        tl_state_dict[f"blocks.{l}.attn.b_O"] = neox_model.gpt_neox.layers[l].attention.dense.bias

        tl_state_dict[f"blocks.{l}.ln2.w"] = neox_model.gpt_neox.layers[l].post_attention_layernorm.weight
        tl_state_dict[f"blocks.{l}.ln2.b"] = neox_model.gpt_neox.layers[l].post_attention_layernorm.bias

        tl_state_dict[f"blocks.{l}.mlp.W_in"] = neox_model.gpt_neox.layers[l].mlp.dense_h_to_4h.weight.T
        tl_state_dict[f"blocks.{l}.mlp.b_in"] = neox_model.gpt_neox.layers[l].mlp.dense_h_to_4h.bias

        tl_state_dict[f"blocks.{l}.mlp.W_out"] = neox_model.gpt_neox.layers[l].mlp.dense_4h_to_h.weight.T
        tl_state_dict[f"blocks.{l}.mlp.b_out"] = neox_model.gpt_neox.layers[l].mlp.dense_4h_to_h.bias
    tl_state_dict["ln_final.w"] = neox_model.gpt_neox.final_layer_norm.weight
    tl_state_dict["ln_final.b"] = neox_model.gpt_neox.final_layer_norm.bias

    tl_state_dict["unembed.W_U"] = neox_model.embed_out.weight.T
    tl_state_dict["unembed.b_U"] = torch.zeros(cfg.d_vocab, dtype=cfg.dtype)
    return tl_state_dict


