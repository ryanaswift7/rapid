from typing import Dict, List, Tuple
import einops
import torch
from transformer_lens import HookedTransformer
from transformer_lens.HookedTransformerConfig import HookedTransformerConfig
import math

def convert_hooked_to_gemma_weights(
    tl_model: HookedTransformer,
    cfg: HookedTransformerConfig,
    verbose: bool = False,
) -> Tuple[Dict[str, torch.Tensor], List[str]]:
    """
    Converts weights from HookedTransformer format back to Hugging Face Gemma format.
    Returns (gemma_state_dict, missing_params).
    """
    assert cfg.n_key_value_heads is not None
    assert cfg.d_mlp is not None

    tl_state_dict = tl_model.state_dict()
    gemma_state_dict: Dict[str, torch.Tensor] = {}

    # Build expected params based on cfg
    expected_params = [
        "embed.W_E",
        "ln_final.w",
        "unembed.W_U",
        "unembed.b_U",
    ]
    for l in range(cfg.n_layers):
        expected_params.extend([
            f"blocks.{l}.ln1.w",
            f"blocks.{l}.attn.W_Q",
            f"blocks.{l}.attn._W_K",
            f"blocks.{l}.attn._W_V",
            f"blocks.{l}.attn.b_Q",
            f"blocks.{l}.attn._b_K",
            f"blocks.{l}.attn._b_V",
            f"blocks.{l}.attn.W_O",
            f"blocks.{l}.attn.b_O",
            f"blocks.{l}.mlp.W_in",
            f"blocks.{l}.mlp.W_gate",
            f"blocks.{l}.mlp.W_out",
            f"blocks.{l}.mlp.b_in",
            f"blocks.{l}.mlp.b_out",
        ])
        if cfg.use_normalization_before_and_after:
            expected_params.extend([
                f"blocks.{l}.ln1_post.w",
                f"blocks.{l}.ln2.w",
                f"blocks.{l}.ln2_post.w",
            ])
        else:
            expected_params.append(f"blocks.{l}.ln2.w")

        if getattr(cfg, "use_qk_norm", False):
            expected_params.extend([
                f"blocks.{l}.attn.q_norm.w",
                f"blocks.{l}.attn.k_norm.w",
            ])

    missing_params = [p for p in expected_params if p not in tl_state_dict]

    # ---- Embedding: reverse sqrt(d_model) scaling ----
    if "embed.W_E" in tl_state_dict:
        emb = tl_state_dict["embed.W_E"]
        gemma_state_dict["model.embed_tokens.weight"] = (emb / math.sqrt(cfg.d_model)).to(dtype=emb.dtype, device=emb.device)

    # ---- Per-layer conversions ----
    for l in range(cfg.n_layers):
        # LN1: subtract the +1 baked in during forward convert
        ln1_key = f"blocks.{l}.ln1.w"
        if ln1_key in tl_state_dict:
            src = tl_state_dict[ln1_key]
            gemma_state_dict[f"model.layers.{l}.input_layernorm.weight"] = (src - torch.ones_like(src)).to(dtype=src.dtype, device=src.device)

        if cfg.use_normalization_before_and_after:
            # Gemma 2: ln1_post -> post_attention_layernorm
            ln1_post_key = f"blocks.{l}.ln1_post.w"
            if ln1_post_key in tl_state_dict:
                src = tl_state_dict[ln1_post_key]
                gemma_state_dict[f"model.layers.{l}.post_attention_layernorm.weight"] = (src - torch.ones_like(src)).to(dtype=src.dtype, device=src.device)

            # Gemma 2: ln2 -> pre_feedforward, ln2_post -> post_feedforward
            ln2_key = f"blocks.{l}.ln2.w"
            if ln2_key in tl_state_dict:
                src = tl_state_dict[ln2_key]
                gemma_state_dict[f"model.layers.{l}.pre_feedforward_layernorm.weight"] = (src - torch.ones_like(src)).to(dtype=src.dtype, device=src.device)

            ln2_post_key = f"blocks.{l}.ln2_post.w"
            if ln2_post_key in tl_state_dict:
                src = tl_state_dict[ln2_post_key]
                gemma_state_dict[f"model.layers.{l}.post_feedforward_layernorm.weight"] = (src - torch.ones_like(src)).to(dtype=src.dtype, device=src.device)
        else:
            # Gemma 1: ln2 -> post_attention_layernorm
            ln2_key = f"blocks.{l}.ln2.w"
            if ln2_key in tl_state_dict:
                src = tl_state_dict[ln2_key]
                gemma_state_dict[f"model.layers.{l}.post_attention_layernorm.weight"] = (src - torch.ones_like(src)).to(dtype=src.dtype, device=src.device)

        # Q/K/V projections: rearrange back to (n*h, m)
        wq_key = f"blocks.{l}.attn.W_Q"
        if wq_key in tl_state_dict:
            src = tl_state_dict[wq_key]
            gemma_state_dict[f"model.layers.{l}.self_attn.q_proj.weight"] = einops.rearrange(src, "n m h -> (n h) m", n=cfg.n_heads).to(dtype=src.dtype, device=src.device)

        wk_key = f"blocks.{l}.attn._W_K"
        if wk_key in tl_state_dict:
            src = tl_state_dict[wk_key]
            gemma_state_dict[f"model.layers.{l}.self_attn.k_proj.weight"] = einops.rearrange(src, "n m h -> (n h) m", n=cfg.n_key_value_heads).to(dtype=src.dtype, device=src.device)

        wv_key = f"blocks.{l}.attn._W_V"
        if wv_key in tl_state_dict:
            src = tl_state_dict[wv_key]
            gemma_state_dict[f"model.layers.{l}.self_attn.v_proj.weight"] = einops.rearrange(src, "n m h -> (n h) m", n=cfg.n_key_value_heads).to(dtype=src.dtype, device=src.device)

        wo_key = f"blocks.{l}.attn.W_O"
        if wo_key in tl_state_dict:
            src = tl_state_dict[wo_key]
            gemma_state_dict[f"model.layers.{l}.self_attn.o_proj.weight"] = einops.rearrange(src, "n h m -> m (n h)", n=cfg.n_heads).to(dtype=src.dtype, device=src.device)

        # QK norms (Gemma 3): subtract the +1 baked in during forward convert
        if getattr(cfg, "use_qk_norm", False):
            qn_key = f"blocks.{l}.attn.q_norm.w"
            if qn_key in tl_state_dict:
                src = tl_state_dict[qn_key]
                gemma_state_dict[f"model.layers.{l}.self_attn.q_norm.weight"] = (src - torch.ones_like(src)).to(dtype=src.dtype, device=src.device)

            kn_key = f"blocks.{l}.attn.k_norm.w"
            if kn_key in tl_state_dict:
                src = tl_state_dict[kn_key]
                gemma_state_dict[f"model.layers.{l}.self_attn.k_norm.weight"] = (src - torch.ones_like(src)).to(dtype=src.dtype, device=src.device)

        # MLP: transpose back (forward convert stored these transposed)
        win_key = f"blocks.{l}.mlp.W_in"
        if win_key in tl_state_dict:
            src = tl_state_dict[win_key]
            gemma_state_dict[f"model.layers.{l}.mlp.up_proj.weight"] = src.T.to(dtype=src.dtype, device=src.device)

        wgate_key = f"blocks.{l}.mlp.W_gate"
        if wgate_key in tl_state_dict:
            src = tl_state_dict[wgate_key]
            gemma_state_dict[f"model.layers.{l}.mlp.gate_proj.weight"] = src.T.to(dtype=src.dtype, device=src.device)

        wout_key = f"blocks.{l}.mlp.W_out"
        if wout_key in tl_state_dict:
            src = tl_state_dict[wout_key]
            gemma_state_dict[f"model.layers.{l}.mlp.down_proj.weight"] = src.T.to(dtype=src.dtype, device=src.device)

        # Biases were zeros in TL; propagate if present (HF Gemma has no biases, but be safe)
        bin_key = f"blocks.{l}.mlp.b_in"
        if bin_key in tl_state_dict:
            src = tl_state_dict[bin_key]
            gemma_state_dict[f"model.layers.{l}.mlp.up_proj.bias"] = src.to(dtype=src.dtype, device=src.device)

        bout_key = f"blocks.{l}.mlp.b_out"
        if bout_key in tl_state_dict:
            src = tl_state_dict[bout_key]
            gemma_state_dict[f"model.layers.{l}.mlp.down_proj.bias"] = src.to(dtype=src.dtype, device=src.device)

    # ---- Final norm: reverse +1 ----
    if "ln_final.w" in tl_state_dict:
        src = tl_state_dict["ln_final.w"]
        gemma_state_dict["model.norm.weight"] = (src - 0 * torch.ones_like(src)).to(dtype=src.dtype, device=src.device)

    # ---- Unembed / lm_head ----
    if "unembed.W_U" in tl_state_dict:
        src = tl_state_dict["unembed.W_U"]
        gemma_state_dict["lm_head.weight"] = src.T.to(dtype=src.dtype, device=src.device)

    if "unembed.b_U" in tl_state_dict:
        src = tl_state_dict["unembed.b_U"]
        gemma_state_dict["lm_head.bias"] = src.to(dtype=src.dtype, device=src.device)

    # ---- Verbose reporting ----
    unexpected_params = [k for k in tl_state_dict.keys() if k not in expected_params]
    if verbose:
        if unexpected_params:
            print("Unexpected parameters in HookedTransformer state_dict (will be ignored):")
            for p in unexpected_params:
                print(f"  - {p}")
        if missing_params:
            print("Expected parameters not found in HookedTransformer state_dict:")
            for p in missing_params:
                print(f"  - {p}")

    return gemma_state_dict, missing_params


def convert_gemma_to_hooked_weights(gemma, cfg: HookedTransformerConfig):
    state_dict = {}

    assert cfg.n_key_value_heads is not None  # keep mypy happy
    assert cfg.d_mlp is not None  # keep mypy happy

    # Check if this is a multimodal model (Gemma3ForConditionalGeneration)
    # Multimodal models have language_model attribute, text-only models don't
    is_multimodal = hasattr(gemma, "language_model")

    # Get the actual model
    # For multimodal: gemma.language_model.model is Gemma3TextModel which has layers/embed_tokens
    # For text-only: gemma has .model which contains layers/embed_tokens
    if is_multimodal:
        # Multimodal structure: gemma.language_model.model contains the text transformer
        # We skip gemma.vision_tower entirely to save memory
        if hasattr(gemma.language_model, "model"):
            base_model = gemma.language_model.model
        else:
            # Fallback if structure is different
            base_model = gemma.language_model
    else:
        # Text-only Gemma3ForCausalLM has .model wrapper
        base_model = gemma.model

    # Gemma Models scale embeddings by multiplying by sqrt(d_model), use hidden state type to match
    # HF implementation
    state_dict["embed.W_E"] = base_model.embed_tokens.weight * torch.tensor(
        cfg.d_model**0.5, dtype=cfg.dtype
    )

    # Gemma has no biases anywhere
    for l in range(cfg.n_layers):
        # GemmaRMSNorm adds 1 to weights before multiplying by input, keep RMS calcs in float32
        state_dict[f"blocks.{l}.ln1.w"] = base_model.layers[
            l
        ].input_layernorm.weight.float() + torch.ones_like(
            base_model.layers[l].input_layernorm.weight, dtype=torch.float32
        )
        if cfg.use_normalization_before_and_after:
            # Only applies for Gemma 2
            state_dict[f"blocks.{l}.ln1_post.w"] = base_model.layers[
                l
            ].post_attention_layernorm.weight.float() + torch.ones_like(
                base_model.layers[l].input_layernorm.weight, dtype=torch.float32
            )

        W_Q = base_model.layers[l].self_attn.q_proj.weight
        W_K = base_model.layers[l].self_attn.k_proj.weight
        W_V = base_model.layers[l].self_attn.v_proj.weight
        W_Q = einops.rearrange(W_Q, "(n h) m->n m h", n=cfg.n_heads)
        W_K = einops.rearrange(W_K, "(n h) m->n m h", n=cfg.n_key_value_heads)
        W_V = einops.rearrange(W_V, "(n h) m->n m h", n=cfg.n_key_value_heads)
        state_dict[f"blocks.{l}.attn.W_Q"] = W_Q
        state_dict[f"blocks.{l}.attn._W_K"] = W_K
        state_dict[f"blocks.{l}.attn._W_V"] = W_V

        # Load q_norm and k_norm if they exist (Gemma 3)
        # Gemma3RMSNorm adds 1 to weights in forward(), so we pre-add it here
        if cfg.use_qk_norm:
            state_dict[f"blocks.{l}.attn.q_norm.w"] = base_model.layers[
                l
            ].self_attn.q_norm.weight.float() + torch.ones_like(
                base_model.layers[l].self_attn.q_norm.weight, dtype=torch.float32
            )
            state_dict[f"blocks.{l}.attn.k_norm.w"] = base_model.layers[
                l
            ].self_attn.k_norm.weight.float() + torch.ones_like(
                base_model.layers[l].self_attn.k_norm.weight, dtype=torch.float32
            )

        state_dict[f"blocks.{l}.attn.b_Q"] = torch.zeros(
            cfg.n_heads, cfg.d_head, dtype=cfg.dtype, device=W_Q.device
        )
        state_dict[f"blocks.{l}.attn._b_K"] = torch.zeros(
            cfg.n_key_value_heads, cfg.d_head, dtype=cfg.dtype, device=W_K.device
        )
        state_dict[f"blocks.{l}.attn._b_V"] = torch.zeros(
            cfg.n_key_value_heads, cfg.d_head, dtype=cfg.dtype, device=W_V.device
        )

        W_O = base_model.layers[l].self_attn.o_proj.weight
        W_O = einops.rearrange(W_O, "m (n h)->n h m", n=cfg.n_heads)
        state_dict[f"blocks.{l}.attn.W_O"] = W_O

        state_dict[f"blocks.{l}.attn.b_O"] = torch.zeros(
            cfg.d_model, dtype=cfg.dtype, device=W_O.device
        )

        # GemmaRMSNorm adds 1 to weights before multiplying by input, keep RMS calcs in float32
        if not cfg.use_normalization_before_and_after:
            # Only applies for Gemma 1. Confusingly post_attention_layernorm is applied to mlp_input in Gemma 1 and attn_out in Gemma 2
            state_dict[f"blocks.{l}.ln2.w"] = base_model.layers[
                l
            ].post_attention_layernorm.weight.float() + torch.ones_like(
                base_model.norm.weight, dtype=torch.float32
            )
        else:
            # Only applies for Gemma 2
            state_dict[f"blocks.{l}.ln2.w"] = base_model.layers[
                l
            ].pre_feedforward_layernorm.weight.float() + torch.ones_like(
                base_model.layers[l].pre_feedforward_layernorm.weight, dtype=torch.float32
            )
            state_dict[f"blocks.{l}.ln2_post.w"] = base_model.layers[
                l
            ].post_feedforward_layernorm.weight.float() + torch.ones_like(
                base_model.layers[l].post_feedforward_layernorm.weight, dtype=torch.float32
            )

        state_dict[f"blocks.{l}.mlp.W_in"] = base_model.layers[l].mlp.up_proj.weight.T
        state_dict[f"blocks.{l}.mlp.W_gate"] = base_model.layers[l].mlp.gate_proj.weight.T
        state_dict[f"blocks.{l}.mlp.b_in"] = torch.zeros(
            cfg.d_mlp, dtype=cfg.dtype, device=base_model.layers[l].mlp.up_proj.weight.device
        )

        state_dict[f"blocks.{l}.mlp.W_out"] = base_model.layers[l].mlp.down_proj.weight.T
        state_dict[f"blocks.{l}.mlp.b_out"] = torch.zeros(
            cfg.d_model, dtype=cfg.dtype, device=base_model.layers[l].mlp.down_proj.weight.device
        )

    # GemmaRMSNorm adds 1 to weights before multiplying by input, keep RMS calcs in float32
    state_dict["ln_final.w"] = base_model.norm.weight.float() + torch.ones_like(
        base_model.norm.weight, dtype=torch.float32
    )

    # For multimodal models, lm_head might not exist or be tied to embeddings
    if hasattr(gemma, "lm_head"):
        state_dict["unembed.W_U"] = gemma.lm_head.weight.T
        unembed_device = gemma.lm_head.weight.device
    else:
        # Multimodal models might use tied embeddings
        state_dict["unembed.W_U"] = base_model.embed_tokens.weight.T
        unembed_device = base_model.embed_tokens.weight.device
    state_dict["unembed.b_U"] = torch.zeros(cfg.d_vocab, dtype=cfg.dtype, device=unembed_device)

    return state_dict