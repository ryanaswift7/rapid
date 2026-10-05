import pytest
import torch
from transformer_lens import HookedTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer

from kdfp.utils.weight_conversions import convert_hooked_to_hf_weights

TEST_MODELS = [
    "EleutherAI/pythia-14m",
    "meta-llama/Llama-3.2-1B",
    "Qwen/Qwen2-0.5B",
    "Qwen/Qwen3-0.6B",
    # "google/gemma-3-270m",
]

class TestRealModelConversion:
    """Test that converted weights match original HuggingFace models in bfloat16."""
    
    def _compare_state_dicts(self, dict1, dict2, tol=1e-3):
        """Compare two state dictionaries with bfloat16-friendly tolerance."""
        keys1 = set(dict1.keys())
        keys2 = set(dict2.keys())
        common_keys = keys1.intersection(keys2)
        
        all_close = True
        max_diff = 0
        problematic_keys = []
        
        for key in common_keys:
            if dict1[key].shape != dict2[key].shape:
                print(f"Shape mismatch for {key}: {dict1[key].shape} vs {dict2[key].shape}")
                all_close = False
                continue
            
            # Cast to float32 ONLY for calculating the difference cleanly. 
            tensor1 = dict1[key].float()
            tensor2 = dict2[key].float()
            
            diff = torch.max(torch.abs(tensor1 - tensor2)).item()
            if diff > tol:
                print(f"Warning: {key} has max difference {diff:.4e} (tolerance {tol:.1e})")
                all_close = False
                problematic_keys.append((key, diff))
                max_diff = max(max_diff, diff)
        
        if not all_close:
            print(f"Maximum difference found: {max_diff:.4e}")
            print("Top 5 problematic parameters:")
            for key, diff in sorted(problematic_keys, key=lambda x: x[1], reverse=True)[:5]:
                print(f"  {key}: {diff:.4e}")
        
        return all_close

    @pytest.mark.integration
    @pytest.mark.slow
    @pytest.mark.parametrize("model_name", TEST_MODELS)
    def test_conversion_matches_hf(self, model_name: str):
        """Test that converted weights match original HF weights for all architectures."""
        try:
            print(f"\n[{model_name.upper()}] Loading original HF model in bf16...")
            hf_original = AutoModelForCausalLM.from_pretrained(
                model_name, torch_dtype=torch.bfloat16
            )
            hf_original_state = hf_original.state_dict()
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            
            print(f"[{model_name.upper()}] Loading HookedTransformer in bf16...")
            hooked_model = HookedTransformer.from_pretrained_no_processing(
                model_name, dtype=torch.bfloat16
            )
            
            print(f"[{model_name.upper()}] Converting Hooked weights back to HF format...")
            converted_state, missing = convert_hooked_to_hf_weights(hooked_model, hooked_model.cfg)

            expected_missing = [m for m in missing if m != "unembed.b_U"]
            assert len(expected_missing) == 0, f"Missing params in conversion: {expected_missing}"
            
            print(f"[{model_name.upper()}] Creating HF model with converted weights (bf16)...")
            hf_converted = AutoModelForCausalLM.from_pretrained(
                model_name, torch_dtype=torch.bfloat16
            )
            load_info = hf_converted.load_state_dict(converted_state, strict=False)

            hf_original.eval()
            hf_converted.eval()
            
            if load_info.unexpected_keys:
                print(f"Warning - Unexpected keys loaded: {load_info.unexpected_keys}")
                
            hf_converted_state = hf_converted.state_dict()
            
            print(f"[{model_name.upper()}] Comparing weights between original and converted...")
            weights_match = self._compare_state_dicts(hf_original_state, hf_converted_state, tol=1e-3)
            assert weights_match, f"Weights mismatch for {model_name}"
            
            print(f"[{model_name.upper()}] Testing forward pass consistency...")
            test_text = "The quick brown fox jumps over the lazy dog."
            inputs = tokenizer(test_text, return_tensors="pt")
            
            with torch.no_grad():
                original_output = hf_original(**inputs)
                converted_output = hf_converted(**inputs)
            
            original_logits = original_output.logits.float()
            converted_logits = converted_output.logits.float()
            
            max_diff = torch.max(torch.abs(original_logits - converted_logits)).item()
            
            assert torch.allclose(original_logits, converted_logits, atol=1e-2, rtol=1e-2), \
                f"Logits mismatch! Max diff: {max_diff:.6f}"
            
            print(f"✅ {model_name.upper()} conversion successful! Max logit diff: {max_diff:.6f}")
            
        except Exception as e:
            pytest.fail(f"{model_name.upper()} test failed: {e}")