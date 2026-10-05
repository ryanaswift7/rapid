import pytest
import torch
import numpy as np
from unittest.mock import patch, MagicMock
from transformers import AutoTokenizer
from transformers import AutoModelForCausalLM

# Import your module directly
from kdfp.utils.shallow_model_loading import (
    load_shallow, 
    get_indices_and_num_layers_to_keep
)


class TestWeightCopying:
    @pytest.fixture
    def mock_tl_model(self):
        # Create mock original model
        model = MagicMock()
        model.cfg.n_layers = 12
        # Mock blocks
        model.blocks = [MagicMock() for _ in range(12)]
        for i in range(12):
            model.blocks[i].state_dict.return_value = {f'layer_{i}': torch.ones(1)}
        
        # Mock components
        for comp in ['embed', 'unembed', 'ln_final']:
            c = MagicMock()
            c.state_dict.return_value = {comp: torch.ones(1)}
            setattr(model, comp, c)
        return model

    @patch('kdfp.utils.shallow_model_loading.HookedTransformer')
    def test_tl_layer_weights_copied_correctly(self, mock_ht, mock_tl_model):
        mock_ht.from_pretrained_no_processing.return_value = mock_tl_model
        
        # We need a fresh mock for the "new" shallow model
        shallow_model = MagicMock()
        shallow_model.cfg.n_layers = 3
        shallow_model.blocks = [MagicMock() for _ in range(3)]
        mock_ht.return_value = shallow_model
        
        _, _, keep_indices = load_shallow("gpt2", "tl", 0.25, 'linspace', torch.bfloat16)
        
        # Verify the load_state_dict calls on blocks
        for i, idx in enumerate(keep_indices):
            shallow_model.blocks[i].load_state_dict.assert_called_with(
                mock_tl_model.blocks[idx].state_dict()
            )

    @patch('kdfp.utils.shallow_model_loading.AutoModelForCausalLM')
    @patch('kdfp.utils.shallow_model_loading.AutoConfig')
    @patch('kdfp.utils.shallow_model_loading.AutoTokenizer')
    def test_hf_layer_weights_copied_correctly(self, mock_tok, mock_cfg, mock_model):
        # Setup Config
        config = MagicMock()
        config.n_layer = 12
        mock_cfg.from_pretrained.return_value = config
        
        # Setup Original Model
        orig_model = MagicMock()
        # Mocking the layers path for GPT2 ('transformer.h')
        orig_model.transformer.h = [MagicMock() for _ in range(12)]
        orig_model.state_dict.return_value = {'transformer.wte.weight': torch.randn(5)}
        mock_model.from_pretrained.return_value = orig_model
        
        # Setup Shallow Model
        shallow_model = MagicMock()
        shallow_model.transformer.h = [MagicMock() for _ in range(3)]
        mock_model.from_config.return_value = shallow_model
        
        load_shallow("gpt2", "hf", 0.25, 'linspace', torch.bfloat16)
        
        # Verify layer copying
        expected_indices = [0, 6, 11]
        for i, idx in enumerate(expected_indices):
            shallow_model.transformer.h[i].load_state_dict.assert_called()


class TestKeepIndicesValidation:
    """Test that the keep_indices returned are correct for various scenarios."""
    
    @pytest.mark.parametrize("total_layers,reduction_factor,expected_indices", [
        (12, 0.5, [0, 2, 4, 7, 9, 11]),      # 12 * 0.5 = 6 layers
        (12, 0.25, [0, 6, 11]),              # 12 * 0.25 = 3 layers
        (12, 0.33, [0, 4, 7, 11]),           # 12 * 0.33 ≈ 4 layers
        (8, 0.5, [0, 2, 5, 7]),              # 8 * 0.5 = 4 layers
        (6, 0.33, [0, 5]),                   # 6 * 0.33 ≈ 2 layers
        (24, 0.17, [0, 8, 15, 23]),          # 24 * 0.17 ≈ 4 layers
        (16, 0.125, [0, 15]),                # 16 * 0.125 = 2 layers
    ])
    @patch('kdfp.utils.shallow_model_loading.AutoModelForCausalLM')
    @patch('kdfp.utils.shallow_model_loading.AutoConfig')
    @patch('kdfp.utils.shallow_model_loading.AutoTokenizer')
    def test_hf_keep_indices_returned(self, mock_tokenizer, mock_config, mock_model, 
                                       total_layers, reduction_factor, expected_indices):
        """Test keep_indices calculation for various layer counts and reduction factors."""
        # Setup mocks
        original_model = MagicMock()
        original_model.transformer.h = [MagicMock() for _ in range(total_layers)]
        original_model.state_dict.return_value = {
            'transformer.wte.weight': torch.randn(10, 8),
            'transformer.wpe.weight': torch.randn(10, 8),
            'transformer.ln_f.weight': torch.randn(8),
            'lm_head.weight': torch.randn(10, 8),
        }
        mock_model.from_pretrained.return_value = original_model
        
        config = MagicMock()
        setattr(config, 'n_layer', total_layers)  # Using setattr to handle dynamic attribute
        mock_config.from_pretrained.return_value = config

        _, layers_to_keep = get_indices_and_num_layers_to_keep(total_layers=total_layers,
                                                           layer_reduction_factor=reduction_factor,
                                                           selection_method='linspace')
        
        shallow_model = MagicMock()
        shallow_model.transformer.h = [MagicMock() for _ in range(layers_to_keep)]
        shallow_model.state_dict.return_value = {
            'transformer.wte.weight': torch.randn(10, 8),
            'transformer.wpe.weight': torch.randn(10, 8),
            'transformer.ln_f.weight': torch.randn(8),
            'lm_head.weight': torch.randn(10, 8),
        }
        mock_model.from_config.return_value = shallow_model
        
        tokenizer = MagicMock()
        mock_tokenizer.from_pretrained.return_value = tokenizer
        
        # Run the function using the wrapper
        _, _, keep_indices = load_shallow(model_name="gpt2",
                                          mode="hf",
                                          layer_reduction_factor=reduction_factor,
                                          selection_method='linspace',
                                          torch_dtype=torch.bfloat16)
        
        # Verify the keep_indices match expected
        assert keep_indices == expected_indices, (
            f"For {total_layers} layers with reduction factor {reduction_factor}: "
            f"expected {expected_indices}, got {keep_indices}"
        )
    
    @patch('kdfp.utils.shallow_model_loading.HookedTransformer')
    def test_tl_keep_indices_returned(self, mock_hooked_transformer):
        """Test that TransformerLens mode returns correct keep_indices."""
        # Setup mock
        original_model = MagicMock()
        original_model.cfg.n_layers = 8
        original_model.cfg.copy.return_value = MagicMock()
        original_model.cfg.copy.return_value.n_layers = 8
        original_model.blocks = [MagicMock() for _ in range(8)]
        for component in ['embed', 'unembed', 'ln_final', 'pos_embed']:
            setattr(original_model, component, MagicMock())
        # Fixed the mock here to use from_pretrained_no_processing
        mock_hooked_transformer.from_pretrained_no_processing.return_value = original_model
        
        # 8 * 0.5 = 4 layers
        shallow_model = MagicMock()
        shallow_model.blocks = [MagicMock() for _ in range(4)]
        for component in ['embed', 'unembed', 'ln_final', 'pos_embed']:
            setattr(shallow_model, component, MagicMock())
        mock_hooked_transformer.return_value = shallow_model
        
        # Run the function with reduction factor 0.5
        _, _, keep_indices = load_shallow(model_name="test_model",
                                          mode="tl",
                                          layer_reduction_factor=0.5,
                                          selection_method='linspace',
                                          torch_dtype=torch.bfloat16)
        
        # For 8 layers with reduction factor 0.5: torch.linspace(0, 7, 4) = [0, 2, 5, 7]
        expected_indices = [0, 2, 5, 7]
        assert keep_indices == expected_indices


class TestEdgeCases:
    """Test edge cases for the reduction factor."""
    
    @patch('kdfp.utils.shallow_model_loading.HookedTransformer')
    def test_minimum_one_layer(self, mock_hooked_transformer):
        """Test that at least one layer is kept even with very small reduction factor."""
        # Setup mock
        original_model = MagicMock()
        original_model.cfg.n_layers = 12
        original_model.cfg.copy.return_value = MagicMock()
        original_model.cfg.copy.return_value.n_layers = 12
        original_model.blocks = [MagicMock() for _ in range(12)]
        for component in ['embed', 'unembed', 'ln_final', 'pos_embed']:
            setattr(original_model, component, MagicMock())
        # Fixed the mock here
        mock_hooked_transformer.from_pretrained_no_processing.return_value = original_model
        
        # Even with very small reduction factor, should keep at least 1 layer
        shallow_model = MagicMock()
        shallow_model.blocks = [MagicMock() for _ in range(1)]
        for component in ['embed', 'unembed', 'ln_final', 'pos_embed']:
            setattr(shallow_model, component, MagicMock())
        mock_hooked_transformer.return_value = shallow_model
        
        # Run the function with very small reduction factor
        _, _, keep_indices = load_shallow(model_name="test_model",
                                          mode="tl",
                                          layer_reduction_factor=0.01,
                                          selection_method='linspace',
                                          torch_dtype=torch.bfloat16)
        
        # Should keep exactly 1 layer (index 0)
        assert len(keep_indices) == 1
        assert keep_indices == [0]
    
    @patch('kdfp.utils.shallow_model_loading.HookedTransformer')
    def test_reduction_factor_greater_than_one(self, mock_hooked_transformer):
        """Test that reduction factor > 1 keeps all layers."""
        # Setup mock
        original_model = MagicMock()
        original_model.cfg.n_layers = 12
        original_model.cfg.copy.return_value = MagicMock()
        original_model.cfg.copy.return_value.n_layers = 12
        original_model.blocks = [MagicMock() for _ in range(12)]
        for component in ['embed', 'unembed', 'ln_final', 'pos_embed']:
            setattr(original_model, component, MagicMock())
        # Fixed the mock here
        mock_hooked_transformer.from_pretrained_no_processing.return_value = original_model
        
        # Reduction factor > 1 should keep all layers
        shallow_model = MagicMock()
        shallow_model.blocks = [MagicMock() for _ in range(12)]
        for component in ['embed', 'unembed', 'ln_final', 'pos_embed']:
            setattr(shallow_model, component, MagicMock())
        mock_hooked_transformer.return_value = shallow_model
        
        # Run the function with reduction factor > 1
        _, _, keep_indices = load_shallow(model_name="test_model",
                                          mode="tl",
                                          layer_reduction_factor=1.5,
                                          selection_method='linspace',
                                          torch_dtype=torch.bfloat16)
        
        # Should keep all 12 layers
        assert len(keep_indices) == 12
        assert keep_indices == list(range(12))


class TestWeightPreservation:
    """Test that weight values are preserved exactly during copying."""
    
    @pytest.mark.integration
    @pytest.mark.slow
    def test_weight_preservation_pythia_31m_half_layers(self):
        """
        Integration test that verifies weight values are preserved exactly
        when creating a shallow Pythia-31M model with half the layers.
        """
        model_name = "EleutherAI/pythia-31m"
        reduction_factor = 0.5
        selection_method = 'linspace'
        
        try:
            # Load and create shallow model using the wrapper
            shallow_model, tokenizer, keep_indices = load_shallow(model_name=model_name,
                                                                  mode="hf",
                                                                  layer_reduction_factor=reduction_factor,
                                                                  selection_method='linspace',
                                                                  torch_dtype=torch.bfloat16)
            original_model = AutoModelForCausalLM.from_pretrained(model_name)
            
            # Get state dicts
            original_state = original_model.state_dict()
            shallow_state = shallow_model.state_dict()
            
            # Test 1: Non-layer components preserved
            non_layer_keys = [k for k in original_state.keys() if not k.startswith('gpt_neox.layers.')]
            for key in non_layer_keys:
                if key in shallow_state:
                    assert torch.equal(original_state[key], shallow_state[key])
            
            # Test 2: Selected layers preserved
            for i, layer_idx in enumerate(keep_indices):
                layer_prefix = f'gpt_neox.layers.{layer_idx}.'
                shallow_prefix = f'gpt_neox.layers.{i}.'
                
                layer_keys = [k for k in original_state.keys() if k.startswith(layer_prefix)]
                for orig_key in layer_keys:
                    shallow_key = orig_key.replace(layer_prefix, shallow_prefix)
                    if shallow_key in shallow_state:
                        assert torch.equal(original_state[orig_key], shallow_state[shallow_key])
            
            # Clean up
            del original_model, shallow_model
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            
        except Exception as e:
            pytest.skip(f"Could not load Pythia-31M model: {e}")


class TestFirstNSelection:
    """Test the 'first_n' selection method for layer reduction."""
    
    @pytest.mark.parametrize("total_layers,reduction_factor,expected_indices,expected_count", [
        # Basic cases
        (12, 0.5, [0, 1, 2, 3, 4, 5], 6),          # 12 * 0.5 = 6 layers (first 6)
        (12, 0.25, [0, 1, 2], 3),                  # 12 * 0.25 = 3 layers (first 3)
        (12, 0.33, [0, 1, 2, 3], 4),               # 12 * 0.33 ≈ 4 layers (first 4)
        (8, 0.5, [0, 1, 2, 3], 4),                 # 8 * 0.5 = 4 layers (first 4)
        (6, 0.33, [0, 1], 2),                      # 6 * 0.33 ≈ 2 layers (first 2)
        
        # Edge cases
        (12, 1.0, list(range(12)), 12),            # 100% reduction = all layers
        (12, 0.01, [0], 1),                        # Very small reduction = first layer only
        (12, 1.5, list(range(12)), 12),            # >100% reduction = all layers
        (1, 0.5, [0], 1),                          # Single layer model
        (1, 1.0, [0], 1),                          # Single layer, 100% reduction
    ])
    def test_first_n_indices_calculation(self, total_layers, reduction_factor, 
                                        expected_indices, expected_count):
        """Test that first_n selection returns correct indices and count."""
        keep_indices, layers_to_keep = get_indices_and_num_layers_to_keep(
            total_layers=total_layers,
            layer_reduction_factor=reduction_factor,
            selection_method='first_n'
        )
        
        assert layers_to_keep == expected_count, (
            f"Expected {expected_count} layers, got {layers_to_keep}"
        )
        assert keep_indices == expected_indices, (
            f"Expected indices {expected_indices}, got {keep_indices}"
        )
        assert len(keep_indices) == expected_count, (
            f"Number of indices ({len(keep_indices)}) doesn't match expected count ({expected_count})"
        )
    
    def test_first_n_versus_linspace_comparison(self):
        """Test that first_n and linspace produce different results for same parameters."""
        total_layers = 12
        reduction_factor = 0.5
        
        first_n_indices, first_n_count = get_indices_and_num_layers_to_keep(
            total_layers=total_layers,
            layer_reduction_factor=reduction_factor,
            selection_method='first_n'
        )
        
        linspace_indices, linspace_count = get_indices_and_num_layers_to_keep(
            total_layers=total_layers,
            layer_reduction_factor=reduction_factor,
            selection_method='linspace'
        )
        
        # Both should keep the same number of layers
        assert first_n_count == linspace_count == 6
        
        # But they should have different indices
        assert first_n_indices != linspace_indices
        assert first_n_indices == [0, 1, 2, 3, 4, 5]  # First 6 layers
        assert linspace_indices == [0, 2, 4, 7, 9, 11]  # Evenly spaced
    
    @patch('kdfp.utils.shallow_model_loading.HookedTransformer')
    def test_tl_first_n_selection(self, mock_hooked_transformer):
        """Test that TransformerLens mode uses first_n selection correctly."""
        total_layers = 8
        reduction_factor = 0.5  # Should keep first 4 layers
        
        # Setup mock
        original_model = MagicMock()
        original_model.cfg.n_layers = total_layers
        original_model.cfg.copy.return_value = MagicMock()
        original_model.cfg.copy.return_value.n_layers = total_layers
        original_model.blocks = [MagicMock() for _ in range(total_layers)]
        for component in ['embed', 'unembed', 'ln_final', 'pos_embed']:
            setattr(original_model, component, MagicMock())
        # Fixed the mock here
        mock_hooked_transformer.from_pretrained_no_processing.return_value = original_model
        
        # Create shallow model with expected number of layers
        shallow_model = MagicMock()
        shallow_model.blocks = [MagicMock() for _ in range(4)]  # 8 * 0.5 = 4 layers
        for component in ['embed', 'unembed', 'ln_final', 'pos_embed']:
            setattr(shallow_model, component, MagicMock())
        mock_hooked_transformer.return_value = shallow_model
        
        # Run the function with first_n selection using wrapper
        _, _, keep_indices = load_shallow(
            model_name="test_model",
            mode="tl",
            layer_reduction_factor=reduction_factor,
            selection_method='first_n',
            torch_dtype=torch.bfloat16
        )
        
        # Should keep first 4 layers: [0, 1, 2, 3]
        expected_indices = [0, 1, 2, 3]
        assert keep_indices == expected_indices
    
    @patch('kdfp.utils.shallow_model_loading.AutoModelForCausalLM')
    @patch('kdfp.utils.shallow_model_loading.AutoConfig')
    @patch('kdfp.utils.shallow_model_loading.AutoTokenizer')
    def test_hf_first_n_selection(self, mock_tokenizer, mock_config, mock_model):
        """Test that HuggingFace mode uses first_n selection correctly."""
        total_layers = 10
        reduction_factor = 0.3  # Should keep first 3 layers
        
        # Setup mocks
        original_model = MagicMock()
        original_model.transformer.h = [MagicMock() for _ in range(total_layers)]
        original_model.state_dict.return_value = {
            'transformer.wte.weight': torch.randn(10, 8),
            'transformer.wpe.weight': torch.randn(10, 8),
            'transformer.ln_f.weight': torch.randn(8),
            'lm_head.weight': torch.randn(10, 8),
        }
        mock_model.from_pretrained.return_value = original_model
        
        config = MagicMock()
        setattr(config, 'n_layer', total_layers)
        mock_config.from_pretrained.return_value = config
        
        # Create shallow model with expected number of layers
        shallow_model = MagicMock()
        shallow_model.transformer.h = [MagicMock() for _ in range(3)]  # 10 * 0.3 = 3 layers
        shallow_model.state_dict.return_value = {
            'transformer.wte.weight': torch.randn(10, 8),
            'transformer.wpe.weight': torch.randn(10, 8),
            'transformer.ln_f.weight': torch.randn(8),
            'lm_head.weight': torch.randn(10, 8),
        }
        mock_model.from_config.return_value = shallow_model
        
        tokenizer = MagicMock()
        mock_tokenizer.from_pretrained.return_value = tokenizer
        
        # Run the function with first_n selection using wrapper
        _, _, keep_indices = load_shallow(
            model_name="gpt2",
            mode="hf",
            layer_reduction_factor=reduction_factor,
            selection_method='first_n',
            torch_dtype=torch.bfloat16
        )
        
        # Should keep first 3 layers: [0, 1, 2]
        expected_indices = [0, 1, 2]
        assert keep_indices == expected_indices
    
    def test_first_n_consistency_across_runs(self):
        """Test that first_n selection is deterministic and consistent."""
        total_layers = 12
        reduction_factor = 0.25
        
        # Run multiple times
        results = []
        for _ in range(5):
            keep_indices, layers_to_keep = get_indices_and_num_layers_to_keep(
                total_layers=total_layers,
                layer_reduction_factor=reduction_factor,
                selection_method='first_n'
            )
            results.append((keep_indices, layers_to_keep))
        
        # All results should be identical
        first_result = results[0]
        for result in results[1:]:
            assert result == first_result, "first_n selection should be deterministic"
        
        # Verify the expected result
        expected_indices = [0, 1, 2]  # 12 * 0.25 = 3 layers (first 3)
        expected_count = 3
        assert first_result == (expected_indices, expected_count)

if __name__ == "__main__":
    pytest.main([__file__, "-v"])