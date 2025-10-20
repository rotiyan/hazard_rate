"""
Unit tests for SAFE model
"""

import unittest
import torch
import numpy as np

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from safe_fraud_detection.models.safe_model import SAFEModel


class TestSAFEModel(unittest.TestCase):
    """Test cases for SAFE model."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.input_dim = 5
        self.hidden_dim = 32
        self.batch_size = 8
        self.seq_len = 10
        
        self.model = SAFEModel(
            input_dim=self.input_dim,
            hidden_dim=self.hidden_dim,
            num_layers=1,
            dropout=0.0
        )
        
        # Create sample input
        self.sample_input = torch.randn(self.batch_size, self.seq_len, self.input_dim)
    
    def test_model_initialization(self):
        """Test model initializes correctly."""
        self.assertIsInstance(self.model, SAFEModel)
        self.assertEqual(self.model.input_dim, self.input_dim)
        self.assertEqual(self.model.hidden_dim, self.hidden_dim)
    
    def test_forward_pass_shape(self):
        """Test forward pass returns correct shapes."""
        hazard_rates, survival_probs, hidden_states = self.model(self.sample_input)
        
        # Check shapes
        self.assertEqual(hazard_rates.shape, (self.batch_size, self.seq_len))
        self.assertEqual(survival_probs.shape, (self.batch_size, self.seq_len))
        self.assertEqual(hidden_states.shape, (1, self.batch_size, self.hidden_dim))
    
    def test_hazard_rates_positive(self):
        """Test that hazard rates are always positive."""
        hazard_rates, _, _ = self.model(self.sample_input)
        
        self.assertTrue(torch.all(hazard_rates > 0))
    
    def test_survival_probability_properties(self):
        """Test survival probabilities have correct properties."""
        _, survival_probs, _ = self.model(self.sample_input)
        
        # Should be between 0 and 1
        self.assertTrue(torch.all(survival_probs >= 0))
        self.assertTrue(torch.all(survival_probs <= 1))
        
        # Should be monotonically decreasing
        for i in range(self.batch_size):
            for t in range(1, self.seq_len):
                self.assertLessEqual(
                    survival_probs[i, t].item(),
                    survival_probs[i, t-1].item()
                )
    
    def test_predict_method(self):
        """Test predict method works correctly."""
        predictions, survival_probs = self.model.predict(
            self.sample_input, 
            threshold=0.5
        )
        
        # Check shapes
        self.assertEqual(predictions.shape, (self.batch_size, self.seq_len))
        self.assertEqual(survival_probs.shape, (self.batch_size, self.seq_len))
        
        # Check predictions are binary
        self.assertTrue(torch.all((predictions == 0) | (predictions == 1)))
    
    def test_early_detection_time(self):
        """Test early detection time calculation."""
        _, survival_probs, _ = self.model(self.sample_input)
        
        detection_times = self.model.get_early_detection_time(
            survival_probs,
            threshold=0.5
        )
        
        # Check shape
        self.assertEqual(detection_times.shape, (self.batch_size,))
        
        # Check values are in valid range
        self.assertTrue(torch.all(detection_times >= 0))
        self.assertTrue(torch.all(detection_times <= self.seq_len))
    
    def test_different_sequence_lengths(self):
        """Test model works with different sequence lengths."""
        for seq_len in [5, 10, 20]:
            input_tensor = torch.randn(self.batch_size, seq_len, self.input_dim)
            hazard_rates, survival_probs, _ = self.model(input_tensor)
            
            self.assertEqual(hazard_rates.shape[1], seq_len)
            self.assertEqual(survival_probs.shape[1], seq_len)
    
    def test_batch_size_one(self):
        """Test model works with batch size of 1."""
        input_tensor = torch.randn(1, self.seq_len, self.input_dim)
        hazard_rates, survival_probs, _ = self.model(input_tensor)
        
        self.assertEqual(hazard_rates.shape, (1, self.seq_len))
        self.assertEqual(survival_probs.shape, (1, self.seq_len))
    
    def test_gradient_flow(self):
        """Test gradients flow through the model."""
        self.model.train()
        
        output_hazards, _, _ = self.model(self.sample_input)
        loss = output_hazards.sum()
        loss.backward()
        
        # Check that gradients exist and are not zero
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.assertIsNotNone(param.grad)
                self.assertFalse(torch.all(param.grad == 0))


class TestSAFEModelMultiLayer(unittest.TestCase):
    """Test multi-layer GRU configuration."""
    
    def test_multi_layer_model(self):
        """Test model with multiple GRU layers."""
        model = SAFEModel(
            input_dim=5,
            hidden_dim=32,
            num_layers=2,
            dropout=0.1
        )
        
        input_tensor = torch.randn(4, 10, 5)
        hazard_rates, survival_probs, hidden_states = model(input_tensor)
        
        # Check hidden state has correct number of layers
        self.assertEqual(hidden_states.shape[0], 2)
        
        # Check output shapes
        self.assertEqual(hazard_rates.shape, (4, 10))
        self.assertEqual(survival_probs.shape, (4, 10))


if __name__ == '__main__':
    unittest.main()
