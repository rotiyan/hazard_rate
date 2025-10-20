"""
Unit tests for loss functions
"""

import unittest
import torch
import numpy as np

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from safe_fraud_detection.models.loss import SAFELoss, RegularSurvivalLoss, WeightedSAFELoss


class TestSAFELoss(unittest.TestCase):
    """Test cases for SAFE loss function."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.loss_fn = SAFELoss(epsilon=1e-7)
        self.batch_size = 8
        self.seq_len = 10
        
        # Create sample data
        self.hazard_rates = torch.rand(self.batch_size, self.seq_len) * 0.1
        self.event_indicator = torch.randint(0, 2, (self.batch_size,)).float()
        self.time_observed = torch.randint(1, self.seq_len + 1, (self.batch_size,))
    
    def test_loss_computation(self):
        """Test loss computation works."""
        loss = self.loss_fn(
            self.hazard_rates,
            self.event_indicator,
            self.time_observed
        )
        
        # Check loss is a scalar
        self.assertEqual(loss.shape, torch.Size([]))
        
        # Check loss is positive
        self.assertGreater(loss.item(), 0)
    
    def test_loss_is_differentiable(self):
        """Test that loss can be backpropagated."""
        hazard_rates = self.hazard_rates.clone().requires_grad_(True)
        
        loss = self.loss_fn(
            hazard_rates,
            self.event_indicator,
            self.time_observed
        )
        
        loss.backward()
        
        # Check gradients exist
        self.assertIsNotNone(hazard_rates.grad)
        self.assertFalse(torch.all(hazard_rates.grad == 0))
    
    def test_event_vs_censored_loss(self):
        """Test loss behaves differently for events vs censored."""
        # All events
        all_events = torch.ones(self.batch_size)
        loss_events = self.loss_fn(
            self.hazard_rates,
            all_events,
            self.time_observed
        )
        
        # All censored
        all_censored = torch.zeros(self.batch_size)
        loss_censored = self.loss_fn(
            self.hazard_rates,
            all_censored,
            self.time_observed
        )
        
        # Losses should be different
        self.assertNotEqual(loss_events.item(), loss_censored.item())
    
    def test_numerical_stability(self):
        """Test loss handles extreme values without NaN/Inf."""
        # Test with very small hazard rates
        small_hazards = torch.ones(self.batch_size, self.seq_len) * 1e-8
        loss = self.loss_fn(small_hazards, self.event_indicator, self.time_observed)
        self.assertFalse(torch.isnan(loss))
        self.assertFalse(torch.isinf(loss))
        
        # Test with larger hazard rates
        large_hazards = torch.ones(self.batch_size, self.seq_len) * 2.0
        loss = self.loss_fn(large_hazards, self.event_indicator, self.time_observed)
        self.assertFalse(torch.isnan(loss))
        self.assertFalse(torch.isinf(loss))
    
    def test_time_observed_effect(self):
        """Test that time observed affects the loss correctly."""
        # Early time
        early_time = torch.ones(self.batch_size, dtype=torch.long) * 2
        loss_early = self.loss_fn(
            self.hazard_rates,
            self.event_indicator,
            early_time
        )
        
        # Late time
        late_time = torch.ones(self.batch_size, dtype=torch.long) * 8
        loss_late = self.loss_fn(
            self.hazard_rates,
            self.event_indicator,
            late_time
        )
        
        # Losses should be different
        self.assertNotEqual(loss_early.item(), loss_late.item())


class TestRegularSurvivalLoss(unittest.TestCase):
    """Test cases for regular survival loss."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.loss_fn = RegularSurvivalLoss(epsilon=1e-7)
        self.batch_size = 8
        self.seq_len = 10
        
        self.hazard_rates = torch.rand(self.batch_size, self.seq_len) * 0.1
        self.event_indicator = torch.randint(0, 2, (self.batch_size,)).float()
        self.time_observed = torch.randint(1, self.seq_len + 1, (self.batch_size,))
    
    def test_regular_loss_computation(self):
        """Test regular survival loss works."""
        loss = self.loss_fn(
            self.hazard_rates,
            self.event_indicator,
            self.time_observed
        )
        
        self.assertEqual(loss.shape, torch.Size([]))
        self.assertGreater(loss.item(), 0)
    
    def test_regular_vs_safe_loss(self):
        """Test that regular loss differs from SAFE loss."""
        safe_loss = SAFELoss()
        
        loss_safe = safe_loss(
            self.hazard_rates,
            self.event_indicator,
            self.time_observed
        )
        
        loss_regular = self.loss_fn(
            self.hazard_rates,
            self.event_indicator,
            self.time_observed
        )
        
        # They should produce different values
        self.assertNotEqual(loss_safe.item(), loss_regular.item())


class TestWeightedSAFELoss(unittest.TestCase):
    """Test cases for weighted SAFE loss."""
    
    def test_weighted_loss(self):
        """Test weighted loss with different weights."""
        loss_fn = WeightedSAFELoss(
            event_weight=2.0,
            censored_weight=1.0
        )
        
        hazard_rates = torch.rand(8, 10) * 0.1
        event_indicator = torch.randint(0, 2, (8,)).float()
        time_observed = torch.randint(1, 11, (8,))
        
        loss = loss_fn(hazard_rates, event_indicator, time_observed)
        
        self.assertGreater(loss.item(), 0)
        self.assertFalse(torch.isnan(loss))
    
    def test_weight_effect(self):
        """Test that weights affect loss magnitude."""
        hazard_rates = torch.rand(8, 10) * 0.1
        event_indicator = torch.ones(8)  # All events
        time_observed = torch.randint(1, 11, (8,))
        
        # Low weight
        loss_fn_low = WeightedSAFELoss(event_weight=0.5, censored_weight=1.0)
        loss_low = loss_fn_low(hazard_rates, event_indicator, time_observed)
        
        # High weight
        loss_fn_high = WeightedSAFELoss(event_weight=2.0, censored_weight=1.0)
        loss_high = loss_fn_high(hazard_rates, event_indicator, time_observed)
        
        # Higher weight should give higher loss
        self.assertGreater(loss_high.item(), loss_low.item())


if __name__ == '__main__':
    unittest.main()
