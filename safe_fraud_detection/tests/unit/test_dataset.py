"""
Unit tests for dataset classes
"""

import unittest
import torch
import numpy as np

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from safe_fraud_detection.data.dataset import (
    FraudDataset, 
    SurvivalDataset,
    TwitterDataset,
    WikiDataset
)


class TestFraudDataset(unittest.TestCase):
    """Test cases for FraudDataset."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.num_samples = 20
        self.seq_len = 10
        self.num_features = 5
        
        self.sequences = np.random.randn(self.num_samples, self.seq_len, self.num_features)
        self.events = np.random.randint(0, 2, self.num_samples)
        self.times = np.random.randint(1, self.seq_len + 1, self.num_samples)
        
        self.dataset = FraudDataset(
            self.sequences,
            self.events,
            self.times
        )
    
    def test_dataset_length(self):
        """Test dataset returns correct length."""
        self.assertEqual(len(self.dataset), self.num_samples)
    
    def test_getitem_returns_correct_types(self):
        """Test __getitem__ returns tensors."""
        sequence, event, time = self.dataset[0]
        
        self.assertIsInstance(sequence, torch.Tensor)
        self.assertIsInstance(event, torch.Tensor)
        self.assertIsInstance(time, torch.Tensor)
    
    def test_getitem_shapes(self):
        """Test __getitem__ returns correct shapes."""
        sequence, event, time = self.dataset[0]
        
        self.assertEqual(sequence.shape, (self.seq_len, self.num_features))
        self.assertEqual(event.shape, (1,))
        self.assertEqual(time.shape, (1,))
    
    def test_get_batch(self):
        """Test get_batch method."""
        indices = [0, 1, 2, 3]
        sequences, events, times = self.dataset.get_batch(indices)
        
        self.assertEqual(sequences.shape, (4, self.seq_len, self.num_features))
        self.assertEqual(events.shape, (4,))
        self.assertEqual(times.shape, (4,))
    
    def test_indexing(self):
        """Test that indexing works correctly."""
        for i in range(5):
            sequence, event, time = self.dataset[i]
            
            # Check values match
            np.testing.assert_array_almost_equal(
                sequence.numpy(),
                self.sequences[i]
            )


class TestSurvivalDataset(unittest.TestCase):
    """Test cases for SurvivalDataset."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.features = np.random.randn(20, 10, 5)
        self.events = np.random.randint(0, 2, 20)
        self.times = np.random.randint(1, 11, 20)
        
        self.dataset = SurvivalDataset(
            self.features,
            self.events,
            self.times
        )
    
    def test_properties(self):
        """Test dataset properties."""
        self.assertEqual(self.dataset.num_features, 5)
        self.assertEqual(self.dataset.max_seq_len, 10)
    
    def test_get_statistics(self):
        """Test get_statistics method."""
        stats = self.dataset.get_statistics()
        
        self.assertIn('num_samples', stats)
        self.assertIn('num_features', stats)
        self.assertIn('max_seq_len', stats)
        self.assertIn('num_events', stats)
        self.assertIn('num_censored', stats)
        self.assertIn('event_rate', stats)
        
        self.assertEqual(stats['num_samples'], 20)
        self.assertEqual(stats['num_features'], 5)
        self.assertEqual(stats['max_seq_len'], 10)
    
    def test_getitem(self):
        """Test __getitem__ method."""
        features, event, time = self.dataset[0]
        
        self.assertEqual(features.shape, (10, 5))
        self.assertIsInstance(event, torch.Tensor)
        self.assertIsInstance(time, torch.Tensor)


class TestSpecializedDatasets(unittest.TestCase):
    """Test specialized dataset classes."""
    
    def test_twitter_dataset(self):
        """Test TwitterDataset."""
        sequences = np.random.randn(10, 15, 5)
        events = np.random.randint(0, 2, 10)
        times = np.random.randint(1, 16, 10)
        
        dataset = TwitterDataset(sequences, events, times)
        
        self.assertEqual(len(dataset), 10)
        self.assertIsNotNone(dataset.feature_names)
        self.assertEqual(len(dataset.feature_names), 5)
    
    def test_wiki_dataset(self):
        """Test WikiDataset."""
        sequences = np.random.randn(10, 15, 8)
        events = np.random.randint(0, 2, 10)
        times = np.random.randint(1, 16, 10)
        
        dataset = WikiDataset(sequences, events, times)
        
        self.assertEqual(len(dataset), 10)
        self.assertIsNotNone(dataset.feature_names)
        self.assertEqual(len(dataset.feature_names), 8)


if __name__ == '__main__':
    unittest.main()
