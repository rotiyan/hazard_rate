"""
Unit tests for preprocessing utilities
"""

import unittest
import numpy as np

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from safe_fraud_detection.data.preprocessing import (
    SequencePreprocessor,
    pad_sequences,
    create_train_val_test_split
)


class TestSequencePreprocessor(unittest.TestCase):
    """Test cases for SequencePreprocessor."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.sequences = np.random.randn(50, 10, 5)
        self.preprocessor = SequencePreprocessor(
            normalize='standard',
            handle_nan='zero'
        )
    
    def test_fit_transform(self):
        """Test fit and transform."""
        transformed = self.preprocessor.fit_transform(self.sequences)
        
        self.assertEqual(transformed.shape, self.sequences.shape)
        self.assertTrue(self.preprocessor.is_fitted)
    
    def test_normalization(self):
        """Test that normalization works."""
        transformed = self.preprocessor.fit_transform(self.sequences)
        
        # Check that mean is approximately 0 and std is approximately 1
        # (per feature across all samples and time)
        for i in range(5):
            feature_values = transformed[:, :, i].flatten()
            self.assertAlmostEqual(np.mean(feature_values), 0.0, places=1)
            self.assertAlmostEqual(np.std(feature_values), 1.0, places=1)
    
    def test_handle_nan_zero(self):
        """Test NaN handling with zero fill."""
        sequences_with_nan = self.sequences.copy()
        sequences_with_nan[0, 5, 2] = np.nan
        
        preprocessor = SequencePreprocessor(
            normalize=None,
            handle_nan='zero'
        )
        
        transformed = preprocessor.fit_transform(sequences_with_nan)
        
        # Check that NaN was replaced with 0
        self.assertFalse(np.isnan(transformed).any())
    
    def test_handle_nan_mean(self):
        """Test NaN handling with mean fill."""
        sequences_with_nan = self.sequences.copy()
        sequences_with_nan[0, 5, 2] = np.nan
        
        preprocessor = SequencePreprocessor(
            normalize=None,
            handle_nan='mean'
        )
        
        transformed = preprocessor.fit_transform(sequences_with_nan)
        
        # Check that NaN was replaced
        self.assertFalse(np.isnan(transformed).any())
    
    def test_transform_without_fit_raises_error(self):
        """Test that transform without fit raises error."""
        new_preprocessor = SequencePreprocessor()
        
        with self.assertRaises(ValueError):
            new_preprocessor.transform(self.sequences)
    
    def test_minmax_normalization(self):
        """Test minmax normalization."""
        preprocessor = SequencePreprocessor(normalize='minmax')
        transformed = preprocessor.fit_transform(self.sequences)
        
        # Values should be between 0 and 1
        self.assertGreaterEqual(transformed.min(), 0.0)
        self.assertLessEqual(transformed.max(), 1.0)
    
    def test_no_normalization(self):
        """Test with normalization disabled."""
        preprocessor = SequencePreprocessor(normalize=None)
        transformed = preprocessor.fit_transform(self.sequences)
        
        # Should be approximately the same (minus NaN handling)
        np.testing.assert_array_almost_equal(transformed, self.sequences)


class TestPadSequences(unittest.TestCase):
    """Test cases for pad_sequences function."""
    
    def test_pad_sequences_basic(self):
        """Test basic padding functionality."""
        sequences = [
            np.random.randn(5, 3),
            np.random.randn(8, 3),
            np.random.randn(6, 3)
        ]
        
        padded = pad_sequences(sequences)
        
        # Should be padded to longest (8)
        self.assertEqual(padded.shape, (3, 8, 3))
    
    def test_pad_sequences_custom_length(self):
        """Test padding with custom max length."""
        sequences = [
            np.random.randn(5, 3),
            np.random.randn(8, 3)
        ]
        
        padded = pad_sequences(sequences, max_len=10)
        
        self.assertEqual(padded.shape, (2, 10, 3))
    
    def test_pad_sequences_truncation(self):
        """Test that sequences longer than max_len are truncated."""
        sequences = [
            np.random.randn(15, 3),
            np.random.randn(12, 3)
        ]
        
        padded = pad_sequences(sequences, max_len=10)
        
        # Should be truncated to 10
        self.assertEqual(padded.shape, (2, 10, 3))


class TestTrainValTestSplit(unittest.TestCase):
    """Test cases for data splitting."""
    
    def test_split_ratios(self):
        """Test that split ratios are correct."""
        sequences = np.random.randn(100, 10, 5)
        events = np.random.randint(0, 2, 100)
        times = np.random.randint(1, 11, 100)
        
        train_data, val_data, test_data = create_train_val_test_split(
            sequences, events, times,
            train_ratio=0.7,
            val_ratio=0.1,
            test_ratio=0.2
        )
        
        # Check sizes
        self.assertEqual(len(train_data[0]), 70)
        self.assertEqual(len(val_data[0]), 10)
        self.assertEqual(len(test_data[0]), 20)
    
    def test_split_reproducibility(self):
        """Test that split is reproducible with same seed."""
        sequences = np.random.randn(100, 10, 5)
        events = np.random.randint(0, 2, 100)
        times = np.random.randint(1, 11, 100)
        
        train1, val1, test1 = create_train_val_test_split(
            sequences, events, times, random_seed=42
        )
        
        train2, val2, test2 = create_train_val_test_split(
            sequences, events, times, random_seed=42
        )
        
        # Check that splits are identical
        np.testing.assert_array_equal(train1[0], train2[0])
        np.testing.assert_array_equal(val1[0], val2[0])
        np.testing.assert_array_equal(test1[0], test2[0])
    
    def test_no_data_leakage(self):
        """Test that there's no overlap between splits."""
        sequences = np.random.randn(30, 10, 5)
        events = np.random.randint(0, 2, 30)
        times = np.random.randint(1, 11, 30)
        
        # Add unique identifiers
        for i in range(30):
            sequences[i, 0, 0] = i
        
        train_data, val_data, test_data = create_train_val_test_split(
            sequences, events, times
        )
        
        # Get unique identifiers from each split
        train_ids = set(train_data[0][:, 0, 0])
        val_ids = set(val_data[0][:, 0, 0])
        test_ids = set(test_data[0][:, 0, 0])
        
        # Check no overlap
        self.assertEqual(len(train_ids & val_ids), 0)
        self.assertEqual(len(train_ids & test_ids), 0)
        self.assertEqual(len(val_ids & test_ids), 0)


if __name__ == '__main__':
    unittest.main()
