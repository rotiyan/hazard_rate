"""
Unit tests for dataset classes
"""

import unittest
import torch
from torch.utils.data import DataLoader
import numpy as np

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from safe_fraud_detection.data.dataset import (
    SurvivalBatch,
    SurvivalDataset,
    CreditCardDataset,
    TwitterDataset,
    WikiDataset
)


class TestSurvivalDatasetFixedLength(unittest.TestCase):
    """Test cases for SurvivalDataset with a 3D array."""

    def setUp(self):
        """Set up test fixtures."""
        self.sequences = np.random.randn(20, 10, 5)
        self.events = np.random.randint(0, 2, 20)
        self.times = np.random.randint(1, 11, 20)

        self.dataset = SurvivalDataset(self.sequences, self.events, self.times)

    def test_dataset_length(self):
        """Test dataset returns correct length."""
        self.assertEqual(len(self.dataset), 20)

    def test_getitem_returns_survival_batch(self):
        """Test __getitem__ returns a SurvivalBatch with correct shapes and types."""
        sample = self.dataset[0]

        self.assertIsInstance(sample, SurvivalBatch)
        self.assertEqual(sample.sequences.shape, (10, 5))
        self.assertEqual(sample.masks.shape, (10,))
        self.assertEqual(sample.events.shape, ())
        self.assertEqual(sample.times.dtype, torch.long)
        self.assertEqual(sample.lengths.item(), 10)
        self.assertTrue(torch.all(sample.masks == 1))

    def test_indexing(self):
        """Test that indexing returns the matching sample."""
        for i in range(5):
            sample = self.dataset[i]
            np.testing.assert_array_almost_equal(sample.sequences.numpy(), self.sequences[i], decimal=5)
            self.assertEqual(sample.times.item(), self.times[i])

    def test_properties(self):
        """Test dataset properties."""
        self.assertEqual(self.dataset.num_features, 5)
        self.assertEqual(self.dataset.max_seq_len, 10)

    def test_get_statistics(self):
        """Test get_statistics method."""
        stats = self.dataset.get_statistics()

        self.assertEqual(stats['num_samples'], 20)
        self.assertEqual(stats['num_features'], 5)
        self.assertEqual(stats['max_seq_len'], 10)
        self.assertEqual(stats['num_events'] + stats['num_censored'], 20)
        self.assertAlmostEqual(stats['event_rate'], self.events.mean())

    def test_default_collate_batches(self):
        """Test DataLoader batches keep the SurvivalBatch structure."""
        batch = next(iter(DataLoader(self.dataset, batch_size=4)))

        self.assertIsInstance(batch, SurvivalBatch)
        self.assertEqual(batch.sequences.shape, (4, 10, 5))
        self.assertEqual(batch.masks.shape, (4, 10))
        self.assertEqual(batch.events.shape, (4,))
        self.assertEqual(batch.times.shape, (4,))
        self.assertEqual(batch.lengths.shape, (4,))


class TestSurvivalDatasetVariableLength(unittest.TestCase):
    """Test cases for SurvivalDataset with variable-length sequences."""

    def setUp(self):
        """Set up variable-length sequences."""
        self.sequences = [np.random.randn(n, 3) for n in (4, 2, 6)]

    def test_padding_and_masks(self):
        """Test sequences are padded and masks mark real positions."""
        dataset = SurvivalDataset(self.sequences, np.array([1, 0, 1]), np.array([4, 2, 5]))
        sample = dataset[1]

        self.assertEqual(sample.sequences.shape, (6, 3))
        np.testing.assert_array_equal(sample.masks.numpy(), [1, 1, 0, 0, 0, 0])
        np.testing.assert_array_equal(sample.sequences[2:].numpy(), 0)
        self.assertEqual(sample.lengths.item(), 2)

    def test_max_seq_len_truncates(self):
        """Test max_seq_len truncates longer sequences and their lengths."""
        dataset = SurvivalDataset(self.sequences, np.array([1, 0, 1]), np.array([4, 2, 5]), max_seq_len=3)

        self.assertEqual(dataset[2].sequences.shape, (3, 3))
        self.assertEqual(dataset[2].lengths.item(), 3)

    def test_get_statistics(self):
        """Test sequence length statistics."""
        stats = SurvivalDataset(self.sequences, np.array([1, 0, 1]), np.array([4, 2, 5])).get_statistics()

        self.assertAlmostEqual(stats['avg_seq_len'], 4.0)
        self.assertEqual(stats['min_seq_len'], 2)
        self.assertEqual(stats['num_events'], 2)

    def test_time_observed_beyond_length_raises(self):
        """Test time_observed pointing into padding is rejected."""
        with self.assertRaisesRegex(ValueError, "index 1"):
            SurvivalDataset(self.sequences, np.array([1, 1, 0]), np.array([4, 3, 6]))

    def test_time_observed_zero_raises(self):
        """Test time_observed below 1 is rejected."""
        with self.assertRaises(ValueError):
            SurvivalDataset(self.sequences, np.array([1, 1, 0]), np.array([0, 2, 6]))

    def test_mismatched_lengths_raise(self):
        """Test inputs of different lengths are rejected."""
        with self.assertRaisesRegex(ValueError, "same length"):
            SurvivalDataset(self.sequences, np.array([1, 0]), np.array([4, 2, 5]))

    def test_empty_raises(self):
        """Test an empty dataset is rejected."""
        with self.assertRaises(ValueError):
            SurvivalDataset([], [], [])

    def test_feature_names_must_match(self):
        """Test explicit feature names must match the number of features."""
        with self.assertRaisesRegex(ValueError, "feature names"):
            SurvivalDataset(self.sequences, np.array([1, 0, 1]), np.array([4, 2, 5]), feature_names=['a', 'b'])


class TestSpecializedDatasets(unittest.TestCase):
    """Test specialized dataset classes."""

    def test_twitter_dataset(self):
        """Test TwitterDataset."""
        dataset = TwitterDataset(np.random.randn(10, 15, 5), np.random.randint(0, 2, 10), np.random.randint(1, 16, 10))

        self.assertIsInstance(dataset, SurvivalDataset)
        self.assertEqual(len(dataset), 10)
        self.assertEqual(len(dataset.feature_names), 5)

    def test_wiki_dataset(self):
        """Test WikiDataset."""
        dataset = WikiDataset(np.random.randn(10, 15, 8), np.random.randint(0, 2, 10), np.random.randint(1, 16, 10))

        self.assertEqual(len(dataset), 10)
        self.assertEqual(len(dataset.feature_names), 8)

    def test_credit_card_default_names_only_for_standard_layout(self):
        """Test default feature names are skipped when the data has a different layout."""
        standard = CreditCardDataset([np.random.randn(4, 10)], [1], [4])
        custom = CreditCardDataset([np.random.randn(4, 21)], [1], [4])

        self.assertEqual(len(standard.feature_names), 10)
        self.assertIsNone(custom.feature_names)


if __name__ == '__main__':
    unittest.main()
