"""
Unit tests for .npz saving and loading
"""

import unittest
import tempfile
import os
import numpy as np

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from safe_fraud_detection.data.npz_io import load_npz_data, save_npz_data


class TestNpzIO(unittest.TestCase):
    """Test cases for save_npz_data and load_npz_data."""

    def setUp(self):
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.path = os.path.join(self.tmp_dir.name, 'data.npz')

    def tearDown(self):
        self.tmp_dir.cleanup()

    def test_variable_length_round_trip(self):
        """Test variable-length sequences load back as an unpadded list."""
        sequences = [np.ones((3, 2)), np.full((5, 2), 2.0)]
        save_npz_data(self.path, sequences, np.array([1, 0]), np.array([3, 5]))

        loaded, events, times = load_npz_data(self.path)

        self.assertIsInstance(loaded, list)
        self.assertEqual([seq.shape for seq in loaded], [(3, 2), (5, 2)])
        np.testing.assert_array_equal(loaded[1], sequences[1])
        np.testing.assert_array_equal(events, [1, 0])
        np.testing.assert_array_equal(times, [3, 5])

    def test_equal_length_list_round_trip(self):
        """Test a list of equal-length sequences saves and loads."""
        sequences = [np.zeros((4, 3)), np.ones((4, 3))]
        save_npz_data(self.path, sequences, np.array([0, 1]), np.array([4, 4]))

        loaded, _, _ = load_npz_data(self.path)

        np.testing.assert_array_equal(np.stack(loaded), np.stack(sequences))

    def test_fixed_length_array_round_trip(self):
        """Test a 3D array loads back unchanged."""
        sequences = np.random.randn(6, 4, 3)
        save_npz_data(self.path, sequences, np.zeros(6), np.full(6, 4))

        loaded, _, _ = load_npz_data(self.path)

        np.testing.assert_array_equal(loaded, sequences)

    def test_alternate_key_names(self):
        """Test event_indicators/time_observed keys are accepted."""
        np.savez(
            self.path,
            sequences=np.zeros((2, 3, 1)),
            event_indicators=np.array([1, 0]),
            time_observed=np.array([2, 3])
        )

        _, events, times = load_npz_data(self.path)

        np.testing.assert_array_equal(events, [1, 0])
        np.testing.assert_array_equal(times, [2, 3])

    def test_missing_key_raises(self):
        """Test a missing events key gives a clear error."""
        np.savez(self.path, sequences=np.zeros((2, 3, 1)), times=np.array([2, 3]))

        with self.assertRaisesRegex(KeyError, "events"):
            load_npz_data(self.path)

    def test_missing_file_raises(self):
        """Test a missing file raises FileNotFoundError."""
        with self.assertRaises(FileNotFoundError):
            load_npz_data(os.path.join(self.tmp_dir.name, 'missing.npz'))


if __name__ == '__main__':
    unittest.main()
