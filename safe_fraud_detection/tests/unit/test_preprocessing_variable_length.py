"""
Unit tests for preprocessing variable-length sequences
"""

import unittest
import tempfile
import os
import numpy as np
import torch

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from safe_fraud_detection.data.preprocessing import SequencePreprocessor, create_train_val_test_split


class TestVariableLengthPreprocessing(unittest.TestCase):
    """Test SequencePreprocessor and splitting with lists of sequences."""

    def setUp(self):
        rng = np.random.default_rng(0)
        self.sequences = [rng.normal(3.0, 2.0, size=(n, 3)) for n in (2, 5, 8)]

    def test_fit_uses_real_timesteps_only(self):
        """Test statistics match the unpadded timesteps."""
        preprocessor = SequencePreprocessor(normalize='standard').fit(self.sequences)
        rows = np.concatenate(self.sequences)

        np.testing.assert_allclose(preprocessor.feature_means, rows.mean(axis=0))
        np.testing.assert_allclose(preprocessor.scaler.mean_, rows.mean(axis=0))

    def test_transform_keeps_list_form(self):
        """Test a list comes back as a list with the same lengths, standardized."""
        preprocessor = SequencePreprocessor(normalize='standard').fit(self.sequences)
        transformed = preprocessor.transform(self.sequences)

        self.assertIsInstance(transformed, list)
        self.assertEqual([len(seq) for seq in transformed], [2, 5, 8])
        np.testing.assert_allclose(np.concatenate(transformed).mean(axis=0), 0, atol=1e-7)

    def test_list_and_array_transforms_match(self):
        """Test equal-length data gives the same result as a list or a 3D array."""
        equal_length = np.stack([seq[:2] for seq in self.sequences])
        preprocessor = SequencePreprocessor(normalize='minmax', clip_outliers=2.0).fit(equal_length)

        np.testing.assert_allclose(
            np.stack(preprocessor.transform(list(equal_length))),
            preprocessor.transform(equal_length)
        )

    def test_forward_fill_does_not_cross_sequences(self):
        """Test a leading NaN isn't filled from the previous sequence."""
        sequences = [np.array([[1.0], [5.0]]), np.array([[np.nan], [2.0]])]
        preprocessor = SequencePreprocessor(normalize=None, handle_nan='forward_fill').fit(sequences)

        self.assertEqual(preprocessor.transform(sequences)[1][0, 0], 0.0)

    def test_state_dict_round_trip(self):
        """Test a restored preprocessor transforms identically after a weights-only load."""
        for normalize in ('standard', 'minmax', None):
            with self.subTest(normalize=normalize):
                preprocessor = SequencePreprocessor(normalize=normalize, clip_outliers=3.0).fit(self.sequences)

                with tempfile.TemporaryDirectory() as tmp_dir:
                    path = os.path.join(tmp_dir, 'state.pt')
                    torch.save(preprocessor.state_dict(), path)
                    restored = SequencePreprocessor.from_state_dict(torch.load(path, weights_only=True))

                for original, loaded in zip(preprocessor.transform(self.sequences), restored.transform(self.sequences)):
                    np.testing.assert_allclose(original, loaded)

    def test_split_accepts_list(self):
        """Test splitting a list keeps sequences paired with their labels."""
        sequences = self.sequences * 4
        times = np.array([len(seq) for seq in sequences])

        splits = create_train_val_test_split(sequences, np.zeros(12), times, 0.5, 0.25, 0.25)

        for split_sequences, _, split_times in splits:
            self.assertIsInstance(split_sequences, list)
            self.assertEqual([len(seq) for seq in split_sequences], split_times.tolist())


if __name__ == '__main__':
    unittest.main()
