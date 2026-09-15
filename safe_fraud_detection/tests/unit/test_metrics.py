"""
Unit tests for evaluation metrics
"""

import unittest
import torch
from torch.utils.data import DataLoader
import numpy as np

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from safe_fraud_detection.data.dataset import SurvivalDataset
from safe_fraud_detection.utils.metrics import EarlyDetectionMetrics, evaluate_at_timestamps


class FixedSurvivalModel(torch.nn.Module):
    """Stub model that returns preset survival probabilities."""

    def __init__(self, survival_probs: torch.Tensor):
        super().__init__()
        self.survival_probs = survival_probs

    def forward(self, x, mask=None):
        return None, self.survival_probs[:x.shape[0]], None


class TestEvaluateAtTimestamps(unittest.TestCase):
    """Test cases for evaluate_at_timestamps."""

    def test_never_detected_fraudster_is_not_early(self):
        """Test a fraudster whose survival never drops below threshold isn't counted."""
        survival_probs = torch.tensor([
            [0.9, 0.9, 0.9, 0.9, 0.9],  # fraudster, never detected
            [0.9, 0.4, 0.3, 0.2, 0.1],  # fraudster, detected at t=1 before time 3
            [0.9, 0.9, 0.9, 0.9, 0.9],  # censored
        ])
        dataset = SurvivalDataset(
            np.zeros((3, 5, 2)),
            np.array([1, 1, 0]),
            np.array([5, 3, 4])
        )
        loader = DataLoader(dataset, batch_size=3, shuffle=False)

        metrics = evaluate_at_timestamps(
            FixedSurvivalModel(survival_probs), loader, timestamps=[0], threshold=0.5
        )
        summary = metrics.get_early_detection_summary()

        self.assertEqual(summary['total_fraudsters'], 2)
        self.assertEqual(summary['early_detected_count'], 1)
        self.assertAlmostEqual(summary['early_detection_rate'], 0.5)
        self.assertAlmostEqual(summary['avg_early_timestamps'], 2.0)

        # Numpy scalars would break torch.load(weights_only=True) on saved checkpoints
        for key, value in summary.items():
            self.assertIn(type(value), (int, float), msg=key)

    def test_timestamp_metrics_use_observed_samples_only(self):
        """Test @t metrics skip samples whose sequences ended before t."""
        survival_probs = torch.tensor([
            [0.9, 0.9, 0.1, 0.1, 0.1],  # fraudster, length 2: low values are padding
            [0.9, 0.9, 0.9, 0.9, 0.9],  # fraudster, length 5, missed
            [0.9, 0.9, 0.9, 0.9, 0.9],  # censored, length 5
        ])
        dataset = SurvivalDataset(
            [np.zeros((2, 2)), np.zeros((5, 2)), np.zeros((5, 2))],
            np.array([1, 1, 0]),
            np.array([2, 5, 5])
        )
        loader = DataLoader(dataset, batch_size=3, shuffle=False)

        metrics = evaluate_at_timestamps(
            FixedSurvivalModel(survival_probs), loader, timestamps=[3], threshold=0.5
        )

        # Only the two length-5 samples are scored at @4: censored correct, fraud missed
        self.assertAlmostEqual(metrics.get_metrics_at_k(4)['accuracy'], 0.5)


class TestEarlyDetectionMetrics(unittest.TestCase):
    """Test cases for EarlyDetectionMetrics."""

    def test_format_summary_orders_timestamps_numerically(self):
        """Test @2 is reported before @10."""
        metrics = EarlyDetectionMetrics()
        for timestamp in (10, 2):
            metrics.update_at_timestamp(timestamp, np.array([0, 1]), np.array([0, 1]), np.array([0.9, 0.1]))

        text = metrics.format_summary()

        self.assertLess(text.index('Metrics @2:'), text.index('Metrics @10:'))
        self.assertIn('Average Metrics (@2 to @10)', text)


if __name__ == '__main__':
    unittest.main()
