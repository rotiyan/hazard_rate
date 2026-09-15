"""
Unit tests for credit card data preparation
"""

import unittest
import numpy as np
import pandas as pd

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from safe_fraud_detection.data.prepare_credit_card_data import prepare_credit_card_data


def make_features(cards, num_weeks=6, start='2023-01-02', skip_weeks=()):
    """Weekly feature rows for each card; feature value equals the week index."""
    weeks = pd.date_range(start, periods=num_weeks, freq='7D')
    rows = [
        {'card_number': card, 'week_start': week, 'week_idx': float(i)}
        for card in cards
        for i, week in enumerate(weeks)
        if i not in skip_weeks
    ]
    return pd.DataFrame(rows)


def prepare(features_df, close_dates):
    close_dates_df = pd.DataFrame({
        'card_number': list(close_dates.keys()),
        'CLOSE_DT': [pd.Timestamp(d) if d else None for d in close_dates.values()]
    })
    return prepare_credit_card_data(features_df, close_dates_df)


class TestPrepareCreditCardData(unittest.TestCase):
    """Test cases for prepare_credit_card_data."""

    def test_close_mid_week_keeps_weeks_before_close(self):
        """Test a card closed during week index 2 keeps weeks 0-2."""
        sequences, events, times = prepare(make_features(['A']), {'A': '2023-01-18'})

        self.assertEqual(events.tolist(), [1])
        self.assertEqual(times.tolist(), [3])
        np.testing.assert_array_equal(sequences[0][:, 0], [0, 1, 2])

    def test_close_before_first_week_is_skipped(self):
        """Test a card closed before any observed week is dropped."""
        sequences, events, times = prepare(
            make_features(['A', 'B']), {'A': '2022-12-01', 'B': None}
        )

        self.assertEqual(len(sequences), 1)
        self.assertEqual(events.tolist(), [0])
        self.assertEqual(times.tolist(), [6])

    def test_close_during_last_observed_week_is_event(self):
        """Test a closure inside the last observed week counts as an event."""
        # Last week starts 2023-02-06
        sequences, events, times = prepare(make_features(['A']), {'A': '2023-02-08'})

        self.assertEqual(events.tolist(), [1])
        self.assertEqual(times.tolist(), [6])

    def test_close_after_observation_is_censored(self):
        """Test a closure after the observation window is censored."""
        sequences, events, times = prepare(make_features(['A']), {'A': '2023-06-01'})

        self.assertEqual(events.tolist(), [0])
        self.assertEqual(times.tolist(), [6])

    def test_missing_weeks_keep_time_equal_to_length(self):
        """Test time_observed counts rows, not calendar weeks, when weeks are missing."""
        features_df = make_features(['A'], skip_weeks=(1, 2))
        sequences, events, times = prepare(features_df, {'A': '2023-01-25'})

        self.assertEqual(events.tolist(), [1])
        self.assertEqual(times.tolist(), [len(sequences[0])])
        np.testing.assert_array_equal(sequences[0][:, 0], [0, 3])

    def test_time_observed_matches_sequence_length(self):
        """Test every card's time_observed equals its sequence length."""
        sequences, events, times = prepare(
            make_features(['A', 'B', 'C', 'D']),
            {'A': '2023-01-10', 'B': None, 'C': '2023-02-01', 'D': '2023-01-02'}
        )

        self.assertEqual(times.tolist(), [len(s) for s in sequences])


if __name__ == '__main__':
    unittest.main()
