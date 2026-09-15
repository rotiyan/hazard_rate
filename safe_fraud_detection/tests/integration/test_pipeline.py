"""
Integration tests for the shared pipeline with variable-length sequences
"""

import unittest
import tempfile
import os
import numpy as np
import torch

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from safe_fraud_detection.data.dataset import SurvivalDataset
from safe_fraud_detection.data.preprocessing import create_train_val_test_split
from safe_fraud_detection.models.loss import SAFELoss, WeightedSAFELoss
from safe_fraud_detection.models.safe_model import SAFEModel
from safe_fraud_detection.pipeline import (
    build_loss,
    build_model,
    build_trainer,
    load_model,
    prepare_dataloaders,
    save_model
)
from safe_fraud_detection.utils.config import Config
from safe_fraud_detection.utils.metrics import evaluate_at_timestamps, predict_survival


def make_variable_length_data(num_samples=80, num_features=4, seed=0):
    """Random sequences of 3-12 timesteps, with time_observed equal to length."""
    rng = np.random.default_rng(seed)
    lengths = rng.integers(3, 13, num_samples)
    sequences = [rng.normal(5.0, 2.0, size=(n, num_features)) for n in lengths]
    events = rng.integers(0, 2, num_samples)
    return sequences, events, lengths


def make_config(**loss):
    return Config.from_dict({
        'model': {'input_dim': 4, 'hidden_dim': 8},
        'training': {'batch_size': 16, 'epochs': 2, 'lr_scheduler': 'plateau'},
        'loss': loss,
        'evaluation': {'eval_timestamps': [0, 2, 5]},
        'device': 'cpu'
    })


class TestPipeline(unittest.TestCase):
    """Test cases for safe_fraud_detection.pipeline."""

    def setUp(self):
        torch.manual_seed(0)
        self.sequences, self.events, self.times = make_variable_length_data()

    def test_preprocessing_fits_on_training_split_only(self):
        """Test the scaler statistics come from the unpadded training split."""
        config = make_config()
        train_loader, _, _, preprocessor = prepare_dataloaders(
            self.sequences, self.events, self.times, config
        )

        train_data, _, _ = create_train_val_test_split(
            self.sequences, self.events, self.times,
            train_ratio=config.data.train_ratio,
            val_ratio=config.data.val_ratio,
            test_ratio=config.data.test_ratio,
            random_seed=config.data.random_seed
        )

        np.testing.assert_allclose(preprocessor.feature_means, np.concatenate(train_data[0]).mean(axis=0))
        np.testing.assert_array_equal(train_loader.dataset.lengths, [len(seq) for seq in train_data[0]])

    def test_padding_does_not_change_loss(self):
        """Test a sample's loss is the same alone or padded next to a longer sequence."""
        model = SAFEModel(input_dim=4, hidden_dim=8)
        loss_fn = SAFELoss()
        short = np.random.randn(3, 4)

        alone = SurvivalDataset([short], [1], [3])
        padded = SurvivalDataset([short, np.random.randn(9, 4)], [1, 0], [3, 9])
        padded.sequences[0, 3:] = 100.0  # Junk in padding must be ignored

        def loss_of(sample):
            x, mask = sample.sequences.unsqueeze(0), sample.masks.unsqueeze(0)
            hazard_rates, _, _ = model(x, mask)
            return loss_fn(hazard_rates, sample.events.unsqueeze(0), sample.times.unsqueeze(0), mask)

        torch.testing.assert_close(loss_of(alone[0]), loss_of(padded[0]))

    def test_auto_weight_balances_classes(self):
        """Test auto_weight computes balanced class weights from training events."""
        loss_fn = build_loss(make_config(loss_type='weighted', auto_weight=True), train_events=np.array([1, 0, 0, 0]))

        self.assertIsInstance(loss_fn, WeightedSAFELoss)
        self.assertAlmostEqual(loss_fn.event_weight, 2.0)
        self.assertAlmostEqual(loss_fn.censored_weight, 4 / 6)

    def test_train_evaluate_save_load_round_trip(self):
        """Test training, evaluation and a checkpoint round trip with variable-length data."""
        config = make_config(loss_type='weighted', auto_weight=True)
        train_loader, val_loader, test_loader, preprocessor = prepare_dataloaders(
            self.sequences, self.events, self.times, config
        )

        model = build_model(config)
        trainer = build_trainer(config, model, train_events=train_loader.dataset.events)
        history = trainer.fit(train_loader, val_loader, epochs=config.training.epochs, verbose=False)
        self.assertEqual(len(history['train_losses']), 2)

        metrics = evaluate_at_timestamps(model, test_loader, config.evaluation.eval_timestamps, device='cpu')
        self.assertIn('@1', metrics.metrics_by_time)

        with tempfile.TemporaryDirectory() as tmp_dir:
            path = os.path.join(tmp_dir, 'nested', 'model.pt')
            save_model(path, model, config, preprocessor,
                       metrics=metrics.get_early_detection_summary(), history=history)
            loaded_model, loaded_config, loaded_preprocessor = load_model(path, device='cpu')

        self.assertEqual(loaded_config.to_dict(), config.to_dict())
        np.testing.assert_allclose(
            predict_survival(loaded_model, test_loader)['survival_probs'],
            predict_survival(model, test_loader)['survival_probs']
        )
        for original, restored in zip(preprocessor.transform(self.sequences[:5]),
                                      loaded_preprocessor.transform(self.sequences[:5])):
            np.testing.assert_allclose(original, restored)


if __name__ == '__main__':
    unittest.main()
