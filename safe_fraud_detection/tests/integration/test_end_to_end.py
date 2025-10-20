"""
Integration tests for end-to-end workflow
"""

import unittest
import torch
from torch.utils.data import DataLoader
import numpy as np
import tempfile
import os

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from safe_fraud_detection.models.safe_model import SAFEModel
from safe_fraud_detection.models.loss import SAFELoss
from safe_fraud_detection.data.dataset import SurvivalDataset
from safe_fraud_detection.data.preprocessing import (
    SequencePreprocessor,
    create_train_val_test_split
)
from safe_fraud_detection.utils.trainer import SAFETrainer
from safe_fraud_detection.utils.metrics import evaluate_at_timestamps


class TestEndToEndWorkflow(unittest.TestCase):
    """Test complete end-to-end workflow."""
    
    def setUp(self):
        """Set up test data and model."""
        # Create synthetic data
        np.random.seed(42)
        torch.manual_seed(42)
        
        self.num_samples = 100
        self.seq_len = 12
        self.num_features = 5
        
        # Generate sequences
        sequences = np.random.randn(self.num_samples, self.seq_len, self.num_features)
        
        # Generate events and times
        events = np.random.randint(0, 2, self.num_samples)
        times = np.random.randint(5, self.seq_len + 1, self.num_samples)
        
        # Split data
        train_data, val_data, test_data = create_train_val_test_split(
            sequences, events, times,
            train_ratio=0.6,
            val_ratio=0.2,
            test_ratio=0.2,
            random_seed=42
        )
        
        # Preprocess
        self.preprocessor = SequencePreprocessor(normalize='standard')
        train_sequences = self.preprocessor.fit_transform(train_data[0])
        val_sequences = self.preprocessor.transform(val_data[0])
        test_sequences = self.preprocessor.transform(test_data[0])
        
        # Create datasets
        self.train_dataset = SurvivalDataset(
            train_sequences, train_data[1], train_data[2]
        )
        self.val_dataset = SurvivalDataset(
            val_sequences, val_data[1], val_data[2]
        )
        self.test_dataset = SurvivalDataset(
            test_sequences, test_data[1], test_data[2]
        )
        
        # Create data loaders
        self.train_loader = DataLoader(self.train_dataset, batch_size=16, shuffle=True)
        self.val_loader = DataLoader(self.val_dataset, batch_size=16, shuffle=False)
        self.test_loader = DataLoader(self.test_dataset, batch_size=16, shuffle=False)
        
        # Create model
        self.model = SAFEModel(
            input_dim=self.num_features,
            hidden_dim=16,
            num_layers=1
        )
        
        self.loss_fn = SAFELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.01)
    
    def test_training_loop(self):
        """Test that training loop runs without errors."""
        trainer = SAFETrainer(
            model=self.model,
            loss_fn=self.loss_fn,
            optimizer=self.optimizer,
            device='cpu'
        )
        
        # Train for a few epochs
        history = trainer.fit(
            train_loader=self.train_loader,
            val_loader=self.val_loader,
            epochs=5,
            verbose=False
        )
        
        # Check that training happened
        self.assertEqual(len(history['train_losses']), 5)
        self.assertEqual(len(history['val_losses']), 5)
        
        # Check that loss decreased
        self.assertLess(history['train_losses'][-1], history['train_losses'][0])
    
    def test_evaluation(self):
        """Test evaluation workflow."""
        # Train briefly
        trainer = SAFETrainer(
            model=self.model,
            loss_fn=self.loss_fn,
            optimizer=self.optimizer,
            device='cpu'
        )
        
        trainer.fit(
            train_loader=self.train_loader,
            val_loader=self.val_loader,
            epochs=3,
            verbose=False
        )
        
        # Evaluate
        metrics = evaluate_at_timestamps(
            model=self.model,
            data_loader=self.test_loader,
            timestamps=[0, 1, 2, 3, 4],
            threshold=0.5,
            device='cpu'
        )
        
        # Check metrics were computed
        self.assertGreater(len(metrics.metrics_by_time), 0)
        
        # Check early detection stats
        summary = metrics.get_early_detection_summary()
        self.assertIn('total_fraudsters', summary)
        self.assertIn('early_detected_count', summary)
    
    def test_save_and_load_checkpoint(self):
        """Test saving and loading model checkpoint."""
        trainer = SAFETrainer(
            model=self.model,
            loss_fn=self.loss_fn,
            optimizer=self.optimizer,
            device='cpu'
        )
        
        # Train a bit
        trainer.fit(
            train_loader=self.train_loader,
            epochs=2,
            verbose=False
        )
        
        # Save checkpoint
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = os.path.join(tmpdir, 'checkpoint.pt')
            trainer.save_checkpoint(checkpoint_path, epoch=2)
            
            # Create new trainer and load
            new_model = SAFEModel(
                input_dim=self.num_features,
                hidden_dim=16,
                num_layers=1
            )
            new_optimizer = torch.optim.Adam(new_model.parameters())
            new_trainer = SAFETrainer(
                model=new_model,
                loss_fn=self.loss_fn,
                optimizer=new_optimizer,
                device='cpu'
            )
            
            checkpoint = new_trainer.load_checkpoint(checkpoint_path)
            
            # Check checkpoint was loaded
            self.assertEqual(checkpoint['epoch'], 2)
            self.assertEqual(len(checkpoint['train_losses']), 2)
    
    def test_prediction_consistency(self):
        """Test that predictions are consistent (monotonic survival probabilities)."""
        self.model.eval()
        
        # Get a batch
        sequences, _, _ = next(iter(self.test_loader))
        
        # Get predictions
        _, survival_probs, _ = self.model(sequences)
        
        # Check monotonicity for each sample
        for i in range(sequences.shape[0]):
            for t in range(1, self.seq_len):
                self.assertLessEqual(
                    survival_probs[i, t].item(),
                    survival_probs[i, t-1].item(),
                    msg=f"Survival probability not monotonic at sample {i}, time {t}"
                )
    
    def test_overfitting_on_small_dataset(self):
        """Test that model can overfit on a tiny dataset (sanity check)."""
        # Create tiny dataset
        tiny_sequences = np.random.randn(10, 10, 5)
        tiny_events = np.ones(10)  # All events
        tiny_times = np.random.randint(5, 11, 10)
        
        tiny_sequences = self.preprocessor.fit_transform(tiny_sequences)
        tiny_dataset = SurvivalDataset(tiny_sequences, tiny_events, tiny_times)
        tiny_loader = DataLoader(tiny_dataset, batch_size=10, shuffle=True)
        
        # Create small model
        model = SAFEModel(input_dim=5, hidden_dim=8)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        trainer = SAFETrainer(model, self.loss_fn, optimizer, device='cpu')
        
        # Train for many epochs
        history = trainer.fit(
            train_loader=tiny_loader,
            epochs=50,
            verbose=False
        )
        
        # Loss should decrease significantly
        initial_loss = history['train_losses'][0]
        final_loss = history['train_losses'][-1]
        
        self.assertLess(final_loss, initial_loss * 0.5)


class TestModelPersistence(unittest.TestCase):
    """Test model saving and loading."""
    
    def test_save_and_load_full_model(self):
        """Test saving and loading complete model with config."""
        model = SAFEModel(input_dim=5, hidden_dim=16)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = os.path.join(tmpdir, 'model.pt')
            
            # Save
            torch.save({
                'model_state_dict': model.state_dict(),
                'config': {
                    'input_dim': 5,
                    'hidden_dim': 16,
                    'num_layers': 1
                }
            }, model_path)
            
            # Load
            checkpoint = torch.load(model_path)
            new_model = SAFEModel(**checkpoint['config'])
            new_model.load_state_dict(checkpoint['model_state_dict'])
            
            # Check models are identical
            for p1, p2 in zip(model.parameters(), new_model.parameters()):
                self.assertTrue(torch.allclose(p1, p2))


if __name__ == '__main__':
    unittest.main()
