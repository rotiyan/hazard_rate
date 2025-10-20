"""
Training script for SAFE model
"""

import os
import sys
import argparse
import logging
from pathlib import Path

import torch
from torch.utils.data import DataLoader
import numpy as np

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from safe_fraud_detection.models.safe_model import SAFEModel
from safe_fraud_detection.models.loss import SAFELoss, RegularSurvivalLoss, WeightedSAFELoss
from safe_fraud_detection.data.dataset import SurvivalDataset
from safe_fraud_detection.data.preprocessing import SequencePreprocessor, create_train_val_test_split
from safe_fraud_detection.utils.trainer import SAFETrainer
from safe_fraud_detection.utils.metrics import evaluate_at_timestamps
from safe_fraud_detection.utils.config import Config


# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_data(data_path: str, config: Config):
    """
    Load and preprocess data.
    
    Args:
        data_path: Path to data file
        config: Configuration object
        
    Returns:
        train_loader, val_loader, test_loader, preprocessor
    """
    logger.info(f"Loading data from {data_path}")
    
    # Load your data here - this is a placeholder
    # You should replace this with actual data loading logic
    # Expected format: sequences (N, T, F), events (N,), times (N,)
    
    # For demonstration, we'll show how to load from numpy files
    if os.path.exists(data_path):
        data = np.load(data_path, allow_pickle=True)
        sequences = data['sequences']
        events = data['events']
        times = data['times']
    else:
        raise FileNotFoundError(f"Data file not found: {data_path}")
    
    logger.info(f"Loaded {len(sequences)} samples")
    
    # Split data
    train_data, val_data, test_data = create_train_val_test_split(
        sequences, events, times,
        train_ratio=config.data.train_ratio,
        val_ratio=config.data.val_ratio,
        test_ratio=config.data.test_ratio,
        random_seed=config.data.random_seed
    )
    
    # Preprocess
    preprocessor = SequencePreprocessor(
        normalize=config.data.normalize,
        handle_nan=config.data.handle_nan
    )
    
    train_sequences, train_events, train_times = train_data
    train_sequences = preprocessor.fit_transform(train_sequences)
    
    val_sequences, val_events, val_times = val_data
    val_sequences = preprocessor.transform(val_sequences)
    
    test_sequences, test_events, test_times = test_data
    test_sequences = preprocessor.transform(test_sequences)
    
    # Create datasets
    train_dataset = SurvivalDataset(train_sequences, train_events, train_times)
    val_dataset = SurvivalDataset(val_sequences, val_events, val_times)
    test_dataset = SurvivalDataset(test_sequences, test_events, test_times)
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.training.batch_size,
        shuffle=True,
        num_workers=0
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.training.batch_size,
        shuffle=False,
        num_workers=0
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.training.batch_size,
        shuffle=False,
        num_workers=0
    )
    
    logger.info(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")
    logger.info(f"Train stats: {train_dataset.get_statistics()}")
    
    return train_loader, val_loader, test_loader, preprocessor


def create_model_and_optimizer(config: Config):
    """
    Create model, loss function, and optimizer.
    
    Args:
        config: Configuration object
        
    Returns:
        model, loss_fn, optimizer
    """
    # Create model
    model = SAFEModel(
        input_dim=config.model.input_dim,
        hidden_dim=config.model.hidden_dim,
        num_layers=config.model.num_layers,
        dropout=config.model.dropout
    )
    
    logger.info(f"Model created with {sum(p.numel() for p in model.parameters())} parameters")
    
    # Create loss function
    if config.loss.loss_type == 'safe':
        loss_fn = SAFELoss(epsilon=config.loss.epsilon)
    elif config.loss.loss_type == 'regular':
        loss_fn = RegularSurvivalLoss(epsilon=config.loss.epsilon)
    elif config.loss.loss_type == 'weighted':
        loss_fn = WeightedSAFELoss(
            event_weight=config.loss.event_weight,
            censored_weight=config.loss.censored_weight,
            epsilon=config.loss.epsilon
        )
    else:
        raise ValueError(f"Unknown loss type: {config.loss.loss_type}")
    
    # Create optimizer
    if config.training.optimizer.lower() == 'adam':
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=config.training.learning_rate,
            weight_decay=config.training.weight_decay
        )
    elif config.training.optimizer.lower() == 'sgd':
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=config.training.learning_rate,
            momentum=0.9,
            weight_decay=config.training.weight_decay
        )
    else:
        raise ValueError(f"Unknown optimizer: {config.training.optimizer}")
    
    return model, loss_fn, optimizer


def main():
    parser = argparse.ArgumentParser(description='Train SAFE model')
    parser.add_argument('--config', type=str, default=None, help='Path to config file')
    parser.add_argument('--data', type=str, required=True, help='Path to data file')
    parser.add_argument('--output', type=str, default='checkpoints/safe_model.pt', help='Output path for model')
    parser.add_argument('--device', type=str, default=None, help='Device (cuda/cpu)')
    
    args = parser.parse_args()
    
    # Load config
    if args.config:
        config = Config.from_yaml(args.config)
    else:
        config = Config()
    
    if args.device:
        config.device = args.device
    
    logger.info(f"Using device: {config.device}")
    logger.info(f"Configuration: {config.to_dict()}")
    
    # Create output directory
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    
    # Load data
    train_loader, val_loader, test_loader, preprocessor = load_data(args.data, config)
    
    # Create model, loss, optimizer
    model, loss_fn, optimizer = create_model_and_optimizer(config)
    
    # Create trainer
    trainer = SAFETrainer(
        model=model,
        loss_fn=loss_fn,
        optimizer=optimizer,
        device=config.device,
        gradient_clip=config.training.gradient_clip
    )
    
    # Train
    logger.info("Starting training...")
    history = trainer.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=config.training.epochs,
        early_stopping_patience=config.training.early_stopping_patience,
        save_best=True,
        verbose=True
    )
    
    logger.info(f"Training completed. Best val loss: {history['best_val_loss']:.4f}")
    
    # Evaluate on test set
    logger.info("Evaluating on test set...")
    metrics = evaluate_at_timestamps(
        model=model,
        data_loader=test_loader,
        timestamps=config.evaluation.eval_timestamps,
        threshold=config.evaluation.threshold,
        device=config.device
    )
    
    metrics.print_summary()
    
    # Save model
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': config.to_dict(),
        'metrics': metrics.get_early_detection_summary(),
        'history': history
    }, args.output)
    
    logger.info(f"Model saved to {args.output}")


if __name__ == '__main__':
    main()
