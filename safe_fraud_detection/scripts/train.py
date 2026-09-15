"""
Training script for SAFE model
"""

import sys
import argparse
import logging
from pathlib import Path

import numpy as np

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from safe_fraud_detection.data.npz_io import load_npz_data
from safe_fraud_detection.pipeline import build_model, build_trainer, prepare_dataloaders, save_model
from safe_fraud_detection.utils.metrics import evaluate_at_timestamps
from safe_fraud_detection.utils.config import Config, resolve_device


# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description='Train SAFE model')
    parser.add_argument('--config', type=str, default=None, help='Path to config file')
    parser.add_argument('--data', type=str, required=True, help='Path to .npz data file')
    parser.add_argument('--output', type=str, default='checkpoints/safe_model.pt', help='Output path for model')
    parser.add_argument('--device', type=str, default=None, help="Device ('auto', 'cuda' or 'cpu')")

    args = parser.parse_args()

    # Load config
    config = Config.from_yaml(args.config) if args.config else Config()
    if args.device:
        config.device = resolve_device(args.device)

    logger.info(f"Using device: {config.device}")

    # Load data (fixed-length array or list of variable-length sequences)
    logger.info(f"Loading data from {args.data}")
    sequences, events, times = load_npz_data(args.data)
    logger.info(f"Loaded {len(events)} samples")

    # The model's input size must match the data; the saved config then records it
    num_features = np.asarray(sequences[0]).shape[-1]
    if config.model.input_dim != num_features:
        logger.warning(
            f"Config model.input_dim={config.model.input_dim} but data has "
            f"{num_features} features; using {num_features}"
        )
        config.model.input_dim = num_features

    logger.info(f"Configuration: {config.to_dict()}")

    # Split, fit preprocessing on the training split, and build loaders
    train_loader, val_loader, test_loader, preprocessor = prepare_dataloaders(
        sequences, events, times, config
    )
    logger.info(f"Train stats: {train_loader.dataset.get_statistics()}")

    # Create model and trainer
    model = build_model(config)
    logger.info(f"Model created with {sum(p.numel() for p in model.parameters())} parameters")
    trainer = build_trainer(config, model, train_events=train_loader.dataset.events)

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

    # Save model with its config and fitted preprocessing
    save_model(
        args.output,
        model,
        config,
        preprocessor,
        metrics=metrics.get_early_detection_summary(),
        history=history
    )

    logger.info(f"Model saved to {args.output}")


if __name__ == '__main__':
    main()
