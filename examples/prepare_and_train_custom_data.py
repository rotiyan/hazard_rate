"""
Complete example: Prepare custom credit card data and train the SAFE model.

This script:
1. Loads a weekly features table and a close-dates table (CSV or Parquet)
2. Combines them into variable-length sequences with prepare_credit_card_data
3. Trains and evaluates SAFE with credit_card_config.yaml
4. Saves the model with its config and preprocessing

Usage:
    python examples/prepare_and_train_custom_data.py \
        --features path/to/features.csv --close-dates path/to/close_dates.csv
"""

import argparse
import logging
import os
import sys

import pandas as pd

# Add parent directory to path
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(REPO_ROOT)

from safe_fraud_detection.data.prepare_credit_card_data import prepare_credit_card_data
from safe_fraud_detection.pipeline import build_model, build_trainer, prepare_dataloaders, save_model
from safe_fraud_detection.utils.config import Config
from safe_fraud_detection.utils.metrics import evaluate_at_timestamps


DEFAULT_CONFIG = os.path.join(REPO_ROOT, 'safe_fraud_detection', 'configs', 'credit_card_config.yaml')


def read_table(path: str) -> pd.DataFrame:
    """Read a CSV or Parquet file."""
    return pd.read_parquet(path) if path.endswith('.parquet') else pd.read_csv(path)


def parse_args():
    parser = argparse.ArgumentParser(description='Prepare credit card data and train SAFE')
    parser.add_argument('--features', required=True, help='CSV/Parquet with one row per card per week')
    parser.add_argument('--close-dates', required=True, help='CSV/Parquet with card close dates')
    parser.add_argument('--card-number-col', default='card_number')
    parser.add_argument('--week-start-col', default='week_start')
    parser.add_argument('--close-date-col', default='CLOSE_DT')
    parser.add_argument('--is-fraud-col', default=None,
                        help='Column marking fraud closures (default: every closure is fraud)')
    parser.add_argument('--config', default=DEFAULT_CONFIG, help='Config YAML')
    parser.add_argument('--epochs', type=int, default=None, help='Override training epochs from the config')
    parser.add_argument('--output', default='outputs/custom_model.pt', help='Where to save the model')
    return parser.parse_args()


def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format='%(message)s')

    # ============= 1. Load and Prepare Data =============
    print("=== Step 1: Loading and Preparing Data ===")
    features_df = read_table(args.features)
    close_dates_df = read_table(args.close_dates)
    print(f"Features dataframe shape: {features_df.shape}")
    print(f"Close dates dataframe shape: {close_dates_df.shape}")

    sequences, events, times = prepare_credit_card_data(
        features_df=features_df,
        close_dates_df=close_dates_df,
        card_number_col=args.card_number_col,
        week_start_col=args.week_start_col,
        close_date_col=args.close_date_col,
        is_fraud_col=args.is_fraud_col
    )

    config = Config.from_yaml(args.config)
    config.model.input_dim = sequences[0].shape[1]
    if args.epochs is not None:
        config.training.epochs = args.epochs

    # ============= 2. Split, Preprocess, Build Loaders =============
    print("\n=== Step 2: Building Datasets ===")
    train_loader, val_loader, test_loader, preprocessor = prepare_dataloaders(
        sequences, events, times, config
    )
    print(f"Train stats: {train_loader.dataset.get_statistics()}")

    # ============= 3. Train =============
    print("\n=== Step 3: Training Model ===")
    model = build_model(config)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    trainer = build_trainer(config, model, train_events=train_loader.dataset.events)
    history = trainer.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=config.training.epochs,
        early_stopping_patience=config.training.early_stopping_patience
    )

    # ============= 4. Evaluate and Save =============
    print("\n=== Step 4: Evaluating on Test Set ===")
    metrics = evaluate_at_timestamps(
        model, test_loader,
        timestamps=config.evaluation.eval_timestamps,
        threshold=config.evaluation.threshold,
        device=config.device
    )
    metrics.print_summary()

    save_model(
        args.output, model, config, preprocessor,
        metrics=metrics.get_early_detection_summary(),
        history=history
    )
    print(f"Model saved to: {args.output}")


if __name__ == "__main__":
    main()
