"""
Evaluation script for SAFE model
"""

import os
import sys
import argparse
import logging
from pathlib import Path

from torch.utils.data import DataLoader

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from safe_fraud_detection.data.dataset import SurvivalDataset
from safe_fraud_detection.data.npz_io import load_npz_data
from safe_fraud_detection.data.preprocessing import SequencePreprocessor
from safe_fraud_detection.pipeline import load_model
from safe_fraud_detection.utils.metrics import evaluate_at_timestamps
from safe_fraud_detection.utils.config import resolve_device


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description='Evaluate SAFE model')
    parser.add_argument('--model', type=str, required=True, help='Path to model saved by safe-train')
    parser.add_argument('--data', type=str, required=True, help='Path to .npz test data')
    parser.add_argument('--threshold', type=float, default=None,
                        help='Classification threshold (default: from the model config)')
    parser.add_argument('--timestamps', type=int, nargs='+', default=None,
                        help='0-indexed timestamps to evaluate at (default: from the model config)')
    parser.add_argument('--batch-size', type=int, default=64, help='Evaluation batch size')
    parser.add_argument('--device', type=str, default='auto', help="Device ('auto', 'cuda' or 'cpu')")

    args = parser.parse_args()
    device = resolve_device(args.device)

    # Load model, config and fitted preprocessing
    logger.info(f"Loading model from {args.model}")
    model, config, preprocessor = load_model(args.model, device)
    logger.info("Model loaded successfully")

    threshold = args.threshold if args.threshold is not None else config.evaluation.threshold
    timestamps = args.timestamps if args.timestamps is not None else config.evaluation.eval_timestamps

    # Load test data
    logger.info(f"Loading test data from {args.data}")
    sequences, events, times = load_npz_data(args.data)

    if preprocessor is None:
        # Older checkpoints don't store preprocessing. Fitting on the evaluation
        # data leaks its statistics, so results will be optimistic.
        logger.warning("Checkpoint has no saved preprocessor; fitting one on the evaluation data")
        preprocessor = SequencePreprocessor(
            normalize=config.data.normalize,
            handle_nan=config.data.handle_nan,
            clip_outliers=config.data.clip_outliers
        ).fit(sequences)

    # Create dataset and loader
    test_dataset = SurvivalDataset(preprocessor.transform(sequences), events, times)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    logger.info(f"Loaded {len(test_dataset)} test samples")
    logger.info(f"Test stats: {test_dataset.get_statistics()}")

    # Evaluate
    logger.info("Running evaluation...")
    metrics = evaluate_at_timestamps(
        model=model,
        data_loader=test_loader,
        timestamps=timestamps,
        threshold=threshold,
        device=device
    )

    # Print and save results
    metrics.print_summary()

    output_path = os.path.splitext(args.model)[0] + '_eval_results.txt'
    with open(output_path, 'w') as f:
        f.write(metrics.format_summary() + "\n")

    logger.info(f"Results saved to {output_path}")


if __name__ == '__main__':
    main()
