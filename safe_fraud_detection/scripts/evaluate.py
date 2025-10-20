"""
Evaluation script for SAFE model
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
from safe_fraud_detection.data.dataset import SurvivalDataset
from safe_fraud_detection.data.preprocessing import SequencePreprocessor
from safe_fraud_detection.utils.metrics import evaluate_at_timestamps
from safe_fraud_detection.utils.config import Config


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description='Evaluate SAFE model')
    parser.add_argument('--model', type=str, required=True, help='Path to trained model')
    parser.add_argument('--data', type=str, required=True, help='Path to test data')
    parser.add_argument('--threshold', type=float, default=0.5, help='Classification threshold')
    parser.add_argument('--timestamps', type=int, nargs='+', default=[0, 1, 2, 3, 4],
                       help='Timestamps to evaluate at')
    parser.add_argument('--device', type=str, default='cpu', help='Device (cuda/cpu)')
    
    args = parser.parse_args()
    
    # Load model
    logger.info(f"Loading model from {args.model}")
    checkpoint = torch.load(args.model, map_location=args.device)
    config_dict = checkpoint.get('config', {})
    
    # Reconstruct model
    model_config = config_dict.get('model', {})
    model = SAFEModel(
        input_dim=model_config.get('input_dim', 5),
        hidden_dim=model_config.get('hidden_dim', 32),
        num_layers=model_config.get('num_layers', 1),
        dropout=model_config.get('dropout', 0.0)
    )
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(args.device)
    model.eval()
    
    logger.info(f"Model loaded successfully")
    
    # Load test data
    logger.info(f"Loading test data from {args.data}")
    
    if os.path.exists(args.data):
        data = np.load(args.data, allow_pickle=True)
        sequences = data['sequences']
        events = data['events']
        times = data['times']
    else:
        raise FileNotFoundError(f"Data file not found: {args.data}")
    
    # Preprocess (use same config as training if available)
    data_config = config_dict.get('data', {})
    preprocessor = SequencePreprocessor(
        normalize=data_config.get('normalize', 'standard'),
        handle_nan=data_config.get('handle_nan', 'zero')
    )
    
    # For evaluation, we fit on the test data itself (or load fitted preprocessor)
    sequences = preprocessor.fit_transform(sequences)
    
    # Create dataset and loader
    test_dataset = SurvivalDataset(sequences, events, times)
    test_loader = DataLoader(
        test_dataset,
        batch_size=32,
        shuffle=False,
        num_workers=0
    )
    
    logger.info(f"Loaded {len(test_dataset)} test samples")
    logger.info(f"Test stats: {test_dataset.get_statistics()}")
    
    # Evaluate
    logger.info("Running evaluation...")
    metrics = evaluate_at_timestamps(
        model=model,
        data_loader=test_loader,
        timestamps=args.timestamps,
        threshold=args.threshold,
        device=args.device
    )
    
    # Print results
    metrics.print_summary()
    
    # Save results
    output_path = args.model.replace('.pt', '_eval_results.txt')
    with open(output_path, 'w') as f:
        f.write("SAFE Model Evaluation Results\n")
        f.write("=" * 60 + "\n\n")
        
        for time_key in sorted(metrics.metrics_by_time.keys()):
            metrics_dict = metrics.metrics_by_time[time_key]
            f.write(f"\nMetrics {time_key}:\n")
            for metric_name, value in metrics_dict.items():
                f.write(f"  {metric_name:12s}: {value:.4f}\n")
        
        early_summary = metrics.get_early_detection_summary()
        f.write("\nEarly Detection Summary:\n")
        for key, value in early_summary.items():
            if isinstance(value, float):
                f.write(f"  {key:25s}: {value:.4f}\n")
            else:
                f.write(f"  {key:25s}: {value}\n")
    
    logger.info(f"Results saved to {output_path}")


if __name__ == '__main__':
    main()
