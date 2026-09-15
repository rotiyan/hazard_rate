"""
Evaluation metrics for fraud detection
"""

import numpy as np
import torch
from typing import Dict, Optional, List
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, roc_auc_score

from ..data.dataset import SurvivalBatch


def compute_metrics(
    predictions: np.ndarray,
    targets: np.ndarray,
    probabilities: Optional[np.ndarray] = None
) -> Dict[str, float]:
    """
    Compute classification metrics.

    Args:
        predictions: Binary predictions
        targets: Ground truth labels
        probabilities: Optional predicted probabilities for AUC

    Returns:
        Dictionary of metrics
    """
    metrics = {
        'precision': precision_score(targets, predictions, zero_division=0),
        'recall': recall_score(targets, predictions, zero_division=0),
        'f1': f1_score(targets, predictions, zero_division=0),
        'accuracy': accuracy_score(targets, predictions)
    }

    if probabilities is not None:
        try:
            metrics['auc'] = roc_auc_score(targets, probabilities)
        except ValueError:
            metrics['auc'] = 0.0

    return metrics


class EarlyDetectionMetrics:
    """
    Metrics specifically for fraud early detection.

    Tracks:
    - Performance at different timestamps (@1, @2, @3, etc.)
    - Percentage of early detected fraudsters
    - Average early detection timestamps
    """

    def __init__(self):
        self.reset()

    def reset(self):
        """Reset all metrics."""
        self.metrics_by_time = {}
        self.early_detection_stats = {
            'num_fraudsters': 0,
            'num_early_detected': 0,
            'early_detection_times': [],
            'suspended_times': []
        }

    def update_at_timestamp(
        self,
        timestamp: int,
        predictions: np.ndarray,
        targets: np.ndarray,
        survival_probs: np.ndarray
    ):
        """
        Update metrics at a specific timestamp.

        Args:
            timestamp: Current timestamp index
            predictions: Binary predictions at this timestamp
            targets: Ground truth labels (event indicators)
            survival_probs: Survival probabilities at this timestamp
        """
        metrics = compute_metrics(
            predictions,
            targets,
            probabilities=1 - survival_probs  # Convert to failure probability
        )

        self.metrics_by_time[f'@{timestamp}'] = metrics

    def update_early_detection(
        self,
        detection_times: np.ndarray,
        actual_times: np.ndarray,
        event_indicators: np.ndarray
    ):
        """
        Update early detection statistics.

        Args:
            detection_times: Predicted detection timestamps
            actual_times: Actual suspended/censored times
            event_indicators: Binary indicators (1=fraudster, 0=censored)
        """
        # Only consider actual fraudsters
        fraudster_mask = event_indicators == 1

        self.early_detection_stats['num_fraudsters'] = int(fraudster_mask.sum())

        # Check how many were detected early (before actual time)
        early_detected = (detection_times[fraudster_mask] < actual_times[fraudster_mask])
        self.early_detection_stats['num_early_detected'] = int(early_detected.sum())

        # Record detection times for early detected fraudsters
        early_times = actual_times[fraudster_mask] - detection_times[fraudster_mask]
        self.early_detection_stats['early_detection_times'] = early_times[early_detected].tolist()
        self.early_detection_stats['suspended_times'] = actual_times[fraudster_mask].tolist()

    def get_metrics_at_k(self, k: int) -> Dict[str, float]:
        """Get metrics at timestamp k."""
        return self.metrics_by_time.get(f'@{k}', {})

    def get_average_metrics(self, k_values: List[int]) -> Dict[str, float]:
        """
        Get average metrics over multiple timestamps.

        Args:
            k_values: List of timestamp indices to average over

        Returns:
            Dictionary of averaged metrics
        """
        avg_metrics = {
            'precision': 0.0,
            'recall': 0.0,
            'f1': 0.0,
            'accuracy': 0.0,
            'auc': 0.0
        }

        count = 0
        for k in k_values:
            metrics = self.get_metrics_at_k(k)
            if metrics:
                for key in avg_metrics.keys():
                    if key in metrics:
                        avg_metrics[key] += metrics[key]
                count += 1

        if count > 0:
            for key in avg_metrics:
                avg_metrics[key] /= count

        return avg_metrics

    def get_early_detection_summary(self) -> Dict:
        """Get summary of early detection performance."""
        stats = self.early_detection_stats

        summary = {
            'total_fraudsters': stats['num_fraudsters'],
            'early_detected_count': stats['num_early_detected'],
            'early_detection_rate': 0.0,
            'avg_early_timestamps': 0.0,
            'median_early_timestamps': 0.0
        }

        if stats['num_fraudsters'] > 0:
            summary['early_detection_rate'] = stats['num_early_detected'] / stats['num_fraudsters']

        # Plain Python floats so the summary can be saved in a checkpoint and
        # loaded with torch.load(weights_only=True), which rejects numpy scalars
        if len(stats['early_detection_times']) > 0:
            summary['avg_early_timestamps'] = float(np.mean(stats['early_detection_times']))
            summary['median_early_timestamps'] = float(np.median(stats['early_detection_times']))

        return summary

    def format_summary(self) -> str:
        """Format all metrics as a readable report."""
        lines = ["=" * 60, "FRAUD EARLY DETECTION METRICS SUMMARY", "=" * 60]

        # Metrics at each timestamp, in numeric order
        k_values = sorted(int(key.replace('@', '')) for key in self.metrics_by_time)
        for k in k_values:
            lines.append(f"\nMetrics @{k}:")
            for metric_name, value in self.get_metrics_at_k(k).items():
                lines.append(f"  {metric_name:12s}: {value:.4f}")

        # Average metrics
        if k_values:
            lines.append(f"\nAverage Metrics (@{k_values[0]} to @{k_values[-1]}):")
            for metric_name, value in self.get_average_metrics(k_values).items():
                lines.append(f"  {metric_name:12s}: {value:.4f}")

        # Early detection summary
        lines.append("\nEarly Detection Summary:")
        for key, value in self.get_early_detection_summary().items():
            if isinstance(value, float):
                lines.append(f"  {key:25s}: {value:.4f}")
            else:
                lines.append(f"  {key:25s}: {value}")

        lines.append("=" * 60)
        return "\n".join(lines)

    def print_summary(self):
        """Print a formatted summary of all metrics."""
        print("\n" + self.format_summary() + "\n")


def predict_survival(model, data_loader, device: str = 'cpu') -> Dict[str, np.ndarray]:
    """
    Run a model over a loader of SurvivalBatch and collect the results.

    Args:
        model: Trained SAFE model
        data_loader: DataLoader yielding SurvivalBatch
        device: Device to run on

    Returns:
        Dictionary with 'survival_probs' (num_samples, seq_len), and 'events',
        'times' and 'lengths' (num_samples,)
    """
    model.eval()
    collected = {'survival_probs': [], 'events': [], 'times': [], 'lengths': []}

    with torch.no_grad():
        for batch in data_loader:
            batch = SurvivalBatch(*batch)
            _, survival_probs, _ = model(batch.sequences.to(device), batch.masks.to(device))

            collected['survival_probs'].append(survival_probs.cpu().numpy())
            collected['events'].append(batch.events.numpy())
            collected['times'].append(batch.times.numpy())
            collected['lengths'].append(batch.lengths.numpy())

    return {key: np.concatenate(values) for key, values in collected.items()}


def evaluate_at_timestamps(
    model,
    data_loader,
    timestamps: List[int],
    threshold: float = 0.5,
    device: str = 'cpu'
):
    """
    Evaluate model at multiple timestamps.

    Args:
        model: Trained SAFE model
        data_loader: DataLoader yielding SurvivalBatch
        timestamps: 0-indexed timestamps to evaluate at (reported 1-indexed)
        threshold: Classification threshold
        device: Device to run evaluation on

    Returns:
        EarlyDetectionMetrics object with results
    """
    results = predict_survival(model, data_loader, device)
    survival_probs = results['survival_probs']
    events, times, lengths = results['events'], results['times'], results['lengths']

    metrics_tracker = EarlyDetectionMetrics()

    # Evaluate each timestamp on the samples still observed at that point, so
    # padded positions of shorter sequences don't count as predictions
    for t in timestamps:
        observed = lengths > t
        if t >= survival_probs.shape[1] or not observed.any():
            continue
        probs_at_t = survival_probs[observed, t]
        metrics_tracker.update_at_timestamp(
            t + 1,  # 1-indexed
            (probs_at_t < threshold).astype(int),
            events[observed],
            probs_at_t
        )

    # Calculate early detection statistics
    # Samples that never drop below the threshold get seq_len (never detected);
    # a bare argmax would return 0 for them and count them as detected early
    below_threshold = survival_probs < threshold
    detection_times = np.where(
        below_threshold.any(axis=1),
        below_threshold.argmax(axis=1),
        survival_probs.shape[1]
    )
    metrics_tracker.update_early_detection(detection_times, times, events)

    return metrics_tracker
