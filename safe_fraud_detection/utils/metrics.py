"""
Evaluation metrics for fraud detection
"""

import numpy as np
import torch
from typing import Dict, Tuple, Optional, List
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, roc_auc_score


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
        
        if len(stats['early_detection_times']) > 0:
            summary['avg_early_timestamps'] = np.mean(stats['early_detection_times'])
            summary['median_early_timestamps'] = np.median(stats['early_detection_times'])
        
        return summary
    
    def print_summary(self):
        """Print a formatted summary of all metrics."""
        print("\n" + "="*60)
        print("FRAUD EARLY DETECTION METRICS SUMMARY")
        print("="*60)
        
        # Print metrics at each timestamp
        for time_key in sorted(self.metrics_by_time.keys()):
            metrics = self.metrics_by_time[time_key]
            print(f"\nMetrics {time_key}:")
            for metric_name, value in metrics.items():
                print(f"  {metric_name:12s}: {value:.4f}")
        
        # Print average metrics
        k_values = [int(k.replace('@', '')) for k in self.metrics_by_time.keys()]
        if k_values:
            avg_metrics = self.get_average_metrics(k_values)
            print(f"\nAverage Metrics (@1 to @{max(k_values)}):")
            for metric_name, value in avg_metrics.items():
                print(f"  {metric_name:12s}: {value:.4f}")
        
        # Print early detection summary
        early_summary = self.get_early_detection_summary()
        print("\nEarly Detection Summary:")
        for key, value in early_summary.items():
            if isinstance(value, float):
                print(f"  {key:25s}: {value:.4f}")
            else:
                print(f"  {key:25s}: {value}")
        
        print("="*60 + "\n")


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
        data_loader: DataLoader with evaluation data
        timestamps: List of timestamps to evaluate at
        threshold: Classification threshold
        device: Device to run evaluation on
        
    Returns:
        EarlyDetectionMetrics object with results
    """
    model.eval()
    metrics_tracker = EarlyDetectionMetrics()
    
    all_survival_probs = []
    all_events = []
    all_times = []
    
    with torch.no_grad():
        for sequences, events, times in data_loader:
            sequences = sequences.to(device)
            _, survival_probs, _ = model(sequences)
            
            all_survival_probs.append(survival_probs.cpu().numpy())
            all_events.append(events.cpu().numpy())
            all_times.append(times.cpu().numpy())
    
    # Concatenate all batches
    all_survival_probs = np.concatenate(all_survival_probs, axis=0)
    all_events = np.concatenate(all_events, axis=0)
    all_times = np.concatenate(all_times, axis=0)
    
    # Evaluate at each timestamp
    for t in timestamps:
        if t < all_survival_probs.shape[1]:
            predictions = (all_survival_probs[:, t] < threshold).astype(int)
            metrics_tracker.update_at_timestamp(
                t + 1,  # 1-indexed
                predictions,
                all_events,
                all_survival_probs[:, t]
            )
    
    # Calculate early detection statistics
    detection_times = np.argmax(all_survival_probs < threshold, axis=1)
    metrics_tracker.update_early_detection(detection_times, all_times, all_events)
    
    return metrics_tracker
