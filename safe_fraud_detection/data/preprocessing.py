"""
Data preprocessing utilities for fraud detection
"""

import numpy as np
from typing import Tuple, Optional, List
from sklearn.preprocessing import StandardScaler, MinMaxScaler


class SequencePreprocessor:
    """
    Preprocessor for sequential fraud detection data.
    
    Handles normalization, padding, and feature engineering for
    time-varying covariate sequences.
    """
    
    def __init__(
        self, 
        normalize: str = 'standard',
        handle_nan: str = 'zero',
        clip_outliers: Optional[float] = None
    ):
        """
        Args:
            normalize: Normalization method ('standard', 'minmax', or None)
            handle_nan: How to handle NaN values ('zero', 'mean', 'forward_fill')
            clip_outliers: If not None, clip values beyond this many std devs
        """
        self.normalize = normalize
        self.handle_nan = handle_nan
        self.clip_outliers = clip_outliers
        
        self.scaler = None
        self.feature_means = None
        self.feature_stds = None
        self.is_fitted = False
        
    def fit(self, sequences: np.ndarray) -> 'SequencePreprocessor':
        """
        Fit the preprocessor on training data.
        
        Args:
            sequences: Array of shape (num_samples, seq_len, num_features)
            
        Returns:
            self
        """
        # Reshape to (num_samples * seq_len, num_features) for fitting
        num_samples, seq_len, num_features = sequences.shape
        sequences_flat = sequences.reshape(-1, num_features)
        
        # Remove NaN for fitting
        mask = ~np.isnan(sequences_flat).any(axis=1)
        sequences_clean = sequences_flat[mask]
        
        # Compute statistics
        self.feature_means = np.nanmean(sequences, axis=(0, 1))
        self.feature_stds = np.nanstd(sequences, axis=(0, 1))
        
        # Fit scaler
        if self.normalize == 'standard':
            self.scaler = StandardScaler()
            self.scaler.fit(sequences_clean)
        elif self.normalize == 'minmax':
            self.scaler = MinMaxScaler()
            self.scaler.fit(sequences_clean)
        
        self.is_fitted = True
        return self
    
    def transform(self, sequences: np.ndarray) -> np.ndarray:
        """
        Transform sequences using fitted preprocessor.
        
        Args:
            sequences: Array of shape (num_samples, seq_len, num_features)
            
        Returns:
            Transformed sequences
        """
        if not self.is_fitted:
            raise ValueError("Preprocessor must be fitted before transform")
        
        sequences = sequences.copy()
        num_samples, seq_len, num_features = sequences.shape
        
        # Handle NaN values
        sequences = self._handle_nan(sequences)
        
        # Clip outliers if specified
        if self.clip_outliers is not None:
            sequences = self._clip_outliers(sequences)
        
        # Normalize
        if self.scaler is not None:
            sequences_flat = sequences.reshape(-1, num_features)
            sequences_flat = self.scaler.transform(sequences_flat)
            sequences = sequences_flat.reshape(num_samples, seq_len, num_features)
        
        return sequences
    
    def fit_transform(self, sequences: np.ndarray) -> np.ndarray:
        """Fit and transform in one step."""
        self.fit(sequences)
        return self.transform(sequences)
    
    def _handle_nan(self, sequences: np.ndarray) -> np.ndarray:
        """Handle NaN values in sequences."""
        if self.handle_nan == 'zero':
            sequences = np.nan_to_num(sequences, nan=0.0)
        elif self.handle_nan == 'mean':
            for i in range(sequences.shape[2]):
                mask = np.isnan(sequences[:, :, i])
                sequences[:, :, i][mask] = self.feature_means[i]
        elif self.handle_nan == 'forward_fill':
            for i in range(sequences.shape[0]):
                for j in range(sequences.shape[2]):
                    seq = sequences[i, :, j]
                    mask = np.isnan(seq)
                    if mask.any():
                        # Forward fill
                        idx = np.where(~mask, np.arange(len(mask)), 0)
                        np.maximum.accumulate(idx, out=idx)
                        sequences[i, :, j] = seq[idx]
                        # If still NaN at start, fill with 0
                        sequences[i, :, j] = np.nan_to_num(sequences[i, :, j], nan=0.0)
        
        return sequences
    
    def _clip_outliers(self, sequences: np.ndarray) -> np.ndarray:
        """Clip outliers beyond specified standard deviations."""
        for i in range(sequences.shape[2]):
            mean = self.feature_means[i]
            std = self.feature_stds[i]
            lower = mean - self.clip_outliers * std
            upper = mean + self.clip_outliers * std
            sequences[:, :, i] = np.clip(sequences[:, :, i], lower, upper)
        
        return sequences
    
    def inverse_transform(self, sequences: np.ndarray) -> np.ndarray:
        """Inverse transform sequences back to original scale."""
        if not self.is_fitted or self.scaler is None:
            return sequences
        
        num_samples, seq_len, num_features = sequences.shape
        sequences_flat = sequences.reshape(-1, num_features)
        sequences_flat = self.scaler.inverse_transform(sequences_flat)
        return sequences_flat.reshape(num_samples, seq_len, num_features)


def pad_sequences(
    sequences: List[np.ndarray], 
    max_len: Optional[int] = None,
    padding_value: float = 0.0
) -> np.ndarray:
    """
    Pad sequences to the same length.
    
    Args:
        sequences: List of arrays with shape (seq_len, num_features)
        max_len: Maximum length to pad to (defaults to longest sequence)
        padding_value: Value to use for padding
        
    Returns:
        Padded array of shape (num_samples, max_len, num_features)
    """
    if max_len is None:
        max_len = max(len(seq) for seq in sequences)
    
    num_features = sequences[0].shape[1] if len(sequences[0].shape) > 1 else 1
    num_samples = len(sequences)
    
    padded = np.full((num_samples, max_len, num_features), padding_value, dtype=np.float32)
    
    for i, seq in enumerate(sequences):
        seq_len = len(seq)
        if seq_len > max_len:
            padded[i] = seq[:max_len]
        else:
            padded[i, :seq_len] = seq
    
    return padded


def create_train_val_test_split(
    sequences: np.ndarray,
    events: np.ndarray,
    times: np.ndarray,
    train_ratio: float = 0.7,
    val_ratio: float = 0.1,
    test_ratio: float = 0.2,
    random_seed: Optional[int] = 42
) -> Tuple[Tuple[np.ndarray, np.ndarray, np.ndarray], ...]:
    """
    Split data into train, validation, and test sets.
    
    Args:
        sequences: Feature sequences
        events: Event indicators
        times: Observed times
        train_ratio: Proportion for training
        val_ratio: Proportion for validation
        test_ratio: Proportion for testing
        random_seed: Random seed for reproducibility
        
    Returns:
        (train_data, val_data, test_data) where each is (sequences, events, times)
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
        "Ratios must sum to 1.0"
    
    num_samples = len(sequences)
    
    if random_seed is not None:
        np.random.seed(random_seed)
    
    # Shuffle indices
    indices = np.random.permutation(num_samples)
    
    # Calculate split points
    train_end = int(num_samples * train_ratio)
    val_end = train_end + int(num_samples * val_ratio)
    
    # Split indices
    train_idx = indices[:train_end]
    val_idx = indices[train_end:val_end]
    test_idx = indices[val_end:]
    
    # Create splits
    train_data = (sequences[train_idx], events[train_idx], times[train_idx])
    val_data = (sequences[val_idx], events[val_idx], times[val_idx])
    test_data = (sequences[test_idx], events[test_idx], times[test_idx])
    
    return train_data, val_data, test_data
