"""
Data preprocessing utilities for fraud detection

Functions accept either a 3D array (num_samples, seq_len, num_features) or a
list of 2D arrays (seq_len, num_features) with varying seq_len.
"""

import numpy as np
from typing import Tuple, Optional, List, Union
from sklearn.preprocessing import StandardScaler, MinMaxScaler


Sequences = Union[np.ndarray, List[np.ndarray]]

# Fitted scaler attributes needed to rebuild a scaler without pickling it
_SCALER_ATTRS = {
    'standard': ('mean_', 'var_', 'scale_'),
    'minmax': ('min_', 'scale_', 'data_min_', 'data_max_', 'data_range_'),
}


def _is_3d_array(sequences) -> bool:
    return isinstance(sequences, np.ndarray) and sequences.dtype != object and sequences.ndim == 3


def _stack_timesteps(sequences: Sequences) -> np.ndarray:
    """Return all timesteps as rows of shape (total_timesteps, num_features)."""
    if _is_3d_array(sequences):
        return sequences.reshape(-1, sequences.shape[2]).astype(np.float64)
    return np.concatenate([np.asarray(seq, dtype=np.float64) for seq in sequences])


class SequencePreprocessor:
    """
    Preprocessor for sequential fraud detection data.

    Handles normalization, NaN handling and outlier clipping for
    time-varying covariate sequences.
    """

    def __init__(
        self,
        normalize: Optional[str] = 'standard',
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

    def fit(self, sequences: Sequences) -> 'SequencePreprocessor':
        """
        Fit the preprocessor on training data.

        Statistics use every timestep given, so pass unpadded variable-length
        sequences as a list to keep padding out of them.

        Args:
            sequences: 3D array or list of 2D arrays

        Returns:
            self
        """
        rows = _stack_timesteps(sequences)

        # Compute statistics
        self.feature_means = np.nanmean(rows, axis=0)
        self.feature_stds = np.nanstd(rows, axis=0)

        # Fit scaler on rows without NaN
        clean_rows = rows[~np.isnan(rows).any(axis=1)]
        if self.normalize == 'standard':
            self.scaler = StandardScaler().fit(clean_rows)
        elif self.normalize == 'minmax':
            self.scaler = MinMaxScaler().fit(clean_rows)

        self.is_fitted = True
        return self

    def transform(self, sequences: Sequences) -> Sequences:
        """
        Transform sequences using fitted preprocessor.

        Args:
            sequences: 3D array or list of 2D arrays

        Returns:
            Transformed sequences in the same form as the input
        """
        if not self.is_fitted:
            raise ValueError("Preprocessor must be fitted before transform")

        if _is_3d_array(sequences):
            sequences = np.array(sequences, dtype=np.float64)
            return self._clip_and_scale(self._handle_nan(sequences))

        if len(sequences) == 0:
            return []

        # Handle NaNs per sequence so forward fill never crosses sequences, then
        # clip and scale all timesteps in one call
        filled = [
            self._handle_nan(np.array(seq, dtype=np.float64)[np.newaxis])[0]
            for seq in sequences
        ]
        lengths = [len(seq) for seq in filled]
        rows = self._clip_and_scale(np.concatenate(filled)[np.newaxis])[0]
        return np.split(rows, np.cumsum(lengths)[:-1])

    def fit_transform(self, sequences: Sequences) -> Sequences:
        """Fit and transform in one step."""
        self.fit(sequences)
        return self.transform(sequences)

    def _clip_and_scale(self, sequences: np.ndarray) -> np.ndarray:
        """Clip outliers and apply the scaler to a 3D array."""
        if self.clip_outliers is not None:
            sequences = self._clip_outliers(sequences)

        if self.scaler is not None:
            num_samples, seq_len, num_features = sequences.shape
            sequences_flat = self.scaler.transform(sequences.reshape(-1, num_features))
            sequences = sequences_flat.reshape(num_samples, seq_len, num_features)

        return sequences

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
        """Inverse transform a 3D array back to original scale."""
        if not self.is_fitted or self.scaler is None:
            return sequences

        num_samples, seq_len, num_features = sequences.shape
        sequences_flat = sequences.reshape(-1, num_features)
        sequences_flat = self.scaler.inverse_transform(sequences_flat)
        return sequences_flat.reshape(num_samples, seq_len, num_features)

    def state_dict(self) -> dict:
        """
        Return the fitted state as plain Python values.

        Safe to store in a checkpoint loaded with torch.load(weights_only=True).
        """
        if not self.is_fitted:
            raise ValueError("Preprocessor must be fitted before saving its state")

        scaler_state = None
        if self.scaler is not None:
            scaler_state = {
                attr: getattr(self.scaler, attr).tolist()
                for attr in _SCALER_ATTRS[self.normalize]
            }

        return {
            'normalize': self.normalize,
            'handle_nan': self.handle_nan,
            'clip_outliers': self.clip_outliers,
            'feature_means': self.feature_means.tolist(),
            'feature_stds': self.feature_stds.tolist(),
            'scaler': scaler_state,
        }

    @classmethod
    def from_state_dict(cls, state: dict) -> 'SequencePreprocessor':
        """Rebuild a fitted preprocessor from state_dict() output."""
        preprocessor = cls(
            normalize=state['normalize'],
            handle_nan=state['handle_nan'],
            clip_outliers=state['clip_outliers']
        )
        preprocessor.feature_means = np.array(state['feature_means'])
        preprocessor.feature_stds = np.array(state['feature_stds'])

        if state['scaler'] is not None:
            scaler = StandardScaler() if state['normalize'] == 'standard' else MinMaxScaler()
            for attr, value in state['scaler'].items():
                setattr(scaler, attr, np.array(value))
            scaler.n_features_in_ = len(preprocessor.feature_means)
            preprocessor.scaler = scaler

        preprocessor.is_fitted = True
        return preprocessor


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
    sequences: Sequences,
    events: np.ndarray,
    times: np.ndarray,
    train_ratio: float = 0.7,
    val_ratio: float = 0.1,
    test_ratio: float = 0.2,
    random_seed: Optional[int] = 42
) -> Tuple[Tuple[Sequences, np.ndarray, np.ndarray], ...]:
    """
    Split data into train, validation, and test sets.

    Args:
        sequences: Feature sequences (3D array or list of 2D arrays)
        events: Event indicators
        times: Observed times
        train_ratio: Proportion for training
        val_ratio: Proportion for validation
        test_ratio: Proportion for testing
        random_seed: Random seed for reproducibility

    Returns:
        (train_data, val_data, test_data) where each is (sequences, events, times)
        and sequences keep the input form
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
        "Ratios must sum to 1.0"

    events = np.asarray(events)
    times = np.asarray(times)
    num_samples = len(events)

    if random_seed is not None:
        np.random.seed(random_seed)

    # Shuffle indices
    indices = np.random.permutation(num_samples)

    # Calculate split points
    train_end = int(num_samples * train_ratio)
    val_end = train_end + int(num_samples * val_ratio)

    def take(split_indices):
        if isinstance(sequences, np.ndarray):
            split_sequences = sequences[split_indices]
        else:
            split_sequences = [sequences[i] for i in split_indices]
        return split_sequences, events[split_indices], times[split_indices]

    return take(indices[:train_end]), take(indices[train_end:val_end]), take(indices[val_end:])
