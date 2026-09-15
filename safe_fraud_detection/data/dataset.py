"""
Dataset classes for fraud detection with survival analysis
"""

import torch
from torch.utils.data import Dataset
import numpy as np
from typing import Tuple, Optional, List, Union


def validate_and_convert_sequences(
    sequences: Union[List[np.ndarray], np.ndarray],
    expected_num_samples: Optional[int] = None
) -> List[np.ndarray]:
    """
    Validate and convert sequences to the correct format for CreditCardDataset.
    
    This function helps ensure your data is in the correct format before creating
    a dataset. It handles common mistakes like passing a single numpy array instead
    of a list of arrays.
    
    Args:
        sequences: Either:
            - List of 2D numpy arrays, each with shape (seq_len, num_features)
            - 3D numpy array with shape (num_samples, seq_len, num_features)
        expected_num_samples: Optional expected number of samples for validation
    
    Returns:
        List of 2D numpy arrays, each with shape (seq_len, num_features)
    
    Raises:
        ValueError: If sequences are in an invalid format
    """
    # Convert numpy array to list if needed
    if isinstance(sequences, np.ndarray):
        if sequences.ndim == 3:
            # 3D array: (num_samples, seq_len, num_features)
            num_samples = len(sequences)
            sequences_list = [sequences[i] for i in range(num_samples)]
            print(f"Converted 3D numpy array (shape {sequences.shape}) to list of {num_samples} sequences")
        elif sequences.ndim == 2:
            # 2D array: treat as single sequence
            sequences_list = [sequences]
            print(f"Converted 2D numpy array (shape {sequences.shape}) to list with 1 sequence")
        else:
            raise ValueError(
                f"Invalid numpy array shape. Expected 2D (seq_len, num_features) or "
                f"3D (num_samples, seq_len, num_features), got {sequences.ndim}D with shape {sequences.shape}"
            )
    elif isinstance(sequences, (list, tuple)):
        sequences_list = list(sequences)
    else:
        raise TypeError(
            f"sequences must be a list of numpy arrays or a numpy array. "
            f"Got {type(sequences)}"
        )
    
    # Validate each sequence
    num_features = None
    for i, seq in enumerate(sequences_list):
        seq = np.asarray(seq)
        
        if seq.ndim == 1:
            # 1D array: assume it's a single timestep with multiple features
            seq = seq.reshape(1, -1)
            sequences_list[i] = seq
        elif seq.ndim != 2:
            raise ValueError(
                f"Sequence {i} must be 2D (seq_len, num_features). "
                f"Got shape {seq.shape}"
            )
        
        # Check feature dimension consistency
        if num_features is None:
            num_features = seq.shape[1]
        elif seq.shape[1] != num_features:
            raise ValueError(
                f"All sequences must have the same number of features. "
                f"Sequence 0 has {num_features} features, but sequence {i} has {seq.shape[1]}"
            )
    
    # Validate number of samples if provided
    if expected_num_samples is not None and len(sequences_list) != expected_num_samples:
        raise ValueError(
            f"Number of sequences ({len(sequences_list)}) does not match "
            f"expected number of samples ({expected_num_samples})"
        )
    
    print(f"✓ Validated {len(sequences_list)} sequences, each with {num_features} features")
    return sequences_list


class FraudDataset(Dataset):
    """
    Dataset for fraud detection with time-varying features.
    
    Args:
        sequences: List or array of feature sequences (num_samples, seq_len, num_features)
        event_indicators: Binary indicators (1=fraudster, 0=censored)
        time_observed: Last observed time for each sample
        transform: Optional transform to apply to sequences
    """
    
    def __init__(
        self,
        sequences: np.ndarray,
        event_indicators: np.ndarray,
        time_observed: np.ndarray,
        transform=None
    ):
        """Initialize the dataset."""
        self.sequences = sequences
        self.event_indicators = event_indicators
        self.time_observed = time_observed
        self.transform = transform
        
        # Validate inputs
        assert len(sequences) == len(event_indicators) == len(time_observed), \
            "All inputs must have the same length"
        
        self.num_samples = len(sequences)
        
    def __len__(self) -> int:
        """Return the number of samples."""
        return self.num_samples
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get a single sample.
        
        Returns:
            sequence: Feature sequence tensor
            event_indicator: Binary event indicator
            time_observed: Last observed time
        """
        sequence = self.sequences[idx]
        event = self.event_indicators[idx]
        time = self.time_observed[idx]
        
        if self.transform:
            sequence = self.transform(sequence)
        
        # Convert to tensors
        sequence = torch.FloatTensor(sequence)
        event = torch.FloatTensor([event])
        time = torch.LongTensor([time])
        
        return sequence, event, time
    
    def get_batch(self, indices: List[int]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Get a batch of samples by indices."""
        sequences = []
        events = []
        times = []
        
        for idx in indices:
            seq, event, time = self.__getitem__(idx)
            sequences.append(seq)
            events.append(event)
            times.append(time)
        
        sequences = torch.stack(sequences)
        events = torch.stack(events).squeeze(-1)
        times = torch.stack(times).squeeze(-1)
        
        return sequences, events, times


class SurvivalDataset(Dataset):
    """
    Generic survival analysis dataset.
    
    This class handles survival data with time-varying covariates,
    event indicators, and observed times.
    """
    
    def __init__(
        self,
        features: np.ndarray,
        events: np.ndarray,
        times: np.ndarray,
        feature_names: Optional[List[str]] = None
    ):
        """
        Args:
            features: Feature array (num_samples, seq_len, num_features)
            events: Event indicators (num_samples,)
            times: Observed times (num_samples,)
            feature_names: Optional list of feature names
        """
        self.features = torch.FloatTensor(features)
        self.events = torch.FloatTensor(events)
        self.times = torch.LongTensor(times)
        self.feature_names = feature_names
        
        assert len(features) == len(events) == len(times), \
            "All inputs must have the same length"
        
    def __len__(self) -> int:
        return len(self.features)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.features[idx], self.events[idx], self.times[idx]
    
    @property
    def num_features(self) -> int:
        """Return number of features per timestamp."""
        return self.features.shape[2]
    
    @property
    def max_seq_len(self) -> int:
        """Return maximum sequence length."""
        return self.features.shape[1]
    
    def get_statistics(self) -> dict:
        """Get dataset statistics."""
        return {
            'num_samples': len(self),
            'num_features': self.num_features,
            'max_seq_len': self.max_seq_len,
            'num_events': int(self.events.sum().item()),
            'num_censored': int((1 - self.events).sum().item()),
            'event_rate': float(self.events.mean().item()),
            'mean_observed_time': float(self.times.float().mean().item()),
            'std_observed_time': float(self.times.float().std().item())
        }


class TwitterDataset(FraudDataset):
    """
    Twitter fraud detection dataset.
    
    Features per timestamp:
    1. Number of followers (change)
    2. Number of followees (change)
    3. Number of tweets (change)
    4. Number of liked tweets (change)
    5. Number of public lists (change)
    """
    
    feature_names = [
        'followers_change',
        'followees_change',
        'tweets_change',
        'likes_change',
        'lists_change'
    ]
    
    def __init__(self, sequences, event_indicators, time_observed, transform=None):
        super().__init__(sequences, event_indicators, time_observed, transform)


class WikiDataset(FraudDataset):
    """
    Wikipedia vandalism detection dataset.
    
    Features per timestamp (edit):
    1. Edit on meta-page
    2. Empty category
    3. Re-edit < 1 min
    4. Re-edit < 3 min
    5. Re-edit < 15 min
    6. Page edited before
    7. Consecutive same page
    8. Common category with previous
    """
    
    feature_names = [
        'is_meta_page',
        'empty_category',
        'reedit_1min',
        'reedit_3min',
        'reedit_15min',
        'page_edited_before',
        'consecutive_same_page',
        'common_category'
    ]
    
    def __init__(self, sequences, event_indicators, time_observed, transform=None):
        super().__init__(sequences, event_indicators, time_observed, transform)


class CreditCardDataset(Dataset):
    """
    Credit Card Fraud Detection Dataset with Variable-Length Sequences.
    
    This dataset handles variable-length sequences of weekly transaction aggregates.
    Cards can be closed at different times due to fraud or other reasons.
    
    Features per week:
    1. Transaction count
    2. Total amount
    3. Average amount
    4. Max amount
    5. Std amount
    6. Foreign transaction ratio
    7. Online transaction ratio
    8. Weekend transaction ratio
    9. Night transaction ratio
    10. Velocity (change in transaction count)
    
    Args:
        sequences: List of variable-length sequences, each (seq_len, num_features)
        event_indicators: Binary indicators (1=fraud, 0=censored)
        time_observed: Last observed time (weeks) for each card
        max_seq_len: Maximum sequence length for padding (if None, uses max in data)
        transform: Optional transform to apply to sequences
    """
    
    feature_names = [
        'txn_count',
        'total_amount',
        'avg_amount',
        'max_amount',
        'std_amount',
        'foreign_ratio',
        'online_ratio',
        'weekend_ratio',
        'night_ratio',
        'velocity'
    ]
    
    def __init__(
        self,
        sequences: Union[List[np.ndarray], np.ndarray],
        event_indicators: np.ndarray,
        time_observed: np.ndarray,
        max_seq_len: Optional[int] = None,
        transform=None
    ):
        """Initialize the dataset."""
        # Validate and convert sequences to list format
        sequences = validate_and_convert_sequences(sequences)
        
        # Convert to numpy arrays if needed
        event_indicators = np.asarray(event_indicators)
        time_observed = np.asarray(time_observed)
        
        # Validate inputs
        if len(sequences) != len(event_indicators) or len(sequences) != len(time_observed):
            raise ValueError(
                f"All inputs must have the same length. "
                f"Got sequences: {len(sequences)}, "
                f"events: {len(event_indicators)}, "
                f"times: {len(time_observed)}"
            )
        
        self.raw_sequences = sequences
        self.event_indicators = event_indicators
        self.time_observed = time_observed
        self.transform = transform
        
        self.num_samples = len(sequences)
        
        # Store actual lengths before padding
        self.actual_lengths = np.array([len(seq) for seq in sequences])

        # The loss sums hazards up to time_observed, so it must not point past
        # the real data into padding
        invalid = (self.time_observed < 1) | (self.time_observed > self.actual_lengths)
        if invalid.any():
            first = int(np.flatnonzero(invalid)[0])
            raise ValueError(
                f"time_observed must be between 1 and the sequence length. "
                f"{int(invalid.sum())} samples violate this; first is index {first} "
                f"with time_observed={self.time_observed[first]} and "
                f"length={self.actual_lengths[first]}"
            )

        # Determine max sequence length
        if max_seq_len is None:
            self.max_seq_len = int(self.actual_lengths.max())
        else:
            self.max_seq_len = max_seq_len
        
        # Pad sequences to max length
        self.sequences = self._pad_sequences(sequences)
        
        # Create masks (1 for real data, 0 for padding)
        self.masks = self._create_masks()
        
        self.num_features = self.sequences.shape[2]
    
    def _pad_sequences(self, sequences: List[np.ndarray]) -> np.ndarray:
        """
        Pad sequences to max_seq_len.
        
        Args:
            sequences: List of variable-length sequences
            
        Returns:
            Padded array of shape (num_samples, max_seq_len, num_features)
        """
        num_features = sequences[0].shape[1]
        padded = np.zeros((self.num_samples, self.max_seq_len, num_features), dtype=np.float32)
        
        for i, seq in enumerate(sequences):
            seq_len = min(len(seq), self.max_seq_len)
            padded[i, :seq_len, :] = seq[:seq_len]
        
        return padded
    
    def _create_masks(self) -> np.ndarray:
        """
        Create masks indicating real data vs padding.
        
        Returns:
            Mask array of shape (num_samples, max_seq_len)
        """
        masks = np.zeros((self.num_samples, self.max_seq_len), dtype=np.float32)
        
        for i, length in enumerate(self.actual_lengths):
            seq_len = min(length, self.max_seq_len)
            masks[i, :seq_len] = 1.0
        
        return masks
    
    def __len__(self) -> int:
        """Return the number of samples."""
        return self.num_samples
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get a single sample.
        
        Returns:
            sequence: Padded feature sequence tensor (max_seq_len, num_features)
            mask: Mask indicating valid positions (max_seq_len,)
            event_indicator: Binary event indicator
            time_observed: Last observed time
            actual_length: Actual sequence length before padding
        """
        sequence = self.sequences[idx]
        mask = self.masks[idx]
        event = self.event_indicators[idx]
        time = self.time_observed[idx]
        length = self.actual_lengths[idx]
        
        if self.transform:
            sequence = self.transform(sequence)
        
        # Convert to tensors
        sequence = torch.FloatTensor(sequence)
        mask = torch.FloatTensor(mask)
        event = torch.FloatTensor([event])
        time = torch.LongTensor([time])
        length = torch.LongTensor([length])
        
        return sequence, mask, event, time, length
    
    def get_statistics(self) -> dict:
        """Get dataset statistics."""
        fraud_mask = self.event_indicators == 1
        
        return {
            'num_samples': self.num_samples,
            'num_features': self.num_features,
            'max_seq_len': self.max_seq_len,
            'avg_seq_len': float(self.actual_lengths.mean()),
            'std_seq_len': float(self.actual_lengths.std()),
            'min_seq_len': int(self.actual_lengths.min()),
            'num_fraud': int(fraud_mask.sum()),
            'num_censored': int((~fraud_mask).sum()),
            'fraud_rate': float(fraud_mask.mean()),
            'avg_time_to_fraud': float(self.time_observed[fraud_mask].mean()) if fraud_mask.any() else 0,
            'avg_time_censored': float(self.time_observed[~fraud_mask].mean()) if (~fraud_mask).any() else 0,
        }
