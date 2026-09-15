"""
Dataset classes for fraud detection with survival analysis

All datasets pad sequences to a common length and return SurvivalBatch samples,
so fixed-length and variable-length data share one training and evaluation path.
"""

import logging
from typing import List, NamedTuple, Optional, Union

import numpy as np
import torch
from torch.utils.data import Dataset

from .preprocessing import pad_sequences


logger = logging.getLogger(__name__)


class SurvivalBatch(NamedTuple):
    """
    One sample from a SurvivalDataset, or a batch of them.

    DataLoader's default collation keeps this structure, so a batch has the
    same fields with a leading batch dimension.
    """
    sequences: torch.Tensor  # (seq_len, num_features), padded
    masks: torch.Tensor      # (seq_len,), 1 for real timesteps, 0 for padding
    events: torch.Tensor     # 1 = fraud (event), 0 = censored
    times: torch.Tensor      # Last observed time, in [1, length]
    lengths: torch.Tensor    # Number of real timesteps


def validate_and_convert_sequences(
    sequences: Union[List[np.ndarray], np.ndarray],
    expected_num_samples: Optional[int] = None
) -> List[np.ndarray]:
    """
    Validate and convert sequences to a list of 2D arrays.

    Handles common mistakes like passing a single numpy array instead of a
    list of arrays.

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
    if isinstance(sequences, np.ndarray) and sequences.dtype != object:
        if sequences.ndim == 3:
            # 3D array: (num_samples, seq_len, num_features)
            sequences_list = [sequences[i] for i in range(len(sequences))]
            logger.debug(f"Converted 3D numpy array (shape {sequences.shape}) to list of sequences")
        elif sequences.ndim == 2:
            # 2D array: treat as single sequence
            sequences_list = [sequences]
            logger.debug(f"Converted 2D numpy array (shape {sequences.shape}) to list with 1 sequence")
        else:
            raise ValueError(
                f"Invalid numpy array shape. Expected 2D (seq_len, num_features) or "
                f"3D (num_samples, seq_len, num_features), got {sequences.ndim}D with shape {sequences.shape}"
            )
    elif isinstance(sequences, (list, tuple, np.ndarray)):
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
        elif seq.ndim != 2:
            raise ValueError(
                f"Sequence {i} must be 2D (seq_len, num_features). "
                f"Got shape {seq.shape}"
            )
        sequences_list[i] = seq

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

    logger.debug(f"Validated {len(sequences_list)} sequences, each with {num_features} features")
    return sequences_list


class SurvivalDataset(Dataset):
    """
    Survival dataset with time-varying features of fixed or variable length.

    Sequences are padded to max_seq_len, with a mask marking real timesteps.

    Args:
        sequences: 3D array (num_samples, seq_len, num_features), or a list of
                   2D arrays (seq_len, num_features) with varying seq_len
        events: Event indicators (1=fraudster, 0=censored)
        times: Last observed time for each sample, between 1 and its length
        max_seq_len: Length to pad or truncate to (default: longest sequence)
        feature_names: Optional names for the features
        transform: Optional transform applied to each padded sequence
    """

    # Subclasses set default names for their standard feature layout
    default_feature_names: Optional[List[str]] = None

    def __init__(
        self,
        sequences: Union[List[np.ndarray], np.ndarray],
        events: np.ndarray,
        times: np.ndarray,
        max_seq_len: Optional[int] = None,
        feature_names: Optional[List[str]] = None,
        transform=None
    ):
        """Initialize the dataset."""
        sequences = validate_and_convert_sequences(sequences)
        events = np.asarray(events, dtype=np.float32)
        times = np.asarray(times, dtype=np.int64)

        if len(sequences) == 0:
            raise ValueError("SurvivalDataset needs at least one sequence")
        if not len(sequences) == len(events) == len(times):
            raise ValueError(
                f"All inputs must have the same length. "
                f"Got sequences: {len(sequences)}, events: {len(events)}, times: {len(times)}"
            )

        lengths = np.array([len(seq) for seq in sequences], dtype=np.int64)

        # The loss sums hazards up to times, so they must not point past the
        # real data into padding
        invalid = (times < 1) | (times > lengths)
        if invalid.any():
            first = int(np.flatnonzero(invalid)[0])
            raise ValueError(
                f"time_observed must be between 1 and the sequence length. "
                f"{int(invalid.sum())} samples violate this; first is index {first} "
                f"with time_observed={times[first]} and length={lengths[first]}"
            )

        self.max_seq_len = int(lengths.max()) if max_seq_len is None else int(max_seq_len)
        self.sequences = pad_sequences(sequences, max_len=self.max_seq_len)
        self.lengths = np.minimum(lengths, self.max_seq_len)
        self.masks = (
            np.arange(self.max_seq_len)[np.newaxis, :] < self.lengths[:, np.newaxis]
        ).astype(np.float32)
        self.events = events
        self.times = times
        self.transform = transform
        self.num_features = self.sequences.shape[2]
        self.feature_names = self._resolve_feature_names(feature_names)

    def _resolve_feature_names(self, feature_names: Optional[List[str]]) -> Optional[List[str]]:
        """Use explicit names if given, else class defaults when they fit the data."""
        if feature_names is not None:
            if len(feature_names) != self.num_features:
                raise ValueError(
                    f"Got {len(feature_names)} feature names for {self.num_features} features"
                )
            return list(feature_names)

        # Class defaults only describe their standard layout; skip them for other data
        defaults = self.default_feature_names
        if defaults is not None and len(defaults) == self.num_features:
            return list(defaults)
        return None

    def __len__(self) -> int:
        """Return the number of samples."""
        return len(self.events)

    def __getitem__(self, idx: int) -> SurvivalBatch:
        """Get a single padded sample with its mask and labels."""
        sequence = self.sequences[idx]
        if self.transform:
            sequence = self.transform(sequence)

        return SurvivalBatch(
            sequences=torch.as_tensor(sequence, dtype=torch.float32),
            masks=torch.as_tensor(self.masks[idx]),
            events=torch.tensor(self.events[idx], dtype=torch.float32),
            times=torch.tensor(self.times[idx], dtype=torch.long),
            lengths=torch.tensor(self.lengths[idx], dtype=torch.long)
        )

    def get_statistics(self) -> dict:
        """Get dataset statistics."""
        event_mask = self.events == 1

        return {
            'num_samples': len(self),
            'num_features': self.num_features,
            'max_seq_len': self.max_seq_len,
            'avg_seq_len': float(self.lengths.mean()),
            'std_seq_len': float(self.lengths.std()),
            'min_seq_len': int(self.lengths.min()),
            'num_events': int(event_mask.sum()),
            'num_censored': int((~event_mask).sum()),
            'event_rate': float(event_mask.mean()),
            'avg_time_event': float(self.times[event_mask].mean()) if event_mask.any() else 0.0,
            'avg_time_censored': float(self.times[~event_mask].mean()) if (~event_mask).any() else 0.0,
        }


class TwitterDataset(SurvivalDataset):
    """
    Twitter fraud detection dataset.

    Features per timestamp:
    1. Number of followers (change)
    2. Number of followees (change)
    3. Number of tweets (change)
    4. Number of liked tweets (change)
    5. Number of public lists (change)
    """

    default_feature_names = [
        'followers_change',
        'followees_change',
        'tweets_change',
        'likes_change',
        'lists_change'
    ]


class WikiDataset(SurvivalDataset):
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

    default_feature_names = [
        'is_meta_page',
        'empty_category',
        'reedit_1min',
        'reedit_3min',
        'reedit_15min',
        'page_edited_before',
        'consecutive_same_page',
        'common_category'
    ]


class CreditCardDataset(SurvivalDataset):
    """
    Credit card fraud detection dataset of weekly transaction aggregates.

    Cards can be closed at different times due to fraud or other reasons, so
    sequences usually have variable length.

    Default features per week (as produced by CreditCardSimulator):
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
    """

    default_feature_names = [
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
