"""
Saving and loading survival data as .npz files

Standard keys are 'sequences', 'events' and 'times'. Variable-length sequences
are stored as an object array and padded when loaded.
"""

import os
import numpy as np
from typing import List, Tuple, Union

from .preprocessing import pad_sequences


EVENT_KEYS = ('events', 'event_indicators')
TIME_KEYS = ('times', 'time_observed')


def save_npz_data(
    path: str,
    sequences: Union[List[np.ndarray], np.ndarray],
    events: np.ndarray,
    times: np.ndarray
) -> None:
    """
    Save survival data with the standard keys.

    Args:
        path: Output .npz path
        sequences: 3D array (num_samples, seq_len, num_features), or a list of
                   2D arrays (seq_len, num_features) with varying seq_len
        events: Event indicators (num_samples,)
        times: Observed times (num_samples,)
    """
    if isinstance(sequences, np.ndarray) and sequences.dtype != object:
        sequences_array = sequences
    else:
        # Assign one by one so equal-length sequences aren't broadcast into a 3D array
        sequences_array = np.empty(len(sequences), dtype=object)
        for i, seq in enumerate(sequences):
            sequences_array[i] = np.asarray(seq)

    np.savez(path, sequences=sequences_array, events=np.asarray(events), times=np.asarray(times))


def load_npz_data(
    path: str,
    padding_value: float = np.nan
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load survival data saved with save_npz_data (or np.savez).

    Accepts 'events'/'event_indicators' and 'times'/'time_observed' as key names.
    Variable-length sequences are padded to the longest sequence. Padding defaults
    to NaN so SequencePreprocessor.fit ignores it when computing statistics.

    Note: loading object arrays uses pickle, so only load files you trust.

    Args:
        path: Path to .npz file
        padding_value: Value used to pad variable-length sequences

    Returns:
        sequences (num_samples, max_seq_len, num_features), events, times
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Data file not found: {path}")

    data = np.load(path, allow_pickle=True)

    if 'sequences' not in data.files:
        raise KeyError(f"'sequences' not found in {path}. Found keys: {data.files}")
    sequences = data['sequences']
    events = _get_first_key(data, EVENT_KEYS, path)
    times = _get_first_key(data, TIME_KEYS, path)

    if sequences.dtype == object:
        sequences = pad_sequences(list(sequences), padding_value=padding_value)

    return sequences, events, times


def _get_first_key(data, keys: Tuple[str, ...], path: str) -> np.ndarray:
    """Return the array stored under the first key present."""
    for key in keys:
        if key in data.files:
            return data[key]
    raise KeyError(f"None of {list(keys)} found in {path}. Found keys: {data.files}")
