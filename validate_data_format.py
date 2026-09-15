"""
Utility script to validate your data format before creating a DataLoader.

This script helps diagnose common data format issues that cause the error:
"RuntimeError: stack expects each tensor to be equal size"

Run this script with your data to check if it's in the correct format.
"""

import numpy as np
import torch
from safe_fraud_detection.data.dataset import CreditCardDataset, validate_and_convert_sequences
from torch.utils.data import DataLoader


def check_data_format(sequences, event_indicators, time_observed):
    """
    Check if data is in the correct format for CreditCardDataset.
    
    Args:
        sequences: Your sequences (list of arrays or numpy array)
        event_indicators: Binary event indicators (1D array)
        time_observed: Observed times (1D array)
    """
    print("=" * 60)
    print("DATA FORMAT VALIDATION")
    print("=" * 60)
    
    # Check sequences
    print("\n1. Checking sequences...")
    print(f"   Type: {type(sequences)}")
    
    if isinstance(sequences, np.ndarray):
        print(f"   Shape: {sequences.shape}")
        print(f"   Dimensions: {sequences.ndim}D")
        
        if sequences.ndim == 3:
            print("   ⚠️  WARNING: You passed a 3D numpy array.")
            print("   This will be automatically converted, but it's better to pass a list.")
            print("   Expected: List of 2D arrays, each with shape (seq_len, num_features)")
        elif sequences.ndim == 2:
            print("   ⚠️  WARNING: You passed a 2D numpy array (single sequence).")
            print("   Expected: List of 2D arrays")
        else:
            print("   ❌ ERROR: Invalid shape for sequences")
            return False
    elif isinstance(sequences, (list, tuple)):
        print(f"   Number of sequences: {len(sequences)}")
        if len(sequences) > 0:
            print(f"   First sequence type: {type(sequences[0])}")
            if isinstance(sequences[0], np.ndarray):
                print(f"   First sequence shape: {sequences[0].shape}")
                print(f"   First sequence dimensions: {sequences[0].ndim}D")
                
                if sequences[0].ndim != 2:
                    print("   ❌ ERROR: Each sequence must be 2D (seq_len, num_features)")
                    return False
            else:
                print(f"   ❌ ERROR: Sequences must contain numpy arrays, got {type(sequences[0])}")
                return False
    else:
        print(f"   ❌ ERROR: sequences must be a list or numpy array, got {type(sequences)}")
        return False
    
    # Check event_indicators
    print("\n2. Checking event_indicators...")
    event_indicators = np.asarray(event_indicators)
    print(f"   Shape: {event_indicators.shape}")
    print(f"   Dimensions: {event_indicators.ndim}D")
    print(f"   Dtype: {event_indicators.dtype}")
    print(f"   Value range: [{event_indicators.min()}, {event_indicators.max()}]")
    
    if event_indicators.ndim != 1:
        print("   ❌ ERROR: event_indicators must be 1D")
        return False
    
    if not np.all(np.isin(event_indicators, [0, 1])):
        print("   ⚠️  WARNING: event_indicators should contain only 0 and 1")
    
    # Check time_observed
    print("\n3. Checking time_observed...")
    time_observed = np.asarray(time_observed)
    print(f"   Shape: {time_observed.shape}")
    print(f"   Dimensions: {time_observed.ndim}D")
    print(f"   Dtype: {time_observed.dtype}")
    print(f"   Value range: [{time_observed.min()}, {time_observed.max()}]")
    
    if time_observed.ndim != 1:
        print("   ❌ ERROR: time_observed must be 1D")
        return False
    
    # Check lengths match
    print("\n4. Checking lengths match...")
    if isinstance(sequences, np.ndarray):
        if sequences.ndim == 3:
            num_samples = len(sequences)
        else:
            num_samples = 1
    else:
        num_samples = len(sequences)
    
    print(f"   Number of sequences: {num_samples}")
    print(f"   Number of events: {len(event_indicators)}")
    print(f"   Number of times: {len(time_observed)}")
    
    if num_samples != len(event_indicators) or num_samples != len(time_observed):
        print("   ❌ ERROR: All inputs must have the same length")
        return False
    
    print("\n5. Testing dataset creation...")
    try:
        # Convert sequences to list format
        sequences_list = validate_and_convert_sequences(sequences, expected_num_samples=num_samples)
        
        # Create dataset
        dataset = CreditCardDataset(
            sequences_list,
            event_indicators,
            time_observed
        )
        print(f"   ✓ Dataset created successfully with {len(dataset)} samples")
        
        # Test __getitem__
        print("\n6. Testing __getitem__...")
        sample = dataset[0]
        sequence, mask, event, time, length = sample
        
        print(f"   Sequence shape: {sequence.shape}")
        print(f"   Mask shape: {mask.shape}")
        print(f"   Event shape: {event.shape}, value: {event.item()}")
        print(f"   Time shape: {time.shape}, value: {time.item()}")
        print(f"   Length shape: {length.shape}, value: {length.item()}")
        
        # Validate shapes
        if sequence.dim() != 2:
            print(f"   ❌ ERROR: Sequence should be 2D, got {sequence.dim()}D")
            return False
        if mask.dim() != 1:
            print(f"   ❌ ERROR: Mask should be 1D, got {mask.dim()}D")
            return False
        if event.dim() > 1 or time.dim() > 1 or length.dim() > 1:
            print(f"   ❌ ERROR: Event, time, and length should be scalars or 1D")
            return False
        
        print("   ✓ All shapes are correct")
        
        # Test DataLoader
        print("\n7. Testing DataLoader...")
        def collate_fn(batch):
            sequences, masks, events, times, lengths = zip(*batch)
            return (
                torch.stack(sequences),
                torch.stack(masks),
                torch.cat(events) if events[0].dim() > 0 else torch.stack(events),
                torch.cat(times) if times[0].dim() > 0 else torch.stack(times),
                torch.cat(lengths) if lengths[0].dim() > 0 else torch.stack(lengths)
            )
        
        dataloader = DataLoader(dataset, batch_size=2, shuffle=False, collate_fn=collate_fn)
        
        # Get one batch
        batch = next(iter(dataloader))
        sequences_batch, masks_batch, events_batch, times_batch, lengths_batch = batch
        
        print(f"   Batch sequences shape: {sequences_batch.shape}")
        print(f"   Batch masks shape: {masks_batch.shape}")
        print(f"   Batch events shape: {events_batch.shape}")
        print(f"   Batch times shape: {times_batch.shape}")
        print(f"   Batch lengths shape: {lengths_batch.shape}")
        print("   ✓ DataLoader works correctly")
        
        print("\n" + "=" * 60)
        print("✓ ALL CHECKS PASSED! Your data format is correct.")
        print("=" * 60)
        return True
        
    except Exception as e:
        print(f"   ❌ ERROR: {type(e).__name__}: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def example_correct_format():
    """Show example of correct data format."""
    print("\n" + "=" * 60)
    print("EXAMPLE: CORRECT DATA FORMAT")
    print("=" * 60)
    
    # Create example data
    num_samples = 100
    num_features = 10
    
    # CORRECT: List of variable-length sequences
    sequences = []
    for i in range(num_samples):
        seq_len = np.random.randint(10, 50)  # Variable length
        seq = np.random.randn(seq_len, num_features)
        sequences.append(seq)
    
    event_indicators = np.random.randint(0, 2, size=num_samples)
    # time_observed must be between 1 and each sequence's length
    time_observed = np.array([len(seq) for seq in sequences])
    
    print("\n✓ CORRECT FORMAT:")
    print(f"  sequences: List of {len(sequences)} arrays")
    print(f"    - Each array has shape (seq_len, {num_features})")
    print(f"    - Example: sequences[0].shape = {sequences[0].shape}")
    print(f"  event_indicators: 1D array with shape {event_indicators.shape}")
    print(f"  time_observed: 1D array with shape {time_observed.shape}")
    
    print("\n" + "=" * 60)
    print("INCORRECT FORMATS TO AVOID:")
    print("=" * 60)
    
    print("\n❌ WRONG: Single 3D numpy array")
    print("  sequences = np.random.randn(100, 40, 10)  # Shape: (num_samples, seq_len, num_features)")
    print("  This will be auto-converted, but it's better to use a list")
    
    print("\n❌ WRONG: All sequences must have same number of features")
    print("  sequences = [np.random.randn(10, 5), np.random.randn(20, 7)]  # Different num_features!")
    
    print("\n❌ WRONG: Sequences must be 2D")
    print("  sequences = [np.random.randn(10), np.random.randn(20)]  # 1D arrays!")
    
    return sequences, event_indicators, time_observed


if __name__ == "__main__":
    print("=" * 60)
    print("DATA FORMAT VALIDATOR")
    print("=" * 60)
    print("\nThis script helps you validate your data format.")
    print("Replace the example data below with your own data.\n")
    
    # Show example
    sequences, event_indicators, time_observed = example_correct_format()
    
    # Validate example
    print("\n" + "=" * 60)
    print("VALIDATING EXAMPLE DATA")
    print("=" * 60)
    check_data_format(sequences, event_indicators, time_observed)
    
    print("\n" + "=" * 60)
    print("TO USE WITH YOUR DATA:")
    print("=" * 60)
    print("""
1. Load your data:
   sequences = [...]  # List of 2D numpy arrays
   event_indicators = np.array([...])  # 1D array
   time_observed = np.array([...])  # 1D array

2. Run validation:
   check_data_format(sequences, event_indicators, time_observed)

3. If validation passes, create your dataset:
   from safe_fraud_detection.data.dataset import CreditCardDataset
   dataset = CreditCardDataset(sequences, event_indicators, time_observed)
    """)

