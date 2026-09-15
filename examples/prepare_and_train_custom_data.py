"""
Complete example: Prepare custom credit card data and train the SAFE model.

This script demonstrates how to:
1. Load your own credit card data
2. Combine features and close dates
3. Prepare data in the required format
4. Train the SAFE fraud detection model
"""

import sys
import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from typing import List, Optional

# Add parent directory to path
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)

from prepare_credit_card_data import prepare_credit_card_data
from safe_fraud_detection.data.dataset import CreditCardDataset
from safe_fraud_detection.data.preprocessing import SequencePreprocessor
from safe_fraud_detection.models.safe_model import SAFEModel
from safe_fraud_detection.models.loss import WeightedSAFELoss
from safe_fraud_detection.utils.trainer import Trainer


def collate_fn(batch):
    """
    Custom collate function to handle variable-length sequences.
    
    Args:
        batch: List of tuples from dataset.__getitem__()
        
    Returns:
        Batched tensors: sequences, masks, events, times, lengths
    """
    # Unpack batch
    sequences, masks, events, times, lengths = zip(*batch)
    
    # Validate shapes before stacking
    if len(batch) == 0:
        raise ValueError("Empty batch received")
    
    # Check that all sequences have the same shape (they should after padding)
    seq_shapes = [s.shape for s in sequences]
    if len(set(seq_shapes)) > 1:
        raise ValueError(
            f"Sequences have inconsistent shapes in batch. "
            f"Expected all sequences to have the same shape after padding. "
            f"Got shapes: {seq_shapes[:5]}..."  # Show first 5
        )
    
    # Check that events, times, and lengths are scalars or 1D
    for i, (e, t, l) in enumerate(zip(events, times, lengths)):
        if e.dim() > 1 or t.dim() > 1 or l.dim() > 1:
            raise ValueError(
                f"Sample {i} has incorrect tensor dimensions. "
                f"Expected scalars or 1D tensors. "
                f"Got event: {e.shape}, time: {t.shape}, length: {l.shape}"
            )
    
    # Stack sequences and masks (they should all have the same shape)
    sequences = torch.stack(sequences)
    masks = torch.stack(masks)
    
    # Concatenate events, times, and lengths (they should be 1D or scalars)
    events = torch.cat(events) if events[0].dim() > 0 else torch.stack(events)
    times = torch.cat(times) if times[0].dim() > 0 else torch.stack(times)
    lengths = torch.cat(lengths) if lengths[0].dim() > 0 else torch.stack(lengths)
    
    return sequences, masks, events, times, lengths


def load_and_prepare_data(
    features_path: str,
    close_dates_path: str,
    card_number_col: str = 'card_number',
    week_start_col: str = 'week_start',
    close_date_col: str = 'CLOSE_DT',
    is_fraud_col: Optional[str] = None
):
    """
    Load and prepare your credit card data.
    
    Args:
        features_path: Path to CSV/Parquet file with features
        close_dates_path: Path to CSV/Parquet file with close dates
        card_number_col: Name of card number column
        week_start_col: Name of week_start column
        close_date_col: Name of close date column
        is_fraud_col: Optional column name for fraud indicator
    
    Returns:
        sequences, event_indicators, time_observed
    """
    print("Loading data...")
    
    # Load dataframes
    if features_path.endswith('.parquet'):
        features_df = pd.read_parquet(features_path)
    else:
        features_df = pd.read_csv(features_path)
    
    if close_dates_path.endswith('.parquet'):
        close_dates_df = pd.read_parquet(close_dates_path)
    else:
        close_dates_df = pd.read_csv(close_dates_path)
    
    print(f"Features dataframe shape: {features_df.shape}")
    print(f"Close dates dataframe shape: {close_dates_df.shape}")
    
    # Prepare data
    sequences, event_indicators, time_observed = prepare_credit_card_data(
        features_df=features_df,
        close_dates_df=close_dates_df,
        card_number_col=card_number_col,
        week_start_col=week_start_col,
        close_date_col=close_date_col,
        is_fraud_col=is_fraud_col
    )
    
    return sequences, event_indicators, time_observed


def train_model(
    sequences: List[np.ndarray],
    event_indicators: np.ndarray,
    time_observed: np.ndarray,
    num_features: int,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    batch_size: int = 32,
    num_epochs: int = 100,
    device: str = 'auto'
):
    """
    Train the SAFE model on prepared data.
    
    Args:
        sequences: List of variable-length sequences
        event_indicators: Binary event indicators
        time_observed: Observed times
        num_features: Number of features per week
        train_ratio: Proportion for training
        val_ratio: Proportion for validation
        test_ratio: Proportion for testing
        batch_size: Batch size for training
        num_epochs: Number of training epochs
        device: Device to use ('auto', 'cuda', or 'cpu')
    """
    
    # Set device
    if device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(device)
    
    print(f"\nUsing device: {device}")
    
    # ============= 1. Preprocess Data =============
    print("\n=== Step 1: Preprocessing Data ===")
    
    # Normalize features
    preprocessor = SequencePreprocessor(
        normalize='standard',
        handle_nan='zero',
        clip_outliers=3.0
    )
    
    # Fit on all sequences
    max_len = max(len(s) for s in sequences)
    temp_padded = np.zeros((len(sequences), max_len, num_features))
    for i, seq in enumerate(sequences):
        temp_padded[i, :len(seq)] = seq
    
    preprocessor.fit(temp_padded)
    
    # Transform sequences
    normalized_sequences = []
    for seq in sequences:
        seq_normalized = preprocessor.transform(seq.reshape(1, len(seq), -1))[0]
        normalized_sequences.append(seq_normalized)
    
    # ============= 2. Create Train/Val/Test Split =============
    print("\n=== Step 2: Creating Train/Val/Test Split ===")
    
    indices = np.arange(len(sequences))
    np.random.seed(42)
    np.random.shuffle(indices)
    
    train_end = int(train_ratio * len(sequences))
    val_end = train_end + int(val_ratio * len(sequences))
    
    train_idx = indices[:train_end]
    val_idx = indices[train_end:val_end]
    test_idx = indices[val_end:]
    
    # Create datasets
    train_dataset = CreditCardDataset(
        [normalized_sequences[i] for i in train_idx],
        event_indicators[train_idx],
        time_observed[train_idx]
    )
    
    val_dataset = CreditCardDataset(
        [normalized_sequences[i] for i in val_idx],
        event_indicators[val_idx],
        time_observed[val_idx]
    )
    
    test_dataset = CreditCardDataset(
        [normalized_sequences[i] for i in test_idx],
        event_indicators[test_idx],
        time_observed[test_idx]
    )
    
    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")
    print(f"Test samples: {len(test_dataset)}")
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size * 2,
        shuffle=False,
        collate_fn=collate_fn
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size * 2,
        shuffle=False,
        collate_fn=collate_fn
    )
    
    # ============= 3. Create Model =============
    print("\n=== Step 3: Creating Model ===")
    
    model = SAFEModel(
        input_dim=num_features,
        hidden_dim=64,
        num_layers=2,
        dropout=0.2
    )
    
    model = model.to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # ============= 4. Setup Loss and Optimizer =============
    print("\n=== Step 4: Setting Up Training ===")
    
    # Calculate class weights
    fraud_count = event_indicators[train_idx].sum()
    total_count = len(train_idx)
    event_weight = total_count / (2 * fraud_count) if fraud_count > 0 else 1.0
    censored_weight = total_count / (2 * (total_count - fraud_count)) if (total_count - fraud_count) > 0 else 1.0
    
    print(f"Event weight: {event_weight:.3f}, Censored weight: {censored_weight:.3f}")
    
    criterion = WeightedSAFELoss(
        event_weight=event_weight,
        censored_weight=censored_weight
    )
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True
    )
    
    # ============= 5. Train Model =============
    print("\n=== Step 5: Training Model ===")
    
    best_val_loss = float('inf')
    patience = 15
    patience_counter = 0
    
    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss = 0.0
        
        for sequences_batch, masks, events_batch, times_batch, lengths_batch in train_loader:
            sequences_batch = sequences_batch.to(device)
            masks = masks.to(device)
            events_batch = events_batch.to(device)
            times_batch = times_batch.to(device)
            
            optimizer.zero_grad()
            
            hazard_rates, survival_probs, _ = model(sequences_batch, masks)
            loss = criterion(hazard_rates, events_batch, times_batch, masks)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        
        # Validation
        model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for sequences_batch, masks, events_batch, times_batch, lengths_batch in val_loader:
                sequences_batch = sequences_batch.to(device)
                masks = masks.to(device)
                events_batch = events_batch.to(device)
                times_batch = times_batch.to(device)
                
                hazard_rates, survival_probs, _ = model(sequences_batch, masks)
                loss = criterion(hazard_rates, events_batch, times_batch, masks)
                
                val_loss += loss.item()
        
        val_loss /= len(val_loader)
        scheduler.step(val_loss)
        
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), 'best_model.pt')
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break
    
    # Load best model
    model.load_state_dict(torch.load('best_model.pt'))
    
    # ============= 6. Evaluate on Test Set =============
    print("\n=== Step 6: Evaluating on Test Set ===")
    
    model.eval()
    test_loss = 0.0
    
    with torch.no_grad():
        for sequences_batch, masks, events_batch, times_batch, lengths_batch in test_loader:
            sequences_batch = sequences_batch.to(device)
            masks = masks.to(device)
            events_batch = events_batch.to(device)
            times_batch = times_batch.to(device)
            
            hazard_rates, survival_probs, _ = model(sequences_batch, masks)
            loss = criterion(hazard_rates, events_batch, times_batch, masks)
            
            test_loss += loss.item()
    
    test_loss /= len(test_loader)
    print(f"Test Loss: {test_loss:.4f}")
    
    print("\n=== Training Complete ===")
    print("Model saved to: best_model.pt")
    
    return model, test_dataset


def main():
    """
    Main function - customize this with your data paths.
    """
    
    # ============= CUSTOMIZE THESE PATHS =============
    features_path = 'path/to/your/features.csv'  # or .parquet
    close_dates_path = 'path/to/your/close_dates.csv'  # or .parquet
    
    # Column names in your dataframes
    card_number_col = 'card_number'  # Adjust to your column name
    week_start_col = 'week_start'    # Adjust to your column name
    close_date_col = 'CLOSE_DT'      # Adjust to your column name
    is_fraud_col = None  # Set to column name if you have explicit fraud indicator
    
    # ============= Load and Prepare Data =============
    sequences, event_indicators, time_observed = load_and_prepare_data(
        features_path=features_path,
        close_dates_path=close_dates_path,
        card_number_col=card_number_col,
        week_start_col=week_start_col,
        close_date_col=close_date_col,
        is_fraud_col=is_fraud_col
    )
    
    # Determine number of features from first sequence
    num_features = sequences[0].shape[1]
    print(f"\nNumber of features per week: {num_features}")
    
    # ============= Train Model =============
    model, test_dataset = train_model(
        sequences=sequences,
        event_indicators=event_indicators,
        time_observed=time_observed,
        num_features=num_features,
        batch_size=32,
        num_epochs=100
    )


if __name__ == "__main__":
    print("="*60)
    print("Credit Card Fraud Detection - Custom Data Training")
    print("="*60)
    print("\nBefore running, please:")
    print("1. Update the file paths in the main() function")
    print("2. Adjust column names to match your data")
    print("3. Ensure your data is in the correct format")
    print("\nSee prepare_credit_card_data.py for data format requirements.")
    print("="*60)
    
    # Uncomment the line below once you've set up your data paths
    # main()

