"""
Credit Card Fraud Detection using SAFE Model

This example demonstrates how to use the SAFE framework for credit card fraud
detection with variable-length transaction sequences.
"""

import sys
import os
import numpy as np
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from safe_fraud_detection.data.credit_card_simulator import CreditCardSimulator, SimulationConfig
from safe_fraud_detection.data.dataset import CreditCardDataset
from safe_fraud_detection.data.preprocessing import SequencePreprocessor, create_train_val_test_split
from safe_fraud_detection.models.safe_model import SAFEModel
from safe_fraud_detection.models.loss import SAFELoss, WeightedSAFELoss
from safe_fraud_detection.utils.trainer import Trainer
from safe_fraud_detection.utils.metrics import calculate_metrics


def collate_fn(batch):
    """Custom collate function to handle variable-length sequences."""
    sequences, masks, events, times, lengths = zip(*batch)
    
    sequences = torch.stack(sequences)
    masks = torch.stack(masks)
    events = torch.cat(events)
    times = torch.cat(times)
    lengths = torch.cat(lengths)
    
    return sequences, masks, events, times, lengths


def visualize_survival_curves(model, dataset, num_samples=5, device='cpu'):
    """Visualize survival curves for sample cards."""
    model.eval()
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    for i in range(min(num_samples, len(dataset))):
        sequence, mask, event, time, length = dataset[i]
        
        # Forward pass
        with torch.no_grad():
            sequence_batch = sequence.unsqueeze(0).to(device)
            mask_batch = mask.unsqueeze(0).to(device)
            hazard_rates, survival_probs, _ = model(sequence_batch, mask_batch)
        
        # Plot
        ax = axes[i]
        time_steps = np.arange(length.item())
        survival_curve = survival_probs[0, :length.item()].cpu().numpy()
        
        ax.plot(time_steps, survival_curve, 'b-', linewidth=2)
        ax.axhline(y=0.5, color='r', linestyle='--', label='Detection threshold')
        
        if event.item() == 1:
            ax.axvline(x=time.item()-1, color='r', linestyle='-', 
                      linewidth=2, label='Fraud event')
            title = f'Card {i} - FRAUD at week {time.item()}'
        else:
            title = f'Card {i} - CENSORED at week {time.item()}'
        
        ax.set_xlabel('Week')
        ax.set_ylabel('Survival Probability')
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim([-0.05, 1.05])
    
    # Remove extra subplots
    for i in range(num_samples, 6):
        fig.delaxes(axes[i])
    
    plt.tight_layout()
    plt.savefig('credit_card_survival_curves.png', dpi=150)
    print("Survival curves saved to credit_card_survival_curves.png")
    plt.close()


def analyze_early_detection(model, dataloader, device='cpu', threshold=0.5):
    """Analyze early detection performance."""
    model.eval()
    
    early_detection_times = []
    actual_fraud_times = []
    weeks_early = []
    
    with torch.no_grad():
        for sequences, masks, events, times, lengths in dataloader:
            sequences = sequences.to(device)
            masks = masks.to(device)
            
            hazard_rates, survival_probs, _ = model(sequences, masks)
            
            # Find early detection time for fraud cases
            for i in range(len(events)):
                if events[i] == 1:  # Fraud case
                    # Find first time survival prob drops below threshold
                    surv_prob = survival_probs[i, :lengths[i]]
                    below_threshold = (surv_prob < threshold).cpu().numpy()
                    
                    if below_threshold.any():
                        detection_week = np.argmax(below_threshold)
                        early_detection_times.append(detection_week)
                        actual_fraud_times.append(times[i].item())
                        weeks_early.append(times[i].item() - detection_week)
    
    if weeks_early:
        stats = {
            'avg_weeks_early': np.mean(weeks_early),
            'median_weeks_early': np.median(weeks_early),
            'pct_detected_early': np.mean(np.array(weeks_early) > 0) * 100,
            'avg_detection_week': np.mean(early_detection_times),
            'avg_fraud_week': np.mean(actual_fraud_times)
        }
        
        print("\n=== Early Detection Analysis ===")
        print(f"Average weeks early: {stats['avg_weeks_early']:.2f}")
        print(f"Median weeks early: {stats['median_weeks_early']:.2f}")
        print(f"% detected early: {stats['pct_detected_early']:.1f}%")
        print(f"Average detection week: {stats['avg_detection_week']:.1f}")
        print(f"Average fraud week: {stats['avg_fraud_week']:.1f}")
        
        return stats
    else:
        print("No fraud cases detected")
        return None


def main():
    """Main training and evaluation pipeline."""
    
    # Set random seeds
    np.random.seed(42)
    torch.manual_seed(42)
    
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # ============= 1. Simulate Data =============
    print("\n=== Step 1: Simulating Credit Card Data ===")
    
    config = SimulationConfig(
        num_cards=2000,
        max_weeks=208,  # 4 years
        fraud_rate=0.15,
        avg_weeks_to_fraud=52,  # 1 year average
        std_weeks_to_fraud=26,
        early_closure_rate=0.10,
        random_seed=42
    )
    
    simulator = CreditCardSimulator(config)
    sequences, events, times, lengths = simulator.simulate()
    
    stats = simulator.get_statistics(sequences, events, times)
    print("\nSimulation Statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    # ============= 2. Preprocess Data =============
    print("\n=== Step 2: Preprocessing Data ===")
    
    # Normalize features (fit on all data for simplicity)
    preprocessor = SequencePreprocessor(
        normalize='standard',
        handle_nan='zero',
        clip_outliers=3.0
    )
    
    # Fit on all sequences
    # Convert list of sequences to padded array for fitting
    max_len = max(len(s) for s in sequences)
    temp_padded = np.zeros((len(sequences), max_len, sequences[0].shape[1]))
    for i, seq in enumerate(sequences):
        temp_padded[i, :len(seq)] = seq
    
    preprocessor.fit(temp_padded)
    
    # Transform sequences
    normalized_sequences = []
    for seq in sequences:
        seq_normalized = preprocessor.transform(seq.reshape(1, len(seq), -1))[0]
        normalized_sequences.append(seq_normalized)
    
    # ============= 3. Create Datasets =============
    print("\n=== Step 3: Creating Datasets ===")
    
    # Split data
    indices = np.arange(len(sequences))
    np.random.shuffle(indices)
    
    train_end = int(0.7 * len(sequences))
    val_end = int(0.85 * len(sequences))
    
    train_idx = indices[:train_end]
    val_idx = indices[train_end:val_end]
    test_idx = indices[val_end:]
    
    # Create datasets
    train_dataset = CreditCardDataset(
        [normalized_sequences[i] for i in train_idx],
        events[train_idx],
        times[train_idx]
    )
    
    val_dataset = CreditCardDataset(
        [normalized_sequences[i] for i in val_idx],
        events[val_idx],
        times[val_idx]
    )
    
    test_dataset = CreditCardDataset(
        [normalized_sequences[i] for i in test_idx],
        events[test_idx],
        times[test_idx]
    )
    
    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")
    print(f"Test samples: {len(test_dataset)}")
    print(f"\nTrain stats: {train_dataset.get_statistics()}")
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=32,
        shuffle=True,
        collate_fn=collate_fn
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=64,
        shuffle=False,
        collate_fn=collate_fn
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=64,
        shuffle=False,
        collate_fn=collate_fn
    )
    
    # ============= 4. Create Model =============
    print("\n=== Step 4: Creating Model ===")
    
    model = SAFEModel(
        input_dim=10,  # 10 features
        hidden_dim=64,
        num_layers=2,
        dropout=0.2
    )
    
    model = model.to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # ============= 5. Train Model =============
    print("\n=== Step 5: Training Model ===")
    
    # Calculate class weights for imbalanced data
    fraud_count = events[train_idx].sum()
    total_count = len(train_idx)
    event_weight = total_count / (2 * fraud_count)
    censored_weight = total_count / (2 * (total_count - fraud_count))
    
    print(f"Event weight: {event_weight:.3f}, Censored weight: {censored_weight:.3f}")
    
    criterion = WeightedSAFELoss(
        event_weight=event_weight,
        censored_weight=censored_weight
    )
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True
    )
    
    # Training loop
    best_val_loss = float('inf')
    patience = 15
    patience_counter = 0
    num_epochs = 100
    
    train_losses = []
    val_losses = []
    
    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss = 0.0
        
        for sequences, masks, events_batch, times_batch, lengths_batch in train_loader:
            sequences = sequences.to(device)
            masks = masks.to(device)
            events_batch = events_batch.to(device)
            times_batch = times_batch.to(device)
            
            optimizer.zero_grad()
            
            hazard_rates, survival_probs, _ = model(sequences, masks)
            loss = criterion(hazard_rates, events_batch, times_batch, masks)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        train_losses.append(train_loss)
        
        # Validation
        model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for sequences, masks, events_batch, times_batch, lengths_batch in val_loader:
                sequences = sequences.to(device)
                masks = masks.to(device)
                events_batch = events_batch.to(device)
                times_batch = times_batch.to(device)
                
                hazard_rates, survival_probs, _ = model(sequences, masks)
                loss = criterion(hazard_rates, events_batch, times_batch, masks)
                
                val_loss += loss.item()
        
        val_loss /= len(val_loader)
        val_losses.append(val_loss)
        
        scheduler.step(val_loss)
        
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            # Save best model
            torch.save(model.state_dict(), 'best_credit_card_model.pt')
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break
    
    # Load best model
    model.load_state_dict(torch.load('best_credit_card_model.pt'))
    
    # ============= 6. Evaluate Model =============
    print("\n=== Step 6: Evaluating Model ===")
    
    # Test evaluation
    model.eval()
    test_loss = 0.0
    all_survival_probs = []
    all_events = []
    all_times = []
    
    with torch.no_grad():
        for sequences, masks, events_batch, times_batch, lengths_batch in test_loader:
            sequences = sequences.to(device)
            masks = masks.to(device)
            events_batch = events_batch.to(device)
            times_batch = times_batch.to(device)
            
            hazard_rates, survival_probs, _ = model(sequences, masks)
            loss = criterion(hazard_rates, events_batch, times_batch, masks)
            
            test_loss += loss.item()
            
            all_survival_probs.append(survival_probs.cpu())
            all_events.append(events_batch.cpu())
            all_times.append(times_batch.cpu())
    
    test_loss /= len(test_loader)
    print(f"Test Loss: {test_loss:.4f}")
    
    # ============= 7. Visualizations =============
    print("\n=== Step 7: Creating Visualizations ===")
    
    # Plot training curves
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('credit_card_training_loss.png', dpi=150)
    print("Training loss curve saved to credit_card_training_loss.png")
    plt.close()
    
    # Visualize survival curves
    visualize_survival_curves(model, test_dataset, num_samples=6, device=device)
    
    # Early detection analysis
    analyze_early_detection(model, test_loader, device=device, threshold=0.5)
    
    print("\n=== Training Complete ===")
    print("Model saved to: best_credit_card_model.pt")


if __name__ == "__main__":
    main()
