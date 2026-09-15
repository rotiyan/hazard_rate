"""
Example usage of SAFE fraud detection model

This script demonstrates how to:
1. Generate synthetic fraud data
2. Train a SAFE model
3. Evaluate the model
4. Make predictions
"""

import numpy as np
import torch
from torch.utils.data import DataLoader
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from safe_fraud_detection.models.safe_model import SAFEModel
from safe_fraud_detection.models.loss import SAFELoss
from safe_fraud_detection.data.dataset import SurvivalDataset
from safe_fraud_detection.data.preprocessing import (
    SequencePreprocessor,
    create_train_val_test_split
)
from safe_fraud_detection.utils.trainer import SAFETrainer
from safe_fraud_detection.utils.metrics import evaluate_at_timestamps


def generate_synthetic_data(num_samples=500, seq_len=15, num_features=5):
    """
    Generate synthetic fraud detection data.
    
    Returns:
        sequences, events, times
    """
    print("Generating synthetic data...")
    np.random.seed(42)
    
    # Generate sequences
    sequences = np.random.randn(num_samples, seq_len, num_features)
    
    # Generate events (50% fraudsters, 50% censored)
    events = np.random.randint(0, 2, num_samples)
    
    # Generate observation times
    # Fraudsters tend to be caught earlier
    times = np.where(
        events == 1,
        np.random.randint(8, seq_len + 1, num_samples),  # Fraudsters: later times
        np.random.randint(5, seq_len + 1, num_samples)   # Censored: any time
    )
    
    # Add some patterns to fraudster sequences
    for i in range(num_samples):
        if events[i] == 1:
            # Fraudsters have higher variance in certain features
            sequences[i, :, 0] *= 2.0
            sequences[i, :, 1] += 1.0
    
    print(f"Generated {num_samples} samples:")
    print(f"  - Fraudsters: {events.sum()}")
    print(f"  - Censored: {(1 - events).sum()}")
    print(f"  - Sequence length: {seq_len}")
    print(f"  - Features: {num_features}")
    
    return sequences, events, times


def main():
    print("="*60)
    print("SAFE Fraud Detection - Example Usage")
    print("="*60 + "\n")
    
    # Set device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}\n")
    
    # 1. Generate synthetic data
    sequences, events, times = generate_synthetic_data(
        num_samples=500,
        seq_len=15,
        num_features=5
    )
    
    # 2. Preprocess data
    print("\nPreprocessing data...")
    preprocessor = SequencePreprocessor(normalize='standard', handle_nan='zero')
    
    # Split data
    train_data, val_data, test_data = create_train_val_test_split(
        sequences, events, times,
        train_ratio=0.6,
        val_ratio=0.2,
        test_ratio=0.2,
        random_seed=42
    )
    
    # Normalize
    train_sequences = preprocessor.fit_transform(train_data[0])
    val_sequences = preprocessor.transform(val_data[0])
    test_sequences = preprocessor.transform(test_data[0])
    
    print(f"Train samples: {len(train_sequences)}")
    print(f"Val samples: {len(val_sequences)}")
    print(f"Test samples: {len(test_sequences)}")
    
    # 3. Create datasets and dataloaders
    print("\nCreating datasets...")
    train_dataset = SurvivalDataset(train_sequences, train_data[1], train_data[2])
    val_dataset = SurvivalDataset(val_sequences, val_data[1], val_data[2])
    test_dataset = SurvivalDataset(test_sequences, test_data[1], test_data[2])
    
    print(f"Train dataset stats: {train_dataset.get_statistics()}")
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    # 4. Create model
    print("\nCreating SAFE model...")
    model = SAFEModel(
        input_dim=5,
        hidden_dim=32,
        num_layers=1,
        dropout=0.0
    )
    
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {num_params:,}")
    
    # 5. Create loss and optimizer
    loss_fn = SAFELoss(epsilon=1e-7)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # 6. Train model
    print("\nTraining model...")
    trainer = SAFETrainer(
        model=model,
        loss_fn=loss_fn,
        optimizer=optimizer,
        device=device,
        gradient_clip=5.0
    )
    
    history = trainer.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=30,
        early_stopping_patience=10,
        save_best=True,
        verbose=True
    )
    
    print(f"\nTraining completed!")
    print(f"Best validation loss: {history['best_val_loss']:.4f}")
    print(f"Final training loss: {history['train_losses'][-1]:.4f}")
    
    # 7. Evaluate model
    print("\nEvaluating model on test set...")
    metrics = evaluate_at_timestamps(
        model=model,
        data_loader=test_loader,
        timestamps=[0, 1, 2, 3, 4],
        threshold=0.5,
        device=device
    )
    
    metrics.print_summary()
    
    # 8. Make predictions on a sample
    print("\nMaking predictions on sample data...")
    sample_sequences = torch.FloatTensor(test_sequences[:5]).to(device)
    
    with torch.no_grad():
        predictions, survival_probs = model.predict(sample_sequences, threshold=0.5)
    
    print("\nSample predictions:")
    print(f"True labels: {test_data[1][:5]}")
    print(f"Predictions at t=5: {predictions[:, 4].cpu().numpy()}")
    print(f"Survival probs at t=5: {survival_probs[:, 4].cpu().numpy()}")
    
    # 9. Save model
    save_path = 'checkpoints/example_model.pt'
    import os
    os.makedirs('checkpoints', exist_ok=True)
    
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': {
            'input_dim': 5,
            'hidden_dim': 32,
            'num_layers': 1
        },
        'metrics': metrics.get_early_detection_summary(),
        'history': history
    }, save_path)
    
    print(f"\nModel saved to {save_path}")
    
    print("\n" + "="*60)
    print("Example completed successfully!")
    print("="*60)


if __name__ == '__main__':
    main()
