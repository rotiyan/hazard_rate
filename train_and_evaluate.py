"""
Train and Evaluate Credit Card Fraud Detection Model
"""

import numpy as np
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import time

from safe_fraud_detection.data.credit_card_simulator import CreditCardSimulator, SimulationConfig
from safe_fraud_detection.data.dataset import CreditCardDataset
from safe_fraud_detection.data.preprocessing import SequencePreprocessor
from safe_fraud_detection.models.safe_model import SAFEModel
from safe_fraud_detection.models.loss import WeightedSAFELoss

# Set random seeds
np.random.seed(42)
torch.manual_seed(42)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}\n")

# ============= 1. SIMULATE DATA =============
print("=" * 60)
print("STEP 1: SIMULATING CREDIT CARD TRANSACTIONS")
print("=" * 60)

config = SimulationConfig(
    num_cards=15000,
    max_weeks=52,  # 4 years
    fraud_rate=0.15,
    avg_weeks_to_fraud=52,
    std_weeks_to_fraud=26,
    early_closure_rate=0.10,
    random_seed=42
)

simulator = CreditCardSimulator(config)
start_time = time.time()
sequences, events, times, lengths = simulator.simulate()
sim_time = time.time() - start_time

# Get statistics
stats = simulator.get_statistics(sequences, events, times)
print(f"\nSimulation completed in {sim_time:.2f} seconds")
print(f"\n?? Data Statistics:")
print(f"   Total cards: {stats['num_cards']}")
print(f"   Fraud cases: {stats['num_fraud']} ({stats['fraud_rate']:.1%})")
print(f"   Normal cases: {stats['num_censored']}")
print(f"   Avg sequence length: {stats['avg_sequence_length']:.1f} weeks")
print(f"   Min/Max length: {stats['min_sequence_length']}/{stats['max_sequence_length']} weeks")
print(f"   Avg time to fraud: {stats['avg_time_to_fraud']:.1f} weeks")
print(f"   Avg time censored: {stats['avg_time_censored']:.1f} weeks")

# ============= 2. PREPROCESS DATA =============
print("\n" + "=" * 60)
print("STEP 2: PREPROCESSING DATA")
print("=" * 60)

# Normalize features
preprocessor = SequencePreprocessor(
    normalize='standard',
    handle_nan='zero',
    clip_outliers=3.0
)

# Fit on all data and transform
max_len = max(len(s) for s in sequences)
temp_padded = np.zeros((len(sequences), max_len, sequences[0].shape[1]))
for i, seq in enumerate(sequences):
    temp_padded[i, :len(seq)] = seq

preprocessor.fit(temp_padded)

normalized_sequences = []
for seq in sequences:
    seq_normalized = preprocessor.transform(seq.reshape(1, len(seq), -1))[0]
    normalized_sequences.append(seq_normalized)

print(f"\n? Normalized {len(sequences)} sequences")
print(f"? Features per week: {sequences[0].shape[1]}")

# ============= 3. CREATE DATASETS =============
print("\n" + "=" * 60)
print("STEP 3: CREATING TRAIN/VAL/TEST SPLITS")
print("=" * 60)

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

print(f"\n?? Dataset Splits:")
print(f"   Train: {len(train_dataset)} cards ({events[train_idx].mean():.1%} fraud)")
print(f"   Val:   {len(val_dataset)} cards ({events[val_idx].mean():.1%} fraud)")
print(f"   Test:  {len(test_dataset)} cards ({events[test_idx].mean():.1%} fraud)")

# Custom collate function
def collate_fn(batch):
    sequences, masks, events, times, lengths = zip(*batch)
    return (
        torch.stack(sequences),
        torch.stack(masks),
        torch.cat(events),
        torch.cat(times),
        torch.cat(lengths)
    )

# Create dataloaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, collate_fn=collate_fn)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, collate_fn=collate_fn)

# ============= 4. CREATE AND TRAIN MODEL =============
print("\n" + "=" * 60)
print("STEP 4: CREATING AND TRAINING MODEL")
print("=" * 60)

model = SAFEModel(
    input_dim=10,
    hidden_dim=64,
    num_layers=2,
    dropout=0.2
).to(device)

total_params = sum(p.numel() for p in model.parameters())
print(f"\n?? Model Architecture:")
print(f"   Input dimension: 10 features")
print(f"   Hidden dimension: 64")
print(f"   GRU layers: 2")
print(f"   Total parameters: {total_params:,}")

# Calculate class weights
fraud_count = events[train_idx].sum()
total_count = len(train_idx)
event_weight = total_count / (2 * fraud_count)
censored_weight = total_count / (2 * (total_count - fraud_count))

print(f"\n??  Class Weights:")
print(f"   Fraud weight: {event_weight:.3f}")
print(f"   Normal weight: {censored_weight:.3f}")

criterion = WeightedSAFELoss(event_weight=event_weight, censored_weight=censored_weight)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

# Training
print(f"\n???  Training for 50 epochs...")
best_val_loss = float('inf')
patience = 10
patience_counter = 0
num_epochs = 50

train_losses = []
val_losses = []

start_time = time.time()

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
    
    if (epoch + 1) % 10 == 0:
        print(f"   Epoch {epoch+1:3d}/{num_epochs} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
    
    # Early stopping
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        patience_counter = 0
        torch.save(model.state_dict(), 'best_credit_card_model.pt')
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print(f"\n? Early stopping at epoch {epoch+1}")
            break

training_time = time.time() - start_time
print(f"\n? Training completed in {training_time:.1f} seconds ({training_time/60:.1f} minutes)")

# Load best model
model.load_state_dict(torch.load('best_credit_card_model.pt'))

# ============= 5. EVALUATE MODEL =============
print("\n" + "=" * 60)
print("STEP 5: EVALUATING MODEL PERFORMANCE")
print("=" * 60)

model.eval()
all_predictions = []
all_survival_probs = []
all_events = []
all_times = []
all_lengths = []

with torch.no_grad():
    for sequences, masks, events_batch, times_batch, lengths_batch in test_loader:
        sequences = sequences.to(device)
        masks = masks.to(device)
        
        hazard_rates, survival_probs, _ = model(sequences, masks)
        
        all_survival_probs.append(survival_probs.cpu())
        all_events.append(events_batch.cpu())
        all_times.append(times_batch.cpu())
        all_lengths.append(lengths_batch.cpu())

all_survival_probs = torch.cat(all_survival_probs)
all_events = torch.cat(all_events).numpy()
all_times = torch.cat(all_times).numpy()
all_lengths = torch.cat(all_lengths).numpy()

# Calculate metrics at different time points
print("\n?? Performance Metrics:")

# Use survival probability at last observed time for each card
final_survival_probs = []
for i in range(len(all_survival_probs)):
    time_idx = min(int(all_times[i]) - 1, all_survival_probs.shape[1] - 1)
    final_survival_probs.append(all_survival_probs[i, time_idx].item())

final_survival_probs = np.array(final_survival_probs)

# Test different thresholds
thresholds = [0.3, 0.4, 0.5, 0.6, 0.7]
print("\n   Threshold | Accuracy | Precision | Recall | F1-Score")
print("   " + "-" * 60)

for threshold in thresholds:
    predictions = (final_survival_probs < threshold).astype(int)
    
    acc = accuracy_score(all_events, predictions)
    prec = precision_score(all_events, predictions, zero_division=0)
    rec = recall_score(all_events, predictions, zero_division=0)
    f1 = f1_score(all_events, predictions, zero_division=0)
    
    print(f"      {threshold:.1f}    |  {acc:.3f}   |   {prec:.3f}    |  {rec:.3f}  |  {f1:.3f}")

# AUC-ROC
try:
    auc = roc_auc_score(all_events, 1 - final_survival_probs)
    print(f"\n   AUC-ROC: {auc:.3f}")
except:
    print("\n   AUC-ROC: Could not compute")

# ============= 6. EARLY DETECTION ANALYSIS =============
print("\n" + "=" * 60)
print("STEP 6: EARLY DETECTION ANALYSIS")
print("=" * 60)

threshold = 0.5
early_detection_times = []
actual_fraud_times = []
weeks_early = []

for i in range(len(all_events)):
    if all_events[i] == 1:  # Fraud case
        surv_prob = all_survival_probs[i, :all_lengths[i]]
        below_threshold = (surv_prob < threshold).cpu().numpy()
        
        if below_threshold.any():
            detection_week = np.argmax(below_threshold)
            early_detection_times.append(detection_week)
            actual_fraud_times.append(all_times[i])
            weeks_early.append(all_times[i] - detection_week)

if weeks_early:
    print(f"\n?? Early Detection Results (threshold={threshold}):")
    print(f"   Total fraud cases: {int(all_events.sum())}")
    print(f"   Detected early: {len(weeks_early)} ({len(weeks_early)/int(all_events.sum())*100:.1f}%)")
    print(f"   Average weeks early: {np.mean(weeks_early):.1f} weeks")
    print(f"   Median weeks early: {np.median(weeks_early):.1f} weeks")
    print(f"   Max weeks early: {np.max(weeks_early):.0f} weeks")
    print(f"   Min weeks early: {np.min(weeks_early):.0f} weeks")
    
    # Distribution of early detection
    early_pct = np.mean(np.array(weeks_early) > 0) * 100
    print(f"   Detected before fraud: {early_pct:.1f}%")

# ============= 7. VISUALIZATIONS =============
print("\n" + "=" * 60)
print("STEP 7: CREATING VISUALIZATIONS")
print("=" * 60)

# Plot 1: Training curves
plt.figure(figsize=(10, 6))
plt.plot(train_losses, label='Train Loss', linewidth=2)
plt.plot(val_losses, label='Validation Loss', linewidth=2)
plt.xlabel('Epoch', fontsize=12)
plt.ylabel('Loss', fontsize=12)
plt.title('Training and Validation Loss', fontsize=14, fontweight='bold')
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('training_loss.png', dpi=150, bbox_inches='tight')
print("   ? Saved training_loss.png")

# Plot 2: Sample survival curves
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.flatten()

sample_indices = np.random.choice(len(test_dataset), 6, replace=False)

for idx, sample_idx in enumerate(sample_indices):
    seq, mask, event, time, length = test_dataset[sample_idx]
    
    with torch.no_grad():
        seq_batch = seq.unsqueeze(0).to(device)
        mask_batch = mask.unsqueeze(0).to(device)
        _, survival_probs, _ = model(seq_batch, mask_batch)
    
    ax = axes[idx]
    weeks = np.arange(length.item())
    survival_curve = survival_probs[0, :length.item()].cpu().numpy()
    
    ax.plot(weeks, survival_curve, 'b-', linewidth=2, label='Survival probability')
    ax.axhline(y=0.5, color='orange', linestyle='--', linewidth=1.5, label='Detection threshold')
    
    if event.item() == 1:
        ax.axvline(x=time.item()-1, color='red', linestyle='-', linewidth=2, label='Fraud event')
        title = f'Card {sample_idx} - FRAUD at week {time.item()}'
        color = 'red'
    else:
        title = f'Card {sample_idx} - NORMAL'
        color = 'green'
    
    ax.set_xlabel('Week', fontsize=10)
    ax.set_ylabel('Survival Probability', fontsize=10)
    ax.set_title(title, fontsize=11, fontweight='bold', color=color)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_ylim([-0.05, 1.05])

plt.tight_layout()
plt.savefig('survival_curves.png', dpi=150, bbox_inches='tight')
print("   ? Saved survival_curves.png")

# Plot 3: Early detection distribution
if weeks_early:
    plt.figure(figsize=(10, 6))
    plt.hist(weeks_early, bins=20, edgecolor='black', alpha=0.7, color='skyblue')
    plt.axvline(x=np.mean(weeks_early), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(weeks_early):.1f} weeks')
    plt.axvline(x=np.median(weeks_early), color='green', linestyle='--', linewidth=2, label=f'Median: {np.median(weeks_early):.1f} weeks')
    plt.xlabel('Weeks Detected Early', fontsize=12)
    plt.ylabel('Number of Fraud Cases', fontsize=12)
    plt.title('Distribution of Early Detection Times', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig('early_detection_distribution.png', dpi=150, bbox_inches='tight')
    print("   ? Saved early_detection_distribution.png")

print("\n" + "=" * 60)
print("? TRAINING AND EVALUATION COMPLETE!")
print("=" * 60)
print("\n?? Generated Files:")
print("   ? best_credit_card_model.pt - Trained model")
print("   ? training_loss.png - Loss curves")
print("   ? survival_curves.png - Sample predictions")
print("   ? early_detection_distribution.png - Early detection analysis")
print("\n?? Success! The model is ready to use.")
