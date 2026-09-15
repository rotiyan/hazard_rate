"""
Credit Card Fraud Detection using SAFE Model

Simulates weekly credit card transaction aggregates with variable-length
histories, trains SAFE with credit_card_config.yaml, evaluates early detection,
and saves plots and the model to the output directory.

Quick run:
    python examples/credit_card_fraud_example.py --num-cards 500 --max-weeks 52 --epochs 5
"""

import argparse
import logging
import os
import sys

import matplotlib
matplotlib.use('Agg')  # Plots are only saved to files
import matplotlib.pyplot as plt
import numpy as np
import torch

# Add parent directory to path
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(REPO_ROOT)

from safe_fraud_detection.data.credit_card_simulator import CreditCardSimulator, SimulationConfig
from safe_fraud_detection.pipeline import build_model, build_trainer, prepare_dataloaders, save_model
from safe_fraud_detection.utils.config import Config
from safe_fraud_detection.utils.metrics import compute_metrics, evaluate_at_timestamps, predict_survival


CONFIG_PATH = os.path.join(REPO_ROOT, 'safe_fraud_detection', 'configs', 'credit_card_config.yaml')


def print_threshold_table(results, thresholds=(0.3, 0.4, 0.5, 0.6, 0.7)):
    """Print metrics using each card's survival probability at its last observed week."""
    num_samples = len(results['times'])
    final_survival = results['survival_probs'][np.arange(num_samples), results['times'] - 1]
    events = results['events']

    print("\nThreshold | Accuracy | Precision | Recall | F1")
    print("-" * 50)
    for threshold in thresholds:
        m = compute_metrics((final_survival < threshold).astype(int), events)
        print(f"   {threshold:.1f}    |  {m['accuracy']:.3f}   |   {m['precision']:.3f}   | "
              f"{m['recall']:.3f}  | {m['f1']:.3f}")

    auc = compute_metrics((final_survival < 0.5).astype(int), events, probabilities=1 - final_survival)['auc']
    print(f"\nAUC-ROC: {auc:.3f}")


def plot_training_curves(history, path):
    """Plot training and validation loss per epoch."""
    plt.figure(figsize=(10, 6))
    plt.plot(history['train_losses'], label='Train loss', linewidth=2)
    if history['val_losses']:
        plt.plot(history['val_losses'], label='Validation loss', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_survival_curves(model, dataset, device, path, threshold, num_samples=6, seed=42):
    """Plot survival curves for randomly chosen cards over their real weeks."""
    rng = np.random.default_rng(seed)
    indices = rng.choice(len(dataset), size=min(num_samples, len(dataset)), replace=False)

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    model.eval()

    for ax, idx in zip(axes, indices):
        sample = dataset[idx]
        with torch.no_grad():
            _, survival_probs, _ = model(
                sample.sequences.unsqueeze(0).to(device),
                sample.masks.unsqueeze(0).to(device)
            )

        length = sample.lengths.item()
        week = sample.times.item()
        ax.plot(np.arange(length), survival_probs[0, :length].cpu().numpy(), 'b-',
                linewidth=2, label='Survival probability')
        ax.axhline(y=threshold, color='orange', linestyle='--', label='Detection threshold')

        if sample.events.item() == 1:
            ax.axvline(x=week - 1, color='red', linewidth=2, label='Fraud event')
            ax.set_title(f'Card {idx} - FRAUD at week {week}', color='red')
        else:
            ax.set_title(f'Card {idx} - CENSORED at week {week}', color='green')

        ax.set_xlabel('Week')
        ax.set_ylabel('Survival probability')
        ax.set_ylim([-0.05, 1.05])
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)

    for ax in axes[len(indices):]:
        fig.delaxes(ax)

    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def plot_early_detection(weeks_early, path):
    """Plot how many weeks before the fraud event each detected card was flagged."""
    plt.figure(figsize=(10, 6))
    plt.hist(weeks_early, bins=20, edgecolor='black', alpha=0.7, color='skyblue')
    plt.axvline(x=np.mean(weeks_early), color='red', linestyle='--', linewidth=2,
                label=f'Mean: {np.mean(weeks_early):.1f} weeks')
    plt.axvline(x=np.median(weeks_early), color='green', linestyle='--', linewidth=2,
                label=f'Median: {np.median(weeks_early):.1f} weeks')
    plt.xlabel('Weeks detected early')
    plt.ylabel('Number of fraud cases')
    plt.title('Distribution of Early Detection Times')
    plt.legend()
    plt.grid(True, alpha=0.3, axis='y')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()


def parse_args():
    parser = argparse.ArgumentParser(description='Credit card fraud detection with SAFE')
    parser.add_argument('--num-cards', type=int, default=2000, help='Number of cards to simulate')
    parser.add_argument('--max-weeks', type=int, default=208, help='Maximum card lifetime in weeks')
    parser.add_argument('--epochs', type=int, default=None, help='Override training epochs from the config')
    parser.add_argument('--output-dir', type=str, default='outputs', help='Directory for plots and model')
    return parser.parse_args()


def main():
    """Main training and evaluation pipeline."""
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format='%(message)s')

    # Set random seeds
    np.random.seed(42)
    torch.manual_seed(42)
    os.makedirs(args.output_dir, exist_ok=True)

    config = Config.from_yaml(CONFIG_PATH)
    if args.epochs is not None:
        config.training.epochs = args.epochs
    print(f"Using device: {config.device}")

    # ============= 1. Simulate Data =============
    print("\n=== Step 1: Simulating Credit Card Data ===")

    simulator = CreditCardSimulator(SimulationConfig(
        num_cards=args.num_cards,
        max_weeks=args.max_weeks,
        fraud_rate=0.15,
        avg_weeks_to_fraud=52,  # 1 year average
        std_weeks_to_fraud=26,
        early_closure_rate=0.10,
        random_seed=42
    ))
    sequences, events, times, _ = simulator.simulate()

    for key, value in simulator.get_statistics(sequences, events, times).items():
        print(f"  {key}: {value}")

    # ============= 2. Split, Preprocess, Build Loaders =============
    print("\n=== Step 2: Preparing Data ===")

    # Preprocessing is fit on the training split only
    train_loader, val_loader, test_loader, preprocessor = prepare_dataloaders(
        sequences, events, times, config, feature_names=simulator.get_feature_names()
    )
    print(f"Train stats: {train_loader.dataset.get_statistics()}")

    # ============= 3. Train Model =============
    print("\n=== Step 3: Training Model ===")

    config.model.input_dim = simulator.num_features
    model = build_model(config)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    trainer = build_trainer(config, model, train_events=train_loader.dataset.events)
    history = trainer.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=config.training.epochs,
        early_stopping_patience=config.training.early_stopping_patience
    )
    print(f"Best validation loss: {history['best_val_loss']:.4f} (epoch {history['best_epoch']})")

    # ============= 4. Evaluate Model =============
    print("\n=== Step 4: Evaluating Model ===")

    metrics = evaluate_at_timestamps(
        model, test_loader,
        timestamps=config.evaluation.eval_timestamps,
        threshold=config.evaluation.threshold,
        device=config.device
    )
    metrics.print_summary()
    print_threshold_table(predict_survival(model, test_loader, config.device))

    # ============= 5. Save Plots and Model =============
    print("\n=== Step 5: Saving Outputs ===")

    plot_training_curves(history, os.path.join(args.output_dir, 'training_loss.png'))
    plot_survival_curves(
        model, test_loader.dataset, config.device,
        os.path.join(args.output_dir, 'survival_curves.png'),
        threshold=config.evaluation.threshold
    )

    weeks_early = metrics.early_detection_stats['early_detection_times']
    if weeks_early:
        plot_early_detection(weeks_early, os.path.join(args.output_dir, 'early_detection_distribution.png'))

    model_path = os.path.join(args.output_dir, 'credit_card_model.pt')
    save_model(
        model_path, model, config, preprocessor,
        metrics=metrics.get_early_detection_summary(),
        history=history
    )

    print(f"Plots and model saved to {args.output_dir}/")
    print(f"Evaluate later with: safe-eval --model {model_path} --data <test_data.npz>")


if __name__ == "__main__":
    main()
