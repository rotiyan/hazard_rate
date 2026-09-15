"""
Training utilities for SAFE model
"""

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from typing import Any, Optional, Dict
from tqdm import tqdm
import logging

from ..data.dataset import SurvivalBatch


logger = logging.getLogger(__name__)


class SAFETrainer:
    """
    Trainer class for SAFE model.

    Handles training loop, validation, checkpointing, and early stopping.
    Expects loaders that yield SurvivalBatch (as SurvivalDataset does); padding
    masks are passed to both the model and the loss.
    """

    def __init__(
        self,
        model: nn.Module,
        loss_fn: nn.Module,
        optimizer: torch.optim.Optimizer,
        device: str = 'cpu',
        scheduler: Optional[Any] = None,
        gradient_clip: Optional[float] = None,
        show_progress: bool = True
    ):
        """
        Args:
            model: SAFE model to train
            loss_fn: Loss function (SAFELoss, RegularSurvivalLoss or WeightedSAFELoss)
            optimizer: Optimizer for training
            device: Device to train on
            scheduler: Optional learning rate scheduler; ReduceLROnPlateau is
                       stepped with the validation loss (training loss if no
                       validation data)
            gradient_clip: Optional gradient clipping value
            show_progress: Whether to show a progress bar per epoch
        """
        self.model = model.to(device)
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.device = device
        self.scheduler = scheduler
        self.gradient_clip = gradient_clip
        self.show_progress = show_progress

        self.train_losses = []
        self.val_losses = []
        self.best_val_loss = float('inf')
        self.best_model_state = None

    def _batch_loss(self, batch) -> torch.Tensor:
        """Compute the loss for one batch, using its padding mask."""
        batch = SurvivalBatch(*(tensor.to(self.device) for tensor in batch))
        hazard_rates, _, _ = self.model(batch.sequences, batch.masks)
        return self.loss_fn(hazard_rates, batch.events, batch.times, batch.masks)

    def train_epoch(
        self,
        train_loader: DataLoader,
        epoch: int
    ) -> float:
        """
        Train for one epoch.

        Args:
            train_loader: DataLoader for training data
            epoch: Current epoch number

        Returns:
            Average training loss
        """
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch} [Train]", disable=not self.show_progress)

        for batch in pbar:
            loss = self._batch_loss(batch)

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()

            # Gradient clipping
            if self.gradient_clip is not None:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.gradient_clip
                )

            self.optimizer.step()

            # Track loss
            total_loss += loss.item()
            num_batches += 1

            # Update progress bar
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})

        return total_loss / num_batches

    def validate(
        self,
        val_loader: DataLoader
    ) -> float:
        """
        Validate the model.

        Args:
            val_loader: DataLoader for validation data

        Returns:
            Average validation loss
        """
        self.model.eval()

        with torch.no_grad():
            losses = [self._batch_loss(batch).item() for batch in val_loader]

        return sum(losses) / len(losses)

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        epochs: int = 100,
        early_stopping_patience: Optional[int] = None,
        save_best: bool = True,
        verbose: bool = True
    ) -> Dict[str, list]:
        """
        Train the model for multiple epochs.

        Args:
            train_loader: DataLoader for training data
            val_loader: Optional DataLoader for validation
            epochs: Number of epochs to train
            early_stopping_patience: Epochs without validation improvement before stopping
            save_best: Whether to restore the best model state at the end
            verbose: Whether to log progress

        Returns:
            Dictionary with training history
        """
        best_epoch = 0
        epochs_without_improvement = 0

        for epoch in range(1, epochs + 1):
            # Training
            train_loss = self.train_epoch(train_loader, epoch)
            self.train_losses.append(train_loss)
            val_loss = None

            # Validation
            if val_loader is not None:
                val_loss = self.validate(val_loader)
                self.val_losses.append(val_loss)

                if verbose:
                    logger.info(
                        f"Epoch {epoch}/{epochs} - "
                        f"Train Loss: {train_loss:.4f}, "
                        f"Val Loss: {val_loss:.4f}"
                    )

                # Track improvement separately from saving, so early stopping
                # works with save_best=False
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    best_epoch = epoch
                    epochs_without_improvement = 0
                    if save_best:
                        self.best_model_state = {
                            k: v.cpu().clone() for k, v in self.model.state_dict().items()
                        }
                else:
                    epochs_without_improvement += 1
            elif verbose:
                logger.info(f"Epoch {epoch}/{epochs} - Train Loss: {train_loss:.4f}")

            # Learning rate scheduling
            if self.scheduler is not None:
                if isinstance(self.scheduler, ReduceLROnPlateau):
                    self.scheduler.step(val_loss if val_loss is not None else train_loss)
                else:
                    self.scheduler.step()

            # Early stopping
            if early_stopping_patience and epochs_without_improvement >= early_stopping_patience:
                logger.info(f"Early stopping at epoch {epoch}")
                break

        if save_best and self.best_model_state is not None:
            logger.info(f"Loading best model from epoch {best_epoch}")
            self.model.load_state_dict(
                {k: v.to(self.device) for k, v in self.best_model_state.items()}
            )

        return {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'best_epoch': best_epoch,
            'best_val_loss': self.best_val_loss
        }

    def save_checkpoint(self, path: str, epoch: int, **kwargs):
        """
        Save a training checkpoint.

        Args:
            path: Path to save checkpoint
            epoch: Current epoch
            **kwargs: Additional items to save
        """
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'best_val_loss': self.best_val_loss,
        }

        if self.scheduler is not None:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()

        checkpoint.update(kwargs)

        torch.save(checkpoint, path)
        logger.info(f"Checkpoint saved to {path}")

    def load_checkpoint(self, path: str) -> Dict:
        """
        Load a training checkpoint.

        Args:
            path: Path to checkpoint file

        Returns:
            Checkpoint dictionary
        """
        checkpoint = torch.load(path, map_location=self.device)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.train_losses = checkpoint.get('train_losses', [])
        self.val_losses = checkpoint.get('val_losses', [])
        self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))

        if self.scheduler is not None and 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        logger.info(f"Checkpoint loaded from {path}")
        return checkpoint
