"""
End-to-end helpers shared by the training scripts and examples

prepare_dataloaders splits the data, fits preprocessing on the training split
only and builds loaders; the build_* functions turn a Config into training
objects; save_model/load_model keep the model, config and preprocessing
together in one checkpoint.
"""

import logging
import os
from dataclasses import asdict
from typing import Dict, List, Optional, Tuple

import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

from .data.dataset import SurvivalDataset
from .data.preprocessing import Sequences, SequencePreprocessor, create_train_val_test_split
from .models.loss import RegularSurvivalLoss, SAFELoss, WeightedSAFELoss, balanced_class_weights
from .models.safe_model import SAFEModel
from .utils.config import Config
from .utils.trainer import SAFETrainer


logger = logging.getLogger(__name__)


def prepare_dataloaders(
    sequences: Sequences,
    events,
    times,
    config: Config,
    feature_names: Optional[List[str]] = None
) -> Tuple[DataLoader, Optional[DataLoader], DataLoader, SequencePreprocessor]:
    """
    Split data, fit preprocessing on the training split, and build data loaders.

    Args:
        sequences: 3D array, or list of unpadded variable-length 2D arrays
        events: Event indicators (1=fraud, 0=censored)
        times: Last observed times
        config: Uses config.data for splitting and preprocessing, and
                config.training.batch_size
        feature_names: Optional feature names stored on the datasets

    Returns:
        train_loader, val_loader (None if the validation split is empty),
        test_loader, fitted preprocessor
    """
    data_config = config.data
    train_data, val_data, test_data = create_train_val_test_split(
        sequences, events, times,
        train_ratio=data_config.train_ratio,
        val_ratio=data_config.val_ratio,
        test_ratio=data_config.test_ratio,
        random_seed=data_config.random_seed
    )

    preprocessor = SequencePreprocessor(
        normalize=data_config.normalize,
        handle_nan=data_config.handle_nan,
        clip_outliers=data_config.clip_outliers
    )
    preprocessor.fit(train_data[0])

    def make_loader(split, shuffle: bool) -> Optional[DataLoader]:
        split_sequences, split_events, split_times = split
        if len(split_events) == 0:
            return None
        dataset = SurvivalDataset(
            preprocessor.transform(split_sequences),
            split_events,
            split_times,
            feature_names=feature_names
        )
        return DataLoader(dataset, batch_size=config.training.batch_size, shuffle=shuffle)

    train_loader = make_loader(train_data, shuffle=True)
    val_loader = make_loader(val_data, shuffle=False)
    test_loader = make_loader(test_data, shuffle=False)

    if train_loader is None or test_loader is None:
        raise ValueError("Train and test splits must not be empty; check the split ratios")

    logger.info(
        f"Train: {len(train_loader.dataset)}, "
        f"Val: {len(val_loader.dataset) if val_loader is not None else 0}, "
        f"Test: {len(test_loader.dataset)}"
    )
    return train_loader, val_loader, test_loader, preprocessor


def build_model(config: Config) -> SAFEModel:
    """Create a SAFE model from config.model."""
    return SAFEModel(**asdict(config.model))


def build_loss(config: Config, train_events=None) -> torch.nn.Module:
    """
    Create the loss function from config.loss.

    Args:
        config: Configuration object
        train_events: Training event indicators, required when loss.auto_weight is set
    """
    loss_config = config.loss

    if loss_config.loss_type == 'safe':
        return SAFELoss(epsilon=loss_config.epsilon)
    if loss_config.loss_type == 'regular':
        return RegularSurvivalLoss(epsilon=loss_config.epsilon)
    if loss_config.loss_type == 'weighted':
        event_weight, censored_weight = loss_config.event_weight, loss_config.censored_weight
        if loss_config.auto_weight:
            if train_events is None:
                raise ValueError("loss.auto_weight needs the training event indicators")
            event_weight, censored_weight = balanced_class_weights(train_events)
            logger.info(f"Class weights - fraud: {event_weight:.3f}, censored: {censored_weight:.3f}")
        return WeightedSAFELoss(
            event_weight=event_weight,
            censored_weight=censored_weight,
            epsilon=loss_config.epsilon
        )

    raise ValueError(f"Unknown loss type: {loss_config.loss_type}")


def build_optimizer(config: Config, model: torch.nn.Module) -> torch.optim.Optimizer:
    """Create the optimizer from config.training."""
    training_config = config.training
    name = training_config.optimizer.lower()

    if name == 'adam':
        return torch.optim.Adam(
            model.parameters(),
            lr=training_config.learning_rate,
            weight_decay=training_config.weight_decay
        )
    if name == 'sgd':
        return torch.optim.SGD(
            model.parameters(),
            lr=training_config.learning_rate,
            momentum=0.9,
            weight_decay=training_config.weight_decay
        )

    raise ValueError(f"Unknown optimizer: {training_config.optimizer}")


def build_scheduler(config: Config, optimizer: torch.optim.Optimizer):
    """Create the learning rate scheduler from config.training, or None."""
    training_config = config.training

    if training_config.lr_scheduler is None:
        return None
    if training_config.lr_scheduler == 'plateau':
        return ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=training_config.scheduler_factor,
            patience=training_config.scheduler_patience
        )

    raise ValueError(f"Unknown lr_scheduler: {training_config.lr_scheduler}")


def build_trainer(config: Config, model: torch.nn.Module, train_events=None) -> SAFETrainer:
    """
    Create a trainer with the loss, optimizer and scheduler from the config.

    Args:
        config: Configuration object
        model: Model to train
        train_events: Training event indicators, used when loss.auto_weight is set
    """
    optimizer = build_optimizer(config, model)
    return SAFETrainer(
        model=model,
        loss_fn=build_loss(config, train_events),
        optimizer=optimizer,
        device=config.device,
        scheduler=build_scheduler(config, optimizer),
        gradient_clip=config.training.gradient_clip
    )


def save_model(
    path: str,
    model: torch.nn.Module,
    config: Config,
    preprocessor: Optional[SequencePreprocessor] = None,
    metrics: Optional[Dict] = None,
    history: Optional[Dict] = None
) -> None:
    """
    Save model weights with the config and preprocessing needed to reuse them.

    Everything except the weights is stored as plain Python values, so the
    file loads with torch.load(weights_only=True).
    """
    output_dir = os.path.dirname(path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    torch.save({
        'model_state_dict': model.state_dict(),
        'config': config.to_dict(),
        'preprocessor': preprocessor.state_dict() if preprocessor is not None else None,
        'metrics': metrics,
        'history': history
    }, path)


def load_model(
    path: str,
    device: str = 'cpu'
) -> Tuple[SAFEModel, Config, Optional[SequencePreprocessor]]:
    """
    Load a model saved with save_model.

    Returns:
        model (in eval mode on device), config, and the fitted preprocessor
        (None for checkpoints saved without one)
    """
    checkpoint = torch.load(path, map_location=device, weights_only=True)

    config = Config.from_dict(checkpoint.get('config') or {})
    config.device = device

    model = build_model(config)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    preprocessor_state = checkpoint.get('preprocessor')
    preprocessor = (
        SequencePreprocessor.from_state_dict(preprocessor_state)
        if preprocessor_state is not None else None
    )

    return model, config, preprocessor
