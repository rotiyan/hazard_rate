"""
Configuration management for SAFE model
"""

import yaml
import json
import torch
from typing import Any, Dict, Optional
from dataclasses import dataclass, asdict, field


@dataclass
class ModelConfig:
    """Model architecture configuration."""
    input_dim: int = 5
    hidden_dim: int = 32
    num_layers: int = 1
    dropout: float = 0.0


@dataclass
class TrainingConfig:
    """Training hyperparameters."""
    batch_size: int = 16
    learning_rate: float = 1e-3
    epochs: int = 100
    early_stopping_patience: Optional[int] = 10
    gradient_clip: Optional[float] = 5.0
    weight_decay: float = 0.0
    optimizer: str = 'adam'


@dataclass
class DataConfig:
    """Data preprocessing configuration."""
    normalize: str = 'standard'  # 'standard', 'minmax', or None
    handle_nan: str = 'zero'  # 'zero', 'mean', 'forward_fill'
    train_ratio: float = 0.7
    val_ratio: float = 0.1
    test_ratio: float = 0.2
    random_seed: int = 42


@dataclass
class LossConfig:
    """Loss function configuration."""
    loss_type: str = 'safe'  # 'safe' or 'regular'
    epsilon: float = 1e-7
    event_weight: float = 1.0
    censored_weight: float = 1.0


@dataclass
class EvaluationConfig:
    """Evaluation configuration."""
    threshold: float = 0.5
    eval_timestamps: list = field(default_factory=lambda: [0, 1, 2, 3, 4])


@dataclass
class Config:
    """Main configuration class."""
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    data: DataConfig = field(default_factory=DataConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    
    # Paths
    data_path: str = 'data/'
    checkpoint_dir: str = 'checkpoints/'
    log_dir: str = 'logs/'
    
    # Device
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    @classmethod
    def from_yaml(cls, path: str) -> 'Config':
        """Load configuration from YAML file."""
        with open(path, 'r') as f:
            config_dict = yaml.safe_load(f)
        return cls.from_dict(config_dict)
    
    @classmethod
    def from_json(cls, path: str) -> 'Config':
        """Load configuration from JSON file."""
        with open(path, 'r') as f:
            config_dict = json.load(f)
        return cls.from_dict(config_dict)
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'Config':
        """Create config from dictionary."""
        model_config = ModelConfig(**config_dict.get('model', {}))
        training_config = TrainingConfig(**config_dict.get('training', {}))
        data_config = DataConfig(**config_dict.get('data', {}))
        loss_config = LossConfig(**config_dict.get('loss', {}))
        eval_config = EvaluationConfig(**config_dict.get('evaluation', {}))
        
        return cls(
            model=model_config,
            training=training_config,
            data=data_config,
            loss=loss_config,
            evaluation=eval_config,
            data_path=config_dict.get('data_path', 'data/'),
            checkpoint_dir=config_dict.get('checkpoint_dir', 'checkpoints/'),
            log_dir=config_dict.get('log_dir', 'logs/'),
            device=config_dict.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return {
            'model': asdict(self.model),
            'training': asdict(self.training),
            'data': asdict(self.data),
            'loss': asdict(self.loss),
            'evaluation': asdict(self.evaluation),
            'data_path': self.data_path,
            'checkpoint_dir': self.checkpoint_dir,
            'log_dir': self.log_dir,
            'device': self.device
        }
    
    def save_yaml(self, path: str):
        """Save configuration to YAML file."""
        with open(path, 'w') as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False)
    
    def save_json(self, path: str):
        """Save configuration to JSON file."""
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
