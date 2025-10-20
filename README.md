# SAFE: A Neural Survival Analysis Model for Fraud Early Detection

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A production-ready PyTorch implementation of SAFE (Survival Analysis for Fraud Early detection), based on the paper ["SAFE: A Neural Survival Analysis Model for Fraud Early Detection"](https://arxiv.org/abs/1809.04683) by Panpan Zheng, Shuhan Yuan, and Xintao Wu (AAAI 2019).

## Overview

SAFE combines survival analysis with recurrent neural networks (GRU) to detect fraudulent users earlier than traditional classification methods. Unlike standard classifiers that make independent predictions at each timestamp, SAFE guarantees consistent, monotonically decreasing survival probabilities over time.

### Key Features

- **Early Detection**: Specifically designed to detect fraud before suspension time
- **Consistent Predictions**: Monotonically decreasing survival probabilities
- **No Distribution Assumptions**: Directly outputs hazard rates without assuming parametric distributions
- **Production-Ready**: Comprehensive tests, logging, and configuration management
- **Flexible Architecture**: Easy to customize for different fraud detection scenarios

## Installation

### Using conda (Recommended)

```bash
# Clone the repository
git clone https://github.com/yourusername/safe-fraud-detection.git
cd safe-fraud-detection

# Create and activate conda environment
conda create -n fraud_hazard python=3.9
conda activate fraud_hazard

# Install dependencies
pip install -r requirements.txt

# Install package in development mode
pip install -e .
```

### Using pip

```bash
pip install -r requirements.txt
pip install -e .
```

## Quick Start

### Training a Model

```python
import torch
from torch.utils.data import DataLoader
import numpy as np

from safe_fraud_detection.models.safe_model import SAFEModel
from safe_fraud_detection.models.loss import SAFELoss
from safe_fraud_detection.data.dataset import SurvivalDataset
from safe_fraud_detection.utils.trainer import SAFETrainer

# Load your data
# sequences: (num_samples, seq_len, num_features)
# events: (num_samples,) - 1 for fraudster, 0 for censored
# times: (num_samples,) - last observed time

sequences = np.load('data/sequences.npy')
events = np.load('data/events.npy')
times = np.load('data/times.npy')

# Create dataset and dataloader
dataset = SurvivalDataset(sequences, events, times)
dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

# Create model
model = SAFEModel(
    input_dim=5,  # number of features
    hidden_dim=32,
    num_layers=1
)

# Create loss and optimizer
loss_fn = SAFELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Create trainer
trainer = SAFETrainer(
    model=model,
    loss_fn=loss_fn,
    optimizer=optimizer,
    device='cuda' if torch.cuda.is_available() else 'cpu'
)

# Train
history = trainer.fit(
    train_loader=dataloader,
    epochs=100,
    early_stopping_patience=10
)

# Save model
torch.save(model.state_dict(), 'checkpoints/safe_model.pt')
```

### Making Predictions

```python
# Load model
model = SAFEModel(input_dim=5, hidden_dim=32)
model.load_state_dict(torch.load('checkpoints/safe_model.pt'))
model.eval()

# Predict
with torch.no_grad():
    predictions, survival_probs = model.predict(
        sequences,
        threshold=0.5
    )
    
# Get early detection times
detection_times = model.get_early_detection_time(
    survival_probs,
    threshold=0.5
)
```

### Using Command Line Scripts

```bash
# Train a model
safe-train --config configs/default_config.yaml \
           --data data/fraud_data.npz \
           --output checkpoints/model.pt

# Evaluate a model
safe-eval --model checkpoints/model.pt \
          --data data/test_data.npz \
          --threshold 0.5
```

## Project Structure

```
safe_fraud_detection/
├── models/
│   ├── safe_model.py          # SAFE model implementation
│   └── loss.py                # Loss functions
├── data/
│   ├── dataset.py             # Dataset classes
│   └── preprocessing.py       # Data preprocessing utilities
├── utils/
│   ├── trainer.py             # Training utilities
│   ├── metrics.py             # Evaluation metrics
│   └── config.py              # Configuration management
├── scripts/
│   ├── train.py               # Training script
│   └── evaluate.py            # Evaluation script
├── tests/
│   ├── unit/                  # Unit tests
│   └── integration/           # Integration tests
└── configs/                   # Configuration files
```

## Configuration

SAFE uses YAML configuration files for easy experimentation. Example configuration:

```yaml
model:
  input_dim: 5
  hidden_dim: 32
  num_layers: 1
  dropout: 0.0

training:
  batch_size: 16
  learning_rate: 0.001
  epochs: 100
  early_stopping_patience: 10

loss:
  loss_type: 'safe'  # 'safe', 'regular', or 'weighted'
  epsilon: 1.0e-7

evaluation:
  threshold: 0.5
  eval_timestamps: [0, 1, 2, 3, 4]
```

See `configs/` directory for more examples.

## Data Format

SAFE expects data in the following format:

- **Sequences**: `(num_samples, seq_len, num_features)` - Time-varying features
- **Events**: `(num_samples,)` - Binary indicator (1=fraudster, 0=censored)
- **Times**: `(num_samples,)` - Last observed time for each sample

### Example Data Preparation

```python
import numpy as np
from safe_fraud_detection.data.preprocessing import (
    SequencePreprocessor,
    create_train_val_test_split
)

# Preprocess sequences
preprocessor = SequencePreprocessor(normalize='standard')
sequences_normalized = preprocessor.fit_transform(sequences)

# Split data
train_data, val_data, test_data = create_train_val_test_split(
    sequences_normalized,
    events,
    times,
    train_ratio=0.7,
    val_ratio=0.1,
    test_ratio=0.2
)

# Save for later use
np.savez('data/train_data.npz',
         sequences=train_data[0],
         events=train_data[1],
         times=train_data[2])
```

## Testing

Run the comprehensive test suite:

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=safe_fraud_detection --cov-report=html

# Run specific test categories
pytest -m unit          # Unit tests only
pytest -m integration   # Integration tests only
```

## Model Architecture

SAFE uses a GRU-based architecture:

1. **GRU Layer**: Processes time-varying covariates
2. **Hazard Prediction**: Linear layer + softplus activation
3. **Survival Calculation**: Cumulative hazard → survival probability

```
Input (batch, seq_len, features)
    ↓
GRU (batch, seq_len, hidden_dim)
    ↓
Linear + Softplus (batch, seq_len)  ← Hazard rates λ(t)
    ↓
Cumulative sum + Exp (batch, seq_len)  ← Survival S(t) = exp(-Σλ)
```

## Loss Function

The SAFE loss function is designed for early detection:

```
L = Σ[(Σλ_t) - c_i * ln(e^(Σλ_t) - 1)]
```

where:
- `λ_t`: Hazard rate at time t
- `c_i`: Event indicator (1 for fraudster, 0 for censored)
- The loss encourages early detection by maximizing P(T < t) for fraudsters

## Evaluation Metrics

SAFE tracks multiple metrics:

- **Standard metrics**: Precision, Recall, F1, Accuracy, AUC
- **Time-specific metrics**: Performance at different timestamps (@1, @2, etc.)
- **Early detection metrics**:
  - Percentage of early detected fraudsters
  - Average early detection timestamps
  - Median early detection timestamps

## Performance

On the Twitter dataset (as reported in the paper):

| Metric    | @1    | @2    | @3    | @4    | @5    |
|-----------|-------|-------|-------|-------|-------|
| Precision | 0.831 | 0.827 | 0.819 | 0.814 | 0.808 |
| Recall    | 0.373 | 0.521 | 0.602 | 0.633 | 0.656 |
| F1        | 0.505 | 0.636 | 0.693 | 0.711 | 0.724 |
| Accuracy  | 0.650 | 0.707 | 0.736 | 0.746 | 0.752 |

## Citation

If you use this implementation in your research, please cite the original paper:

```bibtex
@inproceedings{zheng2019safe,
  title={SAFE: A Neural Survival Analysis Model for Fraud Early Detection},
  author={Zheng, Panpan and Yuan, Shuhan and Wu, Xintao},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={33},
  pages={1278--1285},
  year={2019}
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## Acknowledgments

- Original paper by Panpan Zheng, Shuhan Yuan, and Xintao Wu
- Implementation inspired by the paper's methodology
- PyTorch team for the deep learning framework

## Contact

For questions or issues, please open an issue on GitHub or contact [your.email@example.com](mailto:your.email@example.com).

## Roadmap

- [ ] Add support for attention mechanisms
- [ ] Implement multi-task learning extensions
- [ ] Add pre-trained models for common datasets
- [ ] Develop interactive visualization tools
- [ ] Add ONNX export for deployment

---

**Note**: This is a research implementation. For production use, please ensure thorough testing on your specific use case.
