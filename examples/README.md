# SAFE Examples

This directory contains example scripts demonstrating how to use the SAFE fraud detection framework.

## Available Examples

### 1. Basic Usage (`example_usage.py`)

Demonstrates the complete workflow:
- Generating synthetic fraud data
- Data preprocessing and splitting
- Model training with early stopping
- Evaluation with multiple metrics
- Making predictions

Run it:
```bash
python examples/example_usage.py
```

## Creating Your Own Dataset

To use SAFE with your own data, prepare it in the following format:

```python
# sequences: (num_samples, seq_len, num_features)
# Time-varying features for each user at each timestamp
sequences = np.array([...])

# events: (num_samples,)
# 1 if user is fraudster (suspended), 0 if censored (not suspended)
events = np.array([...])

# times: (num_samples,)
# Last observed time for each user (1-indexed)
times = np.array([...])

# Save as .npz file
np.savez('my_fraud_data.npz',
         sequences=sequences,
         events=events,
         times=times)
```

Then load and use:
```python
from safe_fraud_detection.data.dataset import SurvivalDataset

data = np.load('my_fraud_data.npz')
dataset = SurvivalDataset(
    data['sequences'],
    data['events'],
    data['times']
)
```

## Tips for Best Results

1. **Feature Engineering**: Create meaningful time-varying features
   - Changes in user behavior over time
   - Cumulative statistics
   - Time since last action

2. **Normalization**: Always normalize your features
   ```python
   from safe_fraud_detection.data.preprocessing import SequencePreprocessor
   
   preprocessor = SequencePreprocessor(normalize='standard')
   sequences = preprocessor.fit_transform(sequences)
   ```

3. **Hyperparameter Tuning**: Key parameters to tune:
   - `hidden_dim`: Size of GRU hidden state (16-64)
   - `learning_rate`: Start with 1e-3, reduce if needed
   - `threshold`: Classification threshold (tune on validation set)

4. **Class Imbalance**: If you have severe class imbalance, use `WeightedSAFELoss`:
   ```python
   from safe_fraud_detection.models.loss import WeightedSAFELoss
   
   loss_fn = WeightedSAFELoss(
       event_weight=2.0,    # Higher weight for fraudsters
       censored_weight=1.0
   )
   ```

5. **Early Stopping**: Monitor validation loss to prevent overfitting:
   ```python
   history = trainer.fit(
       train_loader=train_loader,
       val_loader=val_loader,
       epochs=100,
       early_stopping_patience=10  # Stop if no improvement for 10 epochs
   )
   ```

## Common Issues

### Issue: Model not learning
- Check data preprocessing (normalization)
- Verify loss is decreasing
- Try different learning rates
- Increase model capacity (hidden_dim)

### Issue: Poor early detection
- Make sure you're using `SAFELoss` (not `RegularSurvivalLoss`)
- Tune the classification threshold on validation set
- Check feature quality - are they informative early on?

### Issue: Overfitting
- Add dropout: `SAFEModel(..., dropout=0.2)`
- Use early stopping
- Reduce model capacity
- Get more training data

## Next Steps

- Read the [main README](../README.md) for detailed documentation
- Check the [configuration examples](../safe_fraud_detection/configs/)
- Run the test suite: `pytest`
- Explore the model code in `safe_fraud_detection/models/`
