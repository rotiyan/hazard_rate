"""
Quick Start: Train SAFE Model on Your Credit Card Data

Simply update the file paths and column names below, then run this script.
"""

import pandas as pd
import numpy as np
from prepare_credit_card_data import prepare_credit_card_data

# ============================================================================
# STEP 1: UPDATE THESE PATHS AND COLUMN NAMES TO MATCH YOUR DATA
# ============================================================================

# Paths to your data files
FEATURES_FILE = 'path/to/your/features.csv'  # or .parquet
CLOSE_DATES_FILE = 'path/to/your/close_dates.csv'  # or .parquet

# Column names in your dataframes
CARD_NUMBER_COL = 'card_number'  # Name of card number column
WEEK_START_COL = 'week_start'     # Name of week_start column  
CLOSE_DATE_COL = 'CLOSE_DT'      # Name of close date column
IS_FRAUD_COL = None               # Set to column name if you have explicit fraud indicator

# ============================================================================
# STEP 2: LOAD AND PREPARE YOUR DATA
# ============================================================================

print("="*60)
print("Loading and preparing your credit card data...")
print("="*60)

# Load dataframes
if FEATURES_FILE.endswith('.parquet'):
    features_df = pd.read_parquet(FEATURES_FILE)
else:
    features_df = pd.read_csv(FEATURES_FILE)

if CLOSE_DATES_FILE.endswith('.parquet'):
    close_dates_df = pd.read_parquet(CLOSE_DATES_FILE)
else:
    close_dates_df = pd.read_csv(CLOSE_DATES_FILE)

print(f"\nFeatures dataframe shape: {features_df.shape}")
print(f"Close dates dataframe shape: {close_dates_df.shape}")

# Prepare data
sequences, event_indicators, time_observed = prepare_credit_card_data(
    features_df=features_df,
    close_dates_df=close_dates_df,
    card_number_col=CARD_NUMBER_COL,
    week_start_col=WEEK_START_COL,
    close_date_col=CLOSE_DATE_COL,
    is_fraud_col=IS_FRAUD_COL
)

# ============================================================================
# STEP 3: SAVE PREPARED DATA (OPTIONAL)
# ============================================================================

# Save prepared data for later use
print("\n" + "="*60)
print("Saving prepared data...")
print("="*60)

np.savez('prepared_data.npz', 
         sequences=sequences,
         event_indicators=event_indicators,
         time_observed=time_observed)

print("Prepared data saved to: prepared_data.npz")
print("\nYou can now use this data to train the model.")
print("See examples/prepare_and_train_custom_data.py for training code.")

# ============================================================================
# STEP 4: QUICK CHECK - CREATE DATASET
# ============================================================================

print("\n" + "="*60)
print("Quick check: Creating dataset...")
print("="*60)

from safe_fraud_detection.data.dataset import CreditCardDataset

dataset = CreditCardDataset(
    sequences=sequences,
    event_indicators=event_indicators,
    time_observed=time_observed
)

print(f"Dataset created successfully!")
print(f"Dataset statistics:")
stats = dataset.get_statistics()
for key, value in stats.items():
    print(f"  {key}: {value}")

print("\n" + "="*60)
print("Data preparation complete!")
print("="*60)
print("\nNext steps:")
print("1. Review the dataset statistics above")
print("2. Use examples/prepare_and_train_custom_data.py to train the model")
print("3. Or load the saved data: data = np.load('prepared_data.npz', allow_pickle=True)")


