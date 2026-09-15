"""
Utility script to prepare credit card data for training the SAFE fraud detection model.

This script combines:
1. A dataframe with (card_number, week_start) as index and 21 covariates
2. A dataframe with (card_number, CLOSE_DT) indicating when cards were closed

And creates the required format:
- sequences: List of variable-length sequences (one per card)
- event_indicators: Binary array (1=fraud, 0=censored)
- time_observed: Last observed time in weeks for each card
"""

import pandas as pd
import numpy as np
from typing import Tuple, List, Optional
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')


def prepare_credit_card_data(
    features_df: pd.DataFrame,
    close_dates_df: pd.DataFrame,
    card_number_col: str = 'card_number',
    week_start_col: str = 'week_start',
    close_date_col: str = 'CLOSE_DT',
    is_fraud_col: Optional[str] = None,
    reference_date: Optional[str] = None
) -> Tuple[List[np.ndarray], np.ndarray, np.ndarray]:
    """
    Prepare credit card data for training.
    
    Args:
        features_df: DataFrame with (card_number, week_start) as index and covariates
        close_dates_df: DataFrame with card_number and CLOSE_DT columns
        card_number_col: Name of card number column (if not in index)
        week_start_col: Name of week_start column (if not in index)
        close_date_col: Name of close date column
        is_fraud_col: Optional column name indicating if closure was due to fraud.
                     If None, assumes all closures with CLOSE_DT are fraud (event=1)
        reference_date: Optional reference date for calculating weeks. 
                       If None, uses earliest week_start in data
    
    Returns:
        sequences: List of variable-length sequences, each (seq_len, num_features)
        event_indicators: Binary array (1=fraud, 0=censored)
        time_observed: Last observed time (in weeks) for each card
    """
    
    # Reset index if needed to access card_number and week_start
    if isinstance(features_df.index, pd.MultiIndex):
        features_df = features_df.reset_index()
    
    # Ensure we have the required columns
    if card_number_col not in features_df.columns:
        raise ValueError(f"Column '{card_number_col}' not found in features_df")
    if week_start_col not in features_df.columns:
        raise ValueError(f"Column '{week_start_col}' not found in features_df")
    
    # Convert week_start to datetime if it's not already
    if not pd.api.types.is_datetime64_any_dtype(features_df[week_start_col]):
        features_df[week_start_col] = pd.to_datetime(features_df[week_start_col])
    
    # Convert CLOSE_DT to datetime if it's not already
    if not pd.api.types.is_datetime64_any_dtype(close_dates_df[close_date_col]):
        close_dates_df[close_date_col] = pd.to_datetime(close_dates_df[close_date_col])
    
    # Set reference date (earliest week_start if not provided)
    if reference_date is None:
        reference_date = features_df[week_start_col].min()
    else:
        reference_date = pd.to_datetime(reference_date)
    
    # Get all unique card numbers
    all_cards = features_df[card_number_col].unique()
    print(f"Found {len(all_cards)} unique cards")
    
    # Get feature columns (exclude card_number and week_start)
    feature_cols = [col for col in features_df.columns 
                   if col not in [card_number_col, week_start_col]]
    num_features = len(feature_cols)
    print(f"Found {num_features} feature columns: {feature_cols[:5]}...")
    
    # Create a dictionary mapping card_number to close date and fraud indicator
    close_dict = {}
    for _, row in close_dates_df.iterrows():
        card_num = row[card_number_col]
        close_dt = row[close_date_col]
        
        # Determine if this is fraud (event=1) or censored (event=0)
        if pd.isna(close_dt):
            # No close date = censored (still active)
            close_dict[card_num] = {'close_date': None, 'is_fraud': False}
        else:
            if is_fraud_col and is_fraud_col in close_dates_df.columns:
                # Use explicit fraud indicator if available
                is_fraud = bool(row[is_fraud_col])
            else:
                # If no fraud column, assume all closures are fraud
                is_fraud = True
            
            close_dict[card_num] = {'close_date': close_dt, 'is_fraud': is_fraud}
    
    # Build sequences for each card
    sequences = []
    event_indicators = []
    time_observed = []
    num_closed_before_first_week = 0

    for card_num in all_cards:
        # Get all rows for this card, sorted by week_start
        card_data = features_df[features_df[card_number_col] == card_num].copy()
        card_data = card_data.sort_values(week_start_col)
        
        if len(card_data) == 0:
            continue
        
        # Extract feature sequence
        feature_sequence = card_data[feature_cols].values.astype(np.float32)
        
        # Get close date information
        card_info = close_dict.get(card_num, {'close_date': None, 'is_fraud': False})
        close_date = card_info['close_date']
        is_fraud = card_info['is_fraud']
        
        # Calculate time_observed as a count of weekly rows, so it indexes the
        # same positions as the sequence (also correct when weeks are missing)
        observation_end = card_data[week_start_col].max() + pd.Timedelta(days=7)

        if close_date is not None and not pd.isna(close_date) and close_date < observation_end:
            # Closed during observation period (including the last observed week):
            # keep the weeks that started before the close date
            time_obs = int((card_data[week_start_col] < close_date).sum())
            if time_obs == 0:
                # Closed before the card's first observed week - no usable history
                num_closed_before_first_week += 1
                continue
            feature_sequence = feature_sequence[:time_obs]
            event_indicator = 1 if is_fraud else 0
        else:
            # No close date (still active) or closed after observation period: censored
            time_obs = len(feature_sequence)
            event_indicator = 0

        sequences.append(feature_sequence)
        event_indicators.append(event_indicator)
        time_observed.append(time_obs)
    
    # Convert to numpy arrays
    event_indicators = np.array(event_indicators, dtype=np.float32)
    time_observed = np.array(time_observed, dtype=np.int32)
    
    print(f"\n=== Data Preparation Summary ===")
    print(f"Total cards processed: {len(sequences)}")
    if num_closed_before_first_week:
        print(f"Skipped cards closed before their first observed week: {num_closed_before_first_week}")
    print(f"Number of features per week: {num_features}")
    print(f"Fraud cases (event=1): {int(event_indicators.sum())}")
    print(f"Censored cases (event=0): {int((1 - event_indicators).sum())}")
    print(f"Average sequence length: {np.mean([len(s) for s in sequences]):.2f} weeks")
    print(f"Min sequence length: {min(len(s) for s in sequences)} weeks")
    print(f"Max sequence length: {max(len(s) for s in sequences)} weeks")
    print(f"Average time observed: {time_observed.mean():.2f} weeks")
    
    return sequences, event_indicators, time_observed


def example_usage():
    """
    Example of how to use the prepare_credit_card_data function.
    """
    # Example: Create sample dataframes
    print("Creating example dataframes...")
    
    # Example features dataframe
    dates = pd.date_range('2023-01-01', periods=52, freq='W')
    cards = ['CARD001', 'CARD002', 'CARD003']
    
    features_data = []
    for card in cards:
        for date in dates[:20]:  # 20 weeks per card
            features_data.append({
                'card_number': card,
                'week_start': date,
                'feature_1': np.random.randn(),
                'feature_2': np.random.randn(),
                # ... 19 more features
            })
    
    features_df = pd.DataFrame(features_data)
    
    # Example close dates dataframe
    close_dates_df = pd.DataFrame({
        'card_number': ['CARD001', 'CARD002', 'CARD003'],
        'CLOSE_DT': [
            pd.Timestamp('2023-05-15'),  # Fraud case
            None,  # Censored (still active)
            pd.Timestamp('2023-08-20')   # Another fraud case
        ]
    })
    
    # Prepare data
    sequences, events, times = prepare_credit_card_data(
        features_df=features_df,
        close_dates_df=close_dates_df,
        card_number_col='card_number',
        week_start_col='week_start',
        close_date_col='CLOSE_DT'
    )
    
    print(f"\nPrepared {len(sequences)} sequences")
    print(f"Event indicators: {events}")
    print(f"Time observed: {times}")


if __name__ == "__main__":
    # Example usage
    example_usage()
    
    print("\n" + "="*60)
    print("To use with your own data:")
    print("="*60)
    print("""
    import pandas as pd
    from prepare_credit_card_data import prepare_credit_card_data
    
    # Load your dataframes
    features_df = pd.read_csv('your_features.csv')  # or pd.read_parquet, etc.
    close_dates_df = pd.read_csv('your_close_dates.csv')
    
    # Prepare data
    sequences, event_indicators, time_observed = prepare_credit_card_data(
        features_df=features_df,
        close_dates_df=close_dates_df,
        card_number_col='card_number',  # Adjust to your column name
        week_start_col='week_start',    # Adjust to your column name
        close_date_col='CLOSE_DT',
        is_fraud_col=None  # Set to column name if you have explicit fraud indicator
    )
    
    # Now you can use sequences, event_indicators, time_observed with CreditCardDataset
    from safe_fraud_detection.data.dataset import CreditCardDataset
    
    dataset = CreditCardDataset(
        sequences=sequences,
        event_indicators=event_indicators,
        time_observed=time_observed
    )
    """)

