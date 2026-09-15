"""
Credit Card Transaction Data Simulator

This module simulates credit card transaction data with variable-length sequences
representing weekly aggregated transactions. Cards can be closed early due to fraud,
creating a survival analysis problem.
"""

import numpy as np
from typing import Tuple, List, Optional, Dict
from dataclasses import dataclass


@dataclass
class SimulationConfig:
    """Configuration for credit card simulation."""
    num_cards: int = 1000
    max_weeks: int = 208  # 4 years (52 weeks * 4)
    fraud_rate: float = 0.15  # Proportion of cards that will experience fraud
    avg_weeks_to_fraud: float = 52  # Average time to fraud event (1 year)
    std_weeks_to_fraud: float = 26  # Std dev of time to fraud
    early_closure_rate: float = 0.10  # Rate of early closure for non-fraud reasons
    random_seed: Optional[int] = 42


class CreditCardSimulator:
    """
    Simulates credit card transaction sequences with fraud events.
    
    Each card has:
    - Variable-length weekly transaction aggregates
    - A potential fraud event (y_T = 1) or censoring (y_T = 0)
    - Time-varying transaction features
    
    Features per week:
    1. Transaction count (number of transactions)
    2. Total amount (sum of transaction amounts)
    3. Average amount (mean transaction amount)
    4. Max amount (maximum single transaction)
    5. Std amount (standard deviation of amounts)
    6. Foreign transaction ratio (proportion of international transactions)
    7. Online transaction ratio (proportion of online transactions)
    8. Weekend transaction ratio (proportion on weekends)
    9. Night transaction ratio (proportion during night hours)
    10. Velocity (change in transaction count from previous week)
    """
    
    def __init__(self, config: SimulationConfig = None):
        """
        Initialize the simulator.
        
        Args:
            config: Simulation configuration
        """
        self.config = config or SimulationConfig()
        if self.config.random_seed is not None:
            np.random.seed(self.config.random_seed)
        
        self.num_features = 10
        
    def simulate(self) -> Tuple[List[np.ndarray], np.ndarray, np.ndarray, np.ndarray]:
        """
        Simulate credit card transaction data.
        
        Returns:
            sequences: List of variable-length sequences, each (seq_len, num_features)
            event_indicators: Binary array indicating fraud (1) or censored (0)
            time_observed: Last observed time (in weeks) for each card
            actual_lengths: Actual sequence length for each card
        """
        sequences = []
        event_indicators = np.zeros(self.config.num_cards)
        time_observed = np.zeros(self.config.num_cards, dtype=int)
        actual_lengths = np.zeros(self.config.num_cards, dtype=int)
        
        # Determine which cards will experience fraud
        fraud_cards = np.random.rand(self.config.num_cards) < self.config.fraud_rate
        
        for i in range(self.config.num_cards):
            is_fraud = fraud_cards[i]
            
            if is_fraud:
                # Sample fraud time from truncated normal distribution
                weeks_to_fraud = int(np.abs(np.random.normal(
                    self.config.avg_weeks_to_fraud,
                    self.config.std_weeks_to_fraud
                )))
                weeks_to_fraud = min(weeks_to_fraud, self.config.max_weeks)
                weeks_to_fraud = max(weeks_to_fraud, 4)  # At least 4 weeks
                
                seq_len = weeks_to_fraud
                event_indicators[i] = 1
            else:
                # Non-fraud cards may close early or reach max weeks
                if np.random.rand() < self.config.early_closure_rate:
                    # Early closure (non-fraud reasons: lost card, customer request, etc.)
                    seq_len = np.random.randint(4, self.config.max_weeks)
                else:
                    # Card reaches max lifetime
                    seq_len = np.random.randint(
                        int(self.config.max_weeks * 0.7), 
                        self.config.max_weeks + 1
                    )
                
                event_indicators[i] = 0
            
            # Generate transaction sequence
            sequence = self._generate_sequence(seq_len, is_fraud)
            
            sequences.append(sequence)
            time_observed[i] = seq_len
            actual_lengths[i] = seq_len
        
        return sequences, event_indicators, time_observed, actual_lengths
    
    def _generate_sequence(self, seq_len: int, is_fraud: bool) -> np.ndarray:
        """
        Generate a single transaction sequence.
        
        Args:
            seq_len: Length of the sequence (in weeks)
            is_fraud: Whether this card will experience fraud
            
        Returns:
            Sequence array of shape (seq_len, num_features)
        """
        sequence = np.zeros((seq_len, self.num_features))
        
        # Base patterns for normal behavior
        base_txn_count = np.random.uniform(5, 20)
        base_amount = np.random.uniform(500, 2000)
        
        # If fraud, pattern changes in later weeks
        fraud_start_week = int(seq_len * 0.7) if is_fraud else seq_len + 1
        
        prev_txn_count = base_txn_count
        
        for t in range(seq_len):
            # Determine if we're in fraud pattern
            in_fraud_pattern = is_fraud and t >= fraud_start_week
            
            # 1. Transaction count
            if in_fraud_pattern:
                # Fraudsters often increase transaction frequency
                txn_count = np.random.poisson(base_txn_count * np.random.uniform(1.5, 3.0))
            else:
                # Normal variation
                seasonal = 1 + 0.2 * np.sin(2 * np.pi * t / 52)  # Yearly seasonality
                txn_count = np.random.poisson(base_txn_count * seasonal)
            
            txn_count = max(1, txn_count)
            
            # 2-5. Amount statistics
            if in_fraud_pattern:
                # Fraudulent transactions often have different amount patterns
                amounts = np.random.lognormal(
                    np.log(base_amount * 1.5), 
                    0.8, 
                    txn_count
                )
            else:
                amounts = np.random.lognormal(np.log(base_amount), 0.6, txn_count)
            
            total_amount = np.sum(amounts)
            avg_amount = np.mean(amounts)
            max_amount = np.max(amounts)
            std_amount = np.std(amounts) if txn_count > 1 else 0
            
            # 6. Foreign transaction ratio
            if in_fraud_pattern:
                foreign_ratio = np.random.uniform(0.3, 0.7)  # More foreign txns in fraud
            else:
                foreign_ratio = np.random.uniform(0.0, 0.15)
            
            # 7. Online transaction ratio
            if in_fraud_pattern:
                online_ratio = np.random.uniform(0.6, 0.95)  # More online in fraud
            else:
                online_ratio = np.random.uniform(0.3, 0.6)
            
            # 8. Weekend transaction ratio
            weekend_ratio = np.random.uniform(0.2, 0.4)
            
            # 9. Night transaction ratio
            if in_fraud_pattern:
                night_ratio = np.random.uniform(0.3, 0.6)  # Unusual hours
            else:
                night_ratio = np.random.uniform(0.05, 0.2)
            
            # 10. Velocity (change in transaction count)
            velocity = (txn_count - prev_txn_count) / max(prev_txn_count, 1)
            prev_txn_count = txn_count
            
            # Store features
            sequence[t, 0] = txn_count
            sequence[t, 1] = total_amount
            sequence[t, 2] = avg_amount
            sequence[t, 3] = max_amount
            sequence[t, 4] = std_amount
            sequence[t, 5] = foreign_ratio
            sequence[t, 6] = online_ratio
            sequence[t, 7] = weekend_ratio
            sequence[t, 8] = night_ratio
            sequence[t, 9] = velocity
        
        return sequence
    
    def get_feature_names(self) -> List[str]:
        """Get list of feature names."""
        return [
            'txn_count',
            'total_amount',
            'avg_amount',
            'max_amount',
            'std_amount',
            'foreign_ratio',
            'online_ratio',
            'weekend_ratio',
            'night_ratio',
            'velocity'
        ]
    
    def get_statistics(
        self, 
        sequences: List[np.ndarray],
        event_indicators: np.ndarray,
        time_observed: np.ndarray
    ) -> Dict:
        """
        Get statistics about the simulated data.
        
        Args:
            sequences: List of sequences
            event_indicators: Fraud indicators
            time_observed: Observed times
            
        Returns:
            Dictionary of statistics
        """
        fraud_mask = event_indicators == 1
        
        stats = {
            'num_cards': len(sequences),
            'num_fraud': int(fraud_mask.sum()),
            'num_censored': int((~fraud_mask).sum()),
            'fraud_rate': float(fraud_mask.mean()),
            'avg_sequence_length': float(np.mean([len(s) for s in sequences])),
            'std_sequence_length': float(np.std([len(s) for s in sequences])),
            'min_sequence_length': int(min(len(s) for s in sequences)),
            'max_sequence_length': int(max(len(s) for s in sequences)),
            'avg_time_to_fraud': float(time_observed[fraud_mask].mean()) if fraud_mask.any() else 0,
            'avg_time_censored': float(time_observed[~fraud_mask].mean()) if (~fraud_mask).any() else 0,
        }
        
        return stats


def create_sample_data(
    num_cards: int = 1000,
    fraud_rate: float = 0.15,
    random_seed: int = 42
) -> Tuple[List[np.ndarray], np.ndarray, np.ndarray, np.ndarray]:
    """
    Convenience function to create sample credit card data.
    
    Args:
        num_cards: Number of cards to simulate
        fraud_rate: Proportion of cards with fraud
        random_seed: Random seed for reproducibility
        
    Returns:
        sequences, event_indicators, time_observed, actual_lengths
    """
    config = SimulationConfig(
        num_cards=num_cards,
        fraud_rate=fraud_rate,
        random_seed=random_seed
    )
    
    simulator = CreditCardSimulator(config)
    return simulator.simulate()


if __name__ == "__main__":
    # Example usage
    print("Simulating credit card transaction data...")
    
    simulator = CreditCardSimulator()
    sequences, events, times, lengths = simulator.simulate()
    
    stats = simulator.get_statistics(sequences, events, times)
    
    print("\n=== Simulation Statistics ===")
    for key, value in stats.items():
        print(f"{key}: {value}")
    
    print("\n=== Feature Names ===")
    for i, name in enumerate(simulator.get_feature_names()):
        print(f"Feature {i}: {name}")
    
    print("\n=== Sample Sequences ===")
    for i in range(3):
        print(f"\nCard {i}:")
        print(f"  Length: {lengths[i]} weeks")
        print(f"  Fraud: {'Yes' if events[i] == 1 else 'No'}")
        print(f"  First week features: {sequences[i][0]}")
        print(f"  Last week features: {sequences[i][-1]}")
