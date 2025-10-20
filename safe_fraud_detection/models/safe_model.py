"""
SAFE Model Implementation
A Neural Survival Analysis Model for Fraud Early Detection using GRU
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


class SAFEModel(nn.Module):
    """
    SAFE: A Neural Survival Analysis Model for Fraud Early Detection
    
    This model uses a GRU to process time-varying covariates and outputs
    hazard rates at each timestamp. Survival probabilities are then calculated
    from the cumulative hazard rates.
    
    Args:
        input_dim: Dimension of input features at each timestamp
        hidden_dim: Dimension of GRU hidden state (default: 32)
        num_layers: Number of GRU layers (default: 1)
        dropout: Dropout probability for GRU (default: 0.0)
        
    Attributes:
        gru: GRU layer for processing sequential data
        hazard_layer: Linear layer to compute hazard rates
    """
    
    def __init__(
        self, 
        input_dim: int, 
        hidden_dim: int = 32, 
        num_layers: int = 1,
        dropout: float = 0.0
    ):
        super(SAFEModel, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # GRU layer to handle time-varying covariates
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        
        # Linear layer to compute hazard rate from hidden state
        self.hazard_layer = nn.Linear(hidden_dim, 1)
        
    def forward(
        self, 
        x: torch.Tensor, 
        hidden: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass through the SAFE model.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, input_dim)
            hidden: Optional initial hidden state
            
        Returns:
            hazard_rates: Hazard rates at each timestamp (batch_size, seq_len)
            survival_probs: Survival probabilities at each timestamp (batch_size, seq_len)
            hidden_states: Final hidden state from GRU
        """
        batch_size, seq_len, _ = x.shape
        
        # Pass through GRU
        gru_out, hidden_states = self.gru(x, hidden)  # (batch_size, seq_len, hidden_dim)
        
        # Compute hazard rates using softplus activation
        # softplus ensures hazard rates are always positive
        hazard_logits = self.hazard_layer(gru_out).squeeze(-1)  # (batch_size, seq_len)
        hazard_rates = F.softplus(hazard_logits)  # λ_t = ln(1 + exp(w_λ * h_t))
        
        # Calculate cumulative hazard for survival probability
        # S(t) = exp(-Σ λ_k for k=1 to t)
        cumulative_hazard = torch.cumsum(hazard_rates, dim=1)  # (batch_size, seq_len)
        survival_probs = torch.exp(-cumulative_hazard)  # (batch_size, seq_len)
        
        return hazard_rates, survival_probs, hidden_states
    
    def predict(
        self, 
        x: torch.Tensor, 
        threshold: float = 0.5,
        hidden: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict whether users are fraudsters at each timestamp.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, input_dim)
            threshold: Survival probability threshold for classification
            hidden: Optional initial hidden state
            
        Returns:
            predictions: Binary predictions (batch_size, seq_len)
            survival_probs: Survival probabilities (batch_size, seq_len)
        """
        self.eval()
        with torch.no_grad():
            _, survival_probs, _ = self.forward(x, hidden)
            # Predict as fraudster if survival probability < threshold
            predictions = (survival_probs < threshold).long()
        
        return predictions, survival_probs
    
    def get_early_detection_time(
        self, 
        survival_probs: torch.Tensor, 
        threshold: float = 0.5
    ) -> torch.Tensor:
        """
        Get the first timestamp where survival probability falls below threshold.
        
        Args:
            survival_probs: Survival probabilities (batch_size, seq_len)
            threshold: Detection threshold
            
        Returns:
            detection_times: First detection timestamp for each sample (batch_size,)
                           Returns seq_len if never detected
        """
        batch_size, seq_len = survival_probs.shape
        
        # Find first time survival prob drops below threshold
        below_threshold = (survival_probs < threshold).long()
        
        # Get the first occurrence
        detection_times = torch.full((batch_size,), seq_len, dtype=torch.long)
        
        for i in range(batch_size):
            detected_indices = torch.where(below_threshold[i] == 1)[0]
            if len(detected_indices) > 0:
                detection_times[i] = detected_indices[0]
        
        return detection_times
