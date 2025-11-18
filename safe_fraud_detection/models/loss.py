"""
Loss functions for SAFE model
Includes both the early detection loss and regular survival analysis loss
"""

import torch
import torch.nn as nn
from typing import Tuple, Optional


class SAFELoss(nn.Module):
    """
    SAFE Loss Function for Fraud Early Detection
    
    This loss function is designed to detect fraudsters earlier than their
    suspended time by maximizing P{T < t^i} for fraudsters and P{T >= t^i}
    for censored users.
    
    Loss = Σ[(Σλ_t) - c_i * ln(e^(Σλ_t) - 1)]
    
    where:
        - λ_t is the hazard rate at time t
        - c_i is the event indicator (1 for fraudster, 0 for censored)
        - t^i is the last observed time
    """
    
    def __init__(self, epsilon: float = 1e-7):
        """
        Args:
            epsilon: Small constant to avoid log(0) and numerical instability
        """
        super(SAFELoss, self).__init__()
        self.epsilon = epsilon
    
    def forward(
        self, 
        hazard_rates: torch.Tensor, 
        event_indicator: torch.Tensor, 
        time_observed: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute the SAFE loss for early detection.
        
        Args:
            hazard_rates: Hazard rates at each timestamp (batch_size, seq_len)
            event_indicator: Binary indicator if event occurred (batch_size,)
                           1 for fraudster (event), 0 for censored
            time_observed: Last observed time index for each sample (batch_size,)
                          Should be in range [1, seq_len]
            mask: Optional mask tensor (batch_size, seq_len) indicating valid positions
            
        Returns:
            loss: Scalar loss value
        """
        batch_size = hazard_rates.shape[0]
        device = hazard_rates.device
        
        # Create mask for observed times
        # We only sum hazard rates up to the observed time
        seq_len = hazard_rates.shape[1]
        time_mask = torch.arange(seq_len, device=device).unsqueeze(0) < time_observed.unsqueeze(1)
        
        # Combine with padding mask if provided
        if mask is not None:
            time_mask = time_mask.float() * mask
        
        # Sum hazard rates up to observed time for each sample
        masked_hazards = hazard_rates * time_mask.float()
        sum_hazards = torch.sum(masked_hazards, dim=1)  # (batch_size,)
        
        # Clamp sum_hazards to avoid numerical issues
        sum_hazards = torch.clamp(sum_hazards, min=self.epsilon, max=20.0)
        
        # Compute loss components
        # For all samples: Σλ_t
        loss_term1 = sum_hazards
        
        # For event samples: -ln(e^(Σλ_t) - 1)
        # Rewrite as: -ln(e^(Σλ_t) - 1) = -Σλ_t - ln(1 - e^(-Σλ_t))
        # This is more numerically stable
        loss_term2 = -sum_hazards - torch.log(1.0 - torch.exp(-sum_hazards) + self.epsilon)
        
        # Combine: L = Σλ_t - c_i * [Σλ_t + ln(1 - e^(-Σλ_t))]
        loss_per_sample = loss_term1 + event_indicator * loss_term2
        
        # Return mean loss over batch
        return torch.mean(loss_per_sample)


class RegularSurvivalLoss(nn.Module):
    """
    Regular Survival Analysis Loss Function (SAFE-r)
    
    This is the standard survival analysis loss that aims for just-in-time
    prediction rather than early detection.
    
    Loss = Σ[(Σλ_t) - c_i * ln(e^(λ_{t^i}) - 1)]
    
    This loss is included for comparison purposes.
    """
    
    def __init__(self, epsilon: float = 1e-7):
        """
        Args:
            epsilon: Small constant to avoid log(0) and numerical instability
        """
        super(RegularSurvivalLoss, self).__init__()
        self.epsilon = epsilon
    
    def forward(
        self, 
        hazard_rates: torch.Tensor, 
        event_indicator: torch.Tensor, 
        time_observed: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute the regular survival analysis loss.
        
        Args:
            hazard_rates: Hazard rates at each timestamp (batch_size, seq_len)
            event_indicator: Binary indicator if event occurred (batch_size,)
            time_observed: Last observed time index for each sample (batch_size,)
            mask: Optional mask tensor (batch_size, seq_len) indicating valid positions
            
        Returns:
            loss: Scalar loss value
        """
        batch_size = hazard_rates.shape[0]
        device = hazard_rates.device
        seq_len = hazard_rates.shape[1]
        
        # Create mask for observed times
        time_mask = torch.arange(seq_len, device=device).unsqueeze(0) < time_observed.unsqueeze(1)
        
        # Combine with padding mask if provided
        if mask is not None:
            time_mask = time_mask.float() * mask
        
        # Sum hazard rates up to observed time
        masked_hazards = hazard_rates * time_mask.float()
        sum_hazards = torch.sum(masked_hazards, dim=1)  # (batch_size,)
        
        # Clamp for numerical stability
        sum_hazards = torch.clamp(sum_hazards, min=self.epsilon, max=20.0)
        
        # Get hazard at the observed time
        # Need to gather the hazard at time_observed - 1 (0-indexed)
        time_indices = (time_observed - 1).clamp(min=0, max=seq_len-1).unsqueeze(1)
        hazard_at_time = torch.gather(hazard_rates, 1, time_indices).squeeze(1)
        hazard_at_time = torch.clamp(hazard_at_time, min=self.epsilon, max=10.0)
        
        # Compute loss with numerical stability
        loss_term1 = sum_hazards
        # Rewrite: -ln(e^λ - 1) = -λ - ln(1 - e^(-λ))
        loss_term2 = -hazard_at_time - torch.log(1.0 - torch.exp(-hazard_at_time) + self.epsilon)
        
        loss_per_sample = loss_term1 + event_indicator * loss_term2
        
        return torch.mean(loss_per_sample)


class WeightedSAFELoss(nn.Module):
    """
    Weighted SAFE Loss for handling class imbalance.
    """
    
    def __init__(self, event_weight: float = 1.0, censored_weight: float = 1.0, epsilon: float = 1e-7):
        """
        Args:
            event_weight: Weight for fraudster samples
            censored_weight: Weight for censored samples
            epsilon: Small constant for numerical stability
        """
        super(WeightedSAFELoss, self).__init__()
        self.event_weight = event_weight
        self.censored_weight = censored_weight
        self.epsilon = epsilon
    
    def forward(
        self, 
        hazard_rates: torch.Tensor, 
        event_indicator: torch.Tensor, 
        time_observed: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Compute weighted SAFE loss."""
        batch_size = hazard_rates.shape[0]
        device = hazard_rates.device
        seq_len = hazard_rates.shape[1]
        
        # Create mask and compute sum of hazards
        time_mask = torch.arange(seq_len, device=device).unsqueeze(0) < time_observed.unsqueeze(1)
        
        # Combine with padding mask if provided
        if mask is not None:
            time_mask = time_mask.float() * mask
        
        masked_hazards = hazard_rates * time_mask.float()
        sum_hazards = torch.sum(masked_hazards, dim=1)
        
        # Clamp for numerical stability
        sum_hazards = torch.clamp(sum_hazards, min=self.epsilon, max=20.0)
        
        # Compute loss components with numerical stability
        loss_term1 = sum_hazards
        # Rewrite: -ln(e^(Σλ) - 1) = -Σλ - ln(1 - e^(-Σλ))
        loss_term2 = -sum_hazards - torch.log(1.0 - torch.exp(-sum_hazards) + self.epsilon)
        
        loss_per_sample = loss_term1 + event_indicator * loss_term2
        
        # Apply weights
        weights = torch.where(
            event_indicator == 1, 
            torch.tensor(self.event_weight, device=device),
            torch.tensor(self.censored_weight, device=device)
        )
        
        weighted_loss = loss_per_sample * weights
        
        return torch.mean(weighted_loss)
