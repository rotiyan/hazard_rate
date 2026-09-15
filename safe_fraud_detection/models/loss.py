"""
Loss functions for SAFE model
Includes both the early detection loss and regular survival analysis loss
"""

import torch
import torch.nn as nn
from typing import Optional


def _sum_observed_hazards(
    hazard_rates: torch.Tensor,
    time_observed: torch.Tensor,
    mask: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """Sum hazard rates up to each sample's observed time, skipping padding."""
    seq_len = hazard_rates.shape[1]
    time_mask = (
        torch.arange(seq_len, device=hazard_rates.device).unsqueeze(0)
        < time_observed.unsqueeze(1)
    ).float()

    # Combine with padding mask if provided
    if mask is not None:
        time_mask = time_mask * mask

    return torch.sum(hazard_rates * time_mask, dim=1)  # (batch_size,)


def _neg_log_one_minus_exp_neg(x: torch.Tensor, epsilon: float) -> torch.Tensor:
    """
    Compute -ln(1 - e^(-x)) stably.

    expm1 keeps precision for small x, and for large x the result goes to 0
    without overflow, so no upper clamp is needed (an upper clamp would zero
    the gradient for large cumulative hazards).
    """
    return -torch.log(-torch.expm1(-torch.clamp(x, min=epsilon)))


def _safe_loss_per_sample(
    hazard_rates: torch.Tensor,
    event_indicator: torch.Tensor,
    time_observed: torch.Tensor,
    mask: Optional[torch.Tensor],
    epsilon: float
) -> torch.Tensor:
    """
    Per-sample SAFE loss: (Σλ_t) - c_i * ln(e^(Σλ_t) - 1)

    Since ln(e^x - 1) = x + ln(1 - e^(-x)), this equals
    (1 - c_i) * Σλ_t + c_i * [-ln(1 - e^(-Σλ_t))].
    """
    sum_hazards = _sum_observed_hazards(hazard_rates, time_observed, mask)
    censored_term = sum_hazards
    event_term = _neg_log_one_minus_exp_neg(sum_hazards, epsilon)
    return (1 - event_indicator) * censored_term + event_indicator * event_term


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
        loss_per_sample = _safe_loss_per_sample(
            hazard_rates, event_indicator, time_observed, mask, self.epsilon
        )
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
        seq_len = hazard_rates.shape[1]
        sum_hazards = _sum_observed_hazards(hazard_rates, time_observed, mask)

        # Get hazard at the observed time (time_observed - 1, 0-indexed)
        time_indices = (time_observed - 1).clamp(min=0, max=seq_len-1).unsqueeze(1)
        hazard_at_time = torch.gather(hazard_rates, 1, time_indices).squeeze(1)

        # -ln(e^λ - 1) = -λ - ln(1 - e^(-λ))
        event_term = -hazard_at_time + _neg_log_one_minus_exp_neg(hazard_at_time, self.epsilon)

        loss_per_sample = sum_hazards + event_indicator * event_term

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
        loss_per_sample = _safe_loss_per_sample(
            hazard_rates, event_indicator, time_observed, mask, self.epsilon
        )

        # Apply weights
        device = hazard_rates.device
        weights = torch.where(
            event_indicator == 1,
            torch.tensor(self.event_weight, device=device),
            torch.tensor(self.censored_weight, device=device)
        )

        return torch.mean(loss_per_sample * weights)
