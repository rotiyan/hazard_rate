"""
SAFE: A Neural Survival Analysis Model for Fraud Early Detection
Implementation based on the paper by Panpan Zheng, Shuhan Yuan, and Xintao Wu
"""

__version__ = "1.0.0"
__author__ = "SAFE Implementation Team"

from .models.safe_model import SAFEModel
from .models.loss import SAFELoss, RegularSurvivalLoss

__all__ = ["SAFEModel", "SAFELoss", "RegularSurvivalLoss"]
