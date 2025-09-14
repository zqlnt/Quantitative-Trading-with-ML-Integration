"""
Validation utilities for Neural Quant.

This module provides robust validation methods including purged cross-validation
and embargo periods to prevent data leakage in time series models.
"""

from .cross_validation import PurgedCrossValidator, TimeSeriesSplit
from .metrics import calculate_comprehensive_metrics, calculate_drawdown_metrics

__all__ = [
    'PurgedCrossValidator',
    'TimeSeriesSplit', 
    'calculate_comprehensive_metrics',
    'calculate_drawdown_metrics'
]
