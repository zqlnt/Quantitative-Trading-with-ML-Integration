"""
Neural Quant - A comprehensive quantitative trading framework.

This package provides tools for strategy development, backtesting, 
and live trading with MLflow integration and Streamlit dashboards.
"""

__version__ = "0.1.0"
__author__ = "Neural Quant Team"

# Core imports
from .core.backtest import Backtester
from .core.metrics import calculate_performance_metrics
from .strategies.momentum import MovingAverageCrossover
from .data.yf_loader import load_yf_data

__all__ = [
    "Backtester",
    "calculate_performance_metrics", 
    "MovingAverageCrossover",
    "load_yf_data",
]
