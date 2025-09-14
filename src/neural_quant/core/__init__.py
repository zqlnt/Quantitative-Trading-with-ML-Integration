"""Core modules for backtesting, metrics, and strategy execution."""

from .backtest import Backtester
from .metrics import calculate_performance_metrics

__all__ = ["Backtester", "calculate_performance_metrics"]
