"""
Volatility Breakout Strategy - ATR-based

This strategy implements breakout logic using Average True Range (ATR):
- Long when close > yesterday's close + multiplier × ATR
- Short when close < yesterday's close - multiplier × ATR
- Flat otherwise
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional
from .base.strategy_base import StrategyBase


class VolatilityBreakoutStrategy(StrategyBase):
    """
    Volatility Breakout strategy using ATR.
    
    Parameters:
    - atr_window: Rolling window for ATR calculation (default: 14)
    - multiplier: ATR multiplier for breakout threshold (default: 1.5)
    """
    
    def __init__(self, atr_window: int = 14, multiplier: float = 1.5, **kwargs):
        super().__init__(**kwargs)
        self.atr_window = atr_window
        self.multiplier = multiplier
        self.name = f"VolatilityBreakout_{atr_window}_{multiplier}"
        
    def calculate_atr(self, data: pd.DataFrame) -> pd.Series:
        """
        Calculate Average True Range (ATR) for given OHLC data.
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            Series of ATR values
        """
        if not all(col in data.columns for col in ['high', 'low', 'close']):
            raise ValueError("Data must contain 'high', 'low', and 'close' columns for ATR calculation")
        
        high = data['high']
        low = data['low']
        close = data['close']
        
        # Calculate True Range components
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        
        # True Range is the maximum of the three components
        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        
        # ATR is the rolling mean of True Range
        atr = true_range.rolling(window=self.atr_window).mean()
        
        return atr
    
    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """
        Generate trading signals based on volatility breakout.
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            Series of signals: +1 (long), -1 (short), 0 (flat)
        """
        if 'close' not in data.columns:
            raise ValueError("Data must contain 'close' column for Volatility Breakout strategy")
        
        close_prices = data['close']
        
        # Calculate ATR
        atr = self.calculate_atr(data)
        
        # Calculate previous close
        prev_close = close_prices.shift(1)
        
        # Generate signals
        signals = pd.Series(0, index=data.index, name='signal')
        
        # Long signal: close > prev_close + multiplier × ATR
        long_condition = close_prices > (prev_close + self.multiplier * atr)
        
        # Short signal: close < prev_close - multiplier × ATR
        short_condition = close_prices < (prev_close - self.multiplier * atr)
        
        # Apply signals
        signals[long_condition] = 1
        signals[short_condition] = -1
        
        # Ensure we have enough data for ATR calculation
        signals.iloc[:self.atr_window] = 0
        
        return signals
    
    def get_strategy_params(self) -> Dict[str, Any]:
        """Return strategy parameters for logging."""
        return {
            'strategy': 'VolatilityBreakout',
            'atr_window': self.atr_window,
            'multiplier': self.multiplier
        }
    
    def validate_parameters(self) -> bool:
        """Validate strategy parameters."""
        if self.atr_window <= 0:
            raise ValueError("ATR window must be positive")
        if self.multiplier <= 0:
            raise ValueError("ATR multiplier must be positive")
        return True


