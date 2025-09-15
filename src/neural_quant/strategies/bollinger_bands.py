"""
Bollinger Bands Strategy - Mean Reversion

This strategy implements mean reversion logic using Bollinger Bands:
- Long when price < lower band (oversold)
- Short when price > upper band (overbought)
- Flat when price is between bands
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional
from .base.strategy_base import StrategyBase


class BollingerBandsStrategy(StrategyBase):
    """
    Bollinger Bands mean reversion strategy.
    
    Parameters:
    - window: Rolling window for moving average calculation (default: 20)
    - num_std: Number of standard deviations for band width (default: 2.0)
    """
    
    def __init__(self, window: int = 20, num_std: float = 2.0, **kwargs):
        super().__init__(**kwargs)
        self.window = window
        self.num_std = num_std
        self.name = f"BollingerBands_{window}_{num_std}"
        
    def calculate_bollinger_bands(self, prices: pd.Series) -> tuple[pd.Series, pd.Series, pd.Series]:
        """
        Calculate Bollinger Bands for given price series.
        
        Args:
            prices: Price series (typically close prices)
            
        Returns:
            tuple: (upper_band, middle_band, lower_band)
        """
        middle_band = prices.rolling(window=self.window).mean()
        std = prices.rolling(window=self.window).std()
        
        upper_band = middle_band + (self.num_std * std)
        lower_band = middle_band - (self.num_std * std)
        
        return upper_band, middle_band, lower_band
    
    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """
        Generate trading signals based on Bollinger Bands.
        
        Args:
            data: DataFrame with OHLCV data, must include 'close' column
            
        Returns:
            Series of signals: +1 (long), -1 (short), 0 (flat)
        """
        if 'close' not in data.columns:
            raise ValueError("Data must contain 'close' column for Bollinger Bands strategy")
        
        close_prices = data['close']
        
        # Calculate Bollinger Bands
        upper_band, middle_band, lower_band = self.calculate_bollinger_bands(close_prices)
        
        # Generate signals
        signals = pd.Series(0, index=data.index, name='signal')
        
        # Long signal: price below lower band (oversold)
        long_condition = close_prices < lower_band
        
        # Short signal: price above upper band (overbought)
        short_condition = close_prices > upper_band
        
        # Apply signals
        signals[long_condition] = 1
        signals[short_condition] = -1
        
        # Ensure we have enough data for rolling calculations
        signals.iloc[:self.window-1] = 0
        
        return signals
    
    def get_strategy_params(self) -> Dict[str, Any]:
        """Return strategy parameters for logging."""
        return {
            'strategy': 'BollingerBands',
            'window': self.window,
            'num_std': self.num_std
        }
    
    def validate_parameters(self) -> bool:
        """Validate strategy parameters."""
        if self.window <= 0:
            raise ValueError("Window must be positive")
        if self.num_std <= 0:
            raise ValueError("Number of standard deviations must be positive")
        return True


