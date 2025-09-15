"""Moving Average Crossover momentum strategy."""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List
from .base.strategy_base import StrategyBase, Signal
from datetime import datetime

class MovingAverageCrossover(StrategyBase):
    """Moving Average Crossover momentum strategy."""
    
    def __init__(self, 
                 ma_fast: int = 10,
                 ma_slow: int = 30,
                 threshold: float = 0.0,
                 min_volume: float = 1000000,
                 max_positions: int = 5,
                 **kwargs):
        """
        Initialize the strategy.
        
        Args:
            ma_fast: Fast moving average period
            ma_slow: Slow moving average period
            threshold: Minimum crossover strength threshold
            min_volume: Minimum volume filter
            max_positions: Maximum number of positions
        """
        self.ma_fast = ma_fast
        self.ma_slow = ma_slow
        self.threshold = threshold
        self.min_volume = min_volume
        self.max_positions = max_positions
        
        # Initialize base class
        super().__init__(
            name=f"MovingAverageCrossover_{ma_fast}_{ma_slow}_{threshold}",
            parameters={
                'ma_fast': ma_fast,
                'ma_slow': ma_slow,
                'threshold': threshold,
                'min_volume': min_volume,
                'max_positions': max_positions
            },
            **kwargs
        )
        
    def _initialize_strategy(self, data: pd.DataFrame):
        """Initialize the strategy with historical data."""
        # No specific initialization needed for MA crossover
        self.logger.info(f"Initialized MovingAverageCrossover strategy with {len(data)} data points")
    
    def _generate_signals(self, data: pd.DataFrame) -> List[Signal]:
        """
        Generate trading signals based on moving average crossover.
        
        Args:
            data: OHLCV data with 'close' column
            
        Returns:
            List of Signal objects
        """
        signals = []
        
        if 'close' not in data.columns:
            return signals
        
        # Calculate moving averages
        ma_fast = data['close'].rolling(window=self.ma_fast).mean()
        ma_slow = data['close'].rolling(window=self.ma_slow).mean()
        
        # Calculate crossover strength
        crossover_strength = (ma_fast - ma_slow) / ma_slow
        
        # Generate signals for each day
        for i in range(1, len(data)):  # Start from index 1 to have previous day
            date = data.index[i]
            row = data.iloc[i]
            
            if pd.isna(ma_fast.iloc[i]) or pd.isna(ma_slow.iloc[i]):
                continue
            
            # Get previous day's values
            ma_fast_prev = ma_fast.iloc[i-1]
            ma_slow_prev = ma_slow.iloc[i-1]
            
            if pd.isna(ma_fast_prev) or pd.isna(ma_slow_prev):
                continue
            
            # Check for buy signal (fast MA crosses above slow MA)
            if (ma_fast.iloc[i] > ma_slow.iloc[i] and 
                ma_fast_prev <= ma_slow_prev and
                crossover_strength.iloc[i] >= self.threshold and
                row.get('volume', 0) >= self.min_volume):
                
                signal = Signal(
                    symbol='SYMBOL',  # Will be set by the caller
                    signal_type='BUY',
                    strength=min(crossover_strength.iloc[i], 1.0),
                    price=row['close'],
                    timestamp=date,
                    metadata={
                        'ma_fast': ma_fast.iloc[i],
                        'ma_slow': ma_slow.iloc[i],
                        'crossover_strength': crossover_strength.iloc[i]
                    }
                )
                signals.append(signal)
            
            # Check for sell signal (fast MA crosses below slow MA)
            elif (ma_fast.iloc[i] < ma_slow.iloc[i] and 
                  ma_fast_prev >= ma_slow_prev):
                
                signal = Signal(
                    symbol='SYMBOL',  # Will be set by the caller
                    signal_type='SELL',
                    strength=min(abs(crossover_strength.iloc[i]), 1.0),
                    price=row['close'],
                    timestamp=date,
                    metadata={
                        'ma_fast': ma_fast.iloc[i],
                        'ma_slow': ma_slow.iloc[i],
                        'crossover_strength': crossover_strength.iloc[i]
                    }
                )
                signals.append(signal)
        
        return signals
    
    def get_strategy_params(self) -> Dict[str, Any]:
        """Return strategy parameters for logging."""
        return {
            'strategy': 'MovingAverageCrossover',
            'ma_fast': self.ma_fast,
            'ma_slow': self.ma_slow,
            'threshold': self.threshold,
            'min_volume': self.min_volume,
            'max_positions': self.max_positions
        }
    
    def validate_parameters(self) -> bool:
        """Validate strategy parameters."""
        if self.ma_fast <= 0:
            raise ValueError("Fast MA period must be positive")
        if self.ma_slow <= 0:
            raise ValueError("Slow MA period must be positive")
        if self.ma_fast >= self.ma_slow:
            raise ValueError("Fast MA period must be less than slow MA period")
        if self.threshold < 0:
            raise ValueError("Threshold must be non-negative")
        if self.min_volume < 0:
            raise ValueError("Minimum volume must be non-negative")
        if self.max_positions <= 0:
            raise ValueError("Maximum positions must be positive")
        return True
