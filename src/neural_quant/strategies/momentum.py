"""Moving Average Crossover momentum strategy."""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)

class MovingAverageCrossover:
    """Moving Average Crossover momentum strategy."""
    
    def __init__(self, 
                 ma_fast: int = 10,
                 ma_slow: int = 30,
                 threshold: float = 0.0,
                 min_volume: float = 1000000,
                 max_positions: int = 5):
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
        self.current_positions = set()
        
    def generate_signals(self, data: pd.DataFrame) -> Dict[pd.Timestamp, Dict[str, str]]:
        """
        Generate trading signals based on moving average crossover.
        
        Args:
            data: OHLCV data
            
        Returns:
            Dictionary of signals by date
        """
        signals = {}
        
        # Get unique symbols from data
        symbols = self._extract_symbols(data)
        
        for symbol in symbols:
            symbol_data = self._get_symbol_data(data, symbol)
            if symbol_data.empty:
                continue
                
            # Calculate moving averages
            ma_fast = symbol_data['close'].rolling(window=self.ma_fast).mean()
            ma_slow = symbol_data['close'].rolling(window=self.ma_slow).mean()
            
            # Calculate crossover strength
            crossover_strength = (ma_fast - ma_slow) / ma_slow
            
            # Generate signals
            for date, row in symbol_data.iterrows():
                if pd.isna(ma_fast[date]) or pd.isna(ma_slow[date]):
                    continue
                
                # Convert timezone-aware timestamp to naive for comparison
                date_naive = date.tz_localize(None) if date.tz else date
                prev_date = date_naive - pd.Timedelta(days=1)
                
                if date_naive not in signals:
                    signals[date_naive] = {}
                
                # Check for buy signal (fast MA crosses above slow MA)
                if (ma_fast[date] > ma_slow[date] and 
                    ma_fast[date - pd.Timedelta(days=1)] <= ma_slow[date - pd.Timedelta(days=1)] and
                    crossover_strength[date] >= self.threshold and
                    row['volume'] >= self.min_volume and
                    len(self.current_positions) < self.max_positions):
                    
                    signals[date_naive][symbol] = 'BUY'
                    self.current_positions.add(symbol)
                    logger.info(f"BUY signal for {symbol} on {date_naive}")
                
                # Check for sell signal (fast MA crosses below slow MA)
                elif (ma_fast[date] < ma_slow[date] and 
                      ma_fast[date - pd.Timedelta(days=1)] >= ma_slow[date - pd.Timedelta(days=1)] and
                      symbol in self.current_positions):
                    
                    signals[date_naive][symbol] = 'SELL'
                    self.current_positions.remove(symbol)
                    logger.info(f"SELL signal for {symbol} on {date_naive}")
        
        return signals
    
    def _extract_symbols(self, data: pd.DataFrame) -> list:
        """Extract unique symbols from the data."""
        symbols = set()
        for col in data.columns:
            if '_close' in col:
                symbols.add(col.split('_close')[0])
        return list(symbols)
    
    def _get_symbol_data(self, data: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """Get data for a specific symbol."""
        symbol_cols = [col for col in data.columns if col.startswith(f"{symbol}_")]
        if not symbol_cols:
            return pd.DataFrame()
        
        symbol_data = data[symbol_cols].copy()
        symbol_data.columns = [col.split('_', 1)[1] for col in symbol_data.columns]
        
        return symbol_data
    
    def get_parameters(self) -> Dict[str, Any]:
        """Get strategy parameters."""
        return {
            'ma_fast': self.ma_fast,
            'ma_slow': self.ma_slow,
            'threshold': self.threshold,
            'min_volume': self.min_volume,
            'max_positions': self.max_positions
        }
