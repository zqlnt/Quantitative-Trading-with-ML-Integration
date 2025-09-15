"""
Cross-Sectional Momentum Strategy

This strategy implements cross-sectional momentum by ranking all selected tickers
by their returns over a lookback window and taking long positions in top N and
short positions in bottom N tickers.

This strategy requires portfolio-level data and cannot be used for single-ticker backtesting.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List
from .base.strategy_base import StrategyBase


class CrossSectionalMomentumStrategy(StrategyBase):
    """
    Cross-Sectional Momentum strategy for portfolio trading.
    
    Parameters:
    - lookback_window: Number of periods to look back for return calculation (default: 20)
    - top_n: Number of top performers to go long (default: 3)
    - bottom_n: Number of bottom performers to go short (default: 3)
    """
    
    def __init__(self, lookback_window: int = 20, top_n: int = 3, bottom_n: int = 3, **kwargs):
        super().__init__(**kwargs)
        self.lookback_window = lookback_window
        self.top_n = top_n
        self.bottom_n = bottom_n
        self.name = f"CrossSectionalMomentum_{lookback_window}_{top_n}_{bottom_n}"
        
    def calculate_returns(self, prices: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate returns for all tickers over the lookback window.
        
        Args:
            prices: DataFrame with close prices for all tickers (columns = tickers)
            
        Returns:
            DataFrame of returns for each ticker
        """
        # Calculate returns over the lookback window
        returns = prices.pct_change(periods=self.lookback_window)
        return returns
    
    def rank_tickers(self, returns: pd.DataFrame) -> pd.DataFrame:
        """
        Rank tickers by their returns (descending order).
        
        Args:
            returns: DataFrame of returns for each ticker
            
        Returns:
            DataFrame with ranks (1 = best, higher numbers = worse)
        """
        # Rank returns in descending order (best performance = rank 1)
        ranks = returns.rank(axis=1, method='dense', ascending=False)
        return ranks
    
    def generate_portfolio_signals(self, prices: pd.DataFrame) -> pd.DataFrame:
        """
        Generate portfolio-level signals for all tickers.
        
        Args:
            prices: DataFrame with close prices for all tickers (columns = tickers)
            
        Returns:
            DataFrame of signals: +1 (long), -1 (short), 0 (flat) for each ticker
        """
        if prices.empty:
            raise ValueError("Price data is empty")
        
        # Calculate returns
        returns = self.calculate_returns(prices)
        
        # Rank tickers by performance
        ranks = self.rank_tickers(returns)
        
        # Initialize signals DataFrame
        signals = pd.DataFrame(0, index=prices.index, columns=prices.columns)
        
        # Generate signals for each time period
        for date in prices.index:
            if date in ranks.index:
                # Get ranks for this date (exclude NaN values)
                date_ranks = ranks.loc[date].dropna()
                
                if len(date_ranks) < (self.top_n + self.bottom_n):
                    # Not enough tickers for the strategy
                    continue
                
                # Long positions: top N performers
                top_performers = date_ranks.nsmallest(self.top_n).index
                signals.loc[date, top_performers] = 1
                
                # Short positions: bottom N performers
                bottom_performers = date_ranks.nlargest(self.bottom_n).index
                signals.loc[date, bottom_performers] = -1
        
        return signals
    
    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """
        Generate signals for single ticker (not applicable for cross-sectional strategy).
        
        This method raises an error because cross-sectional momentum requires
        portfolio-level data across multiple tickers.
        """
        raise NotImplementedError(
            "Cross-Sectional Momentum strategy requires portfolio-level data. "
            "Use generate_portfolio_signals() method instead."
        )
    
    def get_strategy_params(self) -> Dict[str, Any]:
        """Return strategy parameters for logging."""
        return {
            'strategy': 'CrossSectionalMomentum',
            'lookback_window': self.lookback_window,
            'top_n': self.top_n,
            'bottom_n': self.bottom_n
        }
    
    def validate_parameters(self) -> bool:
        """Validate strategy parameters."""
        if self.lookback_window <= 0:
            raise ValueError("Lookback window must be positive")
        if self.top_n <= 0:
            raise ValueError("Top N must be positive")
        if self.bottom_n <= 0:
            raise ValueError("Bottom N must be positive")
        if self.top_n + self.bottom_n > 12:  # Assuming max 12 tickers
            raise ValueError("Top N + Bottom N cannot exceed total number of tickers")
        return True
    
    def is_portfolio_strategy(self) -> bool:
        """Return True if this strategy requires portfolio-level data."""
        return True


