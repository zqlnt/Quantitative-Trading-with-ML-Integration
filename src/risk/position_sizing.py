"""
Position Sizing and Risk Management for Neural Quant.

This module provides various position sizing algorithms and risk management
functions including Kelly Criterion, volatility targeting, and fixed percentage.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class PositionSizingConfig:
    """Configuration for position sizing."""
    method: str = "fixed_percentage"  # fixed_percentage, kelly, volatility_targeting, equal_weight
    max_position_pct: float = 0.1  # Maximum position size as percentage of portfolio
    kelly_cap: float = 0.2  # Maximum Kelly fraction
    volatility_target: float = 0.15  # Target volatility for volatility targeting
    min_position_pct: float = 0.01  # Minimum position size
    rebalance_threshold: float = 0.05  # Rebalance when drift exceeds this threshold


class PositionSizer:
    """
    Position sizing calculator with multiple algorithms.
    
    This class provides various position sizing methods including:
    - Fixed percentage
    - Kelly Criterion
    - Volatility targeting
    - Equal weight
    - Risk parity
    """
    
    def __init__(self, config: Optional[PositionSizingConfig] = None):
        """
        Initialize position sizer.
        
        Args:
            config: Position sizing configuration
        """
        self.config = config or PositionSizingConfig()
        self.logger = logging.getLogger(__name__)
    
    def calculate_position_size(self, signal: Dict[str, Any], 
                              portfolio_value: float,
                              current_prices: Dict[str, float],
                              historical_returns: Optional[pd.DataFrame] = None) -> float:
        """
        Calculate position size based on configured method.
        
        Args:
            signal: Trading signal with symbol, strength, etc.
            portfolio_value: Current portfolio value
            current_prices: Current prices for all symbols
            historical_returns: Historical returns for risk calculations
            
        Returns:
            float: Position size in shares
        """
        method = self.config.method.lower()
        
        if method == "fixed_percentage":
            return self._fixed_percentage_sizing(signal, portfolio_value, current_prices)
        elif method == "kelly":
            return self._kelly_sizing(signal, portfolio_value, current_prices, historical_returns)
        elif method == "volatility_targeting":
            return self._volatility_targeting_sizing(signal, portfolio_value, current_prices, historical_returns)
        elif method == "equal_weight":
            return self._equal_weight_sizing(signal, portfolio_value, current_prices)
        elif method == "risk_parity":
            return self._risk_parity_sizing(signal, portfolio_value, current_prices, historical_returns)
        else:
            self.logger.warning(f"Unknown position sizing method: {method}")
            return self._fixed_percentage_sizing(signal, portfolio_value, current_prices)
    
    def _fixed_percentage_sizing(self, signal: Dict[str, Any], 
                                portfolio_value: float,
                                current_prices: Dict[str, float]) -> float:
        """
        Fixed percentage position sizing.
        
        Args:
            signal: Trading signal
            portfolio_value: Portfolio value
            current_prices: Current prices
            
        Returns:
            float: Position size in shares
        """
        symbol = signal['symbol']
        current_price = current_prices.get(symbol, 0)
        
        if current_price <= 0:
            return 0
        
        # Calculate position value
        position_value = portfolio_value * self.config.max_position_pct
        
        # Calculate shares
        shares = position_value / current_price
        
        # Apply signal strength
        shares *= signal.get('strength', 1.0)
        
        return max(0, int(shares))
    
    def _kelly_sizing(self, signal: Dict[str, Any], 
                     portfolio_value: float,
                     current_prices: Dict[str, float],
                     historical_returns: Optional[pd.DataFrame] = None) -> float:
        """
        Kelly Criterion position sizing.
        
        Args:
            signal: Trading signal
            portfolio_value: Portfolio value
            current_prices: Current prices
            historical_returns: Historical returns
            
        Returns:
            float: Position size in shares
        """
        symbol = signal['symbol']
        current_price = current_prices.get(symbol, 0)
        
        if current_price <= 0 or historical_returns is None:
            return self._fixed_percentage_sizing(signal, portfolio_value, current_prices)
        
        # Calculate Kelly fraction
        kelly_fraction = self._calculate_kelly_fraction(symbol, historical_returns)
        
        # Cap Kelly fraction
        kelly_fraction = min(kelly_fraction, self.config.kelly_cap)
        
        # Calculate position value
        position_value = portfolio_value * kelly_fraction
        
        # Calculate shares
        shares = position_value / current_price
        
        # Apply signal strength
        shares *= signal.get('strength', 1.0)
        
        return max(0, int(shares))
    
    def _volatility_targeting_sizing(self, signal: Dict[str, Any], 
                                   portfolio_value: float,
                                   current_prices: Dict[str, float],
                                   historical_returns: Optional[pd.DataFrame] = None) -> float:
        """
        Volatility targeting position sizing.
        
        Args:
            signal: Trading signal
            portfolio_value: Portfolio value
            current_prices: Current prices
            historical_returns: Historical returns
            
        Returns:
            float: Position size in shares
        """
        symbol = signal['symbol']
        current_price = current_prices.get(symbol, 0)
        
        if current_price <= 0 or historical_returns is None:
            return self._fixed_percentage_sizing(signal, portfolio_value, current_prices)
        
        # Calculate asset volatility
        asset_returns = historical_returns[symbol].dropna()
        if len(asset_returns) < 20:  # Need sufficient data
            return self._fixed_percentage_sizing(signal, portfolio_value, current_prices)
        
        asset_volatility = asset_returns.std() * np.sqrt(252)  # Annualized
        
        if asset_volatility <= 0:
            return 0
        
        # Calculate volatility-adjusted position size
        volatility_ratio = self.config.volatility_target / asset_volatility
        
        # Calculate position value
        position_value = portfolio_value * volatility_ratio * self.config.max_position_pct
        
        # Calculate shares
        shares = position_value / current_price
        
        # Apply signal strength
        shares *= signal.get('strength', 1.0)
        
        return max(0, int(shares))
    
    def _equal_weight_sizing(self, signal: Dict[str, Any], 
                           portfolio_value: float,
                           current_prices: Dict[str, float]) -> float:
        """
        Equal weight position sizing.
        
        Args:
            signal: Trading signal
            portfolio_value: Portfolio value
            current_prices: Current prices
            
        Returns:
            float: Position size in shares
        """
        symbol = signal['symbol']
        current_price = current_prices.get(symbol, 0)
        
        if current_price <= 0:
            return 0
        
        # Calculate equal weight position value
        # Assume we want to hold N positions, so each gets 1/N of portfolio
        num_positions = 10  # Default number of positions
        position_value = portfolio_value / num_positions
        
        # Calculate shares
        shares = position_value / current_price
        
        # Apply signal strength
        shares *= signal.get('strength', 1.0)
        
        return max(0, int(shares))
    
    def _risk_parity_sizing(self, signal: Dict[str, Any], 
                          portfolio_value: float,
                          current_prices: Dict[str, float],
                          historical_returns: Optional[pd.DataFrame] = None) -> float:
        """
        Risk parity position sizing.
        
        Args:
            signal: Trading signal
            portfolio_value: Portfolio value
            current_prices: Current prices
            historical_returns: Historical returns
            
        Returns:
            float: Position size in shares
        """
        symbol = signal['symbol']
        current_price = current_prices.get(symbol, 0)
        
        if current_price <= 0 or historical_returns is None:
            return self._fixed_percentage_sizing(signal, portfolio_value, current_prices)
        
        # Calculate asset volatility
        asset_returns = historical_returns[symbol].dropna()
        if len(asset_returns) < 20:
            return self._fixed_percentage_sizing(signal, portfolio_value, current_prices)
        
        asset_volatility = asset_returns.std() * np.sqrt(252)
        
        if asset_volatility <= 0:
            return 0
        
        # Risk parity: equal risk contribution
        # Position size inversely proportional to volatility
        risk_contribution = 1.0 / asset_volatility
        
        # Normalize to get position weight
        # This is simplified - real risk parity requires solving optimization problem
        position_weight = risk_contribution / (risk_contribution + 1)  # Simplified normalization
        
        # Calculate position value
        position_value = portfolio_value * position_weight * self.config.max_position_pct
        
        # Calculate shares
        shares = position_value / current_price
        
        # Apply signal strength
        shares *= signal.get('strength', 1.0)
        
        return max(0, int(shares))
    
    def _calculate_kelly_fraction(self, symbol: str, 
                                 historical_returns: pd.DataFrame) -> float:
        """
        Calculate Kelly fraction for a symbol.
        
        Args:
            symbol: Trading symbol
            historical_returns: Historical returns
            
        Returns:
            float: Kelly fraction
        """
        returns = historical_returns[symbol].dropna()
        
        if len(returns) < 20:
            return 0.1  # Default conservative fraction
        
        # Calculate win rate and average win/loss
        positive_returns = returns[returns > 0]
        negative_returns = returns[returns < 0]
        
        if len(positive_returns) == 0 or len(negative_returns) == 0:
            return 0.1  # Default conservative fraction
        
        win_rate = len(positive_returns) / len(returns)
        avg_win = positive_returns.mean()
        avg_loss = abs(negative_returns.mean())
        
        if avg_loss == 0:
            return 0.1  # Default conservative fraction
        
        # Kelly formula: f = (bp - q) / b
        # where b = avg_win/avg_loss, p = win_rate, q = 1 - win_rate
        b = avg_win / avg_loss
        p = win_rate
        q = 1 - win_rate
        
        kelly_fraction = (b * p - q) / b
        
        # Ensure non-negative
        return max(0, kelly_fraction)
    
    def calculate_portfolio_risk(self, positions: Dict[str, float],
                               current_prices: Dict[str, float],
                               historical_returns: Optional[pd.DataFrame] = None) -> Dict[str, float]:
        """
        Calculate portfolio risk metrics.
        
        Args:
            positions: Current positions (symbol -> quantity)
            current_prices: Current prices
            historical_returns: Historical returns
            
        Returns:
            Dict[str, float]: Risk metrics
        """
        if not positions or historical_returns is None:
            return {'portfolio_volatility': 0, 'var_95': 0, 'max_drawdown': 0}
        
        # Calculate position values
        position_values = {
            symbol: quantity * current_prices.get(symbol, 0)
            for symbol, quantity in positions.items()
        }
        
        total_value = sum(position_values.values())
        
        if total_value == 0:
            return {'portfolio_volatility': 0, 'var_95': 0, 'max_drawdown': 0}
        
        # Calculate weights
        weights = {
            symbol: value / total_value
            for symbol, value in position_values.items()
        }
        
        # Calculate portfolio returns
        portfolio_returns = pd.Series(0, index=historical_returns.index)
        
        for symbol, weight in weights.items():
            if symbol in historical_returns.columns:
                symbol_returns = historical_returns[symbol].dropna()
                portfolio_returns += weight * symbol_returns
        
        # Calculate risk metrics
        portfolio_volatility = portfolio_returns.std() * np.sqrt(252)
        var_95 = np.percentile(portfolio_returns, 5)
        
        # Calculate max drawdown
        cumulative_returns = (1 + portfolio_returns).cumprod()
        running_max = cumulative_returns.expanding().max()
        drawdown = (cumulative_returns - running_max) / running_max
        max_drawdown = drawdown.min()
        
        return {
            'portfolio_volatility': portfolio_volatility,
            'var_95': var_95,
            'max_drawdown': max_drawdown,
            'num_positions': len(positions),
            'concentration_risk': max(weights.values()) if weights else 0
        }


def apply_stop_loss(signal: Dict[str, Any], 
                   current_price: float,
                   stop_loss_pct: float = 0.05) -> Optional[Dict[str, Any]]:
    """
    Apply stop loss to a signal.
    
    Args:
        signal: Original trading signal
        current_price: Current market price
        stop_loss_pct: Stop loss percentage
        
    Returns:
        Optional[Dict[str, Any]]: Modified signal with stop loss or None if stop hit
    """
    if signal['signal_type'] not in ['BUY', 'SELL']:
        return signal
    
    # Calculate stop loss price
    if signal['signal_type'] == 'BUY':
        stop_price = current_price * (1 - stop_loss_pct)
        # Check if current price is below stop loss
        if current_price <= stop_price:
            return None  # Stop loss hit
    else:  # SELL
        stop_price = current_price * (1 + stop_loss_pct)
        # Check if current price is above stop loss
        if current_price >= stop_price:
            return None  # Stop loss hit
    
    # Add stop loss information to signal
    modified_signal = signal.copy()
    modified_signal['stop_loss_price'] = stop_price
    modified_signal['stop_loss_pct'] = stop_loss_pct
    
    return modified_signal


def apply_take_profit(signal: Dict[str, Any], 
                     current_price: float,
                     take_profit_pct: float = 0.15) -> Optional[Dict[str, Any]]:
    """
    Apply take profit to a signal.
    
    Args:
        signal: Original trading signal
        current_price: Current market price
        take_profit_pct: Take profit percentage
        
    Returns:
        Optional[Dict[str, Any]]: Modified signal with take profit or None if target hit
    """
    if signal['signal_type'] not in ['BUY', 'SELL']:
        return signal
    
    # Calculate take profit price
    if signal['signal_type'] == 'BUY':
        target_price = current_price * (1 + take_profit_pct)
        # Check if current price is above target
        if current_price >= target_price:
            return None  # Take profit hit
    else:  # SELL
        target_price = current_price * (1 - take_profit_pct)
        # Check if current price is below target
        if current_price <= target_price:
            return None  # Take profit hit
    
    # Add take profit information to signal
    modified_signal = signal.copy()
    modified_signal['take_profit_price'] = target_price
    modified_signal['take_profit_pct'] = take_profit_pct
    
    return modified_signal
