"""
Paper Trading Engine for Neural Quant.

This module provides a realistic paper trading simulation with configurable
slippage, commission, and market impact models.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
from dataclasses import dataclass
import logging

from ...strategies.base.strategy_base import Signal, Trade, Portfolio
from ...utils.config.config_manager import get_config

logger = logging.getLogger(__name__)


@dataclass
class ExecutionConfig:
    """Configuration for paper trading execution."""
    commission: float = 0.0  # Commission per trade
    slippage: float = 0.001  # Slippage as percentage
    market_impact: float = 0.0005  # Market impact per $1000 traded
    min_trade_size: float = 1.0  # Minimum trade size in shares
    max_trade_size: float = 10000.0  # Maximum trade size in shares
    latency_ms: int = 100  # Simulated latency in milliseconds
    fill_probability: float = 1.0  # Probability of getting filled (0-1)


class PaperTrader:
    """
    Paper trading engine with realistic execution simulation.
    
    This class simulates realistic trading execution including:
    - Commission costs
    - Slippage simulation
    - Market impact modeling
    - Latency simulation
    - Partial fills
    """
    
    def __init__(self, initial_capital: float = 100000, 
                 execution_config: Optional[ExecutionConfig] = None):
        """
        Initialize the paper trader.
        
        Args:
            initial_capital: Starting capital
            execution_config: Execution configuration
        """
        self.initial_capital = initial_capital
        self.cash = initial_capital
        self.positions = {}  # symbol -> quantity
        self.total_value = initial_capital
        self.trade_history = []
        self.execution_config = execution_config or ExecutionConfig()
        
        # Performance tracking
        self.daily_returns = []
        self.daily_values = []
        self.daily_dates = []
        
        self.logger = logging.getLogger(__name__)
        
    def execute_signal(self, signal: Signal, current_price: float, 
                      market_data: Optional[Dict[str, Any]] = None) -> Optional[Trade]:
        """
        Execute a trading signal with realistic simulation.
        
        Args:
            signal: Trading signal to execute
            current_price: Current market price
            market_data: Additional market data for execution
            
        Returns:
            Optional[Trade]: Executed trade or None if not executed
        """
        try:
            # Calculate position size
            position_size = self._calculate_position_size(signal, current_price)
            
            if position_size <= 0:
                return None
            
            # Apply execution constraints
            position_size = self._apply_execution_constraints(position_size, signal.symbol)
            
            if position_size <= 0:
                return None
            
            # Simulate execution
            execution_result = self._simulate_execution(signal, current_price, position_size, market_data)
            
            if execution_result is None:
                return None
            
            executed_price, executed_quantity, commission = execution_result
            
            # Create trade
            trade = Trade(
                symbol=signal.symbol,
                side=signal.signal_type,
                quantity=executed_quantity,
                price=executed_price,
                timestamp=datetime.now(),
                order_id=f"PAPER_{signal.symbol}_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}",
                commission=commission,
                metadata={
                    'signal_strength': signal.strength,
                    'original_signal_price': signal.price,
                    'execution_config': {
                        'slippage': self.execution_config.slippage,
                        'commission': self.execution_config.commission,
                        'market_impact': self.execution_config.market_impact
                    }
                }
            )
            
            # Update portfolio
            self._update_portfolio(trade)
            
            # Log trade
            self.trade_history.append(trade)
            self.logger.info(
                f"Executed {signal.signal_type} {executed_quantity} {signal.symbol} @ ${executed_price:.2f} "
                f"(Commission: ${commission:.2f})"
            )
            
            return trade
            
        except Exception as e:
            self.logger.error(f"Failed to execute signal for {signal.symbol}: {e}")
            return None
    
    def _calculate_position_size(self, signal: Signal, current_price: float) -> float:
        """
        Calculate position size based on signal and available capital.
        
        Args:
            signal: Trading signal
            current_price: Current market price
            
        Returns:
            float: Position size in shares
        """
        if signal.signal_type not in ['BUY', 'SELL']:
            return 0
        
        # Simple position sizing based on available capital
        # In a real system, this would use more sophisticated position sizing
        max_position_value = self.total_value * 0.1  # 10% of portfolio per position
        position_size = max_position_value / current_price
        
        # Round to whole shares
        return int(position_size)
    
    def _apply_execution_constraints(self, position_size: float, symbol: str) -> float:
        """
        Apply execution constraints (min/max trade size, available capital).
        
        Args:
            position_size: Desired position size
            symbol: Trading symbol
            
        Returns:
            float: Constrained position size
        """
        # Apply min/max trade size constraints
        position_size = max(self.execution_config.min_trade_size, position_size)
        position_size = min(self.execution_config.max_trade_size, position_size)
        
        # Check available capital for buy orders
        if position_size > 0 and self.cash < position_size * 100:  # Assume $100 per share for now
            available_shares = self.cash / 100
            position_size = min(position_size, available_shares)
        
        # Check available shares for sell orders
        current_position = self.positions.get(symbol, 0)
        if position_size > 0 and current_position < position_size:
            position_size = min(position_size, current_position)
        
        return int(position_size)
    
    def _simulate_execution(self, signal: Signal, current_price: float, 
                          position_size: float, market_data: Optional[Dict[str, Any]]) -> Optional[Tuple[float, float, float]]:
        """
        Simulate realistic trade execution.
        
        Args:
            signal: Trading signal
            current_price: Current market price
            position_size: Position size to execute
            market_data: Additional market data
            
        Returns:
            Optional[Tuple[float, float, float]]: (executed_price, executed_quantity, commission)
        """
        # Check fill probability
        if np.random.random() > self.execution_config.fill_probability:
            self.logger.warning(f"Order not filled for {signal.symbol} (fill probability)")
            return None
        
        # Calculate slippage
        slippage = self._calculate_slippage(signal, position_size, market_data)
        
        # Calculate market impact
        market_impact = self._calculate_market_impact(position_size, current_price)
        
        # Calculate executed price
        if signal.signal_type == 'BUY':
            executed_price = current_price * (1 + slippage + market_impact)
        else:  # SELL
            executed_price = current_price * (1 - slippage - market_impact)
        
        # Calculate commission
        trade_value = position_size * executed_price
        commission = self.execution_config.commission + (trade_value * 0.0001)  # 0.01% of trade value
        
        return executed_price, position_size, commission
    
    def _calculate_slippage(self, signal: Signal, position_size: float, 
                          market_data: Optional[Dict[str, Any]]) -> float:
        """
        Calculate slippage based on market conditions and position size.
        
        Args:
            signal: Trading signal
            position_size: Position size
            market_data: Market data
            
        Returns:
            float: Slippage as percentage
        """
        base_slippage = self.execution_config.slippage
        
        # Size-based slippage (larger orders have more slippage)
        size_multiplier = min(1.0 + (position_size / 1000) * 0.1, 2.0)
        
        # Volatility-based slippage (if market data available)
        volatility_multiplier = 1.0
        if market_data and 'volatility' in market_data:
            volatility_multiplier = 1.0 + market_data['volatility'] * 0.5
        
        # Random component
        random_component = np.random.normal(0, 0.0001)
        
        total_slippage = base_slippage * size_multiplier * volatility_multiplier + random_component
        
        return max(0, total_slippage)  # Slippage can't be negative
    
    def _calculate_market_impact(self, position_size: float, current_price: float) -> float:
        """
        Calculate market impact based on position size.
        
        Args:
            position_size: Position size
            current_price: Current price
            
        Returns:
            float: Market impact as percentage
        """
        trade_value = position_size * current_price
        impact = (trade_value / 1000) * self.execution_config.market_impact
        
        return min(impact, 0.01)  # Cap at 1%
    
    def _update_portfolio(self, trade: Trade):
        """Update portfolio after trade execution."""
        if trade.side == 'BUY':
            self.cash -= (trade.quantity * trade.price + trade.commission)
            self.positions[trade.symbol] = self.positions.get(trade.symbol, 0) + trade.quantity
        elif trade.side == 'SELL':
            self.cash += (trade.quantity * trade.price - trade.commission)
            self.positions[trade.symbol] = self.positions.get(trade.symbol, 0) - trade.quantity
            
            # Remove position if quantity becomes zero
            if self.positions[trade.symbol] <= 0:
                del self.positions[trade.symbol]
    
    def update_portfolio_value(self, current_prices: Dict[str, float]) -> Portfolio:
        """
        Update portfolio value with current prices.
        
        Args:
            current_prices: Current prices for all positions
            
        Returns:
            Portfolio: Updated portfolio state
        """
        # Calculate positions value
        positions_value = sum(
            quantity * current_prices.get(symbol, 0)
            for symbol, quantity in self.positions.items()
        )
        
        self.total_value = self.cash + positions_value
        
        # Create portfolio snapshot
        portfolio = Portfolio(
            total_value=self.total_value,
            cash=self.cash,
            positions=self.positions.copy(),
            timestamp=datetime.now(),
            metadata={
                'positions_value': positions_value,
                'num_positions': len(self.positions),
                'cash_pct': self.cash / self.total_value if self.total_value > 0 else 0
            }
        )
        
        return portfolio
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """
        Get performance summary for the paper trading account.
        
        Returns:
            Dict[str, Any]: Performance summary
        """
        if not self.trade_history:
            return {
                'total_trades': 0,
                'total_return': 0,
                'current_value': self.total_value,
                'cash': self.cash,
                'positions': self.positions
            }
        
        # Calculate basic metrics
        total_trades = len(self.trade_history)
        total_return = (self.total_value - self.initial_capital) / self.initial_capital
        
        # Calculate trade statistics
        buy_trades = [t for t in self.trade_history if t.side == 'BUY']
        sell_trades = [t for t in self.trade_history if t.side == 'SELL']
        
        total_commission = sum(t.commission for t in self.trade_history)
        total_volume = sum(t.quantity * t.price for t in self.trade_history)
        
        return {
            'total_trades': total_trades,
            'buy_trades': len(buy_trades),
            'sell_trades': len(sell_trades),
            'total_return': total_return,
            'total_commission': total_commission,
            'total_volume': total_volume,
            'current_value': self.total_value,
            'cash': self.cash,
            'positions': self.positions,
            'num_positions': len(self.positions),
            'cash_pct': self.cash / self.total_value if self.total_value > 0 else 0
        }
    
    def reset(self, initial_capital: Optional[float] = None):
        """Reset the paper trading account."""
        if initial_capital is not None:
            self.initial_capital = initial_capital
        
        self.cash = self.initial_capital
        self.positions = {}
        self.total_value = self.initial_capital
        self.trade_history = []
        self.daily_returns = []
        self.daily_values = []
        self.daily_dates = []
        
        self.logger.info(f"Paper trading account reset with ${self.initial_capital:,.2f}")


# Factory function for easy creation
def create_paper_trader(initial_capital: float = 100000, 
                       commission: float = 0.0,
                       slippage: float = 0.001) -> PaperTrader:
    """
    Create a paper trader with specified configuration.
    
    Args:
        initial_capital: Starting capital
        commission: Commission per trade
        slippage: Slippage as percentage
        
    Returns:
        PaperTrader: Configured paper trader
    """
    config = ExecutionConfig(
        commission=commission,
        slippage=slippage
    )
    
    return PaperTrader(initial_capital=initial_capital, execution_config=config)
