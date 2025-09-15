"""
Trade Logger for Neural Quant

This module provides logging functionality for trades, signals, and performance metrics.
"""

import logging
from typing import Dict, Any, Optional
from datetime import datetime

logger = logging.getLogger(__name__)

class TradeLogger:
    """Logger for trading activities."""
    
    def __init__(self):
        """Initialize the trade logger."""
        self.trades = []
        self.signals = []
        self.performance_metrics = []
        self.portfolio_snapshots = []
    
    def log_trade(self, 
                  strategy_name: str,
                  symbol: str,
                  side: str,
                  quantity: float,
                  price: float,
                  order_id: Optional[str] = None,
                  commission: float = 0.0,
                  metadata: Optional[Dict[str, Any]] = None):
        """
        Log a trade.
        
        Args:
            strategy_name: Name of the strategy
            symbol: Trading symbol
            side: BUY or SELL
            quantity: Number of shares
            price: Price per share
            order_id: Order identifier
            commission: Commission paid
            metadata: Additional trade metadata
        """
        trade = {
            'timestamp': datetime.now(),
            'strategy_name': strategy_name,
            'symbol': symbol,
            'side': side,
            'quantity': quantity,
            'price': price,
            'order_id': order_id,
            'commission': commission,
            'metadata': metadata or {}
        }
        self.trades.append(trade)
        logger.info(f"Trade logged: {side} {quantity} {symbol} @ {price}")
    
    def log_signal(self,
                   strategy_name: str,
                   symbol: str,
                   signal_type: str,
                   strength: float,
                   price: float,
                   metadata: Optional[Dict[str, Any]] = None):
        """
        Log a trading signal.
        
        Args:
            strategy_name: Name of the strategy
            symbol: Trading symbol
            signal_type: Type of signal (BUY, SELL, HOLD)
            strength: Signal strength (0-1)
            price: Current price
            metadata: Additional signal metadata
        """
        signal = {
            'timestamp': datetime.now(),
            'strategy_name': strategy_name,
            'symbol': symbol,
            'signal_type': signal_type,
            'strength': strength,
            'price': price,
            'metadata': metadata or {}
        }
        self.signals.append(signal)
        logger.info(f"Signal logged: {signal_type} {symbol} @ {price} (strength: {strength})")
    
    def log_performance_metric(self,
                              strategy_name: str,
                              metric_name: str,
                              metric_value: float):
        """
        Log a performance metric.
        
        Args:
            strategy_name: Name of the strategy
            metric_name: Name of the metric
            metric_value: Value of the metric
        """
        metric = {
            'timestamp': datetime.now(),
            'strategy_name': strategy_name,
            'metric_name': metric_name,
            'metric_value': metric_value
        }
        self.performance_metrics.append(metric)
        logger.debug(f"Performance metric logged: {metric_name} = {metric_value}")
    
    def log_portfolio_snapshot(self,
                              strategy_name: str,
                              total_value: float,
                              cash: float,
                              positions: Dict[str, float],
                              metadata: Optional[Dict[str, Any]] = None):
        """
        Log a portfolio snapshot.
        
        Args:
            strategy_name: Name of the strategy
            total_value: Total portfolio value
            cash: Cash balance
            positions: Dictionary of positions (symbol -> quantity)
            metadata: Additional portfolio metadata
        """
        snapshot = {
            'timestamp': datetime.now(),
            'strategy_name': strategy_name,
            'total_value': total_value,
            'cash': cash,
            'positions': positions,
            'metadata': metadata or {}
        }
        self.portfolio_snapshots.append(snapshot)
        logger.debug(f"Portfolio snapshot logged: {total_value} total value")
    
    def get_trades(self) -> list:
        """Get all logged trades."""
        return self.trades.copy()
    
    def get_signals(self) -> list:
        """Get all logged signals."""
        return self.signals.copy()
    
    def get_performance_metrics(self) -> list:
        """Get all logged performance metrics."""
        return self.performance_metrics.copy()
    
    def get_portfolio_snapshots(self) -> list:
        """Get all logged portfolio snapshots."""
        return self.portfolio_snapshots.copy()
    
    def clear(self):
        """Clear all logged data."""
        self.trades.clear()
        self.signals.clear()
        self.performance_metrics.clear()
        self.portfolio_snapshots.clear()

# Global instance
_trade_logger = None

def get_trade_logger() -> TradeLogger:
    """Get the global trade logger instance."""
    global _trade_logger
    if _trade_logger is None:
        _trade_logger = TradeLogger()
    return _trade_logger


