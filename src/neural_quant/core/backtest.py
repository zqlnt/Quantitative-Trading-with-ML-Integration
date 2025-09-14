"""Backtesting engine for trading strategies."""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any
import mlflow
import mlflow.sklearn
from datetime import datetime

class Backtester:
    """High-fidelity backtesting engine with realistic transaction costs."""
    
    def __init__(self, 
                 initial_capital: float = 100000.0,
                 commission: float = 0.001,
                 slippage: float = 0.0005):
        """
        Initialize the backtester.
        
        Args:
            initial_capital: Starting capital
            commission: Commission rate per trade
            slippage: Slippage rate per trade
        """
        self.initial_capital = initial_capital
        self.commission = commission
        self.slippage = slippage
        self.reset()
    
    def reset(self):
        """Reset the backtester state."""
        self.capital = self.initial_capital
        self.positions = {}
        self.trades = []
        self.equity_curve = []
        self.current_date = None
    
    def run_backtest(self, 
                    data: pd.DataFrame, 
                    strategy, 
                    start_date: Optional[str] = None,
                    end_date: Optional[str] = None) -> Dict[str, Any]:
        """
        Run a backtest on the given data with the specified strategy.
        
        Args:
            data: Price data with OHLCV columns
            strategy: Strategy instance with generate_signals method
            start_date: Start date for backtest
            end_date: End date for backtest
            
        Returns:
            Dictionary containing backtest results
        """
        self.reset()
        
        # Filter data by date range
        if start_date:
            data = data[data.index >= start_date]
        if end_date:
            data = data[data.index <= end_date]
        
        # Generate signals
        signals = strategy.generate_signals(data)
        
        # Process each day
        for date, row in data.iterrows():
            # Convert timezone-aware timestamp to naive for consistency
            self.current_date = date.tz_localize(None) if date.tz else date
            self._process_signals(self.current_date, row, signals)
            self._update_equity_curve(self.current_date, row)
        
        # Calculate performance metrics
        results = self._calculate_results(data)
        
        # Log to MLflow
        self._log_to_mlflow(results, strategy)
        
        return results
    
    def _process_signals(self, date, row, signals):
        """Process trading signals for the current date."""
        if date not in signals:
            return
        
        signal = signals[date]
        
        for symbol, action in signal.items():
            if action == 'BUY' and symbol not in self.positions:
                self._open_position(symbol, row[f'{symbol}_close'] if f'{symbol}_close' in row else row['close'])
            elif action == 'SELL' and symbol in self.positions:
                self._close_position(symbol, row[f'{symbol}_close'] if f'{symbol}_close' in row else row['close'])
    
    def _open_position(self, symbol: str, price: float):
        """Open a new position."""
        # Apply slippage
        entry_price = price * (1 + self.slippage)
        
        # Calculate position size (simple equal weight for now)
        position_value = self.capital * 0.1  # 10% per position
        shares = int(position_value / entry_price)
        
        if shares > 0:
            cost = shares * entry_price * (1 + self.commission)
            if cost <= self.capital:
                self.positions[symbol] = {
                    'shares': shares,
                    'entry_price': entry_price,
                    'entry_date': self.current_date
                }
                self.capital -= cost
    
    def _close_position(self, symbol: str, price: float):
        """Close an existing position."""
        if symbol not in self.positions:
            return
        
        position = self.positions[symbol]
        exit_price = price * (1 - self.slippage)
        
        # Calculate P&L
        pnl = (exit_price - position['entry_price']) * position['shares']
        proceeds = position['shares'] * exit_price * (1 - self.commission)
        
        # Record trade
        trade = {
            'symbol': symbol,
            'entry_date': position['entry_date'],
            'exit_date': self.current_date,
            'shares': position['shares'],
            'entry_price': position['entry_price'],
            'exit_price': exit_price,
            'pnl': pnl,
            'return': pnl / (position['shares'] * position['entry_price'])
        }
        self.trades.append(trade)
        
        # Update capital
        self.capital += proceeds
        
        # Remove position
        del self.positions[symbol]
    
    def _update_equity_curve(self, date, row):
        """Update the equity curve with current portfolio value."""
        total_value = self.capital
        
        # Add position values
        for symbol, position in self.positions.items():
            current_price = row[f'{symbol}_close'] if f'{symbol}_close' in row else row['close']
            total_value += position['shares'] * current_price
        
        self.equity_curve.append({
            'date': date,
            'equity': total_value,
            'cash': self.capital,
            'positions': len(self.positions)
        })
    
    def _calculate_results(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Calculate performance metrics."""
        if not self.equity_curve:
            return {}
        
        equity_df = pd.DataFrame(self.equity_curve).set_index('date')
        equity_df['returns'] = equity_df['equity'].pct_change()
        
        # Calculate metrics
        total_return = (equity_df['equity'].iloc[-1] / self.initial_capital) - 1
        annualized_return = (1 + total_return) ** (252 / len(equity_df)) - 1
        volatility = equity_df['returns'].std() * np.sqrt(252)
        sharpe_ratio = annualized_return / volatility if volatility > 0 else 0
        
        # Max drawdown
        rolling_max = equity_df['equity'].expanding().max()
        drawdown = (equity_df['equity'] - rolling_max) / rolling_max
        max_drawdown = drawdown.min()
        
        return {
            'total_return': total_return,
            'annualized_return': annualized_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'total_trades': len(self.trades),
            'win_rate': len([t for t in self.trades if t['pnl'] > 0]) / len(self.trades) if self.trades else 0,
            'equity_curve': equity_df,
            'trades': self.trades
        }
    
    def _log_to_mlflow(self, results: Dict[str, Any], strategy):
        """Log backtest results to MLflow."""
        with mlflow.start_run():
            # Log parameters
            mlflow.log_params({
                'strategy': strategy.__class__.__name__,
                'initial_capital': self.initial_capital,
                'commission': self.commission,
                'slippage': self.slippage
            })
            
            # Log metrics
            mlflow.log_metrics({
                'total_return': results.get('total_return', 0),
                'annualized_return': results.get('annualized_return', 0),
                'volatility': results.get('volatility', 0),
                'sharpe_ratio': results.get('sharpe_ratio', 0),
                'max_drawdown': results.get('max_drawdown', 0),
                'total_trades': results.get('total_trades', 0),
                'win_rate': results.get('win_rate', 0)
            })
            
            # Log equity curve
            if 'equity_curve' in results:
                results['equity_curve'].to_csv('equity_curve.csv')
                mlflow.log_artifact('equity_curve.csv')
