"""
Base strategy class for Neural Quant.

This module provides the base class and interface for all trading strategies,
ensuring consistent behavior and integration with the MLflow tracking system.
"""

import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple, Union
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from dataclasses import dataclass

from ...utils.config.config_manager import get_config
from ...logging.mlflow.mlflow_manager import get_mlflow_manager
from ...logging.trades.trade_logger import get_trade_logger
from ...utils.validation.metrics import calculate_comprehensive_metrics, format_metrics_report
from ...utils.validation.sanity_checks import run_sanity_checks
from ...utils.helpers.determinism import set_global_seed, create_data_manifest, create_config_fingerprint


@dataclass
class Signal:
    """Trading signal data structure."""
    symbol: str
    signal_type: str  # BUY, SELL, HOLD
    strength: float  # Signal strength (0-1)
    price: float
    timestamp: datetime
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class Trade:
    """Trade execution data structure."""
    symbol: str
    side: str  # BUY, SELL
    quantity: float
    price: float
    timestamp: datetime
    order_id: Optional[str] = None
    commission: float = 0.0
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class Portfolio:
    """Portfolio state data structure."""
    total_value: float
    cash: float
    positions: Dict[str, float]  # symbol -> quantity
    timestamp: datetime
    metadata: Optional[Dict[str, Any]] = None


class StrategyBase(ABC):
    """
    Abstract base class for all trading strategies.
    
    This class provides the common interface and functionality that all
    trading strategies must implement, including signal generation,
    risk management, and performance tracking.
    """
    
    def __init__(self, name: str, parameters: Optional[Dict[str, Any]] = None):
        """
        Initialize the strategy.
        
        Args:
            name: Name of the strategy.
            parameters: Strategy parameters.
        """
        self.name = name
        self.parameters = parameters or {}
        self.logger = logging.getLogger(f"{__name__}.{name}")
        
        # Configuration
        self.config = get_config()
        
        # MLflow and logging
        self.mlflow_manager = get_mlflow_manager()
        self.trade_logger = get_trade_logger()
        
        # Strategy state
        self.is_initialized = False
        self.current_positions = {}
        self.cash = self.config.trading.paper_trading.initial_capital
        self.total_value = self.cash
        self.trade_history = []
        self.signal_history = []
        self.performance_metrics = {}
        
        # Risk management
        self.max_position_size = self.config.trading.max_position_size
        self.max_daily_loss = self.config.trading.max_daily_loss
        self.max_drawdown = self.config.trading.max_drawdown
        
        self.logger.info(f"Initialized strategy: {name}")
    
    def initialize(self, data: pd.DataFrame) -> bool:
        """
        Initialize the strategy with historical data.
        
        Args:
            data: Historical data for initialization.
            
        Returns:
            bool: True if initialization successful, False otherwise.
        """
        try:
            self._initialize_strategy(data)
            self.is_initialized = True
            self.logger.info(f"Strategy {self.name} initialized successfully")
            return True
        except Exception as e:
            self.logger.error(f"Failed to initialize strategy {self.name}: {e}")
            return False
    
    def generate_signals(self, data: pd.DataFrame) -> List[Signal]:
        """
        Generate trading signals based on current data.
        
        Args:
            data: Current market data.
            
        Returns:
            List[Signal]: List of generated signals.
        """
        if not self.is_initialized:
            self.logger.warning(f"Strategy {self.name} not initialized")
            return []
        
        try:
            signals = self._generate_signals(data)
            
            # Log signals
            for signal in signals:
                self.trade_logger.log_signal(
                    strategy_name=self.name,
                    symbol=signal.symbol,
                    signal_type=signal.signal_type,
                    strength=signal.strength,
                    price=signal.price,
                    metadata=signal.metadata
                )
                self.signal_history.append(signal)
            
            return signals
            
        except Exception as e:
            self.logger.error(f"Failed to generate signals for strategy {self.name}: {e}")
            return []
    
    def execute_trades(self, signals: List[Signal], current_prices: Dict[str, float]) -> List[Trade]:
        """
        Execute trades based on signals.
        
        Args:
            signals: List of signals to execute.
            current_prices: Current prices for all symbols.
            
        Returns:
            List[Trade]: List of executed trades.
        """
        trades = []
        
        for signal in signals:
            try:
                trade = self._execute_signal(signal, current_prices)
                if trade:
                    trades.append(trade)
                    self.trade_history.append(trade)
                    
                    # Update portfolio
                    self._update_portfolio(trade)
                    
                    # Log trade
                    self.trade_logger.log_trade(
                        strategy_name=self.name,
                        symbol=trade.symbol,
                        side=trade.side,
                        quantity=trade.quantity,
                        price=trade.price,
                        order_id=trade.order_id,
                        commission=trade.commission,
                        metadata=trade.metadata
                    )
                    
            except Exception as e:
                self.logger.error(f"Failed to execute signal for {signal.symbol}: {e}")
        
        return trades
    
    def update_portfolio(self, current_prices: Dict[str, float]) -> Portfolio:
        """
        Update portfolio with current prices.
        
        Args:
            current_prices: Current prices for all symbols.
            
        Returns:
            Portfolio: Updated portfolio state.
        """
        # Calculate current portfolio value
        positions_value = sum(
            quantity * current_prices.get(symbol, 0)
            for symbol, quantity in self.current_positions.items()
        )
        
        self.total_value = self.cash + positions_value
        
        # Create portfolio snapshot
        portfolio = Portfolio(
            total_value=self.total_value,
            cash=self.cash,
            positions=self.current_positions.copy(),
            timestamp=datetime.now()
        )
        
        # Log portfolio snapshot
        self.trade_logger.log_portfolio_snapshot(
            strategy_name=self.name,
            total_value=portfolio.total_value,
            cash=portfolio.cash,
            positions=portfolio.positions,
            metadata=portfolio.metadata
        )
        
        return portfolio
    
    def calculate_performance_metrics(self, benchmark_data: Optional[pd.DataFrame] = None) -> Dict[str, float]:
        """
        Calculate comprehensive performance metrics for the strategy.
        
        Args:
            benchmark_data: Optional benchmark data for comparison.
            
        Returns:
            Dict[str, float]: Dictionary of performance metrics.
        """
        try:
            if not self.trade_history:
                return {}
            
            # Convert trade history to DataFrame
            trades_df = pd.DataFrame([
                {
                    'timestamp': trade.timestamp,
                    'symbol': trade.symbol,
                    'side': trade.side,
                    'quantity': trade.quantity,
                    'price': trade.price,
                    'value': trade.quantity * trade.price,
                    'commission': trade.commission
                }
                for trade in self.trade_history
            ])
            
            # Calculate basic metrics
            total_trades = len(trades_df)
            total_value = trades_df['value'].sum()
            total_commission = trades_df['commission'].sum()
            
            # Calculate P&L (simplified)
            buy_trades = trades_df[trades_df['side'] == 'BUY']
            sell_trades = trades_df[trades_df['side'] == 'SELL']
            
            buy_value = buy_trades['value'].sum() if not buy_trades.empty else 0
            sell_value = sell_trades['value'].sum() if not sell_trades.empty else 0
            net_pnl = sell_value - buy_value - total_commission
            
            # Calculate returns
            initial_capital = self.config.trading.paper_trading.initial_capital
            total_return = (self.total_value - initial_capital) / initial_capital
            
            # Calculate win rate (simplified)
            profitable_trades = len(trades_df[trades_df['value'] > 0])
            win_rate = profitable_trades / total_trades if total_trades > 0 else 0
            
            # Calculate Sharpe ratio (simplified)
            if len(trades_df) > 1:
                returns = trades_df['value'].pct_change().dropna()
                sharpe_ratio = returns.mean() / returns.std() if returns.std() > 0 else 0
            else:
                sharpe_ratio = 0
            
            # Basic metrics
            basic_metrics = {
                'total_trades': total_trades,
                'total_value': total_value,
                'total_commission': total_commission,
                'net_pnl': net_pnl,
                'total_return': total_return,
                'win_rate': win_rate,
                'sharpe_ratio': sharpe_ratio,
                'current_portfolio_value': self.total_value,
                'current_cash': self.cash,
                'current_positions_count': len(self.current_positions)
            }
            
            # Calculate comprehensive metrics if we have enough data
            comprehensive_metrics = {}
            if len(trades_df) > 10:  # Need sufficient data for comprehensive metrics
                try:
                    # Create returns series from trade values
                    trades_df['date'] = pd.to_datetime(trades_df['timestamp'])
                    trades_df = trades_df.set_index('date').sort_index()
                    
                    # Calculate daily returns (simplified)
                    daily_values = trades_df['value'].resample('D').sum()
                    daily_returns = daily_values.pct_change().dropna()
                    
                    if len(daily_returns) > 0:
                        # Calculate comprehensive metrics
                        comprehensive_metrics = calculate_comprehensive_metrics(
                            daily_returns, 
                            benchmark_data=benchmark_data
                        )
                        
                        # Log comprehensive metrics
                        for metric_name, metric_value in comprehensive_metrics.items():
                            self.trade_logger.log_performance_metric(
                                strategy_name=self.name,
                                metric_name=metric_name,
                                metric_value=metric_value
                            )
                            
                except Exception as e:
                    self.logger.warning(f"Failed to calculate comprehensive metrics: {e}")
            
            # Combine all metrics
            all_metrics = {**basic_metrics, **comprehensive_metrics}
            self.performance_metrics = all_metrics
            
            # Log basic performance metrics
            for metric_name, metric_value in basic_metrics.items():
                self.trade_logger.log_performance_metric(
                    strategy_name=self.name,
                    metric_name=metric_name,
                    metric_value=metric_value
                )
            
            return all_metrics
            
        except Exception as e:
            self.logger.error(f"Failed to calculate performance metrics: {e}")
            return {}
    
    def run_backtest(self, data: pd.DataFrame, start_date: str, end_date: str) -> Dict[str, Any]:
        """
        Run a backtest of the strategy with comprehensive sanity checks.
        
        Args:
            data: Historical data for backtesting.
            start_date: Start date for backtest.
            end_date: End date for backtest.
            
        Returns:
            Dict[str, Any]: Backtest results.
        """
        try:
            # Set global seed for reproducibility
            set_global_seed(42)
            
            # Filter data by date range
            start_dt = pd.to_datetime(start_date)
            end_dt = pd.to_datetime(end_date)
            backtest_data = data[(data.index >= start_dt) & (data.index <= end_dt)]
            
            if backtest_data.empty:
                self.logger.warning(f"No data available for backtest period {start_date} to {end_date}")
                return {}
            
            # Create data manifest
            symbols = backtest_data['Symbol'].unique().tolist() if 'Symbol' in backtest_data.columns else ['UNKNOWN']
            data_manifest = create_data_manifest(
                backtest_data, symbols, start_date, end_date, "1d"
            )
            
            # Create config fingerprint
            config_dict = self.config.dict()
            config_fingerprint = create_config_fingerprint(config_dict)
            
            # Initialize strategy
            if not self.initialize(backtest_data):
                return {}
            
            # Run backtest
            results = {
                'strategy_name': self.name,
                'start_date': start_date,
                'end_date': end_date,
                'total_days': len(backtest_data),
                'trades': [],
                'signals': [],
                'performance_metrics': {},
                'data_manifest': data_manifest,
                'config_fingerprint': config_fingerprint,
                'sanity_checks': {}
            }
            
            # Process each day
            for date, day_data in backtest_data.groupby(backtest_data.index.date):
                # Generate signals
                signals = self.generate_signals(day_data)
                results['signals'].extend(signals)
                
                # Execute trades
                current_prices = {symbol: day_data[day_data['Symbol'] == symbol]['Close'].iloc[-1] 
                                for symbol in day_data['Symbol'].unique()}
                trades = self.execute_trades(signals, current_prices)
                results['trades'].extend(trades)
                
                # Update portfolio
                self.update_portfolio(current_prices)
            
            # Calculate final performance metrics
            results['performance_metrics'] = self.calculate_performance_metrics()
            
            # Run sanity checks
            returns = backtest_data['Close'].pct_change().dropna() if 'Close' in backtest_data.columns else pd.Series()
            sanity_results = run_sanity_checks(
                backtest_data, 
                results['performance_metrics'],
                returns=returns
            )
            results['sanity_checks'] = sanity_results
            
            # Fail fast if sanity checks fail
            if not sanity_results['overall_passed']:
                self.logger.error(f"Sanity checks failed: {sanity_results['issues']}")
                return results  # Return results but mark as failed
            
            # Log results to MLflow
            self._log_backtest_results(results)
            
            return results
            
        except Exception as e:
            self.logger.error(f"Failed to run backtest: {e}")
            return {}
    
    def _log_backtest_results(self, results: Dict[str, Any]):
        """Log backtest results to MLflow."""
        try:
            with self.mlflow_manager.start_run(run_name=f"{self.name}_backtest_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
                # Log strategy parameters
                self.mlflow_manager.log_parameters(self.parameters)
                
                # Log performance metrics
                if 'performance_metrics' in results:
                    self.mlflow_manager.log_metrics(results['performance_metrics'])
                
                # Log additional results
                self.mlflow_manager.log_parameters({
                    'strategy_name': self.name,
                    'start_date': results.get('start_date', ''),
                    'end_date': results.get('end_date', ''),
                    'total_days': results.get('total_days', 0),
                    'total_trades': len(results.get('trades', [])),
                    'total_signals': len(results.get('signals', []))
                })
                
                # Add strategy-specific tags
                import mlflow
                mlflow.set_tag("strategy_name", self.name)
                mlflow.set_tag("run_type", "backtest")
                
        except Exception as e:
            self.logger.error(f"Failed to log backtest results to MLflow: {e}")
    
    def _execute_signal(self, signal: Signal, current_prices: Dict[str, float]) -> Optional[Trade]:
        """
        Execute a single signal.
        
        Args:
            signal: Signal to execute.
            current_prices: Current prices for all symbols.
            
        Returns:
            Optional[Trade]: Executed trade or None if not executed.
        """
        if signal.signal_type not in ['BUY', 'SELL']:
            return None
        
        current_price = current_prices.get(signal.symbol, signal.price)
        
        # Calculate position size based on risk management
        position_size = self._calculate_position_size(signal, current_price)
        
        if position_size <= 0:
            return None
        
        # Check if we have enough cash for buy orders
        if signal.signal_type == 'BUY' and self.cash < position_size * current_price:
            self.logger.warning(f"Insufficient cash for {signal.symbol} buy order")
            return None
        
        # Check if we have enough shares for sell orders
        if signal.signal_type == 'SELL' and self.current_positions.get(signal.symbol, 0) < position_size:
            self.logger.warning(f"Insufficient shares for {signal.symbol} sell order")
            return None
        
        # Create trade
        trade = Trade(
            symbol=signal.symbol,
            side=signal.signal_type,
            quantity=position_size,
            price=current_price,
            timestamp=datetime.now(),
            order_id=f"{self.name}_{signal.symbol}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            commission=0.0,  # Paper trading
            metadata=signal.metadata
        )
        
        return trade
    
    def _calculate_position_size(self, signal: Signal, current_price: float) -> float:
        """
        Calculate position size based on risk management rules.
        
        Args:
            signal: Trading signal.
            current_price: Current price of the symbol.
            
        Returns:
            float: Position size in shares.
        """
        # Simple position sizing based on max position size
        max_position_value = self.total_value * self.max_position_size
        position_size = max_position_value / current_price
        
        # Round to whole shares
        return int(position_size)
    
    def _update_portfolio(self, trade: Trade):
        """Update portfolio after a trade."""
        if trade.side == 'BUY':
            self.cash -= trade.quantity * trade.price + trade.commission
            self.current_positions[trade.symbol] = self.current_positions.get(trade.symbol, 0) + trade.quantity
        elif trade.side == 'SELL':
            self.cash += trade.quantity * trade.price - trade.commission
            self.current_positions[trade.symbol] = self.current_positions.get(trade.symbol, 0) - trade.quantity
            
            # Remove position if quantity becomes zero
            if self.current_positions[trade.symbol] <= 0:
                del self.current_positions[trade.symbol]
    
    @abstractmethod
    def _initialize_strategy(self, data: pd.DataFrame):
        """
        Initialize the strategy with historical data.
        
        This method must be implemented by subclasses.
        
        Args:
            data: Historical data for initialization.
        """
        pass
    
    @abstractmethod
    def _generate_signals(self, data: pd.DataFrame) -> List[Signal]:
        """
        Generate trading signals based on current data.
        
        This method must be implemented by subclasses.
        
        Args:
            data: Current market data.
            
        Returns:
            List[Signal]: List of generated signals.
        """
        pass
    
    def is_portfolio_strategy(self) -> bool:
        """
        Check if this strategy requires portfolio-level data.
        
        Returns:
            bool: True if strategy requires multiple tickers, False otherwise.
        """
        return False
    
    def generate_portfolio_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Generate portfolio-level signals for multiple tickers.
        
        This method should be implemented by portfolio strategies.
        Default implementation raises NotImplementedError.
        
        Args:
            data: DataFrame with price data for multiple tickers (columns = tickers)
            
        Returns:
            DataFrame of signals: +1 (long), -1 (short), 0 (flat) for each ticker
        """
        if not self.is_portfolio_strategy():
            raise NotImplementedError(
                f"Strategy {self.name} is not a portfolio strategy. "
                "Use generate_signals() for single-ticker strategies."
            )
        raise NotImplementedError(
            f"Portfolio strategy {self.name} must implement generate_portfolio_signals()"
        )
    
    def get_strategy_info(self) -> Dict[str, Any]:
        """
        Get information about the strategy.
        
        Returns:
            Dict[str, Any]: Strategy information.
        """
        return {
            'name': self.name,
            'parameters': self.parameters,
            'is_initialized': self.is_initialized,
            'is_portfolio_strategy': self.is_portfolio_strategy(),
            'current_positions': self.current_positions,
            'cash': self.cash,
            'total_value': self.total_value,
            'total_trades': len(self.trade_history),
            'total_signals': len(self.signal_history),
            'performance_metrics': self.performance_metrics
        }
