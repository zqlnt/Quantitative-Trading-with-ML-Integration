"""Backtesting engine for trading strategies."""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any
import mlflow
import mlflow.sklearn
from datetime import datetime
from ..utils.time_utils import ensure_tz_naive_daily_index, is_daily_data
from ..analysis.mcpt import MonteCarloPermutationTester, MCPTConfig
from ..analysis.bootstrap import BootstrapAnalyzer, BootstrapConfig
from ..analysis.regime_filter import RegimeFilter, RegimeFilterConfig
from ..analysis.volatility_targeting import VolatilityTargeting, VolatilityTargetingConfig

class Backtester:
    """High-fidelity backtesting engine with realistic transaction costs."""
    
    def __init__(self, 
                 initial_capital: float = 100000.0,
                 commission: float = 0.001,
                 slippage: float = 0.0005,
                 enable_mcpt: bool = True,
                 mcpt_config: MCPTConfig = None,
                 enable_bootstrap: bool = True,
                 bootstrap_config: BootstrapConfig = None,
                 enable_regime_filter: bool = False,
                 regime_filter_config: RegimeFilterConfig = None,
                 enable_vol_targeting: bool = False,
                 vol_targeting_config: VolatilityTargetingConfig = None):
        """
        Initialize the backtester.
        
        Args:
            initial_capital: Starting capital
            commission: Commission rate per trade
            slippage: Slippage rate per trade
            enable_mcpt: Whether to run Monte Carlo Permutation Tests
            mcpt_config: Configuration for MCPT testing
            enable_bootstrap: Whether to run Bootstrap Confidence Intervals
            bootstrap_config: Configuration for Bootstrap analysis
            enable_regime_filter: Whether to enable regime filtering
            regime_filter_config: Configuration for regime filtering
            enable_vol_targeting: Whether to enable volatility targeting
            vol_targeting_config: Configuration for volatility targeting
        """
        self.initial_capital = initial_capital
        self.commission = commission
        self.slippage = slippage
        self.enable_mcpt = enable_mcpt
        self.mcpt_config = mcpt_config or MCPTConfig()
        self.enable_bootstrap = enable_bootstrap
        self.bootstrap_config = bootstrap_config or BootstrapConfig()
        self.enable_regime_filter = enable_regime_filter
        self.regime_filter_config = regime_filter_config or RegimeFilterConfig()
        self.enable_vol_targeting = enable_vol_targeting
        self.vol_targeting_config = vol_targeting_config or VolatilityTargetingConfig()
        self.regime_filter = None
        self.vol_targeting = None
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
        
        # Normalize timezone for daily data
        if is_daily_data(data):
            data = ensure_tz_naive_daily_index(data, market="US")
        
        # Filter data by date range
        if start_date:
            data = data[data.index >= start_date]
        if end_date:
            data = data[data.index <= end_date]
        
        # Initialize strategy if not already initialized
        if not strategy.is_initialized:
            success = strategy.initialize(data)
            if not success:
                return {'error': 'Strategy initialization failed'}
        
        # Initialize regime filter if enabled
        if self.enable_regime_filter:
            self.regime_filter = RegimeFilter(self.regime_filter_config)
            if not self.regime_filter.load_proxy_data(start_date, end_date):
                return {'error': 'Failed to load regime filter proxy data'}
        
        # Initialize volatility targeting if enabled
        if self.enable_vol_targeting:
            self.vol_targeting = VolatilityTargeting(self.vol_targeting_config)
        
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
        
        # Apply volatility targeting if enabled
        if self.enable_vol_targeting and self.vol_targeting and 'equity_curve' in results:
            original_equity = results['equity_curve']
            scaled_equity = self.vol_targeting.scale_equity_curve(original_equity)
            results['equity_curve'] = scaled_equity
            
            # Recalculate metrics with scaled equity curve
            results = self._calculate_results(data, scaled_equity)
            
            # Add volatility targeting metrics
            vol_summary = self.vol_targeting.get_scaling_summary()
            results['volatility_targeting'] = vol_summary
        
        # Run MCPT significance testing if enabled
        if self.enable_mcpt and 'equity_curve' in results:
            mcpt_results = self._run_mcpt_test(results, strategy)
            results['mcpt_results'] = mcpt_results
        
        # Run Bootstrap confidence intervals if enabled
        if self.enable_bootstrap and 'trades' in results and results['trades']:
            bootstrap_results = self._run_bootstrap_test(results, strategy)
            results['bootstrap_results'] = bootstrap_results
        
        # Log to MLflow
        self._log_to_mlflow(results, strategy)
        
        return results
    
    def _process_signals(self, date, row, signals):
        """Process trading signals for the current date."""
        # Check regime filter first
        if self.regime_filter and not self.regime_filter.should_trade(date):
            return  # Skip trading if regime filter blocks it
        
        # Find signals for this date
        date_signals = [s for s in signals if s.timestamp.date() == date.date()]
        
        for signal in date_signals:
            symbol = signal.symbol
            action = signal.signal_type
            
            if action == 'BUY' and symbol not in self.positions:
                self._open_position(symbol, signal.price)
            elif action == 'SELL' and symbol in self.positions:
                self._close_position(symbol, signal.price)
    
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
    
    def _calculate_results(self, data: pd.DataFrame, equity_curve: Optional[pd.Series] = None) -> Dict[str, Any]:
        """Calculate performance metrics."""
        if equity_curve is not None:
            # Use provided equity curve (for volatility targeting)
            equity_df = pd.DataFrame({'equity': equity_curve})
            equity_df['returns'] = equity_df['equity'].pct_change()
        elif not self.equity_curve:
            return {}
        else:
            # Use internal equity curve
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
    
    def _run_mcpt_test(self, results: Dict[str, Any], strategy) -> Dict[str, Any]:
        """Run Monte Carlo Permutation Test on backtest results."""
        try:
            equity_df = results['equity_curve']
            returns = equity_df['returns'].dropna()
            
            # Create MCPT tester
            tester = MonteCarloPermutationTester(self.mcpt_config)
            
            # Test returns significance
            mcpt_results = tester.test_returns_significance(
                returns, 
                strategy_name=strategy.__class__.__name__
            )
            
            # Get summary
            summary = tester.get_summary()
            
            # Log to MLflow
            tester.log_to_mlflow()
            
            return {
                'results': mcpt_results,
                'summary': summary,
                'config': {
                    'n_permutations': self.mcpt_config.n_permutations,
                    'block_size': self.mcpt_config.block_size,
                    'confidence_level': self.mcpt_config.confidence_level,
                    'significance_level': self.mcpt_config.significance_level
                }
            }
            
        except Exception as e:
            print(f"Warning: MCPT testing failed: {e}")
            return {'error': str(e)}
    
    def _run_bootstrap_test(self, results: Dict[str, Any], strategy) -> Dict[str, Any]:
        """Run Bootstrap Confidence Intervals on backtest results."""
        try:
            trades = results.get('trades', [])
            
            if not trades:
                return {'error': 'No trades available for bootstrap analysis'}
            
            # Create Bootstrap analyzer
            analyzer = BootstrapAnalyzer(self.bootstrap_config)
            
            # Analyze trades
            bootstrap_results = analyzer.analyze_trades(
                trades, 
                self.initial_capital,
                strategy_name=strategy.__class__.__name__
            )
            
            # Get summary
            summary = analyzer.get_summary()
            
            # Create histograms
            plots = analyzer.create_histogram_plots()
            
            # Log to MLflow
            analyzer.log_to_mlflow()
            
            return {
                'results': bootstrap_results,
                'summary': summary,
                'plots': plots,
                'config': {
                    'n_bootstrap': self.bootstrap_config.n_bootstrap,
                    'confidence_level': self.bootstrap_config.confidence_level,
                    'resample_method': self.bootstrap_config.resample_method
                }
            }
            
        except Exception as e:
            print(f"Warning: Bootstrap testing failed: {e}")
            return {'error': str(e)}
    
    def _log_to_mlflow(self, results: Dict[str, Any], strategy):
        """Log backtest results to MLflow."""
        try:
            import mlflow
            
            # Set tracking URI if not already set
            if not mlflow.get_tracking_uri():
                mlflow.set_tracking_uri("sqlite:///mlflow.db")
            
            # End any existing run before starting a new one
            try:
                mlflow.end_run()
            except:
                pass  # No active run to end
            
            with mlflow.start_run() as run:
                # Log parameters
                params = {
                    'strategy': strategy.__class__.__name__,
                    'initial_capital': self.initial_capital,
                    'commission': self.commission,
                    'slippage': self.slippage
                }
                
                # Add regime filter parameters if enabled
                if self.enable_regime_filter and self.regime_filter:
                    regime_summary = self.regime_filter.get_regime_summary()
                    params.update({
                        'regime_filter_enabled': regime_summary['regime_filter_enabled'],
                        'regime_proxy_symbol': regime_summary['proxy_symbol'],
                        'regime_rule': regime_summary['regime_rule']
                    })
                
                # Add volatility targeting parameters if enabled
                if self.enable_vol_targeting and self.vol_targeting:
                    vol_summary = self.vol_targeting.get_scaling_summary()
                    params.update({
                        'volatility_targeting_enabled': vol_summary['volatility_targeting_enabled'],
                        'vol_target': vol_summary['target_vol'],
                        'vol_lookback_window': vol_summary['lookback_window'],
                        'vol_scale_cap': vol_summary['scale_cap']
                    })
                
                mlflow.log_params(params)
                
                # Log metrics
                metrics = {
                    'total_return': results.get('total_return', 0),
                    'annualized_return': results.get('annualized_return', 0),
                    'volatility': results.get('volatility', 0),
                    'sharpe_ratio': results.get('sharpe_ratio', 0),
                    'max_drawdown': results.get('max_drawdown', 0),
                    'total_trades': results.get('total_trades', 0),
                    'win_rate': results.get('win_rate', 0)
                }
                
                # Add regime filter metrics if enabled
                if self.enable_regime_filter and self.regime_filter:
                    regime_summary = self.regime_filter.get_regime_summary()
                    metrics.update({
                        'regime_hit_rate': regime_summary['regime_hit_rate'],
                        'regime_trading_days': regime_summary['trading_days'],
                        'regime_total_days': regime_summary['total_days']
                    })
                
                # Add volatility targeting metrics if enabled
                if self.enable_vol_targeting and self.vol_targeting:
                    vol_summary = self.vol_targeting.get_scaling_summary()
                    metrics.update({
                        'realized_vol_pre': vol_summary['realized_vol_pre'],
                        'realized_vol_post': vol_summary['realized_vol_post'],
                        'avg_scaling': vol_summary['avg_scaling'],
                        'vol_reduction': vol_summary['vol_reduction']
                    })
                
                mlflow.log_metrics(metrics)
                
                # Log equity curve
                if 'equity_curve' in results:
                    results['equity_curve'].to_csv('equity_curve.csv')
                    mlflow.log_artifact('equity_curve.csv')
                
                # Try to generate AI summary if available
                try:
                    from ..utils.llm_assistant import NeuralQuantAssistant
                    assistant = NeuralQuantAssistant()
                    summary = assistant.generate_experiment_summary(
                        run.info.run_id,
                        results,
                        {
                            'strategy': strategy.__class__.__name__,
                            'initial_capital': self.initial_capital,
                            'commission': self.commission,
                            'slippage': self.slippage
                        }
                    )
                    mlflow.set_tag("ai_summary", summary)
                except Exception as ai_error:
                    print(f"Warning: AI summary generation failed: {ai_error}")
                    
        except Exception as e:
            # If MLflow logging fails, just continue without it
            print(f"Warning: MLflow logging failed: {e}")
