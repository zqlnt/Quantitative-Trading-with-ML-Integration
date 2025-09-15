"""
Portfolio Backtesting Engine for Neural Quant

This module extends the backtesting engine to handle portfolio strategies
that work across multiple tickers simultaneously.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Union
import mlflow
from datetime import datetime
from .backtest import Backtester
from ..utils.time_utils import ensure_tz_naive_daily_index, is_daily_data
from ..strategies.base.strategy_base import StrategyBase
from ..analysis.mcpt import MonteCarloPermutationTester, MCPTConfig
from ..analysis.bootstrap import BootstrapAnalyzer, BootstrapConfig
from ..analysis.regime_filter import RegimeFilter, RegimeFilterConfig
from ..analysis.volatility_targeting import VolatilityTargeting, VolatilityTargetingConfig
from ..analysis.allocation_methods import AllocationMethods, AllocationMethodConfig
from ..analysis.position_management import PositionManager, PositionManagementConfig
from ..analysis.basic_exits import BasicExits, BasicExitsConfig
from ..logging.artifacts import ArtifactManager


class PortfolioBacktester(Backtester):
    """Extended backtesting engine for portfolio strategies."""
    
    def __init__(self, 
                 initial_capital: float = 100000.0,
                 commission: float = 0.001,
                 slippage: float = 0.0005,
                 max_positions: int = 12,
                 enable_mcpt: bool = True,
                 mcpt_config: MCPTConfig = None,
                 enable_bootstrap: bool = True,
                 bootstrap_config: BootstrapConfig = None,
                 enable_regime_filter: bool = False,
                 regime_filter_config: RegimeFilterConfig = None,
                 enable_vol_targeting: bool = False,
                 vol_targeting_config: VolatilityTargetingConfig = None,
                 allocation_method: str = "equal_weight",
                 allocation_config: AllocationMethodConfig = None,
                 position_management_config: PositionManagementConfig = None,
                 enable_basic_exits: bool = False,
                 basic_exits_config: BasicExitsConfig = None):
        """
        Initialize the portfolio backtester.
        
        Args:
            initial_capital: Starting capital
            commission: Commission rate per trade
            slippage: Slippage rate per trade
            max_positions: Maximum number of positions to hold
            enable_mcpt: Whether to run Monte Carlo Permutation Tests
            mcpt_config: Configuration for MCPT testing
            enable_bootstrap: Whether to run Bootstrap Confidence Intervals
            bootstrap_config: Configuration for Bootstrap analysis
            enable_regime_filter: Whether to enable regime filtering
            regime_filter_config: Configuration for regime filtering
            enable_vol_targeting: Whether to enable volatility targeting
            vol_targeting_config: Configuration for volatility targeting
            allocation_method: Portfolio allocation method
            allocation_config: Configuration for allocation methods
            position_management_config: Configuration for position management
        """
        super().__init__(initial_capital, commission, slippage, enable_mcpt, mcpt_config, enable_bootstrap, bootstrap_config, enable_regime_filter, regime_filter_config, enable_vol_targeting, vol_targeting_config, enable_basic_exits, basic_exits_config)
        self.max_positions = max_positions
        self.symbols = []
        
        # Initialize allocation and position management
        self.allocation_method = allocation_method
        self.allocation_config = allocation_config or AllocationMethodConfig(method=allocation_method)
        self.position_management_config = position_management_config or PositionManagementConfig()
        self.allocation_manager = AllocationMethods(self.allocation_config)
        self.position_manager = PositionManager(self.position_management_config)
    
    def run_portfolio_backtest(self, 
                              data_dict: Dict[str, pd.DataFrame], 
                              strategy: StrategyBase,
                              start_date: Optional[str] = None,
                              end_date: Optional[str] = None) -> Dict[str, Any]:
        """
        Run a portfolio backtest across multiple tickers.
        
        Args:
            data_dict: Dictionary mapping symbol to price data
            strategy: Strategy instance (must be portfolio strategy)
            start_date: Start date for backtest
            end_date: End date for backtest
            
        Returns:
            Dictionary containing backtest results
        """
        if not strategy.is_portfolio_strategy():
            raise ValueError(f"Strategy {strategy.__class__.__name__} is not a portfolio strategy")
        
        self.reset()
        self.symbols = list(data_dict.keys())
        
        # Prepare portfolio data
        portfolio_data = self._prepare_portfolio_data(data_dict, start_date, end_date)
        
        if portfolio_data.empty:
            return {'error': 'No data available for backtest period'}
        
        # Generate portfolio signals
        signals_df = strategy.generate_portfolio_signals(portfolio_data)
        
        # Process each day
        for date, row in portfolio_data.iterrows():
            self.current_date = date.tz_localize(None) if date.tz else date
            self._process_portfolio_signals(self.current_date, row, signals_df)
            self._update_equity_curve(self.current_date, row)
        
        # Calculate performance metrics
        results = self._calculate_portfolio_results(portfolio_data)
        
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
        
        # Create and save artifacts
        self._create_portfolio_artifacts(results, strategy, start_date, end_date, data_dict)
        
        return results
    
    def _prepare_portfolio_data(self, 
                               data_dict: Dict[str, pd.DataFrame], 
                               start_date: Optional[str] = None,
                               end_date: Optional[str] = None) -> pd.DataFrame:
        """
        Prepare portfolio data by aligning all ticker data.
        
        Args:
            data_dict: Dictionary mapping symbol to price data
            start_date: Start date for backtest
            end_date: End date for backtest
            
        Returns:
            DataFrame with aligned price data for all tickers
        """
        # Normalize timezone for all data
        normalized_data = {}
        for symbol, data in data_dict.items():
            if is_daily_data(data):
                data = ensure_tz_naive_daily_index(data, market="US")
            
            # Filter by date range
            if start_date:
                data = data[data.index >= start_date]
            if end_date:
                data = data[data.index <= end_date]
            
            normalized_data[symbol] = data
        
        # Find common date range
        all_dates = set()
        for data in normalized_data.values():
            all_dates.update(data.index)
        
        common_dates = sorted(all_dates)
        
        # Create portfolio DataFrame
        portfolio_data = pd.DataFrame(index=common_dates)
        
        for symbol, data in normalized_data.items():
            # Align data to common dates
            aligned_data = data.reindex(common_dates, method='ffill')
            
            # Add close prices
            if 'close' in aligned_data.columns:
                portfolio_data[f'{symbol}_close'] = aligned_data['close']
            elif 'Close' in aligned_data.columns:
                portfolio_data[f'{symbol}_close'] = aligned_data['Close']
            else:
                # Try to find any price column
                price_cols = [col for col in aligned_data.columns if 'close' in col.lower() or 'price' in col.lower()]
                if price_cols:
                    portfolio_data[f'{symbol}_close'] = aligned_data[price_cols[0]]
                else:
                    raise ValueError(f"No price column found for {symbol}")
        
        return portfolio_data
    
    def _process_portfolio_signals(self, date, row, signals_df: pd.DataFrame):
        """Process portfolio trading signals for the current date."""
        if date not in signals_df.index:
            return
        
        date_signals = signals_df.loc[date]
        
        # Get current portfolio weights
        current_weights = self._get_current_weights()
        
        # Calculate target weights based on signals and allocation method
        target_weights = self._calculate_target_weights(date, date_signals, row)
        
        # Apply position management (caps and rebalancing)
        final_weights, was_rebalanced = self.position_manager.rebalance_portfolio(
            date, current_weights, target_weights
        )
        
        # Update positions based on final weights
        if was_rebalanced:
            self._rebalance_positions(date, final_weights, row)
        else:
            # Process individual signals for new positions
            for symbol in self.symbols:
                if symbol not in date_signals:
                    continue
                
                signal = date_signals[symbol]
                price_col = f'{symbol}_close'
                
                if price_col not in row or pd.isna(row[price_col]):
                    continue
                
                price = row[price_col]
                
                if signal == 1 and symbol not in self.positions:  # New long signal
                    self._open_position(symbol, price, final_weights.get(symbol, 0))
                elif signal == -1 and symbol in self.positions:  # Close signal
                    self._close_position(symbol, price)
    
    def _get_current_weights(self) -> Dict[str, float]:
        """Get current portfolio weights."""
        if not self.positions:
            return {}
        
        total_value = self.capital
        weights = {}
        
        for symbol, position in self.positions.items():
            position_value = position['shares'] * position['entry_price']
            weights[symbol] = position_value / total_value if total_value > 0 else 0
        
        return weights
    
    def _calculate_target_weights(self, date, signals, row) -> Dict[str, float]:
        """Calculate target weights based on signals and allocation method."""
        # Get symbols with buy signals
        buy_symbols = [symbol for symbol in self.symbols 
                      if symbol in signals and signals[symbol] == 1]
        
        if not buy_symbols:
            return {}
        
        # Calculate returns data for volatility weighting if needed
        returns_data = None
        if self.allocation_config.method == "volatility_weighted":
            returns_data = self._get_returns_data(buy_symbols, date)
        
        # Calculate weights using allocation method
        target_weights = self.allocation_manager.calculate_weights(
            buy_symbols, returns_data
        )
        
        # Log weights for this date
        self.allocation_manager.log_weights(date, target_weights)
        
        return target_weights
    
    def _get_returns_data(self, symbols: List[str], current_date) -> Dict[str, pd.Series]:
        """Get returns data for volatility weighting."""
        returns_data = {}
        
        for symbol in symbols:
            price_col = f'{symbol}_close'
            if price_col in self.portfolio_data.columns:
                prices = self.portfolio_data[price_col].dropna()
                if len(prices) > 1:
                    returns = prices.pct_change().dropna()
                    returns_data[symbol] = returns
        
        return returns_data
    
    def _rebalance_positions(self, date, target_weights: Dict[str, float], row):
        """Rebalance portfolio to target weights."""
        # Close all existing positions
        for symbol in list(self.positions.keys()):
            price_col = f'{symbol}_close'
            if price_col in row and not pd.isna(row[price_col]):
                self._close_position(symbol, row[price_col])
        
        # Open new positions based on target weights
        for symbol, weight in target_weights.items():
            if weight > 0:
                price_col = f'{symbol}_close'
                if price_col in row and not pd.isna(row[price_col]):
                    self._open_position(symbol, row[price_col], weight)
    
    def _open_position(self, symbol: str, price: float, target_weight: float = None):
        """Open a new position with portfolio constraints."""
        # Check if we already have a position
        if symbol in self.positions:
            return
        
        # Check position limit
        if len(self.positions) >= self.max_positions:
            return
        
        # Apply slippage
        entry_price = price * (1 + self.slippage)
        
        # Calculate position size based on target weight
        if target_weight is None:
            # Fallback to equal weight if no target weight provided
            target_weight = 1.0 / self.max_positions
        
        # Calculate position value based on target weight
        position_value = self.capital * target_weight
        
        # Don't exceed available capital
        position_value = min(position_value, available_capital)
        
        shares = int(position_value / entry_price)
        
        if shares > 0:
            cost = shares * entry_price * (1 + self.commission)
            if cost <= available_capital:
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
            price_col = f'{symbol}_close'
            if price_col in row and not pd.isna(row[price_col]):
                current_price = row[price_col]
                total_value += position['shares'] * current_price
        
        self.equity_curve.append({
            'date': date,
            'equity': total_value,
            'cash': self.capital,
            'positions': len(self.positions),
            'symbols': list(self.positions.keys())
        })
    
    def _calculate_portfolio_results(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Calculate portfolio performance metrics."""
        if not self.equity_curve:
            return {}
        
        equity_df = pd.DataFrame(self.equity_curve).set_index('date')
        equity_df['returns'] = equity_df['equity'].pct_change()
        
        # Calculate basic metrics
        total_return = (equity_df['equity'].iloc[-1] / self.initial_capital) - 1
        annualized_return = (1 + total_return) ** (252 / len(equity_df)) - 1
        volatility = equity_df['returns'].std() * np.sqrt(252)
        sharpe_ratio = annualized_return / volatility if volatility > 0 else 0
        
        # Calculate CAGR (Compound Annual Growth Rate)
        years = len(equity_df) / 252
        cagr = (equity_df['equity'].iloc[-1] / self.initial_capital) ** (1 / years) - 1 if years > 0 else 0
        
        # Max drawdown
        rolling_max = equity_df['equity'].expanding().max()
        drawdown = (equity_df['equity'] - rolling_max) / rolling_max
        max_drawdown = drawdown.min()
        
        # Portfolio-specific metrics
        avg_positions = equity_df['positions'].mean()
        max_positions = equity_df['positions'].max()
        
        # Calculate turnover (simplified)
        total_turnover = len(self.trades) / len(equity_df) if len(equity_df) > 0 else 0
        
        # Calculate individual symbol performance
        symbol_performance = {}
        for symbol in self.symbols:
            symbol_trades = [t for t in self.trades if t['symbol'] == symbol]
            if symbol_trades:
                symbol_pnl = sum(t['pnl'] for t in symbol_trades)
                symbol_trades_count = len(symbol_trades)
                symbol_performance[symbol] = {
                    'pnl': symbol_pnl,
                    'trades': symbol_trades_count,
                    'avg_return': np.mean([t['return'] for t in symbol_trades]) if symbol_trades else 0
                }
        
        return {
            'total_return': total_return,
            'annualized_return': annualized_return,
            'cagr': cagr,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'total_trades': len(self.trades),
            'win_rate': len([t for t in self.trades if t['pnl'] > 0]) / len(self.trades) if self.trades else 0,
            'avg_positions': avg_positions,
            'max_positions': max_positions,
            'turnover': total_turnover,
            'symbol_performance': symbol_performance,
            'equity_curve': equity_df,
            'trades': self.trades,
            'symbols_traded': list(set(t['symbol'] for t in self.trades))
        }
    
    def _log_to_mlflow(self, results: Dict[str, Any], strategy: StrategyBase):
        """Log portfolio backtest results to MLflow."""
        try:
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
                mlflow.log_params({
                    'strategy': strategy.__class__.__name__,
                    'strategy_type': 'portfolio',
                    'initial_capital': self.initial_capital,
                    'commission': self.commission,
                    'slippage': self.slippage,
                    'max_positions': self.max_positions,
                    'symbols': ','.join(self.symbols)
                })
                
                # Log metrics
                mlflow.log_metrics({
                    'total_return': results.get('total_return', 0),
                    'annualized_return': results.get('annualized_return', 0),
                    'cagr': results.get('cagr', 0),
                    'volatility': results.get('volatility', 0),
                    'sharpe_ratio': results.get('sharpe_ratio', 0),
                    'max_drawdown': results.get('max_drawdown', 0),
                    'total_trades': results.get('total_trades', 0),
                    'win_rate': results.get('win_rate', 0),
                    'avg_positions': results.get('avg_positions', 0),
                    'max_positions': results.get('max_positions', 0),
                    'turnover': results.get('turnover', 0)
                })
                
                # Log equity curve
                if 'equity_curve' in results:
                    results['equity_curve'].to_csv('portfolio_equity_curve.csv')
                    mlflow.log_artifact('portfolio_equity_curve.csv')
                
                # Log symbol performance
                if 'symbol_performance' in results:
                    symbol_perf_df = pd.DataFrame(results['symbol_performance']).T
                    symbol_perf_df.to_csv('symbol_performance.csv')
                    mlflow.log_artifact('symbol_performance.csv')
                
                # Try to generate AI summary if available
                try:
                    from ..utils.llm_assistant import NeuralQuantAssistant
                    assistant = NeuralQuantAssistant()
                    summary = assistant.generate_experiment_summary(
                        run.info.run_id,
                        results,
                        {
                            'strategy': strategy.__class__.__name__,
                            'strategy_type': 'portfolio',
                            'initial_capital': self.initial_capital,
                            'commission': self.commission,
                            'slippage': self.slippage,
                            'max_positions': self.max_positions,
                            'symbols': self.symbols
                        }
                    )
                    mlflow.set_tag("ai_summary", summary)
                except Exception as ai_error:
                    print(f"Warning: AI summary generation failed: {ai_error}")
                    
        except Exception as e:
            # If MLflow logging fails, just continue without it
            print(f"Warning: MLflow logging failed: {e}")
    
    def _create_portfolio_artifacts(self, results: Dict[str, Any], strategy, start_date: str, end_date: str, data_dict: Dict[str, pd.DataFrame]):
        """Create and save structured artifacts for the portfolio run."""
        try:
            import mlflow
            
            # Get current run ID
            run_id = mlflow.active_run().info.run_id if mlflow.active_run() else "unknown"
            
            # Create artifact manager
            artifact_manager = ArtifactManager(run_id)
            
            # Create parameters artifact
            strategy_params = getattr(strategy, 'params', {})
            tickers = list(data_dict.keys())
            
            # Get configuration dictionaries
            regime_config = self.regime_filter.get_regime_summary() if self.regime_filter else None
            vol_config = self.vol_targeting.get_scaling_summary() if self.vol_targeting else None
            allocation_config = {
                'method': self.allocation_method,
                'vol_lookback': self.allocation_config.vol_lookback
            }
            position_config = {
                'max_position_pct': self.position_management_config.max_position_pct,
                'rebalance_frequency': self.position_management_config.rebalance_frequency,
                'min_rebalance_interval': self.position_management_config.min_rebalance_interval,
                'turnover_threshold': self.position_management_config.turnover_threshold
            }
            basic_exits_config = self.basic_exits.get_exits_summary() if self.basic_exits else None
            
            artifact_manager.create_params_artifact(
                strategy=strategy.__class__.__name__,
                strategy_params=strategy_params,
                tickers=tickers,
                start_date=start_date or "unknown",
                end_date=end_date or "unknown",
                initial_capital=self.initial_capital,
                commission=self.commission,
                slippage=self.slippage,
                regime_filter_config=regime_config,
                vol_targeting_config=vol_config,
                allocation_config=allocation_config,
                position_config=position_config,
                basic_exits_config=basic_exits_config
            )
            
            # Create metrics artifact
            portfolio_metrics = {
                'total_return': results.get('total_return', 0.0),
                'annualized_return': results.get('annualized_return', 0.0),
                'volatility': results.get('volatility', 0.0),
                'sharpe_ratio': results.get('sharpe_ratio', 0.0),
                'max_drawdown': results.get('max_drawdown', 0.0),
                'total_trades': results.get('total_trades', 0),
                'win_rate': results.get('win_rate', 0.0),
                'profit_factor': results.get('profit_factor', 0.0),
                'cagr': results.get('cagr', 0.0)
            }
            
            # Get per-ticker metrics
            per_ticker_metrics = results.get('per_ticker_metrics', {})
            
            # Get position management metrics
            position_metrics = {
                'total_turnover': results.get('total_turnover', 0.0),
                'num_rebalances': results.get('num_rebalances', 0),
                'num_cap_breaches': results.get('num_cap_breaches', 0)
            }
            
            artifact_manager.create_metrics_artifact(
                portfolio_metrics=portfolio_metrics,
                per_ticker_metrics=per_ticker_metrics,
                regime_metrics=regime_config,
                vol_targeting_metrics=vol_config,
                position_metrics=position_metrics,
                basic_exits_metrics=basic_exits_config
            )
            
            # Create equity artifact
            if 'equity_curve' in results and 'daily_returns' in results:
                artifact_manager.create_equity_artifact(
                    equity_curve=results['equity_curve'],
                    daily_returns=results['daily_returns']
                )
            
            # Create trades artifact
            if 'trades' in results:
                artifact_manager.create_trades_artifact(results['trades'])
            
            # Create MCPT artifact
            if 'mcpt_results' in results and results['mcpt_results']:
                mcpt_data = results['mcpt_results']
                artifact_manager.create_mcpt_artifact(
                    method=mcpt_data.get('method', 'returns_permutation'),
                    permutations=mcpt_data.get('permutations', 1000),
                    block_size=mcpt_data.get('block_size'),
                    significance_level=mcpt_data.get('significance_level', 0.05),
                    results=mcpt_data.get('results', [])
                )
            
            # Create bootstrap artifact
            if 'bootstrap_results' in results and results['bootstrap_results']:
                bootstrap_data = results['bootstrap_results']
                artifact_manager.create_bootstrap_artifact(
                    method=bootstrap_data.get('method', 'bootstrap'),
                    samples=bootstrap_data.get('samples', 1000),
                    confidence_level=bootstrap_data.get('confidence_level', 0.95),
                    results=bootstrap_data.get('results', [])
                )
            
            # Create weights artifact
            if hasattr(self.allocation_manager, 'weights_history') and not self.allocation_manager.weights_history.empty:
                artifact_manager.create_weights_artifact(self.allocation_manager.weights_history)
            
            # Save all artifacts
            saved_files = artifact_manager.save_all_artifacts()
            
            # Log artifacts to MLflow
            if mlflow.active_run():
                for artifact_name, filepath in saved_files.items():
                    try:
                        mlflow.log_artifact(filepath)
                    except Exception as e:
                        print(f"Warning: Failed to log artifact {artifact_name}: {e}")
            
        except Exception as e:
            print(f"Warning: Portfolio artifact creation failed: {e}")
