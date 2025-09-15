"""
Bootstrap Confidence Intervals Module

This module implements bootstrap resampling for computing confidence intervals
of key performance metrics by resampling trade P&L with replacement.

Author: Neural Quant Team
Date: 2024
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

logger = logging.getLogger(__name__)


@dataclass
class BootstrapResult:
    """Container for bootstrap confidence interval results."""
    metric_name: str
    observed_value: float
    ci_lower: float
    ci_upper: float
    confidence_level: float
    bootstrap_values: List[float]
    mean_bootstrap: float
    std_bootstrap: float


@dataclass
class BootstrapConfig:
    """Configuration for bootstrap resampling."""
    n_bootstrap: int = 1000
    confidence_level: float = 0.95
    random_seed: Optional[int] = None
    n_jobs: int = 4
    resample_method: str = 'trades'  # 'trades' or 'returns'


class BootstrapAnalyzer:
    """
    Bootstrap Confidence Intervals Analyzer for backtest results.
    
    This class implements bootstrap resampling by:
    1. Resampling trade P&L with replacement
    2. Computing performance metrics for each bootstrap sample
    3. Calculating confidence intervals for key metrics
    4. Generating histograms and visualizations
    """
    
    def __init__(self, config: BootstrapConfig = None):
        """Initialize the bootstrap analyzer."""
        self.config = config or BootstrapConfig()
        self.results: Dict[str, BootstrapResult] = {}
        
        # Set random seed for reproducibility
        if self.config.random_seed is not None:
            np.random.seed(self.config.random_seed)
    
    def analyze_trades(self, 
                      trades: List[Dict[str, Any]], 
                      initial_capital: float = 100000.0,
                      strategy_name: str = "strategy") -> Dict[str, BootstrapResult]:
        """
        Analyze trades using bootstrap resampling.
        
        Args:
            trades: List of trade dictionaries with P&L data
            initial_capital: Initial capital for calculations
            strategy_name: Name of the strategy for logging
            
        Returns:
            Dictionary of metric names to BootstrapResult objects
        """
        logger.info(f"Running bootstrap analysis on trades for {strategy_name}")
        logger.info(f"Number of trades: {len(trades)}, N bootstrap: {self.config.n_bootstrap}")
        
        if not trades:
            logger.warning("No trades available for bootstrap analysis")
            return {}
        
        # Extract P&L data
        pnl_data = self._extract_pnl_data(trades)
        
        # Calculate observed metrics
        observed_metrics = self._calculate_observed_metrics(trades, initial_capital)
        
        # Run bootstrap resampling
        bootstrap_samples = self._run_bootstrap_resampling(pnl_data, initial_capital)
        
        # Calculate confidence intervals
        results = {}
        for metric_name, observed_value in observed_metrics.items():
            if metric_name in bootstrap_samples:
                bootstrap_values = bootstrap_samples[metric_name]
                
                result = self._calculate_confidence_interval(
                    metric_name, observed_value, bootstrap_values
                )
                results[metric_name] = result
                
                logger.info(f"{metric_name}: observed={observed_value:.4f}, "
                           f"CI=[{result.ci_lower:.4f}, {result.ci_upper:.4f}]")
        
        self.results.update(results)
        return results
    
    def analyze_returns(self, 
                       returns: pd.Series, 
                       strategy_name: str = "strategy") -> Dict[str, BootstrapResult]:
        """
        Analyze returns using bootstrap resampling.
        
        Args:
            returns: Series of strategy returns
            strategy_name: Name of the strategy for logging
            
        Returns:
            Dictionary of metric names to BootstrapResult objects
        """
        logger.info(f"Running bootstrap analysis on returns for {strategy_name}")
        logger.info(f"Returns length: {len(returns)}, N bootstrap: {self.config.n_bootstrap}")
        
        if returns.empty:
            logger.warning("No returns available for bootstrap analysis")
            return {}
        
        # Calculate observed metrics
        observed_metrics = self._calculate_returns_metrics(returns)
        
        # Run bootstrap resampling
        bootstrap_samples = self._run_returns_bootstrap(returns)
        
        # Calculate confidence intervals
        results = {}
        for metric_name, observed_value in observed_metrics.items():
            if metric_name in bootstrap_samples:
                bootstrap_values = bootstrap_samples[metric_name]
                
                result = self._calculate_confidence_interval(
                    metric_name, observed_value, bootstrap_values
                )
                results[metric_name] = result
                
                logger.info(f"{metric_name}: observed={observed_value:.4f}, "
                           f"CI=[{result.ci_lower:.4f}, {result.ci_upper:.4f}]")
        
        self.results.update(results)
        return results
    
    def _extract_pnl_data(self, trades: List[Dict[str, Any]]) -> np.ndarray:
        """Extract P&L data from trades."""
        pnl_values = []
        for trade in trades:
            if 'pnl' in trade and trade['pnl'] is not None:
                pnl_values.append(trade['pnl'])
            elif 'return' in trade and trade['return'] is not None:
                # Convert return to P&L if needed
                if 'entry_price' in trade and 'shares' in trade:
                    pnl_values.append(trade['return'] * trade['entry_price'] * trade['shares'])
        
        return np.array(pnl_values)
    
    def _calculate_observed_metrics(self, trades: List[Dict[str, Any]], initial_capital: float) -> Dict[str, float]:
        """Calculate observed performance metrics from trades."""
        if not trades:
            return {}
        
        # Extract P&L data
        pnl_data = self._extract_pnl_data(trades)
        
        if len(pnl_data) == 0:
            return {}
        
        # Calculate basic metrics
        total_pnl = np.sum(pnl_data)
        total_return = total_pnl / initial_capital
        
        # Calculate annualized return (assuming 252 trading days)
        n_trades = len(trades)
        if n_trades > 0:
            # Estimate time period from trade dates
            entry_dates = [trade.get('entry_date') for trade in trades if 'entry_date' in trade]
            exit_dates = [trade.get('exit_date') for trade in trades if 'exit_date' in trade]
            
            if entry_dates and exit_dates:
                try:
                    start_date = min(entry_dates)
                    end_date = max(exit_dates)
                    if isinstance(start_date, str):
                        start_date = pd.to_datetime(start_date)
                    if isinstance(end_date, str):
                        end_date = pd.to_datetime(end_date)
                    
                    days = (end_date - start_date).days
                    years = max(days / 252, 1)  # At least 1 year
                except:
                    years = 1
            else:
                years = 1
            
            annualized_return = (1 + total_return) ** (1 / years) - 1
        else:
            annualized_return = 0
        
        # Calculate volatility (from trade returns)
        trade_returns = [trade.get('return', 0) for trade in trades if 'return' in trade]
        if trade_returns:
            volatility = np.std(trade_returns) * np.sqrt(252)
        else:
            volatility = 0
        
        # Calculate Sharpe ratio
        sharpe_ratio = annualized_return / volatility if volatility > 0 else 0
        
        # Calculate CAGR (Compound Annual Growth Rate)
        cagr = annualized_return
        
        # Calculate max drawdown (simplified)
        cumulative_pnl = np.cumsum(pnl_data)
        running_max = np.maximum.accumulate(cumulative_pnl)
        drawdown = cumulative_pnl - running_max
        max_drawdown = np.min(drawdown) / initial_capital if len(drawdown) > 0 else 0
        
        # Calculate win rate
        win_rate = np.mean(pnl_data > 0) if len(pnl_data) > 0 else 0
        
        # Calculate profit factor
        positive_pnl = np.sum(pnl_data[pnl_data > 0])
        negative_pnl = np.sum(np.abs(pnl_data[pnl_data < 0]))
        profit_factor = positive_pnl / negative_pnl if negative_pnl > 0 else np.inf
        
        return {
            'total_return': total_return,
            'annualized_return': annualized_return,
            'cagr': cagr,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'profit_factor': profit_factor
        }
    
    def _calculate_returns_metrics(self, returns: pd.Series) -> Dict[str, float]:
        """Calculate performance metrics from returns series."""
        if returns.empty:
            return {}
        
        # Basic return metrics
        total_return = (1 + returns).prod() - 1
        annualized_return = (1 + returns).prod() ** (252 / len(returns)) - 1
        volatility = returns.std() * np.sqrt(252)
        sharpe_ratio = annualized_return / volatility if volatility > 0 else 0
        
        # Risk metrics
        cumulative_returns = (1 + returns).cumprod()
        running_max = cumulative_returns.expanding().max()
        drawdown = (cumulative_returns - running_max) / running_max
        max_drawdown = drawdown.min()
        
        # Win rate
        win_rate = (returns > 0).mean()
        
        # Profit factor
        positive_returns = returns[returns > 0].sum()
        negative_returns = abs(returns[returns < 0].sum())
        profit_factor = positive_returns / negative_returns if negative_returns > 0 else np.inf
        
        return {
            'total_return': total_return,
            'annualized_return': annualized_return,
            'cagr': annualized_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'profit_factor': profit_factor
        }
    
    def _run_bootstrap_resampling(self, pnl_data: np.ndarray, initial_capital: float) -> Dict[str, List[float]]:
        """Run bootstrap resampling on P&L data."""
        bootstrap_samples = {
            'total_return': [],
            'annualized_return': [],
            'cagr': [],
            'volatility': [],
            'sharpe_ratio': [],
            'max_drawdown': [],
            'win_rate': [],
            'profit_factor': []
        }
        
        # Use parallel processing for speed
        with ThreadPoolExecutor(max_workers=self.config.n_jobs) as executor:
            futures = []
            
            for i in range(self.config.n_bootstrap):
                future = executor.submit(self._single_bootstrap_sample, pnl_data, initial_capital)
                futures.append(future)
            
            for future in as_completed(futures):
                try:
                    sample_metrics = future.result()
                    for metric_name, value in sample_metrics.items():
                        if metric_name in bootstrap_samples:
                            bootstrap_samples[metric_name].append(value)
                except Exception as e:
                    logger.warning(f"Bootstrap sample failed: {e}")
                    continue
        
        return bootstrap_samples
    
    def _run_returns_bootstrap(self, returns: pd.Series) -> Dict[str, List[float]]:
        """Run bootstrap resampling on returns data."""
        bootstrap_samples = {
            'total_return': [],
            'annualized_return': [],
            'cagr': [],
            'volatility': [],
            'sharpe_ratio': [],
            'max_drawdown': [],
            'win_rate': [],
            'profit_factor': []
        }
        
        # Use parallel processing for speed
        with ThreadPoolExecutor(max_workers=self.config.n_jobs) as executor:
            futures = []
            
            for i in range(self.config.n_bootstrap):
                future = executor.submit(self._single_returns_bootstrap_sample, returns)
                futures.append(future)
            
            for future in as_completed(futures):
                try:
                    sample_metrics = future.result()
                    for metric_name, value in sample_metrics.items():
                        if metric_name in bootstrap_samples:
                            bootstrap_samples[metric_name].append(value)
                except Exception as e:
                    logger.warning(f"Bootstrap sample failed: {e}")
                    continue
        
        return bootstrap_samples
    
    def _single_bootstrap_sample(self, pnl_data: np.ndarray, initial_capital: float) -> Dict[str, float]:
        """Run a single bootstrap sample on P&L data."""
        # Resample P&L data with replacement
        bootstrap_pnl = np.random.choice(pnl_data, size=len(pnl_data), replace=True)
        
        # Calculate metrics for this sample
        total_pnl = np.sum(bootstrap_pnl)
        total_return = total_pnl / initial_capital
        
        # Annualized return (simplified)
        annualized_return = total_return  # Simplified for bootstrap
        
        # Volatility (from P&L)
        volatility = np.std(bootstrap_pnl) * np.sqrt(252) / initial_capital
        
        # Sharpe ratio
        sharpe_ratio = annualized_return / volatility if volatility > 0 else 0
        
        # Max drawdown
        cumulative_pnl = np.cumsum(bootstrap_pnl)
        running_max = np.maximum.accumulate(cumulative_pnl)
        drawdown = cumulative_pnl - running_max
        max_drawdown = np.min(drawdown) / initial_capital if len(drawdown) > 0 else 0
        
        # Win rate
        win_rate = np.mean(bootstrap_pnl > 0)
        
        # Profit factor
        positive_pnl = np.sum(bootstrap_pnl[bootstrap_pnl > 0])
        negative_pnl = np.sum(np.abs(bootstrap_pnl[bootstrap_pnl < 0]))
        profit_factor = positive_pnl / negative_pnl if negative_pnl > 0 else np.inf
        
        return {
            'total_return': total_return,
            'annualized_return': annualized_return,
            'cagr': annualized_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'profit_factor': profit_factor
        }
    
    def _single_returns_bootstrap_sample(self, returns: pd.Series) -> Dict[str, float]:
        """Run a single bootstrap sample on returns data."""
        # Resample returns with replacement
        bootstrap_returns = returns.sample(n=len(returns), replace=True)
        
        # Calculate metrics for this sample
        return self._calculate_returns_metrics(bootstrap_returns)
    
    def _calculate_confidence_interval(self, 
                                     metric_name: str, 
                                     observed_value: float, 
                                     bootstrap_values: List[float]) -> BootstrapResult:
        """Calculate confidence interval for a metric."""
        bootstrap_values = np.array(bootstrap_values)
        
        # Calculate confidence interval
        alpha = 1 - self.config.confidence_level
        lower_percentile = (alpha / 2) * 100
        upper_percentile = (1 - alpha / 2) * 100
        
        ci_lower = np.percentile(bootstrap_values, lower_percentile)
        ci_upper = np.percentile(bootstrap_values, upper_percentile)
        
        return BootstrapResult(
            metric_name=metric_name,
            observed_value=observed_value,
            ci_lower=ci_lower,
            ci_upper=ci_upper,
            confidence_level=self.config.confidence_level,
            bootstrap_values=bootstrap_values.tolist(),
            mean_bootstrap=bootstrap_values.mean(),
            std_bootstrap=bootstrap_values.std()
        )
    
    def create_histogram_plots(self) -> Dict[str, go.Figure]:
        """Create histogram plots for bootstrap results."""
        plots = {}
        
        for metric_name, result in self.results.items():
            fig = go.Figure()
            
            # Add histogram
            fig.add_trace(go.Histogram(
                x=result.bootstrap_values,
                nbinsx=50,
                name=f'Bootstrap {metric_name}',
                opacity=0.7,
                marker_color='lightblue'
            ))
            
            # Add observed value as vertical line
            fig.add_vline(
                x=result.observed_value,
                line_dash="dash",
                line_color="red",
                line_width=3,
                annotation_text=f"Observed: {result.observed_value:.4f}",
                annotation_position="top"
            )
            
            # Add confidence interval lines
            fig.add_vline(
                x=result.ci_lower,
                line_dash="dot",
                line_color="green",
                line_width=2,
                annotation_text=f"CI Lower: {result.ci_lower:.4f}",
                annotation_position="bottom"
            )
            
            fig.add_vline(
                x=result.ci_upper,
                line_dash="dot",
                line_color="green",
                line_width=2,
                annotation_text=f"CI Upper: {result.ci_upper:.4f}",
                annotation_position="bottom"
            )
            
            # Update layout
            fig.update_layout(
                title=f'Bootstrap Distribution: {metric_name.replace("_", " ").title()}',
                xaxis_title=metric_name.replace("_", " ").title(),
                yaxis_title='Frequency',
                showlegend=False,
                template='plotly_white'
            )
            
            plots[metric_name] = fig
        
        return plots
    
    def get_summary(self) -> pd.DataFrame:
        """Get a summary of all bootstrap results."""
        if not self.results:
            return pd.DataFrame()
        
        data = []
        for result in self.results.values():
            data.append({
                'Metric': result.metric_name,
                'Observed': result.observed_value,
                'CI Lower': result.ci_lower,
                'CI Upper': result.ci_upper,
                'CI Width': result.ci_upper - result.ci_lower,
                'Bootstrap Mean': result.mean_bootstrap,
                'Bootstrap Std': result.std_bootstrap,
                'Confidence Level': f"{result.confidence_level:.1%}"
            })
        
        return pd.DataFrame(data)
    
    def log_to_mlflow(self, run_id: str = None):
        """Log bootstrap results to MLflow."""
        try:
            import mlflow
            
            # Check if there's an active run
            if mlflow.active_run() is None:
                logger.warning("No active MLflow run, skipping Bootstrap logging")
                return
            
            if run_id:
                mlflow.set_tag("run_id", run_id)
            
            # Log configuration
            mlflow.log_params({
                "bootstrap_n_samples": self.config.n_bootstrap,
                "bootstrap_confidence_level": self.config.confidence_level,
                "bootstrap_resample_method": self.config.resample_method
            })
            
            # Log results
            for result in self.results.values():
                mlflow.log_metrics({
                    f"bootstrap_{result.metric_name}_observed": result.observed_value,
                    f"bootstrap_{result.metric_name}_ci_lower": result.ci_lower,
                    f"bootstrap_{result.metric_name}_ci_upper": result.ci_upper,
                    f"bootstrap_{result.metric_name}_ci_width": result.ci_upper - result.ci_lower,
                    f"bootstrap_{result.metric_name}_bootstrap_mean": result.mean_bootstrap,
                    f"bootstrap_{result.metric_name}_bootstrap_std": result.std_bootstrap
                })
            
            logger.info("Bootstrap results logged to MLflow")
            
        except ImportError:
            logger.warning("MLflow not available, skipping logging")
        except Exception as e:
            logger.error(f"Failed to log bootstrap results to MLflow: {e}")


def run_bootstrap_analysis(
    trades: List[Dict[str, Any]] = None,
    returns: pd.Series = None,
    config: BootstrapConfig = None,
    initial_capital: float = 100000.0,
    strategy_name: str = "strategy"
) -> Tuple[Dict[str, BootstrapResult], pd.DataFrame, Dict[str, go.Figure]]:
    """
    Convenience function to run bootstrap analysis.
    
    Args:
        trades: List of trade dictionaries with P&L data
        returns: Series of strategy returns
        config: Bootstrap configuration
        initial_capital: Initial capital for calculations
        strategy_name: Name of the strategy
        
    Returns:
        Tuple of (results_dict, summary_dataframe, histogram_plots)
    """
    analyzer = BootstrapAnalyzer(config)
    
    # Analyze trades if provided
    if trades:
        results = analyzer.analyze_trades(trades, initial_capital, strategy_name)
    elif returns is not None:
        results = analyzer.analyze_returns(returns, strategy_name)
    else:
        raise ValueError("Either trades or returns must be provided")
    
    # Get summary
    summary = analyzer.get_summary()
    
    # Create histograms
    plots = analyzer.create_histogram_plots()
    
    return results, summary, plots
