"""
Monte Carlo Permutation Testing (MCPT) Module

This module implements statistical significance testing for backtest results
using Monte Carlo permutation tests. It helps determine if observed performance
could have occurred by chance.

Author: Neural Quant Team
Date: 2024
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

logger = logging.getLogger(__name__)


@dataclass
class MCPTResult:
    """Container for MCPT test results."""
    metric_name: str
    observed_value: float
    null_mean: float
    null_std: float
    p_value: float
    confidence_interval: Tuple[float, float]
    is_significant: bool
    significance_level: float = 0.05
    null_values: List[float] = None  # Add null_values for UI visualization
    confidence_level: float = 0.95  # Add confidence_level for UI compatibility


@dataclass
class MCPTConfig:
    """Configuration for MCPT testing."""
    n_permutations: int = 1000
    block_size: Optional[int] = None
    confidence_level: float = 0.95
    significance_level: float = 0.05
    random_seed: Optional[int] = None
    n_jobs: int = 4


class MonteCarloPermutationTester:
    """
    Monte Carlo Permutation Tester for backtest results.
    
    This class implements statistical significance testing by:
    1. Shuffling returns or permuting signals
    2. Running multiple permutations
    3. Computing null distributions for key metrics
    4. Calculating p-values and confidence intervals
    """
    
    def __init__(self, config: MCPTConfig = None):
        """Initialize the MCPT tester."""
        self.config = config or MCPTConfig()
        self.results: Dict[str, MCPTResult] = {}
        
        # Set random seed for reproducibility
        if self.config.random_seed is not None:
            np.random.seed(self.config.random_seed)
    
    def test_returns_significance(
        self, 
        returns: pd.Series, 
        strategy_name: str = "strategy"
    ) -> Dict[str, MCPTResult]:
        """
        Test significance of returns-based metrics.
        
        Args:
            returns: Series of strategy returns
            strategy_name: Name of the strategy for logging
            
        Returns:
            Dictionary of metric names to MCPTResult objects
        """
        logger.info(f"Running MCPT on returns for {strategy_name}")
        logger.info(f"Returns length: {len(returns)}, N permutations: {self.config.n_permutations}")
        
        # Calculate observed metrics
        observed_metrics = self._calculate_returns_metrics(returns)
        
        # Run permutations
        null_distributions = self._run_returns_permutations(returns)
        
        # Calculate p-values and confidence intervals
        results = {}
        for metric_name, observed_value in observed_metrics.items():
            null_values = null_distributions[metric_name]
            
            result = self._calculate_significance(
                metric_name, observed_value, null_values
            )
            results[metric_name] = result
            
            logger.info(f"{metric_name}: observed={observed_value:.4f}, "
                       f"null_mean={result.null_mean:.4f}, "
                       f"p_value={result.p_value:.4f}, "
                       f"significant={result.is_significant}")
        
        self.results.update(results)
        return results
    
    def test_signals_significance(
        self, 
        signals: pd.Series, 
        returns: pd.Series,
        strategy_name: str = "strategy"
    ) -> Dict[str, MCPTResult]:
        """
        Test significance of signal-based metrics.
        
        Args:
            signals: Series of trading signals (1, 0, -1)
            returns: Series of market returns
            strategy_name: Name of the strategy for logging
            
        Returns:
            Dictionary of metric names to MCPTResult objects
        """
        logger.info(f"Running MCPT on signals for {strategy_name}")
        logger.info(f"Signals length: {len(signals)}, N permutations: {self.config.n_permutations}")
        
        # Calculate observed metrics
        observed_metrics = self._calculate_signal_metrics(signals, returns)
        
        # Run permutations
        null_distributions = self._run_signal_permutations(signals, returns)
        
        # Calculate p-values and confidence intervals
        results = {}
        for metric_name, observed_value in observed_metrics.items():
            null_values = null_distributions[metric_name]
            
            result = self._calculate_significance(
                metric_name, observed_value, null_values
            )
            results[metric_name] = result
            
            logger.info(f"{metric_name}: observed={observed_value:.4f}, "
                       f"null_mean={result.null_mean:.4f}, "
                       f"p_value={result.p_value:.4f}, "
                       f"significant={result.is_significant}")
        
        self.results.update(results)
        return results
    
    def _calculate_returns_metrics(self, returns: pd.Series) -> Dict[str, float]:
        """Calculate key performance metrics from returns."""
        metrics = {}
        
        # Basic return metrics
        metrics['total_return'] = (1 + returns).prod() - 1
        metrics['annualized_return'] = (1 + returns).prod() ** (252 / len(returns)) - 1
        metrics['volatility'] = returns.std() * np.sqrt(252)
        metrics['sharpe_ratio'] = metrics['annualized_return'] / metrics['volatility'] if metrics['volatility'] > 0 else 0
        
        # Risk metrics
        cumulative_returns = (1 + returns).cumprod()
        running_max = cumulative_returns.expanding().max()
        drawdown = (cumulative_returns - running_max) / running_max
        metrics['max_drawdown'] = drawdown.min()
        
        # Win rate
        metrics['win_rate'] = (returns > 0).mean()
        
        # Profit factor
        positive_returns = returns[returns > 0].sum()
        negative_returns = abs(returns[returns < 0].sum())
        metrics['profit_factor'] = positive_returns / negative_returns if negative_returns > 0 else np.inf
        
        return metrics
    
    def _calculate_signal_metrics(self, signals: pd.Series, returns: pd.Series) -> Dict[str, float]:
        """Calculate signal-based performance metrics."""
        metrics = {}
        
        # Signal-return correlation
        metrics['signal_return_correlation'] = signals.corr(returns)
        
        # Signal accuracy
        metrics['signal_accuracy'] = (signals * returns > 0).mean()
        
        # Signal frequency
        metrics['signal_frequency'] = (signals != 0).mean()
        
        # Signal-return alignment
        metrics['signal_return_alignment'] = (signals * returns).mean()
        
        return metrics
    
    def _run_returns_permutations(self, returns: pd.Series) -> Dict[str, List[float]]:
        """Run permutations on returns data."""
        null_distributions = {metric: [] for metric in self._calculate_returns_metrics(returns).keys()}
        
        # Use parallel processing for speed
        with ThreadPoolExecutor(max_workers=self.config.n_jobs) as executor:
            futures = []
            
            for i in range(self.config.n_permutations):
                future = executor.submit(self._single_returns_permutation, returns)
                futures.append(future)
            
            for future in as_completed(futures):
                try:
                    permuted_metrics = future.result()
                    for metric_name, value in permuted_metrics.items():
                        null_distributions[metric_name].append(value)
                except Exception as e:
                    logger.warning(f"Permutation failed: {e}")
                    continue
        
        return null_distributions
    
    def _run_signal_permutations(self, signals: pd.Series, returns: pd.Series) -> Dict[str, List[float]]:
        """Run permutations on signals data."""
        null_distributions = {metric: [] for metric in self._calculate_signal_metrics(signals, returns).keys()}
        
        # Use parallel processing for speed
        with ThreadPoolExecutor(max_workers=self.config.n_jobs) as executor:
            futures = []
            
            for i in range(self.config.n_permutations):
                future = executor.submit(self._single_signal_permutation, signals, returns)
                futures.append(future)
            
            for future in as_completed(futures):
                try:
                    permuted_metrics = future.result()
                    for metric_name, value in permuted_metrics.items():
                        null_distributions[metric_name].append(value)
                except Exception as e:
                    logger.warning(f"Permutation failed: {e}")
                    continue
        
        return null_distributions
    
    def _single_returns_permutation(self, returns: pd.Series) -> Dict[str, float]:
        """Run a single permutation on returns."""
        if self.config.block_size is None:
            # Simple shuffle
            permuted_returns = returns.sample(frac=1.0).reset_index(drop=True)
        else:
            # Block permutation to preserve autocorrelation
            permuted_returns = self._block_permute(returns, self.config.block_size)
        
        return self._calculate_returns_metrics(permuted_returns)
    
    def _single_signal_permutation(self, signals: pd.Series, returns: pd.Series) -> Dict[str, float]:
        """Run a single permutation on signals."""
        if self.config.block_size is None:
            # Simple shuffle
            permuted_signals = signals.sample(frac=1.0).reset_index(drop=True)
        else:
            # Block permutation to preserve autocorrelation
            permuted_signals = self._block_permute(signals, self.config.block_size)
        
        return self._calculate_signal_metrics(permuted_signals, returns)
    
    def _block_permute(self, series: pd.Series, block_size: int) -> pd.Series:
        """Permute series in blocks to preserve autocorrelation."""
        n = len(series)
        n_blocks = n // block_size
        
        # Create blocks
        blocks = []
        for i in range(n_blocks):
            start_idx = i * block_size
            end_idx = min((i + 1) * block_size, n)
            blocks.append(series.iloc[start_idx:end_idx])
        
        # Shuffle blocks
        np.random.shuffle(blocks)
        
        # Reconstruct series
        permuted_series = pd.concat(blocks, ignore_index=True)
        
        # Handle remaining elements
        if n % block_size != 0:
            remaining = series.iloc[n_blocks * block_size:]
            permuted_series = pd.concat([permuted_series, remaining], ignore_index=True)
        
        return permuted_series
    
    def _calculate_significance(
        self, 
        metric_name: str, 
        observed_value: float, 
        null_values: List[float]
    ) -> MCPTResult:
        """Calculate significance statistics for a metric."""
        null_values = np.array(null_values)
        
        # Calculate p-value
        if metric_name in ['max_drawdown']:
            # For max drawdown, we want to know if observed is better (less negative)
            p_value = (null_values <= observed_value).mean()
        else:
            # For other metrics, we want to know if observed is better (higher)
            p_value = (null_values >= observed_value).mean()
        
        # Calculate confidence interval
        alpha = 1 - self.config.confidence_level
        lower_percentile = (alpha / 2) * 100
        upper_percentile = (1 - alpha / 2) * 100
        
        confidence_interval = (
            np.percentile(null_values, lower_percentile),
            np.percentile(null_values, upper_percentile)
        )
        
        # Determine significance
        is_significant = p_value < self.config.significance_level
        
        return MCPTResult(
            metric_name=metric_name,
            observed_value=observed_value,
            null_mean=null_values.mean(),
            null_std=null_values.std(),
            p_value=p_value,
            confidence_interval=confidence_interval,
            is_significant=is_significant,
            significance_level=self.config.significance_level,
            null_values=null_values.tolist(),
            confidence_level=self.config.confidence_level
        )
    
    def get_summary(self) -> pd.DataFrame:
        """Get a summary of all MCPT results."""
        if not self.results:
            return pd.DataFrame()
        
        data = []
        for result in self.results.values():
            data.append({
                'Metric': result.metric_name,
                'Observed': result.observed_value,
                'Null Mean': result.null_mean,
                'Null Std': result.null_std,
                'P-Value': result.p_value,
                'Significant': result.is_significant,
                'CI Lower': result.confidence_interval[0],
                'CI Upper': result.confidence_interval[1]
            })
        
        return pd.DataFrame(data)
    
    def log_to_mlflow(self, run_id: str = None):
        """Log MCPT results to MLflow."""
        try:
            import mlflow
            
            # Check if there's an active run
            if mlflow.active_run() is None:
                logger.warning("No active MLflow run, skipping MCPT logging")
                return
            
            if run_id:
                mlflow.set_tag("run_id", run_id)
            
            # Log configuration
            mlflow.log_params({
                "mcpt_n_permutations": self.config.n_permutations,
                "mcpt_block_size": self.config.block_size,
                "mcpt_confidence_level": self.config.confidence_level,
                "mcpt_significance_level": self.config.significance_level
            })
            
            # Log results
            for result in self.results.values():
                mlflow.log_metrics({
                    f"mcpt_{result.metric_name}_observed": result.observed_value,
                    f"mcpt_{result.metric_name}_null_mean": result.null_mean,
                    f"mcpt_{result.metric_name}_null_std": result.null_std,
                    f"mcpt_{result.metric_name}_p_value": result.p_value,
                    f"mcpt_{result.metric_name}_significant": float(result.is_significant)
                })
            
            logger.info("MCPT results logged to MLflow")
            
        except ImportError:
            logger.warning("MLflow not available, skipping logging")
        except Exception as e:
            logger.error(f"Failed to log MCPT results to MLflow: {e}")


def run_mcpt_test(
    returns: pd.Series,
    signals: pd.Series = None,
    config: MCPTConfig = None,
    strategy_name: str = "strategy"
) -> Tuple[Dict[str, MCPTResult], pd.DataFrame]:
    """
    Convenience function to run MCPT test.
    
    Args:
        returns: Series of strategy returns
        signals: Optional series of trading signals
        config: MCPT configuration
        strategy_name: Name of the strategy
        
    Returns:
        Tuple of (results_dict, summary_dataframe)
    """
    tester = MonteCarloPermutationTester(config)
    
    # Test returns significance
    results = tester.test_returns_significance(returns, strategy_name)
    
    # Test signals significance if provided
    if signals is not None:
        signal_results = tester.test_signals_significance(signals, returns, strategy_name)
        results.update(signal_results)
    
    # Get summary
    summary = tester.get_summary()
    
    return results, summary
