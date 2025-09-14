"""
Comprehensive metrics for trading strategy evaluation.

This module provides robust metrics including Sharpe ratio, maximum drawdown,
turnover, and drawdown distributions for proper strategy evaluation.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union
from scipy import stats
import logging

logger = logging.getLogger(__name__)


def calculate_comprehensive_metrics(returns: pd.Series, 
                                  benchmark_returns: Optional[pd.Series] = None,
                                  risk_free_rate: float = 0.02,
                                  periods_per_year: int = 252) -> Dict[str, float]:
    """
    Calculate comprehensive trading strategy metrics.
    
    Args:
        returns: Strategy returns series
        benchmark_returns: Benchmark returns for comparison
        risk_free_rate: Annual risk-free rate
        periods_per_year: Number of periods per year
        
    Returns:
        Dict[str, float]: Dictionary of calculated metrics
    """
    if returns.empty:
        return {}
        
    # Convert to numpy array for calculations
    returns_array = returns.dropna().values
    if len(returns_array) == 0:
        return {}
    
    # Basic return metrics
    total_return = (1 + returns_array).prod() - 1
    annualized_return = (1 + returns_array).prod() ** (periods_per_year / len(returns_array)) - 1
    
    # Volatility metrics
    volatility = returns_array.std() * np.sqrt(periods_per_year)
    
    # Risk-adjusted metrics
    sharpe_ratio = (annualized_return - risk_free_rate) / volatility if volatility > 0 else 0
    
    # Sortino ratio (downside deviation)
    downside_returns = returns_array[returns_array < 0]
    downside_volatility = downside_returns.std() * np.sqrt(periods_per_year) if len(downside_returns) > 0 else 0
    sortino_ratio = (annualized_return - risk_free_rate) / downside_volatility if downside_volatility > 0 else 0
    
    # Calmar ratio (return / max drawdown)
    max_drawdown = calculate_max_drawdown(returns)
    calmar_ratio = annualized_return / abs(max_drawdown) if max_drawdown != 0 else 0
    
    # Drawdown metrics
    drawdown_metrics = calculate_drawdown_metrics(returns)
    
    # Turnover metrics (if position data available)
    turnover_metrics = calculate_turnover_metrics(returns)
    
    # Win rate and profit factor
    win_rate = (returns_array > 0).mean()
    profit_factor = calculate_profit_factor(returns_array)
    
    # VaR and CVaR
    var_95 = np.percentile(returns_array, 5)
    var_99 = np.percentile(returns_array, 1)
    cvar_95 = returns_array[returns_array <= var_95].mean()
    cvar_99 = returns_array[returns_array <= var_99].mean()
    
    # Skewness and Kurtosis
    skewness = stats.skew(returns_array)
    kurtosis = stats.kurtosis(returns_array)
    
    # Tail ratio (95th percentile / 5th percentile)
    tail_ratio = np.percentile(returns_array, 95) / abs(np.percentile(returns_array, 5)) if np.percentile(returns_array, 5) != 0 else 0
    
    # Common sense ratio (mean return / mean absolute return)
    common_sense_ratio = returns_array.mean() / np.abs(returns_array).mean() if np.abs(returns_array).mean() != 0 else 0
    
    # Compile metrics
    metrics = {
        # Return metrics
        'total_return': total_return,
        'annualized_return': annualized_return,
        'volatility': volatility,
        
        # Risk-adjusted metrics
        'sharpe_ratio': sharpe_ratio,
        'sortino_ratio': sortino_ratio,
        'calmar_ratio': calmar_ratio,
        
        # Drawdown metrics
        'max_drawdown': max_drawdown,
        'max_drawdown_duration': drawdown_metrics['max_drawdown_duration'],
        'avg_drawdown': drawdown_metrics['avg_drawdown'],
        'drawdown_std': drawdown_metrics['drawdown_std'],
        
        # Turnover metrics
        'avg_turnover': turnover_metrics['avg_turnover'],
        'turnover_std': turnover_metrics['turnover_std'],
        
        # Risk metrics
        'var_95': var_95,
        'var_99': var_99,
        'cvar_95': cvar_95,
        'cvar_99': cvar_99,
        
        # Distribution metrics
        'skewness': skewness,
        'kurtosis': kurtosis,
        'tail_ratio': tail_ratio,
        'common_sense_ratio': common_sense_ratio,
        
        # Performance metrics
        'win_rate': win_rate,
        'profit_factor': profit_factor,
        
        # Additional metrics
        'total_periods': len(returns_array),
        'positive_periods': (returns_array > 0).sum(),
        'negative_periods': (returns_array < 0).sum(),
    }
    
    # Benchmark comparison if provided
    if benchmark_returns is not None:
        benchmark_array = benchmark_returns.dropna().values
        if len(benchmark_array) > 0:
            # Align returns and benchmark
            common_dates = returns.index.intersection(benchmark_returns.index)
            if len(common_dates) > 0:
                aligned_returns = returns[common_dates]
                aligned_benchmark = benchmark_returns[common_dates]
                
                # Calculate excess returns
                excess_returns = aligned_returns - aligned_benchmark
                excess_metrics = calculate_comprehensive_metrics(excess_returns, risk_free_rate=risk_free_rate, periods_per_year=periods_per_year)
                
                # Add benchmark-specific metrics
                metrics.update({
                    'excess_return': excess_metrics.get('annualized_return', 0),
                    'excess_sharpe': excess_metrics.get('sharpe_ratio', 0),
                    'information_ratio': excess_metrics.get('sharpe_ratio', 0),
                    'tracking_error': excess_metrics.get('volatility', 0),
                })
    
    return metrics


def calculate_drawdown_metrics(returns: pd.Series) -> Dict[str, float]:
    """
    Calculate comprehensive drawdown metrics.
    
    Args:
        returns: Strategy returns series
        
    Returns:
        Dict[str, float]: Drawdown metrics
    """
    if returns.empty:
        return {
            'max_drawdown': 0,
            'max_drawdown_duration': 0,
            'avg_drawdown': 0,
            'drawdown_std': 0,
            'drawdown_skewness': 0,
            'drawdown_kurtosis': 0
        }
    
    # Calculate cumulative returns
    cumulative_returns = (1 + returns).cumprod()
    
    # Calculate running maximum
    running_max = cumulative_returns.expanding().max()
    
    # Calculate drawdown
    drawdown = (cumulative_returns - running_max) / running_max
    
    # Drawdown periods (consecutive negative drawdowns)
    drawdown_periods = []
    current_period = 0
    
    for dd in drawdown:
        if dd < 0:
            current_period += 1
        else:
            if current_period > 0:
                drawdown_periods.append(current_period)
            current_period = 0
    
    # Add final period if it ends in drawdown
    if current_period > 0:
        drawdown_periods.append(current_period)
    
    # Calculate metrics
    max_drawdown = drawdown.min()
    max_drawdown_duration = max(drawdown_periods) if drawdown_periods else 0
    avg_drawdown = drawdown[drawdown < 0].mean() if (drawdown < 0).any() else 0
    drawdown_std = drawdown[drawdown < 0].std() if (drawdown < 0).any() else 0
    
    # Distribution metrics for drawdowns
    drawdown_values = drawdown[drawdown < 0].values
    if len(drawdown_values) > 0:
        drawdown_skewness = stats.skew(drawdown_values)
        drawdown_kurtosis = stats.kurtosis(drawdown_values)
    else:
        drawdown_skewness = 0
        drawdown_kurtosis = 0
    
    return {
        'max_drawdown': max_drawdown,
        'max_drawdown_duration': max_drawdown_duration,
        'avg_drawdown': avg_drawdown,
        'drawdown_std': drawdown_std,
        'drawdown_skewness': drawdown_skewness,
        'drawdown_kurtosis': drawdown_kurtosis,
        'drawdown_periods_count': len(drawdown_periods),
        'avg_drawdown_duration': np.mean(drawdown_periods) if drawdown_periods else 0
    }


def calculate_turnover_metrics(returns: pd.Series, 
                              position_changes: Optional[pd.Series] = None) -> Dict[str, float]:
    """
    Calculate turnover metrics.
    
    Args:
        returns: Strategy returns series
        position_changes: Position changes series (if available)
        
    Returns:
        Dict[str, float]: Turnover metrics
    """
    if position_changes is not None and not position_changes.empty:
        # Use actual position changes if available
        turnover = position_changes.abs()
    else:
        # Estimate turnover from returns (simplified)
        # This is a rough approximation - real turnover requires position data
        turnover = returns.abs() * 0.1  # Assume 10% of return magnitude as turnover
    
    return {
        'avg_turnover': turnover.mean(),
        'turnover_std': turnover.std(),
        'max_turnover': turnover.max(),
        'turnover_skewness': stats.skew(turnover.dropna()) if not turnover.empty else 0
    }


def calculate_profit_factor(returns: np.ndarray) -> float:
    """
    Calculate profit factor (gross profit / gross loss).
    
    Args:
        returns: Returns array
        
    Returns:
        float: Profit factor
    """
    gross_profit = returns[returns > 0].sum()
    gross_loss = abs(returns[returns < 0].sum())
    
    return gross_profit / gross_loss if gross_loss != 0 else float('inf')


def calculate_max_drawdown(returns: pd.Series) -> float:
    """
    Calculate maximum drawdown.
    
    Args:
        returns: Strategy returns series
        
    Returns:
        float: Maximum drawdown
    """
    if returns.empty:
        return 0.0
    
    cumulative_returns = (1 + returns).cumprod()
    running_max = cumulative_returns.expanding().max()
    drawdown = (cumulative_returns - running_max) / running_max
    
    return drawdown.min()


def calculate_rolling_metrics(returns: pd.Series, window: int = 252) -> pd.DataFrame:
    """
    Calculate rolling metrics for time series analysis.
    
    Args:
        returns: Strategy returns series
        window: Rolling window size
        
    Returns:
        pd.DataFrame: Rolling metrics
    """
    if returns.empty:
        return pd.DataFrame()
    
    rolling_metrics = pd.DataFrame(index=returns.index)
    
    # Rolling returns
    rolling_metrics['rolling_return'] = returns.rolling(window).apply(lambda x: (1 + x).prod() - 1)
    
    # Rolling volatility
    rolling_metrics['rolling_volatility'] = returns.rolling(window).std() * np.sqrt(252)
    
    # Rolling Sharpe ratio
    rolling_metrics['rolling_sharpe'] = rolling_metrics['rolling_return'] / rolling_metrics['rolling_volatility']
    
    # Rolling maximum drawdown
    rolling_metrics['rolling_max_dd'] = returns.rolling(window).apply(
        lambda x: calculate_max_drawdown(pd.Series(x, index=returns.index[:len(x)]))
    )
    
    return rolling_metrics


def calculate_regime_metrics(returns: pd.Series, 
                           regime_indicator: pd.Series) -> Dict[str, Dict[str, float]]:
    """
    Calculate metrics for different market regimes.
    
    Args:
        returns: Strategy returns series
        regime_indicator: Market regime indicator (e.g., 'bull', 'bear', 'sideways')
        
    Returns:
        Dict[str, Dict[str, float]]: Metrics for each regime
    """
    regime_metrics = {}
    
    for regime in regime_indicator.unique():
        regime_returns = returns[regime_indicator == regime]
        if not regime_returns.empty:
            regime_metrics[regime] = calculate_comprehensive_metrics(regime_returns)
    
    return regime_metrics


def format_metrics_report(metrics: Dict[str, float], 
                         decimal_places: int = 4) -> str:
    """
    Format metrics into a readable report.
    
    Args:
        metrics: Dictionary of metrics
        decimal_places: Number of decimal places to show
        
    Returns:
        str: Formatted metrics report
    """
    report = "=" * 60 + "\n"
    report += "STRATEGY PERFORMANCE METRICS\n"
    report += "=" * 60 + "\n\n"
    
    # Return metrics
    report += "RETURN METRICS:\n"
    report += f"  Total Return:        {metrics.get('total_return', 0):.{decimal_places}%}\n"
    report += f"  Annualized Return:   {metrics.get('annualized_return', 0):.{decimal_places}%}\n"
    report += f"  Volatility:          {metrics.get('volatility', 0):.{decimal_places}%}\n\n"
    
    # Risk-adjusted metrics
    report += "RISK-ADJUSTED METRICS:\n"
    report += f"  Sharpe Ratio:        {metrics.get('sharpe_ratio', 0):.{decimal_places}f}\n"
    report += f"  Sortino Ratio:       {metrics.get('sortino_ratio', 0):.{decimal_places}f}\n"
    report += f"  Calmar Ratio:        {metrics.get('calmar_ratio', 0):.{decimal_places}f}\n\n"
    
    # Drawdown metrics
    report += "DRAWDOWN METRICS:\n"
    report += f"  Max Drawdown:        {metrics.get('max_drawdown', 0):.{decimal_places}%}\n"
    report += f"  Max DD Duration:     {metrics.get('max_drawdown_duration', 0):.0f} periods\n"
    report += f"  Avg Drawdown:        {metrics.get('avg_drawdown', 0):.{decimal_places}%}\n\n"
    
    # Risk metrics
    report += "RISK METRICS:\n"
    report += f"  VaR (95%):           {metrics.get('var_95', 0):.{decimal_places}%}\n"
    report += f"  VaR (99%):           {metrics.get('var_99', 0):.{decimal_places}%}\n"
    report += f"  CVaR (95%):          {metrics.get('cvar_95', 0):.{decimal_places}%}\n"
    report += f"  CVaR (99%):          {metrics.get('cvar_99', 0):.{decimal_places}%}\n\n"
    
    # Performance metrics
    report += "PERFORMANCE METRICS:\n"
    report += f"  Win Rate:            {metrics.get('win_rate', 0):.{decimal_places}%}\n"
    report += f"  Profit Factor:       {metrics.get('profit_factor', 0):.{decimal_places}f}\n"
    report += f"  Tail Ratio:          {metrics.get('tail_ratio', 0):.{decimal_places}f}\n\n"
    
    # Distribution metrics
    report += "DISTRIBUTION METRICS:\n"
    report += f"  Skewness:            {metrics.get('skewness', 0):.{decimal_places}f}\n"
    report += f"  Kurtosis:            {metrics.get('kurtosis', 0):.{decimal_places}f}\n"
    report += f"  Common Sense Ratio:  {metrics.get('common_sense_ratio', 0):.{decimal_places}f}\n\n"
    
    report += "=" * 60
    
    return report
