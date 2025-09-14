"""Performance metrics calculation for trading strategies."""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional

def calculate_performance_metrics(equity_curve: pd.DataFrame, 
                                trades: list = None,
                                risk_free_rate: float = 0.02) -> Dict[str, Any]:
    """
    Calculate comprehensive performance metrics.
    
    Args:
        equity_curve: DataFrame with equity values over time
        trades: List of trade dictionaries
        risk_free_rate: Risk-free rate for Sharpe calculation
        
    Returns:
        Dictionary of performance metrics
    """
    if equity_curve.empty:
        return {}
    
    # Ensure we have returns
    if 'returns' not in equity_curve.columns:
        equity_curve = equity_curve.copy()
        equity_curve['returns'] = equity_curve['equity'].pct_change()
    
    returns = equity_curve['returns'].dropna()
    
    # Basic metrics
    total_return = (equity_curve['equity'].iloc[-1] / equity_curve['equity'].iloc[0]) - 1
    annualized_return = (1 + total_return) ** (252 / len(equity_curve)) - 1
    volatility = returns.std() * np.sqrt(252)
    
    # Risk-adjusted metrics
    excess_returns = returns - risk_free_rate / 252
    sharpe_ratio = excess_returns.mean() / returns.std() * np.sqrt(252) if returns.std() > 0 else 0
    
    # Drawdown analysis
    rolling_max = equity_curve['equity'].expanding().max()
    drawdown = (equity_curve['equity'] - rolling_max) / rolling_max
    max_drawdown = drawdown.min()
    
    # Additional metrics
    win_rate = len(returns[returns > 0]) / len(returns) if len(returns) > 0 else 0
    profit_factor = returns[returns > 0].sum() / abs(returns[returns < 0].sum()) if len(returns[returns < 0]) > 0 else np.inf
    
    # VaR and CVaR (95% confidence)
    var_95 = np.percentile(returns, 5)
    cvar_95 = returns[returns <= var_95].mean()
    
    # Skewness and Kurtosis
    skewness = returns.skew()
    kurtosis = returns.kurtosis()
    
    # Tail ratio
    tail_ratio = abs(returns.quantile(0.95)) / abs(returns.quantile(0.05)) if abs(returns.quantile(0.05)) > 0 else np.inf
    
    metrics = {
        'total_return': total_return,
        'annualized_return': annualized_return,
        'volatility': volatility,
        'sharpe_ratio': sharpe_ratio,
        'max_drawdown': max_drawdown,
        'win_rate': win_rate,
        'profit_factor': profit_factor,
        'var_95': var_95,
        'cvar_95': cvar_95,
        'skewness': skewness,
        'kurtosis': kurtosis,
        'tail_ratio': tail_ratio
    }
    
    # Trade-based metrics
    if trades:
        trade_returns = [t['return'] for t in trades if 'return' in t]
        if trade_returns:
            metrics.update({
                'total_trades': len(trades),
                'winning_trades': len([r for r in trade_returns if r > 0]),
                'losing_trades': len([r for r in trade_returns if r < 0]),
                'avg_win': np.mean([r for r in trade_returns if r > 0]) if any(r > 0 for r in trade_returns) else 0,
                'avg_loss': np.mean([r for r in trade_returns if r < 0]) if any(r < 0 for r in trade_returns) else 0,
                'largest_win': max(trade_returns) if trade_returns else 0,
                'largest_loss': min(trade_returns) if trade_returns else 0,
            })
    
    return metrics
