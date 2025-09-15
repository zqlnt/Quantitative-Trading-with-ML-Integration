"""
Artifact Type Definitions

This module defines the structure for different types of artifacts
that will be saved to MLflow runs.

Author: Neural Quant Team
Date: 2024
"""

from enum import Enum
from dataclasses import dataclass, asdict
from typing import Dict, Any, List, Optional, Union
import pandas as pd
import json
from datetime import datetime


class ArtifactType(Enum):
    """Types of artifacts that can be saved."""
    PARAMS = "params.json"
    METRICS = "metrics.json"
    EQUITY = "equity.csv"
    TRADES = "trades.csv"
    MCPT = "mcpt.json"
    BOOTSTRAP = "bootstrap.json"
    WALKFORWARD = "walkforward.parquet"
    WEIGHTS = "weights.csv"
    SUMMARY = "summary.md"


@dataclass
class ParamsArtifact:
    """Parameters artifact containing strategy and configuration details."""
    strategy: str
    strategy_params: Dict[str, Any]
    tickers: List[str]
    start_date: str
    end_date: str
    initial_capital: float
    commission: float
    slippage: float
    
    # C1 - Regime Filter
    regime_filter_enabled: bool = False
    regime_proxy_symbol: Optional[str] = None
    regime_rule: Optional[str] = None
    regime_sma_window: Optional[int] = None
    
    # C2 - Volatility Targeting
    volatility_targeting_enabled: bool = False
    target_vol: Optional[float] = None
    vol_lookback_window: Optional[int] = None
    vol_scale_cap: Optional[float] = None
    
    # C3 - Allocation Method
    allocation_method: str = "equal_weight"
    vol_lookback: Optional[int] = None
    
    # C4 - Position Management
    max_position_pct: Optional[float] = None
    rebalance_frequency: Optional[str] = None
    min_rebalance_interval: Optional[int] = None
    turnover_threshold: Optional[float] = None
    
    # C5 - Basic Exits
    basic_exits_enabled: bool = False
    atr_stop_enabled: bool = False
    atr_window: Optional[int] = None
    atr_multiplier: Optional[float] = None
    time_stop_enabled: bool = False
    time_stop_bars: Optional[int] = None
    
    # Data split info
    data_split: str = "train"
    walk_forward_window: Optional[int] = None
    walk_forward_step: Optional[int] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)
    
    def save_json(self, filepath: str) -> None:
        """Save as JSON file."""
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2, default=str)


@dataclass
class MetricsArtifact:
    """Metrics artifact containing portfolio and per-ticker performance metrics."""
    # Portfolio metrics
    total_return: float
    annualized_return: float
    volatility: float
    sharpe_ratio: float
    max_drawdown: float
    total_trades: int
    win_rate: float
    profit_factor: float
    cagr: float
    
    # Per-ticker metrics
    per_ticker_metrics: Dict[str, Dict[str, float]]
    
    # Regime filter metrics
    regime_hit_rate: Optional[float] = None
    regime_trading_days: Optional[int] = None
    regime_total_days: Optional[int] = None
    
    # Volatility targeting metrics
    realized_vol_pre: Optional[float] = None
    realized_vol_post: Optional[float] = None
    avg_scaling: Optional[float] = None
    vol_reduction: Optional[float] = None
    
    # Position management metrics
    total_turnover: Optional[float] = None
    num_rebalances: Optional[int] = None
    num_cap_breaches: Optional[int] = None
    
    # Basic exits metrics
    total_stops: Optional[int] = None
    atr_stops: Optional[int] = None
    time_stops: Optional[int] = None
    atr_stop_rate: Optional[float] = None
    time_stop_rate: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)
    
    def save_json(self, filepath: str) -> None:
        """Save as JSON file."""
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2, default=str)


@dataclass
class EquityArtifact:
    """Equity curve artifact containing portfolio value and returns."""
    equity_curve: pd.Series
    daily_returns: pd.Series
    benchmark_returns: Optional[pd.Series] = None
    
    def save_csv(self, filepath: str) -> None:
        """Save as CSV file."""
        df = pd.DataFrame({
            'equity_curve': self.equity_curve,
            'daily_returns': self.daily_returns
        })
        if self.benchmark_returns is not None:
            df['benchmark_returns'] = self.benchmark_returns
        df.to_csv(filepath)


@dataclass
class TradesArtifact:
    """Trades artifact containing detailed trade log."""
    trades: List[Dict[str, Any]]
    
    def save_csv(self, filepath: str) -> None:
        """Save as CSV file."""
        if not self.trades:
            # Create empty CSV with expected columns
            empty_df = pd.DataFrame(columns=[
                'symbol', 'entry_date', 'exit_date', 'shares', 'entry_price', 
                'exit_price', 'pnl', 'return', 'exit_reason'
            ])
            empty_df.to_csv(filepath, index=False)
        else:
            df = pd.DataFrame(self.trades)
            df.to_csv(filepath, index=False)


@dataclass
class MCPTArtifact:
    """MCPT artifact containing significance test results."""
    method: str
    permutations: int
    block_size: Optional[int]
    significance_level: float
    results: List[Dict[str, Any]]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            'method': self.method,
            'permutations': self.permutations,
            'block_size': self.block_size,
            'significance_level': self.significance_level,
            'results': self.results,
            'timestamp': datetime.now().isoformat()
        }
    
    def save_json(self, filepath: str) -> None:
        """Save as JSON file."""
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2, default=str)


@dataclass
class BootstrapArtifact:
    """Bootstrap artifact containing confidence interval results."""
    method: str
    samples: int
    confidence_level: float
    results: List[Dict[str, Any]]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            'method': self.method,
            'samples': self.samples,
            'confidence_level': self.confidence_level,
            'results': self.results,
            'timestamp': datetime.now().isoformat()
        }
    
    def save_json(self, filepath: str) -> None:
        """Save as JSON file."""
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2, default=str)


@dataclass
class WalkForwardArtifact:
    """Walk-forward analysis artifact containing rolling metrics."""
    window_length: int
    step_size: int
    min_trades: int
    results: pd.DataFrame
    
    def save_parquet(self, filepath: str) -> None:
        """Save as Parquet file."""
        self.results.to_parquet(filepath, index=True)


@dataclass
class WeightsArtifact:
    """Weights artifact containing allocation weights per rebalance."""
    weights_history: pd.DataFrame
    
    def save_csv(self, filepath: str) -> None:
        """Save as CSV file."""
        self.weights_history.to_csv(filepath)


@dataclass
class SummaryArtifact:
    """Summary artifact containing human-readable run summary."""
    title: str
    strategy: str
    period: str
    performance_summary: str
    key_insights: List[str]
    recommendations: List[str]
    technical_details: str
    
    def save_markdown(self, filepath: str) -> None:
        """Save as Markdown file."""
        content = f"""# {self.title}

## Strategy: {self.strategy}
**Period:** {self.period}

## Performance Summary
{self.performance_summary}

## Key Insights
{chr(10).join(f"- {insight}" for insight in self.key_insights)}

## Recommendations
{chr(10).join(f"- {rec}" for rec in self.recommendations)}

## Technical Details
{self.technical_details}

---
*Generated by Neural Quant on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*
"""
        with open(filepath, 'w') as f:
            f.write(content)
