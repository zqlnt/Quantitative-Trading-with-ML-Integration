"""
Walk-Forward Analysis Module

This module implements walk-forward analysis with statistical significance testing
on rolling windows to assess strategy performance over time.

Author: Neural Quant Team
Date: 2024
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
import logging
from datetime import datetime, timedelta
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from ..analysis.mcpt import MonteCarloPermutationTester, MCPTConfig, MCPTResult
from ..strategies.base.strategy_base import StrategyBase

logger = logging.getLogger(__name__)


@dataclass
class WalkForwardWindow:
    """Container for walk-forward window results."""
    start_date: str
    end_date: str
    window_length: int
    metrics: Dict[str, float]
    mcpt_results: Dict[str, MCPTResult]
    trades: List[Dict[str, Any]]
    equity_curve: pd.DataFrame


@dataclass
class WalkForwardConfig:
    """Configuration for walk-forward analysis."""
    window_length: int = 252  # 1 year of trading days
    step_size: int = 21  # 1 month step
    min_trades: int = 10  # Minimum trades required for analysis
    mcpt_config: MCPTConfig = None
    significance_level: float = 0.05


class WalkForwardAnalyzer:
    """
    Walk-Forward Analysis with Statistical Significance Testing.
    
    This class implements walk-forward analysis by:
    1. Creating rolling windows of specified length
    2. Running backtests on each window
    3. Computing MCPT significance tests on each window
    4. Creating visualizations of rolling performance and significance
    """
    
    def __init__(self, config: WalkForwardConfig = None):
        """Initialize the walk-forward analyzer."""
        self.config = config or WalkForwardConfig()
        self.windows: List[WalkForwardWindow] = []
        self.rolling_metrics: pd.DataFrame = None
        self.rolling_pvalues: pd.DataFrame = None
        
        # Set up MCPT config
        if self.config.mcpt_config is None:
            self.config.mcpt_config = MCPTConfig(n_permutations=500)  # Smaller for speed
    
    def analyze_strategy(self, 
                        data: pd.DataFrame,
                        strategy: StrategyBase,
                        backtester,  # Any backtester instance
                        start_date: Optional[str] = None,
                        end_date: Optional[str] = None) -> Dict[str, Any]:
        """
        Run walk-forward analysis on a strategy.
        
        Args:
            data: Price data with OHLCV columns
            strategy: Strategy instance
            backtester: Backtester instance
            start_date: Start date for analysis
            end_date: End date for analysis
            
        Returns:
            Dictionary containing walk-forward results
        """
        logger.info(f"Starting walk-forward analysis for {strategy.__class__.__name__}")
        
        # Prepare data
        if start_date:
            data = data[data.index >= start_date]
        if end_date:
            data = data[data.index <= end_date]
        
        if data.empty:
            logger.warning("No data available for walk-forward analysis")
            return {'error': 'No data available'}
        
        # Create rolling windows
        windows = self._create_rolling_windows(data)
        logger.info(f"Created {len(windows)} rolling windows")
        
        # Run analysis on each window
        self.windows = []
        for i, (window_start, window_end) in enumerate(windows):
            logger.info(f"Processing window {i+1}/{len(windows)}: {window_start} to {window_end}")
            
            try:
                window_data = data[(data.index >= window_start) & (data.index <= window_end)]
                if len(window_data) < self.config.window_length // 2:  # Skip if too short
                    continue
                
                # Run backtest on window
                window_results = backtester.run_backtest(window_data, strategy, str(window_start), str(window_end))
                
                if not window_results or 'trades' not in window_results or not window_results['trades']:
                    logger.warning(f"No trades in window {window_start} to {window_end}")
                    continue
                
                # Run MCPT on window
                mcpt_results = self._run_window_mcpt(window_results, strategy)
                
                # Create window result
                window_result = WalkForwardWindow(
                    start_date=str(window_start),
                    end_date=str(window_end),
                    window_length=len(window_data),
                    metrics=self._extract_window_metrics(window_results),
                    mcpt_results=mcpt_results,
                    trades=window_results.get('trades', []),
                    equity_curve=window_results.get('equity_curve', pd.DataFrame())
                )
                
                self.windows.append(window_result)
                
            except Exception as e:
                logger.warning(f"Failed to process window {window_start} to {window_end}: {e}")
                continue
        
        if not self.windows:
            logger.warning("No valid windows processed")
            return {'error': 'No valid windows processed'}
        
        # Create rolling analysis
        self._create_rolling_analysis()
        
        # Generate visualizations
        plots = self._create_visualizations()
        
        logger.info(f"Walk-forward analysis completed with {len(self.windows)} windows")
        
        return {
            'windows': self.windows,
            'rolling_metrics': self.rolling_metrics,
            'rolling_pvalues': self.rolling_pvalues,
            'plots': plots,
            'config': {
                'window_length': self.config.window_length,
                'step_size': self.config.step_size,
                'min_trades': self.config.min_trades,
                'significance_level': self.config.significance_level
            }
        }
    
    def _create_rolling_windows(self, data: pd.DataFrame) -> List[Tuple[datetime, datetime]]:
        """Create rolling windows for analysis."""
        windows = []
        
        start_date = data.index[0]
        end_date = data.index[-1]
        
        current_start = start_date
        while current_start + timedelta(days=self.config.window_length) <= end_date:
            current_end = current_start + timedelta(days=self.config.window_length)
            windows.append((current_start, current_end))
            current_start += timedelta(days=self.config.step_size)
        
        return windows
    
    def _run_window_mcpt(self, window_results: Dict[str, Any], strategy: StrategyBase) -> Dict[str, MCPTResult]:
        """Run MCPT on a single window."""
        try:
            # Extract returns from equity curve
            equity_curve = window_results.get('equity_curve')
            if equity_curve is None or equity_curve.empty:
                return {}
            
            if isinstance(equity_curve, pd.DataFrame):
                returns = equity_curve['returns'].dropna()
            else:
                returns = equity_curve.pct_change().dropna()
            
            if len(returns) < 10:  # Need minimum data for MCPT
                return {}
            
            # Run MCPT
            tester = MonteCarloPermutationTester(self.config.mcpt_config)
            mcpt_results = tester.test_returns_significance(
                returns, 
                strategy_name=f"{strategy.__class__.__name__}_window"
            )
            
            return mcpt_results
            
        except Exception as e:
            logger.warning(f"MCPT failed for window: {e}")
            return {}
    
    def _extract_window_metrics(self, window_results: Dict[str, Any]) -> Dict[str, float]:
        """Extract key metrics from window results."""
        metrics = {}
        
        # Basic metrics
        metrics['total_return'] = window_results.get('total_return', 0)
        metrics['annualized_return'] = window_results.get('annualized_return', 0)
        metrics['cagr'] = window_results.get('cagr', 0)
        metrics['volatility'] = window_results.get('volatility', 0)
        metrics['sharpe_ratio'] = window_results.get('sharpe_ratio', 0)
        metrics['max_drawdown'] = window_results.get('max_drawdown', 0)
        metrics['win_rate'] = window_results.get('win_rate', 0)
        metrics['profit_factor'] = window_results.get('profit_factor', 0)
        metrics['total_trades'] = window_results.get('total_trades', 0)
        
        return metrics
    
    def _create_rolling_analysis(self):
        """Create rolling analysis DataFrames."""
        if not self.windows:
            return
        
        # Create rolling metrics DataFrame
        metrics_data = []
        pvalues_data = []
        
        for window in self.windows:
            # Add metrics
            row = {
                'start_date': window.start_date,
                'end_date': window.end_date,
                'window_length': window.window_length,
                **window.metrics
            }
            metrics_data.append(row)
            
            # Add p-values
            pvalue_row = {
                'start_date': window.start_date,
                'end_date': window.end_date,
                'window_length': window.window_length
            }
            
            for metric_name, mcpt_result in window.mcpt_results.items():
                pvalue_row[f'{metric_name}_pvalue'] = mcpt_result.p_value
                pvalue_row[f'{metric_name}_significant'] = mcpt_result.is_significant
            
            pvalues_data.append(pvalue_row)
        
        self.rolling_metrics = pd.DataFrame(metrics_data)
        self.rolling_pvalues = pd.DataFrame(pvalues_data)
        
        # Convert dates
        self.rolling_metrics['start_date'] = pd.to_datetime(self.rolling_metrics['start_date'])
        self.rolling_metrics['end_date'] = pd.to_datetime(self.rolling_metrics['end_date'])
        self.rolling_pvalues['start_date'] = pd.to_datetime(self.rolling_pvalues['start_date'])
        self.rolling_pvalues['end_date'] = pd.to_datetime(self.rolling_pvalues['end_date'])
    
    def _create_visualizations(self) -> Dict[str, go.Figure]:
        """Create walk-forward visualizations."""
        plots = {}
        
        if self.rolling_metrics is None or self.rolling_pvalues is None:
            return plots
        
        # 1. Rolling Sharpe with P-values
        plots['rolling_sharpe'] = self._create_rolling_sharpe_plot()
        
        # 2. Significance Heatmap
        plots['significance_heatmap'] = self._create_significance_heatmap()
        
        # 3. Rolling P-values Strip Chart
        plots['pvalues_strip'] = self._create_pvalues_strip_chart()
        
        # 4. Performance vs Significance Scatter
        plots['performance_significance'] = self._create_performance_significance_plot()
        
        return plots
    
    def _create_rolling_sharpe_plot(self) -> go.Figure:
        """Create rolling Sharpe ratio with p-values plot."""
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=('Rolling Sharpe Ratio', 'Statistical Significance (P-values)'),
            vertical_spacing=0.1,
            row_heights=[0.7, 0.3]
        )
        
        # Rolling Sharpe
        fig.add_trace(
            go.Scatter(
                x=self.rolling_metrics['end_date'],
                y=self.rolling_metrics['sharpe_ratio'],
                mode='lines+markers',
                name='Sharpe Ratio',
                line=dict(color='blue', width=2),
                marker=dict(size=6)
            ),
            row=1, col=1
        )
        
        # Add significance threshold line
        fig.add_hline(
            y=0, 
            line_dash="dash", 
            line_color="gray",
            row=1, col=1
        )
        
        # P-values
        if 'sharpe_ratio_pvalue' in self.rolling_pvalues.columns:
            # Color points by significance
            colors = ['red' if p < self.config.significance_level else 'green' 
                     for p in self.rolling_pvalues['sharpe_ratio_pvalue']]
            
            fig.add_trace(
                go.Scatter(
                    x=self.rolling_pvalues['end_date'],
                    y=self.rolling_pvalues['sharpe_ratio_pvalue'],
                    mode='markers',
                    name='P-value',
                    marker=dict(
                        color=colors,
                        size=8,
                        line=dict(width=1, color='black')
                    ),
                    text=[f"P-value: {p:.3f}" for p in self.rolling_pvalues['sharpe_ratio_pvalue']],
                    hovertemplate='%{text}<br>Date: %{x}<extra></extra>'
                ),
                row=2, col=1
            )
            
            # Add significance threshold line
            fig.add_hline(
                y=self.config.significance_level, 
                line_dash="dash", 
                line_color="red",
                annotation_text=f"Significance Level ({self.config.significance_level})",
                row=2, col=1
            )
        
        fig.update_layout(
            title='Walk-Forward Analysis: Rolling Sharpe Ratio and Statistical Significance',
            height=600,
            showlegend=True
        )
        
        fig.update_xaxes(title_text="Date", row=2, col=1)
        fig.update_yaxes(title_text="Sharpe Ratio", row=1, col=1)
        fig.update_yaxes(title_text="P-value", row=2, col=1)
        
        return fig
    
    def _create_significance_heatmap(self) -> go.Figure:
        """Create significance heatmap across metrics and time."""
        if self.rolling_pvalues.empty:
            return go.Figure()
        
        # Prepare data for heatmap
        metrics = ['sharpe_ratio', 'total_return', 'cagr', 'max_drawdown']
        available_metrics = [m for m in metrics if f'{m}_significant' in self.rolling_pvalues.columns]
        
        if not available_metrics:
            return go.Figure()
        
        # Create significance matrix
        significance_data = []
        for _, row in self.rolling_pvalues.iterrows():
            significance_row = []
            for metric in available_metrics:
                significance_row.append(1 if row[f'{metric}_significant'] else 0)
            significance_data.append(significance_row)
        
        # Create heatmap
        fig = go.Figure(data=go.Heatmap(
            z=significance_data,
            x=[m.replace('_', ' ').title() for m in available_metrics],
            y=[f"{row['start_date'].strftime('%Y-%m')} to {row['end_date'].strftime('%Y-%m')}" 
               for _, row in self.rolling_pvalues.iterrows()],
            colorscale=[[0, 'red'], [1, 'green']],
            showscale=True,
            colorbar=dict(
                title="Significant",
                tickvals=[0, 1],
                ticktext=['No', 'Yes']
            ),
            hovertemplate='Metric: %{x}<br>Period: %{y}<br>Significant: %{z}<extra></extra>'
        ))
        
        fig.update_layout(
            title='Statistical Significance Heatmap Across Time and Metrics',
            xaxis_title='Metrics',
            yaxis_title='Time Periods',
            height=max(400, len(self.rolling_pvalues) * 30)
        )
        
        return fig
    
    def _create_pvalues_strip_chart(self) -> go.Figure:
        """Create strip chart of p-values across metrics."""
        if self.rolling_pvalues.empty:
            return go.Figure()
        
        # Prepare data for strip chart
        metrics = ['sharpe_ratio', 'total_return', 'cagr', 'max_drawdown']
        available_metrics = [m for m in metrics if f'{m}_pvalue' in self.rolling_pvalues.columns]
        
        if not available_metrics:
            return go.Figure()
        
        fig = go.Figure()
        
        for metric in available_metrics:
            pvalues = self.rolling_pvalues[f'{metric}_pvalue'].dropna()
            
            # Color by significance
            colors = ['red' if p < self.config.significance_level else 'green' for p in pvalues]
            
            fig.add_trace(go.Scatter(
                x=[metric.replace('_', ' ').title()] * len(pvalues),
                y=pvalues,
                mode='markers',
                name=metric.replace('_', ' ').title(),
                marker=dict(
                    color=colors,
                    size=8,
                    line=dict(width=1, color='black')
                ),
                text=[f"P-value: {p:.3f}" for p in pvalues],
                hovertemplate='%{text}<extra></extra>'
            ))
        
        # Add significance threshold line
        fig.add_hline(
            y=self.config.significance_level, 
            line_dash="dash", 
            line_color="red",
            annotation_text=f"Significance Level ({self.config.significance_level})"
        )
        
        fig.update_layout(
            title='P-values Distribution Across Metrics',
            xaxis_title='Metrics',
            yaxis_title='P-value',
            yaxis=dict(type='log'),
            height=500
        )
        
        return fig
    
    def _create_performance_significance_plot(self) -> go.Figure:
        """Create scatter plot of performance vs significance."""
        if self.rolling_metrics.empty or self.rolling_pvalues.empty:
            return go.Figure()
        
        # Merge metrics and p-values
        merged = pd.merge(
            self.rolling_metrics[['end_date', 'sharpe_ratio']],
            self.rolling_pvalues[['end_date', 'sharpe_ratio_pvalue']],
            on='end_date',
            how='inner'
        )
        
        if merged.empty:
            return go.Figure()
        
        # Color by significance
        colors = ['red' if p < self.config.significance_level else 'green' 
                 for p in merged['sharpe_ratio_pvalue']]
        
        fig = go.Figure(data=go.Scatter(
            x=merged['sharpe_ratio_pvalue'],
            y=merged['sharpe_ratio'],
            mode='markers',
            marker=dict(
                color=colors,
                size=10,
                line=dict(width=2, color='black')
            ),
            text=[f"Date: {d.strftime('%Y-%m')}<br>Sharpe: {s:.3f}<br>P-value: {p:.3f}" 
                  for d, s, p in zip(merged['end_date'], merged['sharpe_ratio'], merged['sharpe_ratio_pvalue'])],
            hovertemplate='%{text}<extra></extra>'
        ))
        
        # Add significance threshold line
        fig.add_vline(
            x=self.config.significance_level, 
            line_dash="dash", 
            line_color="red",
            annotation_text=f"Significance Level ({self.config.significance_level})"
        )
        
        fig.update_layout(
            title='Performance vs Statistical Significance',
            xaxis_title='P-value (log scale)',
            yaxis_title='Sharpe Ratio',
            xaxis=dict(type='log'),
            height=500
        )
        
        return fig
    
    def get_summary(self) -> pd.DataFrame:
        """Get summary of walk-forward analysis."""
        if not self.windows:
            return pd.DataFrame()
        
        summary_data = []
        for window in self.windows:
            row = {
                'Start Date': window.start_date,
                'End Date': window.end_date,
                'Window Length': window.window_length,
                'Total Trades': window.metrics.get('total_trades', 0),
                'Sharpe Ratio': window.metrics.get('sharpe_ratio', 0),
                'Total Return': window.metrics.get('total_return', 0),
                'Max Drawdown': window.metrics.get('max_drawdown', 0)
            }
            
            # Add significance info
            if 'sharpe_ratio' in window.mcpt_results:
                row['Sharpe P-value'] = window.mcpt_results['sharpe_ratio'].p_value
                row['Sharpe Significant'] = window.mcpt_results['sharpe_ratio'].is_significant
            
            summary_data.append(row)
        
        return pd.DataFrame(summary_data)


def run_walkforward_analysis(
    data: pd.DataFrame,
    strategy: StrategyBase,
    backtester,  # Any backtester instance
    config: WalkForwardConfig = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None
) -> Dict[str, Any]:
    """
    Convenience function to run walk-forward analysis.
    
    Args:
        data: Price data with OHLCV columns
        strategy: Strategy instance
        backtester: Backtester instance
        config: Walk-forward configuration
        start_date: Start date for analysis
        end_date: End date for analysis
        
    Returns:
        Dictionary containing walk-forward results
    """
    analyzer = WalkForwardAnalyzer(config)
    return analyzer.analyze_strategy(data, strategy, backtester, start_date, end_date)
