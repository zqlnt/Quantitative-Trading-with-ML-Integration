"""
Artifact Manager

This module manages the creation and saving of structured artifacts
for MLflow runs, ensuring all required artifacts are saved consistently.

Author: Neural Quant Team
Date: 2024
"""

import os
import json
import pandas as pd
from typing import Dict, Any, List, Optional, Union
from datetime import datetime
import logging

from .artifact_types import (
    ArtifactType,
    ParamsArtifact,
    MetricsArtifact,
    EquityArtifact,
    TradesArtifact,
    MCPTArtifact,
    BootstrapArtifact,
    WalkForwardArtifact,
    WeightsArtifact,
    SummaryArtifact
)

logger = logging.getLogger(__name__)


class ArtifactManager:
    """
    Manages the creation and saving of structured artifacts for MLflow runs.
    
    This class ensures that all required artifacts are created and saved
    consistently for each backtest run, providing a complete record of
    the strategy performance and analysis.
    """
    
    def __init__(self, run_id: str, output_dir: str = "artifacts"):
        """
        Initialize the artifact manager.
        
        Args:
            run_id: Unique identifier for the MLflow run
            output_dir: Directory to save artifacts (relative to MLflow run)
        """
        self.run_id = run_id
        self.output_dir = output_dir
        self.artifacts = {}
        
        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)
    
    def create_params_artifact(self, 
                             strategy: str,
                             strategy_params: Dict[str, Any],
                             tickers: List[str],
                             start_date: str,
                             end_date: str,
                             initial_capital: float,
                             commission: float,
                             slippage: float,
                             regime_filter_config: Optional[Dict[str, Any]] = None,
                             vol_targeting_config: Optional[Dict[str, Any]] = None,
                             allocation_config: Optional[Dict[str, Any]] = None,
                             position_config: Optional[Dict[str, Any]] = None,
                             basic_exits_config: Optional[Dict[str, Any]] = None,
                             walk_forward_config: Optional[Dict[str, Any]] = None) -> ParamsArtifact:
        """
        Create parameters artifact.
        
        Args:
            strategy: Strategy name
            strategy_params: Strategy-specific parameters
            tickers: List of tickers
            start_date: Start date
            end_date: End date
            initial_capital: Initial capital
            commission: Commission rate
            slippage: Slippage rate
            regime_filter_config: Regime filter configuration
            vol_targeting_config: Volatility targeting configuration
            allocation_config: Allocation method configuration
            position_config: Position management configuration
            basic_exits_config: Basic exits configuration
            walk_forward_config: Walk-forward analysis configuration
            
        Returns:
            ParamsArtifact instance
        """
        artifact = ParamsArtifact(
            strategy=strategy,
            strategy_params=strategy_params,
            tickers=tickers,
            start_date=start_date,
            end_date=end_date,
            initial_capital=initial_capital,
            commission=commission,
            slippage=slippage
        )
        
        # Add C1 - Regime Filter
        if regime_filter_config:
            artifact.regime_filter_enabled = regime_filter_config.get('enabled', False)
            artifact.regime_proxy_symbol = regime_filter_config.get('proxy_symbol')
            artifact.regime_rule = regime_filter_config.get('regime_rule')
            artifact.regime_sma_window = regime_filter_config.get('sma_window')
        
        # Add C2 - Volatility Targeting
        if vol_targeting_config:
            artifact.volatility_targeting_enabled = vol_targeting_config.get('enabled', False)
            artifact.target_vol = vol_targeting_config.get('target_vol')
            artifact.vol_lookback_window = vol_targeting_config.get('lookback_window')
            artifact.vol_scale_cap = vol_targeting_config.get('scale_cap')
        
        # Add C3 - Allocation Method
        if allocation_config:
            artifact.allocation_method = allocation_config.get('method', 'equal_weight')
            artifact.vol_lookback = allocation_config.get('vol_lookback')
        
        # Add C4 - Position Management
        if position_config:
            artifact.max_position_pct = position_config.get('max_position_pct')
            artifact.rebalance_frequency = position_config.get('rebalance_frequency')
            artifact.min_rebalance_interval = position_config.get('min_rebalance_interval')
            artifact.turnover_threshold = position_config.get('turnover_threshold')
        
        # Add C5 - Basic Exits
        if basic_exits_config:
            artifact.basic_exits_enabled = basic_exits_config.get('enabled', False)
            artifact.atr_stop_enabled = basic_exits_config.get('enable_atr_stop', False)
            artifact.atr_window = basic_exits_config.get('atr_window')
            artifact.atr_multiplier = basic_exits_config.get('atr_multiplier')
            artifact.time_stop_enabled = basic_exits_config.get('enable_time_stop', False)
            artifact.time_stop_bars = basic_exits_config.get('time_stop_bars')
        
        # Add walk-forward config
        if walk_forward_config:
            artifact.walk_forward_window = walk_forward_config.get('window_length')
            artifact.walk_forward_step = walk_forward_config.get('step_size')
        
        self.artifacts[ArtifactType.PARAMS] = artifact
        return artifact
    
    def create_metrics_artifact(self, 
                              portfolio_metrics: Dict[str, Any],
                              per_ticker_metrics: Optional[Dict[str, Dict[str, float]]] = None,
                              regime_metrics: Optional[Dict[str, Any]] = None,
                              vol_targeting_metrics: Optional[Dict[str, Any]] = None,
                              position_metrics: Optional[Dict[str, Any]] = None,
                              basic_exits_metrics: Optional[Dict[str, Any]] = None) -> MetricsArtifact:
        """
        Create metrics artifact.
        
        Args:
            portfolio_metrics: Portfolio-level performance metrics
            per_ticker_metrics: Per-ticker performance metrics
            regime_metrics: Regime filter metrics
            vol_targeting_metrics: Volatility targeting metrics
            position_metrics: Position management metrics
            basic_exits_metrics: Basic exits metrics
            
        Returns:
            MetricsArtifact instance
        """
        artifact = MetricsArtifact(
            total_return=portfolio_metrics.get('total_return', 0.0),
            annualized_return=portfolio_metrics.get('annualized_return', 0.0),
            volatility=portfolio_metrics.get('volatility', 0.0),
            sharpe_ratio=portfolio_metrics.get('sharpe_ratio', 0.0),
            max_drawdown=portfolio_metrics.get('max_drawdown', 0.0),
            total_trades=portfolio_metrics.get('total_trades', 0),
            win_rate=portfolio_metrics.get('win_rate', 0.0),
            profit_factor=portfolio_metrics.get('profit_factor', 0.0),
            cagr=portfolio_metrics.get('cagr', 0.0),
            per_ticker_metrics=per_ticker_metrics or {}
        )
        
        # Add regime filter metrics
        if regime_metrics:
            artifact.regime_hit_rate = regime_metrics.get('regime_hit_rate')
            artifact.regime_trading_days = regime_metrics.get('trading_days')
            artifact.regime_total_days = regime_metrics.get('total_days')
        
        # Add volatility targeting metrics
        if vol_targeting_metrics:
            artifact.realized_vol_pre = vol_targeting_metrics.get('realized_vol_pre')
            artifact.realized_vol_post = vol_targeting_metrics.get('realized_vol_post')
            artifact.avg_scaling = vol_targeting_metrics.get('avg_scaling')
            artifact.vol_reduction = vol_targeting_metrics.get('vol_reduction')
        
        # Add position management metrics
        if position_metrics:
            artifact.total_turnover = position_metrics.get('total_turnover')
            artifact.num_rebalances = position_metrics.get('num_rebalances')
            artifact.num_cap_breaches = position_metrics.get('num_cap_breaches')
        
        # Add basic exits metrics
        if basic_exits_metrics:
            artifact.total_stops = basic_exits_metrics.get('total_stops')
            artifact.atr_stops = basic_exits_metrics.get('atr_stops')
            artifact.time_stops = basic_exits_metrics.get('time_stops')
            artifact.atr_stop_rate = basic_exits_metrics.get('atr_stop_rate')
            artifact.time_stop_rate = basic_exits_metrics.get('time_stop_rate')
        
        self.artifacts[ArtifactType.METRICS] = artifact
        return artifact
    
    def create_equity_artifact(self, 
                             equity_curve: pd.Series,
                             daily_returns: pd.Series,
                             benchmark_returns: Optional[pd.Series] = None) -> EquityArtifact:
        """
        Create equity curve artifact.
        
        Args:
            equity_curve: Portfolio equity curve
            daily_returns: Daily returns
            benchmark_returns: Optional benchmark returns
            
        Returns:
            EquityArtifact instance
        """
        artifact = EquityArtifact(
            equity_curve=equity_curve,
            daily_returns=daily_returns,
            benchmark_returns=benchmark_returns
        )
        
        self.artifacts[ArtifactType.EQUITY] = artifact
        return artifact
    
    def create_trades_artifact(self, trades: List[Dict[str, Any]]) -> TradesArtifact:
        """
        Create trades artifact.
        
        Args:
            trades: List of trade records
            
        Returns:
            TradesArtifact instance
        """
        artifact = TradesArtifact(trades=trades)
        self.artifacts[ArtifactType.TRADES] = artifact
        return artifact
    
    def create_mcpt_artifact(self, 
                           method: str,
                           permutations: int,
                           block_size: Optional[int],
                           significance_level: float,
                           results: List[Dict[str, Any]]) -> MCPTArtifact:
        """
        Create MCPT artifact.
        
        Args:
            method: MCPT method used
            permutations: Number of permutations
            block_size: Block size for permutations
            significance_level: Significance level
            results: MCPT results
            
        Returns:
            MCPTArtifact instance
        """
        artifact = MCPTArtifact(
            method=method,
            permutations=permutations,
            block_size=block_size,
            significance_level=significance_level,
            results=results
        )
        
        self.artifacts[ArtifactType.MCPT] = artifact
        return artifact
    
    def create_bootstrap_artifact(self, 
                                method: str,
                                samples: int,
                                confidence_level: float,
                                results: List[Dict[str, Any]]) -> BootstrapArtifact:
        """
        Create bootstrap artifact.
        
        Args:
            method: Bootstrap method used
            samples: Number of bootstrap samples
            confidence_level: Confidence level
            results: Bootstrap results
            
        Returns:
            BootstrapArtifact instance
        """
        artifact = BootstrapArtifact(
            method=method,
            samples=samples,
            confidence_level=confidence_level,
            results=results
        )
        
        self.artifacts[ArtifactType.BOOTSTRAP] = artifact
        return artifact
    
    def create_walkforward_artifact(self, 
                                  window_length: int,
                                  step_size: int,
                                  min_trades: int,
                                  results: pd.DataFrame) -> WalkForwardArtifact:
        """
        Create walk-forward analysis artifact.
        
        Args:
            window_length: Window length
            step_size: Step size
            min_trades: Minimum trades required
            results: Walk-forward results DataFrame
            
        Returns:
            WalkForwardArtifact instance
        """
        artifact = WalkForwardArtifact(
            window_length=window_length,
            step_size=step_size,
            min_trades=min_trades,
            results=results
        )
        
        self.artifacts[ArtifactType.WALKFORWARD] = artifact
        return artifact
    
    def create_weights_artifact(self, weights_history: pd.DataFrame) -> WeightsArtifact:
        """
        Create weights artifact.
        
        Args:
            weights_history: Weights history DataFrame
            
        Returns:
            WeightsArtifact instance
        """
        artifact = WeightsArtifact(weights_history=weights_history)
        self.artifacts[ArtifactType.WEIGHTS] = artifact
        return artifact
    
    def create_summary_artifact(self, 
                              title: str,
                              strategy: str,
                              period: str,
                              performance_summary: str,
                              key_insights: List[str],
                              recommendations: List[str],
                              technical_details: str) -> SummaryArtifact:
        """
        Create summary artifact.
        
        Args:
            title: Summary title
            strategy: Strategy name
            period: Analysis period
            performance_summary: Performance summary text
            key_insights: List of key insights
            recommendations: List of recommendations
            technical_details: Technical details text
            
        Returns:
            SummaryArtifact instance
        """
        artifact = SummaryArtifact(
            title=title,
            strategy=strategy,
            period=period,
            performance_summary=performance_summary,
            key_insights=key_insights,
            recommendations=recommendations,
            technical_details=technical_details
        )
        
        self.artifacts[ArtifactType.SUMMARY] = artifact
        return artifact
    
    def save_all_artifacts(self) -> Dict[str, str]:
        """
        Save all artifacts to files.
        
        Returns:
            Dictionary mapping artifact types to file paths
        """
        saved_files = {}
        
        for artifact_type, artifact in self.artifacts.items():
            try:
                filepath = os.path.join(self.output_dir, artifact_type.value)
                
                if artifact_type == ArtifactType.PARAMS:
                    artifact.save_json(filepath)
                elif artifact_type == ArtifactType.METRICS:
                    artifact.save_json(filepath)
                elif artifact_type == ArtifactType.EQUITY:
                    artifact.save_csv(filepath)
                elif artifact_type == ArtifactType.TRADES:
                    artifact.save_csv(filepath)
                elif artifact_type == ArtifactType.MCPT:
                    artifact.save_json(filepath)
                elif artifact_type == ArtifactType.BOOTSTRAP:
                    artifact.save_json(filepath)
                elif artifact_type == ArtifactType.WALKFORWARD:
                    artifact.save_parquet(filepath)
                elif artifact_type == ArtifactType.WEIGHTS:
                    artifact.save_csv(filepath)
                elif artifact_type == ArtifactType.SUMMARY:
                    artifact.save_markdown(filepath)
                
                saved_files[artifact_type.value] = filepath
                logger.info(f"Saved {artifact_type.value} to {filepath}")
                
            except Exception as e:
                logger.error(f"Error saving {artifact_type.value}: {e}")
        
        return saved_files
    
    def get_artifact(self, artifact_type: ArtifactType) -> Optional[Any]:
        """
        Get an artifact by type.
        
        Args:
            artifact_type: Type of artifact to retrieve
            
        Returns:
            Artifact instance or None if not found
        """
        return self.artifacts.get(artifact_type)
    
    def has_artifact(self, artifact_type: ArtifactType) -> bool:
        """
        Check if an artifact exists.
        
        Args:
            artifact_type: Type of artifact to check
            
        Returns:
            True if artifact exists
        """
        return artifact_type in self.artifacts
    
    def list_artifacts(self) -> List[str]:
        """
        List all available artifacts.
        
        Returns:
            List of artifact type names
        """
        return [artifact_type.value for artifact_type in self.artifacts.keys()]
