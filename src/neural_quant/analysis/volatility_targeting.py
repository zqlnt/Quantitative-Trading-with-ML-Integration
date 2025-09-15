"""
Volatility Targeting Module

This module implements portfolio volatility targeting by scaling exposures
to hit a target annual volatility.

Author: Neural Quant Team
Date: 2024
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, Tuple
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class VolatilityTargetingConfig:
    """Configuration for volatility targeting."""
    enabled: bool = False
    target_vol: float = 0.10  # 10% annualized
    lookback_window: int = 20  # days
    scale_cap: float = 2.0  # maximum scaling factor
    realized_vol_pre: float = 0.0  # realized vol before scaling
    realized_vol_post: float = 0.0  # realized vol after scaling
    avg_scaling: float = 1.0  # average scaling factor applied


class VolatilityTargeting:
    """
    Volatility targeting for portfolio strategies.
    
    Scales portfolio exposures to maintain target annual volatility.
    """
    
    def __init__(self, config: VolatilityTargetingConfig):
        self.config = config
        self.scaling_factors: Optional[pd.Series] = None
        self.realized_vol_series: Optional[pd.Series] = None
        self.scaled_returns: Optional[pd.Series] = None
    
    def calculate_realized_volatility(self, returns: pd.Series, window: int = None) -> pd.Series:
        """
        Calculate rolling realized volatility.
        
        Args:
            returns: Daily returns series
            window: Rolling window size (defaults to config lookback_window)
            
        Returns:
            Rolling realized volatility series (annualized)
        """
        if window is None:
            window = self.config.lookback_window
        
        # Calculate rolling standard deviation and annualize
        rolling_std = returns.rolling(window=window).std()
        realized_vol = rolling_std * np.sqrt(252)  # Annualize
        
        return realized_vol
    
    def calculate_scaling_factors(self, returns: pd.Series) -> pd.Series:
        """
        Calculate scaling factors to target volatility.
        
        Args:
            returns: Daily returns series
            
        Returns:
            Scaling factors series
        """
        # Calculate realized volatility
        realized_vol = self.calculate_realized_volatility(returns)
        
        # Calculate scaling factors
        scaling_factors = self.config.target_vol / realized_vol
        
        # Apply scale cap
        scaling_factors = np.clip(scaling_factors, 0, self.config.scale_cap)
        
        # Fill NaN values with 1.0 (no scaling)
        scaling_factors = scaling_factors.fillna(1.0)
        
        self.scaling_factors = scaling_factors
        self.realized_vol_series = realized_vol
        
        return scaling_factors
    
    def scale_returns(self, returns: pd.Series) -> pd.Series:
        """
        Scale returns to target volatility.
        
        Args:
            returns: Original returns series
            
        Returns:
            Scaled returns series
        """
        if not self.config.enabled:
            return returns
        
        # Calculate scaling factors
        scaling_factors = self.calculate_scaling_factors(returns)
        
        # Scale returns
        scaled_returns = returns * scaling_factors
        
        # Store for analysis
        self.scaled_returns = scaled_returns
        
        # Calculate metrics
        self._calculate_metrics(returns, scaled_returns)
        
        return scaled_returns
    
    def scale_equity_curve(self, equity_curve: pd.Series) -> pd.Series:
        """
        Scale equity curve to target volatility.
        
        Args:
            equity_curve: Original equity curve
            
        Returns:
            Scaled equity curve
        """
        if not self.config.enabled:
            return equity_curve
        
        # Calculate returns from equity curve
        returns = equity_curve.pct_change().dropna()
        
        # Scale returns
        scaled_returns = self.scale_returns(returns)
        
        # Reconstruct equity curve from scaled returns
        scaled_equity = (1 + scaled_returns).cumprod() * equity_curve.iloc[0]
        
        return scaled_equity
    
    def _calculate_metrics(self, original_returns: pd.Series, scaled_returns: pd.Series):
        """Calculate volatility targeting metrics."""
        # Calculate realized volatility before and after scaling
        self.config.realized_vol_pre = original_returns.std() * np.sqrt(252)
        self.config.realized_vol_post = scaled_returns.std() * np.sqrt(252)
        
        # Calculate average scaling factor
        if self.scaling_factors is not None:
            self.config.avg_scaling = self.scaling_factors.mean()
        else:
            self.config.avg_scaling = 1.0
    
    def get_scaling_summary(self) -> Dict[str, Any]:
        """
        Get summary of volatility targeting performance.
        
        Returns:
            Dictionary with volatility targeting metrics
        """
        if not self.config.enabled:
            return {
                'volatility_targeting_enabled': False,
                'target_vol': 0.0,
                'realized_vol_pre': 0.0,
                'realized_vol_post': 0.0,
                'avg_scaling': 1.0
            }
        
        return {
            'volatility_targeting_enabled': True,
            'target_vol': self.config.target_vol,
            'lookback_window': self.config.lookback_window,
            'scale_cap': self.config.scale_cap,
            'realized_vol_pre': self.config.realized_vol_pre,
            'realized_vol_post': self.config.realized_vol_post,
            'avg_scaling': self.config.avg_scaling,
            'vol_reduction': (self.config.realized_vol_pre - self.config.realized_vol_post) / self.config.realized_vol_pre if self.config.realized_vol_pre > 0 else 0.0
        }
    
    def log_to_mlflow(self):
        """Log volatility targeting configuration and results to MLflow."""
        try:
            import mlflow
            
            if mlflow.active_run() is None:
                logger.warning("No active MLflow run, skipping volatility targeting logging")
                return
            
            # Log configuration
            mlflow.log_param("volatility_targeting_enabled", self.config.enabled)
            if self.config.enabled:
                mlflow.log_param("vol_target", self.config.target_vol)
                mlflow.log_param("vol_lookback_window", self.config.lookback_window)
                mlflow.log_param("vol_scale_cap", self.config.scale_cap)
                
                # Log results
                summary = self.get_scaling_summary()
                mlflow.log_metric("realized_vol_pre", summary['realized_vol_pre'])
                mlflow.log_metric("realized_vol_post", summary['realized_vol_post'])
                mlflow.log_metric("avg_scaling", summary['avg_scaling'])
                mlflow.log_metric("vol_reduction", summary['vol_reduction'])
                
        except Exception as e:
            logger.error(f"Error logging volatility targeting to MLflow: {e}")


def create_volatility_targeting_config(enabled: bool = False,
                                     target_vol: float = 0.10,
                                     lookback_window: int = 20,
                                     scale_cap: float = 2.0) -> VolatilityTargetingConfig:
    """
    Create a volatility targeting configuration.
    
    Args:
        enabled: Whether volatility targeting is enabled
        target_vol: Target annual volatility (e.g., 0.10 for 10%)
        lookback_window: Lookback window for realized vol calculation
        scale_cap: Maximum scaling factor
        
    Returns:
        VolatilityTargetingConfig instance
    """
    return VolatilityTargetingConfig(
        enabled=enabled,
        target_vol=target_vol,
        lookback_window=lookback_window,
        scale_cap=scale_cap
    )
