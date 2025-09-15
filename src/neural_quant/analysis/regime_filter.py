"""
Regime Filter Module

This module implements market regime filtering for trading strategies.
It allows strategies to only trade when a market proxy is in a specified regime.

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
class RegimeFilterConfig:
    """Configuration for regime filtering."""
    enabled: bool = False
    proxy_symbol: str = "SPY"
    regime_rule: str = "Bull only (proxy > SMA(200))"
    sma_window: int = 200
    regime_hit_rate: float = 0.0


class RegimeFilter:
    """
    Regime filter for trading strategies.
    
    Filters trading signals based on market proxy regime conditions.
    """
    
    def __init__(self, config: RegimeFilterConfig):
        self.config = config
        self.proxy_data: Optional[pd.DataFrame] = None
        self.regime_signals: Optional[pd.Series] = None
        self.regime_hit_rate: float = 0.0
    
    def load_proxy_data(self, start_date: str, end_date: str) -> bool:
        """
        Load proxy data for regime analysis.
        
        Args:
            start_date: Start date for data
            end_date: End date for data
            
        Returns:
            True if data loaded successfully, False otherwise
        """
        try:
            from ..data.yf_loader import load_yf_data
            
            # Load proxy data
            self.proxy_data = load_yf_data([self.config.proxy_symbol], start_date, end_date)
            
            if self.proxy_data.empty:
                logger.warning(f"No data loaded for proxy symbol {self.config.proxy_symbol}")
                return False
            
            # Calculate SMA
            close_col = f"{self.config.proxy_symbol}_close"
            if close_col not in self.proxy_data.columns:
                logger.error(f"Close column {close_col} not found in proxy data")
                return False
            
            sma = self.proxy_data[close_col].rolling(window=self.config.sma_window).mean()
            
            # Generate regime signals based on rule
            if "Bull only" in self.config.regime_rule:
                self.regime_signals = (self.proxy_data[close_col] > sma).astype(int)
            elif "Bear only" in self.config.regime_rule:
                self.regime_signals = (self.proxy_data[close_col] < sma).astype(int)
            else:  # Both (no filter)
                self.regime_signals = pd.Series(1, index=self.proxy_data.index)
            
            # Calculate regime hit rate
            self.regime_hit_rate = self.regime_signals.mean()
            self.config.regime_hit_rate = self.regime_hit_rate
            
            logger.info(f"Regime filter loaded: {self.config.proxy_symbol}, hit rate: {self.regime_hit_rate:.2%}")
            return True
            
        except Exception as e:
            logger.error(f"Error loading proxy data: {e}")
            return False
    
    def should_trade(self, date: pd.Timestamp) -> bool:
        """
        Check if trading should be allowed on a given date.
        
        Args:
            date: Trading date
            
        Returns:
            True if trading is allowed, False otherwise
        """
        if not self.config.enabled or self.regime_signals is None:
            return True
        
        # Find the closest date in regime signals
        try:
            # Convert to date for comparison
            date_only = date.date()
            
            # Find matching date in regime signals
            matching_dates = self.regime_signals.index.date == date_only
            if matching_dates.any():
                return bool(self.regime_signals[matching_dates].iloc[0])
            else:
                # If exact date not found, use the last available signal
                return bool(self.regime_signals.iloc[-1])
                
        except Exception as e:
            logger.warning(f"Error checking regime for date {date}: {e}")
            return True  # Default to allowing trade if error
    
    def get_regime_summary(self) -> Dict[str, Any]:
        """
        Get summary of regime filter performance.
        
        Returns:
            Dictionary with regime filter metrics
        """
        if not self.config.enabled:
            return {
                'regime_filter_enabled': False,
                'regime_hit_rate': 0.0,
                'proxy_symbol': None,
                'regime_rule': None
            }
        
        return {
            'regime_filter_enabled': True,
            'regime_hit_rate': self.regime_hit_rate,
            'proxy_symbol': self.config.proxy_symbol,
            'regime_rule': self.config.regime_rule,
            'sma_window': self.config.sma_window,
            'total_days': len(self.regime_signals) if self.regime_signals is not None else 0,
            'trading_days': int(self.regime_hit_rate * len(self.regime_signals)) if self.regime_signals is not None else 0
        }
    
    def log_to_mlflow(self):
        """Log regime filter configuration and results to MLflow."""
        try:
            import mlflow
            
            if mlflow.active_run() is None:
                logger.warning("No active MLflow run, skipping regime filter logging")
                return
            
            # Log configuration
            mlflow.log_param("regime_filter_enabled", self.config.enabled)
            if self.config.enabled:
                mlflow.log_param("regime_proxy_symbol", self.config.proxy_symbol)
                mlflow.log_param("regime_rule", self.config.regime_rule)
                mlflow.log_param("regime_sma_window", self.config.sma_window)
                
                # Log results
                summary = self.get_regime_summary()
                mlflow.log_metric("regime_hit_rate", summary['regime_hit_rate'])
                mlflow.log_metric("regime_trading_days", summary['trading_days'])
                mlflow.log_metric("regime_total_days", summary['total_days'])
                
        except Exception as e:
            logger.error(f"Error logging regime filter to MLflow: {e}")


def create_regime_filter_config(enabled: bool = False, 
                               proxy_symbol: str = "SPY",
                               regime_rule: str = "Bull only (proxy > SMA(200))") -> RegimeFilterConfig:
    """
    Create a regime filter configuration.
    
    Args:
        enabled: Whether regime filtering is enabled
        proxy_symbol: Market proxy symbol
        regime_rule: Regime rule to apply
        
    Returns:
        RegimeFilterConfig instance
    """
    return RegimeFilterConfig(
        enabled=enabled,
        proxy_symbol=proxy_symbol,
        regime_rule=regime_rule
    )
