"""
Basic Exits Module

This module implements basic exit strategies including ATR stops and time stops
for realistic trade management.

Author: Neural Quant Team
Date: 2024
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)


@dataclass
class BasicExitsConfig:
    """Configuration for basic exit strategies."""
    enable_atr_stop: bool = False
    atr_window: int = 14
    atr_multiplier: float = 2.5
    enable_time_stop: bool = False
    time_stop_bars: int = 30
    min_atr_value: float = 0.001  # Minimum ATR value to avoid division by zero


class BasicExits:
    """
    Basic exit strategies for trade management.
    
    Implements ATR-based stops and time-based stops.
    """
    
    def __init__(self, config: BasicExitsConfig):
        self.config = config
        self.atr_stops = {}  # symbol -> trailing stop level
        self.time_stops = {}  # symbol -> entry date
        self.stop_hits = []
        self.atr_values = {}  # symbol -> ATR series
        
    def calculate_atr(self, data: pd.DataFrame, symbol: str) -> pd.Series:
        """
        Calculate Average True Range (ATR) for a symbol.
        
        Args:
            data: Price data with OHLC columns
            symbol: Symbol name
            
        Returns:
            ATR series
        """
        high_col = f'{symbol}_high'
        low_col = f'{symbol}_low'
        close_col = f'{symbol}_close'
        
        if not all(col in data.columns for col in [high_col, low_col, close_col]):
            logger.warning(f"Missing OHLC data for {symbol}, using close price for ATR")
            # Fallback to using close price volatility
            close_prices = data[close_col].dropna()
            if len(close_prices) > 1:
                returns = close_prices.pct_change().dropna()
                atr = returns.rolling(window=self.config.atr_window).std() * close_prices
                return atr.fillna(self.config.min_atr_value)
            else:
                return pd.Series(index=data.index, data=self.config.min_atr_value)
        
        high = data[high_col]
        low = data[low_col]
        close = data[close_col]
        
        # Calculate True Range
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        
        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        
        # Calculate ATR as rolling mean of True Range
        atr = true_range.rolling(window=self.config.atr_window).mean()
        
        # Fill NaN values with minimum ATR
        atr = atr.fillna(self.config.min_atr_value)
        
        # Ensure minimum ATR value
        atr = atr.clip(lower=self.config.min_atr_value)
        
        return atr
    
    def initialize_stops(self, symbol: str, entry_date: datetime, entry_price: float) -> bool:
        """
        Initialize stop levels for a new position.
        
        Args:
            symbol: Symbol name
            entry_date: Entry date
            entry_price: Entry price
            
        Returns:
            True if stops were initialized successfully
        """
        try:
            # Initialize ATR stop if enabled
            if self.config.enable_atr_stop:
                # Set initial trailing stop using a simple volatility estimate
                # We'll calculate proper ATR when we have more data
                initial_atr = entry_price * 0.02  # 2% of price as initial estimate
                stop_distance = initial_atr * self.config.atr_multiplier
                self.atr_stops[symbol] = entry_price - stop_distance
                logger.debug(f"Initialized ATR stop for {symbol}: {self.atr_stops[symbol]:.4f}")
            
            # Initialize time stop if enabled
            if self.config.enable_time_stop:
                self.time_stops[symbol] = entry_date
                logger.debug(f"Initialized time stop for {symbol}: {entry_date}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error initializing stops for {symbol}: {e}")
            return False
    
    def update_atr_stop(self, symbol: str, current_price: float, current_date: datetime) -> bool:
        """
        Update trailing ATR stop for a position.
        
        Args:
            symbol: Symbol name
            current_price: Current price
            current_date: Current date
            
        Returns:
            True if stop was updated
        """
        if symbol not in self.atr_stops:
            return False
        
        try:
            # Use a simple volatility estimate based on price movement
            # This is a simplified approach that doesn't require full historical data
            price_change = abs(current_price - self.atr_stops[symbol]) / current_price
            estimated_atr = current_price * max(price_change, 0.01)  # Minimum 1% volatility
            
            # Calculate new stop level
            stop_distance = estimated_atr * self.config.atr_multiplier
            new_stop = current_price - stop_distance
            
            # Update stop only if it's higher (trailing up)
            if new_stop > self.atr_stops[symbol]:
                self.atr_stops[symbol] = new_stop
                logger.debug(f"Updated ATR stop for {symbol}: {new_stop:.4f}")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error updating ATR stop for {symbol}: {e}")
            return False
    
    def check_atr_stop(self, symbol: str, current_price: float) -> bool:
        """
        Check if ATR stop was hit.
        
        Args:
            symbol: Symbol name
            current_price: Current price
            
        Returns:
            True if stop was hit
        """
        if symbol not in self.atr_stops:
            return False
        
        if current_price <= self.atr_stops[symbol]:
            # Record stop hit
            self.stop_hits.append({
                'symbol': symbol,
                'stop_type': 'ATR',
                'stop_level': self.atr_stops[symbol],
                'exit_price': current_price,
                'date': datetime.now()
            })
            logger.info(f"ATR stop hit for {symbol}: {current_price:.4f} <= {self.atr_stops[symbol]:.4f}")
            return True
        
        return False
    
    def check_time_stop(self, symbol: str, current_date: datetime) -> bool:
        """
        Check if time stop was hit.
        
        Args:
            symbol: Symbol name
            current_date: Current date
            
        Returns:
            True if stop was hit
        """
        if symbol not in self.time_stops:
            return False
        
        entry_date = self.time_stops[symbol]
        
        # Calculate days held
        days_held = (current_date - entry_date).days
        
        if days_held >= self.config.time_stop_bars:
            # Record stop hit
            self.stop_hits.append({
                'symbol': symbol,
                'stop_type': 'TIME',
                'stop_level': None,
                'exit_price': None,  # Will be set by caller
                'date': current_date,
                'days_held': days_held
            })
            logger.info(f"Time stop hit for {symbol}: {days_held} days held")
            return True
        
        return False
    
    def check_exits(self, symbol: str, current_price: float, current_date: datetime) -> Tuple[bool, str]:
        """
        Check all exit conditions for a position.
        
        Args:
            symbol: Symbol name
            current_price: Current price
            current_date: Current date
            
        Returns:
            Tuple of (should_exit, exit_reason)
        """
        # Check ATR stop first
        if self.config.enable_atr_stop and self.check_atr_stop(symbol, current_price):
            return True, "ATR_STOP"
        
        # Check time stop
        if self.config.enable_time_stop and self.check_time_stop(symbol, current_date):
            return True, "TIME_STOP"
        
        return False, ""
    
    def update_stops(self, symbol: str, current_price: float, current_date: datetime):
        """
        Update all stop levels for a position.
        
        Args:
            symbol: Symbol name
            current_price: Current price
            current_date: Current date
        """
        # Update ATR stop
        if self.config.enable_atr_stop:
            self.update_atr_stop(symbol, current_price, current_date)
    
    def close_position(self, symbol: str):
        """
        Clean up stop tracking when position is closed.
        
        Args:
            symbol: Symbol name
        """
        if symbol in self.atr_stops:
            del self.atr_stops[symbol]
        if symbol in self.time_stops:
            del self.time_stops[symbol]
        if symbol in self.atr_values:
            del self.atr_values[symbol]
    
    def get_exits_summary(self) -> Dict[str, Any]:
        """
        Get summary of exit strategy performance.
        
        Returns:
            Dictionary with exit strategy metrics
        """
        if not self.stop_hits:
            return {
                'total_stops': 0,
                'atr_stops': 0,
                'time_stops': 0,
                'avg_pnl_atr': 0.0,
                'avg_pnl_time': 0.0
            }
        
        atr_stops = [hit for hit in self.stop_hits if hit['stop_type'] == 'ATR']
        time_stops = [hit for hit in self.stop_hits if hit['stop_type'] == 'TIME']
        
        return {
            'total_stops': len(self.stop_hits),
            'atr_stops': len(atr_stops),
            'time_stops': len(time_stops),
            'atr_stop_rate': len(atr_stops) / len(self.stop_hits) if self.stop_hits else 0,
            'time_stop_rate': len(time_stops) / len(self.stop_hits) if self.stop_hits else 0
        }
    
    def log_to_mlflow(self):
        """Log exit strategy configuration and results to MLflow."""
        try:
            import mlflow
            
            if mlflow.active_run() is None:
                logger.warning("No active MLflow run, skipping basic exits logging")
                return
            
            # Log configuration
            mlflow.log_param("enable_atr_stop", self.config.enable_atr_stop)
            if self.config.enable_atr_stop:
                mlflow.log_param("atr_window", self.config.atr_window)
                mlflow.log_param("atr_multiplier", self.config.atr_multiplier)
            
            mlflow.log_param("enable_time_stop", self.config.enable_time_stop)
            if self.config.enable_time_stop:
                mlflow.log_param("time_stop_bars", self.config.time_stop_bars)
            
            # Log results
            summary = self.get_exits_summary()
            mlflow.log_metric("total_stops", summary['total_stops'])
            mlflow.log_metric("atr_stops", summary['atr_stops'])
            mlflow.log_metric("time_stops", summary['time_stops'])
            mlflow.log_metric("atr_stop_rate", summary['atr_stop_rate'])
            mlflow.log_metric("time_stop_rate", summary['time_stop_rate'])
            
        except Exception as e:
            logger.error(f"Error logging basic exits to MLflow: {e}")


def create_basic_exits_config(enable_atr_stop: bool = False,
                             atr_window: int = 14,
                             atr_multiplier: float = 2.5,
                             enable_time_stop: bool = False,
                             time_stop_bars: int = 30) -> BasicExitsConfig:
    """
    Create a basic exits configuration.
    
    Args:
        enable_atr_stop: Whether to enable ATR stops
        atr_window: ATR calculation window
        atr_multiplier: ATR stop multiplier
        enable_time_stop: Whether to enable time stops
        time_stop_bars: Number of bars for time stop
        
    Returns:
        BasicExitsConfig instance
    """
    return BasicExitsConfig(
        enable_atr_stop=enable_atr_stop,
        atr_window=atr_window,
        atr_multiplier=atr_multiplier,
        enable_time_stop=enable_time_stop,
        time_stop_bars=time_stop_bars
    )
