"""
Position Management Module

This module implements position caps and rebalancing frequency controls
for portfolio risk management.

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
class PositionManagementConfig:
    """Configuration for position management."""
    max_position_pct: float = 0.15  # 15% max position per name
    rebalance_frequency: str = "monthly"  # "daily", "weekly", "monthly"
    min_rebalance_interval: int = 1  # minimum days between rebalances
    last_rebalance_date: Optional[datetime] = None
    turnover_threshold: float = 0.05  # 5% minimum turnover to trigger rebalance


class PositionManager:
    """
    Position management for portfolio strategies.
    
    Handles position caps and rebalancing frequency controls.
    """
    
    def __init__(self, config: PositionManagementConfig):
        self.config = config
        self.rebalance_count = 0
        self.cap_breaches = 0
        self.total_turnover = 0.0
        self.rebalance_dates = []
        self.weight_history = []
    
    def should_rebalance(self, current_date: datetime, 
                        current_weights: Dict[str, float],
                        target_weights: Dict[str, float]) -> bool:
        """
        Determine if portfolio should be rebalanced.
        
        Args:
            current_date: Current trading date
            current_weights: Current portfolio weights
            target_weights: Target portfolio weights
            
        Returns:
            True if rebalancing should occur
        """
        # Check if it's the first rebalance
        if self.config.last_rebalance_date is None:
            return True
        
        # Check minimum interval
        days_since_rebalance = (current_date - self.config.last_rebalance_date).days
        if days_since_rebalance < self.config.min_rebalance_interval:
            return False
        
        # Check frequency-based rebalancing
        if self.config.rebalance_frequency == "daily":
            return True
        elif self.config.rebalance_frequency == "weekly":
            return current_date.weekday() == 0  # Monday
        elif self.config.rebalance_frequency == "monthly":
            return current_date.day == 1  # First day of month
        else:
            logger.warning(f"Unknown rebalance frequency: {self.config.rebalance_frequency}")
            return False
    
    def apply_position_caps(self, weights: Dict[str, float]) -> Dict[str, float]:
        """
        Apply position caps to weights and renormalize.
        
        Args:
            weights: Original weights dictionary
            
        Returns:
            Capped and renormalized weights
        """
        if not weights:
            return weights
        
        # Apply position caps
        capped_weights = {}
        excess_weight = 0.0
        
        for symbol, weight in weights.items():
            if weight > self.config.max_position_pct:
                # Cap the weight and track excess
                capped_weight = self.config.max_position_pct
                excess_weight += weight - self.config.max_position_pct
                self.cap_breaches += 1
                logger.debug(f"Position cap applied to {symbol}: {weight:.3f} -> {capped_weight:.3f}")
            else:
                capped_weight = weight
            
            capped_weights[symbol] = capped_weight
        
        # Redistribute excess weight proportionally among uncapped positions
        if excess_weight > 0:
            uncapped_symbols = [symbol for symbol, weight in weights.items() 
                              if weight <= self.config.max_position_pct]
            
            if uncapped_symbols:
                # Calculate total weight of uncapped positions
                total_uncapped_weight = sum(capped_weights[symbol] for symbol in uncapped_symbols)
                
                if total_uncapped_weight > 0:
                    # Redistribute excess proportionally
                    for symbol in uncapped_symbols:
                        additional_weight = (capped_weights[symbol] / total_uncapped_weight) * excess_weight
                        capped_weights[symbol] += additional_weight
                else:
                    # If no uncapped positions, distribute equally
                    weight_per_symbol = excess_weight / len(uncapped_symbols)
                    for symbol in uncapped_symbols:
                        capped_weights[symbol] += weight_per_symbol
        
        # Final normalization to ensure weights sum to 1
        total_weight = sum(capped_weights.values())
        if total_weight > 0:
            capped_weights = {symbol: weight / total_weight 
                            for symbol, weight in capped_weights.items()}
        
        return capped_weights
    
    def calculate_turnover(self, old_weights: Dict[str, float], 
                          new_weights: Dict[str, float]) -> float:
        """
        Calculate portfolio turnover between two weight sets.
        
        Args:
            old_weights: Previous weights
            new_weights: New weights
            
        Returns:
            Turnover percentage
        """
        if not old_weights or not new_weights:
            return 0.0
        
        # Get all symbols from both weight sets
        all_symbols = set(old_weights.keys()) | set(new_weights.keys())
        
        # Calculate turnover as sum of absolute weight changes
        turnover = 0.0
        for symbol in all_symbols:
            old_weight = old_weights.get(symbol, 0.0)
            new_weight = new_weights.get(symbol, 0.0)
            turnover += abs(new_weight - old_weight)
        
        return turnover / 2.0  # Divide by 2 to get one-way turnover
    
    def rebalance_portfolio(self, current_date: datetime,
                           current_weights: Dict[str, float],
                           target_weights: Dict[str, float]) -> Tuple[Dict[str, float], bool]:
        """
        Rebalance portfolio if conditions are met.
        
        Args:
            current_date: Current trading date
            current_weights: Current portfolio weights
            target_weights: Target portfolio weights
            
        Returns:
            Tuple of (final_weights, was_rebalanced)
        """
        # Check if rebalancing should occur
        if not self.should_rebalance(current_date, current_weights, target_weights):
            return current_weights, False
        
        # Apply position caps to target weights
        capped_weights = self.apply_position_caps(target_weights)
        
        # Calculate turnover
        turnover = self.calculate_turnover(current_weights, capped_weights)
        
        # Only rebalance if turnover exceeds threshold
        if turnover < self.config.turnover_threshold:
            logger.debug(f"Turnover {turnover:.3f} below threshold {self.config.turnover_threshold}, skipping rebalance")
            return current_weights, False
        
        # Update tracking variables
        self.rebalance_count += 1
        self.total_turnover += turnover
        self.config.last_rebalance_date = current_date
        self.rebalance_dates.append(current_date)
        
        # Log weight change
        self.weight_history.append({
            'date': current_date,
            'old_weights': current_weights.copy(),
            'new_weights': capped_weights.copy(),
            'turnover': turnover
        })
        
        logger.info(f"Portfolio rebalanced on {current_date.strftime('%Y-%m-%d')}: "
                   f"turnover={turnover:.3f}, caps_breached={self.cap_breaches}")
        
        return capped_weights, True
    
    def get_management_summary(self) -> Dict[str, Any]:
        """
        Get summary of position management performance.
        
        Returns:
            Dictionary with position management metrics
        """
        return {
            'max_position_pct': self.config.max_position_pct,
            'rebalance_frequency': self.config.rebalance_frequency,
            'total_rebalances': self.rebalance_count,
            'total_turnover': self.total_turnover,
            'avg_turnover': self.total_turnover / max(self.rebalance_count, 1),
            'cap_breaches': self.cap_breaches,
            'last_rebalance_date': self.config.last_rebalance_date,
            'rebalance_dates': self.rebalance_dates
        }
    
    def log_to_mlflow(self):
        """Log position management configuration and results to MLflow."""
        try:
            import mlflow
            
            if mlflow.active_run() is None:
                logger.warning("No active MLflow run, skipping position management logging")
                return
            
            # Log configuration
            mlflow.log_param("max_position_pct", self.config.max_position_pct)
            mlflow.log_param("rebalance_frequency", self.config.rebalance_frequency)
            mlflow.log_param("min_rebalance_interval", self.config.min_rebalance_interval)
            mlflow.log_param("turnover_threshold", self.config.turnover_threshold)
            
            # Log results
            summary = self.get_management_summary()
            mlflow.log_metric("total_rebalances", summary['total_rebalances'])
            mlflow.log_metric("total_turnover", summary['total_turnover'])
            mlflow.log_metric("avg_turnover", summary['avg_turnover'])
            mlflow.log_metric("cap_breaches", summary['cap_breaches'])
            
        except Exception as e:
            logger.error(f"Error logging position management to MLflow: {e}")


def create_position_management_config(max_position_pct: float = 0.15,
                                    rebalance_frequency: str = "monthly",
                                    min_rebalance_interval: int = 1,
                                    turnover_threshold: float = 0.05) -> PositionManagementConfig:
    """
    Create a position management configuration.
    
    Args:
        max_position_pct: Maximum position percentage per name
        rebalance_frequency: Rebalancing frequency ("daily", "weekly", "monthly")
        min_rebalance_interval: Minimum days between rebalances
        turnover_threshold: Minimum turnover to trigger rebalance
        
    Returns:
        PositionManagementConfig instance
    """
    return PositionManagementConfig(
        max_position_pct=max_position_pct,
        rebalance_frequency=rebalance_frequency,
        min_rebalance_interval=min_rebalance_interval,
        turnover_threshold=turnover_threshold
    )
