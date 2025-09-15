"""
Allocation Methods Module

This module implements different portfolio allocation methods including
equal-weight and volatility-weighted allocation.

Author: Neural Quant Team
Date: 2024
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class AllocationMethodConfig:
    """Configuration for allocation methods."""
    method: str = "equal_weight"  # "equal_weight" or "volatility_weighted"
    vol_lookback: int = 20  # days for volatility calculation
    min_weight: float = 0.01  # minimum weight per asset (1%)
    max_weight: float = 0.50  # maximum weight per asset (50%)


class AllocationMethods:
    """
    Portfolio allocation methods for multi-asset strategies.
    
    Supports equal-weight and volatility-weighted allocation schemes.
    """
    
    def __init__(self, config: AllocationMethodConfig):
        self.config = config
        self.weights_history: List[Dict[str, Any]] = []
        self.current_weights: Dict[str, float] = {}
    
    def calculate_equal_weights(self, symbols: List[str]) -> Dict[str, float]:
        """
        Calculate equal weights for all symbols.
        
        Args:
            symbols: List of asset symbols
            
        Returns:
            Dictionary of symbol -> weight
        """
        if not symbols:
            return {}
        
        weight = 1.0 / len(symbols)
        weights = {symbol: weight for symbol in symbols}
        
        # Apply min/max constraints
        weights = self._apply_weight_constraints(weights)
        
        self.current_weights = weights
        return weights
    
    def calculate_volatility_weights(self, returns_data: Dict[str, pd.Series]) -> Dict[str, float]:
        """
        Calculate volatility-weighted allocation (1/σ normalized).
        
        Args:
            returns_data: Dictionary of symbol -> returns series
            
        Returns:
            Dictionary of symbol -> weight
        """
        if not returns_data:
            return {}
        
        # Calculate realized volatility for each asset
        volatilities = {}
        for symbol, returns in returns_data.items():
            if len(returns) >= self.config.vol_lookback:
                # Use rolling volatility
                vol = returns.rolling(window=self.config.vol_lookback).std().iloc[-1]
            else:
                # Use full sample volatility if not enough data
                vol = returns.std()
            
            # Annualize volatility
            vol_annualized = vol * np.sqrt(252)
            volatilities[symbol] = vol_annualized
        
        # Calculate inverse volatility weights (1/σ)
        inv_volatilities = {symbol: 1.0 / vol if vol > 0 else 0.0 
                           for symbol, vol in volatilities.items()}
        
        # Normalize to sum to 1
        total_inv_vol = sum(inv_volatilities.values())
        if total_inv_vol > 0:
            weights = {symbol: inv_vol / total_inv_vol 
                      for symbol, inv_vol in inv_volatilities.items()}
        else:
            # Fallback to equal weights if all volatilities are zero
            weights = self.calculate_equal_weights(list(returns_data.keys()))
        
        # Apply min/max constraints
        weights = self._apply_weight_constraints(weights)
        
        self.current_weights = weights
        return weights
    
    def _apply_weight_constraints(self, weights: Dict[str, float]) -> Dict[str, float]:
        """
        Apply minimum and maximum weight constraints.
        
        Args:
            weights: Original weights dictionary
            
        Returns:
            Constrained weights dictionary
        """
        # Apply minimum weight constraint
        constrained_weights = {}
        for symbol, weight in weights.items():
            constrained_weights[symbol] = max(weight, self.config.min_weight)
        
        # Normalize after applying minimum weights
        total_weight = sum(constrained_weights.values())
        if total_weight > 0:
            constrained_weights = {symbol: weight / total_weight 
                                 for symbol, weight in constrained_weights.items()}
        
        # Apply maximum weight constraint
        for symbol in constrained_weights:
            if constrained_weights[symbol] > self.config.max_weight:
                constrained_weights[symbol] = self.config.max_weight
        
        # Final normalization
        total_weight = sum(constrained_weights.values())
        if total_weight > 0:
            constrained_weights = {symbol: weight / total_weight 
                                 for symbol, weight in constrained_weights.items()}
        
        return constrained_weights
    
    def calculate_weights(self, symbols: List[str], 
                         returns_data: Optional[Dict[str, pd.Series]] = None) -> Dict[str, float]:
        """
        Calculate portfolio weights based on selected method.
        
        Args:
            symbols: List of asset symbols
            returns_data: Optional returns data for volatility weighting
            
        Returns:
            Dictionary of symbol -> weight
        """
        if self.config.method == "equal_weight":
            return self.calculate_equal_weights(symbols)
        elif self.config.method == "volatility_weighted":
            if returns_data is None:
                logger.warning("No returns data provided for volatility weighting, falling back to equal weights")
                return self.calculate_equal_weights(symbols)
            return self.calculate_volatility_weights(returns_data)
        else:
            logger.warning(f"Unknown allocation method: {self.config.method}, falling back to equal weights")
            return self.calculate_equal_weights(symbols)
    
    def log_weights(self, date: pd.Timestamp, weights: Dict[str, float]):
        """
        Log weights for a specific date.
        
        Args:
            date: Trading date
            weights: Weights dictionary
        """
        weight_entry = {
            'date': date,
            'method': self.config.method,
            **weights
        }
        self.weights_history.append(weight_entry)
    
    def get_weights_summary(self) -> Dict[str, Any]:
        """
        Get summary of allocation method performance.
        
        Returns:
            Dictionary with allocation method metrics
        """
        if not self.weights_history:
            return {
                'allocation_method': self.config.method,
                'total_allocations': 0,
                'avg_weights': {},
                'weight_volatility': {}
            }
        
        # Convert to DataFrame for analysis
        df = pd.DataFrame(self.weights_history)
        df = df.set_index('date')
        
        # Calculate average weights per symbol
        weight_cols = [col for col in df.columns if col != 'method']
        avg_weights = df[weight_cols].mean().to_dict()
        
        # Calculate weight volatility (how much weights change over time)
        weight_volatility = df[weight_cols].std().to_dict()
        
        return {
            'allocation_method': self.config.method,
            'total_allocations': len(self.weights_history),
            'avg_weights': avg_weights,
            'weight_volatility': weight_volatility,
            'vol_lookback': self.config.vol_lookback if self.config.method == "volatility_weighted" else None
        }
    
    def save_weights_csv(self, filename: str = "weights.csv"):
        """
        Save weights history to CSV file.
        
        Args:
            filename: Output filename
        """
        if not self.weights_history:
            logger.warning("No weights history to save")
            return
        
        df = pd.DataFrame(self.weights_history)
        df.to_csv(filename, index=False)
        logger.info(f"Weights saved to {filename}")
    
    def log_to_mlflow(self):
        """Log allocation method configuration and results to MLflow."""
        try:
            import mlflow
            
            if mlflow.active_run() is None:
                logger.warning("No active MLflow run, skipping allocation method logging")
                return
            
            # Log configuration
            mlflow.log_param("allocation_method", self.config.method)
            if self.config.method == "volatility_weighted":
                mlflow.log_param("vol_lookback", self.config.vol_lookback)
            mlflow.log_param("min_weight", self.config.min_weight)
            mlflow.log_param("max_weight", self.config.max_weight)
            
            # Log results
            summary = self.get_weights_summary()
            mlflow.log_metric("total_allocations", summary['total_allocations'])
            
            # Log average weights for each symbol
            for symbol, weight in summary['avg_weights'].items():
                mlflow.log_metric(f"avg_weight_{symbol}", weight)
            
            # Log weight volatility for each symbol
            for symbol, vol in summary['weight_volatility'].items():
                mlflow.log_metric(f"weight_vol_{symbol}", vol)
            
            # Save weights CSV as artifact
            self.save_weights_csv()
            mlflow.log_artifact("weights.csv")
            
        except Exception as e:
            logger.error(f"Error logging allocation method to MLflow: {e}")


def create_allocation_method_config(method: str = "equal_weight",
                                  vol_lookback: int = 20,
                                  min_weight: float = 0.01,
                                  max_weight: float = 0.50) -> AllocationMethodConfig:
    """
    Create an allocation method configuration.
    
    Args:
        method: Allocation method ("equal_weight" or "volatility_weighted")
        vol_lookback: Lookback window for volatility calculation
        min_weight: Minimum weight per asset
        max_weight: Maximum weight per asset
        
    Returns:
        AllocationMethodConfig instance
    """
    return AllocationMethodConfig(
        method=method,
        vol_lookback=vol_lookback,
        min_weight=min_weight,
        max_weight=max_weight
    )
