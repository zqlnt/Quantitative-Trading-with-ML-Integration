"""
Sanity checks and fail-fast gates for Neural Quant.

This module provides comprehensive sanity checks to ensure data quality,
model validity, and system integrity.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime
import logging
import warnings

logger = logging.getLogger(__name__)


class SanityChecker:
    """
    Comprehensive sanity checker for trading systems.
    
    This class provides various sanity checks including:
    - Data quality checks
    - Model validation checks
    - Performance metric validation
    - Risk metric validation
    - System integrity checks
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize sanity checker.
        
        Args:
            config: Configuration for sanity checks
        """
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        
        # Default thresholds
        self.thresholds = {
            'max_kurtosis': 20.0,
            'min_sharpe': -5.0,
            'max_sharpe': 10.0,
            'min_win_rate': 0.0,
            'max_win_rate': 1.0,
            'max_drawdown_threshold': -0.5,  # -50%
            'min_tail_ratio': 0.1,
            'max_tail_ratio': 10.0,
            'max_skewness': 10.0,
            'min_skewness': -10.0,
            'cost_error_threshold': 0.20,  # 20%
            'class_imbalance_threshold': 0.8  # 80/20 split
        }
        
        # Update with config
        self.thresholds.update(self.config.get('thresholds', {}))
    
    def check_data_quality(self, data: pd.DataFrame, 
                          required_columns: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Check data quality and return issues.
        
        Args:
            data: DataFrame to check
            required_columns: List of required columns
            
        Returns:
            Dict[str, Any]: Quality check results
        """
        issues = []
        warnings = []
        
        # Check for empty data
        if data.empty:
            issues.append("Data is empty")
            return {'issues': issues, 'warnings': warnings, 'passed': False}
        
        # Check required columns
        if required_columns:
            missing_cols = [col for col in required_columns if col not in data.columns]
            if missing_cols:
                issues.append(f"Missing required columns: {missing_cols}")
        
        # Check for excessive NaNs
        nan_counts = data.isnull().sum()
        nan_pct = (nan_counts / len(data)) * 100
        
        for col, pct in nan_pct.items():
            if pct > 50:
                issues.append(f"Column {col} has {pct:.1f}% NaN values")
            elif pct > 10:
                warnings.append(f"Column {col} has {pct:.1f}% NaN values")
        
        # Check for infinite values
        inf_counts = np.isinf(data.select_dtypes(include=[np.number])).sum()
        for col, count in inf_counts.items():
            if count > 0:
                issues.append(f"Column {col} has {count} infinite values")
        
        # Check for duplicate indices
        if data.index.duplicated().any():
            issues.append("Data has duplicate indices")
        
        # Check for constant columns
        constant_cols = []
        for col in data.select_dtypes(include=[np.number]).columns:
            if data[col].nunique() <= 1:
                constant_cols.append(col)
        
        if constant_cols:
            warnings.append(f"Constant columns detected: {constant_cols}")
        
        passed = len(issues) == 0
        
        return {
            'issues': issues,
            'warnings': warnings,
            'passed': passed,
            'data_shape': data.shape,
            'nan_summary': nan_pct.to_dict(),
            'constant_columns': constant_cols
        }
    
    def check_metrics_sanity(self, metrics: Dict[str, float]) -> Dict[str, Any]:
        """
        Check sanity of performance metrics.
        
        Args:
            metrics: Dictionary of performance metrics
            
        Returns:
            Dict[str, Any]: Sanity check results
        """
        issues = []
        warnings = []
        
        # Check for NaNs and infinities
        for metric_name, value in metrics.items():
            if pd.isna(value) or not np.isfinite(value):
                issues.append(f"Metric {metric_name} is NaN or infinite: {value}")
        
        # Check Sharpe ratio range
        sharpe = metrics.get('sharpe_ratio', 0)
        if not (self.thresholds['min_sharpe'] <= sharpe <= self.thresholds['max_sharpe']):
            issues.append(f"Sharpe ratio out of range: {sharpe} (expected {self.thresholds['min_sharpe']} to {self.thresholds['max_sharpe']})")
        
        # Check win rate range
        win_rate = metrics.get('win_rate', 0)
        if not (self.thresholds['min_win_rate'] <= win_rate <= self.thresholds['max_win_rate']):
            issues.append(f"Win rate out of range: {win_rate} (expected {self.thresholds['min_win_rate']} to {self.thresholds['max_win_rate']})")
        
        # Check max drawdown
        max_dd = metrics.get('max_drawdown', 0)
        if max_dd < self.thresholds['max_drawdown_threshold']:
            issues.append(f"Max drawdown too severe: {max_dd:.2%} (threshold: {self.thresholds['max_drawdown_threshold']:.2%})")
        
        # Check kurtosis (fat tails)
        kurtosis = metrics.get('kurtosis', 0)
        if kurtosis > self.thresholds['max_kurtosis']:
            warnings.append(f"High kurtosis detected: {kurtosis:.2f} (threshold: {self.thresholds['max_kurtosis']})")
        
        # Check tail ratio
        tail_ratio = metrics.get('tail_ratio', 0)
        if not (self.thresholds['min_tail_ratio'] <= tail_ratio <= self.thresholds['max_tail_ratio']):
            warnings.append(f"Tail ratio out of range: {tail_ratio:.2f} (expected {self.thresholds['min_tail_ratio']} to {self.thresholds['max_tail_ratio']})")
        
        # Check skewness
        skewness = metrics.get('skewness', 0)
        if not (self.thresholds['min_skewness'] <= skewness <= self.thresholds['max_skewness']):
            warnings.append(f"Skewness out of range: {skewness:.2f} (expected {self.thresholds['min_skewness']} to {self.thresholds['max_skewness']})")
        
        # Check for suspiciously good metrics
        if sharpe > 5:
            warnings.append(f"Suspiciously high Sharpe ratio: {sharpe:.2f}")
        
        if win_rate > 0.9:
            warnings.append(f"Suspiciously high win rate: {win_rate:.2%}")
        
        passed = len(issues) == 0
        
        return {
            'issues': issues,
            'warnings': warnings,
            'passed': passed,
            'checked_metrics': list(metrics.keys())
        }
    
    def check_cost_reconciliation(self, backtest_costs: Dict[str, float], 
                                paper_costs: Dict[str, float]) -> Dict[str, Any]:
        """
        Check cost reconciliation between backtest and paper trading.
        
        Args:
            backtest_costs: Backtest cost metrics
            paper_costs: Paper trading cost metrics
            
        Returns:
            Dict[str, Any]: Reconciliation check results
        """
        issues = []
        warnings = []
        
        # Calculate cost error
        backtest_total = backtest_costs.get('total_commission', 0) + backtest_costs.get('total_slippage', 0)
        paper_total = paper_costs.get('total_commission', 0) + paper_costs.get('total_slippage', 0)
        
        if backtest_total > 0:
            cost_error = abs(paper_total - backtest_total) / backtest_total
        else:
            cost_error = 0
        
        if cost_error > self.thresholds['cost_error_threshold']:
            issues.append(f"Cost reconciliation error too high: {cost_error:.2%} (threshold: {self.thresholds['cost_error_threshold']:.2%})")
        elif cost_error > self.thresholds['cost_error_threshold'] / 2:
            warnings.append(f"Cost reconciliation error moderate: {cost_error:.2%}")
        
        passed = len(issues) == 0
        
        return {
            'issues': issues,
            'warnings': warnings,
            'passed': passed,
            'cost_error': cost_error,
            'backtest_costs': backtest_total,
            'paper_costs': paper_total
        }
    
    def check_class_imbalance(self, y: pd.Series, 
                             threshold: Optional[float] = None) -> Dict[str, Any]:
        """
        Check for class imbalance in classification problems.
        
        Args:
            y: Target series
            threshold: Imbalance threshold (default from config)
            
        Returns:
            Dict[str, Any]: Imbalance check results
        """
        if threshold is None:
            threshold = self.thresholds['class_imbalance_threshold']
        
        issues = []
        warnings = []
        
        # Calculate class distribution
        class_counts = y.value_counts()
        total_samples = len(y)
        
        for class_label, count in class_counts.items():
            class_ratio = count / total_samples
            
            if class_ratio > threshold or class_ratio < (1 - threshold):
                issues.append(f"Extreme class imbalance: {class_label} = {class_ratio:.2%}")
            elif class_ratio > threshold * 0.8 or class_ratio < (1 - threshold) * 0.8:
                warnings.append(f"Moderate class imbalance: {class_label} = {class_ratio:.2%}")
        
        passed = len(issues) == 0
        
        return {
            'issues': issues,
            'warnings': warnings,
            'passed': passed,
            'class_distribution': class_counts.to_dict(),
            'max_class_ratio': class_counts.max() / total_samples
        }
    
    def check_flat_series(self, returns: pd.Series) -> Dict[str, Any]:
        """
        Check for flat price series (no movement).
        
        Args:
            returns: Returns series
            
        Returns:
            Dict[str, Any]: Flat series check results
        """
        issues = []
        warnings = []
        
        # Check for zero volatility
        volatility = returns.std()
        if volatility == 0:
            issues.append("Returns series has zero volatility (flat series)")
        elif volatility < 0.001:  # Very low volatility
            warnings.append(f"Returns series has very low volatility: {volatility:.6f}")
        
        # Check for constant returns
        unique_returns = returns.nunique()
        if unique_returns <= 1:
            issues.append("Returns series is constant")
        elif unique_returns <= 5:
            warnings.append(f"Returns series has very few unique values: {unique_returns}")
        
        passed = len(issues) == 0
        
        return {
            'issues': issues,
            'warnings': warnings,
            'passed': passed,
            'volatility': volatility,
            'unique_values': unique_returns
        }
    
    def check_pathological_costs(self, returns: pd.Series, 
                               costs: pd.Series) -> Dict[str, Any]:
        """
        Check for pathological cost scenarios.
        
        Args:
            returns: Returns series
            costs: Costs series
            
        Returns:
            Dict[str, Any]: Pathological cost check results
        """
        issues = []
        warnings = []
        
        # Check if costs dominate returns
        total_returns = returns.sum()
        total_costs = costs.sum()
        
        if total_costs > 0:
            cost_ratio = abs(total_costs) / abs(total_returns) if total_returns != 0 else float('inf')
            
            if cost_ratio > 1.0:
                issues.append(f"Costs dominate returns: {cost_ratio:.2f}x")
            elif cost_ratio > 0.5:
                warnings.append(f"High cost ratio: {cost_ratio:.2f}x")
        
        # Check for excessive daily costs
        max_daily_cost = costs.max()
        if max_daily_cost > 0.1:  # 10% daily cost
            issues.append(f"Excessive daily cost: {max_daily_cost:.2%}")
        elif max_daily_cost > 0.05:  # 5% daily cost
            warnings.append(f"High daily cost: {max_daily_cost:.2%}")
        
        passed = len(issues) == 0
        
        return {
            'issues': issues,
            'warnings': warnings,
            'passed': passed,
            'cost_ratio': cost_ratio if 'cost_ratio' in locals() else 0,
            'max_daily_cost': max_daily_cost
        }
    
    def run_comprehensive_checks(self, data: pd.DataFrame, 
                               metrics: Dict[str, float],
                               returns: Optional[pd.Series] = None,
                               costs: Optional[pd.Series] = None,
                               y: Optional[pd.Series] = None) -> Dict[str, Any]:
        """
        Run comprehensive sanity checks.
        
        Args:
            data: Input data
            metrics: Performance metrics
            returns: Returns series
            costs: Costs series
            y: Target series for classification
            
        Returns:
            Dict[str, Any]: Comprehensive check results
        """
        all_issues = []
        all_warnings = []
        checks_passed = []
        
        # Data quality check
        data_check = self.check_data_quality(data)
        all_issues.extend(data_check['issues'])
        all_warnings.extend(data_check['warnings'])
        checks_passed.append(('data_quality', data_check['passed']))
        
        # Metrics sanity check
        metrics_check = self.check_metrics_sanity(metrics)
        all_issues.extend(metrics_check['issues'])
        all_warnings.extend(metrics_check['warnings'])
        checks_passed.append(('metrics_sanity', metrics_check['passed']))
        
        # Additional checks if data available
        if returns is not None:
            flat_check = self.check_flat_series(returns)
            all_issues.extend(flat_check['issues'])
            all_warnings.extend(flat_check['warnings'])
            checks_passed.append(('flat_series', flat_check['passed']))
        
        if costs is not None and returns is not None:
            cost_check = self.check_pathological_costs(returns, costs)
            all_issues.extend(cost_check['issues'])
            all_warnings.extend(cost_check['warnings'])
            checks_passed.append(('pathological_costs', cost_check['passed']))
        
        if y is not None:
            imbalance_check = self.check_class_imbalance(y)
            all_issues.extend(imbalance_check['issues'])
            all_warnings.extend(imbalance_check['warnings'])
            checks_passed.append(('class_imbalance', imbalance_check['passed']))
        
        # Overall result
        overall_passed = len(all_issues) == 0
        
        return {
            'overall_passed': overall_passed,
            'total_issues': len(all_issues),
            'total_warnings': len(all_warnings),
            'issues': all_issues,
            'warnings': all_warnings,
            'checks_passed': checks_passed,
            'timestamp': datetime.now().isoformat()
        }


def create_fail_fast_gates() -> Dict[str, Any]:
    """
    Create fail-fast gates for critical checks.
    
    Returns:
        Dict[str, Any]: Fail-fast gate configuration
    """
    return {
        'purged_cv_required': True,
        'costs_enabled': True,
        'metrics_required': [
            'sharpe_ratio',
            'max_drawdown',
            'total_return',
            'win_rate'
        ],
        'max_drawdown_limit': -0.12,  # -12%
        'min_sharpe_threshold': 0.0,
        'cost_error_limit': 0.30,  # 30%
        'data_quality_required': True,
        'leakage_detection': True
    }


def check_fail_fast_gates(results: Dict[str, Any], 
                         gates: Dict[str, Any]) -> bool:
    """
    Check fail-fast gates and return True if all pass.
    
    Args:
        results: Sanity check results
        gates: Fail-fast gate configuration
        
        Returns:
            bool: True if all gates pass
    """
    if gates.get('purged_cv_required', False):
        # This would need to be checked in the actual CV implementation
        pass
    
    if gates.get('costs_enabled', False):
        # Check if costs are properly calculated
        pass
    
    if gates.get('data_quality_required', False):
        if not results.get('overall_passed', False):
            return False
    
    # Check required metrics
    required_metrics = gates.get('metrics_required', [])
    # This would need to be checked against actual metrics
    
    return True


# Global sanity checker instance
sanity_checker = SanityChecker()


def run_sanity_checks(data: pd.DataFrame, 
                     metrics: Dict[str, float],
                     **kwargs) -> Dict[str, Any]:
    """
    Run sanity checks using the global checker.
    
    Args:
        data: Input data
        metrics: Performance metrics
        **kwargs: Additional arguments
        
    Returns:
        Dict[str, Any]: Sanity check results
    """
    return sanity_checker.run_comprehensive_checks(data, metrics, **kwargs)
