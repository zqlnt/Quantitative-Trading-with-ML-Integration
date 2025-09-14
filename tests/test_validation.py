"""
Validation tests for Neural Quant.

This module tests the purged cross-validation implementation and ensures
no data leakage occurs in time series models.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from src.utils.validation.cross_validation import PurgedCrossValidator, TimeSeriesSplit, validate_no_leakage
from src.utils.validation.metrics import calculate_comprehensive_metrics


class TestPurgedCrossValidation:
    """Test purged cross-validation correctness."""
    
    def test_purged_cv_no_leakage(self):
        """Test that purged CV prevents data leakage."""
        # Create synthetic time series data
        dates = pd.date_range('2020-01-01', '2023-12-31', freq='D')
        n_samples = len(dates)
        
        # Create features with clear temporal structure
        np.random.seed(42)
        X = pd.DataFrame({
            'feature1': np.cumsum(np.random.randn(n_samples)),
            'feature2': np.sin(np.arange(n_samples) * 2 * np.pi / 252),  # Annual cycle
            'feature3': np.random.randn(n_samples)
        }, index=dates)
        
        y = pd.Series(np.random.randn(n_samples), index=dates)
        
        # Test purged CV
        cv = PurgedCrossValidator(n_splits=5, embargo_pct=0.01, purge_pct=0.01)
        
        leakage_detected = False
        for train_idx, test_idx in cv.split(X, y):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
            
            # Check temporal ordering
            max_train_date = X_train.index.max()
            min_test_date = X_test.index.min()
            
            # Assert no temporal overlap
            assert max_train_date < min_test_date, f"Temporal leakage: max_train={max_train_date}, min_test={min_test_date}"
            
            # Check for index overlaps
            train_indices = set(X_train.index)
            test_indices = set(X_test.index)
            overlap = train_indices.intersection(test_indices)
            
            assert len(overlap) == 0, f"Index overlap detected: {len(overlap)} overlapping indices"
            
            # Validate no leakage using the validation function
            if not validate_no_leakage(X_train, X_test, y_train, y_test):
                leakage_detected = True
        
        assert not leakage_detected, "Data leakage detected in purged CV"
    
    def test_embargo_period(self):
        """Test that embargo period is properly enforced."""
        dates = pd.date_range('2020-01-01', '2023-12-31', freq='D')
        n_samples = len(dates)
        
        X = pd.DataFrame(np.random.randn(n_samples, 3), index=dates)
        y = pd.Series(np.random.randn(n_samples), index=dates)
        
        # Test with larger embargo period
        cv = PurgedCrossValidator(n_splits=3, embargo_pct=0.05, purge_pct=0.05)
        
        for train_idx, test_idx in cv.split(X, y):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            
            # Calculate gap between train and test
            max_train_date = X_train.index.max()
            min_test_date = X_test.index.min()
            gap_days = (min_test_date - max_train_date).days
            
            # Should have at least 5% gap (embargo + purge)
            min_gap = int(n_samples * 0.05)
            assert gap_days >= min_gap, f"Gap too small: {gap_days} days < {min_gap} days"
    
    def test_time_series_split(self):
        """Test time series split with datetime index."""
        dates = pd.date_range('2020-01-01', '2023-12-31', freq='D')
        n_samples = len(dates)
        
        X = pd.DataFrame(np.random.randn(n_samples, 3), index=dates)
        y = pd.Series(np.random.randn(n_samples), index=dates)
        
        ts_cv = TimeSeriesSplit(n_splits=3, embargo_days=5, purge_days=5)
        
        for train_idx, test_idx in ts_cv.split(X, y):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            
            # Check temporal ordering
            max_train_date = X_train.index.max()
            min_test_date = X_test.index.min()
            
            assert max_train_date < min_test_date, "Temporal ordering violated"
            
            # Check embargo period
            gap_days = (min_test_date - max_train_date).days
            assert gap_days >= 5, f"Embargo period violated: {gap_days} days < 5 days"


class TestLeakageSentinels:
    """Test leakage detection with synthetic data."""
    
    def test_trending_series_leakage_detection(self):
        """Test that trending series shows inflated metrics without purging."""
        # Create synthetic trending series
        dates = pd.date_range('2020-01-01', '2023-12-31', freq='D')
        n_samples = len(dates)
        
        # Strong upward trend
        trend = np.linspace(0, 10, n_samples)
        noise = np.random.randn(n_samples) * 0.1
        prices = trend + noise
        
        # Create features that leak future information
        X_leaky = pd.DataFrame({
            'price': prices,
            'future_price': np.roll(prices, -1),  # Future price leak
            'future_return': np.roll(np.diff(prices, prepend=prices[0]), -1)
        }, index=dates)
        
        y = pd.Series(np.diff(prices, prepend=prices[0]), index=dates)
        
        # Test with regular CV (should show inflated performance)
        from sklearn.model_selection import TimeSeriesSplit as SklearnTS
        from sklearn.linear_model import LinearRegression
        from sklearn.metrics import mean_squared_error
        
        # Regular time series split
        regular_cv = SklearnTS(n_splits=5)
        regular_scores = []
        
        for train_idx, test_idx in regular_cv.split(X_leaky):
            X_train, X_test = X_leaky.iloc[train_idx], X_leaky.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
            
            model = LinearRegression()
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            mse = mean_squared_error(y_test, y_pred)
            regular_scores.append(mse)
        
        # Test with purged CV (should show realistic performance)
        purged_cv = PurgedCrossValidator(n_splits=5, embargo_pct=0.05, purge_pct=0.05)
        purged_scores = []
        
        for train_idx, test_idx in purged_cv.split(X_leaky, y):
            X_train, X_test = X_leaky.iloc[train_idx], X_leaky.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
            
            model = LinearRegression()
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            mse = mean_squared_error(y_test, y_pred)
            purged_scores.append(mse)
        
        # Purged CV should have higher (worse) MSE due to no leakage
        avg_regular_mse = np.mean(regular_scores)
        avg_purged_mse = np.mean(purged_scores)
        
        # The difference should be significant
        assert avg_purged_mse > avg_regular_mse, f"Purged CV not working: {avg_purged_mse} <= {avg_regular_mse}"
        
        # Purged CV should be much worse (at least 50% higher MSE)
        assert avg_purged_mse >= avg_regular_mse * 1.5, f"Leakage not properly prevented: {avg_purged_mse} vs {avg_regular_mse}"


class TestMetricsSanity:
    """Test metrics sanity checks."""
    
    def test_metrics_no_nans(self):
        """Test that metrics don't contain NaNs."""
        # Create synthetic returns
        np.random.seed(42)
        returns = pd.Series(np.random.randn(252), index=pd.date_range('2023-01-01', periods=252, freq='D'))
        
        metrics = calculate_comprehensive_metrics(returns)
        
        # Check for NaNs
        for metric_name, metric_value in metrics.items():
            assert not pd.isna(metric_value), f"Metric {metric_name} contains NaN: {metric_value}"
            assert np.isfinite(metric_value), f"Metric {metric_name} is not finite: {metric_value}"
    
    def test_metrics_ranges(self):
        """Test that metrics are within reasonable ranges."""
        # Create synthetic returns
        np.random.seed(42)
        returns = pd.Series(np.random.randn(252), index=pd.date_range('2023-01-01', periods=252, freq='D'))
        
        metrics = calculate_comprehensive_metrics(returns)
        
        # Check ranges
        assert -1 <= metrics.get('total_return', 0) <= 10, f"Total return out of range: {metrics.get('total_return', 0)}"
        assert -5 <= metrics.get('sharpe_ratio', 0) <= 5, f"Sharpe ratio out of range: {metrics.get('sharpe_ratio', 0)}"
        assert -1 <= metrics.get('max_drawdown', 0) <= 0, f"Max drawdown out of range: {metrics.get('max_drawdown', 0)}"
        assert 0 <= metrics.get('win_rate', 0) <= 1, f"Win rate out of range: {metrics.get('win_rate', 0)}"
        assert -10 <= metrics.get('skewness', 0) <= 10, f"Skewness out of range: {metrics.get('skewness', 0)}"
        assert -5 <= metrics.get('kurtosis', 0) <= 20, f"Kurtosis out of range: {metrics.get('kurtosis', 0)}"
        assert 0 <= metrics.get('tail_ratio', 0) <= 10, f"Tail ratio out of range: {metrics.get('tail_ratio', 0)}"
    
    def test_flat_series_metrics(self):
        """Test metrics for flat price series."""
        # Create flat returns (all zeros)
        returns = pd.Series(np.zeros(252), index=pd.date_range('2023-01-01', periods=252, freq='D'))
        
        metrics = calculate_comprehensive_metrics(returns)
        
        # Flat series should have near-zero Sharpe
        assert abs(metrics.get('sharpe_ratio', 0)) < 0.1, f"Flat series has non-zero Sharpe: {metrics.get('sharpe_ratio', 0)}"
        
        # Volatility should be zero
        assert metrics.get('volatility', 0) == 0, f"Flat series has non-zero volatility: {metrics.get('volatility', 0)}"
        
        # Max drawdown should be zero
        assert metrics.get('max_drawdown', 0) == 0, f"Flat series has non-zero max drawdown: {metrics.get('max_drawdown', 0)}"


class TestEdgeCases:
    """Test edge cases and pathological scenarios."""
    
    def test_all_nan_features(self):
        """Test handling of all-NaN feature windows."""
        dates = pd.date_range('2020-01-01', '2023-12-31', freq='D')
        n_samples = len(dates)
        
        # Create data with NaN windows
        X = pd.DataFrame(np.random.randn(n_samples, 3), index=dates)
        X.iloc[100:200, :] = np.nan  # NaN window
        
        y = pd.Series(np.random.randn(n_samples), index=dates)
        
        # Should not crash
        cv = PurgedCrossValidator(n_splits=3)
        
        for train_idx, test_idx in cv.split(X, y):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
            
            # Should handle NaNs gracefully
            assert len(X_train) > 0, "Training set is empty"
            assert len(X_test) > 0, "Test set is empty"
    
    def test_sparse_labels(self):
        """Test handling of sparse labels (class imbalance)."""
        dates = pd.date_range('2020-01-01', '2023-12-31', freq='D')
        n_samples = len(dates)
        
        # Create highly imbalanced labels (90% negative, 10% positive)
        y = pd.Series(np.random.choice([0, 1], n_samples, p=[0.9, 0.1]), index=dates)
        X = pd.DataFrame(np.random.randn(n_samples, 3), index=dates)
        
        # Should handle imbalanced data
        cv = PurgedCrossValidator(n_splits=3)
        
        for train_idx, test_idx in cv.split(X, y):
            y_train = y.iloc[train_idx]
            
            # Check class distribution
            positive_ratio = y_train.mean()
            assert 0 <= positive_ratio <= 1, f"Invalid positive ratio: {positive_ratio}"
            
            # Warn if extreme imbalance
            if positive_ratio > 0.8 or positive_ratio < 0.2:
                print(f"Warning: Extreme class imbalance detected: {positive_ratio:.2%} positive")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
