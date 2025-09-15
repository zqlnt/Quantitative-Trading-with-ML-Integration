"""Unit tests for time utilities."""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timezone
from src.neural_quant.utils.time_utils import (
    ensure_tz_naive_daily_index,
    clamp_to_dates,
    normalize_intraday_index,
    is_daily_data,
    get_market_calendar_info
)

class TestTimeUtils:
    """Test time utility functions."""
    
    def test_ensure_tz_naive_daily_index_timezone_aware(self):
        """Test converting timezone-aware data to naive daily index."""
        # Create timezone-aware data
        dates = pd.date_range('2023-01-01', periods=5, freq='D', tz='UTC')
        series = pd.Series([100, 101, 102, 103, 104], index=dates)
        
        result = ensure_tz_naive_daily_index(series, market="US")
        
        # Should be timezone-naive
        assert result.index.tz is None
        # Should have 5 data points
        assert len(result) == 5
        # Should be normalized to market close time
        assert all(result.index.hour == 16)  # 4 PM US market close
    
    def test_ensure_tz_naive_daily_index_already_naive(self):
        """Test handling already timezone-naive data."""
        # Create timezone-naive data
        dates = pd.date_range('2023-01-01', periods=5, freq='D')
        series = pd.Series([100, 101, 102, 103, 104], index=dates)
        
        result = ensure_tz_naive_daily_index(series, market="US")
        
        # Should still be timezone-naive
        assert result.index.tz is None
        # Should have 5 data points
        assert len(result) == 5
    
    def test_clamp_to_dates(self):
        """Test clamping data to date range."""
        dates = pd.date_range('2023-01-01', periods=10, freq='D')
        series = pd.Series(range(10), index=dates)
        
        # Clamp to middle range
        result = clamp_to_dates(series, '2023-01-03', '2023-01-07')
        
        assert len(result) == 5
        assert result.index[0] == pd.Timestamp('2023-01-03')
        assert result.index[-1] == pd.Timestamp('2023-01-07')
    
    def test_clamp_to_dates_no_clamping(self):
        """Test clamping with no date restrictions."""
        dates = pd.date_range('2023-01-01', periods=5, freq='D')
        series = pd.Series(range(5), index=dates)
        
        result = clamp_to_dates(series)
        
        assert len(result) == 5
        assert result.equals(series)
    
    def test_normalize_intraday_index(self):
        """Test normalizing intraday data to UTC."""
        # Create timezone-aware intraday data
        dates = pd.date_range('2023-01-01 09:00', periods=5, freq='H', tz='America/New_York')
        series = pd.Series([100, 101, 102, 103, 104], index=dates)
        
        result = normalize_intraday_index(series, target_tz="UTC")
        
        # Should be timezone-naive
        assert result.index.tz is None
        # Should have 5 data points
        assert len(result) == 5
    
    def test_is_daily_data_true(self):
        """Test detecting daily data."""
        dates = pd.date_range('2023-01-01', periods=10, freq='D')
        series = pd.Series(range(10), index=dates)
        
        assert is_daily_data(series) == True
    
    def test_is_daily_data_false(self):
        """Test detecting non-daily data."""
        dates = pd.date_range('2023-01-01 09:00', periods=10, freq='H')
        series = pd.Series(range(10), index=dates)
        
        assert is_daily_data(series) == False
    
    def test_is_daily_data_insufficient_data(self):
        """Test handling insufficient data."""
        dates = pd.date_range('2023-01-01', periods=1, freq='D')
        series = pd.Series([100], index=dates)
        
        assert is_daily_data(series) == True  # Should assume daily
    
    def test_get_market_calendar_info(self):
        """Test getting market calendar information."""
        us_info = get_market_calendar_info("US")
        assert us_info["timezone"] == "America/New_York"
        assert us_info["open_time"] == "09:30"
        assert us_info["close_time"] == "16:00"
        
        eu_info = get_market_calendar_info("EU")
        assert eu_info["timezone"] == "Europe/London"
        
        asia_info = get_market_calendar_info("ASIA")
        assert asia_info["timezone"] == "Asia/Tokyo"
    
    def test_empty_series_handling(self):
        """Test handling empty series."""
        empty_series = pd.Series(dtype=float)
        
        result1 = ensure_tz_naive_daily_index(empty_series)
        assert len(result1) == 0
        
        result2 = clamp_to_dates(empty_series)
        assert len(result2) == 0
        
        result3 = normalize_intraday_index(empty_series)
        assert len(result3) == 0

class TestTimezoneIntegration:
    """Test timezone handling integration with real data."""
    
    def test_yf_data_timezone_normalization(self):
        """Test that yfinance data gets properly normalized."""
        from src.neural_quant.data.yf_loader import load_yf_data
        
        # Load a small amount of data
        data = load_yf_data(['AAPL'], '2023-01-01', '2023-01-10', '1d')
        
        if not data.empty:
            # Check that index is timezone-naive
            assert data.index.tz is None
            # Check that we have data
            assert len(data) > 0
    
    def test_backtest_timezone_handling(self):
        """Test that backtester handles timezone normalization."""
        from src.neural_quant.core.backtest import Backtester
        from src.neural_quant.strategies.momentum import MovingAverageCrossover
        from src.neural_quant.data.yf_loader import load_yf_data
        
        # Load data
        data = load_yf_data(['AAPL'], '2023-01-01', '2023-01-31', '1d')
        
        if not data.empty:
            # Create backtester and strategy
            backtester = Backtester(initial_capital=100000)
            strategy = MovingAverageCrossover(ma_fast=5, ma_slow=10, threshold=0.0)
            
            # Run backtest
            results = backtester.run_backtest(data, strategy)
            
            # Should complete without timezone errors
            assert 'total_return' in results
            assert isinstance(results['total_return'], (int, float))
