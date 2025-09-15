"""Time utilities for handling timezone normalization in trading data."""

import pandas as pd
from typing import Union, Optional, Tuple
import logging

logger = logging.getLogger(__name__)

def ensure_tz_naive_daily_index(series: pd.Series, 
                               market: str = "US",
                               tz: Optional[str] = None) -> pd.Series:
    """
    Ensure a price series has a timezone-naive daily index.
    
    For daily bars:
    - Convert to market timezone if needed
    - Normalize to market close time
    - Convert to timezone-naive
    
    Args:
        series: Price series with datetime index
        market: Market identifier ("US", "EU", "ASIA")
        tz: Override timezone (if None, uses market default)
        
    Returns:
        Series with timezone-naive daily index
    """
    if series.empty:
        return series
    
    # Market timezone defaults
    market_tz = {
        "US": "America/New_York",
        "EU": "Europe/London", 
        "ASIA": "Asia/Tokyo"
    }
    
    target_tz = tz or market_tz.get(market, "America/New_York")
    
    # If already timezone-naive, assume it's in the target timezone
    if series.index.tz is None:
        series = series.copy()
        series.index = series.index.tz_localize(target_tz)
    
    # Convert to target timezone
    if series.index.tz != target_tz:
        series = series.tz_convert(target_tz)
    
    # Normalize to market close time (4 PM for US markets)
    if market == "US":
        series.index = series.index.normalize() + pd.Timedelta(hours=16)
    else:
        # For other markets, normalize to end of day
        series.index = series.index.normalize() + pd.Timedelta(hours=23, minutes=59)
    
    # Convert to timezone-naive
    series.index = series.index.tz_localize(None)
    
    return series

def clamp_to_dates(series: pd.Series, 
                   start_date: Optional[str] = None,
                   end_date: Optional[str] = None) -> pd.Series:
    """
    Clamp a series to the specified date range.
    
    Args:
        series: Price series with datetime index
        start_date: Start date string (YYYY-MM-DD)
        end_date: End date string (YYYY-MM-DD)
        
    Returns:
        Clamped series
    """
    if series.empty:
        return series
    
    if start_date:
        start_dt = pd.to_datetime(start_date)
        series = series[series.index >= start_dt]
    
    if end_date:
        end_dt = pd.to_datetime(end_date)
        series = series[series.index <= end_dt]
    
    return series

def normalize_intraday_index(series: pd.Series, 
                           target_tz: str = "UTC") -> pd.Series:
    """
    Normalize intraday data to UTC timezone-naive.
    
    Args:
        series: Price series with datetime index
        target_tz: Target timezone (default: UTC)
        
    Returns:
        Series with UTC timezone-naive index
    """
    if series.empty:
        return series
    
    # If already timezone-naive, assume it's in the target timezone
    if series.index.tz is None:
        series = series.copy()
        series.index = series.index.tz_localize(target_tz)
    
    # Convert to target timezone
    if series.index.tz != target_tz:
        series = series.tz_convert(target_tz)
    
    # Convert to timezone-naive
    series.index = series.index.tz_localize(None)
    
    return series

def is_daily_data(series: pd.Series) -> bool:
    """
    Check if a series represents daily data based on index frequency.
    
    Args:
        series: Price series with datetime index
        
    Returns:
        True if daily data, False otherwise
    """
    if series.empty or len(series) < 2:
        return True  # Assume daily if insufficient data
    
    # Calculate time differences
    time_diffs = series.index.to_series().diff().dropna()
    
    # Check if most differences are around 1 day
    daily_threshold = pd.Timedelta(hours=20)  # Allow some flexibility
    daily_count = (time_diffs >= daily_threshold).sum()
    
    return daily_count / len(time_diffs) > 0.8  # 80% of intervals are daily

def get_market_calendar_info(market: str = "US") -> dict:
    """
    Get market calendar information.
    
    Args:
        market: Market identifier
        
    Returns:
        Dictionary with market calendar info
    """
    calendars = {
        "US": {
            "timezone": "America/New_York",
            "open_time": "09:30",
            "close_time": "16:00",
            "trading_days": ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
        },
        "EU": {
            "timezone": "Europe/London", 
            "open_time": "08:00",
            "close_time": "16:30",
            "trading_days": ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
        },
        "ASIA": {
            "timezone": "Asia/Tokyo",
            "open_time": "09:00", 
            "close_time": "15:00",
            "trading_days": ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
        }
    }
    
    return calendars.get(market, calendars["US"])
