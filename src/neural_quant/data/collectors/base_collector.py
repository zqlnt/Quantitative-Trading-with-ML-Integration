"""
Base data collector for Neural Quant.

This module provides the base class and interface for all data collectors,
ensuring consistent API across different data sources.
"""

import time
import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union, Tuple
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from pathlib import Path
import pickle
import json

from ...utils.config.config_manager import get_config


class DataCollector(ABC):
    """
    Abstract base class for all data collectors.
    
    This class defines the interface that all data collectors must implement,
    providing common functionality for rate limiting, caching, and error handling.
    """
    
    def __init__(self, name: str, rate_limit: float = 1.0, cache_dir: Optional[str] = None):
        """
        Initialize the data collector.
        
        Args:
            name: Name of the data collector.
            rate_limit: Minimum time between requests in seconds.
            cache_dir: Directory for caching data. If None, uses default.
        """
        self.name = name
        self.rate_limit = rate_limit
        self.last_request_time = 0
        self.logger = logging.getLogger(f"{__name__}.{name}")
        
        # Set up caching
        self.cache_dir = Path(cache_dir or "data/raw")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Configuration
        self.config = get_config()
        
    def _rate_limit_check(self):
        """Check if enough time has passed since the last request."""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        
        if time_since_last < self.rate_limit:
            sleep_time = self.rate_limit - time_since_last
            self.logger.debug(f"Rate limiting: sleeping for {sleep_time:.2f} seconds")
            time.sleep(sleep_time)
        
        self.last_request_time = time.time()
    
    def _get_cache_path(self, symbol: str, timeframe: str, start_date: str, end_date: str) -> Path:
        """
        Get the cache file path for given parameters.
        
        Args:
            symbol: Trading symbol.
            timeframe: Data timeframe.
            start_date: Start date string.
            end_date: End date string.
            
        Returns:
            Path: Path to the cache file.
        """
        filename = f"{symbol}_{timeframe}_{start_date}_{end_date}.pkl"
        return self.cache_dir / filename
    
    def _is_cache_valid(self, cache_path: Path, max_age_hours: int = 24) -> bool:
        """
        Check if cached data is still valid.
        
        Args:
            cache_path: Path to the cache file.
            max_age_hours: Maximum age of cache in hours.
            
        Returns:
            bool: True if cache is valid, False otherwise.
        """
        if not cache_path.exists():
            return False
        
        file_age = time.time() - cache_path.stat().st_mtime
        max_age_seconds = max_age_hours * 3600
        
        return file_age < max_age_seconds
    
    def _load_from_cache(self, cache_path: Path) -> Optional[pd.DataFrame]:
        """
        Load data from cache.
        
        Args:
            cache_path: Path to the cache file.
            
        Returns:
            Optional[pd.DataFrame]: Cached data or None if not available.
        """
        try:
            with open(cache_path, 'rb') as f:
                data = pickle.load(f)
            self.logger.debug(f"Loaded data from cache: {cache_path}")
            return data
        except Exception as e:
            self.logger.warning(f"Failed to load from cache: {e}")
            return None
    
    def _save_to_cache(self, data: pd.DataFrame, cache_path: Path):
        """
        Save data to cache.
        
        Args:
            data: DataFrame to cache.
            cache_path: Path to save the cache file.
        """
        try:
            with open(cache_path, 'wb') as f:
                pickle.dump(data, f)
            self.logger.debug(f"Saved data to cache: {cache_path}")
        except Exception as e:
            self.logger.warning(f"Failed to save to cache: {e}")
    
    def get_historical_data(self, symbol: str, start_date: str, end_date: str,
                           timeframe: str = "1d", use_cache: bool = True,
                           max_cache_age_hours: int = 24) -> pd.DataFrame:
        """
        Get historical data for a symbol.
        
        Args:
            symbol: Trading symbol.
            start_date: Start date in YYYY-MM-DD format.
            end_date: End date in YYYY-MM-DD format.
            timeframe: Data timeframe (1d, 1h, 5m, etc.).
            use_cache: Whether to use cached data if available.
            max_cache_age_hours: Maximum age of cached data in hours.
            
        Returns:
            pd.DataFrame: Historical data with OHLCV columns.
        """
        # Check cache first
        if use_cache:
            cache_path = self._get_cache_path(symbol, timeframe, start_date, end_date)
            if self._is_cache_valid(cache_path, max_cache_age_hours):
                cached_data = self._load_from_cache(cache_path)
                if cached_data is not None:
                    return cached_data
        
        # Rate limiting
        self._rate_limit_check()
        
        # Fetch data from source
        try:
            data = self._fetch_historical_data(symbol, start_date, end_date, timeframe)
            
            # Validate data
            data = self._validate_data(data, symbol, start_date, end_date)
            
            # Save to cache
            if use_cache:
                self._save_to_cache(data, cache_path)
            
            self.logger.info(f"Fetched {len(data)} records for {symbol} from {start_date} to {end_date}")
            return data
            
        except Exception as e:
            self.logger.error(f"Failed to fetch data for {symbol}: {e}")
            raise
    
    def get_latest_data(self, symbol: str, timeframe: str = "1d") -> pd.DataFrame:
        """
        Get the latest data for a symbol.
        
        Args:
            symbol: Trading symbol.
            timeframe: Data timeframe.
            
        Returns:
            pd.DataFrame: Latest data.
        """
        # Rate limiting
        self._rate_limit_check()
        
        try:
            data = self._fetch_latest_data(symbol, timeframe)
            data = self._validate_data(data, symbol)
            
            self.logger.info(f"Fetched latest data for {symbol}: {len(data)} records")
            return data
            
        except Exception as e:
            self.logger.error(f"Failed to fetch latest data for {symbol}: {e}")
            raise
    
    def get_multiple_symbols(self, symbols: List[str], start_date: str, end_date: str,
                           timeframe: str = "1d", use_cache: bool = True) -> Dict[str, pd.DataFrame]:
        """
        Get historical data for multiple symbols.
        
        Args:
            symbols: List of trading symbols.
            start_date: Start date in YYYY-MM-DD format.
            end_date: End date in YYYY-MM-DD format.
            timeframe: Data timeframe.
            use_cache: Whether to use cached data.
            
        Returns:
            Dict[str, pd.DataFrame]: Dictionary mapping symbols to their data.
        """
        results = {}
        
        for symbol in symbols:
            try:
                data = self.get_historical_data(symbol, start_date, end_date, timeframe, use_cache)
                results[symbol] = data
            except Exception as e:
                self.logger.error(f"Failed to fetch data for {symbol}: {e}")
                results[symbol] = pd.DataFrame()  # Empty DataFrame for failed symbols
        
        return results
    
    def _validate_data(self, data: pd.DataFrame, symbol: str, 
                      start_date: Optional[str] = None, end_date: Optional[str] = None) -> pd.DataFrame:
        """
        Validate and clean the fetched data.
        
        Args:
            data: Raw data from the source.
            symbol: Trading symbol.
            start_date: Expected start date.
            end_date: Expected end date.
            
        Returns:
            pd.DataFrame: Validated and cleaned data.
        """
        if data.empty:
            self.logger.warning(f"No data received for {symbol}")
            return data
        
        # Ensure we have the required columns
        required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        missing_columns = [col for col in required_columns if col not in data.columns]
        
        if missing_columns:
            self.logger.error(f"Missing required columns for {symbol}: {missing_columns}")
            raise ValueError(f"Missing required columns: {missing_columns}")
        
        # Convert index to datetime if it's not already
        if not isinstance(data.index, pd.DatetimeIndex):
            try:
                data.index = pd.to_datetime(data.index)
            except Exception as e:
                self.logger.error(f"Failed to convert index to datetime for {symbol}: {e}")
                raise
        
        # Sort by date
        data = data.sort_index()
        
        # Remove any rows with NaN values in critical columns
        initial_rows = len(data)
        data = data.dropna(subset=required_columns)
        removed_rows = initial_rows - len(data)
        
        if removed_rows > 0:
            self.logger.warning(f"Removed {removed_rows} rows with NaN values for {symbol}")
        
        # Validate date range if provided
        if start_date and end_date:
            start_dt = pd.to_datetime(start_date)
            end_dt = pd.to_datetime(end_date)
            
            data = data[(data.index >= start_dt) & (data.index <= end_dt)]
            
            if data.empty:
                self.logger.warning(f"No data in specified date range for {symbol}")
        
        # Validate OHLC relationships
        invalid_ohlc = data[(data['High'] < data['Low']) | 
                           (data['High'] < data['Open']) | 
                           (data['High'] < data['Close']) |
                           (data['Low'] > data['Open']) | 
                           (data['Low'] > data['Close'])]
        
        if not invalid_ohlc.empty:
            self.logger.warning(f"Found {len(invalid_ohlc)} rows with invalid OHLC relationships for {symbol}")
            # Remove invalid rows
            data = data[~data.index.isin(invalid_ohlc.index)]
        
        return data
    
    @abstractmethod
    def _fetch_historical_data(self, symbol: str, start_date: str, end_date: str, 
                              timeframe: str) -> pd.DataFrame:
        """
        Fetch historical data from the data source.
        
        This method must be implemented by subclasses.
        
        Args:
            symbol: Trading symbol.
            start_date: Start date in YYYY-MM-DD format.
            end_date: End date in YYYY-MM-DD format.
            timeframe: Data timeframe.
            
        Returns:
            pd.DataFrame: Historical data with OHLCV columns.
        """
        pass
    
    @abstractmethod
    def _fetch_latest_data(self, symbol: str, timeframe: str) -> pd.DataFrame:
        """
        Fetch the latest data from the data source.
        
        This method must be implemented by subclasses.
        
        Args:
            symbol: Trading symbol.
            timeframe: Data timeframe.
            
        Returns:
            pd.DataFrame: Latest data.
        """
        pass
    
    def get_info(self) -> Dict[str, Any]:
        """
        Get information about the data collector.
        
        Returns:
            Dict[str, Any]: Information about the collector.
        """
        return {
            "name": self.name,
            "rate_limit": self.rate_limit,
            "cache_dir": str(self.cache_dir),
            "last_request_time": self.last_request_time
        }
    
    def clear_cache(self, symbol: Optional[str] = None):
        """
        Clear cached data.
        
        Args:
            symbol: Specific symbol to clear cache for. If None, clears all cache.
        """
        if symbol:
            # Clear cache for specific symbol
            pattern = f"{symbol}_*.pkl"
            for cache_file in self.cache_dir.glob(pattern):
                cache_file.unlink()
                self.logger.info(f"Cleared cache for {symbol}: {cache_file}")
        else:
            # Clear all cache
            for cache_file in self.cache_dir.glob("*.pkl"):
                cache_file.unlink()
                self.logger.info(f"Cleared cache file: {cache_file}")
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics.
        
        Returns:
            Dict[str, Any]: Cache statistics.
        """
        cache_files = list(self.cache_dir.glob("*.pkl"))
        total_size = sum(f.stat().st_size for f in cache_files)
        
        return {
            "cache_dir": str(self.cache_dir),
            "total_files": len(cache_files),
            "total_size_bytes": total_size,
            "total_size_mb": total_size / (1024 * 1024)
        }
