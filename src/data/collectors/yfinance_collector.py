"""
YFinance data collector for Neural Quant.

This module provides data collection from Yahoo Finance using the yfinance library.
"""

import yfinance as yf
import pandas as pd
import logging
from typing import Optional
from datetime import datetime, timedelta

from .base_collector import DataCollector
from ...utils.config.config_manager import get_config, get_data_source_config


class YFinanceCollector(DataCollector):
    """
    Data collector for Yahoo Finance data.
    
    This collector fetches historical and real-time data from Yahoo Finance
    using the yfinance library.
    """
    
    def __init__(self, rate_limit: Optional[float] = None):
        """
        Initialize the YFinance collector.
        
        Args:
            rate_limit: Rate limit in seconds. If None, uses config default.
        """
        config = get_data_source_config("yfinance")
        if config is None:
            raise ValueError("YFinance configuration not found")
        
        if not config.enabled:
            raise ValueError("YFinance data source is not enabled")
        
        rate_limit = rate_limit or config.rate_limit
        
        super().__init__(
            name="yfinance",
            rate_limit=rate_limit,
            cache_dir="data/raw/yfinance"
        )
        
        self.logger = logging.getLogger(f"{__name__}.YFinanceCollector")
    
    def _fetch_historical_data(self, symbol: str, start_date: str, end_date: str, 
                              timeframe: str) -> pd.DataFrame:
        """
        Fetch historical data from Yahoo Finance.
        
        Args:
            symbol: Trading symbol.
            start_date: Start date in YYYY-MM-DD format.
            end_date: End date in YYYY-MM-DD format.
            timeframe: Data timeframe (1d, 1h, 5m, etc.).
            
        Returns:
            pd.DataFrame: Historical data with OHLCV columns.
        """
        try:
            # Create ticker object
            ticker = yf.Ticker(symbol)
            
            # Map timeframe to yfinance interval
            interval_map = {
                "1m": "1m",
                "2m": "2m", 
                "5m": "5m",
                "15m": "15m",
                "30m": "30m",
                "60m": "60m",
                "90m": "90m",
                "1h": "1h",
                "1d": "1d",
                "5d": "5d",
                "1wk": "1wk",
                "1mo": "1mo",
                "3mo": "3mo"
            }
            
            interval = interval_map.get(timeframe, "1d")
            
            # Fetch data
            data = ticker.history(
                start=start_date,
                end=end_date,
                interval=interval,
                auto_adjust=True,
                prepost=True,
                threads=True
            )
            
            if data.empty:
                self.logger.warning(f"No data found for {symbol} from {start_date} to {end_date}")
                return data
            
            # Rename columns to standard format
            data.columns = data.columns.str.title()
            
            # Ensure we have the required columns
            required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
            for col in required_columns:
                if col not in data.columns:
                    self.logger.error(f"Missing required column {col} for {symbol}")
                    raise ValueError(f"Missing required column: {col}")
            
            # Remove any extra columns we don't need
            data = data[required_columns]
            
            # Add symbol column
            data['Symbol'] = symbol
            
            self.logger.debug(f"Fetched {len(data)} records for {symbol}")
            return data
            
        except Exception as e:
            self.logger.error(f"Failed to fetch historical data for {symbol}: {e}")
            raise
    
    def _fetch_latest_data(self, symbol: str, timeframe: str) -> pd.DataFrame:
        """
        Fetch the latest data from Yahoo Finance.
        
        Args:
            symbol: Trading symbol.
            timeframe: Data timeframe.
            
        Returns:
            pd.DataFrame: Latest data.
        """
        try:
            # Get data for the last 5 days to ensure we have recent data
            end_date = datetime.now()
            start_date = end_date - timedelta(days=5)
            
            return self._fetch_historical_data(
                symbol, 
                start_date.strftime("%Y-%m-%d"), 
                end_date.strftime("%Y-%m-%d"), 
                timeframe
            )
            
        except Exception as e:
            self.logger.error(f"Failed to fetch latest data for {symbol}: {e}")
            raise
    
    def get_company_info(self, symbol: str) -> dict:
        """
        Get company information for a symbol.
        
        Args:
            symbol: Trading symbol.
            
        Returns:
            dict: Company information.
        """
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            
            # Extract relevant information
            company_info = {
                "symbol": symbol,
                "name": info.get("longName", ""),
                "sector": info.get("sector", ""),
                "industry": info.get("industry", ""),
                "market_cap": info.get("marketCap", 0),
                "employees": info.get("fullTimeEmployees", 0),
                "website": info.get("website", ""),
                "description": info.get("longBusinessSummary", ""),
                "country": info.get("country", ""),
                "currency": info.get("currency", "USD"),
                "exchange": info.get("exchange", ""),
                "quote_type": info.get("quoteType", ""),
                "shares_outstanding": info.get("sharesOutstanding", 0),
                "float_shares": info.get("floatShares", 0),
                "regular_market_price": info.get("regularMarketPrice", 0),
                "previous_close": info.get("previousClose", 0),
                "open": info.get("open", 0),
                "day_low": info.get("dayLow", 0),
                "day_high": info.get("dayHigh", 0),
                "volume": info.get("volume", 0),
                "avg_volume": info.get("averageVolume", 0),
                "pe_ratio": info.get("trailingPE", 0),
                "forward_pe": info.get("forwardPE", 0),
                "peg_ratio": info.get("pegRatio", 0),
                "price_to_book": info.get("priceToBook", 0),
                "price_to_sales": info.get("priceToSalesTrailing12Months", 0),
                "dividend_yield": info.get("dividendYield", 0),
                "beta": info.get("beta", 0),
                "52_week_low": info.get("fiftyTwoWeekLow", 0),
                "52_week_high": info.get("fiftyTwoWeekHigh", 0),
                "last_updated": datetime.now().isoformat()
            }
            
            return company_info
            
        except Exception as e:
            self.logger.error(f"Failed to get company info for {symbol}: {e}")
            return {}
    
    def get_dividends(self, symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
        """
        Get dividend data for a symbol.
        
        Args:
            symbol: Trading symbol.
            start_date: Start date in YYYY-MM-DD format.
            end_date: End date in YYYY-MM-DD format.
            
        Returns:
            pd.DataFrame: Dividend data.
        """
        try:
            ticker = yf.Ticker(symbol)
            dividends = ticker.dividends
            
            if dividends.empty:
                return pd.DataFrame()
            
            # Filter by date range
            start_dt = pd.to_datetime(start_date)
            end_dt = pd.to_datetime(end_date)
            dividends = dividends[(dividends.index >= start_dt) & (dividends.index <= end_dt)]
            
            # Convert to DataFrame
            df = pd.DataFrame({
                'Date': dividends.index,
                'Dividend': dividends.values,
                'Symbol': symbol
            })
            
            return df
            
        except Exception as e:
            self.logger.error(f"Failed to get dividends for {symbol}: {e}")
            return pd.DataFrame()
    
    def get_splits(self, symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
        """
        Get stock split data for a symbol.
        
        Args:
            symbol: Trading symbol.
            start_date: Start date in YYYY-MM-DD format.
            end_date: End date in YYYY-MM-DD format.
            
        Returns:
            pd.DataFrame: Stock split data.
        """
        try:
            ticker = yf.Ticker(symbol)
            splits = ticker.splits
            
            if splits.empty:
                return pd.DataFrame()
            
            # Filter by date range
            start_dt = pd.to_datetime(start_date)
            end_dt = pd.to_datetime(end_date)
            splits = splits[(splits.index >= start_dt) & (splits.index <= end_dt)]
            
            # Convert to DataFrame
            df = pd.DataFrame({
                'Date': splits.index,
                'Split': splits.values,
                'Symbol': symbol
            })
            
            return df
            
        except Exception as e:
            self.logger.error(f"Failed to get splits for {symbol}: {e}")
            return pd.DataFrame()
    
    def search_symbols(self, query: str) -> list:
        """
        Search for symbols matching a query.
        
        Args:
            query: Search query.
            
        Returns:
            list: List of matching symbols.
        """
        try:
            # This is a simplified search - in practice, you might want to use
            # a more sophisticated search API or maintain a symbol database
            ticker = yf.Ticker(query)
            info = ticker.info
            
            if info and 'symbol' in info:
                return [info['symbol']]
            else:
                return []
                
        except Exception as e:
            self.logger.error(f"Failed to search symbols for query '{query}': {e}")
            return []
    
    def is_market_open(self) -> bool:
        """
        Check if the market is currently open.
        
        Returns:
            bool: True if market is open, False otherwise.
        """
        try:
            # Use a major index to check market status
            ticker = yf.Ticker("^GSPC")  # S&P 500
            info = ticker.info
            
            # Check if we have recent data (within last few minutes)
            if 'regularMarketTime' in info:
                market_time = pd.to_datetime(info['regularMarketTime'], unit='s')
                current_time = datetime.now()
                time_diff = (current_time - market_time).total_seconds()
                
                # If data is less than 5 minutes old, market is likely open
                return time_diff < 300
            
            return False
            
        except Exception as e:
            self.logger.error(f"Failed to check market status: {e}")
            return False
    
    def get_market_calendar(self, year: int) -> pd.DataFrame:
        """
        Get market calendar for a given year.
        
        Args:
            year: Year to get calendar for.
            
        Returns:
            pd.DataFrame: Market calendar with holidays and special dates.
        """
        try:
            # This is a simplified implementation
            # In practice, you might want to use a more comprehensive calendar API
            import pandas_market_calendars as mcal
            
            # Get NYSE calendar
            nyse = mcal.get_calendar('NYSE')
            calendar = nyse.schedule(start_date=f'{year}-01-01', end_date=f'{year}-12-31')
            
            # Convert to DataFrame
            df = pd.DataFrame({
                'Date': calendar.index,
                'Market_Open': calendar['market_open'],
                'Market_Close': calendar['market_close'],
                'Is_Trading_Day': True
            })
            
            return df
            
        except ImportError:
            self.logger.warning("pandas_market_calendars not available. Install with: pip install pandas_market_calendars")
            return pd.DataFrame()
        except Exception as e:
            self.logger.error(f"Failed to get market calendar for {year}: {e}")
            return pd.DataFrame()
