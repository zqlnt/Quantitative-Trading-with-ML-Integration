"""Yahoo Finance data loader."""

import yfinance as yf
import pandas as pd
from typing import List, Optional, Dict, Any
import logging
from ..utils.time_utils import ensure_tz_naive_daily_index, clamp_to_dates, is_daily_data

logger = logging.getLogger(__name__)

def load_yf_data(symbols: List[str], 
                start_date: Optional[str] = None, 
                end_date: Optional[str] = None,
                timeframe: str = '1d',
                days_back: Optional[int] = None) -> pd.DataFrame:
    """
    Load data from Yahoo Finance for multiple symbols.
    
    Args:
        symbols: List of symbols to load
        start_date: Start date in YYYY-MM-DD format (optional if days_back provided)
        end_date: End date in YYYY-MM-DD format (optional if days_back provided)
        timeframe: Data timeframe (1d, 1h, etc.)
        days_back: Number of days back from today (alternative to start_date/end_date)
        
    Returns:
        DataFrame with OHLCV data for all symbols
    """
    # Handle days_back parameter
    if days_back is not None:
        from datetime import datetime, timedelta
        end_date = datetime.now().strftime('%Y-%m-%d')
        
        if days_back == "ytd":
            # Year to date
            start_date = datetime.now().replace(month=1, day=1).strftime('%Y-%m-%d')
        else:
            # Regular days back
            start_date = (datetime.now() - timedelta(days=days_back)).strftime('%Y-%m-%d')
    
    if not start_date or not end_date:
        raise ValueError("Either start_date/end_date or days_back must be provided")
    
    logger.info(f"Loading data for {symbols} from {start_date} to {end_date}")
    
    all_data = []
    
    for symbol in symbols:
        try:
            ticker = yf.Ticker(symbol)
            data = ticker.history(start=start_date, end=end_date, interval=timeframe)
            
            if data.empty:
                logger.warning(f"No data found for {symbol}")
                continue
            
            # Normalize column names
            data.columns = [col.lower() for col in data.columns]
            
            # Add symbol prefix to columns
            data.columns = [f"{symbol}_{col}" if col != 'date' else col for col in data.columns]
            
            # Add symbol column
            data['symbol'] = symbol
            
            # Normalize timezone for daily data
            if timeframe == '1d' or is_daily_data(data):
                data = ensure_tz_naive_daily_index(data, market="US")
            else:
                # For intraday data, normalize to UTC
                from ..utils.time_utils import normalize_intraday_index
                data = normalize_intraday_index(data, target_tz="UTC")
            
            # Clamp to date range
            data = clamp_to_dates(data, start_date, end_date)
            
            all_data.append(data)
            logger.info(f"Loaded {len(data)} records for {symbol}")
            
        except Exception as e:
            logger.error(f"Error loading data for {symbol}: {e}")
            continue
    
    if not all_data:
        logger.error("No data loaded for any symbols")
        return pd.DataFrame()
    
    # Combine all data
    combined_data = pd.concat(all_data, axis=1, sort=True)
    
    # Forward fill missing values
    combined_data = combined_data.ffill()
    
    logger.info(f"Combined data shape: {combined_data.shape}")
    return combined_data

def get_symbol_info(symbol: str) -> Dict[str, Any]:
    """
    Get basic information about a symbol.
    
    Args:
        symbol: Symbol to get info for
        
    Returns:
        Dictionary with symbol information
    """
    try:
        ticker = yf.Ticker(symbol)
        info = ticker.info
        return {
            'symbol': symbol,
            'name': info.get('longName', ''),
            'sector': info.get('sector', ''),
            'industry': info.get('industry', ''),
            'market_cap': info.get('marketCap', 0),
            'currency': info.get('currency', 'USD')
        }
    except Exception as e:
        logger.error(f"Error getting info for {symbol}: {e}")
        return {'symbol': symbol, 'error': str(e)}
