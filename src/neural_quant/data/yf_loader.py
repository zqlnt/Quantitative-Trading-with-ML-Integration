"""Yahoo Finance data loader."""

import yfinance as yf
import pandas as pd
from typing import List, Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)

def load_yf_data(symbols: List[str], 
                start_date: str, 
                end_date: str,
                timeframe: str = '1d') -> pd.DataFrame:
    """
    Load data from Yahoo Finance for multiple symbols.
    
    Args:
        symbols: List of symbols to load
        start_date: Start date in YYYY-MM-DD format
        end_date: End date in YYYY-MM-DD format
        timeframe: Data timeframe (1d, 1h, etc.)
        
    Returns:
        DataFrame with OHLCV data for all symbols
    """
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
