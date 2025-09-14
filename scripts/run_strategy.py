#!/usr/bin/env python3
"""
Neural Quant Strategy Runner

This script demonstrates how to run a trading strategy with the Neural Quant framework.
"""

import sys
import logging
from pathlib import Path
from datetime import datetime, timedelta
import pandas as pd

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from data.collectors.yfinance_collector import YFinanceCollector
from strategies.baselines.momentum_strategy import MomentumStrategy
from logging.mlflow.mlflow_manager import get_mlflow_manager

def setup_logging():
    """Set up logging."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)

def main():
    """Run a momentum strategy example."""
    logger = setup_logging()
    
    logger.info("=" * 60)
    logger.info("Neural Quant Strategy Runner")
    logger.info("=" * 60)
    
    try:
        # Get symbols from config
        from utils.config.config_manager import get_config
        config = get_config()
        symbols = config.trading.default_symbols[:3]  # Use first 3 symbols
        
        logger.info(f"Running strategy for symbols: {symbols}")
        
        # Create data collector
        collector = YFinanceCollector()
        
        # Create strategy
        strategy = MomentumStrategy({
            'lookback_period': 20,
            'threshold': 0.02,
            'min_volume': 1000000,
            'max_positions': 5
        })
        
        # Get historical data
        end_date = datetime.now().strftime("%Y-%m-%d")
        start_date = (datetime.now() - timedelta(days=90)).strftime("%Y-%m-%d")
        
        logger.info(f"Fetching data from {start_date} to {end_date}")
        
        all_data = []
        for symbol in symbols:
            try:
                data = collector.get_historical_data(
                    symbol=symbol,
                    start_date=start_date,
                    end_date=end_date,
                    timeframe="1d"
                )
                if not data.empty:
                    all_data.append(data)
                    logger.info(f"  ✓ {symbol}: {len(data)} records")
                else:
                    logger.warning(f"  ✗ {symbol}: No data")
            except Exception as e:
                logger.error(f"  ✗ {symbol}: {e}")
        
        if not all_data:
            logger.error("No data available for any symbols")
            return 1
        
        # Combine all data
        combined_data = pd.concat(all_data, ignore_index=False)
        combined_data = combined_data.sort_index()
        
        logger.info(f"Total data points: {len(combined_data)}")
        
        # Run backtest
        logger.info("Running backtest...")
        results = strategy.run_backtest(combined_data, start_date, end_date)
        
        if not results:
            logger.error("Backtest failed")
            return 1
        
        # Display results
        logger.info("\n" + "=" * 60)
        logger.info("BACKTEST RESULTS")
        logger.info("=" * 60)
        
        logger.info(f"Strategy: {results['strategy_name']}")
        logger.info(f"Period: {results['start_date']} to {results['end_date']}")
        logger.info(f"Total days: {results['total_days']}")
        logger.info(f"Total trades: {len(results.get('trades', []))}")
        logger.info(f"Total signals: {len(results.get('signals', []))}")
        
        if 'performance_metrics' in results and results['performance_metrics']:
            metrics = results['performance_metrics']
            logger.info("\nPerformance Metrics:")
            logger.info(f"  Total Return: {metrics.get('total_return', 0):.2%}")
            logger.info(f"  Net P&L: ${metrics.get('net_pnl', 0):.2f}")
            logger.info(f"  Win Rate: {metrics.get('win_rate', 0):.2%}")
            logger.info(f"  Sharpe Ratio: {metrics.get('sharpe_ratio', 0):.2f}")
            logger.info(f"  Total Trades: {metrics.get('total_trades', 0)}")
            logger.info(f"  Current Portfolio Value: ${metrics.get('current_portfolio_value', 0):.2f}")
        
        # Show recent trades
        if 'trades' in results and results['trades']:
            logger.info(f"\nRecent Trades ({min(5, len(results['trades']))}):")
            for trade in results['trades'][-5:]:
                logger.info(f"  {trade.timestamp.strftime('%Y-%m-%d')} | {trade.symbol} | {trade.side} | {trade.quantity} @ ${trade.price:.2f}")
        
        logger.info("\n" + "=" * 60)
        logger.info("Strategy run completed successfully!")
        logger.info("Check MLflow UI for detailed results: mlflow ui")
        
        return 0
        
    except Exception as e:
        logger.error(f"Strategy run failed: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
