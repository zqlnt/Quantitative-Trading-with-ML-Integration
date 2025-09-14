#!/usr/bin/env python3
"""
Neural Quant Strategy Runner

This script demonstrates how to run a trading strategy with the Neural Quant framework.
"""

import sys
import logging
import argparse
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

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Run Neural Quant trading strategy')
    parser.add_argument('--strategy', default='momentum', help='Strategy to run (default: momentum)')
    parser.add_argument('--symbols', nargs='+', default=['AAPL', 'MSFT', 'GOOGL'], help='Symbols to trade (default: AAPL MSFT GOOGL)')
    parser.add_argument('--timeframe', default='1d', help='Timeframe for data (default: 1d)')
    parser.add_argument('--years', type=int, default=1, help='Number of years of data (default: 1)')
    parser.add_argument('--lookback', type=int, default=20, help='Lookback period for strategy (default: 20)')
    parser.add_argument('--ma_fast', type=int, default=10, help='Fast moving average period (default: 10)')
    parser.add_argument('--ma_slow', type=int, default=30, help='Slow moving average period (default: 30)')
    parser.add_argument('--threshold', type=float, default=0.02, help='Signal threshold (default: 0.02)')
    return parser.parse_args()

def main():
    """Run a trading strategy with specified parameters."""
    args = parse_arguments()
    logger = setup_logging()
    
    logger.info("=" * 60)
    logger.info("Neural Quant Strategy Runner")
    logger.info("=" * 60)
    
    try:
        # Calculate date range
        end_date = datetime.now()
        start_date = end_date - timedelta(days=args.years * 365)
        
        logger.info(f"Running {args.strategy} strategy for symbols: {args.symbols}")
        logger.info(f"Timeframe: {args.timeframe}, Period: {start_date.date()} to {end_date.date()}")
        
        # Create data collector
        collector = YFinanceCollector()
        
        # Create strategy based on type
        if args.strategy.lower() == 'momentum':
            strategy = MomentumStrategy({
                'lookback_period': args.lookback,
                'threshold': args.threshold,
                'ma_fast': args.ma_fast,
                'ma_slow': args.ma_slow,
                'min_volume': 1000000,
                'max_positions': 5
            })
        
        # Get historical data
        start_date_str = start_date.strftime("%Y-%m-%d")
        end_date_str = end_date.strftime("%Y-%m-%d")
        
        logger.info(f"Fetching data from {start_date_str} to {end_date_str}")
        
        all_data = []
        for symbol in args.symbols:
            try:
                data = collector.get_historical_data(
                    symbol=symbol,
                    start_date=start_date_str,
                    end_date=end_date_str,
                    timeframe=args.timeframe
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
        results = strategy.run_backtest(combined_data, start_date_str, end_date_str)
        
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
