#!/usr/bin/env python3
"""Run trading strategies with command-line interface."""

import argparse
import sys
import logging
from datetime import datetime, timedelta
from pathlib import Path

# Add the src directory to the path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from neural_quant.data.yf_loader import load_yf_data
from neural_quant.strategies.momentum import MovingAverageCrossover
from neural_quant.core.backtest import Backtester
from neural_quant.core.metrics import calculate_performance_metrics

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

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
    parser.add_argument('--threshold', type=float, default=0.0, help='Signal threshold (default: 0.0)')
    parser.add_argument('--initial_capital', type=float, default=100000.0, help='Initial capital (default: 100000)')
    parser.add_argument('--commission', type=float, default=0.001, help='Commission rate (default: 0.001)')
    parser.add_argument('--slippage', type=float, default=0.0005, help='Slippage rate (default: 0.0005)')
    return parser.parse_args()

def main():
    """Main function."""
    args = parse_arguments()
    
    logger.info("=" * 80)
    logger.info("NEURAL QUANT - STRATEGY RUNNER")
    logger.info("=" * 80)
    logger.info(f"Strategy: {args.strategy}")
    logger.info(f"Symbols: {args.symbols}")
    logger.info(f"Timeframe: {args.timeframe}")
    logger.info(f"Years: {args.years}")
    logger.info(f"MA Fast: {args.ma_fast}")
    logger.info(f"MA Slow: {args.ma_slow}")
    logger.info(f"Threshold: {args.threshold}")
    logger.info("")
    
    # Calculate date range
    end_date = datetime.now()
    start_date = end_date - timedelta(days=args.years * 365)
    start_date_str = start_date.strftime('%Y-%m-%d')
    end_date_str = end_date.strftime('%Y-%m-%d')
    
    logger.info(f"Date range: {start_date_str} to {end_date_str}")
    logger.info("")
    
    try:
        # Load data
        logger.info("Loading market data...")
        data = load_yf_data(args.symbols, start_date_str, end_date_str, args.timeframe)
        
        if data.empty:
            logger.error("No data loaded. Exiting.")
            return 1
        
        logger.info(f"✓ Loaded {len(data)} records")
        logger.info("")
        
        # Initialize strategy
        if args.strategy.lower() == 'momentum':
            strategy = MovingAverageCrossover(
                ma_fast=args.ma_fast,
                ma_slow=args.ma_slow,
                threshold=args.threshold,
                min_volume=1000000,
                max_positions=5
            )
        else:
            logger.error(f"Unknown strategy: {args.strategy}")
            return 1
        
        # Initialize backtester
        backtester = Backtester(
            initial_capital=args.initial_capital,
            commission=args.commission,
            slippage=args.slippage
        )
        
        # Run backtest
        logger.info("Running backtest...")
        results = backtester.run_backtest(data, strategy, start_date_str, end_date_str)
        
        # Display results
        logger.info("=" * 80)
        logger.info("BACKTEST RESULTS")
        logger.info("=" * 80)
        logger.info(f"Total Return: {results.get('total_return', 0):.2%}")
        logger.info(f"Annualized Return: {results.get('annualized_return', 0):.2%}")
        logger.info(f"Volatility: {results.get('volatility', 0):.2%}")
        logger.info(f"Sharpe Ratio: {results.get('sharpe_ratio', 0):.2f}")
        logger.info(f"Max Drawdown: {results.get('max_drawdown', 0):.2%}")
        logger.info(f"Total Trades: {results.get('total_trades', 0)}")
        logger.info(f"Win Rate: {results.get('win_rate', 0):.2%}")
        logger.info("")
        
        # Show recent trades
        if results.get('trades'):
            logger.info("Recent trades:")
            for trade in results['trades'][-5:]:
                logger.info(f"  {trade['symbol']}: {trade['entry_date']} -> {trade['exit_date']} | P&L: {trade['pnl']:.2f}")
        
        logger.info("=" * 80)
        logger.info("BACKTEST COMPLETED SUCCESSFULLY!")
        logger.info("=" * 80)
        
        return 0
        
    except Exception as e:
        logger.error(f"Error running strategy: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
