#!/usr/bin/env python3
"""
Neural Quant Setup Verification Script

This script verifies that the Neural Quant development environment is properly
set up and all components are working correctly.
"""

import sys
import os
import logging
from pathlib import Path
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

def setup_logging():
    """Set up logging for the verification script."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)

def test_imports():
    """Test that all required modules can be imported."""
    logger = logging.getLogger(__name__)
    logger.info("Testing imports...")
    
    try:
        # Test core imports
        import pandas as pd
        import numpy as np
        import yfinance as yf
        import mlflow
        import torch
        import sklearn
        logger.info("✓ Core libraries imported successfully")
        
        # Test Neural Quant imports
        from utils.config.config_manager import get_config, ConfigManager
        from data.collectors.yfinance_collector import YFinanceCollector
        from strategies.baselines.momentum_strategy import MomentumStrategy
        from logging.mlflow.mlflow_manager import get_mlflow_manager
        from logging.trades.trade_logger import get_trade_logger
        logger.info("✓ Neural Quant modules imported successfully")
        
        return True
        
    except ImportError as e:
        logger.error(f"✗ Import failed: {e}")
        return False

def test_configuration():
    """Test configuration system."""
    logger = logging.getLogger(__name__)
    logger.info("Testing configuration system...")
    
    try:
        from utils.config.config_manager import get_config
        
        config = get_config()
        
        # Test basic config properties
        assert hasattr(config, 'environment')
        assert hasattr(config, 'mlflow')
        assert hasattr(config, 'trading')
        assert hasattr(config, 'data_sources')
        
        logger.info(f"✓ Configuration loaded successfully")
        logger.info(f"  Environment: {config.environment}")
        logger.info(f"  MLflow tracking URI: {config.mlflow.tracking_uri}")
        logger.info(f"  Default symbols: {config.trading.default_symbols}")
        
        return True
        
    except Exception as e:
        logger.error(f"✗ Configuration test failed: {e}")
        return False

def test_mlflow():
    """Test MLflow integration."""
    logger = logging.getLogger(__name__)
    logger.info("Testing MLflow integration...")
    
    try:
        from logging.mlflow.mlflow_manager import get_mlflow_manager
        
        mlflow_manager = get_mlflow_manager()
        
        # Test starting a run
        with mlflow_manager.start_run(run_name="verification_test"):
            mlflow_manager.log_parameters({"test_param": "test_value"})
            mlflow_manager.log_metrics({"test_metric": 0.95})
            
        logger.info("✓ MLflow integration working correctly")
        return True
        
    except Exception as e:
        logger.error(f"✗ MLflow test failed: {e}")
        return False

def test_data_collector():
    """Test data collector functionality."""
    logger = logging.getLogger(__name__)
    logger.info("Testing data collector...")
    
    try:
        from data.collectors.yfinance_collector import YFinanceCollector
        
        collector = YFinanceCollector()
        
        # Test fetching data for a simple symbol
        end_date = datetime.now().strftime("%Y-%m-%d")
        start_date = (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")
        
        data = collector.get_historical_data(
            symbol="AAPL",
            start_date=start_date,
            end_date=end_date,
            timeframe="1d"
        )
        
        assert not data.empty, "No data returned"
        assert 'Close' in data.columns, "Missing Close column"
        assert 'Volume' in data.columns, "Missing Volume column"
        assert len(data) > 0, "Empty data returned"
        
        logger.info(f"✓ Data collector working correctly")
        logger.info(f"  Fetched {len(data)} records for AAPL")
        logger.info(f"  Date range: {data.index[0]} to {data.index[-1]}")
        
        return True
        
    except Exception as e:
        logger.error(f"✗ Data collector test failed: {e}")
        return False

def test_strategy():
    """Test strategy functionality."""
    logger = logging.getLogger(__name__)
    logger.info("Testing strategy system...")
    
    try:
        from strategies.baselines.momentum_strategy import MomentumStrategy
        from data.collectors.yfinance_collector import YFinanceCollector
        
        # Create strategy
        strategy = MomentumStrategy({
            'lookback_period': 10,
            'threshold': 0.01,
            'min_volume': 1000000,
            'max_positions': 5
        })
        
        # Get some test data
        collector = YFinanceCollector()
        end_date = datetime.now().strftime("%Y-%m-%d")
        start_date = (datetime.now() - timedelta(days=60)).strftime("%Y-%m-%d")
        
        data = collector.get_historical_data(
            symbol="AAPL",
            start_date=start_date,
            end_date=end_date,
            timeframe="1d"
        )
        
        # Add Symbol column if not present
        if 'Symbol' not in data.columns:
            data['Symbol'] = 'AAPL'
        
        # Initialize strategy
        success = strategy.initialize(data)
        assert success, "Strategy initialization failed"
        
        # Test signal generation
        signals = strategy.generate_signals(data.tail(5))
        logger.info(f"  Generated {len(signals)} signals")
        
        # Test strategy info
        info = strategy.get_strategy_info()
        assert 'name' in info, "Missing strategy name in info"
        assert 'is_initialized' in info, "Missing initialization status in info"
        
        logger.info("✓ Strategy system working correctly")
        return True
        
    except Exception as e:
        logger.error(f"✗ Strategy test failed: {e}")
        return False

def test_trade_logging():
    """Test trade logging functionality."""
    logger = logging.getLogger(__name__)
    logger.info("Testing trade logging...")
    
    try:
        from logging.trades.trade_logger import get_trade_logger
        
        trade_logger = get_trade_logger()
        
        # Test logging a trade
        trade_logger.log_trade(
            strategy_name="test_strategy",
            symbol="AAPL",
            side="BUY",
            quantity=100,
            price=150.0,
            commission=0.0
        )
        
        # Test logging a signal
        trade_logger.log_signal(
            strategy_name="test_strategy",
            symbol="AAPL",
            signal_type="BUY",
            strength=0.8,
            price=150.0
        )
        
        # Test logging performance metrics
        trade_logger.log_performance_metric(
            strategy_name="test_strategy",
            metric_name="total_return",
            metric_value=0.15
        )
        
        logger.info("✓ Trade logging working correctly")
        return True
        
    except Exception as e:
        logger.error(f"✗ Trade logging test failed: {e}")
        return False

def test_backtest():
    """Test backtesting functionality."""
    logger = logging.getLogger(__name__)
    logger.info("Testing backtesting...")
    
    try:
        from strategies.baselines.momentum_strategy import MomentumStrategy
        from data.collectors.yfinance_collector import YFinanceCollector
        
        # Create strategy
        strategy = MomentumStrategy({
            'lookback_period': 5,
            'threshold': 0.01,
            'min_volume': 1000000,
            'max_positions': 3
        })
        
        # Get test data
        collector = YFinanceCollector()
        end_date = datetime.now().strftime("%Y-%m-%d")
        start_date = (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")
        
        data = collector.get_historical_data(
            symbol="AAPL",
            start_date=start_date,
            end_date=end_date,
            timeframe="1d"
        )
        
        # Add Symbol column if not present
        if 'Symbol' not in data.columns:
            data['Symbol'] = 'AAPL'
        
        # Run backtest
        results = strategy.run_backtest(data, start_date, end_date)
        
        assert 'strategy_name' in results, "Missing strategy name in results"
        assert 'performance_metrics' in results, "Missing performance metrics in results"
        
        logger.info("✓ Backtesting working correctly")
        logger.info(f"  Strategy: {results['strategy_name']}")
        logger.info(f"  Total trades: {len(results.get('trades', []))}")
        logger.info(f"  Total signals: {len(results.get('signals', []))}")
        
        if 'performance_metrics' in results and results['performance_metrics']:
            metrics = results['performance_metrics']
            logger.info(f"  Total return: {metrics.get('total_return', 0):.2%}")
            logger.info(f"  Total trades: {metrics.get('total_trades', 0)}")
        
        return True
        
    except Exception as e:
        logger.error(f"✗ Backtesting test failed: {e}")
        return False

def test_directory_structure():
    """Test that the directory structure is correct."""
    logger = logging.getLogger(__name__)
    logger.info("Testing directory structure...")
    
    base_path = Path(__file__).parent.parent
    
    required_dirs = [
        "src/data/collectors",
        "src/data/processors", 
        "src/data/feeds",
        "src/models/baselines",
        "src/models/neural",
        "src/models/hybrid",
        "src/strategies/base",
        "src/strategies/signal",
        "src/strategies/portfolio",
        "src/execution/brokers",
        "src/execution/orders",
        "src/execution/paper",
        "src/risk/monitors",
        "src/risk/controls",
        "src/risk/attribution",
        "src/logging/mlflow",
        "src/logging/trades",
        "src/logging/metrics",
        "src/utils/config",
        "src/utils/helpers",
        "src/utils/visualization",
        "experiments",
        "data/raw",
        "data/processed",
        "data/live",
        "notebooks",
        "configs",
        "logs",
        "docs",
        "tests",
        "scripts"
    ]
    
    missing_dirs = []
    for dir_path in required_dirs:
        full_path = base_path / dir_path
        if not full_path.exists():
            missing_dirs.append(dir_path)
    
    if missing_dirs:
        logger.error(f"✗ Missing directories: {missing_dirs}")
        return False
    
    logger.info("✓ Directory structure is correct")
    return True

def main():
    """Run all verification tests."""
    logger = setup_logging()
    
    logger.info("=" * 60)
    logger.info("Neural Quant Setup Verification")
    logger.info("=" * 60)
    
    tests = [
        ("Directory Structure", test_directory_structure),
        ("Imports", test_imports),
        ("Configuration", test_configuration),
        ("MLflow Integration", test_mlflow),
        ("Data Collector", test_data_collector),
        ("Strategy System", test_strategy),
        ("Trade Logging", test_trade_logging),
        ("Backtesting", test_backtest)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        logger.info(f"\n--- {test_name} ---")
        try:
            if test_func():
                passed += 1
                logger.info(f"✓ {test_name} PASSED")
            else:
                logger.error(f"✗ {test_name} FAILED")
        except Exception as e:
            logger.error(f"✗ {test_name} FAILED with exception: {e}")
    
    logger.info("\n" + "=" * 60)
    logger.info(f"Verification Results: {passed}/{total} tests passed")
    logger.info("=" * 60)
    
    if passed == total:
        logger.info("🎉 All tests passed! Neural Quant is ready to use.")
        logger.info("\nNext steps:")
        logger.info("1. Copy configs/config.example.yaml to configs/config.yaml")
        logger.info("2. Update config.yaml with your API keys and preferences")
        logger.info("3. Start MLflow server: mlflow server --backend-store-uri sqlite:///mlflow.db")
        logger.info("4. Run your first strategy: python scripts/run_strategy.py")
        return 0
    else:
        logger.error(f"❌ {total - passed} tests failed. Please fix the issues above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
