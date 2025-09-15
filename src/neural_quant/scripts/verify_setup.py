#!/usr/bin/env python3
"""Verify Neural Quant setup and dependencies."""

import sys
import logging
from pathlib import Path

# Add the src directory to the path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from neural_quant.utils.config.config_manager import get_config
from neural_quant.data.yf_loader import load_yf_data
from neural_quant.strategies.momentum import MovingAverageCrossover
from neural_quant.core.backtest import Backtester

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_imports():
    """Test that all imports work correctly."""
    logger.info("Testing imports...")
    
    try:
        import pandas as pd
        import numpy as np
        import yfinance as yf
        import mlflow
        import streamlit as st
        logger.info("All core dependencies imported successfully")
        return True
    except ImportError as e:
        logger.error(f"Import error: {e}")
        return False

def test_config():
    """Test configuration loading."""
    logger.info("Testing configuration...")
    
    try:
        config = get_config()
        logger.info(f"Configuration loaded successfully")
        logger.info(f"  Environment: {config.environment}")
        logger.info(f"  MLflow URI: {config.mlflow.tracking_uri}")
        logger.info(f"  Default symbols: {config.trading.default_symbols}")
        return True
    except Exception as e:
        logger.error(f"Configuration error: {e}")
        return False

def test_mlflow():
    """Test MLflow connection."""
    logger.info("Testing MLflow connection...")
    
    try:
        import mlflow
        mlflow.set_tracking_uri("sqlite:///mlflow.db")
        with mlflow.start_run():
            mlflow.log_param("test", "value")
        logger.info("MLflow connection successful")
        return True
    except Exception as e:
        logger.error(f"MLflow error: {e}")
        return False

def test_data_loading():
    """Test data loading."""
    logger.info("Testing data loading...")
    
    try:
        data = load_yf_data(['AAPL'], '2023-01-01', '2023-12-31', '1d')
        if data.empty:
            logger.warning("No data loaded (this might be normal)")
        else:
            logger.info(f"Data loading successful: {len(data)} records")
        return True
    except Exception as e:
        logger.error(f"Data loading error: {e}")
        return False

def test_strategy():
    """Test strategy initialization."""
    logger.info("Testing strategy initialization...")
    
    try:
        strategy = MovingAverageCrossover(ma_fast=10, ma_slow=30, threshold=0.0)
        logger.info("Strategy initialization successful")
        return True
    except Exception as e:
        logger.error(f"Strategy error: {e}")
        return False

def test_backtester():
    """Test backtester initialization."""
    logger.info("Testing backtester initialization...")
    
    try:
        backtester = Backtester(initial_capital=100000, commission=0.001, slippage=0.0005)
        logger.info("Backtester initialization successful")
        return True
    except Exception as e:
        logger.error(f"Backtester error: {e}")
        return False

def main():
    """Main verification function."""
    logger.info("=" * 80)
    logger.info("NEURAL QUANT - SETUP VERIFICATION")
    logger.info("=" * 80)
    
    tests = [
        ("Imports", test_imports),
        ("Configuration", test_config),
        ("MLflow", test_mlflow),
        ("Data Loading", test_data_loading),
        ("Strategy", test_strategy),
        ("Backtester", test_backtester)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        logger.info("")
        if test_func():
            passed += 1
        else:
            logger.error(f"Test '{test_name}' failed")
    
    logger.info("")
    logger.info("=" * 80)
    logger.info(f"VERIFICATION COMPLETE: {passed}/{total} tests passed")
    logger.info("=" * 80)
    
    if passed == total:
        logger.info("All tests passed! Neural Quant is ready to use.")
        return 0
    else:
        logger.error("Some tests failed. Please check the errors above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
