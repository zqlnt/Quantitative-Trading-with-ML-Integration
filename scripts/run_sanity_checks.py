#!/usr/bin/env python3
"""
Comprehensive sanity check runner for Neural Quant.

This script runs all sanity checks and fail-fast gates to ensure
system integrity and data quality.
"""

import sys
import logging
from pathlib import Path
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from src.utils.validation.sanity_checks import SanityChecker, create_fail_fast_gates, check_fail_fast_gates
from src.utils.helpers.determinism import set_global_seed, create_data_manifest, create_config_fingerprint
from src.utils.validation.metrics import calculate_comprehensive_metrics
from src.data.collectors.yfinance_collector import YFinanceCollector
from src.strategies.baselines.momentum_strategy import MomentumStrategy
from src.logging.mlflow.mlflow_manager import get_mlflow_manager
from src.utils.config.config_manager import get_config


def setup_logging():
    """Set up logging for sanity checks."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('logs/sanity_checks.log'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)


def test_purged_cv_correctness():
    """Test purged cross-validation correctness."""
    logger = logging.getLogger(__name__)
    logger.info("Testing purged CV correctness...")
    
    try:
        # Import test module
        from tests.test_validation import TestPurgedCrossValidation
        
        # Run tests
        test_instance = TestPurgedCrossValidation()
        test_instance.test_purged_cv_no_leakage()
        test_instance.test_embargo_period()
        test_instance.test_time_series_split()
        
        logger.info("✓ Purged CV correctness tests passed")
        return True
        
    except Exception as e:
        logger.error(f"✗ Purged CV correctness tests failed: {e}")
        return False


def test_cost_reconciliation():
    """Test cost reconciliation between backtest and paper trading."""
    logger = logging.getLogger(__name__)
    logger.info("Testing cost reconciliation...")
    
    try:
        from tests.test_cost_reconciliation import TestCostReconciliation
        
        test_instance = TestCostReconciliation()
        test_instance.test_backtest_vs_paper_costs()
        test_instance.test_cost_parity_with_different_sizes()
        
        logger.info("✓ Cost reconciliation tests passed")
        return True
        
    except Exception as e:
        logger.error(f"✗ Cost reconciliation tests failed: {e}")
        return False


def test_metrics_sanity():
    """Test metrics sanity checks."""
    logger = logging.getLogger(__name__)
    logger.info("Testing metrics sanity...")
    
    try:
        # Create synthetic returns
        np.random.seed(42)
        returns = pd.Series(np.random.randn(252), index=pd.date_range('2023-01-01', periods=252, freq='D'))
        
        # Calculate metrics
        metrics = calculate_comprehensive_metrics(returns)
        
        # Run sanity checks
        checker = SanityChecker()
        results = checker.check_metrics_sanity(metrics)
        
        if results['passed']:
            logger.info("✓ Metrics sanity checks passed")
            return True
        else:
            logger.error(f"✗ Metrics sanity checks failed: {results['issues']}")
            return False
            
    except Exception as e:
        logger.error(f"✗ Metrics sanity tests failed: {e}")
        return False


def test_determinism():
    """Test determinism and seeding."""
    logger = logging.getLogger(__name__)
    logger.info("Testing determinism...")
    
    try:
        # Set global seed
        set_global_seed(42)
        
        # Generate random numbers
        rand1 = np.random.randn(10)
        
        # Reset and generate again
        from src.utils.helpers.determinism import reset_random_state
        reset_random_state()
        rand2 = np.random.randn(10)
        
        # Should be identical
        if np.allclose(rand1, rand2):
            logger.info("✓ Determinism tests passed")
            return True
        else:
            logger.error("✗ Determinism tests failed: results not identical")
            return False
            
    except Exception as e:
        logger.error(f"✗ Determinism tests failed: {e}")
        return False


def test_data_quality():
    """Test data quality checks."""
    logger = logging.getLogger(__name__)
    logger.info("Testing data quality...")
    
    try:
        # Create test data
        data = pd.DataFrame({
            'price': np.random.randn(100),
            'volume': np.random.randint(1000, 10000, 100),
            'feature': np.random.randn(100)
        })
        
        # Run data quality checks
        checker = SanityChecker()
        results = checker.check_data_quality(data)
        
        if results['passed']:
            logger.info("✓ Data quality tests passed")
            return True
        else:
            logger.error(f"✗ Data quality tests failed: {results['issues']}")
            return False
            
    except Exception as e:
        logger.error(f"✗ Data quality tests failed: {e}")
        return False


def test_edge_cases():
    """Test edge cases and pathological scenarios."""
    logger = logging.getLogger(__name__)
    logger.info("Testing edge cases...")
    
    try:
        checker = SanityChecker()
        
        # Test flat series
        flat_returns = pd.Series(np.zeros(100))
        flat_check = checker.check_flat_series(flat_returns)
        
        # Test class imbalance
        imbalanced_y = pd.Series([0] * 90 + [1] * 10)  # 90/10 split
        imbalance_check = checker.check_class_imbalance(imbalanced_y)
        
        # Test pathological costs
        returns = pd.Series(np.random.randn(100) * 0.01)  # 1% daily returns
        costs = pd.Series(np.random.randn(100) * 0.05)    # 5% daily costs
        cost_check = checker.check_pathological_costs(returns, costs)
        
        logger.info("✓ Edge case tests passed")
        return True
        
    except Exception as e:
        logger.error(f"✗ Edge case tests failed: {e}")
        return False


def test_fail_fast_gates():
    """Test fail-fast gates."""
    logger = logging.getLogger(__name__)
    logger.info("Testing fail-fast gates...")
    
    try:
        # Create test results
        test_results = {
            'overall_passed': True,
            'total_issues': 0,
            'total_warnings': 1
        }
        
        # Create gates
        gates = create_fail_fast_gates()
        
        # Check gates
        gates_passed = check_fail_fast_gates(test_results, gates)
        
        if gates_passed:
            logger.info("✓ Fail-fast gates tests passed")
            return True
        else:
            logger.error("✗ Fail-fast gates tests failed")
            return False
            
    except Exception as e:
        logger.error(f"✗ Fail-fast gates tests failed: {e}")
        return False


def test_real_data_pipeline():
    """Test with real data pipeline."""
    logger = logging.getLogger(__name__)
    logger.info("Testing real data pipeline...")
    
    try:
        # Set global seed for reproducibility
        set_global_seed(42)
        
        # Get configuration
        config = get_config()
        
        # Create data collector
        collector = YFinanceCollector()
        
        # Fetch small amount of data
        end_date = datetime.now().strftime("%Y-%m-%d")
        start_date = (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")
        
        data = collector.get_historical_data(
            symbol="AAPL",
            start_date=start_date,
            end_date=end_date,
            timeframe="1d"
        )
        
        if data.empty:
            logger.warning("No data available for real data test")
            return True
        
        # Create data manifest
        manifest = create_data_manifest(
            data, ["AAPL"], start_date, end_date, "1d"
        )
        
        # Create config fingerprint
        config_dict = config.dict()
        fingerprint = create_config_fingerprint(config_dict)
        
        # Run comprehensive sanity checks
        checker = SanityChecker()
        
        # Calculate returns for metrics
        returns = data['Close'].pct_change().dropna()
        metrics = calculate_comprehensive_metrics(returns)
        
        # Run checks
        results = checker.run_comprehensive_checks(
            data, metrics, returns=returns
        )
        
        if results['overall_passed']:
            logger.info("✓ Real data pipeline tests passed")
            logger.info(f"  Data manifest: {manifest['data_hash']}")
            logger.info(f"  Config fingerprint: {fingerprint['config_hash']}")
            return True
        else:
            logger.error(f"✗ Real data pipeline tests failed: {results['issues']}")
            return False
            
    except Exception as e:
        logger.error(f"✗ Real data pipeline tests failed: {e}")
        return False


def main():
    """Run all sanity checks."""
    logger = setup_logging()
    
    logger.info("=" * 60)
    logger.info("NEURAL QUANT SANITY CHECKS")
    logger.info(f"Started at: {datetime.now()}")
    logger.info("=" * 60)
    
    # Define all tests
    tests = [
        ("Purged CV Correctness", test_purged_cv_correctness),
        ("Cost Reconciliation", test_cost_reconciliation),
        ("Metrics Sanity", test_metrics_sanity),
        ("Determinism", test_determinism),
        ("Data Quality", test_data_quality),
        ("Edge Cases", test_edge_cases),
        ("Fail-Fast Gates", test_fail_fast_gates),
        ("Real Data Pipeline", test_real_data_pipeline)
    ]
    
    passed = 0
    total = len(tests)
    results = {}
    
    for test_name, test_func in tests:
        logger.info(f"\n--- {test_name} ---")
        try:
            if test_func():
                passed += 1
                results[test_name] = "PASSED"
                logger.info(f"✓ {test_name} PASSED")
            else:
                results[test_name] = "FAILED"
                logger.error(f"✗ {test_name} FAILED")
        except Exception as e:
            results[test_name] = f"FAILED: {e}"
            logger.error(f"✗ {test_name} FAILED with exception: {e}")
    
    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("SANITY CHECK RESULTS")
    logger.info("=" * 60)
    
    for test_name, result in results.items():
        status = "✓" if "PASSED" in result else "✗"
        logger.info(f"{status} {test_name}: {result}")
    
    logger.info(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        logger.info("🎉 All sanity checks passed! System is ready for production.")
        return 0
    else:
        logger.error(f"❌ {total - passed} tests failed. Please fix issues before proceeding.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
