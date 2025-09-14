#!/usr/bin/env python3
"""
Neural Quant CLI

Command-line interface for Neural Quant trading system.
"""

import sys
import argparse
import logging
from pathlib import Path

def setup_logging(level=logging.INFO):
    """Set up logging."""
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

def verify_setup():
    """Run setup verification."""
    from scripts.verify_setup import main
    return main()

def run_strategy():
    """Run a trading strategy."""
    from scripts.run_strategy import main
    return main()

def start_mlflow():
    """Start MLflow server."""
    import subprocess
    import os
    
    # Change to project directory
    project_dir = Path(__file__).parent.parent
    os.chdir(project_dir)
    
    # Start MLflow server
    cmd = [
        "mlflow", "server",
        "--backend-store-uri", "sqlite:///mlflow.db",
        "--default-artifact-root", "./mlruns",
        "--host", "0.0.0.0",
        "--port", "5000"
    ]
    
    print("Starting MLflow server...")
    print(f"Command: {' '.join(cmd)}")
    print("MLflow UI will be available at: http://localhost:5000")
    
    try:
        subprocess.run(cmd)
    except KeyboardInterrupt:
        print("\nMLflow server stopped.")

def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(description="Neural Quant Trading System")
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Verify setup command
    verify_parser = subparsers.add_parser('verify', help='Verify setup')
    verify_parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    
    # Run strategy command
    run_parser = subparsers.add_parser('run', help='Run a trading strategy')
    run_parser.add_argument('--strategy', '-s', default='momentum', help='Strategy to run')
    run_parser.add_argument('--symbols', nargs='+', help='Symbols to trade')
    run_parser.add_argument('--days', type=int, default=90, help='Number of days for backtest')
    
    # MLflow command
    mlflow_parser = subparsers.add_parser('mlflow', help='Start MLflow server')
    mlflow_parser.add_argument('--port', type=int, default=5000, help='MLflow server port')
    mlflow_parser.add_argument('--host', default='0.0.0.0', help='MLflow server host')
    
    args = parser.parse_args()
    
    if args.command == 'verify':
        if args.verbose:
            setup_logging(logging.DEBUG)
        else:
            setup_logging(logging.INFO)
        return verify_setup()
    
    elif args.command == 'run':
        setup_logging(logging.INFO)
        return run_strategy()
    
    elif args.command == 'mlflow':
        setup_logging(logging.INFO)
        start_mlflow()
        return 0
    
    else:
        parser.print_help()
        return 1

if __name__ == "__main__":
    sys.exit(main())
