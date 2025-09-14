"""
Trade logging system for Neural Quant.

This module provides comprehensive logging for all trading activities including
trades, signals, performance metrics, and audit trails.
"""

import os
import json
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional, Union
from pathlib import Path
import pandas as pd
import sqlite3
from contextlib import contextmanager

from ...utils.config.config_manager import get_config


class TradeLogger:
    """
    Comprehensive trade logging system for Neural Quant.
    
    This class provides logging for trades, signals, performance metrics,
    and maintains audit trails for all trading activities.
    """
    
    def __init__(self, log_dir: Optional[str] = None):
        """
        Initialize the trade logger.
        
        Args:
            log_dir: Directory for log files. If None, uses config default.
        """
        self.config = get_config()
        self.log_dir = Path(log_dir or self.config.logging.file_logging.log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Set up logging
        self.logger = logging.getLogger(__name__)
        self._setup_logging()
        
        # Database for structured logging
        self.db_path = self.log_dir / "trades.db"
        self._setup_database()
        
    def _setup_logging(self):
        """Set up file logging for trades."""
        if not self.config.logging.trade_logging.enabled:
            return
            
        # Create trade log file handler
        trade_log_file = self.log_dir / "trades.log"
        trade_handler = logging.FileHandler(trade_log_file)
        trade_handler.setLevel(logging.INFO)
        
        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        trade_handler.setFormatter(formatter)
        
        # Add handler to logger
        self.logger.addHandler(trade_handler)
        self.logger.setLevel(logging.INFO)
        
    def _setup_database(self):
        """Set up SQLite database for structured trade logging."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                # Create trades table
                conn.execute('''
                    CREATE TABLE IF NOT EXISTS trades (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                        strategy_name TEXT NOT NULL,
                        symbol TEXT NOT NULL,
                        side TEXT NOT NULL,
                        quantity REAL NOT NULL,
                        price REAL NOT NULL,
                        value REAL NOT NULL,
                        commission REAL DEFAULT 0.0,
                        order_id TEXT,
                        status TEXT DEFAULT 'FILLED',
                        metadata TEXT
                    )
                ''')
                
                # Create signals table
                conn.execute('''
                    CREATE TABLE IF NOT EXISTS signals (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                        strategy_name TEXT NOT NULL,
                        symbol TEXT NOT NULL,
                        signal_type TEXT NOT NULL,
                        strength REAL,
                        price REAL,
                        metadata TEXT
                    )
                ''')
                
                # Create performance table
                conn.execute('''
                    CREATE TABLE IF NOT EXISTS performance (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                        strategy_name TEXT NOT NULL,
                        metric_name TEXT NOT NULL,
                        metric_value REAL NOT NULL,
                        period_start DATETIME,
                        period_end DATETIME,
                        metadata TEXT
                    )
                ''')
                
                # Create portfolio table
                conn.execute('''
                    CREATE TABLE IF NOT EXISTS portfolio (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                        strategy_name TEXT NOT NULL,
                        total_value REAL NOT NULL,
                        cash REAL NOT NULL,
                        positions TEXT,
                        metadata TEXT
                    )
                ''')
                
                conn.commit()
                
        except Exception as e:
            self.logger.error(f"Failed to set up trade database: {e}")
            raise
    
    @contextmanager
    def get_db_connection(self):
        """Get database connection with proper error handling."""
        conn = None
        try:
            conn = sqlite3.connect(self.db_path)
            yield conn
        except Exception as e:
            if conn:
                conn.rollback()
            self.logger.error(f"Database error: {e}")
            raise
        finally:
            if conn:
                conn.close()
    
    def log_trade(self, strategy_name: str, symbol: str, side: str, quantity: float,
                  price: float, order_id: Optional[str] = None, commission: float = 0.0,
                  status: str = "FILLED", metadata: Optional[Dict[str, Any]] = None):
        """
        Log a trade execution.
        
        Args:
            strategy_name: Name of the strategy that generated the trade.
            symbol: Trading symbol.
            side: Trade side (BUY, SELL).
            quantity: Number of shares/units.
            price: Execution price.
            order_id: Optional order ID from broker.
            commission: Commission paid.
            status: Trade status (FILLED, PARTIAL, CANCELLED, etc.).
            metadata: Additional trade metadata.
        """
        try:
            value = quantity * price
            metadata_json = json.dumps(metadata) if metadata else None
            
            with self.get_db_connection() as conn:
                conn.execute('''
                    INSERT INTO trades (strategy_name, symbol, side, quantity, price, 
                                      value, commission, order_id, status, metadata)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (strategy_name, symbol, side, quantity, price, value, 
                     commission, order_id, status, metadata_json))
                conn.commit()
            
            # Log to file
            self.logger.info(
                f"TRADE: {strategy_name} | {symbol} | {side} | {quantity} @ {price} | "
                f"Value: {value:.2f} | Commission: {commission:.2f} | Status: {status}"
            )
            
        except Exception as e:
            self.logger.error(f"Failed to log trade: {e}")
            raise
    
    def log_signal(self, strategy_name: str, symbol: str, signal_type: str,
                   strength: Optional[float] = None, price: Optional[float] = None,
                   metadata: Optional[Dict[str, Any]] = None):
        """
        Log a trading signal.
        
        Args:
            strategy_name: Name of the strategy.
            symbol: Trading symbol.
            signal_type: Type of signal (BUY, SELL, HOLD).
            strength: Signal strength (0-1).
            price: Price at signal generation.
            metadata: Additional signal metadata.
        """
        try:
            metadata_json = json.dumps(metadata) if metadata else None
            
            with self.get_db_connection() as conn:
                conn.execute('''
                    INSERT INTO signals (strategy_name, symbol, signal_type, strength, 
                                       price, metadata)
                    VALUES (?, ?, ?, ?, ?, ?)
                ''', (strategy_name, symbol, signal_type, strength, price, metadata_json))
                conn.commit()
            
            # Log to file
            strength_str = f" | Strength: {strength:.3f}" if strength else ""
            price_str = f" | Price: {price:.2f}" if price else ""
            self.logger.info(
                f"SIGNAL: {strategy_name} | {symbol} | {signal_type}{strength_str}{price_str}"
            )
            
        except Exception as e:
            self.logger.error(f"Failed to log signal: {e}")
            raise
    
    def log_performance_metric(self, strategy_name: str, metric_name: str, metric_value: float,
                              period_start: Optional[datetime] = None, 
                              period_end: Optional[datetime] = None,
                              metadata: Optional[Dict[str, Any]] = None):
        """
        Log a performance metric.
        
        Args:
            strategy_name: Name of the strategy.
            metric_name: Name of the metric.
            metric_value: Value of the metric.
            period_start: Start of the measurement period.
            period_end: End of the measurement period.
            metadata: Additional metadata.
        """
        try:
            metadata_json = json.dumps(metadata) if metadata else None
            
            with self.get_db_connection() as conn:
                conn.execute('''
                    INSERT INTO performance (strategy_name, metric_name, metric_value,
                                          period_start, period_end, metadata)
                    VALUES (?, ?, ?, ?, ?, ?)
                ''', (strategy_name, metric_name, metric_value, period_start, 
                     period_end, metadata_json))
                conn.commit()
            
            # Log to file
            period_str = ""
            if period_start and period_end:
                period_str = f" | Period: {period_start} to {period_end}"
            self.logger.info(
                f"PERFORMANCE: {strategy_name} | {metric_name}: {metric_value:.4f}{period_str}"
            )
            
        except Exception as e:
            self.logger.error(f"Failed to log performance metric: {e}")
            raise
    
    def log_portfolio_snapshot(self, strategy_name: str, total_value: float, cash: float,
                              positions: Dict[str, float], metadata: Optional[Dict[str, Any]] = None):
        """
        Log a portfolio snapshot.
        
        Args:
            strategy_name: Name of the strategy.
            total_value: Total portfolio value.
            cash: Cash balance.
            positions: Dictionary of symbol -> quantity positions.
            metadata: Additional metadata.
        """
        try:
            positions_json = json.dumps(positions)
            metadata_json = json.dumps(metadata) if metadata else None
            
            with self.get_db_connection() as conn:
                conn.execute('''
                    INSERT INTO portfolio (strategy_name, total_value, cash, positions, metadata)
                    VALUES (?, ?, ?, ?, ?)
                ''', (strategy_name, total_value, cash, positions_json, metadata_json))
                conn.commit()
            
            # Log to file
            positions_str = ", ".join([f"{sym}: {qty}" for sym, qty in positions.items()])
            self.logger.info(
                f"PORTFOLIO: {strategy_name} | Total: {total_value:.2f} | "
                f"Cash: {cash:.2f} | Positions: {positions_str}"
            )
            
        except Exception as e:
            self.logger.error(f"Failed to log portfolio snapshot: {e}")
            raise
    
    def get_trades(self, strategy_name: Optional[str] = None, 
                   symbol: Optional[str] = None,
                   start_date: Optional[datetime] = None,
                   end_date: Optional[datetime] = None) -> pd.DataFrame:
        """
        Get trades from the database.
        
        Args:
            strategy_name: Filter by strategy name.
            symbol: Filter by symbol.
            start_date: Filter by start date.
            end_date: Filter by end date.
            
        Returns:
            pd.DataFrame: DataFrame of trades.
        """
        try:
            query = "SELECT * FROM trades WHERE 1=1"
            params = []
            
            if strategy_name:
                query += " AND strategy_name = ?"
                params.append(strategy_name)
            
            if symbol:
                query += " AND symbol = ?"
                params.append(symbol)
            
            if start_date:
                query += " AND timestamp >= ?"
                params.append(start_date)
            
            if end_date:
                query += " AND timestamp <= ?"
                params.append(end_date)
            
            query += " ORDER BY timestamp DESC"
            
            with self.get_db_connection() as conn:
                df = pd.read_sql_query(query, conn, params=params)
            
            return df
            
        except Exception as e:
            self.logger.error(f"Failed to get trades: {e}")
            raise
    
    def get_signals(self, strategy_name: Optional[str] = None,
                   symbol: Optional[str] = None,
                   signal_type: Optional[str] = None) -> pd.DataFrame:
        """
        Get signals from the database.
        
        Args:
            strategy_name: Filter by strategy name.
            symbol: Filter by symbol.
            signal_type: Filter by signal type.
            
        Returns:
            pd.DataFrame: DataFrame of signals.
        """
        try:
            query = "SELECT * FROM signals WHERE 1=1"
            params = []
            
            if strategy_name:
                query += " AND strategy_name = ?"
                params.append(strategy_name)
            
            if symbol:
                query += " AND symbol = ?"
                params.append(symbol)
            
            if signal_type:
                query += " AND signal_type = ?"
                params.append(signal_type)
            
            query += " ORDER BY timestamp DESC"
            
            with self.get_db_connection() as conn:
                df = pd.read_sql_query(query, conn, params=params)
            
            return df
            
        except Exception as e:
            self.logger.error(f"Failed to get signals: {e}")
            raise
    
    def get_performance_metrics(self, strategy_name: Optional[str] = None,
                               metric_name: Optional[str] = None) -> pd.DataFrame:
        """
        Get performance metrics from the database.
        
        Args:
            strategy_name: Filter by strategy name.
            metric_name: Filter by metric name.
            
        Returns:
            pd.DataFrame: DataFrame of performance metrics.
        """
        try:
            query = "SELECT * FROM performance WHERE 1=1"
            params = []
            
            if strategy_name:
                query += " AND strategy_name = ?"
                params.append(strategy_name)
            
            if metric_name:
                query += " AND metric_name = ?"
                params.append(metric_name)
            
            query += " ORDER BY timestamp DESC"
            
            with self.get_db_connection() as conn:
                df = pd.read_sql_query(query, conn, params=params)
            
            return df
            
        except Exception as e:
            self.logger.error(f"Failed to get performance metrics: {e}")
            raise
    
    def get_portfolio_history(self, strategy_name: Optional[str] = None) -> pd.DataFrame:
        """
        Get portfolio history from the database.
        
        Args:
            strategy_name: Filter by strategy name.
            
        Returns:
            pd.DataFrame: DataFrame of portfolio snapshots.
        """
        try:
            query = "SELECT * FROM portfolio WHERE 1=1"
            params = []
            
            if strategy_name:
                query += " AND strategy_name = ?"
                params.append(strategy_name)
            
            query += " ORDER BY timestamp DESC"
            
            with self.get_db_connection() as conn:
                df = pd.read_sql_query(query, conn, params=params)
            
            return df
            
        except Exception as e:
            self.logger.error(f"Failed to get portfolio history: {e}")
            raise
    
    def calculate_strategy_performance(self, strategy_name: str) -> Dict[str, float]:
        """
        Calculate performance metrics for a strategy.
        
        Args:
            strategy_name: Name of the strategy.
            
        Returns:
            Dict[str, float]: Dictionary of performance metrics.
        """
        try:
            trades_df = self.get_trades(strategy_name)
            
            if trades_df.empty:
                return {}
            
            # Calculate basic metrics
            total_trades = len(trades_df)
            buy_trades = trades_df[trades_df['side'] == 'BUY']
            sell_trades = trades_df[trades_df['side'] == 'SELL']
            
            total_value = trades_df['value'].sum()
            total_commission = trades_df['commission'].sum()
            
            # Calculate P&L (simplified - would need position tracking for accurate calculation)
            buy_value = buy_trades['value'].sum() if not buy_trades.empty else 0
            sell_value = sell_trades['value'].sum() if not sell_trades.empty else 0
            net_pnl = sell_value - buy_value - total_commission
            
            # Calculate win rate (simplified)
            profitable_trades = len(trades_df[trades_df['value'] > 0])  # Simplified
            win_rate = profitable_trades / total_trades if total_trades > 0 else 0
            
            performance = {
                'total_trades': total_trades,
                'total_value': total_value,
                'total_commission': total_commission,
                'net_pnl': net_pnl,
                'win_rate': win_rate,
                'avg_trade_value': total_value / total_trades if total_trades > 0 else 0
            }
            
            return performance
            
        except Exception as e:
            self.logger.error(f"Failed to calculate strategy performance: {e}")
            return {}


# Global trade logger instance
trade_logger = TradeLogger()


def get_trade_logger() -> TradeLogger:
    """
    Get the global trade logger instance.
    
    Returns:
        TradeLogger: Global trade logger instance.
    """
    return trade_logger


def log_trade(strategy_name: str, symbol: str, side: str, quantity: float,
              price: float, **kwargs):
    """
    Log a trade using the global trade logger.
    
    Args:
        strategy_name: Name of the strategy.
        symbol: Trading symbol.
        side: Trade side.
        quantity: Number of shares.
        price: Execution price.
        **kwargs: Additional trade parameters.
    """
    trade_logger.log_trade(strategy_name, symbol, side, quantity, price, **kwargs)


def log_signal(strategy_name: str, symbol: str, signal_type: str, **kwargs):
    """
    Log a signal using the global trade logger.
    
    Args:
        strategy_name: Name of the strategy.
        symbol: Trading symbol.
        signal_type: Type of signal.
        **kwargs: Additional signal parameters.
    """
    trade_logger.log_signal(strategy_name, symbol, signal_type, **kwargs)
