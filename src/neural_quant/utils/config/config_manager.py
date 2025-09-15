"""
Configuration Manager for Neural Quant

This module provides configuration management for the trading system.
"""

from dataclasses import dataclass
from typing import Dict, Any
import os

@dataclass
class TradingConfig:
    """Trading configuration."""
    max_position_size: float = 0.1  # 10% max position size
    max_daily_loss: float = 0.05    # 5% max daily loss
    max_drawdown: float = 0.2       # 20% max drawdown
    
    @dataclass
    class PaperTrading:
        initial_capital: float = 100000.0
    
    paper_trading: PaperTrading = PaperTrading()

class ConfigManager:
    """Configuration manager for Neural Quant."""
    
    def __init__(self):
        """Initialize the configuration manager."""
        self.trading = TradingConfig()
    
    def dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            'trading': {
                'max_position_size': self.trading.max_position_size,
                'max_daily_loss': self.trading.max_daily_loss,
                'max_drawdown': self.trading.max_drawdown,
                'paper_trading': {
                    'initial_capital': self.trading.paper_trading.initial_capital
                }
            }
        }

# Global instance
_config_manager = None

def get_config() -> ConfigManager:
    """Get the global configuration manager instance."""
    global _config_manager
    if _config_manager is None:
        _config_manager = ConfigManager()
    return _config_manager