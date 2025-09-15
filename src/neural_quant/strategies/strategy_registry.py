"""
Strategy Registry for Neural Quant

This module provides a centralized registry for all available trading strategies,
making it easy to discover, instantiate, and manage strategies.
"""

from typing import Dict, Type, Any, List, Optional
from .base.strategy_base import StrategyBase
from .momentum import MovingAverageCrossover
from .bollinger_bands import BollingerBandsStrategy
from .volatility_breakout import VolatilityBreakoutStrategy
from .cross_sectional_momentum import CrossSectionalMomentumStrategy


class StrategyRegistry:
    """Registry for managing trading strategies."""
    
    def __init__(self):
        """Initialize the strategy registry."""
        self._strategies: Dict[str, Type[StrategyBase]] = {}
        self._register_default_strategies()
    
    def _register_default_strategies(self):
        """Register default strategies."""
        self.register_strategy('momentum', MovingAverageCrossover)
        self.register_strategy('moving_average_crossover', MovingAverageCrossover)
        self.register_strategy('bollinger_bands', BollingerBandsStrategy)
        self.register_strategy('volatility_breakout', VolatilityBreakoutStrategy)
        self.register_strategy('cross_sectional_momentum', CrossSectionalMomentumStrategy)
    
    def register_strategy(self, name: str, strategy_class: Type[StrategyBase]):
        """
        Register a strategy class.
        
        Args:
            name: Strategy name/identifier
            strategy_class: Strategy class that inherits from StrategyBase
        """
        if not issubclass(strategy_class, StrategyBase):
            raise ValueError(f"Strategy class must inherit from StrategyBase")
        
        self._strategies[name.lower()] = strategy_class
    
    def get_strategy(self, name: str, **kwargs) -> StrategyBase:
        """
        Get a strategy instance.
        
        Args:
            name: Strategy name
            **kwargs: Strategy parameters
            
        Returns:
            Strategy instance
        """
        name = name.lower()
        if name not in self._strategies:
            available = ', '.join(self._strategies.keys())
            raise ValueError(f"Strategy '{name}' not found. Available strategies: {available}")
        
        strategy_class = self._strategies[name]
        return strategy_class(**kwargs)
    
    def list_strategies(self) -> List[str]:
        """
        List all available strategy names.
        
        Returns:
            List of strategy names
        """
        return list(self._strategies.keys())
    
    def get_strategy_info(self, name: str) -> Dict[str, Any]:
        """
        Get information about a strategy.
        
        Args:
            name: Strategy name
            
        Returns:
            Dictionary with strategy information
        """
        name = name.lower()
        if name not in self._strategies:
            raise ValueError(f"Strategy '{name}' not found")
        
        strategy_class = self._strategies[name]
        
        # Get strategy parameters from docstring or class attributes
        # Check if it's a portfolio strategy by creating a temporary instance
        try:
            temp_instance = strategy_class()
            is_portfolio = temp_instance.is_portfolio_strategy()
        except Exception:
            # If we can't create an instance, assume it's not a portfolio strategy
            is_portfolio = False
        
        info = {
            'name': name,
            'class_name': strategy_class.__name__,
            'description': strategy_class.__doc__ or "No description available",
            'is_portfolio_strategy': is_portfolio
        }
        
        # Try to get parameter information from __init__ method
        try:
            import inspect
            sig = inspect.signature(strategy_class.__init__)
            params = {}
            for param_name, param in sig.parameters.items():
                if param_name != 'self' and param_name != 'kwargs':
                    params[param_name] = {
                        'default': param.default if param.default != inspect.Parameter.empty else None,
                        'annotation': str(param.annotation) if param.annotation != inspect.Parameter.empty else None
                    }
            info['parameters'] = params
        except Exception:
            info['parameters'] = {}
        
        return info
    
    def get_portfolio_strategies(self) -> List[str]:
        """
        Get list of portfolio strategies (those that require multiple tickers).
        
        Returns:
            List of portfolio strategy names
        """
        portfolio_strategies = []
        for name, strategy_class in self._strategies.items():
            try:
                temp_instance = strategy_class()
                if temp_instance.is_portfolio_strategy():
                    portfolio_strategies.append(name)
            except Exception:
                # If we can't create an instance, skip this strategy
                continue
        
        return portfolio_strategies
    
    def get_single_ticker_strategies(self) -> List[str]:
        """
        Get list of single-ticker strategies.
        
        Returns:
            List of single-ticker strategy names
        """
        all_strategies = set(self.list_strategies())
        portfolio_strategies = set(self.get_portfolio_strategies())
        return list(all_strategies - portfolio_strategies)


# Global registry instance
_strategy_registry = None

def get_strategy_registry() -> StrategyRegistry:
    """Get the global strategy registry instance."""
    global _strategy_registry
    if _strategy_registry is None:
        _strategy_registry = StrategyRegistry()
    return _strategy_registry

def register_strategy(name: str, strategy_class: Type[StrategyBase]):
    """Register a strategy with the global registry."""
    registry = get_strategy_registry()
    registry.register_strategy(name, strategy_class)

def get_strategy(name: str, **kwargs) -> StrategyBase:
    """Get a strategy instance from the global registry."""
    registry = get_strategy_registry()
    return registry.get_strategy(name, **kwargs)

def list_strategies() -> List[str]:
    """List all available strategies."""
    registry = get_strategy_registry()
    return registry.list_strategies()

def get_strategy_info(name: str) -> Dict[str, Any]:
    """Get strategy information."""
    registry = get_strategy_registry()
    return registry.get_strategy_info(name)
