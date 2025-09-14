"""
Momentum strategy implementation for Neural Quant.

This module implements a simple momentum-based trading strategy that buys
stocks when they show positive momentum and sells when momentum turns negative.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Any
from datetime import datetime

from ..base.strategy_base import StrategyBase, Signal


class MomentumStrategy(StrategyBase):
    """
    Momentum-based trading strategy.
    
    This strategy identifies stocks with strong momentum and trades based on
    momentum signals. It uses multiple timeframes and technical indicators
    to generate buy/sell signals.
    """
    
    def __init__(self, parameters: Dict[str, Any] = None):
        """
        Initialize the momentum strategy.
        
        Args:
            parameters: Strategy parameters including:
                - lookback_period: Number of periods for momentum calculation (default: 20)
                - threshold: Momentum threshold for signal generation (default: 0.02)
                - min_volume: Minimum volume requirement (default: 1000000)
                - max_positions: Maximum number of positions (default: 10)
        """
        default_params = {
            'lookback_period': 20,
            'threshold': 0.02,
            'min_volume': 1000000,
            'max_positions': 10
        }
        
        if parameters:
            default_params.update(parameters)
        
        super().__init__("MomentumStrategy", default_params)
        
        # Strategy-specific attributes
        self.lookback_period = self.parameters['lookback_period']
        self.threshold = self.parameters['threshold']
        self.min_volume = self.parameters['min_volume']
        self.max_positions = self.parameters['max_positions']
        
        # Internal state
        self.price_history = {}
        self.momentum_scores = {}
        
    def _initialize_strategy(self, data: pd.DataFrame):
        """
        Initialize the strategy with historical data.
        
        Args:
            data: Historical data for initialization.
        """
        self.logger.info(f"Initializing momentum strategy with {len(data)} records")
        
        # Group data by symbol
        for symbol, symbol_data in data.groupby('Symbol'):
            if symbol not in self.price_history:
                self.price_history[symbol] = []
                self.momentum_scores[symbol] = []
            
            # Store price history
            prices = symbol_data['Close'].values
            self.price_history[symbol].extend(prices.tolist())
            
            # Calculate initial momentum scores
            if len(self.price_history[symbol]) >= self.lookback_period:
                momentum = self._calculate_momentum(symbol)
                self.momentum_scores[symbol].append(momentum)
        
        self.logger.info(f"Initialized momentum strategy for {len(self.price_history)} symbols")
    
    def _generate_signals(self, data: pd.DataFrame) -> List[Signal]:
        """
        Generate trading signals based on momentum analysis.
        
        Args:
            data: Current market data.
            
        Returns:
            List[Signal]: List of generated signals.
        """
        signals = []
        
        for symbol, symbol_data in data.groupby('Symbol'):
            try:
                # Get current price and volume
                current_price = symbol_data['Close'].iloc[-1]
                current_volume = symbol_data['Volume'].iloc[-1]
                
                # Update price history
                if symbol not in self.price_history:
                    self.price_history[symbol] = []
                    self.momentum_scores[symbol] = []
                
                self.price_history[symbol].append(current_price)
                
                # Keep only recent history
                if len(self.price_history[symbol]) > self.lookback_period * 2:
                    self.price_history[symbol] = self.price_history[symbol][-self.lookback_period * 2:]
                
                # Check minimum volume requirement
                if current_volume < self.min_volume:
                    continue
                
                # Calculate momentum
                if len(self.price_history[symbol]) >= self.lookback_period:
                    momentum = self._calculate_momentum(symbol)
                    self.momentum_scores[symbol].append(momentum)
                    
                    # Keep only recent momentum scores
                    if len(self.momentum_scores[symbol]) > 10:
                        self.momentum_scores[symbol] = self.momentum_scores[symbol][-10:]
                    
                    # Generate signal based on momentum
                    signal = self._generate_momentum_signal(symbol, current_price, momentum)
                    if signal:
                        signals.append(signal)
                
            except Exception as e:
                self.logger.error(f"Error processing symbol {symbol}: {e}")
                continue
        
        return signals
    
    def _calculate_momentum(self, symbol: str) -> float:
        """
        Calculate momentum score for a symbol.
        
        Args:
            symbol: Trading symbol.
            
        Returns:
            float: Momentum score.
        """
        if len(self.price_history[symbol]) < self.lookback_period:
            return 0.0
        
        prices = np.array(self.price_history[symbol][-self.lookback_period:])
        
        # Calculate multiple momentum indicators
        # 1. Simple price momentum
        price_momentum = (prices[-1] - prices[0]) / prices[0]
        
        # 2. Rate of change
        roc = (prices[-1] - prices[-5]) / prices[-5] if len(prices) >= 5 else 0
        
        # 3. Moving average momentum
        short_ma = np.mean(prices[-5:]) if len(prices) >= 5 else prices[-1]
        long_ma = np.mean(prices[-self.lookback_period:])
        ma_momentum = (short_ma - long_ma) / long_ma
        
        # 4. Volatility-adjusted momentum
        returns = np.diff(prices) / prices[:-1]
        volatility = np.std(returns) if len(returns) > 1 else 0
        vol_adjusted_momentum = price_momentum / volatility if volatility > 0 else 0
        
        # Combine momentum indicators with weights
        momentum_score = (
            0.4 * price_momentum +
            0.2 * roc +
            0.2 * ma_momentum +
            0.2 * vol_adjusted_momentum
        )
        
        return momentum_score
    
    def _generate_momentum_signal(self, symbol: str, current_price: float, momentum: float) -> Signal:
        """
        Generate a trading signal based on momentum.
        
        Args:
            symbol: Trading symbol.
            current_price: Current price.
            momentum: Momentum score.
            
        Returns:
            Signal: Generated signal or None.
        """
        # Check if we already have a position
        current_position = self.current_positions.get(symbol, 0)
        
        # Generate buy signal for strong positive momentum
        if momentum > self.threshold and current_position == 0:
            # Check if we haven't exceeded max positions
            if len(self.current_positions) < self.max_positions:
                return Signal(
                    symbol=symbol,
                    signal_type='BUY',
                    strength=min(momentum / self.threshold, 1.0),
                    price=current_price,
                    timestamp=datetime.now(),
                    metadata={
                        'momentum_score': momentum,
                        'strategy': 'momentum',
                        'lookback_period': self.lookback_period
                    }
                )
        
        # Generate sell signal for negative momentum
        elif momentum < -self.threshold and current_position > 0:
            return Signal(
                symbol=symbol,
                signal_type='SELL',
                strength=min(abs(momentum) / self.threshold, 1.0),
                price=current_price,
                timestamp=datetime.now(),
                metadata={
                    'momentum_score': momentum,
                    'strategy': 'momentum',
                    'lookback_period': self.lookback_period
                }
            )
        
        # Generate hold signal for neutral momentum
        elif abs(momentum) <= self.threshold and current_position > 0:
            return Signal(
                symbol=symbol,
                signal_type='HOLD',
                strength=0.5,
                price=current_price,
                timestamp=datetime.now(),
                metadata={
                    'momentum_score': momentum,
                    'strategy': 'momentum',
                    'lookback_period': self.lookback_period
                }
            )
        
        return None
    
    def get_momentum_analysis(self, symbol: str) -> Dict[str, Any]:
        """
        Get detailed momentum analysis for a symbol.
        
        Args:
            symbol: Trading symbol.
            
        Returns:
            Dict[str, Any]: Momentum analysis results.
        """
        if symbol not in self.price_history or len(self.price_history[symbol]) < self.lookback_period:
            return {}
        
        prices = np.array(self.price_history[symbol][-self.lookback_period:])
        
        # Calculate various momentum metrics
        analysis = {
            'symbol': symbol,
            'current_price': prices[-1],
            'price_change': prices[-1] - prices[0],
            'price_change_pct': (prices[-1] - prices[0]) / prices[0],
            'momentum_score': self.momentum_scores[symbol][-1] if self.momentum_scores[symbol] else 0,
            'volatility': np.std(np.diff(prices) / prices[:-1]) if len(prices) > 1 else 0,
            'trend_direction': 'up' if prices[-1] > prices[0] else 'down',
            'lookback_period': self.lookback_period,
            'data_points': len(prices)
        }
        
        # Add moving average analysis
        if len(prices) >= 10:
            short_ma = np.mean(prices[-5:])
            long_ma = np.mean(prices[-10:])
            analysis['short_ma'] = short_ma
            analysis['long_ma'] = long_ma
            analysis['ma_signal'] = 'bullish' if short_ma > long_ma else 'bearish'
        
        return analysis
    
    def get_strategy_summary(self) -> Dict[str, Any]:
        """
        Get a summary of the strategy's current state.
        
        Returns:
            Dict[str, Any]: Strategy summary.
        """
        summary = self.get_strategy_info()
        
        # Add momentum-specific information
        summary.update({
            'lookback_period': self.lookback_period,
            'threshold': self.threshold,
            'min_volume': self.min_volume,
            'max_positions': self.max_positions,
            'tracked_symbols': len(self.price_history),
            'active_positions': len(self.current_positions),
            'available_positions': self.max_positions - len(self.current_positions)
        })
        
        # Add momentum analysis for current positions
        position_analysis = {}
        for symbol in self.current_positions:
            analysis = self.get_momentum_analysis(symbol)
            if analysis:
                position_analysis[symbol] = analysis
        
        summary['position_analysis'] = position_analysis
        
        return summary
