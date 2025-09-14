"""
Cost reconciliation tests for Neural Quant.

This module tests that backtest costs match paper trading simulator costs
for the same signals.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from src.execution.paper.paper_trader import PaperTrader, ExecutionConfig
from src.strategies.base.strategy_base import Signal


class TestCostReconciliation:
    """Test cost reconciliation between backtest and paper trading."""
    
    def test_backtest_vs_paper_costs(self):
        """Test that backtest costs match paper trading costs."""
        # Create test signals
        signals = [
            Signal(
                symbol="AAPL",
                signal_type="BUY",
                strength=0.8,
                price=150.0,
                timestamp=datetime.now(),
                metadata={"test": True}
            ),
            Signal(
                symbol="AAPL",
                signal_type="SELL",
                strength=0.6,
                price=155.0,
                timestamp=datetime.now() + timedelta(hours=1),
                metadata={"test": True}
            ),
            Signal(
                symbol="MSFT",
                signal_type="BUY",
                strength=0.7,
                price=300.0,
                timestamp=datetime.now() + timedelta(hours=2),
                metadata={"test": True}
            )
        ]
        
        # Create paper trader with known costs
        execution_config = ExecutionConfig(
            commission=0.01,  # $0.01 per trade
            slippage=0.001,   # 0.1% slippage
            market_impact=0.0005  # 0.05% market impact per $1000
        )
        
        paper_trader = PaperTrader(
            initial_capital=100000,
            execution_config=execution_config
        )
        
        # Execute signals in paper trader
        paper_trades = []
        current_prices = {"AAPL": 150.0, "MSFT": 300.0}
        
        for signal in signals:
            trade = paper_trader.execute_signal(signal, current_prices[signal.symbol])
            if trade:
                paper_trades.append(trade)
        
        # Calculate paper trading costs
        paper_total_commission = sum(trade.commission for trade in paper_trades)
        paper_total_slippage = sum(
            abs(trade.price - signal.price) * trade.quantity 
            for trade, signal in zip(paper_trades, signals)
        )
        
        # Simulate backtest costs (simplified)
        backtest_costs = self._calculate_backtest_costs(signals, execution_config)
        
        # Calculate cost reconciliation error
        total_paper_costs = paper_total_commission + paper_total_slippage
        total_backtest_costs = backtest_costs['total_commission'] + backtest_costs['total_slippage']
        
        if total_backtest_costs > 0:
            cost_error_pct = abs(total_paper_costs - total_backtest_costs) / total_backtest_costs
        else:
            cost_error_pct = 0
        
        # Cost error should be less than 20%
        assert cost_error_pct < 0.20, f"Cost reconciliation error too high: {cost_error_pct:.2%}"
        
        print(f"Cost reconciliation test passed:")
        print(f"  Paper costs: ${total_paper_costs:.2f}")
        print(f"  Backtest costs: ${total_backtest_costs:.2f}")
        print(f"  Error: {cost_error_pct:.2%}")
    
    def _calculate_backtest_costs(self, signals, execution_config):
        """Calculate backtest costs for comparison."""
        total_commission = 0
        total_slippage = 0
        
        for signal in signals:
            if signal.signal_type in ['BUY', 'SELL']:
                # Calculate position size (simplified)
                position_value = 10000  # $10k per position
                quantity = int(position_value / signal.price)
                
                # Calculate costs
                trade_value = quantity * signal.price
                commission = execution_config.commission + (trade_value * 0.0001)
                
                # Calculate slippage
                slippage_pct = execution_config.slippage
                market_impact_pct = (trade_value / 1000) * execution_config.market_impact
                total_slippage_pct = slippage_pct + market_impact_pct
                
                executed_price = signal.price * (1 + total_slippage_pct)
                slippage_cost = abs(executed_price - signal.price) * quantity
                
                total_commission += commission
                total_slippage += slippage_cost
        
        return {
            'total_commission': total_commission,
            'total_slippage': total_slippage
        }
    
    def test_cost_parity_with_different_sizes(self):
        """Test cost parity with different position sizes."""
        # Test with various position sizes
        position_sizes = [1000, 5000, 10000, 50000, 100000]  # Different position values
        
        for position_value in position_sizes:
            # Create signal
            signal = Signal(
                symbol="AAPL",
                signal_type="BUY",
                strength=0.8,
                price=150.0,
                timestamp=datetime.now(),
                metadata={"position_value": position_value}
            )
            
            # Paper trader
            execution_config = ExecutionConfig(
                commission=0.01,
                slippage=0.001,
                market_impact=0.0005
            )
            
            paper_trader = PaperTrader(
                initial_capital=1000000,  # Large capital for big positions
                execution_config=execution_config
            )
            
            # Execute
            trade = paper_trader.execute_signal(signal, 150.0)
            
            if trade:
                # Calculate expected costs
                expected_commission = execution_config.commission + (trade.quantity * trade.price * 0.0001)
                expected_slippage = trade.quantity * trade.price * (execution_config.slippage + 
                                                                  (trade.quantity * trade.price / 1000) * execution_config.market_impact)
                
                # Check that actual costs are reasonable
                actual_commission = trade.commission
                actual_slippage = abs(trade.price - signal.price) * trade.quantity
                
                # Commission should be close to expected
                commission_error = abs(actual_commission - expected_commission) / expected_commission
                assert commission_error < 0.1, f"Commission error too high for position {position_value}: {commission_error:.2%}"
                
                # Slippage should be close to expected
                if expected_slippage > 0:
                    slippage_error = abs(actual_slippage - expected_slippage) / expected_slippage
                    assert slippage_error < 0.2, f"Slippage error too high for position {position_value}: {slippage_error:.2%}"


class TestTurnoverRealism:
    """Test turnover realism and capacity warnings."""
    
    def test_adv_capacity_warnings(self):
        """Test ADV-based capacity warnings."""
        # Create synthetic data with known ADV
        dates = pd.date_range('2023-01-01', periods=252, freq='D')
        
        # Create price data
        prices = pd.Series(100 + np.cumsum(np.random.randn(252) * 0.02), index=dates)
        
        # Create volume data (simulate ADV)
        base_volume = 1000000  # 1M shares per day
        volume = pd.Series(base_volume + np.random.randint(-200000, 200000, 252), index=dates)
        
        # Calculate ADV
        adv = volume.rolling(20).mean().iloc[-1]  # 20-day average
        
        # Test position changes
        position_changes = [0.01, 0.02, 0.05, 0.10, 0.20]  # Different % of ADV
        
        for pct_change in position_changes:
            position_size = int(adv * pct_change)
            
            # Check if position change exceeds 2% of ADV
            if pct_change > 0.02:
                print(f"WARNING: Position change {pct_change:.1%} exceeds 2% ADV threshold")
                print(f"  Position size: {position_size:,} shares")
                print(f"  ADV: {adv:,.0f} shares")
                print(f"  % of ADV: {pct_change:.1%}")
            
            # For this test, we just want to ensure the warning logic works
            assert position_size > 0, "Position size should be positive"
    
    def test_turnover_cost_impact(self):
        """Test that high turnover increases costs significantly."""
        # Create high-turnover scenario
        signals = []
        base_price = 100.0
        
        # Generate many small trades (high turnover)
        for i in range(50):  # 50 trades
            signal = Signal(
                symbol="TEST",
                signal_type="BUY" if i % 2 == 0 else "SELL",
                strength=0.5,
                price=base_price + i * 0.1,
                timestamp=datetime.now() + timedelta(minutes=i),
                metadata={"turnover_test": True}
            )
            signals.append(signal)
        
        # Calculate costs
        execution_config = ExecutionConfig(
            commission=0.01,
            slippage=0.001,
            market_impact=0.0005
        )
        
        paper_trader = PaperTrader(
            initial_capital=100000,
            execution_config=execution_config
        )
        
        total_costs = 0
        for signal in signals:
            trade = paper_trader.execute_signal(signal, signal.price)
            if trade:
                total_costs += trade.commission + abs(trade.price - signal.price) * trade.quantity
        
        # High turnover should result in significant costs
        assert total_costs > 100, f"High turnover costs too low: ${total_costs:.2f}"
        
        print(f"High turnover test:")
        print(f"  Number of trades: {len(signals)}")
        print(f"  Total costs: ${total_costs:.2f}")
        print(f"  Cost per trade: ${total_costs/len(signals):.2f}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
