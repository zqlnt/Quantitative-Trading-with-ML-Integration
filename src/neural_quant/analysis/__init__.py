"""
Analysis module for Neural Quant.

This module contains statistical analysis tools for backtest results,
including Monte Carlo Permutation Testing (MCPT) for significance testing
and Bootstrap Confidence Intervals for robust metric estimation.
"""

from .mcpt import (
    MonteCarloPermutationTester,
    MCPTConfig,
    MCPTResult,
    run_mcpt_test
)

from .bootstrap import (
    BootstrapAnalyzer,
    BootstrapConfig,
    BootstrapResult,
    run_bootstrap_analysis
)

from .walkforward import (
    WalkForwardAnalyzer,
    WalkForwardConfig,
    WalkForwardWindow,
    run_walkforward_analysis
)

from .regime_filter import (
    RegimeFilter,
    RegimeFilterConfig,
    create_regime_filter_config
)

from .volatility_targeting import (
    VolatilityTargeting,
    VolatilityTargetingConfig,
    create_volatility_targeting_config
)

__all__ = [
    'MonteCarloPermutationTester',
    'MCPTConfig', 
    'MCPTResult',
    'run_mcpt_test',
    'BootstrapAnalyzer',
    'BootstrapConfig',
    'BootstrapResult',
    'run_bootstrap_analysis',
    'WalkForwardAnalyzer',
    'WalkForwardConfig',
    'WalkForwardWindow',
    'run_walkforward_analysis',
    'RegimeFilter',
    'RegimeFilterConfig',
    'create_regime_filter_config',
    'VolatilityTargeting',
    'VolatilityTargetingConfig',
    'create_volatility_targeting_config'
]
