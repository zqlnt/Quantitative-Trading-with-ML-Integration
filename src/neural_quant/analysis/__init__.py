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

from .allocation_methods import (
    AllocationMethods,
    AllocationMethodConfig,
    create_allocation_method_config
)

from .position_management import (
    PositionManager,
    PositionManagementConfig,
    create_position_management_config
)

from .basic_exits import (
    BasicExits,
    BasicExitsConfig,
    create_basic_exits_config
)

from ..logging.artifacts import (
    ArtifactManager,
    ArtifactType,
    ParamsArtifact,
    MetricsArtifact,
    EquityArtifact,
    TradesArtifact,
    MCPTArtifact,
    BootstrapArtifact,
    WalkForwardArtifact,
    WeightsArtifact,
    SummaryArtifact
)

from .summary_generator import StrategyAnalyst, PromotionRules
from .run_qa import RunQASystem
from .weekly_memo import WeeklyMemoGenerator
from .quant_researcher import QuantResearcher
from .strategy_developer import StrategyDeveloper
from .orchestration import OrchestrationEngine

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
    'create_volatility_targeting_config',
    'AllocationMethods',
    'AllocationMethodConfig',
    'create_allocation_method_config',
    'PositionManager',
    'PositionManagementConfig',
    'create_position_management_config',
    'BasicExits',
    'BasicExitsConfig',
    'create_basic_exits_config',
    'ArtifactManager',
    'ArtifactType',
    'ParamsArtifact',
    'MetricsArtifact',
    'EquityArtifact',
    'TradesArtifact',
    'MCPTArtifact',
    'BootstrapArtifact',
    'WalkForwardArtifact',
    'WeightsArtifact',
    'SummaryArtifact',
    'StrategyAnalyst',
    'PromotionRules',
    'RunQASystem',
    'WeeklyMemoGenerator',
    'QuantResearcher',
    'StrategyDeveloper',
    'OrchestrationEngine'
]
