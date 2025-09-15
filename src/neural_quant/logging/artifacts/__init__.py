"""
Artifact Management Module

This module handles saving structured results for each MLflow run,
including parameters, metrics, equity curves, trades, and analysis results.

Author: Neural Quant Team
Date: 2024
"""

from .artifact_manager import ArtifactManager
from .artifact_types import (
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

__all__ = [
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
    'SummaryArtifact'
]
