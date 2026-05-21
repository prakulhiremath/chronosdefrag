"""
ChronosDefrag — Temporal Defragmentation for Quantitative Finance.

Public API:
    ChronosDefragEngine  — full training and inference orchestrator
    DefragConfig         — typed configuration entry point
    DeploymentMode       — enum for operational mode selection
"""

from .config import DefragConfig, DeploymentMode
from .engine import ChronosDefragEngine, LivePrediction

__all__ = [
    "DefragConfig",
    "DeploymentMode",
    "ChronosDefragEngine",
    "LivePrediction",
]

__version__ = "0.1.0"
