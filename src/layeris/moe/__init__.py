"""
Layeris Mixture-of-Experts (MoE) Architecture Subsystem
======================================================
"""

from layeris.moe.config import MoEConfig, ValidationOutcome, detect_moe_architecture
from layeris.moe.observation import RouterObservation
from layeris.moe.adapter import MoEArchitectureAdapter
from layeris.moe.qwen import Qwen36MoEAdapter
from layeris.moe.router_hooks import RouterHookManager
from layeris.moe.live_predictor import LiveMoEPredictor, StepTelemetry

__all__ = [
    "MoEConfig",
    "ValidationOutcome",
    "detect_moe_architecture",
    "RouterObservation",
    "MoEArchitectureAdapter",
    "Qwen36MoEAdapter",
    "RouterHookManager",
    "LiveMoEPredictor",
    "StepTelemetry",
]
