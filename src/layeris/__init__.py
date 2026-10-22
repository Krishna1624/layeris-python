"""
Layeris Python Inference Engine
================================
A modular Python framework for:
1. Multi-layer expert prediction & predictive prefetching for Mixture-of-Experts (MoE) LLMs.
2. Hardware-balanced speculative decoding & interactive model inference.
"""

__version__ = "0.1.0"

from layeris.prefetch.predictor import (
    ColiPredictConfig,
    RoutingHistory,
    ColiMultiLayerPredictRuntime,
    calculate_priority,
    passes_confidence_filter,
)
from layeris.prefetch.simulator import (
    PrefetchCacheSimulator,
    SimulationStats,
)
from layeris.prefetch.trace import (
    MoERoutingTrace,
    generate_coupled_trace,
    generate_zipf_trace,
)

__all__ = [
    "ColiPredictConfig",
    "RoutingHistory",
    "ColiMultiLayerPredictRuntime",
    "calculate_priority",
    "passes_confidence_filter",
    "PrefetchCacheSimulator",
    "SimulationStats",
    "MoERoutingTrace",
    "generate_coupled_trace",
    "generate_zipf_trace",
]
