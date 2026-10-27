"""
Layeris Inference Subsystem: Speculative Decoding & Hardware Safety Guard
"""

from layeris.inference.guard import (
    configure_safe_execution_environment,
    detect_hardware_capabilities,
    HardwareProfile,
)
from layeris.inference.speculative import (
    SpeculativeDecoder,
    PromptLookupDraftGenerator,
    SpeculativeStats,
    TwoModelSpeculativeEngine,
    verify_candidates,
    make_model_verifier,
)
from layeris.inference.chat import (
    InteractiveChatEngine,
)
from layeris.inference.kv_cache import (
    KVCacheSettings,
    TurboQuantCache,
    create_kv_cache,
    get_kv_cache_settings,
)
from layeris.inference.memory_balancer import (
    BalancedPlacementPlan,
    MemoryBalancer,
    MemoryBreakdown,
    ModelGeometry,
    calculate_balanced_memory_plan,
    resolve_model_geometry,
)

__all__ = [
    "configure_safe_execution_environment",
    "detect_hardware_capabilities",
    "HardwareProfile",
    "SpeculativeDecoder",
    "PromptLookupDraftGenerator",
    "SpeculativeStats",
    "TwoModelSpeculativeEngine",
    "verify_candidates",
    "make_model_verifier",
    "InteractiveChatEngine",
    "KVCacheSettings",
    "TurboQuantCache",
    "create_kv_cache",
    "get_kv_cache_settings",
    "BalancedPlacementPlan",
    "MemoryBalancer",
    "MemoryBreakdown",
    "ModelGeometry",
    "calculate_balanced_memory_plan",
    "resolve_model_geometry",
]
