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

__all__ = [
    "configure_safe_execution_environment",
    "detect_hardware_capabilities",
    "HardwareProfile",
    "SpeculativeDecoder",
    "PromptLookupDraftGenerator",
    "SpeculativeStats",
    "InteractiveChatEngine",
    "KVCacheSettings",
    "TurboQuantCache",
    "create_kv_cache",
    "get_kv_cache_settings",
]
