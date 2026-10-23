"""
Common MoE Architecture Adapter Interface
=========================================

Abstract interface ensuring every supported MoE model architecture adheres
to identical routing observation contracts without altering authoritative routing semantics.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple

from layeris.moe.config import MoEConfig
from layeris.moe.observation import RouterObservation


class MoEArchitectureAdapter(ABC):
    """Abstract base adapter for architecture-specific MoE routing semantics."""

    def __init__(self, config: MoEConfig):
        self.config = config

    @abstractmethod
    def get_router_modules(self, model: Any) -> Dict[int, Any]:
        """Locates and returns the authoritative router gate module for each layer.
        
        Returns:
            Dict mapping layer_id (0 .. num_layers - 1) to the PyTorch router module.
        """
        pass

    @abstractmethod
    def extract_observation(
        self,
        module_output: Any,
        layer_id: int,
        step_id: int,
        is_prefill: bool,
    ) -> RouterObservation:
        """Parses the native router module's forward output into a standardized RouterObservation.
        
        Must preserve the model's exact routing choices without applying assumed re-ranking.
        """
        pass

    @property
    def total_layers(self) -> int:
        return self.config.num_layers

    @property
    def num_experts(self) -> int:
        return self.config.num_experts

    @property
    def top_k(self) -> int:
        return self.config.num_experts_per_tok
