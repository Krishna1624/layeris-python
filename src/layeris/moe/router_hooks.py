"""
Router Hook Lifecycle & Live Observation Manager
================================================

Attaches non-invasive PyTorch forward hooks to MoE router modules, streaming
authoritative routing observations without modifying model outputs, weights, or graph state.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Dict, List, Optional

import torch

from layeris.moe.adapter import MoEArchitectureAdapter
from layeris.moe.observation import RouterObservation

logger = logging.getLogger(__name__)


class RouterHookManager:
    """Manages PyTorch forward hooks for live MoE routing observation streaming."""

    def __init__(
        self,
        adapter: MoEArchitectureAdapter,
        on_observation: Optional[Callable[[RouterObservation], None]] = None,
    ):
        self.adapter = adapter
        self.on_observation = on_observation
        self.hooks: List[torch.utils.hooks.RemovableHandle] = []
        self.observations: List[RouterObservation] = []
        self.current_step: int = 0
        self.is_prefill: bool = True
        self._is_attached: bool = False

    def attach(self, model: Any) -> None:
        """Attaches forward hooks to all layer router modules identified by the adapter."""
        if self._is_attached:
            self.remove()

        routers = self.adapter.get_router_modules(model)
        for layer_idx, router_module in routers.items():
            hook = router_module.register_forward_hook(self._create_hook_fn(layer_idx))
            self.hooks.append(hook)

        self._is_attached = True
        logger.debug("Attached router observation hooks across %d layers.", len(routers))

    def remove(self) -> None:
        """Safely removes all active PyTorch forward hooks."""
        for hook in self.hooks:
            try:
                hook.remove()
            except Exception as e:
                logger.warning("Error removing router hook: %s", e)
        self.hooks.clear()
        self._is_attached = False

    def reset(self) -> None:
        """Clears captured observation history and resets step counter."""
        self.observations.clear()
        self.current_step = 0
        self.is_prefill = True

    def set_prefill(self, is_prefill: bool) -> None:
        """Sets prefill state flag."""
        self.is_prefill = is_prefill

    def advance_step(self) -> None:
        """Advances token generation step counter for subsequent decode passes."""
        self.current_step += 1
        self.is_prefill = False

    def _create_hook_fn(self, layer_idx: int) -> Callable:
        """Generates a closed forward hook callback for a specific layer."""
        def hook(module: torch.nn.Module, input_args: Any, output: Any) -> None:
            try:
                obs = self.adapter.extract_observation(
                    module_output=output,
                    layer_id=layer_idx,
                    step_id=self.current_step,
                    is_prefill=self.is_prefill,
                )
                self.observations.append(obs)
                if self.on_observation is not None:
                    self.on_observation(obs)
            except Exception as ex:
                logger.error("Failed to extract routing observation at layer %d: %s", layer_idx, ex)

        return hook

    def __enter__(self) -> "RouterHookManager":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.remove()
