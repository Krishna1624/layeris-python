"""
Qwen3.6 MoE Architecture Adapter
================================

Authoritative adapter for Qwen 3.5 / 3.6 / 3.8 MoE models (e.g. Qwen3_5MoeForCausalLM).
Hooks into Qwen3_5MoeTopKRouter (`model.layers[i].mlp.gate`) to extract the exact
router indices and weights without altering execution.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import torch

from layeris.moe.adapter import MoEArchitectureAdapter
from layeris.moe.config import MoEConfig
from layeris.moe.observation import RouterObservation


class Qwen36MoEAdapter(MoEArchitectureAdapter):
    """Architecture adapter for Qwen 3.6 MoE."""

    def __init__(self, config: MoEConfig):
        super().__init__(config)

    def get_router_modules(self, model: Any) -> Dict[int, torch.nn.Module]:
        """Locates the `gate` module inside each transformer layer's MoE block."""
        trunk = getattr(model, "model", model)
        if not hasattr(trunk, "layers"):
            raise AttributeError(f"Model {type(model)} does not have a 'layers' attribute.")

        routers: Dict[int, torch.nn.Module] = {}
        for idx, layer in enumerate(trunk.layers):
            if not hasattr(layer, "mlp"):
                continue
            mlp = layer.mlp
            if hasattr(mlp, "gate"):
                routers[idx] = mlp.gate
            elif hasattr(mlp, "router"):
                routers[idx] = mlp.router

        if not routers:
            raise RuntimeError(
                f"No router modules found in Qwen MoE model layers (inspected {len(trunk.layers)} layers)."
            )
        return routers

    def extract_observation(
        self,
        module_output: Any,
        layer_id: int,
        step_id: int,
        is_prefill: bool,
    ) -> RouterObservation:
        """Parses Qwen3_5MoeTopKRouter output: (router_logits, router_scores, router_indices)."""
        router_logits_t: Optional[torch.Tensor] = None
        router_scores_t: Optional[torch.Tensor] = None
        router_indices_t: Optional[torch.Tensor] = None

        if isinstance(module_output, (tuple, list)):
            if len(module_output) >= 3:
                router_logits_t, router_scores_t, router_indices_t = module_output[:3]
            elif len(module_output) == 2:
                router_scores_t, router_indices_t = module_output[:2]
        elif isinstance(module_output, dict):
            router_logits_t = module_output.get("router_logits")
            router_scores_t = module_output.get("router_scores") or module_output.get("weights")
            router_indices_t = module_output.get("router_indices") or module_output.get("selected_experts")
        elif isinstance(module_output, torch.Tensor):
            # Fallback if only logits returned
            router_logits_t = module_output
            probs = torch.softmax(router_logits_t, dim=-1)
            router_scores_t, router_indices_t = torch.topk(probs, self.top_k, dim=-1)

        if router_indices_t is None:
            raise ValueError(f"Unable to extract router indices from gate output at layer {layer_id}: {type(module_output)}")

        # Detach cleanly without retaining computation graph
        indices_cpu = router_indices_t.detach().cpu()
        seq_len = indices_cpu.shape[0] if indices_cpu.ndim >= 2 else 1

        # Flatten into list of integers
        selected_eids: List[int] = indices_cpu.reshape(-1).tolist()

        routing_weights: Optional[List[float]] = None
        if router_scores_t is not None:
            routing_weights = router_scores_t.detach().cpu().reshape(-1).tolist()

        router_logits: Optional[List[float]] = None
        if router_logits_t is not None and not is_prefill:
            # Only retain logits during single-token decode to bound memory
            router_logits = router_logits_t.detach().cpu().reshape(-1).tolist()

        return RouterObservation(
            layer_id=layer_id,
            step_id=step_id,
            selected_expert_ids=selected_eids,
            routing_weights=routing_weights,
            router_logits=router_logits,
            seq_len=seq_len,
            is_prefill=is_prefill,
        )
