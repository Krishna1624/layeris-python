"""
Layeris Router Observation Interface
====================================

Defines the authoritative observation structure emitted during live MoE forward passes.
Distinguishes between prefill sequences and iterative decode steps.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass(frozen=True)
class RouterObservation:
    """Live routing observation recorded from an authoritative MoE router gate.
    
    Attributes:
        layer_id: Transformer block layer index (0 .. total_layers - 1).
        step_id: Token generation step index.
        selected_expert_ids: List of active expert IDs selected by the router.
            For single-token decode: [e_1, e_2, ..., e_k]
            For multi-token prefill or speculative verification: flattened or primary active set.
        routing_weights: Optional normalized routing weights corresponding to selected_expert_ids.
        router_logits: Optional raw router logits for all experts in the layer.
        seq_len: Sequence length of the current forward pass (1 for standard decode).
        is_prefill: True if this observation originates from prompt prefill.
    """
    layer_id: int
    step_id: int
    selected_expert_ids: List[int]
    routing_weights: Optional[List[float]] = None
    router_logits: Optional[List[float]] = None
    seq_len: int = 1
    is_prefill: bool = False

    @property
    def primary_experts(self) -> List[int]:
        """Returns unique expert IDs active in this observation."""
        seen = set()
        unique = []
        for eid in self.selected_expert_ids:
            if eid not in seen:
                seen.add(eid)
                unique.append(eid)
        return unique

    def get_token_experts(self, token_idx: int, top_k: int) -> List[int]:
        """Extracts the top-k experts for a specific token index in a multi-token batch."""
        if self.seq_len <= 1:
            return list(self.selected_expert_ids[:top_k])
        start = token_idx * top_k
        end = start + top_k
        return list(self.selected_expert_ids[start:end])
