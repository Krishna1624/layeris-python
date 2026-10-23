"""
Layeris Live MoE Predictor and Telemetry Coordinator
===================================================

Connects the multi-layer expert predictor (from `layeris.prefetch.predictor`)
with the live MoE routing observation stream from `RouterHookManager`.
Provides non-invasive online candidate prediction, priority scheduling,
live cache telemetry, and closed-loop feedback tracking during model generation.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

from layeris.moe.config import MoEConfig
from layeris.moe.observation import RouterObservation
from layeris.prefetch.predictor import (
    CandidateExpert,
    CandidateQueue,
    LayerPrediction,
    MultiLayerPrediction,
    MultiLayerPredictor,
    PredictConfig,
    ResourceState,
)
from layeris.prefetch.simulator import ExpertCacheSimulator, SimulationStats

logger = logging.getLogger(__name__)


@dataclass
class StepTelemetry:
    """Telemetry diagnostic record for a single layer observation step."""
    layer_id: int
    step_id: int
    is_prefill: bool
    routed_experts: List[int]
    planned_candidates: List[Tuple[int, int, float, float]]  # (target_layer, expert_id, score, conf)
    effective_depth: int
    effective_k: int
    confidence_bias: float


class LiveMoEPredictor:
    """Coordinates live router observations with multi-layer expert prediction.
    
    Guarantees:
    1. Zero side effects on authoritative model outputs, tokens, or weights.
    2. Strict temporal ordering: predictions are made only for upcoming layers (dL >= 1).
    3. Transparent tracking of precision, recall, useful/wasted prefetches, and cache residency.
    """

    def __init__(
        self,
        moe_config: MoEConfig,
        predict_config: Optional[PredictConfig] = None,
        cache_capacity_per_layer: Optional[int] = None,
        resource_state: Optional[ResourceState] = None,
    ):
        self.moe_config = moe_config
        self.config = predict_config or PredictConfig(
            depth=3,
            top_k=min(moe_config.num_experts_per_tok * 2, moe_config.num_experts),
            enabled=True,
            min_confidence=0.0,
            adaptive_resource=True,
        )
        self.total_layers = moe_config.num_layers
        self.num_experts = moe_config.num_experts
        self.predictor = MultiLayerPredictor(
            total_layers=self.total_layers,
            num_experts=self.num_experts,
            config=self.config,
        )
        self.resource_state = resource_state or ResourceState()

        # Optional bounded live cache tracker (mirrors hardware slab cache residency)
        self.cache: Optional[ExpertCacheSimulator] = None
        if cache_capacity_per_layer is not None and cache_capacity_per_layer > 0:
            self.cache = ExpertCacheSimulator(
                capacity_per_layer=cache_capacity_per_layer,
                num_layers=self.total_layers,
                num_experts=self.num_experts,
                eviction_policy="predictive_lfru",
            )

        self.telemetry_history: List[StepTelemetry] = []
        self.current_step: int = 0
        self.total_observations: int = 0

    def reset(self) -> None:
        """Resets predictor state, history, and evaluation telemetry."""
        self.predictor = MultiLayerPredictor(
            total_layers=self.total_layers,
            num_experts=self.num_experts,
            config=self.config,
        )
        if self.cache is not None:
            self.cache.reset()
        self.telemetry_history.clear()
        self.current_step = 0
        self.total_observations = 0

    def on_observation(self, obs: RouterObservation) -> Optional[CandidateQueue]:
        """Processes a live routing observation emitted by a router forward hook.
        
        Args:
            obs: The authoritative routing observation from the current layer.
            
        Returns:
            The scheduled candidate queue for upcoming layers, or None if disabled or during prefill.
        """
        self.total_observations += 1

        if not self.config.enabled:
            return None

        # 1. Handle prompt prefill (prime historical routing frequencies)
        if obs.is_prefill:
            if obs.seq_len > 1:
                # Multiple tokens in prefill sequence
                for t in range(obs.seq_len):
                    tok_eids = obs.get_token_experts(t, self.moe_config.num_experts_per_tok)
                    self.predictor.history.record(obs.layer_id, tok_eids)
            else:
                self.predictor.history.record(obs.layer_id, obs.primary_experts)
            return None

        # 2. Decode pass step progression
        if obs.layer_id == 0:
            self.predictor.advance_token()
            self.current_step = obs.step_id

        # 3. Closed-loop verification of previously planned candidates for this layer
        routed = obs.primary_experts
        self.predictor.on_layer_routed(
            layer=obs.layer_id,
            routed_experts=routed,
            is_timely=True,
            step_latency_us=0,
        )

        if self.cache is not None:
            for eid in routed:
                self.cache.access_expert(obs.layer_id, eid)

        # 4. Multi-layer lookahead planning for upcoming layers (horizon dL = 1 .. depth)
        bundle_layers: List[LayerPrediction] = []
        eff_depth = min(self.config.depth, self.total_layers - 1 - obs.layer_id)

        for dL in range(1, eff_depth + 1):
            target_l = obs.layer_id + dL
            if target_l < self.total_layers:
                lp = self.predictor.predict_from_history(target_l, top_k=self.config.top_k)
                bundle_layers.append(lp)

        if not bundle_layers:
            # Reached final layers of the model
            return None

        ml_pred = MultiLayerPrediction(source_layer=obs.layer_id, layers=bundle_layers)

        # 5. Candidate priority scheduling and resource adaptation
        is_res_fn = self.cache.is_resident if self.cache else None
        planned_q = self.predictor.schedule_candidates(
            pred=ml_pred,
            current_layer=obs.layer_id,
            resource=self.resource_state,
            is_resident_fn=is_res_fn,
        )

        # 6. Dispatch and protect upcoming candidates in cache
        if self.cache is not None:
            self.cache.set_protected_experts(planned_q.to_keys(uncached_only=False))
            for target_l, eid in planned_q.to_keys(uncached_only=True):
                self.cache.dispatch_prefetch(target_l, eid)
            self.cache.step_layer_compute()

        # 7. Record structured step telemetry
        candidates_summary = [
            (c.layer, c.expert_id, round(c.priority_score, 4), round(c.confidence, 4))
            for c in planned_q.items
        ]
        telemetry = StepTelemetry(
            layer_id=obs.layer_id,
            step_id=obs.step_id,
            is_prefill=obs.is_prefill,
            routed_experts=list(routed),
            planned_candidates=candidates_summary,
            effective_depth=len(bundle_layers),
            effective_k=self.config.top_k,
            confidence_bias=round(self.predictor.feedback.current_confidence_bias, 4),
        )
        self.telemetry_history.append(telemetry)

        return planned_q

    def get_diagnostics(self) -> Dict[str, Any]:
        """Consolidates predictor accuracy, feedback bias, and cache simulation telemetry."""
        metrics = self.predictor.get_metrics()
        diag: Dict[str, Any] = {
            "enabled": self.config.enabled,
            "total_layers": self.total_layers,
            "num_experts": self.num_experts,
            "prediction_depth": self.config.depth,
            "top_k": self.config.top_k,
            "total_observations": self.total_observations,
            "predictor_metrics": metrics,
            "feedback_confidence_bias": self.predictor.feedback.current_confidence_bias,
            "waste_count": self.predictor.feedback.wasted_prefetches,
            "useful_count": self.predictor.feedback.useful_prefetches,
            "recent_telemetry_steps": len(self.telemetry_history),
        }
        if self.cache is not None:
            stats = self.cache.stats
            diag["cache_simulation"] = {
                "capacity_per_layer": self.cache.capacity_per_layer,
                "hit_rate_pct": round(stats.hit_rate, 2),
                "miss_rate_pct": round(stats.miss_rate, 2),
                "cache_hits": stats.cache_hits,
                "cache_misses": stats.cache_misses,
                "demand_loads": stats.demand_loads,
                "prefetch_dispatches": stats.prefetch_dispatches,
                "useful_prefetches": stats.useful_prefetches,
                "wasted_prefetches": stats.wasted_prefetches,
                "useful_prefetch_rate_pct": round(stats.useful_prefetch_rate, 2),
                "wasted_prefetch_rate_pct": round(stats.wasted_prefetch_rate, 2),
                "peak_resident_experts": stats.peak_resident_experts,
            }
        return diag
