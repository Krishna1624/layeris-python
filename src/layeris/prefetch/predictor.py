"""Multi-Layer Expert Predictor prototype for Layeris MoE inference runtime.

Implements the multi-layer expert prediction, priority scoring, resource adaptation,
and closed-loop feedback tracking faithfully mirroring `c/multi_layer_predict.h`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Tuple


# Architectural limits from c/multi_layer_predict.h
COLI_PREDICT_MAX_DEPTH = 16
COLI_PREDICT_MAX_K = 64
COLI_MAX_HIST_LAYERS = 128
COLI_MAX_HIST_TOKENS = 16
COLI_MAX_HIST_TOPK = 16
COLI_PREDICT_WS_MAX_DIM = 16384
COLI_PREDICT_WS_MAX_EXPERTS = 1024
COLI_PREDICT_MAX_CANDIDATES = COLI_PREDICT_MAX_DEPTH * COLI_PREDICT_MAX_K


@dataclass
class PredictConfig:
    """Configuration for multi-layer expert prediction."""
    depth: int = 3
    top_k: int = 4
    enabled: bool = True
    min_score: float = -1e20
    min_confidence: float = 0.0
    top_p_mass: float = 0.0
    use_history: bool = True
    max_history_tokens: int = 8
    adaptive_resource: bool = True

    def __post_init__(self):
        if self.depth < 1:
            self.depth = 1
        elif self.depth > COLI_PREDICT_MAX_DEPTH:
            self.depth = COLI_PREDICT_MAX_DEPTH

        if self.top_k < 1:
            self.top_k = 1
        elif self.top_k > COLI_PREDICT_MAX_K:
            self.top_k = COLI_PREDICT_MAX_K


@dataclass
class LayerPrediction:
    """Predicted candidate experts for a single target layer."""
    layer: int
    expert_ids: List[int] = field(default_factory=list)
    scores: List[float] = field(default_factory=list)
    confidences: List[float] = field(default_factory=list)

    @property
    def num_candidates(self) -> int:
        return len(self.expert_ids)


@dataclass
class MultiLayerPrediction:
    """Bundle of candidate predictions spanning upcoming target layers."""
    source_layer: int
    layers: List[LayerPrediction] = field(default_factory=list)

    @property
    def num_layers(self) -> int:
        return len(self.layers)


@dataclass
class CandidateExpert:
    """Individual candidate expert enriched with priority and dispatch metadata."""
    layer: int
    expert_id: int
    raw_score: float = 0.0
    confidence: float = 0.0
    priority_score: float = 0.0
    layer_distance: int = 1
    token_distance: int = 0
    is_resident: bool = False
    is_late: bool = False
    is_duplicate: bool = False


@dataclass
class CandidateQueue:
    """Bounded, prioritized queue of candidate experts ready for prefetch dispatch."""
    items: List[CandidateExpert] = field(default_factory=list)

    @property
    def count(self) -> int:
        return len(self.items)

    def to_keys(self, uncached_only: bool = True) -> List[Tuple[int, int]]:
        """Flatten candidate queue to a list of (layer, expert_id) keys."""
        keys = []
        for cand in self.items:
            if uncached_only and cand.is_resident:
                continue
            keys.append((cand.layer, cand.expert_id))
        return keys


@dataclass
class ResourceState:
    """System memory and I/O queue resource headroom."""
    available_ram_bytes: int = 16 * 1024 * 1024 * 1024
    available_vram_bytes: int = 8 * 1024 * 1024 * 1024
    resident_expert_bytes: int = 4 * 1024 * 1024 * 1024
    expert_size_bytes: int = 16 * 1024 * 1024
    pending_io_jobs: int = 0
    max_concurrent_io: int = 16


@dataclass
class CacheFeedbackTracker:
    """Tracks prefetch utility and dynamically modulates confidence bias.
    
    Mirrors ColiCacheFeedbackTracker in c/multi_layer_predict.h.
    """
    total_prefetches: int = 0
    useful_prefetches: int = 0
    wasted_prefetches: int = 0
    demand_hits: int = 0
    demand_misses: int = 0
    current_confidence_bias: float = 0.0

    def record(self, was_prefetched: bool, was_hit: bool) -> None:
        if was_prefetched:
            self.total_prefetches += 1
            if was_hit:
                self.useful_prefetches += 1
            else:
                self.wasted_prefetches += 1
        else:
            if was_hit:
                self.demand_hits += 1
            else:
                self.demand_misses += 1

        # Dynamic bias adjustment: if waste ratio > 35% across at least 16 prefetches, raise bar
        if self.total_prefetches >= 16:
            waste_ratio = self.wasted_prefetches / self.total_prefetches
            if waste_ratio > 0.35:
                self.current_confidence_bias = (waste_ratio - 0.35) * 0.5
            else:
                self.current_confidence_bias = 0.0


@dataclass
class PredictionEvaluator:
    """Closed-loop prediction accuracy and prefetch timeliness tracker."""
    total_predictions: int = 0
    total_actual_routed: int = 0
    true_positives: int = 0
    unused_prefetches: int = 0
    cache_misses: int = 0
    timely_prefetches: int = 0
    total_latency_us: int = 0
    step_count: int = 0

    def record_step(
        self,
        planned_queue: Optional[CandidateQueue],
        layer: int,
        actual_eids: List[int],
        is_timely: bool = True,
        step_latency_us: int = 0,
        feedback: Optional[CacheFeedbackTracker] = None,
    ) -> None:
        if not actual_eids:
            return

        self.step_count += 1
        self.total_actual_routed += len(actual_eids)
        self.total_latency_us += step_latency_us

        hits_for_layer = 0
        actual_set = set(actual_eids)

        if planned_queue:
            for item in planned_queue.items:
                if item.layer == layer:
                    self.total_predictions += 1
                    was_hit = item.expert_id in actual_set
                    if was_hit:
                        self.true_positives += 1
                        hits_for_layer += 1
                        if is_timely:
                            self.timely_prefetches += 1
                        if feedback:
                            feedback.record(was_prefetched=True, was_hit=True)
                    else:
                        self.unused_prefetches += 1
                        if feedback:
                            feedback.record(was_prefetched=True, was_hit=False)

        misses = max(0, len(actual_eids) - hits_for_layer)
        self.cache_misses += misses

    def compute_summary(self) -> dict:
        precision = (
            self.true_positives / self.total_predictions
            if self.total_predictions > 0
            else 0.0
        )
        recall = (
            self.true_positives / self.total_actual_routed
            if self.total_actual_routed > 0
            else 0.0
        )
        unused_rate = (
            self.unused_prefetches / self.total_predictions
            if self.total_predictions > 0
            else 0.0
        )
        timely_rate = (
            self.timely_prefetches / self.total_predictions
            if self.total_predictions > 0
            else 0.0
        )
        miss_rate = (
            self.cache_misses / self.total_actual_routed
            if self.total_actual_routed > 0
            else 0.0
        )
        avg_latency = (
            self.total_latency_us / self.step_count if self.step_count > 0 else 0.0
        )

        return {
            "total_predictions": self.total_predictions,
            "total_actual_routed": self.total_actual_routed,
            "true_positives": self.true_positives,
            "unused_prefetches": self.unused_prefetches,
            "cache_misses": self.cache_misses,
            "timely_prefetches": self.timely_prefetches,
            "precision": precision,
            "recall": recall,
            "unused_prefetch_rate": unused_rate,
            "timely_prefetch_rate": timely_rate,
            "cache_miss_rate": miss_rate,
            "avg_step_latency_us": avg_latency,
        }


class RoutingHistory:
    """Bounded historical routing observations across tokens and layers.
    
    Matches ColiRoutingHistory in c/multi_layer_predict.h:
    - Ring buffer of size 16 tokens.
    - Exponential decay by 0.90 per token step.
    - Direct recency tracking per token slot.
    """

    def __init__(self, total_layers: int = 32, num_experts: int = 64):
        self.total_layers = min(max(1, total_layers), COLI_MAX_HIST_LAYERS)
        self.num_experts = min(max(1, num_experts), COLI_PREDICT_WS_MAX_EXPERTS)
        self.ring_head = 0
        self.recorded_tokens = 0
        # [layer][slot] -> list of expert IDs
        self.history: List[List[List[int]]] = [
            [[] for _ in range(COLI_MAX_HIST_TOKENS)] for _ in range(self.total_layers)
        ]
        # [layer][expert] -> decayed frequency
        self.expert_freq: List[List[float]] = [
            [0.0 for _ in range(self.num_experts)] for _ in range(self.total_layers)
        ]

    def advance_token(self) -> None:
        """Advance token ring buffer and apply exponential decay (0.90) to frequencies."""
        self.ring_head = (self.ring_head + 1) % COLI_MAX_HIST_TOKENS
        if self.recorded_tokens < COLI_MAX_HIST_TOKENS:
            self.recorded_tokens += 1

        slot = self.ring_head
        for l in range(self.total_layers):
            self.history[l][slot].clear()
            for e in range(self.num_experts):
                self.expert_freq[l][e] *= 0.90

    def record(self, layer: int, expert_ids: List[int]) -> None:
        """Record routed experts for a layer at the current token position."""
        if not (0 <= layer < self.total_layers) or not expert_ids:
            return

        slot = self.ring_head
        # Deduplicate and bound to COLI_MAX_HIST_TOPK
        seen = set()
        recorded = []
        for eid in expert_ids:
            if eid not in seen and 0 <= eid < self.num_experts:
                seen.add(eid)
                recorded.append(eid)
                self.expert_freq[layer][eid] += 1.0
                if len(recorded) >= COLI_MAX_HIST_TOPK:
                    break
        self.history[layer][slot] = recorded


# Priority scoring logic from Phase G (c/multi_layer_predict.h)
def calculate_priority(
    confidence: float,
    layer_distance: int,
    hist_freq: float,
    is_resident: bool,
    confidence_bias: float = 0.0,
) -> float:
    """Calculates composite priority score for candidate prefetching.
    
    1. Adjusted confidence: penalized by dynamic feedback bias
    2. Temporal discount factor: gamma(dL) = 1.0 / (1.0 + 0.30 * (dL - 1))
    3. Historical activation component normalized via tanh(hist_freq * 0.20)
    4. Base utility = 0.75 * adj_conf + 0.25 * hist_factor
    5. Residency discount: 0.05 if resident (save I/O bandwidth), else 1.0
    """
    adj_conf = max(0.0, confidence - confidence_bias)
    dist = max(1, layer_distance)
    temporal_decay = 1.0 / (1.0 + 0.30 * float(dist - 1))
    hist_factor = math.tanh(hist_freq * 0.20)
    base_utility = 0.75 * adj_conf + 0.25 * hist_factor
    avail_multiplier = 0.05 if is_resident else 1.0
    return base_utility * temporal_decay * avail_multiplier


def passes_confidence(
    confidence: float,
    score: float,
    min_confidence: float,
    min_score: float,
    confidence_bias: float = 0.0,
) -> bool:
    """Independent confidence threshold decision."""
    effective_min_conf = min_confidence + confidence_bias
    if score < min_score:
        return False
    if confidence < effective_min_conf:
        return False
    return True


def adapt_resources(
    cfg: PredictConfig,
    res: Optional[ResourceState],
) -> Tuple[int, int]:
    """Dynamically adapt prefetch depth and top_k based on system resource headroom.
    
    Throttles depth if I/O queue is saturated; reduces K if memory capacity is constrained.
    """
    effective_depth = cfg.depth
    effective_k = cfg.top_k
    if not cfg.adaptive_resource or not res:
        return effective_depth, effective_k

    # Throttling when I/O queue is backed up
    if res.max_concurrent_io > 0 and res.pending_io_jobs > res.max_concurrent_io // 2:
        effective_depth = max(1, effective_depth // 2)
        effective_k = max(2, (effective_k * 3) // 4)

    # Memory constraint adaptation
    if res.expert_size_bytes > 0 and res.available_ram_bytes > 0:
        max_experts_fit = res.available_ram_bytes // res.expert_size_bytes
        if max_experts_fit < effective_depth * effective_k:
            effective_k = max(1, int(max_experts_fit // max(1, effective_depth)))

    return effective_depth, effective_k


def topk_filtered(
    router_logits: List[float],
    num_experts: int,
    top_k: int,
    min_score: float = -1e20,
    min_confidence: float = 0.0,
    top_p_mass: float = 0.0,
    is_sigmoid: bool = False,
) -> Tuple[List[int], List[float], List[float]]:
    """Extract top-K candidate expert IDs with confidence scoring, filtering, and normalization."""
    if not router_logits or num_experts <= 0 or top_k <= 0:
        return [], [], []

    k_target = min(top_k, num_experts, COLI_PREDICT_MAX_K)
    logits = list(router_logits[:num_experts])

    # Convert logits to probabilities
    probs = [0.0] * num_experts
    if is_sigmoid:
        sum_p = 0.0
        for e in range(num_experts):
            val = logits[e]
            if math.isnan(val) or math.isinf(val):
                sig = 0.0
            else:
                sig = 1.0 / (1.0 + math.exp(-max(-50.0, min(50.0, val))))
            probs[e] = sig
            sum_p += sig
        if sum_p > 1e-12:
            for e in range(num_experts):
                probs[e] /= sum_p
    else:
        max_l = -1e30
        for e in range(num_experts):
            val = logits[e]
            if not math.isnan(val) and not math.isinf(val) and val > max_l:
                max_l = val
        if max_l <= -1e29:
            max_l = 0.0

        sum_exp = 0.0
        for e in range(num_experts):
            val = logits[e]
            if math.isnan(val) or math.isinf(val):
                ex = 0.0
            else:
                ex = math.exp(max(-50.0, min(50.0, val - max_l)))
            probs[e] = ex
            sum_exp += ex
        if sum_exp > 1e-12:
            for e in range(num_experts):
                probs[e] /= sum_exp

    # Greedy extraction of top candidates
    selected_ids: List[int] = []
    selected_scores: List[float] = []
    selected_confs: List[float] = []
    cumulative_mass = 0.0

    # Sort finite candidates by logit descending
    valid_candidates = []
    for e in range(num_experts):
        val = logits[e]
        if not math.isnan(val) and not math.isinf(val):
            valid_candidates.append((val, e))
    valid_candidates.sort(key=lambda item: item[0], reverse=True)

    for val, e in valid_candidates:
        if len(selected_ids) >= k_target:
            break
        if val < min_score:
            break
        if probs[e] < min_confidence:
            break

        selected_ids.append(e)
        selected_scores.append(val)
        selected_confs.append(probs[e])
        cumulative_mass += probs[e]

        if top_p_mass > 0.0 and cumulative_mass >= top_p_mass and len(selected_ids) >= 1:
            break

    return selected_ids, selected_scores, selected_confs


class MultiLayerPredictor:
    """Complete predictor coordinating history, hidden router projection, and scheduling."""

    def __init__(
        self,
        total_layers: int = 32,
        num_experts: int = 64,
        config: Optional[PredictConfig] = None,
    ):
        self.total_layers = total_layers
        self.num_experts = num_experts
        self.config = config or PredictConfig()
        self.history = RoutingHistory(total_layers, num_experts)
        self.feedback = CacheFeedbackTracker()
        self.evaluator = PredictionEvaluator()
        self.last_planned_queue = CandidateQueue()

    def advance_token(self) -> None:
        """Call at start of each token step."""
        self.history.advance_token()

    def predict_from_history(
        self, target_layer: int, top_k: Optional[int] = None
    ) -> LayerPrediction:
        """Predict candidate experts using historical frequency and recent token activation."""
        if not (0 <= target_layer < self.total_layers):
            return LayerPrediction(target_layer)

        k = top_k if top_k is not None else self.config.top_k
        scores = [0.0] * self.num_experts

        # 1. Base scores from decayed expert frequency
        for e in range(self.num_experts):
            scores[e] = self.history.expert_freq[target_layer][e]

        # 2. Boost experts that were active in the most recent token at target layer
        slot = self.history.ring_head
        recent_eids = self.history.history[target_layer][slot]
        for eid in recent_eids:
            if 0 <= eid < self.num_experts:
                scores[eid] += 2.0

        ids, scs, confs = topk_filtered(
            scores,
            self.num_experts,
            k,
            min_score=self.config.min_score,
            min_confidence=self.config.min_confidence,
            top_p_mass=self.config.top_p_mass,
            is_sigmoid=False,
        )
        return LayerPrediction(target_layer, ids, scs, confs)

    def predict_from_hidden(
        self,
        hidden_state: List[float],
        router_weight: List[List[float]],
        router_bias: Optional[List[float]] = None,
        ln_weight: Optional[List[float]] = None,
        ln_eps: float = 1e-6,
        top_k: Optional[int] = None,
        is_sigmoid: bool = False,
        target_layer: int = 0,
    ) -> LayerPrediction:
        """Predict candidate experts from hidden representations via RMSNorm + Router projection."""
        hidden_dim = len(hidden_state)
        if hidden_dim == 0 or len(router_weight) != self.num_experts:
            return LayerPrediction(target_layer)

        k = top_k if top_k is not None else self.config.top_k

        # Apply RMSNorm if normalization weights are present
        if ln_weight and len(ln_weight) == hidden_dim:
            sumsq = sum(x * x for x in hidden_state)
            scale = 1.0 / math.sqrt(sumsq / float(hidden_dim) + ln_eps)
            nrm = [hidden_state[i] * scale * ln_weight[i] for i in range(hidden_dim)]
        else:
            nrm = list(hidden_state)

        # Router projection: logits = nrm * router_weight + bias
        logits = [0.0] * self.num_experts
        for e in range(self.num_experts):
            rw = router_weight[e]
            dot = sum(nrm[d] * rw[d] for d in range(hidden_dim))
            if router_bias and e < len(router_bias):
                dot += router_bias[e]
            logits[e] = dot

        ids, scs, confs = topk_filtered(
            logits,
            self.num_experts,
            k,
            min_score=self.config.min_score,
            min_confidence=self.config.min_confidence,
            top_p_mass=self.config.top_p_mass,
            is_sigmoid=is_sigmoid,
        )
        return LayerPrediction(target_layer, ids, scs, confs)

    def predict_from_speculative(
        self,
        draft_tokens: List[int],
        draft_confs: Optional[List[float]],
        target_layer: int,
        top_k: Optional[int] = None,
    ) -> LayerPrediction:
        """Predict upcoming layer candidate experts using speculative draft token signals."""
        if not draft_tokens or not (0 <= target_layer < self.total_layers):
            return LayerPrediction(target_layer)

        k = top_k if top_k is not None else self.config.top_k
        scores = [0.0] * self.num_experts

        for e in range(self.num_experts):
            scores[e] = self.history.expert_freq[target_layer][e] * 0.5

        for idx, tok in enumerate(draft_tokens):
            conf = draft_confs[idx] if draft_confs and idx < len(draft_confs) else 1.0
            pseudo_expert = (tok % self.num_experts) if tok >= 0 else 0
            scores[pseudo_expert] += conf * 2.5

        ids, scs, confs = topk_filtered(
            scores,
            self.num_experts,
            k,
            min_score=self.config.min_score,
            min_confidence=self.config.min_confidence,
            top_p_mass=self.config.top_p_mass,
            is_sigmoid=False,
        )
        return LayerPrediction(target_layer, ids, scs, confs)

    def schedule_candidates(
        self,
        pred: MultiLayerPrediction,
        current_layer: int,
        resource: Optional[ResourceState] = None,
        is_resident_fn: Optional[Callable[[int, int], bool]] = None,
    ) -> CandidateQueue:
        """Score, rank, and schedule candidate experts into a bounded candidate queue.
        
        Applies:
        - Resource adaptation (effective depth and K)
        - Dynamic confidence filtering penalized by waste feedback bias
        - Priority scoring with temporal decay, historical activation, and residency discounting
        - Priority descending ranking
        - I/O budget truncation
        """
        if not self.config.enabled or pred.num_layers == 0:
            return CandidateQueue()

        eff_depth, eff_k = adapt_resources(self.config, resource)

        max_io_budget = COLI_PREDICT_MAX_CANDIDATES
        if resource and resource.max_concurrent_io > 0:
            avail_io = resource.max_concurrent_io - resource.pending_io_jobs
            max_io_budget = max(1, avail_io)

        bias = self.feedback.current_confidence_bias
        candidates: List[CandidateExpert] = []
        max_layers = min(eff_depth, pred.num_layers)

        for l_idx in range(max_layers):
            lp = pred.layers[l_idx]
            target_layer = lp.layer
            dL = target_layer - current_layer
            if dL <= 0:
                dL = l_idx + 1

            k_limit = min(eff_k, lp.num_candidates)
            for k in range(k_limit):
                eid = lp.expert_ids[k]
                score = lp.scores[k]
                conf = lp.confidences[k]

                # Confidence check with dynamic bias
                if not passes_confidence(
                    conf, score, self.config.min_confidence, self.config.min_score, bias
                ):
                    continue

                is_res = False
                if is_resident_fn:
                    is_res = is_resident_fn(target_layer, eid)

                freq = 0.0
                if 0 <= target_layer < self.history.total_layers and 0 <= eid < self.history.num_experts:
                    freq = self.history.expert_freq[target_layer][eid]

                prio = calculate_priority(conf, dL, freq, is_res, bias)

                cand = CandidateExpert(
                    layer=target_layer,
                    expert_id=eid,
                    raw_score=score,
                    confidence=conf,
                    priority_score=prio,
                    layer_distance=dL,
                    is_resident=is_res,
                )
                candidates.append(cand)

        # Rank candidates by priority descending
        candidates.sort(key=lambda c: c.priority_score, reverse=True)

        # Budget truncation
        if len(candidates) > max_io_budget:
            candidates = candidates[:max_io_budget]

        queue = CandidateQueue(items=candidates)
        self.last_planned_queue = queue
        return queue

    def on_layer_routed(
        self,
        layer: int,
        routed_experts: List[int],
        is_timely: bool = True,
        step_latency_us: int = 0,
    ) -> None:
        """Closed-loop notification when layer routing decision is finalized."""
        self.history.record(layer, routed_experts)
        self.evaluator.record_step(
            planned_queue=self.last_planned_queue,
            layer=layer,
            actual_eids=routed_experts,
            is_timely=is_timely,
            step_latency_us=step_latency_us,
            feedback=self.feedback,
        )

    def get_metrics(self) -> dict:
        """Fetch aggregate evaluation metrics."""
        return self.evaluator.compute_summary()


# Parity aliases with C runtime naming
ColiPredictConfig = PredictConfig
ColiMultiLayerPredictRuntime = MultiLayerPredictor
passes_confidence_filter = passes_confidence


def schedule_candidates(
    predictor: MultiLayerPredictor,
    prediction: MultiLayerPrediction,
    is_resident_fn: Optional[Callable[[int, int], bool]] = None,
) -> PlannedCandidateQueue:
    """Module-level wrapper for candidate scheduling."""
    return predictor.schedule_candidates(prediction, is_resident_fn)
