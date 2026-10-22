"""Reproducible loading and generation of expert-routing sequences for Layeris MoE evaluation.

Supports:
1. Synthetic coupled MoE routing (cross-layer transition clusters modeling real MoE copula).
2. Zipf-skewed temporal routing with topic drift.
3. Uniform random adversarial routing.
4. Parsing and exporting standard Layeris ROUTE_TRACE logs (<call> <row> <layer> <expert>:<gate>).
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple


@dataclass(frozen=True)
class RoutingEvent:
    """A single layer's expert routing activation event within a token forward pass."""
    token_idx: int
    layer: int
    experts: Tuple[int, ...]
    gates: Tuple[float, ...] = ()
    call_idx: int = 0
    row_idx: int = 0


@dataclass
class RoutingTrace:
    """A complete sequence of routing events across tokens and layers."""
    num_layers: int
    num_experts: int
    top_k: int
    num_tokens: int
    events: List[RoutingEvent] = field(default_factory=list)
    metadata: Dict[str, str] = field(default_factory=dict)

    def events_for_token(self, token_idx: int) -> List[RoutingEvent]:
        return [e for e in self.events if e.token_idx == token_idx]

    def to_route_trace_lines(self) -> List[str]:
        """Convert events to canonical Layeris ROUTE_TRACE text lines."""
        lines = []
        for ev in self.events:
            gates = ev.gates if len(ev.gates) == len(ev.experts) else [1.0 / len(ev.experts)] * len(ev.experts)
            pairs = " ".join(f"{e}:{g:.4f}" for e, g in zip(ev.experts, gates))
            lines.append(f"{ev.call_idx} {ev.row_idx} {ev.layer} {pairs}")
        return lines

    def save_to_file(self, path: Path) -> None:
        """Write trace in ROUTE_TRACE format."""
        path.write_text("\n".join(self.to_route_trace_lines()) + "\n", encoding="utf-8")


def parse_route_trace_file(path: Path, top_k: int = 4) -> RoutingTrace:
    """Parse standard Layeris ROUTE_TRACE file (<call> <row> <layer> <expert>:<gate> ...)."""
    events = []
    max_layer = -1
    max_expert = -1
    token_idx = 0
    prev_layer = -1

    lines = path.read_text(encoding="utf-8").splitlines()
    for raw in lines:
        parts = raw.strip().split()
        if len(parts) < 4:
            continue
        try:
            call = int(parts[0])
            row = int(parts[1])
            layer = int(parts[2])
        except ValueError:
            continue

        if layer < prev_layer:
            token_idx += 1
        prev_layer = layer

        experts = []
        gates = []
        for item in parts[3:]:
            if ":" in item:
                e_str, g_str = item.split(":", 1)
                try:
                    eid = int(e_str)
                    gate = float(g_str)
                except ValueError:
                    continue
            else:
                try:
                    eid = int(item)
                    gate = 1.0
                except ValueError:
                    continue

            experts.append(eid)
            gates.append(gate)
            if eid > max_expert:
                max_expert = eid

        if layer > max_layer:
            max_layer = layer

        events.append(
            RoutingEvent(
                token_idx=token_idx,
                layer=layer,
                experts=tuple(experts),
                gates=tuple(gates),
                call_idx=call,
                row_idx=row,
            )
        )

    num_layers = max_layer + 1 if max_layer >= 0 else 32
    num_experts = max_expert + 1 if max_expert >= 0 else 64
    num_tokens = token_idx + 1

    return RoutingTrace(
        num_layers=num_layers,
        num_experts=num_experts,
        top_k=top_k,
        num_tokens=num_tokens,
        events=events,
        metadata={"source": str(path)},
    )


def generate_coupled_trace(
    num_layers: int = 32,
    num_experts: int = 64,
    top_k: int = 4,
    num_tokens: int = 100,
    coupling_strength: float = 0.85,
    seed: int = 42,
) -> RoutingTrace:
    """Generate synthetic routing sequence with cross-layer Markovian coupling.
    
    Models real MoE properties where experts in layer L strongly condition activations
    in layer L+1 and L+2 into affinity clusters (as observed in Layeris route_pairs.py).
    """
    rng = random.Random(seed)
    events: List[RoutingEvent] = []

    # Create synthetic cluster communities (e.g. 8 functional clusters of experts)
    num_clusters = max(2, num_experts // 8)
    cluster_experts: List[List[int]] = [[] for _ in range(num_clusters)]
    for e in range(num_experts):
        c_id = e % num_clusters
        cluster_experts[c_id].append(e)

    for tok in range(num_tokens):
        # A token activates 1-2 dominant topic clusters
        active_clusters = rng.sample(range(num_clusters), k=min(2, num_clusters))
        current_active_experts: List[int] = []

        for l in range(num_layers):
            layer_experts: List[int] = []

            # With probability coupling_strength, pull from current cluster or coupled predecessor
            if l == 0 or rng.random() > coupling_strength or not current_active_experts:
                # Sample from active clusters
                pool = []
                for c in active_clusters:
                    pool.extend(cluster_experts[c])
                layer_experts = rng.sample(pool, k=min(top_k, len(pool)))
            else:
                # Coupled transition: pick affine experts linked to predecessor layer's experts
                pool = []
                for pred_e in current_active_experts:
                    c = pred_e % num_clusters
                    pool.extend(cluster_experts[c])
                pool = list(set(pool))
                if len(pool) >= top_k:
                    layer_experts = rng.sample(pool, k=top_k)
                else:
                    layer_experts = list(pool)
                    remaining = [e for e in range(num_experts) if e not in layer_experts]
                    layer_experts.extend(rng.sample(remaining, k=top_k - len(layer_experts)))

            current_active_experts = layer_experts

            # Synthetic gate softmax
            raw_gates = [rng.uniform(0.5, 3.0) for _ in range(top_k)]
            sum_g = sum(raw_gates)
            norm_gates = tuple(g / sum_g for g in raw_gates)

            events.append(
                RoutingEvent(
                    token_idx=tok,
                    layer=l,
                    experts=tuple(layer_experts),
                    gates=norm_gates,
                    call_idx=tok * num_layers + l,
                    row_idx=0,
                )
            )

    return RoutingTrace(
        num_layers=num_layers,
        num_experts=num_experts,
        top_k=top_k,
        num_tokens=num_tokens,
        events=events,
        metadata={"pattern": "coupled_markovian", "coupling_strength": str(coupling_strength), "seed": str(seed)},
    )


def generate_zipf_skewed_trace(
    num_layers: int = 32,
    num_experts: int = 64,
    top_k: int = 4,
    num_tokens: int = 100,
    zipf_s: float = 1.25,
    drift_interval: int = 25,
    seed: int = 42,
) -> RoutingTrace:
    """Generate synthetic routing sequence with Zipfian skew and periodic topic drift.
    
    A small fraction of experts receive disproportionately high activation heat,
    with ranking shuffling every drift_interval tokens (e.g. prompt -> generation transition).
    """
    rng = random.Random(seed)
    events: List[RoutingEvent] = []

    # Zipfian probabilities
    harm = sum(1.0 / (i ** zipf_s) for i in range(1, num_experts + 1))
    base_probs = [(1.0 / (i ** zipf_s)) / harm for i in range(1, num_experts + 1)]

    # Current permutation of experts per layer
    expert_ranks: List[List[int]] = [list(range(num_experts)) for _ in range(num_layers)]
    for l in range(num_layers):
        rng.shuffle(expert_ranks[l])

    for tok in range(num_tokens):
        # Topic drift: occasionally re-shuffle popular experts
        if tok > 0 and tok % drift_interval == 0:
            for l in range(num_layers):
                # Shift ranking partially
                pivot = num_experts // 4
                head = expert_ranks[l][:pivot]
                tail = expert_ranks[l][pivot:]
                rng.shuffle(head)
                rng.shuffle(tail)
                expert_ranks[l] = head + tail

        for l in range(num_layers):
            ranks = expert_ranks[l]
            # Weighted sampling without replacement
            # Sample top_k experts according to base_probs
            chosen = []
            available_indices = list(range(num_experts))
            available_weights = [base_probs[i] for i in range(num_experts)]

            for _ in range(top_k):
                sum_w = sum(available_weights)
                r = rng.uniform(0.0, sum_w)
                acc = 0.0
                selected_idx = 0
                for i, w in enumerate(available_weights):
                    acc += w
                    if acc >= r:
                        selected_idx = i
                        break
                chosen.append(ranks[available_indices[selected_idx]])
                del available_indices[selected_idx]
                del available_weights[selected_idx]

            raw_gates = [rng.uniform(0.5, 3.0) for _ in range(top_k)]
            sum_g = sum(raw_gates)
            norm_gates = tuple(g / sum_g for g in raw_gates)

            events.append(
                RoutingEvent(
                    token_idx=tok,
                    layer=l,
                    experts=tuple(chosen),
                    gates=norm_gates,
                    call_idx=tok * num_layers + l,
                    row_idx=0,
                )
            )

    return RoutingTrace(
        num_layers=num_layers,
        num_experts=num_experts,
        top_k=top_k,
        num_tokens=num_tokens,
        events=events,
        metadata={"pattern": "zipf_skewed", "zipf_s": str(zipf_s), "drift_interval": str(drift_interval), "seed": str(seed)},
    )


def generate_uniform_random_trace(
    num_layers: int = 32,
    num_experts: int = 64,
    top_k: int = 4,
    num_tokens: int = 100,
    seed: int = 42,
) -> RoutingTrace:
    """Generate synthetic uniform random routing sequence (adversarial / stress test).
    
    Zero temporal locality and zero cross-layer correlation. Used to evaluate
    feedback bias stabilization (waste-ratio thresholding prevents runaway prefetching).
    """
    rng = random.Random(seed)
    events: List[RoutingEvent] = []

    for tok in range(num_tokens):
        for l in range(num_layers):
            chosen = rng.sample(range(num_experts), k=top_k)
            raw_gates = [1.0 / top_k] * top_k

            events.append(
                RoutingEvent(
                    token_idx=tok,
                    layer=l,
                    experts=tuple(chosen),
                    gates=tuple(raw_gates),
                    call_idx=tok * num_layers + l,
                    row_idx=0,
                )
            )

    return RoutingTrace(
        num_layers=num_layers,
        num_experts=num_experts,
        top_k=top_k,
        num_tokens=num_tokens,
        events=events,
        metadata={"pattern": "uniform_random", "seed": str(seed)},
    )


def generate_trace_by_pattern(
    pattern: str = "coupled",
    num_layers: int = 32,
    num_experts: int = 64,
    top_k: int = 4,
    num_tokens: int = 100,
    seed: int = 42,
) -> RoutingTrace:
    """Convenience factory for generating routing traces by pattern name."""
    if pattern in ("coupled", "coupled_markovian"):
        return generate_coupled_trace(
            num_layers=num_layers,
            num_experts=num_experts,
            top_k=top_k,
            num_tokens=num_tokens,
            seed=seed,
        )
    elif pattern in ("zipf", "skewed", "zipf_skewed"):
        return generate_zipf_skewed_trace(
            num_layers=num_layers,
            num_experts=num_experts,
            top_k=top_k,
            num_tokens=num_tokens,
            seed=seed,
        )
    elif pattern in ("uniform", "random", "uniform_random"):
        return generate_uniform_random_trace(
            num_layers=num_layers,
            num_experts=num_experts,
            top_k=top_k,
            num_tokens=num_tokens,
            seed=seed,
        )
    else:
        raise ValueError(f"Unknown trace pattern: {pattern}")


# Aliases for intuitive API naming
MoERoutingTrace = RoutingTrace
generate_zipf_trace = generate_zipf_skewed_trace
generate_uniform_trace = generate_uniform_random_trace
