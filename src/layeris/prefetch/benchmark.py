"""Benchmark comparing multi-layer predictive prefetching algorithms for Layeris MoE inference.

Baselines evaluated:
1. No Predictive Prefetching (Demand Only): Pure demand paging upon cache miss.
2. Simple Recent-History / Frequency Baseline: Static Top-K or recency-only prefetch.
3. Original Uncoordinated Multi-Layer Prediction: Naive per-layer lookahead without residency awareness or feedback bias.
4. Coordinated Adaptive Multi-Layer Prefetching (Proposed): Full closed-loop priority scoring, resource adaptation, and waste feedback bias.

Outputs:
- Machine-readable JSON and CSV results.
- Detailed head-to-head comparison tables.
- Measured execution overhead vs simulated stall latency separation.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
import tracemalloc
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Import prototype modules
try:
    from .predictor import (
        MultiLayerPredictor,
        MultiLayerPrediction,
        LayerPrediction,
        PredictConfig,
        ResourceState,
        topk_filtered,
    )
    try:
        from .simulator import ExpertCacheSimulator, SimulationStats
    except ImportError:
        from .prefetch_simulator import ExpertCacheSimulator, SimulationStats

    try:
        from .trace import (
            RoutingTrace,
            generate_coupled_trace,
            generate_zipf_skewed_trace,
            generate_uniform_random_trace,
            generate_trace_by_pattern,
        )
    except ImportError:
        from .routing_trace import (
            RoutingTrace,
            generate_coupled_trace,
            generate_zipf_skewed_trace,
            generate_uniform_random_trace,
            generate_trace_by_pattern,
        )
except ImportError:
    # Allow running directly as standalone script
    from predictor import (
        MultiLayerPredictor,
        MultiLayerPrediction,
        LayerPrediction,
        PredictConfig,
        ResourceState,
        topk_filtered,
    )
    try:
        from simulator import ExpertCacheSimulator, SimulationStats
    except ImportError:
        from prefetch_simulator import ExpertCacheSimulator, SimulationStats
    try:
        from trace import (
            RoutingTrace,
            generate_coupled_trace,
            generate_zipf_skewed_trace,
            generate_uniform_random_trace,
            generate_trace_by_pattern,
        )
    except ImportError:
        from routing_trace import (
            RoutingTrace,
            generate_coupled_trace,
            generate_zipf_skewed_trace,
            generate_uniform_random_trace,
            generate_trace_by_pattern,
        )


@dataclass
class BenchmarkResult:
    """Consolidated metrics for a single algorithm run."""
    algorithm_name: str
    trace_pattern: str
    cache_capacity_per_layer: int
    prediction_depth: int
    num_tokens: int
    num_layers: int
    num_experts: int
    total_expert_requests: int
    cache_hits: int
    cache_misses: int
    hit_rate_pct: float
    demand_loads: int
    prefetch_dispatches: int
    redundant_prefetches_filtered: int
    useful_prefetches: int
    wasted_prefetches: int
    wasted_prefetch_ratio_pct: float
    timely_prefetches: int
    timely_prefetch_rate_pct: float
    precision_pct: float
    recall_pct: float
    simulated_compute_time_ms: float
    simulated_stall_time_ms: float
    simulated_total_time_ms: float
    simulated_tokens_per_sec: float
    measured_prediction_overhead_ms: float
    measured_avg_overhead_us_per_layer: float
    peak_resident_experts: int
    peak_memory_kb: int


# ============================================================================
# Runner for Baseline 1: No Predictive Prefetching (Demand Only)
# ============================================================================
def run_baseline_demand_only(
    trace: RoutingTrace,
    capacity_per_layer: int,
    layer_compute_us: float = 120.0,
    expert_load_us: float = 250.0,
) -> Tuple[BenchmarkResult, SimulationStats]:
    sim = ExpertCacheSimulator(
        capacity_per_layer=capacity_per_layer,
        num_layers=trace.num_layers,
        num_experts=trace.num_experts,
        layer_compute_us=layer_compute_us,
        expert_load_us=expert_load_us,
        eviction_policy="lru",
    )

    t0_overhead = time.perf_counter_ns()
    # No prediction overhead
    measured_overhead_ns = time.perf_counter_ns() - t0_overhead

    for tok in range(trace.num_tokens):
        events = trace.events_for_token(tok)
        for ev in events:
            for eid in ev.experts:
                sim.access_expert(ev.layer, eid)
            sim.step_layer_compute()

    stats = sim.finalize()
    overhead_ms = measured_overhead_ns / 1_000_000.0
    total_layers_stepped = trace.num_tokens * trace.num_layers

    res = BenchmarkResult(
        algorithm_name="1. Demand Only (No Prefetch)",
        trace_pattern=trace.metadata.get("pattern", "unknown"),
        cache_capacity_per_layer=capacity_per_layer,
        prediction_depth=0,
        num_tokens=trace.num_tokens,
        num_layers=trace.num_layers,
        num_experts=trace.num_experts,
        total_expert_requests=stats.total_expert_requests,
        cache_hits=stats.cache_hits,
        cache_misses=stats.cache_misses,
        hit_rate_pct=stats.hit_rate,
        demand_loads=stats.demand_loads,
        prefetch_dispatches=stats.prefetch_dispatches,
        redundant_prefetches_filtered=stats.redundant_prefetches_filtered,
        useful_prefetches=stats.useful_prefetches,
        wasted_prefetches=stats.wasted_prefetches,
        wasted_prefetch_ratio_pct=stats.wasted_prefetch_rate,
        timely_prefetches=stats.timely_prefetches,
        timely_prefetch_rate_pct=stats.timely_rate,
        precision_pct=0.0,
        recall_pct=0.0,
        simulated_compute_time_ms=stats.simulated_compute_time_us / 1000.0,
        simulated_stall_time_ms=stats.simulated_stall_time_us / 1000.0,
        simulated_total_time_ms=stats.simulated_total_time_us / 1000.0,
        simulated_tokens_per_sec=(
            trace.num_tokens / (stats.simulated_total_time_us / 1_000_000.0)
            if stats.simulated_total_time_us > 0
            else 0.0
        ),
        measured_prediction_overhead_ms=overhead_ms,
        measured_avg_overhead_us_per_layer=(
            (overhead_ms * 1000.0) / total_layers_stepped if total_layers_stepped > 0 else 0.0
        ),
        peak_resident_experts=stats.peak_resident_experts,
        peak_memory_kb=0,
    )
    return res, stats


# ============================================================================
# Runner for Baseline 2: Simple Recent-History / Frequency-Based Prefetch
# ============================================================================
def run_baseline_frequency_history(
    trace: RoutingTrace,
    capacity_per_layer: int,
    top_k: int = 4,
    layer_compute_us: float = 120.0,
    expert_load_us: float = 250.0,
) -> Tuple[BenchmarkResult, SimulationStats]:
    sim = ExpertCacheSimulator(
        capacity_per_layer=capacity_per_layer,
        num_layers=trace.num_layers,
        num_experts=trace.num_experts,
        layer_compute_us=layer_compute_us,
        expert_load_us=expert_load_us,
        eviction_policy="lru",
    )

    # Frequency counters per layer
    freq_counts: List[List[int]] = [
        [0 for _ in range(trace.num_experts)] for _ in range(trace.num_layers)
    ]

    t0_overhead = time.perf_counter_ns()

    for tok in range(trace.num_tokens):
        events = trace.events_for_token(tok)
        for ev in events:
            l = ev.layer
            # At each layer, prefetch top frequent experts for immediate next layer (L+1)
            next_l = l + 1
            if next_l < trace.num_layers:
                # Rank experts by static cumulative frequency
                scored = sorted(
                    range(trace.num_experts),
                    key=lambda e: freq_counts[next_l][e],
                    reverse=True,
                )
                candidates = scored[:top_k]
                for eid in candidates:
                    if freq_counts[next_l][eid] > 0:
                        sim.dispatch_prefetch(next_l, eid)

            # Demand access
            for eid in ev.experts:
                sim.access_expert(l, eid)
                freq_counts[l][eid] += 1

            sim.step_layer_compute()

    measured_overhead_ns = time.perf_counter_ns() - t0_overhead
    stats = sim.finalize()
    overhead_ms = measured_overhead_ns / 1_000_000.0
    total_layers_stepped = trace.num_tokens * trace.num_layers

    precision = stats.useful_prefetch_rate
    recall = (
        (stats.useful_prefetches / stats.total_expert_requests) * 100.0
        if stats.total_expert_requests > 0
        else 0.0
    )

    res = BenchmarkResult(
        algorithm_name="2. Simple Frequency Baseline",
        trace_pattern=trace.metadata.get("pattern", "unknown"),
        cache_capacity_per_layer=capacity_per_layer,
        prediction_depth=1,
        num_tokens=trace.num_tokens,
        num_layers=trace.num_layers,
        num_experts=trace.num_experts,
        total_expert_requests=stats.total_expert_requests,
        cache_hits=stats.cache_hits,
        cache_misses=stats.cache_misses,
        hit_rate_pct=stats.hit_rate,
        demand_loads=stats.demand_loads,
        prefetch_dispatches=stats.prefetch_dispatches,
        redundant_prefetches_filtered=stats.redundant_prefetches_filtered,
        useful_prefetches=stats.useful_prefetches,
        wasted_prefetches=stats.wasted_prefetches,
        wasted_prefetch_ratio_pct=stats.wasted_prefetch_rate,
        timely_prefetches=stats.timely_prefetches,
        timely_prefetch_rate_pct=stats.timely_rate,
        precision_pct=precision,
        recall_pct=recall,
        simulated_compute_time_ms=stats.simulated_compute_time_us / 1000.0,
        simulated_stall_time_ms=stats.simulated_stall_time_us / 1000.0,
        simulated_total_time_ms=stats.simulated_total_time_us / 1000.0,
        simulated_tokens_per_sec=(
            trace.num_tokens / (stats.simulated_total_time_us / 1_000_000.0)
            if stats.simulated_total_time_us > 0
            else 0.0
        ),
        measured_prediction_overhead_ms=overhead_ms,
        measured_avg_overhead_us_per_layer=(
            (overhead_ms * 1000.0) / total_layers_stepped if total_layers_stepped > 0 else 0.0
        ),
        peak_resident_experts=stats.peak_resident_experts,
        peak_memory_kb=0,
    )
    return res, stats


# ============================================================================
# Runner for Baseline 3: Original Uncoordinated Multi-Layer Prediction
# ============================================================================
def run_baseline_original_uncoordinated(
    trace: RoutingTrace,
    capacity_per_layer: int,
    depth: int = 3,
    top_k: int = 4,
    layer_compute_us: float = 120.0,
    expert_load_us: float = 250.0,
) -> Tuple[BenchmarkResult, SimulationStats]:
    """Mirrors run_benchmark_original in c/tests/bench_prefetch_optimization.c.
    
    Uncoordinated lookahead: loops dL = 1..depth calling independent prefetch,
    without coordinated residency checking, without temporal priority discounting,
    and without dynamic feedback bias.
    """
    sim = ExpertCacheSimulator(
        capacity_per_layer=capacity_per_layer,
        num_layers=trace.num_layers,
        num_experts=trace.num_experts,
        layer_compute_us=layer_compute_us,
        expert_load_us=expert_load_us,
        eviction_policy="lru",
    )

    cfg = PredictConfig(depth=depth, top_k=top_k, adaptive_resource=False, min_confidence=0.10)
    predictor = MultiLayerPredictor(trace.num_layers, trace.num_experts, cfg)

    t0_overhead = time.perf_counter_ns()

    for tok in range(trace.num_tokens):
        predictor.advance_token()
        events = trace.events_for_token(tok)

        for ev in events:
            l = ev.layer

            # Uncoordinated: independently predict for each target layer dL=1..depth
            for dL in range(1, depth + 1):
                target_l = l + dL
                if target_l < trace.num_layers:
                    lp = predictor.predict_from_history(target_l, top_k=top_k)
                    for eid in lp.expert_ids:
                        sim.dispatch_prefetch(target_l, eid)

            # Demand execution
            for eid in ev.experts:
                sim.access_expert(l, eid)

            predictor.on_layer_routed(l, list(ev.experts))
            sim.step_layer_compute()

    measured_overhead_ns = time.perf_counter_ns() - t0_overhead
    stats = sim.finalize()
    overhead_ms = measured_overhead_ns / 1_000_000.0
    total_layers_stepped = trace.num_tokens * trace.num_layers

    precision = stats.useful_prefetch_rate
    recall = (
        (stats.useful_prefetches / stats.total_expert_requests) * 100.0
        if stats.total_expert_requests > 0
        else 0.0
    )

    res = BenchmarkResult(
        algorithm_name="3. Original Uncoordinated Multi-Layer",
        trace_pattern=trace.metadata.get("pattern", "unknown"),
        cache_capacity_per_layer=capacity_per_layer,
        prediction_depth=depth,
        num_tokens=trace.num_tokens,
        num_layers=trace.num_layers,
        num_experts=trace.num_experts,
        total_expert_requests=stats.total_expert_requests,
        cache_hits=stats.cache_hits,
        cache_misses=stats.cache_misses,
        hit_rate_pct=stats.hit_rate,
        demand_loads=stats.demand_loads,
        prefetch_dispatches=stats.prefetch_dispatches,
        redundant_prefetches_filtered=stats.redundant_prefetches_filtered,
        useful_prefetches=stats.useful_prefetches,
        wasted_prefetches=stats.wasted_prefetches,
        wasted_prefetch_ratio_pct=stats.wasted_prefetch_rate,
        timely_prefetches=stats.timely_prefetches,
        timely_prefetch_rate_pct=stats.timely_rate,
        precision_pct=precision,
        recall_pct=recall,
        simulated_compute_time_ms=stats.simulated_compute_time_us / 1000.0,
        simulated_stall_time_ms=stats.simulated_stall_time_us / 1000.0,
        simulated_total_time_ms=stats.simulated_total_time_us / 1000.0,
        simulated_tokens_per_sec=(
            trace.num_tokens / (stats.simulated_total_time_us / 1_000_000.0)
            if stats.simulated_total_time_us > 0
            else 0.0
        ),
        measured_prediction_overhead_ms=overhead_ms,
        measured_avg_overhead_us_per_layer=(
            (overhead_ms * 1000.0) / total_layers_stepped if total_layers_stepped > 0 else 0.0
        ),
        peak_resident_experts=stats.peak_resident_experts,
        peak_memory_kb=0,
    )
    return res, stats


# ============================================================================
# Runner for Proposed Method: Coordinated Adaptive Multi-Layer Prefetching
# ============================================================================
def run_proposed_coordinated_adaptive(
    trace: RoutingTrace,
    capacity_per_layer: int,
    depth: int = 3,
    top_k: int = 4,
    layer_compute_us: float = 120.0,
    expert_load_us: float = 250.0,
    max_concurrent_io: int = 16,
) -> Tuple[BenchmarkResult, SimulationStats]:
    """Mirrors run_benchmark_optimized in c/tests/bench_prefetch_optimization.c.
    
    Implements:
    - Bundled multi-layer lookahead planning
    - Composite priority ranking (confidence, temporal discount, history factor, residency discount)
    - Closed-loop dynamic feedback bias penalty when waste > 35%
    - Resource headroom adaptation
    - Predictive LFRU eviction protecting upcoming candidates
    """
    sim = ExpertCacheSimulator(
        capacity_per_layer=capacity_per_layer,
        num_layers=trace.num_layers,
        num_experts=trace.num_experts,
        layer_compute_us=layer_compute_us,
        expert_load_us=expert_load_us,
        max_concurrent_io=max_concurrent_io,
        eviction_policy="predictive_lfru",
    )

    cfg = PredictConfig(
        depth=depth,
        top_k=top_k,
        adaptive_resource=True,
        min_confidence=0.15,
    )
    predictor = MultiLayerPredictor(trace.num_layers, trace.num_experts, cfg)
    res_state = ResourceState(
        available_ram_bytes=16 * 1024 * 1024 * 1024,
        expert_size_bytes=16 * 1024 * 1024,
        max_concurrent_io=max_concurrent_io,
    )

    t0_overhead = time.perf_counter_ns()

    for tok in range(trace.num_tokens):
        predictor.advance_token()
        events = trace.events_for_token(tok)

        for ev in events:
            l = ev.layer

            # 1. Plan multi-layer bundle for upcoming horizon dL = 1..depth
            bundle_layers: List[LayerPrediction] = []
            for dL in range(1, depth + 1):
                target_l = l + dL
                if target_l < trace.num_layers:
                    lp = predictor.predict_from_history(target_l, top_k=top_k)
                    bundle_layers.append(lp)

            ml_pred = MultiLayerPrediction(source_layer=l, layers=bundle_layers)

            # 2. Priority schedule and adapt candidates
            res_state.pending_io_jobs = len(sim.in_flight_io)
            planned_q = predictor.schedule_candidates(
                pred=ml_pred,
                current_layer=l,
                resource=res_state,
                is_resident_fn=sim.is_resident,
            )

            # Inform cache of upcoming protected experts for Predictive LFRU
            sim.set_protected_experts(planned_q.to_keys(uncached_only=False))

            # 3. Dispatch uncached keys
            dispatch_keys = planned_q.to_keys(uncached_only=True)
            for target_l, eid in dispatch_keys:
                sim.dispatch_prefetch(target_l, eid)

            # 4. Demand execution & feedback record
            for eid in ev.experts:
                sim.access_expert(l, eid)

            predictor.on_layer_routed(l, list(ev.experts))
            sim.step_layer_compute()

    measured_overhead_ns = time.perf_counter_ns() - t0_overhead
    stats = sim.finalize()
    overhead_ms = measured_overhead_ns / 1_000_000.0
    total_layers_stepped = trace.num_tokens * trace.num_layers

    precision = stats.useful_prefetch_rate
    recall = (
        (stats.useful_prefetches / stats.total_expert_requests) * 100.0
        if stats.total_expert_requests > 0
        else 0.0
    )

    res = BenchmarkResult(
        algorithm_name="4. Coordinated Adaptive Multi-Layer (Proposed)",
        trace_pattern=trace.metadata.get("pattern", "unknown"),
        cache_capacity_per_layer=capacity_per_layer,
        prediction_depth=depth,
        num_tokens=trace.num_tokens,
        num_layers=trace.num_layers,
        num_experts=trace.num_experts,
        total_expert_requests=stats.total_expert_requests,
        cache_hits=stats.cache_hits,
        cache_misses=stats.cache_misses,
        hit_rate_pct=stats.hit_rate,
        demand_loads=stats.demand_loads,
        prefetch_dispatches=stats.prefetch_dispatches,
        redundant_prefetches_filtered=stats.redundant_prefetches_filtered,
        useful_prefetches=stats.useful_prefetches,
        wasted_prefetches=stats.wasted_prefetches,
        wasted_prefetch_ratio_pct=stats.wasted_prefetch_rate,
        timely_prefetches=stats.timely_prefetches,
        timely_prefetch_rate_pct=stats.timely_rate,
        precision_pct=precision,
        recall_pct=recall,
        simulated_compute_time_ms=stats.simulated_compute_time_us / 1000.0,
        simulated_stall_time_ms=stats.simulated_stall_time_us / 1000.0,
        simulated_total_time_ms=stats.simulated_total_time_us / 1000.0,
        simulated_tokens_per_sec=(
            trace.num_tokens / (stats.simulated_total_time_us / 1_000_000.0)
            if stats.simulated_total_time_us > 0
            else 0.0
        ),
        measured_prediction_overhead_ms=overhead_ms,
        measured_avg_overhead_us_per_layer=(
            (overhead_ms * 1000.0) / total_layers_stepped if total_layers_stepped > 0 else 0.0
        ),
        peak_resident_experts=stats.peak_resident_experts,
        peak_memory_kb=0,
    )
    return res, stats


# ============================================================================
# Benchmark Runner & Formatted Table Reporting
# ============================================================================
def run_full_suite(
    patterns: List[str] = ["coupled", "zipf", "uniform"],
    capacities: List[int] = [2, 4, 6],
    depths: List[int] = [1, 2, 3, 4],
    num_tokens: int = 60,
    num_layers: int = 16,
    num_experts: int = 32,
    top_k: int = 4,
    seed: int = 42,
    output_dir: Optional[Path] = None,
) -> List[BenchmarkResult]:
    """Run full evaluation matrix across trace patterns, cache capacities, and horizons."""
    tracemalloc.start()
    all_results: List[BenchmarkResult] = []

    for pattern in patterns:
        trace = generate_trace_by_pattern(
            pattern=pattern,
            num_layers=num_layers,
            num_experts=num_experts,
            top_k=top_k,
            num_tokens=num_tokens,
            seed=seed,
        )

        for cap in capacities:
            # Baseline 1: Demand Only
            res_demand, _ = run_baseline_demand_only(trace, capacity_per_layer=cap)
            current_mem, peak_mem = tracemalloc.get_traced_memory()
            res_demand.peak_memory_kb = peak_mem // 1024
            all_results.append(res_demand)

            # Baseline 2: Frequency History
            res_freq, _ = run_baseline_frequency_history(trace, capacity_per_layer=cap, top_k=top_k)
            _, peak_mem = tracemalloc.get_traced_memory()
            res_freq.peak_memory_kb = peak_mem // 1024
            all_results.append(res_freq)

            for d in depths:
                # Baseline 3: Original Uncoordinated
                res_uncoord, _ = run_baseline_original_uncoordinated(
                    trace, capacity_per_layer=cap, depth=d, top_k=top_k
                )
                _, peak_mem = tracemalloc.get_traced_memory()
                res_uncoord.peak_memory_kb = peak_mem // 1024
                all_results.append(res_uncoord)

                # Proposed: Coordinated Adaptive
                res_coord, _ = run_proposed_coordinated_adaptive(
                    trace, capacity_per_layer=cap, depth=d, top_k=top_k
                )
                _, peak_mem = tracemalloc.get_traced_memory()
                res_coord.peak_memory_kb = peak_mem // 1024
                all_results.append(res_coord)

    tracemalloc.stop()

    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        # Export JSON
        json_path = output_dir / "benchmark_results.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump([asdict(r) for r in all_results], f, indent=2)

        # Export CSV
        csv_path = output_dir / "benchmark_results.csv"
        if all_results:
            fieldnames = list(asdict(all_results[0]).keys())
            with open(csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                for r in all_results:
                    writer.writerow(asdict(r))

    return all_results


def print_comparison_table(results: List[BenchmarkResult], pattern_filter: str = "coupled", cap_filter: int = 4, depth_filter: int = 3) -> None:
    """Print clean comparison table for a selected experiment condition."""
    subset = [
        r for r in results
        if pattern_filter in r.trace_pattern
        and r.cache_capacity_per_layer == cap_filter
        and (r.prediction_depth == depth_filter or r.prediction_depth in (0, 1))
    ]

    print("\n" + "=" * 105)
    print(f"  HEAD-TO-HEAD EVALUATION (Pattern: {pattern_filter}, Cap: {cap_filter}/layer, Depth: {depth_filter})")
    print("=" * 105)
    header = f"{'Algorithm':<38} | {'HitRate':<8} | {'UsefulPref':<10} | {'WastedPref':<10} | {'StallTime':<10} | {'Tok/s':<7} | {'Overhead':<9}"
    print(header)
    print("-" * 105)

    for r in subset:
        name = r.algorithm_name
        hit = f"{r.hit_rate_pct:.1f}%"
        useful = f"{r.useful_prefetches}"
        wasted = f"{r.wasted_prefetches} ({r.wasted_prefetch_ratio_pct:.1f}%)"
        stall = f"{r.simulated_stall_time_ms:.1f} ms"
        toks = f"{r.simulated_tokens_per_sec:.1f}"
        ovh = f"{r.measured_prediction_overhead_ms:.2f} ms"
        print(f"{name:<38} | {hit:<8} | {useful:<10} | {wasted:<10} | {stall:<10} | {toks:<7} | {ovh:<9}")

    print("=" * 105 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Layeris Multi-Layer Expert Prefetch Benchmark Suite")
    parser.add_argument("--tokens", type=int, default=60, help="Number of simulated tokens")
    parser.add_argument("--layers", type=int, default=16, help="Number of model layers")
    parser.add_argument("--experts", type=int, default=32, help="Number of experts per layer")
    parser.add_argument("--top_k", type=int, default=4, help="Top-K experts routed per layer")
    parser.add_argument("--seed", type=int, default=42, help="Deterministic random seed")
    parser.add_argument(
        "--output_dir",
        type=str,
        default=str(Path(__file__).parent / "results"),
        help="Directory to save JSON/CSV benchmark output",
    )
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    print(f"Running Multi-Layer Prefetch Benchmark Suite...")
    print(f"Tokens: {args.tokens}, Layers: {args.layers}, Experts: {args.experts}, Top-K: {args.top_k}, Seed: {args.seed}")

    results = run_full_suite(
        patterns=["coupled", "zipf", "uniform"],
        capacities=[2, 4, 6],
        depths=[1, 2, 3, 4],
        num_tokens=args.tokens,
        num_layers=args.layers,
        num_experts=args.experts,
        top_k=args.top_k,
        seed=args.seed,
        output_dir=out_dir,
    )

    print(f"\nExecution finished! Generated {len(results)} experiment data points.")
    print(f"Results saved to: {out_dir}")

    # Print comparison tables for key patterns
    for pat in ["coupled", "zipf", "uniform"]:
        print_comparison_table(results, pattern_filter=pat, cap_filter=4, depth_filter=3)


if __name__ == "__main__":
    main()
