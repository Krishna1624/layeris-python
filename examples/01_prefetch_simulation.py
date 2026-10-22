"""
Example 01: Multi-Layer Predictive Prefetching Simulation
=========================================================
Runs a quick simulation showing the cache hit-rate and latency gains from
coordinated multi-layer prefetching compared to demand-only loading.
"""

import sys
from pathlib import Path

# Add src to sys.path if running without pip install -e .
src_path = Path(__file__).resolve().parent.parent / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

from layeris.prefetch.predictor import (
    MultiLayerPredictor,
    MultiLayerPrediction,
    LayerPrediction,
    PredictConfig,
    ResourceState,
)
from layeris.prefetch.simulator import ExpertCacheSimulator
from layeris.prefetch.trace import generate_coupled_trace


def main():
    print("=================================================================")
    print("  Layeris MoE Predictive Prefetching Simulation Example")
    print("=================================================================")

    num_layers = 16
    num_experts = 32
    top_k = 4
    num_tokens = 50
    cache_capacity_per_layer = 8  # 8 out of 32 experts resident (~25% capacity)
    depth = 3

    print(f"[*] Generating coupled MoE routing trace ({num_layers} layers, {num_experts} experts, top-{top_k}, {num_tokens} tokens)...")
    trace = generate_coupled_trace(
        num_layers=num_layers,
        num_experts=num_experts,
        top_k=top_k,
        num_tokens=num_tokens,
        seed=42,
    )

    # 1. Demand Only Simulation (No prefetching)
    print("\n--- [Run 1: Demand-Only Baseline] ---")
    sim_demand = ExpertCacheSimulator(
        num_layers=num_layers,
        num_experts=num_experts,
        capacity_per_layer=cache_capacity_per_layer,
        eviction_policy="lru",
    )
    for tok in range(trace.num_tokens):
        events = trace.events_for_token(tok)
        for ev in events:
            for eid in ev.experts:
                sim_demand.access_expert(ev.layer, eid)
            sim_demand.step_layer_compute()

    stats_demand = sim_demand.finalize()
    print(f"  Expert Requests : {stats_demand.total_expert_requests}")
    print(f"  Cache Hits      : {stats_demand.cache_hits}")
    print(f"  Cache Misses    : {stats_demand.cache_misses}")
    print(f"  Hit Rate        : {stats_demand.hit_rate:.2f}%")

    # 2. Coordinated Adaptive Prefetching (Proposed)
    print(f"\n--- [Run 2: Coordinated Multi-Layer Prefetching (Depth={depth}, Top-K={top_k})] ---")
    predictor = MultiLayerPredictor(
        config=PredictConfig(depth=depth, top_k=top_k, enabled=True, adaptive_resource=True),
        total_layers=num_layers,
        num_experts=num_experts,
    )
    sim_prefetch = ExpertCacheSimulator(
        num_layers=num_layers,
        num_experts=num_experts,
        capacity_per_layer=cache_capacity_per_layer,
        eviction_policy="predictive_lfru",
    )
    res_state = ResourceState(available_ram_bytes=4 * 1024**3, max_concurrent_io=8)

    for tok in range(trace.num_tokens):
        predictor.advance_token()
        events = trace.events_for_token(tok)
        for ev in events:
            l = ev.layer

            # 1. Multi-layer lookahead prediction
            bundle_layers = []
            for dL in range(1, depth + 1):
                target_l = l + dL
                if target_l < trace.num_layers:
                    lp = predictor.predict_from_history(target_l, top_k=top_k)
                    bundle_layers.append(lp)

            ml_pred = MultiLayerPrediction(source_layer=l, layers=bundle_layers)

            # 2. Priority schedule and adapt candidates
            res_state.pending_io_jobs = len(sim_prefetch.in_flight_io)
            planned_q = predictor.schedule_candidates(
                pred=ml_pred,
                current_layer=l,
                resource=res_state,
                is_resident_fn=sim_prefetch.is_resident,
            )

            # Inform cache of upcoming protected experts for Predictive LFRU
            sim_prefetch.set_protected_experts(planned_q.to_keys(uncached_only=False))

            # 3. Dispatch uncached keys
            dispatch_keys = planned_q.to_keys(uncached_only=True)
            for target_l, eid in dispatch_keys:
                sim_prefetch.dispatch_prefetch(target_l, eid)

            # 4. Demand execution & feedback record
            for eid in ev.experts:
                sim_prefetch.access_expert(l, eid)

            predictor.on_layer_routed(l, list(ev.experts))
            sim_prefetch.step_layer_compute()

    stats_prefetch = sim_prefetch.finalize()
    print(f"  Expert Requests : {stats_prefetch.total_expert_requests}")
    print(f"  Cache Hits      : {stats_prefetch.cache_hits}")
    print(f"  Cache Misses    : {stats_prefetch.cache_misses}")
    print(f"  Hit Rate        : {stats_prefetch.hit_rate:.2f}%")
    print(f"  Useful Prefetch : {stats_prefetch.useful_prefetches}")
    print(f"  Wasted Prefetch : {stats_prefetch.wasted_prefetches}")

    gain = stats_prefetch.hit_rate - stats_demand.hit_rate
    print(f"\n[+] Net Cache Hit Rate Improvement: +{gain:.2f}%")
    print("=================================================================\n")


if __name__ == "__main__":
    main()
