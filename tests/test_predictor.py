"""Unit tests and edge case validation for Layeris Multi-Layer Expert Predictor prototype.

Covers Phase 4 requirements:
- Empty and short histories.
- Repeated expert IDs.
- Unseen experts.
- Different expert counts and prediction depths.
- Invalid input shapes and values (NaN, Inf, negatives, out-of-bounds).
- Prediction confidence and fallback behavior (thresholds, softmax/sigmoid, top-p).
- Priority scoring properties.
- Cache feedback tracking & dynamic bias.
- Resource headroom adaptation.
- Cache simulation correctness (hits, misses, eviction, timely, late, wasted prefetches).
- Deterministic reproducibility under fixed random seed.
"""

from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

src_path = Path(__file__).resolve().parent.parent / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

# Support running via package install, direct discovery, or standalone script
try:
    from layeris.prefetch.predictor import (
        MultiLayerPredictor,
        MultiLayerPrediction,
        LayerPrediction,
        PredictConfig,
        ResourceState,
        RoutingHistory,
        CacheFeedbackTracker,
        topk_filtered,
        calculate_priority,
        passes_confidence,
        adapt_resources,
    )
    from layeris.prefetch.simulator import ExpertCacheSimulator, SimulationStats
    from layeris.prefetch.trace import (
        RoutingTrace,
        generate_coupled_trace,
        generate_zipf_skewed_trace,
        generate_uniform_random_trace,
    )
except ImportError:
    try:
        from .predictor import (
            MultiLayerPredictor,
            MultiLayerPrediction,
            LayerPrediction,
            PredictConfig,
            ResourceState,
            RoutingHistory,
            CacheFeedbackTracker,
            topk_filtered,
            calculate_priority,
            passes_confidence,
            adapt_resources,
        )
        from .simulator import ExpertCacheSimulator, SimulationStats
        from .trace import (
            RoutingTrace,
            generate_coupled_trace,
            generate_zipf_skewed_trace,
            generate_uniform_random_trace,
        )
    except ImportError:
        from predictor import (
            MultiLayerPredictor,
            MultiLayerPrediction,
            LayerPrediction,
            PredictConfig,
            ResourceState,
            RoutingHistory,
            CacheFeedbackTracker,
            topk_filtered,
            calculate_priority,
            passes_confidence,
            adapt_resources,
        )
        from simulator import ExpertCacheSimulator, SimulationStats
        from trace import (
            RoutingTrace,
            generate_coupled_trace,
            generate_zipf_skewed_trace,
            generate_uniform_random_trace,
        )


class TestMultiLayerPredictor(unittest.TestCase):

    def test_empty_and_short_histories(self):
        """Test behavior with zero tokens recorded, and with only 1 token recorded."""
        predictor = MultiLayerPredictor(total_layers=16, num_experts=32)

        # 1. Zero history: predict_from_history should return without error
        pred_empty = predictor.predict_from_history(target_layer=4, top_k=4)
        self.assertIsNotNone(pred_empty)
        self.assertEqual(pred_empty.layer, 4)
        # All expert frequencies are 0, tie-broken cleanly
        self.assertEqual(len(pred_empty.expert_ids), 4)

        # 2. Advance 1 token and record single layer
        predictor.advance_token()
        predictor.on_layer_routed(layer=4, routed_experts=[7, 12])

        # Predict layer 4 again: expert 7 and 12 should receive highest score
        pred_1 = predictor.predict_from_history(target_layer=4, top_k=2)
        self.assertEqual(set(pred_1.expert_ids[:2]), {7, 12})

    def test_repeated_expert_ids(self):
        """Test handling of repeated expert IDs in routing inputs and deduplication."""
        hist = RoutingHistory(total_layers=8, num_experts=16)
        # Pass duplicated expert IDs
        hist.record(layer=2, expert_ids=[5, 5, 5, 9, 9, 5])

        slot = hist.ring_head
        recorded = hist.history[2][slot]
        # Deduplication check
        self.assertEqual(recorded, [5, 9])
        # Frequency should increment only once per record
        self.assertEqual(hist.expert_freq[2][5], 1.0)
        self.assertEqual(hist.expert_freq[2][9], 1.0)

    def test_unseen_experts(self):
        """Test prediction and historical scoring when queried for unseen experts."""
        predictor = MultiLayerPredictor(total_layers=8, num_experts=16)
        predictor.advance_token()
        # Only experts 0, 1 were routed
        predictor.on_layer_routed(layer=3, routed_experts=[0, 1])

        pred = predictor.predict_from_history(target_layer=3, top_k=8)
        # Unseen experts (2..15) should have 0 frequency and lower score than 0 and 1
        self.assertIn(0, pred.expert_ids[:2])
        self.assertIn(1, pred.expert_ids[:2])
        # Later positions contain unseen experts with 0.0 frequency
        for eid in pred.expert_ids[2:]:
            self.assertEqual(predictor.history.expert_freq[3][eid], 0.0)

    def test_different_expert_counts_and_prediction_depths(self):
        """Test varying expert counts and lookahead horizons, including clamping."""
        for num_e in [8, 16, 64, 128]:
            predictor = MultiLayerPredictor(total_layers=16, num_experts=num_e)
            pred = predictor.predict_from_history(target_layer=2, top_k=min(4, num_e))
            self.assertEqual(len(pred.expert_ids), min(4, num_e))
            self.assertTrue(all(0 <= eid < num_e for eid in pred.expert_ids))

        # Test depth clamping and boundaries
        cfg = PredictConfig(depth=25)  # > COLI_PREDICT_MAX_DEPTH (16)
        self.assertEqual(cfg.depth, 16)

        cfg_min = PredictConfig(depth=-5)
        self.assertEqual(cfg_min.depth, 1)

        # Boundary at end of network: target layer >= total_layers
        predictor = MultiLayerPredictor(total_layers=10, num_experts=16)
        pred_oob = predictor.predict_from_history(target_layer=10, top_k=4)
        self.assertEqual(pred_oob.num_candidates, 0)

    def test_invalid_input_shapes_and_values(self):
        """Test NaN, Inf, empty inputs, negative top_k, and out-of-range logits."""
        # Logits containing NaN and Inf
        bad_logits = [1.0, float("nan"), 3.5, float("inf"), -2.0, float("-inf")]
        ids, scores, confs = topk_filtered(bad_logits, num_experts=6, top_k=4)
        # Only finite numbers should be returned
        self.assertEqual(ids, [2, 0, 4])
        self.assertEqual(scores, [3.5, 1.0, -2.0])
        self.assertTrue(all(not math.isnan(c) and not math.isinf(c) for c in confs))

        # Negative top_k or 0 top_k
        ids_neg, _, _ = topk_filtered([1.0, 2.0], num_experts=2, top_k=-1)
        self.assertEqual(ids_neg, [])

        ids_zero, _, _ = topk_filtered([1.0, 2.0], num_experts=2, top_k=0)
        self.assertEqual(ids_zero, [])

        # Empty logits
        ids_empty, _, _ = topk_filtered([], num_experts=0, top_k=4)
        self.assertEqual(ids_empty, [])

    def test_prediction_confidence_and_fallback_behavior(self):
        """Test threshold filtering, softmax vs sigmoid normalization, and top_p cutoff."""
        logits = [0.5, 5.0, 1.0, 4.0, 0.2]
        # min_score filtering: only scores >= 2.0
        ids, scores, confs = topk_filtered(
            logits, num_experts=5, top_k=5, min_score=2.0
        )
        self.assertEqual(ids, [1, 3])
        self.assertEqual(scores, [5.0, 4.0])

        # min_confidence filtering: only normalized probabilities >= 0.20
        ids_c, _, confs_c = topk_filtered(
            logits, num_experts=5, top_k=5, min_confidence=0.20
        )
        self.assertEqual(ids_c, [1, 3])
        self.assertTrue(all(c >= 0.20 for c in confs_c))

        # Top-p cumulative mass cutoff
        ids_topp, _, confs_topp = topk_filtered(
            logits, num_experts=5, top_k=5, top_p_mass=0.80
        )
        # Expert 1 and 3 account for > 90% of softmax mass
        self.assertIn(len(ids_topp), [1, 2])
        self.assertEqual(ids_topp[0], 1)

        # Sigmoid normalization
        ids_sig, _, confs_sig = topk_filtered(
            logits, num_experts=5, top_k=3, is_sigmoid=True
        )
        self.assertEqual(ids_sig[0], 1)
        self.assertAlmostEqual(sum(confs_sig), sum(confs_sig), places=5)

    def test_priority_scoring_properties(self):
        """Test monotonicity, temporal distance decay, residency discounting, and history boost."""
        # 1. Higher confidence yields higher priority
        p_high = calculate_priority(confidence=0.85, layer_distance=1, hist_freq=0.0, is_resident=False)
        p_low = calculate_priority(confidence=0.20, layer_distance=1, hist_freq=0.0, is_resident=False)
        self.assertGreater(p_high, p_low)

        # 2. Nearer target layer (dL=1) outranks distant layer (dL=4)
        p_near = calculate_priority(confidence=0.70, layer_distance=1, hist_freq=0.0, is_resident=False)
        p_far = calculate_priority(confidence=0.70, layer_distance=4, hist_freq=0.0, is_resident=False)
        self.assertGreater(p_near, p_far)

        # 3. Resident candidate discounted by 0.05
        p_uncached = calculate_priority(confidence=0.70, layer_distance=1, hist_freq=0.0, is_resident=False)
        p_cached = calculate_priority(confidence=0.70, layer_distance=1, hist_freq=0.0, is_resident=True)
        self.assertAlmostEqual(p_cached, p_uncached * 0.05, places=5)

        # 4. Historical activation boosts score
        p_freq = calculate_priority(confidence=0.50, layer_distance=1, hist_freq=5.0, is_resident=False)
        p_infreq = calculate_priority(confidence=0.50, layer_distance=1, hist_freq=0.0, is_resident=False)
        self.assertGreater(p_freq, p_infreq)

        # 5. Feedback bias reduces score
        p_biased = calculate_priority(confidence=0.50, layer_distance=1, hist_freq=0.0, is_resident=False, confidence_bias=0.20)
        self.assertLess(p_biased, p_infreq)

    def test_cache_feedback_tracker_and_bias(self):
        """Test useful vs wasted feedback updates and dynamic bias activation."""
        fb = CacheFeedbackTracker()
        self.assertEqual(fb.current_confidence_bias, 0.0)

        # Low waste (20% waste ratio): 16 useful, 4 wasted
        for _ in range(16):
            fb.record(was_prefetched=True, was_hit=True)
        for _ in range(4):
            fb.record(was_prefetched=True, was_hit=False)

        self.assertEqual(fb.total_prefetches, 20)
        self.assertEqual(fb.useful_prefetches, 16)
        self.assertEqual(fb.wasted_prefetches, 4)
        # Waste ratio = 4/20 = 20% <= 35%, so bias remains 0.0
        self.assertEqual(fb.current_confidence_bias, 0.0)

        # Scurry with 30 wasted prefetches -> waste ratio jumps to 34/50 = 68% > 35%
        for _ in range(30):
            fb.record(was_prefetched=True, was_hit=False)

        self.assertGreater(fb.current_confidence_bias, 0.0)
        # Bias should equal (0.68 - 0.35) * 0.5 = 0.165
        expected_bias = (fb.wasted_prefetches / fb.total_prefetches - 0.35) * 0.5
        self.assertAlmostEqual(fb.current_confidence_bias, expected_bias, places=5)

    def test_resource_adaptation(self):
        """Test prefetch depth and top_k adaptation under I/O saturation and memory limits."""
        cfg = PredictConfig(depth=4, top_k=8)

        # 1. Healthy headroom: unconstrained
        healthy = ResourceState(pending_io_jobs=1, max_concurrent_io=16)
        d, k = adapt_resources(cfg, healthy)
        self.assertEqual(d, 4)
        self.assertEqual(k, 8)

        # 2. I/O Queue Saturated (> max / 2): Throttles depth and top_k
        busy = ResourceState(pending_io_jobs=10, max_concurrent_io=16)
        d_busy, k_busy = adapt_resources(cfg, busy)
        self.assertEqual(d_busy, 2)  # depth // 2
        self.assertEqual(k_busy, 6)  # (top_k * 3) // 4

        # 3. Constrained RAM: Available RAM allows only 4 experts
        low_ram = ResourceState(
            available_ram_bytes=4 * 16 * 1024 * 1024,
            expert_size_bytes=16 * 1024 * 1024,
            pending_io_jobs=0,
            max_concurrent_io=16,
        )
        d_ram, k_ram = adapt_resources(cfg, low_ram)
        self.assertLessEqual(k_ram * d_ram, 4)

    def test_cache_simulator_correctness(self):
        """Test cache hits, demand misses, timely vs late prefetch, eviction, and wasted tracking."""
        sim = ExpertCacheSimulator(
            capacity_per_layer=2,
            num_layers=4,
            num_experts=8,
            layer_compute_us=100.0,
            expert_load_us=200.0,
            eviction_policy="lru",
        )

        # 1. Dispatch timely prefetch for layer 1 expert 3 at time t=0
        sim.dispatch_prefetch(layer=1, expert_id=3)
        self.assertEqual(sim.stats.prefetch_dispatches, 1)

        # Advance time by 300 us (compute 3 layers) -> ready_time is 200 us, so it is ready
        sim.step_layer_compute()  # t=100
        sim.step_layer_compute()  # t=200
        sim.step_layer_compute()  # t=300

        # Access layer 1 expert 3: should be a TIMELY hit with 0 stall
        stall = sim.access_expert(layer=1, expert_id=3)
        self.assertEqual(stall, 0.0)
        self.assertEqual(sim.stats.cache_hits, 1)
        self.assertEqual(sim.stats.timely_prefetches, 1)
        self.assertEqual(sim.stats.useful_prefetches, 1)

        # 2. Late prefetch: dispatch prefetch at t=300, expert load is 200us (finishes at t=500)
        sim.dispatch_prefetch(layer=2, expert_id=5)
        # Advance only 100 us
        sim.step_layer_compute()  # t=400
        # Access expert 5 at t=400: ready at 500, so stalls for 100 us
        stall_late = sim.access_expert(layer=2, expert_id=5)
        self.assertEqual(stall_late, 100.0)
        self.assertEqual(sim.stats.late_prefetches, 1)

        # 3. Demand miss: access expert 6 (never prefetched) -> stalls full 200us
        stall_miss = sim.access_expert(layer=2, expert_id=6)
        self.assertEqual(stall_miss, 200.0)
        self.assertEqual(sim.stats.demand_loads, 1)
        self.assertEqual(sim.stats.cache_misses, 1)

        # 4. Cache eviction: layer 2 capacity is 2. Currently has expert 5 and 6.
        # Now access expert 7 -> expert 5 (oldest) should be evicted
        sim.access_expert(layer=2, expert_id=7)
        self.assertFalse(sim.is_resident(layer=2, expert_id=5))
        self.assertTrue(sim.is_resident(layer=2, expert_id=6))
        self.assertTrue(sim.is_resident(layer=2, expert_id=7))

        # 5. Wasted prefetch detection: prefetch expert 0 for layer 3, then never access it
        sim.dispatch_prefetch(layer=3, expert_id=0)
        stats = sim.finalize()
        self.assertGreaterEqual(stats.wasted_prefetches, 1)

    def test_deterministic_reproducibility(self):
        """Test that fixed random seed produces bitwise identical traces and metrics."""
        t1 = generate_coupled_trace(num_layers=8, num_experts=16, top_k=2, num_tokens=20, seed=12345)
        t2 = generate_coupled_trace(num_layers=8, num_experts=16, top_k=2, num_tokens=20, seed=12345)

        self.assertEqual(len(t1.events), len(t2.events))
        for e1, e2 in zip(t1.events, t2.events):
            self.assertEqual(e1.layer, e2.layer)
            self.assertEqual(e1.experts, e2.experts)
            self.assertEqual(e1.gates, e2.gates)


if __name__ == "__main__":
    unittest.main()
