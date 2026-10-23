"""
Unit tests for Layeris MoE architecture adapters, configuration, and router hooks.
"""

from __future__ import annotations

import unittest
from pathlib import Path

import torch
from transformers import AutoConfig, AutoModelForCausalLM

from layeris.moe.config import MoEConfig, ValidationOutcome, detect_moe_architecture
from layeris.moe.observation import RouterObservation
from layeris.moe.qwen import Qwen36MoEAdapter
from layeris.moe.router_hooks import RouterHookManager


class TestMoEAdapter(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Locate verified Qwen3.6 MoE tiny checkpoint
        root_dir = Path(__file__).resolve().parent.parent.parent
        cls.qwen_dir = root_dir / "c" / "qwen36_tiny"
        if not cls.qwen_dir.is_dir():
            # Alternative relative path if running from within layeris-python
            cls.qwen_dir = Path(__file__).resolve().parent.parent / "c" / "qwen36_tiny"

    def test_dense_model_rejection(self):
        """Verifies that dense model types and single-expert models are strictly rejected."""
        dense_cases = [
            {"model_type": "llama", "num_hidden_layers": 32, "hidden_size": 4096},
            {"model_type": "qwen2", "num_hidden_layers": 28, "hidden_size": 1536},
            {"model_type": "mistral", "num_hidden_layers": 32, "hidden_size": 4096},
            {"model_type": "qwen3_5_moe_text", "num_experts": 1, "num_experts_per_tok": 1},
            {"model_type": "custom_moe", "num_experts": 8, "num_experts_per_tok": 0},
            {"model_type": "custom_moe", "num_experts": 8, "num_experts_per_tok": 16},  # top_k > experts
        ]
        for case in dense_cases:
            with self.subTest(case=case):
                with self.assertRaises(ValueError):
                    detect_moe_architecture(case)

    def test_moe_config_validation(self):
        """Validates detection of genuine MoE checkpoint configuration."""
        if not self.qwen_dir.is_dir():
            self.skipTest("qwen36_tiny directory not found")

        cfg = detect_moe_architecture(self.qwen_dir)
        self.assertTrue(cfg.is_moe)
        self.assertEqual(cfg.num_layers, 8)
        self.assertEqual(cfg.num_experts, 8)
        self.assertEqual(cfg.num_experts_per_tok, 2)
        self.assertEqual(cfg.shared_expert_intermediate_size, 32)
        self.assertEqual(cfg.validation_outcome, ValidationOutcome.SUPPORTED_MOE)

    def test_qwen_adapter_router_discovery_and_hooks(self):
        """Tests router gate discovery, live observation extraction, and hook removal."""
        if not self.qwen_dir.is_dir():
            self.skipTest("qwen36_tiny directory not found")

        cfg = detect_moe_architecture(self.qwen_dir)
        adapter = Qwen36MoEAdapter(cfg)

        model_cfg = AutoConfig.from_pretrained(str(self.qwen_dir))
        model = AutoModelForCausalLM.from_pretrained(str(self.qwen_dir), config=model_cfg)
        model.eval()

        # Check router modules
        routers = adapter.get_router_modules(model)
        self.assertEqual(len(routers), 8)

        observations = []
        hook_mgr = RouterHookManager(adapter, on_observation=observations.append)
        hook_mgr.attach(model)
        self.assertEqual(len(hook_mgr.hooks), 8)

        # Single forward pass
        input_ids = torch.tensor([[10, 20, 30]])
        with torch.no_grad():
            outputs = model(input_ids)

        # 8 layers should have emitted observations
        self.assertEqual(len(observations), 8)
        for idx, obs in enumerate(observations):
            self.assertEqual(obs.layer_id, idx)
            self.assertEqual(obs.seq_len, 3)
            # 3 tokens * 2 experts per token = 6 selected expert indices
            self.assertEqual(len(obs.selected_expert_ids), 6)
            for eid in obs.selected_expert_ids:
                self.assertGreaterEqual(eid, 0)
                self.assertLess(eid, 8)

        # Test hook removal
        hook_mgr.remove()
        self.assertEqual(len(hook_mgr.hooks), 0)

        # Forward pass without hooks should not add observations
        with torch.no_grad():
            model(input_ids)
        self.assertEqual(len(observations), 8)

    def test_token_parity_with_and_without_hooks(self):
        """Verifies that attaching router observation hooks causes zero alteration in logits."""
        if not self.qwen_dir.is_dir():
            self.skipTest("qwen36_tiny directory not found")

        cfg = detect_moe_architecture(self.qwen_dir)
        adapter = Qwen36MoEAdapter(cfg)

        model_cfg = AutoConfig.from_pretrained(str(self.qwen_dir))
        model = AutoModelForCausalLM.from_pretrained(str(self.qwen_dir), config=model_cfg)
        model.eval()

        input_ids = torch.tensor([[5, 12, 45, 99]])

        # Baseline run without hooks
        with torch.no_grad():
            baseline_out = model(input_ids).logits

        # Run with hooks attached
        hook_mgr = RouterHookManager(adapter)
        hook_mgr.attach(model)
        with torch.no_grad():
            hooked_out = model(input_ids).logits
        hook_mgr.remove()

        # Logits must be bit-exact identical
        diff = torch.max(torch.abs(baseline_out - hooked_out)).item()
        self.assertEqual(diff, 0.0, f"Max absolute difference between baseline and hooked logits: {diff}")

    def test_live_moe_predictor_integration(self):
        """Tests live integration between router hooks, MultiLayerPredictor, and cache simulation."""
        if not self.qwen_dir.is_dir():
            self.skipTest("qwen36_tiny directory not found")

        cfg = detect_moe_architecture(self.qwen_dir)
        adapter = Qwen36MoEAdapter(cfg)

        model_cfg = AutoConfig.from_pretrained(str(self.qwen_dir))
        model = AutoModelForCausalLM.from_pretrained(str(self.qwen_dir), config=model_cfg)
        model.eval()

        from layeris.moe.live_predictor import LiveMoEPredictor
        from layeris.prefetch.predictor import PredictConfig

        pred_cfg = PredictConfig(depth=2, top_k=2, enabled=True, min_confidence=0.0)
        live_pred = LiveMoEPredictor(
            moe_config=cfg,
            predict_config=pred_cfg,
            cache_capacity_per_layer=4,
        )

        hook_mgr = RouterHookManager(adapter, on_observation=live_pred.on_observation)
        hook_mgr.attach(model)

        # 1. Simulate prefill (step 0, 3 tokens)
        input_ids = torch.tensor([[1, 2, 3]])
        hook_mgr.set_prefill(True)
        with torch.no_grad():
            out = model(input_ids)
        next_tok = torch.argmax(out.logits[:, -1, :], dim=-1, keepdim=True)

        # Prefill should have recorded into routing history
        self.assertEqual(live_pred.total_observations, 8)
        self.assertEqual(len(live_pred.telemetry_history), 0)  # No decode telemetry in prefill

        # 2. Simulate 4 iterative decode steps
        curr_ids = torch.cat([input_ids, next_tok], dim=-1)
        for step in range(1, 5):
            hook_mgr.advance_step()
            with torch.no_grad():
                out = model(curr_ids)
            next_tok = torch.argmax(out.logits[:, -1, :], dim=-1, keepdim=True)
            curr_ids = torch.cat([curr_ids, next_tok], dim=-1)

        hook_mgr.remove()

        # Telemetry should be captured for decode steps
        diag = live_pred.get_diagnostics()
        self.assertTrue(diag["enabled"])
        self.assertEqual(diag["total_layers"], 8)
        self.assertEqual(diag["num_experts"], 8)
        self.assertGreater(len(live_pred.telemetry_history), 0)
        self.assertIn("predictor_metrics", diag)
        self.assertIn("cache_simulation", diag)

        cache_sim = diag["cache_simulation"]
        self.assertEqual(cache_sim["capacity_per_layer"], 4)
        self.assertGreater(cache_sim["cache_hits"] + cache_sim["cache_misses"], 0)

    def test_live_moe_predictor_token_parity(self):
        """Verifies that enabling live prediction produces identical tokens to vanilla generation."""
        if not self.qwen_dir.is_dir():
            self.skipTest("qwen36_tiny directory not found")

        cfg = detect_moe_architecture(self.qwen_dir)
        adapter = Qwen36MoEAdapter(cfg)

        model_cfg = AutoConfig.from_pretrained(str(self.qwen_dir))
        model = AutoModelForCausalLM.from_pretrained(str(self.qwen_dir), config=model_cfg)
        model.eval()

        from layeris.moe.live_predictor import LiveMoEPredictor

        prompt = torch.tensor([[7, 14, 21, 28]])

        # Baseline greedy generation (5 tokens)
        gen_baseline = prompt.clone()
        for _ in range(5):
            with torch.no_grad():
                logits = model(gen_baseline).logits
            next_t = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
            gen_baseline = torch.cat([gen_baseline, next_t], dim=-1)

        # Instrumented generation with LiveMoEPredictor active
        live_pred = LiveMoEPredictor(moe_config=cfg)
        hook_mgr = RouterHookManager(adapter, on_observation=live_pred.on_observation)
        hook_mgr.attach(model)

        gen_predicted = prompt.clone()
        hook_mgr.set_prefill(True)
        with torch.no_grad():
            logits = model(gen_predicted).logits
        next_t = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
        gen_predicted = torch.cat([gen_predicted, next_t], dim=-1)

        for _ in range(4):
            hook_mgr.advance_step()
            with torch.no_grad():
                logits = model(gen_predicted).logits
            next_t = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
            gen_predicted = torch.cat([gen_predicted, next_t], dim=-1)

        hook_mgr.remove()

        # Token sequences must be strictly identical
        self.assertTrue(torch.equal(gen_baseline, gen_predicted),
                        f"Mismatch between baseline {gen_baseline} and predicted {gen_predicted}")


if __name__ == "__main__":
    unittest.main()
