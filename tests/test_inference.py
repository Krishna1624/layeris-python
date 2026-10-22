"""
Unit tests for Layeris Inference & Hardware Guard components.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

src_path = Path(__file__).resolve().parent.parent / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

from layeris.inference.guard import detect_hardware_capabilities, get_cpu_info, get_ram_info
from layeris.inference.speculative import PromptLookupDraftGenerator, SpeculativeStats


class TestInferenceComponents(unittest.TestCase):
    def test_cpu_and_ram_guard(self):
        cpu = get_cpu_info()
        self.assertGreaterEqual(cpu["physical_cores"], 1)
        self.assertGreaterEqual(cpu["logical_threads"], 1)
        self.assertLessEqual(cpu["safe_threads"], cpu["physical_cores"])
        self.assertLessEqual(cpu["safe_threads"], 10)

        ram = get_ram_info()
        self.assertGreater(ram["total_gb"], 0.0)
        self.assertGreater(ram["available_gb"], 0.0)
        self.assertGreaterEqual(ram["safe_budget_gb"], 0.5)

    def test_hardware_profile_detection(self):
        profile = detect_hardware_capabilities()
        self.assertIn(profile.recommended_device, ["cuda", "cpu"])
        self.assertGreaterEqual(profile.safe_threads, 1)

    def test_prompt_lookup_generator(self):
        gen = PromptLookupDraftGenerator(max_draft_tokens=3, ngram_size=2)
        # Sequence with repeated n-gram: [10, 20, 30, 40, 50, 10, 20] -> match at index 0, candidate continuation: [30, 40, 50]
        tokens = [10, 20, 30, 40, 50, 10, 20]
        draft = gen.generate_draft(tokens)
        self.assertEqual(draft, [30, 40, 50])

    def test_prompt_lookup_no_match(self):
        gen = PromptLookupDraftGenerator(max_draft_tokens=3, ngram_size=2)
        tokens = [1, 2, 3, 4, 5]
        draft = gen.generate_draft(tokens)
        self.assertEqual(draft, [])

    def test_speculative_stats(self):
        stats = SpeculativeStats(
            total_tokens_generated=10,
            total_forward_passes=5,
            total_draft_tokens=8,
            accepted_draft_tokens=6,
            start_time_s=1.0,
            end_time_s=2.0,
        )
        self.assertEqual(stats.tokens_per_forward, 2.0)
        self.assertEqual(stats.acceptance_rate, 0.75)
        self.assertEqual(stats.tokens_per_second, 10.0)


if __name__ == "__main__":
    unittest.main()
