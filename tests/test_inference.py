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
from layeris.inference.speculative import (
    PromptLookupDraftGenerator,
    SpeculativeDecoder,
    SpeculativeStats,
)


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

        stats.reset()
        self.assertEqual(stats.total_tokens_generated, 0)
        self.assertEqual(stats.total_forward_passes, 0)
        self.assertEqual(stats.total_draft_tokens, 0)
        self.assertEqual(stats.accepted_draft_tokens, 0)

    def test_candidate_verification_full_acceptance(self):
        # Target sequence after [1, 2]: [3, 4, 5, 6]
        def mock_predict(prefix):
            target_seq = [1, 2, 3, 4, 5, 6, 7]
            idx = len(prefix)
            return target_seq[idx] if idx < len(target_seq) else 99

        prompt = [1, 2]
        drafts = [3, 4, 5]
        from layeris.inference.speculative import verify_candidates, SpeculativeDecoder

        accepted, bonus, is_eos = verify_candidates(mock_predict, prompt, drafts, eos_token_id=99)
        self.assertEqual(accepted, [3, 4, 5])
        self.assertEqual(bonus, 6)
        self.assertFalse(is_eos)

    def test_candidate_verification_partial_acceptance(self):
        # Target sequence: [1, 2, 3, 4, 999] (mismatch at index 4)
        def mock_predict(prefix):
            mapping = {
                (1, 2): 3,
                (1, 2, 3): 4,
                (1, 2, 3, 4): 999,
            }
            return mapping.get(tuple(prefix), 0)

        prompt = [1, 2]
        drafts = [3, 4, 888] # 3 and 4 match, 888 mismatches
        from layeris.inference.speculative import verify_candidates

        accepted, correction, is_eos = verify_candidates(mock_predict, prompt, drafts)
        self.assertEqual(accepted, [3, 4])
        self.assertEqual(correction, 999) # Emits target model's correction token
        self.assertFalse(is_eos)

    def test_candidate_verification_zero_acceptance(self):
        # Target sequence: [1, 2, 999]
        def mock_predict(prefix):
            return 999

        prompt = [1, 2]
        drafts = [100, 101, 102] # All mismatch
        from layeris.inference.speculative import verify_candidates

        accepted, correction, is_eos = verify_candidates(mock_predict, prompt, drafts)
        self.assertEqual(accepted, [])
        self.assertEqual(correction, 999)
        self.assertFalse(is_eos)

    def test_candidate_verification_eos_handling(self):
        EOS = 1000
        # Target generates [3, EOS]
        def mock_predict(prefix):
            if prefix == [1, 2]:
                return 3
            if prefix == [1, 2, 3]:
                return EOS
            return 0

        from layeris.inference.speculative import verify_candidates
        # Case A: EOS in draft tokens
        accepted, next_tok, is_eos = verify_candidates(mock_predict, [1, 2], [3, EOS], eos_token_id=EOS)
        self.assertEqual(accepted, [3, EOS])
        self.assertIsNone(next_tok)
        self.assertTrue(is_eos)

        # Case B: EOS in correction token
        accepted, next_tok, is_eos = verify_candidates(mock_predict, [1, 2], [3, 4], eos_token_id=EOS)
        self.assertEqual(accepted, [3])
        self.assertEqual(next_tok, EOS)
        self.assertTrue(is_eos)

    def test_speculative_decoder_full_generation_and_fallback(self):
        from layeris.inference.speculative import SpeculativeDecoder, verify_candidates

        # Target generates repeated pattern: [10, 20, 30, 40, 10, 20, 30, 40, 10, 20]
        ground_truth = [10, 20, 30, 40, 10, 20, 30, 40, 10, 20, 30, 40]
        def target_predict(prefix):
            idx = len(prefix)
            return ground_truth[idx] if idx < len(ground_truth) else 0

        def verifier(prompt, drafts):
            return verify_candidates(target_predict, prompt, drafts)

        # 1. Speculative mode (draft_k=2)
        decoder = SpeculativeDecoder(draft_k=2)
        # Prompt: [10, 20, 30, 40, 10, 20] -> prompt-lookup will draft [30, 40]!
        prompt = [10, 20, 30, 40, 10, 20]
        generated = decoder.generate(prompt, verifier, max_new_tokens=4)
        self.assertEqual(generated, [30, 40, 10, 20])
        self.assertGreater(decoder.stats.accepted_draft_tokens, 0)
        self.assertGreater(decoder.stats.acceptance_rate, 0.0)

        # 2. Disabled speculative mode (draft_k=0) -> 100% autoregressive fallback
        baseline_decoder = SpeculativeDecoder(draft_k=0)
        gen_base = baseline_decoder.generate(prompt, verifier, max_new_tokens=4)
        self.assertEqual(gen_base, [30, 40, 10, 20])
        self.assertEqual(baseline_decoder.stats.total_draft_tokens, 0)
        self.assertEqual(baseline_decoder.stats.accepted_draft_tokens, 0)
        self.assertEqual(baseline_decoder.stats.tokens_per_forward, 1.0)

    def test_speculative_decoder_accepted_tokens(self):
        decoder = SpeculativeDecoder(draft_k=3)
        # Suffix [1, 2, 3] matches at index 0, candidates are [4, 5]
        prompt_tokens = [1, 2, 3, 4, 5, 1, 2, 3]

        def verifier_stub(prefix, drafts):
            self.assertEqual(drafts, [4, 5, 1])
            # Target model accepts draft tokens and generates next token 6
            return [4, 5, 1], 6, False

        new_tokens, is_eos = decoder.step(prompt_tokens, verifier_stub)
        self.assertEqual(new_tokens, [4, 5, 1, 6])
        self.assertFalse(is_eos)
        self.assertEqual(decoder.stats.total_tokens_generated, 4)
        self.assertEqual(decoder.stats.total_draft_tokens, 3)
        self.assertEqual(decoder.stats.accepted_draft_tokens, 3)
        self.assertEqual(decoder.stats.total_forward_passes, 1)
        self.assertEqual(decoder.stats.acceptance_rate, 1.0)

    def test_speculative_decoder_rejected_fallback(self):
        decoder = SpeculativeDecoder(draft_k=3)
        prompt_tokens = [10, 20, 30, 40]  # No repeated n-gram -> drafts is []

        def verifier_stub(prefix, drafts):
            self.assertEqual(drafts, [])
            # Target model rejects/has no drafts, emits single autoregressive token 50
            return [], 50, False

        new_tokens, is_eos = decoder.step(prompt_tokens, verifier_stub)
        self.assertEqual(new_tokens, [50])
        self.assertFalse(is_eos)
        self.assertEqual(decoder.stats.total_tokens_generated, 1)
        self.assertEqual(decoder.stats.total_draft_tokens, 0)
        self.assertEqual(decoder.stats.accepted_draft_tokens, 0)
        self.assertEqual(decoder.stats.total_forward_passes, 1)

    def test_speculative_decoder_eos_handling(self):
        decoder = SpeculativeDecoder(draft_k=2)
        prompt_tokens = [100, 200, 100, 200]  # Drafts: [] or whatever

        def verifier_stub(prefix, drafts):
            # Target model hits end-of-sequence
            return [], None, True

        new_tokens, is_eos = decoder.step(prompt_tokens, verifier_stub)
        self.assertEqual(new_tokens, [])
        self.assertTrue(is_eos)

    def test_prompt_lookup_repeated_patterns_vs_non_reusable(self):
        gen = PromptLookupDraftGenerator(max_draft_tokens=4, ngram_size=3)
        # Pattern-heavy code sequence: 'def binary_search(arr, target): ... def binary_search(arr'
        code_tokens = [101, 202, 303, 404, 505, 606, 707, 101, 202, 303]
        draft_code = gen.generate_draft(code_tokens)
        self.assertEqual(draft_code, [404, 505, 606, 707])

        # Non-reusable arbitrary prose sequence without repeated 3-grams
        prose_tokens = [11, 22, 33, 44, 55, 66, 77, 88, 99]
        draft_prose = gen.generate_draft(prose_tokens)
        self.assertEqual(draft_prose, [])


if __name__ == "__main__":
    unittest.main()

