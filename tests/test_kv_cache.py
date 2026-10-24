"""Tests for the optional compressed Hugging Face KV cache."""

from __future__ import annotations

import os
import unittest
from unittest.mock import patch

import torch

from layeris.inference.kv_cache import (
    KVCacheSettings,
    TurboQuantDynamicLayer,
    _pack_3bit,
    _unpack_3bit,
    create_kv_cache,
    get_kv_cache_settings,
)


class TestTurboQuantDynamicLayer(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)

    def test_prefill_and_incremental_decode_preserve_shape_and_dtype(self):
        layer = TurboQuantDynamicLayer(layer_idx=0)
        keys = torch.randn(2, 3, 12, 64, dtype=torch.float32)
        values = torch.randn_like(keys)

        full_keys, full_values = layer.update(keys, values)
        self.assertEqual(full_keys.shape, keys.shape)
        self.assertEqual(full_values.shape, values.shape)
        self.assertEqual(full_keys.dtype, keys.dtype)
        self.assertTrue(torch.isfinite(full_keys).all())
        self.assertTrue(torch.isfinite(full_values).all())

        next_keys = torch.randn(2, 3, 1, 64)
        next_values = torch.randn_like(next_keys)
        full_keys, full_values = layer.update(next_keys, next_values)
        self.assertEqual(full_keys.shape, (2, 3, 13, 64))
        self.assertEqual(full_values.shape, (2, 3, 13, 64))
        self.assertEqual(layer.get_seq_length(), 13)
        self.assertIsNone(layer._key_original)

    def test_quantization_error_is_finite_and_bounded(self):
        layer = TurboQuantDynamicLayer(layer_idx=2)
        source = torch.randn(1, 2, 32, 64)
        restored, _ = layer.update(source, source)
        mse = torch.mean((restored - source) ** 2).item()
        self.assertTrue(torch.isfinite(torch.tensor(mse)))
        self.assertLess(mse, 0.2)
        self.assertGreater(mse, 0.0)

    def test_compressed_storage_is_smaller_than_fp32_cache(self):
        layer = TurboQuantDynamicLayer(layer_idx=0)
        source = torch.randn(2, 4, 24, 64)
        layer.update(source, source)
        original_nbytes = source.numel() * source.element_size() * 2
        compressed_nbytes = sum(
            value.numel() * value.element_size()
            for value in (layer.keys, layer.values, layer._key_scales, layer._value_scales)
        )
        self.assertLess(compressed_nbytes, original_nbytes)

    def test_unsupported_dimensions_keep_original_precision(self):
        layer = TurboQuantDynamicLayer(layer_idx=1)
        keys = torch.randn(1, 2, 3, 48)
        values = torch.randn_like(keys)
        cached_keys, cached_values = layer.update(keys, values)
        self.assertIs(layer._key_original, cached_keys)
        self.assertTrue(torch.equal(cached_keys, keys))
        self.assertTrue(torch.equal(cached_values, values))

    def test_non_finite_values_fall_back_without_modifying_input(self):
        layer = TurboQuantDynamicLayer(layer_idx=0)
        keys = torch.randn(1, 2, 2, 64)
        values = torch.randn_like(keys)
        keys[0, 0, 0, 0] = float("nan")
        cached_keys, cached_values = layer.update(keys, values)
        self.assertTrue(torch.isnan(cached_keys[0, 0, 0, 0]))
        self.assertTrue(torch.equal(cached_values, values))
        self.assertIs(layer._key_original, cached_keys)

    def test_zero_vectors_and_fp16_remain_stable(self):
        layer = TurboQuantDynamicLayer(layer_idx=3)
        zeros = torch.zeros(1, 1, 4, 64, dtype=torch.float16)
        restored, _ = layer.update(zeros, zeros)
        self.assertEqual(restored.dtype, torch.float16)
        self.assertTrue(torch.equal(restored, zeros))

    def test_three_bit_polar_packs_three_bits_and_preserves_norm(self):
        layer = TurboQuantDynamicLayer(layer_idx=0, bits=3, polar=True)
        source = torch.randn(1, 2, 12, 64)
        restored, _ = layer.update(source, source)
        self.assertEqual(layer.keys.shape[-1], 24)
        torch.testing.assert_close(
            restored.norm(dim=-1), source.norm(dim=-1), rtol=1e-5, atol=1e-5
        )
        self.assertTrue(torch.isfinite(restored).all())

    def test_polar_quantization_preserves_zero_vectors(self):
        layer = TurboQuantDynamicLayer(layer_idx=0, bits=3, polar=True)
        zeros = torch.zeros(1, 1, 3, 64)
        restored, _ = layer.update(zeros, zeros)
        self.assertTrue(torch.equal(restored, zeros))

    def test_rejects_incompatible_bit_and_polar_modes(self):
        with self.assertRaises(ValueError):
            TurboQuantDynamicLayer(layer_idx=0, bits=4, polar=True)
        with self.assertRaises(ValueError):
            TurboQuantDynamicLayer(layer_idx=0, bits=3, polar=False)

    def test_three_bit_packing_round_trips_all_code_values(self):
        codes = torch.arange(8, dtype=torch.uint8).repeat(8)
        self.assertTrue(torch.equal(_unpack_3bit(_pack_3bit(codes), 64), codes))

    def test_long_context_remains_finite(self):
        layer = TurboQuantDynamicLayer(layer_idx=1)
        source = torch.randn(1, 2, 512, 64)
        restored_keys, restored_values = layer.update(source, source)
        self.assertEqual(restored_keys.shape, source.shape)
        self.assertEqual(restored_values.shape, source.shape)
        self.assertTrue(torch.isfinite(restored_keys).all())
        self.assertTrue(torch.isfinite(restored_values).all())

    def test_non_finite_increment_falls_back_preserving_existing_prefix(self):
        layer = TurboQuantDynamicLayer(layer_idx=0)
        initial = torch.randn(1, 2, 3, 64)
        initial_keys, initial_values = layer.update(initial, initial)
        invalid = torch.randn(1, 2, 1, 64)
        invalid[0, 0, 0, 0] = float("inf")
        keys, values = layer.update(invalid, invalid)
        self.assertEqual(keys.shape[-2], 4)
        self.assertTrue(torch.equal(keys[..., :3, :], initial_keys))
        self.assertTrue(torch.equal(values[..., :3, :], initial_values))
        self.assertTrue(torch.isinf(keys[0, 0, 3, 0]))

    def test_finite_but_overflowing_transform_falls_back(self):
        layer = TurboQuantDynamicLayer(layer_idx=0)
        large_finite = torch.full((1, 1, 2, 64), 1e38, dtype=torch.float32)
        cached_keys, cached_values = layer.update(large_finite, large_finite)
        self.assertIs(layer._key_original, cached_keys)
        self.assertTrue(torch.equal(cached_keys, large_finite))
        self.assertTrue(torch.equal(cached_values, large_finite))

    def test_batch_reorder_repeat_and_crop(self):
        layer = TurboQuantDynamicLayer(layer_idx=0)
        keys = torch.randn(2, 2, 8, 64)
        layer.update(keys, keys)
        layer.batch_repeat_interleave(2)
        self.assertEqual(layer.keys.shape[0], 4)
        layer.batch_select_indices(torch.tensor([3, 1]))
        self.assertEqual(layer.keys.shape[0], 2)
        layer.crop(-3)
        self.assertEqual(layer.get_seq_length(), 5)
        self.assertEqual(layer._decode(layer.keys, layer._key_scales).shape[-2], 5)

    def test_reset_allows_reinitialization(self):
        layer = TurboQuantDynamicLayer(layer_idx=0)
        source = torch.randn(1, 2, 3, 64)
        layer.update(source, source)
        layer.reset()
        self.assertEqual(layer.get_seq_length(), 0)
        next_source = torch.randn(1, 2, 1, 32)
        restored, _ = layer.update(next_source, next_source)
        self.assertEqual(restored.shape, next_source.shape)
        self.assertIsNone(layer._key_original)

    def test_rejects_invalid_tensor_rank_and_empty_updates(self):
        layer = TurboQuantDynamicLayer(layer_idx=0)
        vector = torch.randn(64)
        with self.assertRaisesRegex(ValueError, "sequence and feature dimensions"):
            layer.update(vector, vector)
        empty = torch.empty(1, 2, 0, 64)
        with self.assertRaisesRegex(ValueError, "at least one token"):
            layer.update(empty, empty)

    def test_rejects_batch_head_dtype_and_device_changes(self):
        layer = TurboQuantDynamicLayer(layer_idx=0)
        source = torch.randn(1, 2, 2, 64)
        layer.update(source, source)

        with self.subTest("batch size"):
            other_batch = torch.randn(2, 2, 1, 64)
            with self.assertRaisesRegex(ValueError, "batch and head dimensions"):
                layer.update(other_batch, other_batch)

        with self.subTest("head count"):
            other_heads = torch.randn(1, 3, 1, 64)
            with self.assertRaisesRegex(ValueError, "batch and head dimensions"):
                layer.update(other_heads, other_heads)

        with self.subTest("dtype"):
            other_dtype = source[:, :, :1, :].to(torch.float16)
            with self.assertRaisesRegex(ValueError, "original dtype and device"):
                layer.update(other_dtype, other_dtype)


class TestTurboQuantConfiguration(unittest.TestCase):
    def test_environment_settings(self):
        with patch.dict(os.environ, {"KV_TQ": "4", "KV_TQ_POLAR": "0"}):
            self.assertEqual(get_kv_cache_settings().bits, 4)
            self.assertFalse(get_kv_cache_settings().polar)
        with patch.dict(os.environ, {"KV_TQ": "0", "KV_TQ_POLAR": "1"}):
            self.assertEqual(get_kv_cache_settings().bits, 0)
            self.assertTrue(get_kv_cache_settings().polar)
        with patch.dict(os.environ, {"KV_TQ": "invalid"}):
            with self.assertRaises(ValueError):
                get_kv_cache_settings()

    def test_polar_request_uses_original_cache(self):
        settings = KVCacheSettings(bits=0, polar=True)
        self.assertIsNone(create_kv_cache(config=None, settings=settings))

    def test_polar_request_creates_3bit_cache(self):
        from transformers import Qwen2Config

        config = Qwen2Config(
            hidden_size=256,
            intermediate_size=384,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
        )
        cache = create_kv_cache(
            config, settings=KVCacheSettings(bits=0, polar=True)
        )
        self.assertIsNotNone(cache)
        self.assertEqual(cache.bits, 3)
        self.assertTrue(cache.polar)

    def test_incompatible_setting_falls_back_to_original_cache(self):
        settings = KVCacheSettings(bits=4, polar=True)
        self.assertIsNone(create_kv_cache(config=None, settings=settings))


class TestTurboQuantModelIntegration(unittest.TestCase):
    def test_qwen2_generation_updates_compressed_cache(self):
        try:
            from transformers import Qwen2Config, Qwen2ForCausalLM
            from layeris.inference.kv_cache_model_benchmark import _run_generation
        except ImportError:
            self.skipTest("Qwen2 model implementation is unavailable in this Transformers install")

        config = Qwen2Config(
            vocab_size=128,
            hidden_size=256,
            intermediate_size=384,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            max_position_embeddings=64,
            use_cache=True,
        )
        model = Qwen2ForCausalLM(config).eval()
        input_ids = torch.tensor([[1, 4, 7, 9], [1, 5, 8, 10]])

        class TokenizerStub:
            pad_token_id = 0
            eos_token_id = 127

            @staticmethod
            def decode(token_ids, skip_special_tokens=True):
                return " ".join(str(token_id) for token_id in token_ids)

        original = _run_generation(
            model, TokenizerStub(), input_ids[:1], max_new_tokens=3, compressed=False
        )
        turboquant = _run_generation(
            model, TokenizerStub(), input_ids[:1], max_new_tokens=3, compressed=True
        )

        self.assertEqual(original["generated_tokens"], 3)
        self.assertEqual(turboquant["generated_tokens"], 3)
        self.assertGreater(original["kv_cache_bytes"], turboquant["kv_cache_bytes"])
        self.assertEqual(turboquant["turboquant_compressed_layers"], 2)
        self.assertEqual(turboquant["original_precision_fallback_layers"], 0)
        self.assertIsNotNone(turboquant["time_to_first_token_seconds"])
        self.assertGreater(turboquant["tokens_per_second"], 0)

        polar = _run_generation(
            model,
            TokenizerStub(),
            input_ids[:1],
            max_new_tokens=3,
            compressed=True,
            cache_settings=KVCacheSettings(bits=3, polar=True),
        )
        self.assertEqual(polar["cache_mode_used"], "3-bit Polar")
        self.assertEqual(polar["turboquant_compressed_layers"], 2)
        self.assertEqual(polar["original_precision_fallback_layers"], 0)


if __name__ == "__main__":
    unittest.main()
