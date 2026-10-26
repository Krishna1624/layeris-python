"""Optional TurboQuant cache support for Transformers dynamic attention caches."""

from __future__ import annotations

import logging
import math
import os
from dataclasses import dataclass
from functools import lru_cache
from typing import Optional

try:
    import torch
    from transformers.cache_utils import CacheLayerMixin, DynamicCache, DynamicLayer
except ImportError:
    torch = None
    CacheLayerMixin = None
    DynamicCache = None
    DynamicLayer = None


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class KVCacheSettings:
    """Resolved environment configuration for the inference KV cache."""

    bits: int = 0
    polar: bool = False


def get_kv_cache_settings() -> KVCacheSettings:
    """Read the opt-in cache settings without changing the process environment."""
    raw_bits = os.environ.get("KV_TQ", "0").strip()
    raw_polar = os.environ.get("KV_TQ_POLAR", "0").strip()

    try:
        bits = int(raw_bits)
    except ValueError as exc:
        raise ValueError("KV_TQ must be 0 or 4.") from exc
    if bits not in (0, 4):
        raise ValueError("KV_TQ must be 0 (original cache) or 4 (TurboQuant).")
    if raw_polar not in ("0", "1"):
        raise ValueError("KV_TQ_POLAR must be 0 or 1.")
    return KVCacheSettings(bits=bits, polar=raw_polar == "1")


def _normal_lloyd_max_codebook(bits: int, device: "torch.device") -> tuple[torch.Tensor, torch.Tensor]:
    """Return Gaussian Lloyd-Max reconstruction levels and decision thresholds."""
    levels_count = 1 << bits

    def cdf(value: float) -> float:
        return 0.5 * (1.0 + math.erf(value / math.sqrt(2.0)))

    def pdf(value: float) -> float:
        return math.exp(-0.5 * value * value) / math.sqrt(2.0 * math.pi)

    def first_moment(value: float) -> float:
        return -pdf(value)

    boundaries = [-float("inf")] + [
        -4.0 + 8.0 * index / levels_count for index in range(1, levels_count)
    ] + [float("inf")]
    centroids = [
        (index - (levels_count - 1) / 2) * (6.0 / levels_count)
        for index in range(levels_count)
    ]

    for _ in range(80):
        updated = []
        for index in range(levels_count):
            left, right = boundaries[index], boundaries[index + 1]
            mass = cdf(right) - cdf(left)
            if mass <= 1e-12:
                updated.append(centroids[index])
            else:
                updated.append((first_moment(right) - first_moment(left)) / mass)
        centroids = updated
        boundaries = [-float("inf")] + [
            (centroids[index] + centroids[index + 1]) / 2
            for index in range(levels_count - 1)
        ] + [float("inf")]

    return (
        torch.tensor(centroids, dtype=torch.float32, device=device),
        torch.tensor(boundaries[1:-1], dtype=torch.float32, device=device),
    )


@lru_cache(maxsize=16)
def _cpu_codebook(bits: int) -> tuple[torch.Tensor, torch.Tensor]:
    return _normal_lloyd_max_codebook(bits, torch.device("cpu"))


def _fwht(values: "torch.Tensor") -> "torch.Tensor":
    """Apply the unnormalized Walsh-Hadamard transform to the last axis."""
    width = values.shape[-1]
    transformed = values.reshape(-1, width)
    stride = 1
    while stride < width:
        transformed = transformed.reshape(-1, width // (2 * stride), 2, stride)
        left = transformed[:, :, 0, :]
        right = transformed[:, :, 1, :]
        transformed = torch.stack((left + right, left - right), dim=2).reshape(-1, width)
        stride *= 2
    return transformed.reshape(values.shape)


def _pack_nibbles(codes: "torch.Tensor") -> "torch.Tensor":
    even = codes[..., 0::2].to(torch.uint8)
    odd = codes[..., 1::2].to(torch.uint8)
    return even | (odd << 4)


def _unpack_nibbles(packed: "torch.Tensor", width: int) -> "torch.Tensor":
    codes = torch.empty(
        (*packed.shape[:-1], width), dtype=torch.uint8, device=packed.device
    )
    codes[..., 0::2] = packed & 0x0F
    codes[..., 1::2] = packed >> 4
    return codes


class TurboQuantDynamicLayer(CacheLayerMixin):
    """A dynamic HF cache layer storing rotated 4-bit TurboQuant K/V codes."""

    is_sliding = False
    is_croppable = True
    is_compileable = False

    def __init__(self, layer_idx: int, bits: int = 4):
        super().__init__()
        self.layer_idx = layer_idx
        self.bits = bits
        self.width: Optional[int] = None
        self.length = 0
        self._key_scales = None
        self._value_scales = None
        self._key_original = None
        self._value_original = None
        self._signs = None
        self._fallback_reason: Optional[str] = None

    def _warn_fallback(self, reason: str) -> None:
        if self._fallback_reason is None:
            logger.warning(
                "TurboQuant cache disabled for layer %d: %s; retaining original-precision KV tensors.",
                self.layer_idx,
                reason,
            )
        self._fallback_reason = reason

    def _can_quantize(self, key_states: "torch.Tensor", value_states: "torch.Tensor") -> bool:
        width = key_states.shape[-1]
        if width < 32 or width & (width - 1):
            self._warn_fallback(f"head dimension {width} is not a supported power of two >= 32")
            return False
        if key_states.shape != value_states.shape:
            self._warn_fallback("key and value tensor shapes differ")
            return False
        if key_states.dtype not in (torch.float16, torch.bfloat16, torch.float32):
            self._warn_fallback(f"unsupported tensor dtype {key_states.dtype}")
            return False
        if not torch.isfinite(key_states).all() or not torch.isfinite(value_states).all():
            self._warn_fallback("input contains NaN or infinity")
            return False
        return True

    def _initialize_signs(self, width: int, device: "torch.device") -> None:
        generator = torch.Generator(device="cpu")
        generator.manual_seed(173 + self.layer_idx)
        signs = torch.randint(0, 2, (width,), generator=generator, dtype=torch.int8)
        self._signs = (signs.to(torch.float32).mul_(2).sub_(1)).to(device=device)
        self.width = width

    def _encode(self, values: "torch.Tensor") -> tuple["torch.Tensor", "torch.Tensor"]:
        rows = values.float()
        scales = rows.norm(dim=-1, keepdim=True) / math.sqrt(self.width)
        scales = scales.clamp_min(torch.finfo(torch.float32).tiny)
        rotated = _fwht(rows * self._signs) / math.sqrt(self.width)
        normalized = rotated / scales
        centers_cpu, thresholds_cpu = _cpu_codebook(self.bits)
        thresholds = thresholds_cpu.to(device=values.device)
        centers = centers_cpu.to(device=values.device)
        codes = torch.bucketize(normalized.contiguous(), thresholds)
        quantized = _pack_nibbles(codes)
        return quantized, scales

    def _decode(self, packed: "torch.Tensor", scales: "torch.Tensor") -> "torch.Tensor":
        codes = _unpack_nibbles(packed, self.width).long()
        centers_cpu, _ = _cpu_codebook(self.bits)
        rotated = centers_cpu.to(device=packed.device)[codes] * scales
        return (_fwht(rotated) / math.sqrt(self.width) * self._signs).to(self.dtype)

    def _enable_fallback(self, key_states: "torch.Tensor", value_states: "torch.Tensor") -> None:
        if self._key_original is None:
            if self.length:
                key_states = torch.cat((self._decode(self.keys, self._key_scales), key_states), dim=-2)
                value_states = torch.cat((self._decode(self.values, self._value_scales), value_states), dim=-2)
            self._key_original = key_states
            self._value_original = value_states
        else:
            self._key_original = torch.cat((self._key_original, key_states), dim=-2)
            self._value_original = torch.cat((self._value_original, value_states), dim=-2)
        self.keys = self._key_original
        self.values = self._value_original
        self._key_scales = self._value_scales = None
        self.length = self.keys.shape[-2]
        self.is_initialized = True

    def lazy_initialization(self, key_states: "torch.Tensor", value_states: "torch.Tensor") -> None:
        self.dtype, self.device = key_states.dtype, key_states.device
        self.is_initialized = True
        self.length = 0

    def update(
        self, key_states: "torch.Tensor", value_states: "torch.Tensor", *args, **kwargs
    ) -> tuple["torch.Tensor", "torch.Tensor"]:
        if key_states.shape != value_states.shape:
            raise ValueError("Key and value cache updates must have identical shapes.")
        if key_states.dtype != value_states.dtype or key_states.device != value_states.device:
            raise ValueError("Key and value cache updates must have identical dtype and device.")
        if self.length and key_states.shape[-1] != self.width:
            raise ValueError(
                f"Cache head dimension changed from {self.width} to {key_states.shape[-1]}."
            )
        if not self.is_initialized:
            self.lazy_initialization(key_states, value_states)
        if self._key_original is not None:
            self._enable_fallback(key_states, value_states)
            return self.keys, self.values
        if not self._can_quantize(key_states, value_states):
            self._enable_fallback(key_states, value_states)
            return self.keys, self.values

        if self.width is None:
            self._initialize_signs(key_states.shape[-1], key_states.device)

        key_codes, key_scales = self._encode(key_states)
        value_codes, value_scales = self._encode(value_states)
        if self.length:
            self.keys = torch.cat((self.keys, key_codes), dim=-2)
            self.values = torch.cat((self.values, value_codes), dim=-2)
            self._key_scales = torch.cat((self._key_scales, key_scales), dim=-2)
            self._value_scales = torch.cat((self._value_scales, value_scales), dim=-2)
        else:
            self.keys, self.values = key_codes, value_codes
            self._key_scales, self._value_scales = key_scales, value_scales
        self.length += key_states.shape[-2]
        return self._decode(self.keys, self._key_scales), self._decode(self.values, self._value_scales)

    def get_mask_sizes(self, query_length: int) -> tuple[int, int]:
        return self.length + query_length, 0

    def get_seq_length(self) -> int:
        return self.length

    def get_max_length(self) -> int:
        return -1

    def reset(self) -> None:
        self.keys = self.values = None
        self._key_scales = self._value_scales = None
        self._key_original = self._value_original = None
        self.length = 0
        self.width = None
        self._signs = None
        self._fallback_reason = None
        self.is_initialized = False

    def crop(self, tokens_to_remove: int) -> None:
        if tokens_to_remove > 0:
            raise ValueError("Dynamic TurboQuant caches only support crop with a non-positive value.")
        if tokens_to_remove == 0:
            return
        end = max(0, self.length + tokens_to_remove)
        if self._key_original is not None:
            self._key_original = self._key_original[..., :end, :]
            self._value_original = self._value_original[..., :end, :]
            self.keys, self.values = self._key_original, self._value_original
        else:
            self.keys = self.keys[..., :end, :]
            self.values = self.values[..., :end, :]
            self._key_scales = self._key_scales[..., :end, :]
            self._value_scales = self._value_scales[..., :end, :]
        self.length = end

    def batch_repeat_interleave(self, repeats: int) -> None:
        if self.length:
            self.keys = self.keys.repeat_interleave(repeats, dim=0)
            self.values = self.values.repeat_interleave(repeats, dim=0)
            if self._key_scales is not None:
                self._key_scales = self._key_scales.repeat_interleave(repeats, dim=0)
                self._value_scales = self._value_scales.repeat_interleave(repeats, dim=0)
            if self._key_original is not None:
                self._key_original = self.keys
                self._value_original = self.values

    def batch_select_indices(self, indices: "torch.Tensor") -> None:
        if self.length:
            self.keys = self.keys[indices, ...]
            self.values = self.values[indices, ...]
            if self._key_scales is not None:
                self._key_scales = self._key_scales[indices, ...]
                self._value_scales = self._value_scales[indices, ...]
            if self._key_original is not None:
                self._key_original = self.keys
                self._value_original = self.values

    def reorder_cache(self, beam_idx: "torch.LongTensor") -> None:
        self.batch_select_indices(beam_idx)


if DynamicCache is not None and CacheLayerMixin is not None:
    class TurboQuantCache(DynamicCache):
        """Hugging Face dynamic cache that keeps supported attention layers quantized."""

        def __init__(self, config, bits: int = 4):
            super().__init__(config=config)
            if not self.layers or any(not isinstance(layer, DynamicLayer) for layer in self.layers):
                raise ValueError("TurboQuant supports only models with standard full-attention dynamic cache layers.")
            self.layers = [
                TurboQuantDynamicLayer(layer_idx=index, bits=bits)
                for index in range(len(self.layers))
            ]
            self.bits = bits

        @property
        def compressed_nbytes(self) -> int:
            return sum(
                tensor.numel() * tensor.element_size()
                for layer in self.layers
                for tensor in (layer.keys, layer.values, layer._key_scales, layer._value_scales)
                if tensor is not None
            )

        @property
        def is_compressed(self) -> bool:
            return any(layer.length and layer._key_original is None for layer in self.layers)

        @property
        def compressed_layers(self) -> int:
            return sum(layer.length > 0 and layer._key_original is None for layer in self.layers)

        @property
        def fallback_layers(self) -> int:
            return sum(layer._key_original is not None for layer in self.layers)


else:
    class TurboQuantCache:
        def __init__(self, config, bits: int = 4):
            raise RuntimeError(
                "TurboQuant KV caching requires PyTorch and a Transformers release exposing CacheLayerMixin."
            )


def create_kv_cache(config, settings: Optional[KVCacheSettings] = None):
    """Create an enabled cache, or return ``None`` to use Transformers' original cache."""
    settings = settings or get_kv_cache_settings()
    if not settings.bits and not settings.polar:
        return None
    if settings.polar:
        logger.warning("3-bit Polar KV quantization is not implemented; using the original-precision cache.")
        return None
    if settings.bits != 4:
        logger.warning("Unsupported TurboQuant bit width %d; using the original-precision cache.", settings.bits)
        return None
    try:
        cache = TurboQuantCache(config=config, bits=settings.bits)
    except (RuntimeError, ValueError) as exc:
        logger.warning("TurboQuant KV cache unavailable: %s; using the original-precision cache.", exc)
        return None
    logger.info(
        "TurboQuant KV cache enabled: bits=%d layers=%d; unsupported layers fall back to original precision.",
        settings.bits,
        len(cache.layers),
    )
    return cache
