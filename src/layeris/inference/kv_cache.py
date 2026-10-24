"""Optional TurboQuant cache support for Transformers dynamic attention caches."""

from __future__ import annotations

import logging
import math
import os
from dataclasses import dataclass
from functools import lru_cache
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    # Pylance/Pyright always takes this branch (TYPE_CHECKING = True at analysis
    # time) so none of the following names are ever widened to None.
    import torch
    from transformers.cache_utils import CacheLayerMixin, DynamicCache, DynamicLayer

    # At type-check time these are the exact HF classes, not a union with object.
    _LayerBase = CacheLayerMixin
    _CacheBase = DynamicCache
else:
    try:
        import torch
        from transformers.cache_utils import CacheLayerMixin, DynamicCache, DynamicLayer

        _LayerBase = CacheLayerMixin
        _CacheBase = DynamicCache
    except ImportError:
        torch = None          # type: ignore[assignment]
        CacheLayerMixin = None  # type: ignore[assignment]
        DynamicCache = None     # type: ignore[assignment]
        DynamicLayer = None     # type: ignore[assignment]
        _LayerBase = object
        _CacheBase = object


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


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
        raise ValueError("KV_TQ must be 0, 3, or 4.") from exc
    if bits not in (0, 3, 4):
        raise ValueError("KV_TQ must be 0 (original cache), 3, or 4 (TurboQuant).")
    if raw_polar not in ("0", "1"):
        raise ValueError("KV_TQ_POLAR must be 0 or 1.")
    polar = raw_polar == "1"
    return KVCacheSettings(bits=bits, polar=polar)


# ---------------------------------------------------------------------------
# Codebook helpers
# ---------------------------------------------------------------------------


def _normal_lloyd_max_codebook(
    bits: int, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return Gaussian Lloyd-Max reconstruction levels and decision thresholds."""
    levels_count = 1 << bits

    def cdf(value: float) -> float:
        return 0.5 * (1.0 + math.erf(value / math.sqrt(2.0)))

    def pdf(value: float) -> float:
        return math.exp(-0.5 * value * value) / math.sqrt(2.0 * math.pi)

    def first_moment(value: float) -> float:
        return -pdf(value)

    boundaries: list[float] = [-float("inf")] + [
        -4.0 + 8.0 * index / levels_count for index in range(1, levels_count)
    ] + [float("inf")]
    centroids: list[float] = [
        (index - (levels_count - 1) / 2) * (6.0 / levels_count)
        for index in range(levels_count)
    ]

    for _ in range(80):
        updated: list[float] = []
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


# ---------------------------------------------------------------------------
# Tensor transform / packing utilities
# ---------------------------------------------------------------------------


def _fwht(values: torch.Tensor) -> torch.Tensor:
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


def _pack_nibbles(codes: torch.Tensor) -> torch.Tensor:
    """Pack 4-bit unsigned integers two-per-byte (nibble packing)."""
    even = codes[..., 0::2].to(torch.uint8)
    odd = codes[..., 1::2].to(torch.uint8)
    return even | (odd << 4)


def _unpack_nibbles(packed: torch.Tensor, width: int) -> torch.Tensor:
    """Unpack nibble-packed 4-bit unsigned integers."""
    codes = torch.empty(
        (*packed.shape[:-1], width), dtype=torch.uint8, device=packed.device
    )
    codes[..., 0::2] = packed & 0x0F
    codes[..., 1::2] = packed >> 4
    return codes


def _pack_3bit(codes: torch.Tensor) -> torch.Tensor:
    """Pack 3-bit unsigned integers along the last dimension (8 values → 3 bytes)."""
    shape = codes.shape
    width = shape[-1]
    flat = codes.reshape(-1, width // 8, 8).to(torch.uint8)
    c0, c1, c2 = flat[..., 0], flat[..., 1], flat[..., 2]
    c3, c4, c5 = flat[..., 3], flat[..., 4], flat[..., 5]
    c6, c7 = flat[..., 6], flat[..., 7]
    b0 = c0 | (c1 << 3) | ((c2 & 0x03) << 6)
    b1 = (c2 >> 2) | (c3 << 1) | (c4 << 4) | ((c5 & 0x01) << 7)
    b2 = (c5 >> 1) | (c6 << 2) | (c7 << 5)
    return torch.stack((b0, b1, b2), dim=-1).reshape(*shape[:-1], width * 3 // 8)


def _unpack_3bit(packed: torch.Tensor, width: int) -> torch.Tensor:
    """Unpack 3-bit unsigned integers from uint8 bytes along the last dimension."""
    shape = packed.shape
    triplets = packed.reshape(-1, width // 8, 3)
    b0, b1, b2 = triplets[..., 0], triplets[..., 1], triplets[..., 2]
    c0 = b0 & 0x07
    c1 = (b0 >> 3) & 0x07
    c2 = ((b0 >> 6) & 0x03) | ((b1 & 0x01) << 2)
    c3 = (b1 >> 1) & 0x07
    c4 = (b1 >> 4) & 0x07
    c5 = ((b1 >> 7) & 0x01) | ((b2 & 0x03) << 1)
    c6 = (b2 >> 2) & 0x07
    c7 = (b2 >> 5) & 0x07
    return torch.stack((c0, c1, c2, c3, c4, c5, c6, c7), dim=-1).reshape(*shape[:-1], width)


# ---------------------------------------------------------------------------
# Cache layer
# ---------------------------------------------------------------------------


class TurboQuantDynamicLayer(_LayerBase):
    """A dynamic HF cache layer storing rotated 3-bit Polar or 4-bit TurboQuant K/V codes."""

    is_sliding: bool = False
    is_croppable: bool = True
    is_compileable: bool = False

    def __init__(self, layer_idx: int, bits: int = 4, polar: bool = False) -> None:
        if bits not in (3, 4):
            raise ValueError("TurboQuant cache bit width must be 3 or 4.")
        if (bits == 3) != polar:
            raise ValueError("3-bit cache mode requires Polar quantization; 4-bit mode does not.")
        super().__init__()
        self.layer_idx: int = layer_idx
        self.bits: int = bits
        self.polar: bool = polar
        self.width: Optional[int] = None
        self.length: int = 0
        self.is_initialized: bool = False
        # Declared without assignment; set in lazy_initialization() before any use.
        self.dtype: torch.dtype
        self.device: torch.device
        # Quantised storage; None until the first update() call.
        self.keys: Optional[torch.Tensor] = None
        self.values: Optional[torch.Tensor] = None
        self._key_scales: Optional[torch.Tensor] = None
        self._value_scales: Optional[torch.Tensor] = None
        self._key_original: Optional[torch.Tensor] = None
        self._value_original: Optional[torch.Tensor] = None
        self._signs: Optional[torch.Tensor] = None
        self._fallback_reason: Optional[str] = None
        self._cache_prefix_shape: Optional[tuple[int, ...]] = None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _warn_fallback(self, reason: str) -> None:
        if self._fallback_reason is None:
            logger.warning(
                "TurboQuant cache disabled for layer %d: %s;"
                " retaining original-precision KV tensors.",
                self.layer_idx,
                reason,
            )
        self._fallback_reason = reason

    def _can_quantize(
        self, key_states: torch.Tensor, value_states: torch.Tensor
    ) -> bool:
        if key_states.ndim < 2:
            self._warn_fallback(
                "key and value tensors must have sequence and feature dimensions"
            )
            return False
        width = key_states.shape[-1]
        if width < 32 or width & (width - 1):
            self._warn_fallback(
                f"head dimension {width} is not a supported power of two >= 32"
            )
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

    def _initialize_signs(self, width: int, device: torch.device) -> None:
        generator = torch.Generator(device="cpu")
        generator.manual_seed(173 + self.layer_idx)
        signs = torch.randint(0, 2, (width,), generator=generator, dtype=torch.int8)
        self._signs = signs.to(torch.float32).mul_(2).sub_(1).to(device=device)
        self.width = width

    def _encode(
        self, values: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        assert self._signs is not None, "_signs not initialised; call _initialize_signs first"
        assert self.width is not None, "width not set"
        rows = values.float()
        tiny = torch.finfo(torch.float32).tiny
        if self.polar:
            # Polar (3-bit): store the exact vector norm; quantise only the direction.
            # Scaling by sqrt(width) maps unit-sphere marginals into the Gaussian
            # codebook range ~N(0, 1) so Lloyd-Max centroids are well-matched.
            norms = rows.norm(dim=-1, keepdim=True)
            unit_rows = rows / norms.clamp_min(tiny)
            rotated = _fwht(unit_rows * self._signs) / math.sqrt(self.width)
            to_quantize = rotated * math.sqrt(self.width)
            if not torch.isfinite(to_quantize).all() or not torch.isfinite(norms).all():
                raise FloatingPointError("Polar transform produced non-finite values.")
            _, thresholds_cpu = _cpu_codebook(self.bits)
            thresholds = thresholds_cpu.to(device=values.device)
            codes = torch.bucketize(to_quantize.contiguous(), thresholds)
            return _pack_3bit(codes), norms
        else:
            # TurboQuant (4-bit): scalar MSE quantisation with per-vector scale.
            scales = rows.norm(dim=-1, keepdim=True) / math.sqrt(self.width)
            scales = scales.clamp_min(tiny)
            rotated = _fwht(rows * self._signs) / math.sqrt(self.width)
            normalized = rotated / scales
            if not torch.isfinite(normalized).all() or not torch.isfinite(scales).all():
                raise FloatingPointError("TurboQuant transform produced non-finite values.")
            _, thresholds_cpu = _cpu_codebook(self.bits)
            thresholds = thresholds_cpu.to(device=values.device)
            codes = torch.bucketize(normalized.contiguous(), thresholds)
            return _pack_nibbles(codes), scales

    def _decode(
        self, packed: torch.Tensor, scales: torch.Tensor
    ) -> torch.Tensor:
        assert self._signs is not None, "_signs not initialised; call _initialize_signs first"
        assert self.width is not None, "width not set"
        tiny = torch.finfo(torch.float32).tiny
        if self.polar:
            # Polar decode: reconstruct the approximate direction, force it back to
            # unit norm, then scale by the stored exact norm — guarantees that
            # ‖decoded‖ == stored_norm == ‖original‖ to float32 precision.
            codes = _unpack_3bit(packed, self.width).long()
            centers_cpu, _ = _cpu_codebook(self.bits)
            direction = centers_cpu.to(device=packed.device)[codes]
            dir_norm = direction.norm(dim=-1, keepdim=True).clamp_min(tiny)
            unit_rotated = direction / dir_norm
            x_hat = _fwht(unit_rotated) / math.sqrt(self.width) * self._signs
            return (x_hat * scales).to(self.dtype)
        else:
            codes = _unpack_nibbles(packed, self.width).long()
            centers_cpu, _ = _cpu_codebook(self.bits)
            rotated = centers_cpu.to(device=packed.device)[codes] * scales
            return (_fwht(rotated) / math.sqrt(self.width) * self._signs).to(self.dtype)

    def _enable_fallback(
        self, key_states: torch.Tensor, value_states: torch.Tensor
    ) -> None:
        if self._key_original is None:
            if self.length:
                assert self.keys is not None
                assert self._key_scales is not None
                assert self.values is not None
                assert self._value_scales is not None
                key_states = torch.cat(
                    (self._decode(self.keys, self._key_scales), key_states), dim=-2
                )
                value_states = torch.cat(
                    (self._decode(self.values, self._value_scales), value_states), dim=-2
                )
            self._key_original = key_states
            self._value_original = value_states
        else:
            self._key_original = torch.cat((self._key_original, key_states), dim=-2)
            self._value_original = torch.cat(
                (self._value_original, value_states), dim=-2
            )
        self.keys = self._key_original
        self.values = self._value_original
        self._key_scales = self._value_scales = None
        self.length = self.keys.shape[-2]
        self.is_initialized = True

    # ------------------------------------------------------------------
    # CacheLayerMixin interface
    # ------------------------------------------------------------------

    def lazy_initialization(
        self, key_states: torch.Tensor, value_states: torch.Tensor
    ) -> None:
        if key_states.ndim < 2:
            raise ValueError(
                "Key and value cache tensors must have sequence and feature dimensions."
            )
        if key_states.shape != value_states.shape:
            raise ValueError("Key and value cache updates must have identical shapes.")
        if key_states.dtype != value_states.dtype or key_states.device != value_states.device:
            raise ValueError(
                "Key and value cache updates must have identical dtype and device."
            )
        self.dtype = key_states.dtype
        self.device = key_states.device
        self._cache_prefix_shape = tuple(key_states.shape[:-2])
        self.is_initialized = True
        self.length = 0

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        *_args: object,
        **_kwargs: object,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        _ = _args, _kwargs
        if key_states.ndim < 2:
            raise ValueError(
                "Key and value cache tensors must have sequence and feature dimensions."
            )
        if key_states.shape != value_states.shape:
            raise ValueError("Key and value cache updates must have identical shapes.")
        if key_states.dtype != value_states.dtype or key_states.device != value_states.device:
            raise ValueError(
                "Key and value cache updates must have identical dtype and device."
            )
        if self.length:
            if key_states.dtype != self.dtype or key_states.device != self.device:
                raise ValueError(
                    "KV cache updates must preserve the original dtype and device."
                )
            if key_states.shape[:-2] != self._cache_prefix_shape:
                raise ValueError(
                    "KV cache updates must preserve batch and head dimensions."
                )
            if key_states.shape[-1] != self.width:
                raise ValueError(
                    f"Cache head dimension changed from {self.width} to {key_states.shape[-1]}."
                )
        if key_states.shape[-2] == 0:
            raise ValueError("KV cache updates must contain at least one token.")
        if not self.is_initialized:
            self.lazy_initialization(key_states, value_states)

        if self._key_original is not None:
            self._enable_fallback(key_states, value_states)
            assert self.keys is not None and self.values is not None
            return self.keys, self.values

        if not self._can_quantize(key_states, value_states):
            self._enable_fallback(key_states, value_states)
            assert self.keys is not None and self.values is not None
            return self.keys, self.values

        if self.width is None:
            self._initialize_signs(key_states.shape[-1], key_states.device)

        try:
            key_codes, key_scales = self._encode(key_states)
            value_codes, value_scales = self._encode(value_states)
        except FloatingPointError as exc:
            self._warn_fallback(str(exc))
            self._enable_fallback(key_states, value_states)
            assert self.keys is not None and self.values is not None
            return self.keys, self.values

        if self.length:
            assert self.keys is not None and self._key_scales is not None
            assert self.values is not None and self._value_scales is not None
            self.keys = torch.cat((self.keys, key_codes), dim=-2)
            self.values = torch.cat((self.values, value_codes), dim=-2)
            self._key_scales = torch.cat((self._key_scales, key_scales), dim=-2)
            self._value_scales = torch.cat((self._value_scales, value_scales), dim=-2)
        else:
            self.keys, self.values = key_codes, value_codes
            self._key_scales, self._value_scales = key_scales, value_scales

        self.length += key_states.shape[-2]
        assert self.keys is not None and self._key_scales is not None
        assert self.values is not None and self._value_scales is not None
        return (
            self._decode(self.keys, self._key_scales),
            self._decode(self.values, self._value_scales),
        )

    def get_mask_sizes(self, query_length: int) -> tuple[int, int]:
        return self.length + query_length, 0

    def get_seq_length(self) -> int:
        return self.length

    def get_max_length(self) -> int:
        return -1

    def reset(self) -> None:
        self.keys = None
        self.values = None
        self._key_scales = None
        self._value_scales = None
        self._key_original = None
        self._value_original = None
        self.length = 0
        self.width = None
        self._signs = None
        self._fallback_reason = None
        self._cache_prefix_shape = None
        self.is_initialized = False

    def crop(self, tokens_to_remove: int) -> None:
        if tokens_to_remove > 0:
            raise ValueError(
                "Dynamic TurboQuant caches only support crop with a non-positive value."
            )
        if tokens_to_remove == 0:
            return
        end = max(0, self.length + tokens_to_remove)
        if self._key_original is not None:
            self._key_original = self._key_original[..., :end, :]
            self._value_original = self._value_original[..., :end, :]
            self.keys = self._key_original
            self.values = self._value_original
        else:
            assert self.keys is not None and self.values is not None
            assert self._key_scales is not None and self._value_scales is not None
            self.keys = self.keys[..., :end, :]
            self.values = self.values[..., :end, :]
            self._key_scales = self._key_scales[..., :end, :]
            self._value_scales = self._value_scales[..., :end, :]
        self.length = end

    def batch_repeat_interleave(self, repeats: int) -> None:
        if self.length:
            assert self.keys is not None and self.values is not None
            self.keys = self.keys.repeat_interleave(repeats, dim=0)
            self.values = self.values.repeat_interleave(repeats, dim=0)
            if self._key_scales is not None:
                self._key_scales = self._key_scales.repeat_interleave(repeats, dim=0)
            if self._value_scales is not None:
                self._value_scales = self._value_scales.repeat_interleave(repeats, dim=0)
            if self._key_original is not None:
                self._key_original = self.keys
                self._value_original = self.values

    def batch_select_indices(self, indices: torch.Tensor) -> None:
        if self.length:
            assert self.keys is not None and self.values is not None
            self.keys = self.keys[indices, ...]
            self.values = self.values[indices, ...]
            if self._key_scales is not None:
                self._key_scales = self._key_scales[indices, ...]
            if self._value_scales is not None:
                self._value_scales = self._value_scales[indices, ...]
            if self._key_original is not None:
                self._key_original = self.keys
                self._value_original = self.values

    def reorder_cache(self, beam_idx: torch.Tensor) -> None:
        self.batch_select_indices(beam_idx)


# ---------------------------------------------------------------------------
# Top-level cache
# ---------------------------------------------------------------------------


class TurboQuantCache(_CacheBase):
    """Hugging Face dynamic cache that keeps supported attention layers quantized."""

    def __init__(self, config: object, bits: int = 4, polar: bool = False) -> None:
        if DynamicCache is None or CacheLayerMixin is None:
            raise RuntimeError(
                "TurboQuant KV caching requires PyTorch and a Transformers release"
                " exposing CacheLayerMixin."
            )
        super().__init__(config=config)
        if not self.layers or any(
            not isinstance(layer, DynamicLayer) for layer in self.layers
        ):
            raise ValueError(
                "TurboQuant supports only models with standard full-attention"
                " dynamic cache layers."
            )
        self.layers = [
            TurboQuantDynamicLayer(layer_idx=index, bits=bits, polar=polar)
            for index in range(len(self.layers))
        ]
        self.bits: int = bits
        self.polar: bool = polar

    @property
    def compressed_nbytes(self) -> int:
        return sum(
            tensor.numel() * tensor.element_size()
            for layer in self.layers
            for tensor in (
                layer.keys,
                layer.values,
                layer._key_scales,
                layer._value_scales,
                layer._signs,
            )
            if tensor is not None
        )

    @property
    def is_compressed(self) -> bool:
        return any(
            layer.length and layer._key_original is None for layer in self.layers
        )

    @property
    def compressed_layers(self) -> int:
        return sum(
            layer.length > 0 and layer._key_original is None for layer in self.layers
        )

    @property
    def fallback_layers(self) -> int:
        return sum(layer._key_original is not None for layer in self.layers)


# ---------------------------------------------------------------------------
# Public factory
# ---------------------------------------------------------------------------


def create_kv_cache(
    config: object, settings: Optional[KVCacheSettings] = None
) -> Optional[TurboQuantCache]:
    """Create an enabled cache, or return ``None`` to use Transformers' original cache."""
    settings = settings or get_kv_cache_settings()
    if not settings.bits and not settings.polar:
        return None
    # Polar mode always uses 3-bit codes; honour an explicit bits override if given.
    bits = settings.bits if settings.bits else (3 if settings.polar else 4)
    if bits not in (3, 4) or (bits == 3) != settings.polar:
        logger.warning(
            "Unsupported KV cache mode bits=%d polar=%s; 3-bit mode requires Polar "
            "quantization and 4-bit mode requires polar=False. Using the "
            "original-precision cache.",
            bits,
            settings.polar,
        )
        return None
    try:
        cache = TurboQuantCache(config=config, bits=bits, polar=settings.polar)
    except (RuntimeError, ValueError) as exc:
        logger.warning(
            "TurboQuant KV cache unavailable: %s; using the original-precision cache.",
            exc,
        )
        return None
    logger.info(
        "TurboQuant KV cache enabled: bits=%d polar=%s layers=%d;"
        " unsupported layers fall back to original precision.",
        bits,
        settings.polar,
        len(cache.layers),
    )
    return cache
