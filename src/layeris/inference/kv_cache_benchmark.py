"""Microbenchmark TurboQuant cache storage and cache-update overhead."""

from __future__ import annotations

import argparse
import json
import time
from statistics import mean

try:
    import psutil
    import torch
    from transformers.cache_utils import DynamicLayer
except ImportError as exc:
    raise RuntimeError(
        "The KV-cache microbenchmark requires the inference dependencies from .[inference]."
    ) from exc

from layeris.inference.kv_cache import TurboQuantDynamicLayer


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _timed(callable_, device: torch.device) -> float:
    _synchronize(device)
    started = time.perf_counter()
    callable_()
    _synchronize(device)
    return time.perf_counter() - started


def run_benchmark(
    contexts: list[int],
    batch_size: int,
    num_heads: int,
    head_dim: int,
    device_name: str,
    repetitions: int,
    bits: int = 4,
    polar: bool = False,
) -> dict:
    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but PyTorch reports that CUDA is unavailable.")
    if head_dim < 32 or head_dim & (head_dim - 1):
        raise ValueError("head_dim must be a supported power of two >= 32.")
    if bits not in (3, 4) or (bits == 3) != polar:
        raise ValueError("Use bits=4 with polar=False, or bits=3 with polar=True.")

    process = psutil.Process()
    rss_start = process.memory_info().rss
    results = []

    for context in contexts:
        if context < 1:
            raise ValueError("Context lengths must be positive.")
        key_states = torch.randn(batch_size, num_heads, context, head_dim, device=device)
        value_states = torch.randn_like(key_states)
        next_key = torch.randn(batch_size, num_heads, 1, head_dim, device=device)
        next_value = torch.randn_like(next_key)

        prefill_times = []
        quantization_times = []
        decompression_times = []
        decode_times = []
        original_decode_times = []
        compressed_bytes = 0
        key_mse = value_mse = 0.0

        for repeat in range(repetitions):
            compressed = TurboQuantDynamicLayer(
                layer_idx=repeat, bits=bits, polar=polar
            )
            prefill_times.append(
                _timed(lambda: compressed.update(key_states, value_states), device)
            )
            key_codes, key_scales = compressed._encode(key_states)
            value_codes, value_scales = compressed._encode(value_states)
            quantization_times.append(
                _timed(
                    lambda: (
                        compressed._encode(key_states),
                        compressed._encode(value_states),
                    ),
                    device,
                )
            )
            decompressed = []
            decompression_times.append(
                _timed(
                    lambda: decompressed.extend(
                        (
                            compressed._decode(key_codes, key_scales),
                            compressed._decode(value_codes, value_scales),
                        )
                    ),
                    device,
                )
            )
            key_mse = torch.mean((decompressed[0] - key_states) ** 2).item()
            value_mse = torch.mean((decompressed[1] - value_states) ** 2).item()
            compressed_bytes = (
                compressed.keys.numel() * compressed.keys.element_size()
                + compressed.values.numel() * compressed.values.element_size()
                + compressed._key_scales.numel() * compressed._key_scales.element_size()
                + compressed._value_scales.numel() * compressed._value_scales.element_size()
                + compressed._signs.numel() * compressed._signs.element_size()
            )
            decode_times.append(
                _timed(lambda: compressed.update(next_key, next_value), device)
            )

            original = DynamicLayer()
            original.update(key_states, value_states)
            original_decode_times.append(
                _timed(lambda: original.update(next_key, next_value), device)
            )

        original_bytes = (
            key_states.numel() * key_states.element_size()
            + value_states.numel() * value_states.element_size()
        )
        results.append(
            {
                "context_tokens": context,
                "original_cache_bytes": original_bytes,
                "compressed_cache_bytes": compressed_bytes,
                "storage_reduction_ratio": original_bytes / compressed_bytes,
                "prefill_update_seconds_mean": mean(prefill_times),
                "quantization_seconds_mean": mean(quantization_times),
                "decompression_seconds_mean": mean(decompression_times),
                "incremental_update_seconds_mean": mean(decode_times),
                "original_incremental_update_seconds_mean": mean(original_decode_times),
                "key_reconstruction_mse": key_mse,
                "value_reconstruction_mse": value_mse,
            }
        )
        del key_states, value_states, next_key, next_value

    return {
        "benchmark_kind": "synthetic KV-cache microbenchmark (not full-model inference)",
        "cache_mode": "3-bit Polar" if polar else "4-bit TurboQuant-style scalar",
        "device": str(device),
        "dtype": "torch.float32",
        "batch_size": batch_size,
        "num_key_value_heads": num_heads,
        "head_dim": head_dim,
        "repetitions": repetitions,
        "process_rss_start_bytes": rss_start,
        "process_rss_end_bytes": process.memory_info().rss,
        "results": results,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contexts", type=int, nargs="+", default=[512, 1024, 2048, 4096])
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-heads", type=int, default=2)
    parser.add_argument("--head-dim", type=int, default=64)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--bits", type=int, choices=[3, 4], default=4)
    parser.add_argument("--polar", action="store_true")
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--output", type=str, default="")
    args = parser.parse_args()

    if args.repetitions < 1:
        parser.error("--repetitions must be positive.")
    if (args.bits == 3) != args.polar:
        parser.error("Use --bits 4 without --polar, or --bits 3 with --polar.")
    result = run_benchmark(
        contexts=args.contexts,
        batch_size=args.batch_size,
        num_heads=args.num_heads,
        head_dim=args.head_dim,
        device_name=args.device,
        repetitions=args.repetitions,
        bits=args.bits,
        polar=args.polar,
    )
    rendered = json.dumps(result, indent=2)
    if args.output:
        with open(args.output, "w", encoding="utf-8") as result_file:
            result_file.write(rendered + "\n")
    print(rendered)


if __name__ == "__main__":
    main()
