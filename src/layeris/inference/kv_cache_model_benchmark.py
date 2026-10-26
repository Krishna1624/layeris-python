"""Compare original and TurboQuant caches during real Transformers generation."""

from __future__ import annotations

import argparse
import json
import logging
import math
import psutil
import time
from difflib import SequenceMatcher
from statistics import mean

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.cache_utils import DynamicCache
from transformers.generation.streamers import BaseStreamer

from layeris.inference.kv_cache import KVCacheSettings, create_kv_cache


class _TimingStreamer(BaseStreamer):
    def __init__(self):
        self._waiting_for_prompt = True
        self.token_timestamps: list[float] = []

    def put(self, value) -> None:
        if self._waiting_for_prompt:
            self._waiting_for_prompt = False
        else:
            now = time.perf_counter()
            self.token_timestamps.extend([now] * value.numel())

    def end(self) -> None:
        return None


def _tensor_cache_bytes(cache) -> int:
    return sum(
        tensor.numel() * tensor.element_size()
        for layer in cache.layers
        for tensor in (layer.keys, layer.values)
        if tensor is not None
    )


def _run_generation(model, tokenizer, input_ids, max_new_tokens: int, compressed: bool) -> dict:
    cache = create_kv_cache(model.config, KVCacheSettings(bits=4)) if compressed else None
    requested_cache_unavailable = compressed and cache is None
    if cache is None:
        cache = DynamicCache(config=model.config)
    streamer = _TimingStreamer()
    kwargs = {
        "input_ids": input_ids,
        "max_new_tokens": max_new_tokens,
        "do_sample": False,
        "pad_token_id": tokenizer.pad_token_id or tokenizer.eos_token_id,
        "streamer": streamer,
    }
    if cache is not None:
        kwargs["past_key_values"] = cache

    process = psutil.Process()
    if torch.cuda.is_available() and input_ids.device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(input_ids.device)
    started = time.perf_counter()
    sequences = model.generate(**kwargs)
    elapsed = time.perf_counter() - started

    sequence = sequences[0]
    generated_ids = sequence[input_ids.shape[-1]:]
    generated_tokens = int(generated_ids.numel())
    timestamps = streamer.token_timestamps
    first_token_latency = timestamps[0] - started if timestamps else None
    decode_latencies = [
        later - earlier for earlier, later in zip(timestamps, timestamps[1:])
    ]
    if hasattr(cache, "compressed_nbytes"):
        actual_cache_bytes = cache.compressed_nbytes
        compressed_layers = cache.compressed_layers
        fallback_layers = cache.fallback_layers
    else:
        actual_cache_bytes = _tensor_cache_bytes(cache)
        compressed_layers = 0
        fallback_layers = len(cache.layers) if requested_cache_unavailable else 0

    return {
        "generated_token_ids": generated_ids.detach().cpu().tolist(),
        "generated_text": tokenizer.decode(
            generated_ids.detach().cpu(), skip_special_tokens=True
        ),
        "generated_tokens": generated_tokens,
        "cache_mode_requested": "turboquant_4bit" if compressed else "original",
        "cache_mode_used": "turboquant" if compressed_layers else "original",
        "turboquant_cache_unavailable": requested_cache_unavailable,
        "elapsed_seconds": elapsed,
        "time_to_first_token_seconds": first_token_latency,
        "average_decode_latency_seconds": mean(decode_latencies) if decode_latencies else None,
        "tokens_per_second": generated_tokens / elapsed if elapsed else 0.0,
        "kv_cache_bytes": actual_cache_bytes,
        "turboquant_compressed_layers": compressed_layers,
        "original_precision_fallback_layers": fallback_layers,
        "process_rss_after_generation_bytes": process.memory_info().rss,
        "cuda_peak_allocated_bytes": (
            torch.cuda.max_memory_allocated(input_ids.device)
            if torch.cuda.is_available() and input_ids.device.type == "cuda"
            else None
        ),
    }


def _make_context(tokenizer, prompt: str, context_tokens: int, device: torch.device):
    token_ids = tokenizer(prompt, add_special_tokens=False, return_tensors="pt").input_ids
    if token_ids.shape[-1] == 0:
        raise ValueError("The benchmark prompt must tokenize to at least one token.")
    repeats = math.ceil(context_tokens / token_ids.shape[-1])
    return token_ids.repeat(1, repeats)[:, :context_tokens].to(device)


def run_benchmark(
    model_id: str,
    contexts: list[int],
    max_new_tokens: int,
    prompt: str,
    device_name: str,
) -> dict:
    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but PyTorch reports that CUDA is unavailable.")
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    process = psutil.Process()
    rss_before_load = process.memory_info().rss
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=dtype,
        device_map="auto" if device.type == "cuda" else None,
        trust_remote_code=True,
    ).eval()
    if device.type == "cpu":
        model = model.to(device)
    model_device = model.get_input_embeddings().weight.device
    rss_after_load = process.memory_info().rss

    measurements = []
    for context_tokens in contexts:
        if context_tokens < 1:
            raise ValueError("Context lengths must be positive.")
        input_ids = _make_context(tokenizer, prompt, context_tokens, model_device)
        original = _run_generation(
            model, tokenizer, input_ids, max_new_tokens, compressed=False
        )
        turboquant = _run_generation(
            model, tokenizer, input_ids, max_new_tokens, compressed=True
        )
        baseline_ids = original.pop("generated_token_ids")
        compressed_ids = turboquant.pop("generated_token_ids")
        similarity = SequenceMatcher(a=baseline_ids, b=compressed_ids).ratio()
        measurements.append(
            {
                "context_tokens": context_tokens,
                "original_cache": original,
                "turboquant_cache": turboquant,
                "generated_token_sequence_similarity": similarity,
            }
        )

    return {
        "benchmark_kind": "full-model Transformers generation",
        "model": model_id,
        "device": str(model_device),
        "dtype": str(dtype),
        "max_new_tokens": max_new_tokens,
        "prompt_repeated_to_requested_context_length": True,
        "process_rss_before_model_load_bytes": rss_before_load,
        "process_rss_after_model_load_bytes": rss_after_load,
        "measurements": measurements,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--contexts", type=int, nargs="+", default=[512, 1024, 2048, 4096])
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--prompt", default="Explain the purpose of a key-value cache.")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--output", default="")
    args = parser.parse_args()
    if args.max_new_tokens < 1:
        parser.error("--max-new-tokens must be positive.")

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    report = run_benchmark(
        model_id=args.model,
        contexts=args.contexts,
        max_new_tokens=args.max_new_tokens,
        prompt=args.prompt,
        device_name=args.device,
    )
    rendered = json.dumps(report, indent=2)
    if args.output:
        with open(args.output, "w", encoding="utf-8") as result_file:
            result_file.write(rendered + "\n")
    print(rendered)


if __name__ == "__main__":
    main()
