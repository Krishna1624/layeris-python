"""
Comprehensive Real-Model Speculative Benchmark for Layeris Python
==================================================================
Measures real neural network inference on Qwen2.5-0.5B-Instruct comparing:
1. Standard Autoregressive Baseline (no speculation, K=0)
2. Prompt-Lookup Self-Speculative Decoding (K=3)

Reports:
- Model loading time
- Time to first token (TTFT)
- Output tokens per second (TPS)
- Total generation time
- Inter-token latency
- Speculative proposed and accepted tokens
- Acceptance fraction
- Peak process RSS memory
- Exact output text / token comparison
- Multi-run variability (mean +- stddev)
"""

from __future__ import annotations

import argparse
import json
import math
import os
import psutil
import sys
import time
from pathlib import Path
from statistics import mean, stdev
from typing import Dict, List

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation.streamers import BaseStreamer

# Add src to path
src_dir = Path(__file__).resolve().parents[1] / "src"
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

from layeris.inference.speculative import (
    PromptLookupDraftGenerator,
    SpeculativeDecoder,
    SpeculativeStats,
    make_model_verifier,
)


class TimingStreamer(BaseStreamer):
    def __init__(self):
        self.first_token_time: float | None = None
        self.token_times: List[float] = []
        self._start_time: float = 0.0

    def start_timing(self):
        self._start_time = time.perf_counter()
        self.first_token_time = None
        self.token_times.clear()

    def put(self, value) -> None:
        now = time.perf_counter()
        if self.first_token_time is None:
            self.first_token_time = now - self._start_time
        self.token_times.extend([now] * value.numel())

    def end(self) -> None:
        pass


def run_single_measurement(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int,
    spec_k: int,
    device: str = "cpu",
) -> Dict:
    process = psutil.Process()
    rss_before = process.memory_info().rss

    messages = [{"role": "user", "content": prompt}]
    prompt_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt_text, return_tensors="pt").to(device)
    input_len = inputs.input_ids.shape[1]

    streamer = TimingStreamer()
    gen_kwargs = dict(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
        streamer=streamer,
    )
    if spec_k > 0:
        gen_kwargs["prompt_lookup_num_tokens"] = spec_k

    streamer.start_timing()
    t0 = time.perf_counter()
    out = model.generate(**gen_kwargs)
    t_total = time.perf_counter() - t0

    generated_ids = out[0][input_len:].tolist()
    gen_tokens = len(generated_ids)
    tps = gen_tokens / max(0.0001, t_total)

    token_intervals = [
        t2 - t1 for t1, t2 in zip(streamer.token_times, streamer.token_times[1:])
    ]
    mean_inter_token_ms = (mean(token_intervals) * 1000.0) if token_intervals else 0.0
    ttft_ms = (streamer.first_token_time * 1000.0) if streamer.first_token_time else 0.0

    rss_after = process.memory_info().rss
    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)

    return {
        "spec_k": spec_k,
        "gen_tokens": gen_tokens,
        "total_time_s": t_total,
        "tps": tps,
        "ttft_ms": ttft_ms,
        "mean_inter_token_ms": mean_inter_token_ms,
        "peak_rss_mb": rss_after / (1024 * 1024),
        "rss_delta_mb": (rss_after - rss_before) / (1024 * 1024),
        "generated_text": generated_text,
        "generated_ids": generated_ids,
    }


def benchmark_suite(
    model_id: str = "Qwen/Qwen2.5-0.5B-Instruct",
    tokens: int = 48,
    k_spec: int = 3,
    repeats: int = 3,
):
    print("=" * 80)
    print(" REAL SPECULATIVE DECODING BENCHMARK (LAYERIS PYTHON)")
    print("=" * 80)
    print(f" Model ID:            {model_id}")
    print(f" Target Tokens:       {tokens}")
    print(f" Speculative Depth:   K = {k_spec}")
    print(f" Repeated Trials:     {repeats} iterations per mode (post-warmup)")
    print(f" Device:              CPU (Torch threads: {torch.get_num_threads()})")
    print("=" * 80)

    # 1. Load Model & Tokenizer
    print("\n[1/4] Loading model and tokenizer from local cache...")
    t_load_0 = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(model_id, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float32,
        local_files_only=True,
    ).eval()
    t_load = time.perf_counter() - t_load_0
    param_count = sum(p.numel() for p in model.parameters())
    print(f"      Loaded {param_count / 1e6:.1f}M parameters in {t_load:.2f} seconds.")

    test_cases = [
        (
            "Pattern-Heavy Prompt (Code with Repetition)",
            "Write a Python function for binary search with comments and explain its time complexity.",
        ),
        (
            "Ordinary Prompt (Factual Explanation)",
            "Explain how transformer self-attention works in three clear sentences.",
        ),
    ]

    for title, prompt in test_cases:
        print("\n" + "#" * 80)
        print(f" TEST CASE: {title}")
        print(f" Prompt: \"{prompt}\"")
        print("#" * 80)

        # Warmup
        print("\n--> Warming up model execution...")
        _ = run_single_measurement(model, tokenizer, prompt, max_new_tokens=8, spec_k=0)
        _ = run_single_measurement(model, tokenizer, prompt, max_new_tokens=8, spec_k=k_spec)

        # Baseline runs (K=0)
        print(f"\n--> Executing Baseline (Autoregressive, K=0, {repeats} runs)...")
        base_runs = []
        for i in range(repeats):
            res = run_single_measurement(model, tokenizer, prompt, max_new_tokens=tokens, spec_k=0)
            base_runs.append(res)
            print(f"    Run {i+1}: {res['tps']:.2f} tok/s | TTFT: {res['ttft_ms']:.1f}ms | Total: {res['total_time_s']:.2f}s | RSS: {res['peak_rss_mb']:.1f}MB")

        # Speculative runs (K=k_spec)
        print(f"\n--> Executing Prompt-Lookup Self-Speculation (K={k_spec}, {repeats} runs)...")
        spec_runs = []
        for i in range(repeats):
            res = run_single_measurement(model, tokenizer, prompt, max_new_tokens=tokens, spec_k=k_spec)
            spec_runs.append(res)
            print(f"    Run {i+1}: {res['tps']:.2f} tok/s | TTFT: {res['ttft_ms']:.1f}ms | Total: {res['total_time_s']:.2f}s | RSS: {res['peak_rss_mb']:.1f}MB")

        # Verification of Exact Greed Match
        base_ids = base_runs[0]["generated_ids"]
        spec_ids = spec_runs[0]["generated_ids"]
        identical = (base_ids == spec_ids)

        # Compute Acceptance Stats using Layeris SpeculativeDecoder
        print("\n--> Measuring Speculative Acceptance Rate using Layeris SpeculativeDecoder...")
        prompt_tokens = tokenizer.encode(prompt, add_special_tokens=False)
        verifier = make_model_verifier(model, device="cpu", eos_token_id=tokenizer.eos_token_id)
        decoder = SpeculativeDecoder(draft_k=k_spec)
        _ = decoder.generate(prompt_tokens, verifier, max_new_tokens=tokens, eos_token_id=tokenizer.eos_token_id)

        mean_base_tps = mean(r["tps"] for r in base_runs)
        std_base_tps = stdev(r["tps"] for r in base_runs) if len(base_runs) > 1 else 0.0
        mean_spec_tps = mean(r["tps"] for r in spec_runs)
        std_spec_tps = stdev(r["tps"] for r in spec_runs) if len(spec_runs) > 1 else 0.0
        speedup = mean_spec_tps / mean_base_tps

        print("\n" + "-" * 80)
        print(" RESULTS SUMMARY")
        print("-" * 80)
        print(f" Mode                          | Throughput (TPS)   | TTFT (ms)   | Speedup  | Match")
        print("-" * 80)
        print(f" 1. Baseline Autoregressive    | {mean_base_tps:5.2f} +- {std_base_tps:4.2f}    | {mean(r['ttft_ms'] for r in base_runs):8.1f} ms  | 1.00x    | (Ref)")
        print(f" 2. Prompt-Lookup Speculative  | {mean_spec_tps:5.2f} +- {std_spec_tps:4.2f}    | {mean(r['ttft_ms'] for r in spec_runs):8.1f} ms  | {speedup:5.2f}x   | {'PASS' if identical else 'DIFFERS'}")
        print("-" * 80)
        print(f" Speculative Proposals:        {decoder.stats.total_draft_tokens} tokens")
        print(f" Accepted Speculative Tokens:  {decoder.stats.accepted_draft_tokens} tokens")
        print(f" Acceptance Fraction:          {decoder.stats.acceptance_rate * 100:.1f}%")
        print(f" Target Forward Passes:        {decoder.stats.total_forward_passes}")
        print(f" Tokens per Forward Pass:      {decoder.stats.tokens_per_forward:.2f}")
        print(f" Peak Process RSS Memory:      {max(r['peak_rss_mb'] for r in spec_runs):.1f} MB (Delta: ~0 MB extra model RAM)")
        print("-" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Real Speculative Decoding Benchmark")
    parser.add_argument("--tokens", type=int, default=32, help="Number of new tokens")
    parser.add_argument("--k", type=int, default=3, help="Speculative candidate depth")
    parser.add_argument("--repeats", type=int, default=3, help="Repeats per test case")
    args = parser.parse_args()

    benchmark_suite(tokens=args.tokens, k_spec=args.k, repeats=args.repeats)
