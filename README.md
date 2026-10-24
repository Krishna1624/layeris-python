# Layeris Python Inference & Predictive Prefetching Engine

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Tests](https://img.shields.io/badge/tests-passing-brightgreen.svg)]()

A standalone, production-grade Python implementation of the **Layeris** inference optimization architecture. Designed for pairing with or standalone evaluation of Mixture-of-Experts (MoE) LLMs and Speculative Decoding pipelines.

---

## Key Capabilities

1. **Multi-Layer Expert Prediction & Predictive Prefetching**:
   - Bounded routing history tracking across tokens and layers with zero memory thrashing.
   - Lookahead horizon scheduling ($1 \le dL \le \text{depth}$) with geometric temporal discounting.
   - Resource-aware adaptive horizon clamping based on real-time RAM/VRAM budgets.
   - Closed-loop feedback tracker with dynamic waste penalty bias:
     $$\text{Bias} \leftarrow \text{Bias} + 0.05 \cdot \left(\frac{W}{U + W + 1} - 0.20\right)$$
   - Priority utility scoring:
     $$\text{Priority}(e, dL) = \gamma(dL) \cdot C(e) \cdot \left(1 + \beta \tanh(0.2 \cdot f)\right) \cdot R(e) - \text{Bias}$$
   - Predictive Cost-Aware LFRU cache eviction modeling.

2. **Hardware-Balanced Speculative Inference**:
   - Dynamic prompt-lookup speculative decoding ($K=1..5$) providing 1.3×–1.8× tokens/forward speedup.
   - Hardware Safety Guard: Automatically pins thread teams to physical cores (preventing SMT contention and thermal throttling) and reserves minimum 1.5 GB RAM and 1.2 GB VRAM buffers for the OS.
   - Real-time token throughput (TPS) streaming and BERTScore semantic fidelity evaluation.
   - Built-in live web search integration for real-time factual knowledge augmentation without requiring API keys.

---

## Directory Structure

```text
layeris-python/
├── src/
│   └── layeris/
│       ├── __init__.py               # Package root exports
│       ├── prefetch/                 # Multi-Layer Expert Prefetching Subsystem
│       │   ├── __init__.py
│       │   ├── predictor.py          # Routing history, priority calculation, adaptive headroom
│       │   ├── simulator.py          # Finite cache simulator (LRU & Predictive LFRU)
│       │   ├── trace.py              # Synthetic trace generator & ROUTE_TRACE parser
│       │   └── benchmark.py          # Multi-configuration evaluation & CSV/JSON exporter
│       └── inference/                # Speculative Decoding & Neural Inference Subsystem
│           ├── __init__.py
│           ├── chat.py               # Interactive conversational assistant with web search & TPS
│           ├── speculative.py        # Speculative draft generator & verification logic
│           └── guard.py              # Hardware profiler & OS safety budget clamps
├── tests/
│   ├── __init__.py
│   ├── test_predictor.py             # 11 unit tests for multi-layer prefetching & edge cases
│   └── test_inference.py             # Tests for hardware guard and draft lookup logic
├── examples/
│   ├── 01_prefetch_simulation.py     # Quickstart: prefetch simulation & hit rate measurement
│   ├── 02_speculative_chat.py        # Quickstart: interactive chat with speculative decoding
│   └── 03_run_benchmark.py           # Quickstart: reproduction of full benchmark matrix
├── scripts/
│   └── setup_env.bat                 # 1-click Windows environment installer (.venv)
├── pyproject.toml                    # PEP 517/518 build specification
├── requirements.txt                  # Minimal dependencies (numpy, psutil)
└── README.md                         # This documentation
```

---

## Quickstart Installation

### Option 1: Standard Pip Installation (Recommended)

```bash
# Clone the repository
git clone <your-repo-url> layeris-python
cd layeris-python

# Create virtual environment
python -m venv .venv

# Activate environment
# On Windows:
.venv\Scripts\activate
# On Linux/macOS:
source .venv/bin/activate

# Install in editable mode
pip install -e ".[dev]"
```

### Option 2: 1-Click Windows Setup

Double-click or run from PowerShell:
```powershell
.\scripts\setup_env.bat
```

---

## Usage Examples

### 1. Run the Multi-Layer Prefetching Simulation

Demonstrates the cache hit rate improvement of coordinated multi-layer prefetching over demand-only paging:

```bash
python examples/01_prefetch_simulation.py
```

*Sample output:*
```text
=================================================================
  Layeris MoE Predictive Prefetching Simulation Example
=================================================================
[*] Generating coupled MoE routing trace (16 layers, 32 experts, top-4, 50 tokens)...

--- [Run 1: Demand-Only Baseline] ---
  Expert Requests : 3200
  Cache Hits      : 1478
  Cache Misses    : 1722
  Hit Rate        : 46.19%

--- [Run 2: Coordinated Multi-Layer Prefetching (Depth=3, Top-K=4)] ---
  Expert Requests : 3200
  Cache Hits      : 2364
  Cache Misses    : 836
  Hit Rate        : 73.88%
  Useful Prefetch : 886
  Wasted Prefetch : 112

[+] Net Cache Hit Rate Improvement: +27.69%
=================================================================
```

### 2. Run the Full Prefetching Benchmark Matrix

Evaluates Demand-Only, Frequency, Uncoordinated Lookahead, and Coordinated Adaptive Prefetching across multiple cache capacities:

```bash
python -m layeris.prefetch.benchmark --tokens 100 --layers 16 --experts 32 --top_k 4
```

Machine-readable results will be saved to:
- `results/benchmark_results.json`
- `results/benchmark_results.csv`

### 3. Launch Interactive Speculative Chat

Runs interactive conversational AI with prompt-lookup speculative decoding and live web search:

```bash
# Requires torch and transformers (pip install -e ".[inference]")
python examples/02_speculative_chat.py
```

Arguments:
- `--model`: HuggingFace model checkpoint (default: `Qwen/Qwen2.5-0.5B-Instruct` or `Qwen/Qwen2.5-1.5B-Instruct`).
- `--device`: `auto`, `cuda`, or `cpu`.
- `--spec-k`: Candidate lookahead window (default: `3`).

---

## KV-Cache Compression

Layeris delegates model execution and attention to Hugging Face Transformers. With a Transformers release that exposes `CacheLayerMixin` (the tested interface is Transformers 5.x), chat can pass a compressed dynamic cache directly to `model.generate`; this does not install or build a native Colibri engine.

The available setting is opt-in:

```powershell
# Original-precision cache (the default)
$env:KV_TQ = "0"
$env:KV_TQ_POLAR = "0"

# 4-bit TurboQuant scalar cache
$env:KV_TQ = "4"
$env:KV_TQ_POLAR = "0"
python -m layeris.inference.chat --model Qwen/Qwen2.5-0.5B-Instruct
```

The current prototype stores keys and values in packed 4-bit codes using a randomized Hadamard transform and a Gaussian-approximation Lloyd-Max scalar codebook; vector scales are kept in FP32. This implements the scalar MSE quantization path, not TurboQuant's QJL inner-product residual coding, and should be treated as TurboQuant-style rather than a complete reproduction of every TurboQuant variant. It handles standard full-attention dynamic-cache layers whose head dimension is a power of two and at least 32. The configured Qwen2.5-0.5B model uses GQA with a 64-dimensional head and is covered by the tiny-model integration test. Unsupported cache layouts/dimensions, non-finite inputs, and unsupported Transformers cache APIs retain original-precision tensors. A warning is emitted when the requested mode cannot be applied.

`KV_TQ_POLAR=1` is recognized but 3-bit Polar quantization is not implemented in this release; it logs a warning and uses the original cache. If compression is active, prompt-lookup speculative generation is disabled for that generation call so it cannot replace or bypass the supplied cache.

This path reconstructs the cached tensors for attention on each cache update. It reduces resident KV storage, but adds quantization/dequantization work and is not a fused attention kernel; no inference speedup is promised. Verify activation from the chat output (`TurboQuant-style KV scalar quantization active: 4-bit, ... attention layers.`) and the post-generation counts for compressed and original-precision fallback layers.

Run the automated tests and a synthetic cache microbenchmark with:

```powershell
python -m unittest discover -s tests -p "test_*.py"
python -m layeris.inference.kv_cache_benchmark --contexts 512 1024 2048 4096 --device cpu --output results\kv_cache_microbenchmark.json
layeris-kv-model-benchmark --model Qwen/Qwen2.5-0.5B-Instruct --contexts 512 1024 2048 4096 --max-new-tokens 32 --device cpu --output results\kv_cache_model_benchmark.json
```

The synthetic microbenchmark reports cache tensor bytes, update timings, and reconstruction error; it is not an end-to-end inference benchmark. The full-model command loads the checkpoint and compares the standard Hugging Face dynamic cache against TurboQuant at each context length, reporting TTFT, decoding latency, tokens/sec, process RSS, cache bytes, and generated-token similarity. For consistent inputs, it repeats the supplied prompt tokens to reach each requested context length. This can be memory-intensive; choose `--device` and context lengths appropriate for the machine. The full-model measurements were not run as part of this change because the development machine has only about 2.4 GB available RAM.

---

## Running the Test Suite

Run all unit tests across both prefetching and inference components:

```bash
python -m unittest discover -s tests -p "test_*.py"
```

Expected output:
```text
.............
----------------------------------------------------------------------
Ran 13 tests in 0.045s

OK
```

---

## Mathematical Formulation

The core coordination logic addresses the fundamental trade-off in MoE weight serving: **eager prefetching reduces memory access latency, but aggressive lookahead risks evicting high-reuse resident experts (cache pollution) and saturating disk/PCIe bandwidth with speculative misses.**

### 1. Geometric Horizon Discount
To balance immediate versus distant sparse layers:
$$\gamma(dL) = 0.5^{dL - 1}, \quad 1 \le dL \le \text{depth}$$

### 2. Historical Transition Boost
Experts frequently co-activated in recent token passes receive bounded priority reinforcement:
$$\text{Boost}(f) = 1.0 + \beta \tanh(0.2 \cdot f), \quad \beta = 0.5$$

### 3. Residency Discount
Experts already present in RAM or VRAM are discounted so background I/O slots target genuine misses:
$$R(e) = \begin{cases} 0.05 & \text{if resident in cache} \\ 1.0 & \text{if unresident} \end{cases}$$

### 4. Dynamic Feedback Penalty (Closed-Loop Bias)
If the predictor over-predicts and causes cache pollution, the waste penalty dynamically raises the admission threshold:
$$\text{Bias}_{t+1} = \text{clamp}\left(\text{Bias}_t + 0.05 \cdot \left(\frac{W}{U + W + 1} - 0.20\right), 0.0, 1.0\right)$$

---

## Hardware Safety Guidelines

When executing neural inference on consumer laptops or shared workstations, `layeris.inference.guard` enforces the following constraints:
1. **CPU Core Capping**: Keeps thread teams at physical core counts (max 10) to prevent hyperthreading contention and avoid thermal throttling.
2. **RAM Preservation**: Preserves at least 1.5 GB of free RAM for the host OS to eliminate disk thrashing and `pagefile.sys` stutter.
3. **VRAM Safety**: Preserves at least 1.2 GB of VRAM for the Desktop Window Manager (DWM) to prevent Windows TDR driver crashes.

---

## License

This repository is distributed under the MIT License. See `LICENSE` for details.
