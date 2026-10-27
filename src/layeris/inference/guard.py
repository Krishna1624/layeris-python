"""
Layeris Hardware Guard & Dynamic Resource Balancer
==================================================
Monitors CPU, RAM, VRAM, and GPU in real-time.
Calculates optimal, safe hardware budgets to maximize throughput without:
  1. Overheating / thermal throttling the CPU
  2. Thrashing memory pagefile / swap (OOM)
  3. Crashing GPU display drivers (TDR timeout)
"""

from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import psutil

try:
    import torch
except ImportError:
    torch = None


@dataclass
class HardwareProfile:
    physical_cores: int
    logical_threads: int
    safe_threads: int
    total_ram_gb: float
    available_ram_gb: float
    safe_ram_gb: float
    has_gpu: bool
    cuda_available: bool
    hardware_cuda_capable: bool
    gpu_name: Optional[str]
    total_vram_gb: float
    free_vram_gb: float
    safe_vram_gb: float
    recommended_device: str
    # 3-Tier Adaptive Hardware Fields
    has_discrete_gpu: bool = False
    discrete_gpu_name: Optional[str] = None
    has_integrated_gpu: bool = False
    integrated_gpu_name: Optional[str] = None
    vulkan_available: bool = False
    adaptive_backend: str = "cpu"  # "cuda", "vulkan", "cpu"
    adaptive_reason: str = "Host CPU execution"


def get_cpu_info() -> Dict[str, int]:
    """Returns physical and logical CPU core counts."""
    physical = psutil.cpu_count(logical=False) or 8
    logical = psutil.cpu_count(logical=True) or 16
    safe_threads = min(physical, 10)
    return {
        "physical_cores": physical,
        "logical_threads": logical,
        "safe_threads": safe_threads,
    }


def get_ram_info() -> Dict[str, float]:
    """Returns total, available, and safe RAM budgets in GB."""
    vm = psutil.virtual_memory()
    total_gb = vm.total / (1024 ** 3)
    available_gb = vm.available / (1024 ** 3)
    safe_ram_gb = max(0.5, available_gb - 1.5)
    return {
        "total_gb": round(total_gb, 2),
        "available_gb": round(available_gb, 2),
        "used_percent": vm.percent,
        "safe_budget_gb": round(safe_ram_gb, 2),
    }


def get_gpu_info() -> Dict[str, any]:
    """Inspects NVIDIA GPU and VRAM availability via PyTorch and nvidia-smi."""
    gpu_info = {
        "has_gpu": False,
        "cuda_in_torch": bool(torch and torch.cuda.is_available()),
        "name": None,
        "total_vram_gb": 0.0,
        "free_vram_gb": 0.0,
        "safe_vram_gb": 0.0,
    }

    try:
        res = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total,memory.free", "--format=csv,noheader,nounits"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=3,
        )
        if res.returncode == 0 and res.stdout.strip():
            parts = [p.strip() for p in res.stdout.strip().split(",")]
            if len(parts) >= 3:
                gpu_info["has_gpu"] = True
                gpu_info["name"] = parts[0]
                total_mb = float(parts[1])
                free_mb = float(parts[2])
                gpu_info["total_vram_gb"] = round(total_mb / 1024, 2)
                gpu_info["free_vram_gb"] = round(free_mb / 1024, 2)
                safe_vram = max(0.5, (free_mb - 1200) / 1024)
                gpu_info["safe_vram_gb"] = round(safe_vram, 2)
    except Exception:
        pass

    if gpu_info["cuda_in_torch"] and torch is not None:
        gpu_info["has_gpu"] = True
        gpu_info["name"] = torch.cuda.get_device_name(0)

    return gpu_info


def get_vulkan_info() -> Dict[str, any]:
    """Detects Vulkan driver availability and categorizes integrated (iGPU) and discrete devices."""
    vk_info = {
        "vulkan_available": False,
        "integrated_gpu": None,
        "discrete_gpu": None,
        "devices": [],
    }

    # Test loading Vulkan library directly
    try:
        if os.name == "nt":
            import ctypes
            ctypes.windll.LoadLibrary("vulkan-1.dll")
            vk_info["vulkan_available"] = True
        else:
            import ctypes.util
            if ctypes.util.find_library("vulkan"):
                vk_info["vulkan_available"] = True
    except Exception:
        pass

    # Inspect device topologies via vulkaninfo
    try:
        res = subprocess.run(
            ["vulkaninfo", "--summary"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=4,
        )
        if res.returncode == 0 and res.stdout:
            vk_info["vulkan_available"] = True
            curr = {}
            for line in res.stdout.splitlines():
                line = line.strip()
                if line.startswith("GPU") and ":" in line:
                    if curr:
                        vk_info["devices"].append(curr)
                    curr = {"id": line.split(":")[0].strip()}
                elif "=" in line and curr:
                    k, v = line.split("=", 1)
                    curr[k.strip()] = v.strip()
            if curr:
                vk_info["devices"].append(curr)

            for d in vk_info["devices"]:
                dev_type = d.get("deviceType", "")
                dev_name = d.get("deviceName", "")
                if "INTEGRATED" in dev_type:
                    vk_info["integrated_gpu"] = dev_name
                elif "DISCRETE" in dev_type:
                    vk_info["discrete_gpu"] = dev_name
    except Exception:
        pass

    # Windows fallback via Win32_VideoController if integrated GPU was not caught
    if os.name == "nt" and not vk_info["integrated_gpu"]:
        try:
            res = subprocess.run(
                ["powershell", "-NoProfile", "-Command", "Get-CimInstance Win32_VideoController | Select-Object -Expand Name"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=3,
            )
            if res.returncode == 0 and res.stdout:
                for name in res.stdout.splitlines():
                    name = name.strip()
                    if any(w in name.lower() for w in ["intel", "iris", "uhd", "radeon(tm)"]) and not any(w in name.lower() for w in ["rtx", "gtx", "geforce", "quadro"]):
                        vk_info["integrated_gpu"] = name
                        vk_info["vulkan_available"] = True
        except Exception:
            pass

    return vk_info


def detect_hardware_capabilities() -> HardwareProfile:
    """Returns a unified HardwareProfile snapshot with 3-tier adaptive hardware categorization."""
    cpu = get_cpu_info()
    ram = get_ram_info()
    gpu = get_gpu_info()
    vulkan = get_vulkan_info()

    has_discrete_gpu = gpu["has_gpu"] and gpu["safe_vram_gb"] >= 1.0
    has_integrated_gpu = bool(vulkan["integrated_gpu"])
    vulkan_available = vulkan["vulkan_available"]

    # 3-Tier Adaptive Backend Selection:
    # Tier 1: Dedicated GPU (CUDA / ROCm) with high-bandwidth discrete VRAM
    # Tier 2: Integrated GPU (Intel UHD / Iris Xe / AMD Radeon) via Vulkan Unified Memory
    # Tier 3: Pure CPU Host execution with thread-capped OpenMP
    if has_discrete_gpu:
        adaptive_backend = "cuda"
        adaptive_reason = f"Dedicated GPU detected: {gpu['name']} ({gpu['total_vram_gb']} GB VRAM). Using CUDA Tensor Cores + Dedicated VRAM tier."
        recommended_device = "cuda" if gpu["cuda_in_torch"] else "cpu"
    elif has_integrated_gpu and vulkan_available:
        adaptive_backend = "vulkan"
        adaptive_reason = f"Integrated GPU detected: {vulkan['integrated_gpu']} (GPU 0). Adapting to Unified Shared Memory (UMA) via Vulkan shaders."
        recommended_device = "cpu"
    else:
        adaptive_backend = "cpu"
        adaptive_reason = "No GPU acceleration detected. Running on Optimized CPU with AVX2/OpenMP."
        recommended_device = "cpu"

    return HardwareProfile(
        physical_cores=cpu["physical_cores"],
        logical_threads=cpu["logical_threads"],
        safe_threads=cpu["safe_threads"],
        total_ram_gb=ram["total_gb"],
        available_ram_gb=ram["available_gb"],
        safe_ram_gb=ram["safe_budget_gb"],
        has_gpu=gpu["has_gpu"] or has_integrated_gpu,
        cuda_available=gpu["cuda_in_torch"],
        hardware_cuda_capable=gpu["has_gpu"],
        gpu_name=gpu["name"] or vulkan["integrated_gpu"],
        total_vram_gb=gpu["total_vram_gb"],
        free_vram_gb=gpu["free_vram_gb"],
        safe_vram_gb=gpu["safe_vram_gb"],
        recommended_device=recommended_device,
        has_discrete_gpu=has_discrete_gpu,
        discrete_gpu_name=gpu["name"],
        has_integrated_gpu=has_integrated_gpu,
        integrated_gpu_name=vulkan["integrated_gpu"],
        vulkan_available=vulkan_available,
        adaptive_backend=adaptive_backend,
        adaptive_reason=adaptive_reason,
    )


def configure_safe_execution_environment() -> HardwareProfile:
    """Configures thread pools and environment variables for safe balanced execution."""
    profile = detect_hardware_capabilities()

    if torch is not None:
        torch.set_num_threads(profile.safe_threads)
    os.environ["OMP_NUM_THREADS"] = str(profile.safe_threads)
    os.environ["MKL_NUM_THREADS"] = str(profile.safe_threads)

    return profile
