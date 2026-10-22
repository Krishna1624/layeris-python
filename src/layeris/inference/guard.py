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
    gpu_name: Optional[str]
    total_vram_gb: float
    free_vram_gb: float
    safe_vram_gb: float
    recommended_device: str


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


def detect_hardware_capabilities() -> HardwareProfile:
    """Returns a unified HardwareProfile snapshot."""
    cpu = get_cpu_info()
    ram = get_ram_info()
    gpu = get_gpu_info()

    if gpu["cuda_in_torch"] and gpu["safe_vram_gb"] >= 1.5:
        device = "cuda"
    else:
        device = "cpu"

    return HardwareProfile(
        physical_cores=cpu["physical_cores"],
        logical_threads=cpu["logical_threads"],
        safe_threads=cpu["safe_threads"],
        total_ram_gb=ram["total_gb"],
        available_ram_gb=ram["available_gb"],
        safe_ram_gb=ram["safe_budget_gb"],
        has_gpu=gpu["has_gpu"],
        cuda_available=gpu["cuda_in_torch"],
        gpu_name=gpu["name"],
        total_vram_gb=gpu["total_vram_gb"],
        free_vram_gb=gpu["free_vram_gb"],
        safe_vram_gb=gpu["safe_vram_gb"],
        recommended_device=device,
    )


def configure_safe_execution_environment() -> HardwareProfile:
    """Configures thread pools and environment variables for safe balanced execution."""
    profile = detect_hardware_capabilities()

    if torch is not None:
        torch.set_num_threads(profile.safe_threads)
    os.environ["OMP_NUM_THREADS"] = str(profile.safe_threads)
    os.environ["MKL_NUM_THREADS"] = str(profile.safe_threads)

    return profile
