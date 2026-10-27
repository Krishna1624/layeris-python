"""
Unit tests for Layeris Dynamic RAM & VRAM Memory Balancer.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

src_path = Path(__file__).resolve().parent.parent / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

from layeris.inference.guard import HardwareProfile
from layeris.inference.memory_balancer import (
    BalancedPlacementPlan,
    MemoryBalancer,
    ModelGeometry,
    calculate_balanced_memory_plan,
    parse_parameter_count,
    resolve_model_geometry,
)


class TestMemoryBalancer(unittest.TestCase):
    def setUp(self):
        # Create a mock 16 GB RAM + 6 GB VRAM hardware profile (standard consumer AI laptop)
        self.laptop_profile = HardwareProfile(
            physical_cores=10,
            logical_threads=16,
            safe_threads=10,
            total_ram_gb=16.0,
            available_ram_gb=12.0,
            safe_ram_gb=10.0,
            has_gpu=True,
            cuda_available=True,
            hardware_cuda_capable=True,
            gpu_name="NVIDIA GeForce RTX 4050 Laptop GPU",
            total_vram_gb=6.0,
            free_vram_gb=5.7,
            safe_vram_gb=4.5,
            recommended_device="cuda",
        )

        # Create a CPU-only hardware profile (16 GB RAM, no GPU)
        self.cpu_only_profile = HardwareProfile(
            physical_cores=8,
            logical_threads=16,
            safe_threads=8,
            total_ram_gb=16.0,
            available_ram_gb=12.0,
            safe_ram_gb=10.0,
            has_gpu=False,
            cuda_available=False,
            hardware_cuda_capable=False,
            gpu_name=None,
            total_vram_gb=0.0,
            free_vram_gb=0.0,
            safe_vram_gb=0.0,
            recommended_device="cpu",
        )

        # Create an Integrated GPU profile (16 GB RAM, no dedicated GPU, Intel Iris Xe on GPU 0 with Vulkan)
        self.igpu_vulkan_profile = HardwareProfile(
            physical_cores=8,
            logical_threads=16,
            safe_threads=8,
            total_ram_gb=16.0,
            available_ram_gb=12.0,
            safe_ram_gb=10.0,
            has_gpu=False,
            cuda_available=False,
            hardware_cuda_capable=False,
            gpu_name=None,
            total_vram_gb=0.0,
            free_vram_gb=0.0,
            safe_vram_gb=0.0,
            recommended_device="cpu",
            has_discrete_gpu=False,
            discrete_gpu_name=None,
            has_integrated_gpu=True,
            integrated_gpu_name="Intel(R) Iris(R) Xe Graphics",
            vulkan_available=True,
            adaptive_backend="vulkan",
            adaptive_reason="Integrated GPU detected: Intel(R) Iris(R) Xe Graphics (GPU 0). Adapting to Unified Shared Memory (UMA) via Vulkan shaders.",
        )

    def test_parse_parameter_count(self):
        self.assertEqual(parse_parameter_count("0.5B"), int(0.5e9))
        self.assertEqual(parse_parameter_count("1.5B"), int(1.5e9))
        self.assertEqual(parse_parameter_count("7B"), int(7e9))
        self.assertEqual(parse_parameter_count("14B"), int(14e9))
        self.assertEqual(parse_parameter_count("35B"), int(35e9))
        self.assertEqual(parse_parameter_count("70B"), int(70e9))
        self.assertEqual(parse_parameter_count("500M"), int(500e6))

    def test_resolve_model_geometry(self):
        geom_05 = resolve_model_geometry("qwen2.5-0.5b")
        self.assertEqual(geom_05.num_layers, 24)
        self.assertFalse(geom_05.is_moe)

        geom_35 = resolve_model_geometry("qwen36-35b")
        self.assertEqual(geom_35.num_layers, 64)
        self.assertTrue(geom_35.is_moe)

    def test_small_model_fits_entirely_in_vram(self):
        balancer = MemoryBalancer(hardware=self.laptop_profile)
        plan = balancer.balance("qwen2.5-0.5b", precision="fp16")

        self.assertEqual(plan.device_mode, "gpu_exclusive")
        self.assertEqual(plan.gpu_layers, 24)
        self.assertEqual(plan.cpu_layers, 0)
        self.assertEqual(plan.disk_layers, 0)
        self.assertEqual(plan.host_ram_pressure_relieved_pct, 100.0)
        self.assertGreater(plan.estimated_speedup_vs_cpu, 5.0)

    def test_medium_model_balances_across_vram_and_ram(self):
        # 14B model in INT4 (~7.27 GB total) on 4.5 GB VRAM + 10 GB RAM
        balancer = MemoryBalancer(hardware=self.laptop_profile)
        plan = balancer.balance("qwen2.5-14b", precision="int4")

        self.assertEqual(plan.device_mode, "hybrid_balanced")
        self.assertGreater(plan.gpu_layers, 20)
        self.assertGreater(plan.cpu_layers, 10)
        self.assertEqual(plan.disk_layers, 0)
        # Verify safe headroom constraints
        self.assertLessEqual(plan.vram_allocated_gb, self.laptop_profile.safe_vram_gb)
        self.assertLessEqual(plan.ram_allocated_gb, self.laptop_profile.safe_ram_gb)
        self.assertGreater(plan.host_ram_pressure_relieved_pct, 40.0)

    def test_large_model_three_tier_streaming(self):
        # 70B model in INT4 (~35 GB total) on 4.5 GB VRAM + 10 GB RAM
        balancer = MemoryBalancer(hardware=self.laptop_profile)
        plan = balancer.balance("llama3-70b", precision="int4")

        self.assertEqual(plan.device_mode, "three_tier_streaming")
        self.assertGreater(plan.gpu_layers, 5)
        self.assertGreater(plan.cpu_layers, 10)
        self.assertGreater(plan.disk_layers, 10)
        self.assertGreater(plan.disk_offload_gb, 10.0)
        self.assertLessEqual(plan.vram_allocated_gb, self.laptop_profile.safe_vram_gb)
        self.assertLessEqual(plan.ram_allocated_gb, self.laptop_profile.safe_ram_gb)

    def test_cpu_only_fallback(self):
        balancer = MemoryBalancer(hardware=self.cpu_only_profile)
        plan = balancer.balance("qwen2.5-7b", precision="int4")

        self.assertEqual(plan.device_mode, "cpu_exclusive")
        self.assertEqual(plan.gpu_layers, 0)
        self.assertEqual(plan.vram_allocated_gb, 0.0)
        self.assertGreater(plan.cpu_layers, 0)
        self.assertEqual(plan.host_ram_pressure_relieved_pct, 0.0)

    def test_vulkan_integrated_gpu_adaptation(self):
        balancer = MemoryBalancer(hardware=self.igpu_vulkan_profile)
        plan = balancer.balance("qwen2.5-7b", precision="int4")

        self.assertEqual(plan.device_mode, "vulkan_uma_streaming")
        self.assertEqual(plan.gpu_layers, 0)
        self.assertEqual(plan.vram_allocated_gb, 0.0)
        self.assertGreater(plan.cpu_layers, 0)
        self.assertIn("COLI_VULKAN", plan.colibri_cli_args)
        self.assertIn("COLI_VK_DENSE", plan.colibri_cli_args)
        self.assertIn("Intel(R) Iris(R) Xe Graphics", plan.status_summary)
        self.assertGreater(plan.estimated_speedup_vs_cpu, 1.0)

    def test_custom_budget_overrides(self):
        balancer = MemoryBalancer(hardware=self.laptop_profile)
        # Override with a 24 GB VRAM GPU and 64 GB RAM workstation
        plan = balancer.balance(
            "llama3-70b",
            precision="int4",
            ram_budget_gb=50.0,
            vram_budget_gb=22.0,
        )
        self.assertEqual(plan.device_mode, "hybrid_balanced")
        self.assertEqual(plan.disk_layers, 0)
        self.assertEqual(plan.disk_offload_gb, 0.0)
        self.assertGreater(plan.gpu_layers, 40)


if __name__ == "__main__":
    unittest.main()
