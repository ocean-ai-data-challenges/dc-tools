"""Tests for adaptive resource detection and Dask config scaling.

Covers:
- MachineResources detection and properties
- parse_memory_string / format_memory_gb helpers
- adapt_dask_cfg scaling logic
- adapt_parallelism_profile scaling logic
- ParallelismConfig integration (auto_adapt / adapt_profile / adapt_dask_cfg_for_dataset)
- BaseDCEvaluation._extract_dask_cfg_from_source with adaptive mode
"""

from __future__ import annotations

import os
from types import SimpleNamespace
from typing import Any, Dict
from unittest.mock import patch

import pytest

from dctools.utilities.adaptive_resources import (
    DEFAULT_REFERENCE_MACHINE,
    MachineResources,
    adapt_dask_cfg,
    adapt_parallelism_profile,
    format_memory_gb,
    parse_memory_string,
    reference_machine_from_dict,
)
from dctools.utilities.parallelism import ParallelismConfig


# =====================================================================
# parse_memory_string
# =====================================================================
class TestParseMemoryString:
    def test_gb_string(self):
        assert parse_memory_string("4GB") == pytest.approx(4.0)

    def test_mb_string(self):
        assert parse_memory_string("512MB") == pytest.approx(0.5)

    def test_numeric_gb(self):
        assert parse_memory_string(8) == pytest.approx(8.0)

    def test_numeric_float(self):
        assert parse_memory_string(2.5) == pytest.approx(2.5)

    def test_no_unit_defaults_to_gb(self):
        assert parse_memory_string("4") == pytest.approx(4.0)

    def test_tb_string(self):
        assert parse_memory_string("1TB") == pytest.approx(1024.0)

    def test_invalid_raises(self):
        with pytest.raises(ValueError):
            parse_memory_string("notamemory")


# =====================================================================
# format_memory_gb
# =====================================================================
class TestFormatMemoryGb:
    def test_whole_gb(self):
        assert format_memory_gb(4.0) == "4GB"

    def test_half_gb(self):
        assert format_memory_gb(2.5) == "2.5GB"

    def test_sub_gb(self):
        assert format_memory_gb(0.25) == "256MB"


# =====================================================================
# MachineResources
# =====================================================================
class TestMachineResources:
    def test_usable_cpus(self):
        m = MachineResources(cpu_count=16, total_ram_gb=64.0, available_ram_gb=56.0, usable_fraction=0.85)
        # 16 * 0.85 = 13.6 -> int = 13
        assert m.usable_cpus == 13

    def test_usable_ram_takes_min(self):
        m = MachineResources(cpu_count=8, total_ram_gb=32.0, available_ram_gb=10.0, usable_fraction=0.85)
        # min(32*0.85=27.2, 10.0) = 10.0
        assert m.usable_ram_gb == pytest.approx(10.0)

    def test_detect_returns_instance(self):
        m = MachineResources.detect()
        assert isinstance(m, MachineResources)
        assert m.cpu_count >= 1
        assert m.total_ram_gb > 0

    def test_usable_fraction_one(self):
        m = MachineResources(cpu_count=4, total_ram_gb=8.0, available_ram_gb=8.0, usable_fraction=1.0)
        assert m.usable_cpus == 4
        assert m.usable_ram_gb == pytest.approx(8.0)


# =====================================================================
# reference_machine_from_dict
# =====================================================================
class TestReferenceMachineFromDict:
    def test_basic(self):
        ref = reference_machine_from_dict({"cpu_count": 32, "total_ram_gb": 128})
        assert ref.cpu_count == 32
        assert ref.total_ram_gb == 128.0

    def test_defaults(self):
        ref = reference_machine_from_dict({})
        assert ref.cpu_count == DEFAULT_REFERENCE_MACHINE.cpu_count
        assert ref.total_ram_gb == DEFAULT_REFERENCE_MACHINE.total_ram_gb


# =====================================================================
# adapt_dask_cfg
# =====================================================================
class TestAdaptDaskCfg:
    """Test that per-dataset Dask config adapts to machine resources."""

    @pytest.fixture
    def reference(self):
        return MachineResources(cpu_count=16, total_ram_gb=64.0, available_ram_gb=64.0, usable_fraction=1.0)

    @pytest.fixture
    def yaml_cfg(self):
        return {"n_workers": 8, "threads_per_worker": 2, "memory_limit": "4GB"}

    def test_same_machine_average_profile(self, reference, yaml_cfg):
        """On an identical machine with worker_scale=1.0, output == input."""
        result = adapt_dask_cfg(
            yaml_cfg, machine=reference, reference_machine=reference, worker_scale=1.0
        )
        assert result["n_workers"] == 8
        assert result["threads_per_worker"] == 2
        # Memory is always kept at the YAML value (never reduced)
        mem = parse_memory_string(result["memory_limit"])
        assert mem == pytest.approx(4.0)

    def test_half_machine_scales_down(self, reference, yaml_cfg):
        """A machine with half the resources should produce roughly half the workers."""
        small = MachineResources(cpu_count=8, total_ram_gb=32.0, available_ram_gb=32.0, usable_fraction=1.0)
        result = adapt_dask_cfg(
            yaml_cfg, machine=small, reference_machine=reference, worker_scale=1.0
        )
        assert result["n_workers"] <= 8
        assert result["n_workers"] >= 1

    def test_double_machine_scales_up(self, reference, yaml_cfg):
        """A machine with double resources should produce more workers."""
        big = MachineResources(cpu_count=32, total_ram_gb=128.0, available_ram_gb=128.0, usable_fraction=1.0)
        result = adapt_dask_cfg(
            yaml_cfg, machine=big, reference_machine=reference, worker_scale=1.0
        )
        assert result["n_workers"] >= 8

    def test_low_profile_reduces(self, reference, yaml_cfg):
        """LOW profile (worker_scale=0.5) should produce fewer workers."""
        result_avg = adapt_dask_cfg(
            yaml_cfg, machine=reference, reference_machine=reference, worker_scale=1.0
        )
        result_low = adapt_dask_cfg(
            yaml_cfg, machine=reference, reference_machine=reference, worker_scale=0.5
        )
        assert result_low["n_workers"] <= result_avg["n_workers"]

    def test_high_profile_increases(self, reference, yaml_cfg):
        """HIGH profile (worker_scale=1.5) should produce more workers."""
        result_avg = adapt_dask_cfg(
            yaml_cfg, machine=reference, reference_machine=reference, worker_scale=1.0
        )
        result_high = adapt_dask_cfg(
            yaml_cfg, machine=reference, reference_machine=reference, worker_scale=1.5
        )
        assert result_high["n_workers"] >= result_avg["n_workers"]

    def test_minimum_one_worker(self):
        """Even a tiny machine should get at least 1 worker."""
        tiny = MachineResources(cpu_count=1, total_ram_gb=1.0, available_ram_gb=1.0, usable_fraction=1.0)
        ref = MachineResources(cpu_count=64, total_ram_gb=256.0, available_ram_gb=256.0, usable_fraction=1.0)
        result = adapt_dask_cfg(
            {"n_workers": 32, "threads_per_worker": 4, "memory_limit": "8GB"},
            machine=tiny, reference_machine=ref, worker_scale=0.5,
        )
        assert result["n_workers"] >= 1
        assert result["threads_per_worker"] >= 1

    def test_memory_never_reduced(self):
        """memory_limit is the dataset's intrinsic need and is never reduced."""
        machine = MachineResources(cpu_count=16, total_ram_gb=16.0, available_ram_gb=14.0, usable_fraction=0.85)
        result = adapt_dask_cfg(
            {"n_workers": 8, "threads_per_worker": 2, "memory_limit": "8GB"},
            machine=machine, worker_scale=1.0,
        )
        assert result["n_workers"] >= 1
        mem = parse_memory_string(result["memory_limit"])
        # Memory must always equal the YAML value
        assert mem == pytest.approx(8.0)

    def test_memory_never_reduced_prevents_oom(self):
        """Critical OOM prevention: memory_limit is NEVER reduced.

        The original OOM bug was caused by scaling memory_limit DOWN
        proportionally to the RAM ratio (5 GB → 1 GB), which killed
        workers before they could decompress SWOT data.  Now memory_limit
        is always preserved and workers are scaled by CPU ratio only.
        """
        ref = MachineResources(cpu_count=16, total_ram_gb=64.0, available_ram_gb=64.0, usable_fraction=1.0)
        machine = MachineResources(cpu_count=22, total_ram_gb=31.0, available_ram_gb=18.5, usable_fraction=0.85)
        result = adapt_dask_cfg(
            {"n_workers": 5, "threads_per_worker": 1, "memory_limit": "5GB"},
            machine=machine, reference_machine=ref, worker_scale=1.0,
        )
        mem = parse_memory_string(result["memory_limit"])
        # Memory is kept at the YAML value — must NEVER be reduced
        assert mem == pytest.approx(5.0), f"Memory {mem:.1f}GB was reduced, workers will OOM"
        # Workers scale by CPU ratio (22/16 = 1.375) → round(5*1.375) = 7
        assert result["n_workers"] >= 5

    def test_very_low_ram_triggers_guard(self):
        """When total RAM < 50% of reference, the last-resort guard reduces workers."""
        ref = MachineResources(cpu_count=16, total_ram_gb=64.0, available_ram_gb=64.0, usable_fraction=1.0)
        tiny_ram = MachineResources(cpu_count=16, total_ram_gb=8.0, available_ram_gb=8.0, usable_fraction=1.0)
        result = adapt_dask_cfg(
            {"n_workers": 8, "threads_per_worker": 2, "memory_limit": "4GB"},
            machine=tiny_ram, reference_machine=ref, worker_scale=1.0,
        )
        mem = parse_memory_string(result["memory_limit"])
        # Memory per worker must still be the full YAML value
        assert mem == pytest.approx(4.0)
        # ram_ratio = 8/64 = 0.125 < 0.5, guard: max_workers = 8*2/4 = 4
        assert result["n_workers"] <= 4
        assert result["n_workers"] >= 1

    def test_moderate_ram_deficit_no_reduction(self):
        """When total RAM >= 50% of reference, workers are NOT reduced for RAM."""
        ref = MachineResources(cpu_count=16, total_ram_gb=64.0, available_ram_gb=64.0, usable_fraction=1.0)
        half_ram = MachineResources(cpu_count=16, total_ram_gb=32.0, available_ram_gb=32.0, usable_fraction=1.0)
        result = adapt_dask_cfg(
            {"n_workers": 8, "threads_per_worker": 2, "memory_limit": "4GB"},
            machine=half_ram, reference_machine=ref, worker_scale=1.0,
        )
        mem = parse_memory_string(result["memory_limit"])
        assert mem == pytest.approx(4.0)
        # Same CPU count, ram_ratio=0.5 (NOT < 0.5), no guard → 8 workers
        assert result["n_workers"] == 8


# =====================================================================
# adapt_parallelism_profile
# =====================================================================
class TestAdaptParallelismProfile:
    def test_same_machine_unchanged(self):
        ref = MachineResources(cpu_count=16, total_ram_gb=64.0, available_ram_gb=64.0, usable_fraction=1.0)
        profile = {
            "obs_batch_size": 30,
            "prefetch_workers": 4,
            "prefetch_obs_workers": 8,
            "worker_cache_size": 4,
            "blosc_threads": 2,
            "compute_timeout": 90,
        }
        result = adapt_parallelism_profile(profile, machine=ref, reference_machine=ref)
        assert result["obs_batch_size"] == 30
        assert result["prefetch_workers"] == 4

    def test_small_machine_increases_timeouts(self):
        ref = MachineResources(cpu_count=16, total_ram_gb=64.0, available_ram_gb=64.0, usable_fraction=1.0)
        small = MachineResources(cpu_count=4, total_ram_gb=16.0, available_ram_gb=16.0, usable_fraction=1.0)
        profile = {"compute_timeout": 90, "obs_compute_timeout": 120, "stall_timeout": 300}
        result = adapt_parallelism_profile(profile, machine=small, reference_machine=ref)
        assert result["compute_timeout"] > 90
        assert result["obs_compute_timeout"] > 120

    def test_big_machine_does_not_change_timeouts(self):
        ref = MachineResources(cpu_count=16, total_ram_gb=64.0, available_ram_gb=64.0, usable_fraction=1.0)
        big = MachineResources(cpu_count=32, total_ram_gb=128.0, available_ram_gb=128.0, usable_fraction=1.0)
        profile = {"compute_timeout": 90}
        result = adapt_parallelism_profile(profile, machine=big, reference_machine=ref)
        # cpu_ratio >= 1.0, so timeouts should not increase
        assert result["compute_timeout"] == 90


# =====================================================================
# ParallelismConfig integration
# =====================================================================
class TestParallelismConfigAdaptive:
    def test_auto_adapt_default_true(self):
        pcfg = ParallelismConfig()
        assert pcfg.auto_adapt is True

    def test_auto_adapt_from_dict(self):
        pcfg = ParallelismConfig.from_dict({"auto_adapt": False}, env_override=False)
        assert pcfg.auto_adapt is False

    def test_env_override_auto_adapt(self, monkeypatch):
        monkeypatch.setenv("DCTOOLS_AUTO_ADAPT", "false")
        pcfg = ParallelismConfig.from_dict({"auto_adapt": True}, env_override=True)
        assert pcfg.auto_adapt is False

    def test_adapt_dask_cfg_legacy_when_disabled(self):
        pcfg = ParallelismConfig(auto_adapt=False, worker_scale=0.5)
        result = pcfg.adapt_dask_cfg_for_dataset(
            {"n_workers": 8, "threads_per_worker": 2, "memory_limit": "4GB"}
        )
        assert result["n_workers"] == 4  # 8 * 0.5 = 4
        assert result["threads_per_worker"] == 1  # 2 * 0.5 = 1
        assert result["memory_limit"] == "4GB"  # unchanged in legacy mode

    def test_adapt_dask_cfg_adaptive_when_enabled(self):
        pcfg = ParallelismConfig(auto_adapt=True, worker_scale=1.0)
        result = pcfg.adapt_dask_cfg_for_dataset(
            {"n_workers": 8, "threads_per_worker": 2, "memory_limit": "4GB"}
        )
        # Should return valid values
        assert result["n_workers"] >= 1
        assert result["threads_per_worker"] >= 1
        assert "memory_limit" in result

    def test_adapt_profile_returns_self_when_disabled(self):
        pcfg = ParallelismConfig(auto_adapt=False)
        adapted = pcfg.adapt_profile()
        assert adapted is pcfg

    def test_adapt_profile_returns_new_when_enabled(self):
        pcfg = ParallelismConfig(auto_adapt=True, obs_batch_size=30)
        adapted = pcfg.adapt_profile()
        assert isinstance(adapted, ParallelismConfig)
        # Batch size should be adjusted (may or may not change depending on machine)
        assert adapted.obs_batch_size >= 1

    def test_adapt_with_reference_machine_dict(self):
        pcfg = ParallelismConfig(auto_adapt=True, worker_scale=1.0)
        result = pcfg.adapt_dask_cfg_for_dataset(
            {"n_workers": 8, "threads_per_worker": 2, "memory_limit": "4GB"},
            reference_machine_dict={"cpu_count": 32, "total_ram_gb": 128},
        )
        assert result["n_workers"] >= 1
