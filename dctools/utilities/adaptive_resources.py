"""Adaptive resource detection and Dask configuration scaling.

Detects the hardware capabilities of the current machine (CPU cores, total
and available RAM) and scales the per-dataset Dask parameters declared in
YAML configs so that they remain *proportional* to what the machine can
actually provide.

The YAML per-dataset values (``n_parallel_workers``, ``nthreads_per_worker``,
``memory_limit_per_worker``) are treated as **indicative** — calibrated for
a *reference machine*.  This module re-computes optimal values for the
*actual* machine while preserving the user-chosen parallelism profile
(LOW / AVERAGE / HIGH via ``worker_scale``).

Usage::

    from dctools.utilities.adaptive_resources import (
        MachineResources,
        adapt_dask_cfg,
    )

    machine = MachineResources.detect()
    adapted = adapt_dask_cfg(
        yaml_cfg={"n_workers": 8, "threads_per_worker": 2, "memory_limit": "4GB"},
        machine=machine,
        reference_machine=MachineResources(cpu_count=16, total_ram_gb=64.0),
        worker_scale=1.0,
        safety_margin=0.85,
    )
"""

from __future__ import annotations

import math
import os
import re
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple

from loguru import logger


# ---------------------------------------------------------------------------
# Memory parsing helpers
# ---------------------------------------------------------------------------
_MEM_UNITS: Dict[str, float] = {
    "B": 1,
    "K": 1024,
    "KB": 1024,
    "M": 1024**2,
    "MB": 1024**2,
    "G": 1024**3,
    "GB": 1024**3,
    "T": 1024**4,
    "TB": 1024**4,
}

_MEM_RE = re.compile(r"^\s*([0-9]+(?:\.[0-9]*)?)\s*([A-Za-z]*)\s*$")


def parse_memory_string(mem: str | int | float) -> float:
    """Parse a human-readable memory string into **gigabytes** (float).

    Accepts:
    - plain numbers (interpreted as bytes when <1024, GB otherwise)
    - strings like ``"4GB"``, ``"512MB"``, ``"2.5G"``
    - int/float pass-through (assumed GB if >=1, bytes otherwise)
    """
    if isinstance(mem, (int, float)):
        # Heuristic: very small numbers are likely already in GB.
        return float(mem) if mem >= 1 else mem / 1024**3

    m = _MEM_RE.match(str(mem))
    if not m:
        raise ValueError(f"Cannot parse memory string: {mem!r}")
    value = float(m.group(1))
    unit = m.group(2).upper() if m.group(2) else "GB"
    multiplier = _MEM_UNITS.get(unit)
    if multiplier is None:
        raise ValueError(f"Unknown memory unit: {m.group(2)!r} in {mem!r}")
    return value * multiplier / (1024**3)  # convert to GB


def format_memory_gb(gb: float) -> str:
    """Format a GB value into a human-readable string (e.g. ``"4GB"``)."""
    if gb >= 1.0:
        # Round to nearest 0.5 GB for clean display
        rounded = round(gb * 2) / 2
        if rounded == int(rounded):
            return f"{int(rounded)}GB"
        return f"{rounded:.1f}GB"
    mb = gb * 1024
    return f"{int(round(mb))}MB"


# ---------------------------------------------------------------------------
# Machine resource detection
# ---------------------------------------------------------------------------
@dataclass
class MachineResources:
    """Snapshot of the current machine's hardware resources.

    All memory values are in **gigabytes**.
    """

    cpu_count: int = 1
    total_ram_gb: float = 4.0
    available_ram_gb: float = 4.0

    # Optional: fraction of resources to consider available for Dask.
    # Accounts for OS overhead, other processes, etc.
    usable_fraction: float = 0.85

    @classmethod
    def detect(cls, *, usable_fraction: float = 0.85) -> MachineResources:
        """Auto-detect CPU and RAM resources of the current machine.

        Parameters
        ----------
        usable_fraction : float
            Fraction of total resources considered usable by Dask.
            Default 0.85 (reserve 15% for OS + driver process).
        """
        cpu_count = _detect_cpu_count()
        total_ram_gb = _detect_total_ram_gb()
        available_ram_gb = _detect_available_ram_gb()

        res = cls(
            cpu_count=cpu_count,
            total_ram_gb=total_ram_gb,
            available_ram_gb=available_ram_gb,
            usable_fraction=usable_fraction,
        )
        logger.info(
            f"Detected machine resources: {res.cpu_count} CPUs, "
            f"{res.total_ram_gb:.1f} GB total RAM, "
            f"{res.available_ram_gb:.1f} GB available RAM "
            f"(usable fraction={res.usable_fraction:.0%})"
        )
        return res

    @property
    def usable_cpus(self) -> int:
        """Number of CPUs considered usable (after applying usable_fraction)."""
        return max(1, int(self.cpu_count * self.usable_fraction))

    @property
    def usable_ram_gb(self) -> float:
        """Usable RAM in GB (the lesser of total*fraction and available)."""
        return min(
            self.total_ram_gb * self.usable_fraction,
            self.available_ram_gb,
        )


def _detect_cpu_count() -> int:
    """Detect the number of available CPU cores."""
    # Respect cgroup limits (Docker, Kubernetes, SLURM)
    try:
        cgroup_quota = _read_cgroup_cpu_quota()
        if cgroup_quota is not None:
            return max(1, cgroup_quota)
    except Exception:
        pass

    # Environment override (SLURM, PBS, etc.)
    for env_var in ("SLURM_CPUS_ON_NODE", "PBS_NUM_PPN", "NSLOTS"):
        val = os.environ.get(env_var)
        if val:
            try:
                return max(1, int(val))
            except ValueError:
                pass

    # os.cpu_count() — logical cores (hyperthreading counted)
    count = os.cpu_count()
    return max(1, count or 1)


def _read_cgroup_cpu_quota() -> Optional[int]:
    """Read CPU quota from cgroup v1 or v2."""
    # cgroup v2
    try:
        with open("/sys/fs/cgroup/cpu.max") as f:
            parts = f.read().strip().split()
            if parts[0] != "max":
                quota = int(parts[0])
                period = int(parts[1])
                return max(1, quota // period)
    except (FileNotFoundError, OSError):
        pass

    # cgroup v1
    try:
        with open("/sys/fs/cgroup/cpu/cpu.cfs_quota_us") as fq, \
             open("/sys/fs/cgroup/cpu/cpu.cfs_period_us") as fp:
            quota = int(fq.read().strip())
            period = int(fp.read().strip())
            if quota > 0:
                return max(1, quota // period)
    except (FileNotFoundError, OSError):
        pass

    return None


def _detect_total_ram_gb() -> float:
    """Detect total system RAM in GB."""
    # Respect cgroup memory limit
    try:
        cgroup_mem = _read_cgroup_memory_limit_gb()
        if cgroup_mem is not None:
            return cgroup_mem
    except Exception:
        pass

    # Environment override (SLURM)
    mem_env = os.environ.get("SLURM_MEM_PER_NODE")
    if mem_env:
        try:
            # SLURM_MEM_PER_NODE is in MB
            return max(1.0, float(mem_env) / 1024)
        except ValueError:
            pass

    # /proc/meminfo (Linux)
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemTotal:"):
                    kb = int(line.split()[1])
                    return kb / (1024 * 1024)
    except (FileNotFoundError, OSError):
        pass

    # psutil fallback
    try:
        import psutil
        return psutil.virtual_memory().total / (1024**3)
    except ImportError:
        pass

    logger.warning("Could not detect total RAM; defaulting to 8 GB")
    return 8.0


def _read_cgroup_memory_limit_gb() -> Optional[float]:
    """Read memory limit from cgroup v1 or v2."""
    # cgroup v2
    try:
        with open("/sys/fs/cgroup/memory.max") as f:
            val = f.read().strip()
            if val != "max":
                return int(val) / (1024**3)
    except (FileNotFoundError, OSError):
        pass

    # cgroup v1
    try:
        with open("/sys/fs/cgroup/memory/memory.limit_in_bytes") as f:
            val = int(f.read().strip())
            # A very large value means "no limit"
            if val < 2**62:
                return val / (1024**3)
    except (FileNotFoundError, OSError):
        pass

    return None


def _detect_available_ram_gb() -> float:
    """Detect currently available RAM in GB."""
    # /proc/meminfo
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    kb = int(line.split()[1])
                    return kb / (1024 * 1024)
    except (FileNotFoundError, OSError):
        pass

    # psutil fallback
    try:
        import psutil
        return psutil.virtual_memory().available / (1024**3)
    except ImportError:
        pass

    # Conservative fallback: assume 50% of detected total
    return _detect_total_ram_gb() * 0.5


# ---------------------------------------------------------------------------
# Reference machine specification
# ---------------------------------------------------------------------------
#: Default reference machine: the machine the YAML per-dataset numbers
#: were originally calibrated on.  If the YAML does not include a
#: ``reference_machine:`` section, these defaults are assumed.
#: Values reflect *usable* resources (after OS overhead) on the calibration
#: machine: 22 logical CPUs, ~18 GB RAM available for Dask.
DEFAULT_REFERENCE_MACHINE = MachineResources(
    cpu_count=22,
    total_ram_gb=18.0,
    available_ram_gb=18.0,
    usable_fraction=1.0,  # reference values are "pure" (already usable)
)


def reference_machine_from_dict(d: Dict[str, Any]) -> MachineResources:
    """Build a :class:`MachineResources` from the ``reference_machine:`` YAML
    section.

    Expected keys: ``cpu_count``, ``total_ram_gb``.

    Example YAML::

        reference_machine:
          cpu_count: 16
          total_ram_gb: 64
    """
    cpu = int(d.get("cpu_count", DEFAULT_REFERENCE_MACHINE.cpu_count))
    ram = float(d.get("total_ram_gb", DEFAULT_REFERENCE_MACHINE.total_ram_gb))
    return MachineResources(
        cpu_count=cpu,
        total_ram_gb=ram,
        available_ram_gb=ram,  # reference is theoretical: total == available
        usable_fraction=1.0,
    )


# ---------------------------------------------------------------------------
# Adaptive Dask configuration scaling
# ---------------------------------------------------------------------------
def adapt_dask_cfg(
    yaml_cfg: Dict[str, Any],
    machine: MachineResources,
    reference_machine: MachineResources | None = None,
    worker_scale: float = 1.0,
    safety_margin: float = 0.85,
) -> Dict[str, Any]:
    """Scale a per-dataset Dask config from *reference_machine* to *machine*.

    **Design principles**:

    - ``memory_limit`` is the dataset's intrinsic memory need and is **never
      reduced**.  Dask uses it as a *spill threshold*, not a physical RAM
      reservation — workers that share physical RAM simply spill to disk
      when pressure rises.  Reducing this value was the root cause of OOM
      kills (worker dies *before* it can spill because the limit is below
      the dataset's decompression footprint).
    - ``n_workers`` is scaled proportionally to the CPU ratio between this
      machine and the reference machine, modulated by ``worker_scale``.
    - Only when a machine has **dramatically** less total RAM than the
      reference (ratio < 0.5) do we additionally reduce workers as a
      last-resort safety net.

    Parameters
    ----------
    yaml_cfg : dict
        Per-dataset Dask config from the YAML (the "indicative" values).
    machine : MachineResources
        Detected resources of the current machine.
    reference_machine : MachineResources, optional
        The machine the YAML values were calibrated for.
        Defaults to :data:`DEFAULT_REFERENCE_MACHINE`.
    worker_scale : float
        Parallelism profile factor (LOW=0.5, AVERAGE=1.0, HIGH=1.5).
    safety_margin : float
        Unused — kept for API compatibility.  Safety is now handled solely
        by the ``usable_fraction`` in :class:`MachineResources`.

    Returns
    -------
    dict
        Adapted config with ``n_workers``, ``threads_per_worker``,
        ``memory_limit`` (as human-readable string).
    """
    ref = reference_machine or DEFAULT_REFERENCE_MACHINE

    # ---- Extract YAML indicative values ----
    yaml_workers = int(yaml_cfg.get("n_workers", 4))
    yaml_threads = int(yaml_cfg.get("threads_per_worker", 1))
    yaml_mem_str = yaml_cfg.get("memory_limit", "4GB")
    yaml_mem_gb = parse_memory_string(yaml_mem_str)

    # ---- Compute scaling ratio (raw values, no usable_fraction) ----
    # On the calibration machine itself this is exactly 1.0.
    cpu_ratio = machine.cpu_count / max(1, ref.cpu_count)

    # ---- Scale workers by CPU capacity × profile factor ----
    ideal_workers = yaml_workers * cpu_ratio * worker_scale
    n_workers = max(1, round(ideal_workers))

    # ---- Scale threads (conservative: capped at 4 due to GIL) ----
    ideal_threads = yaml_threads * worker_scale
    threads_per_worker = max(1, min(4, round(ideal_threads)))

    # ---- Cap total CPU slots (allow mild ~1.5× oversubscription) ----
    total_cpu_slots = n_workers * threads_per_worker
    max_slots = max(2, int(machine.cpu_count * 1.5))
    if total_cpu_slots > max_slots:
        if threads_per_worker > 1:
            threads_per_worker = max(1, machine.cpu_count // n_workers)
        total_cpu_slots = n_workers * threads_per_worker
        if total_cpu_slots > max_slots:
            n_workers = max(1, max_slots // threads_per_worker)

    # ---- Last-resort RAM guard ----
    # Only kick in when the machine has dramatically less total RAM than
    # the reference (ratio < 0.5).  In that case, reduce workers so the
    # total *theoretical* memory footprint doesn't exceed total RAM × 2.
    # (Factor 2 because memory_limit is a spill threshold, not a hard
    # reservation: Dask workers spill to disk before hitting the limit.)
    total_ram = machine.total_ram_gb
    ram_ratio = total_ram / max(1.0, ref.total_ram_gb)
    if ram_ratio < 0.5:
        max_workers_by_ram = max(1, int(total_ram * 2 / yaml_mem_gb))
        n_workers = min(n_workers, max_workers_by_ram)

    # ---- Memory limit: always use the YAML value ----
    # This is the dataset's intrinsic decompression/processing need.
    # Never reduce it — a SWOT file needs ~5 GB regardless of machine size.
    mem_per_worker = yaml_mem_gb

    result = {
        "n_workers": n_workers,
        "threads_per_worker": threads_per_worker,
        "memory_limit": format_memory_gb(mem_per_worker),
    }

    logger.debug(
        f"Adaptive Dask config: YAML indicative=({yaml_workers}w, {yaml_threads}t, "
        f"{yaml_mem_str}) → adapted=({n_workers}w, {threads_per_worker}t, "
        f"{result['memory_limit']}) | machine={machine.cpu_count}cpu/"
        f"{machine.total_ram_gb:.1f}GB, ref={ref.cpu_count}cpu/"
        f"{ref.total_ram_gb:.0f}GB, scale={worker_scale}"
    )

    return result


def adapt_parallelism_profile(
    pcfg_dict: Dict[str, Any],
    machine: MachineResources,
    reference_machine: MachineResources | None = None,
) -> Dict[str, Any]:
    """Scale parallelism profile parameters to the current machine.

    Parameters that scale with CPU count:
    - ``prep_workers`` (0=auto stays 0)
    - ``prefetch_workers``
    - ``prefetch_obs_workers``
    - ``obs_batch_size``
    - ``gridded_batch_size``

    Parameters that scale with RAM:
    - ``worker_cache_size``
    - ``blosc_threads``

    Parameters that don't scale (ratios/thresholds):
    - ``memory_target``, ``memory_spill``, ``memory_pause``, ``memory_terminate``
    - ``max_memory_increase``, ``max_memory_fraction``
    - ``reduce_precision``, ``restart_workers_per_batch``
    - ``worker_scale``

    Returns a new dict with adapted values.
    """
    ref = reference_machine or DEFAULT_REFERENCE_MACHINE
    result = dict(pcfg_dict)

    # Use raw cpu_count for ratios to avoid asymmetric usable_fraction.
    cpu_ratio = machine.cpu_count / max(1, ref.cpu_count)
    ram_ratio = machine.usable_ram_gb / max(1.0, ref.usable_ram_gb)

    # -- CPU-proportional parameters --
    for key in ("prefetch_workers", "prefetch_obs_workers"):
        if key in result:
            result[key] = max(1, round(int(result[key]) * cpu_ratio))

    # prep_workers: 0 means auto, keep it
    if result.get("prep_workers", 0) > 0:
        result["prep_workers"] = max(1, round(int(result["prep_workers"]) * cpu_ratio))

    # Batch sizes: scale with CPU but be conservative (floor division)
    for key in ("obs_batch_size", "gridded_batch_size"):
        if key in result and result[key] is not None:
            result[key] = max(1, round(int(result[key]) * cpu_ratio))

    # -- RAM-proportional parameters --
    if "worker_cache_size" in result:
        result["worker_cache_size"] = max(1, round(int(result["worker_cache_size"]) * ram_ratio))

    # blosc_threads: scale with CPU but cap at 4
    if "blosc_threads" in result:
        result["blosc_threads"] = max(1, min(4, round(int(result["blosc_threads"]) * cpu_ratio)))

    # -- Timeouts: on weaker machines, increase timeouts proportionally --
    # A slower machine needs relatively more time per task.
    if cpu_ratio < 1.0:
        timeout_scale = 1.0 / max(0.2, cpu_ratio)
        for key in ("compute_timeout", "obs_compute_timeout", "stall_timeout"):
            if key in result:
                result[key] = round(int(result[key]) * timeout_scale)

    logger.debug(
        f"Adaptive parallelism profile: cpu_ratio={cpu_ratio:.2f}, "
        f"ram_ratio={ram_ratio:.2f}"
    )

    return result
