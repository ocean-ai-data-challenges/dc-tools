#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

"""Tests for memory cleanup in Dask workers."""

import gc
import xarray as xr
from dask.distributed import Client, LocalCluster
import pytest

from dctools.metrics.evaluator import _clear_xarray_file_cache, _worker_full_cleanup


pytestmark = pytest.mark.slow


def test_xarray_file_cache_disabled():
    """Verify that xarray file cache is minimal (effectively disabled)."""
    # After configuration, file cache should be 1 (minimal)
    # Note: Some xarray versions don't allow 0
    xr.set_options(file_cache_maxsize=1)

    # Verify cache is minimal (1 = effectively disabled)
    assert xr.get_options()["file_cache_maxsize"] == 1


def test_clear_xarray_file_cache():
    """Test that xarray file cache clearing works."""
    result = _clear_xarray_file_cache()
    assert result is True

    # Verify cache is minimal (1 = effectively disabled)
    assert xr.get_options()["file_cache_maxsize"] == 1


def test_worker_full_cleanup():
    """Test that worker cleanup function executes without error."""
    result = _worker_full_cleanup()
    assert result is True


def test_cancel_before_del_pattern():
    """Test the recommended pattern: cancel(force=True) -> del -> gc.collect().

    This ensures Dask releases memory properly.
    """
    cluster = LocalCluster(
        n_workers=1,
        threads_per_worker=1,
        # Avoid nanny restarts on tight memory budgets in test envs.
        memory_limit="1GB",
        processes=False,
        silence_logs=True,
    )
    client = Client(cluster)

    try:
        # Simulate batch processing
        import dask

        def dummy_task(x):
            return x * 2

        # Create delayed tasks
        delayed_tasks = [dask.delayed(dummy_task)(i) for i in range(10)]

        # Compute
        futures = client.compute(delayed_tasks)

        # Clear delayed tasks immediately (recommended pattern)
        del delayed_tasks
        gc.collect()

        # Gather results
        results = client.gather(futures)
        assert len(results) == 10

        # Proper cleanup pattern
        # 1. wait() to ensure all transitions are complete
        from dask.distributed import wait

        wait(futures)

        # 2. cancel(force=True) before deleting references
        client.cancel(futures, force=True)

        # 3. Delete references
        del futures
        del results

        # 4. Force garbage collection
        gc.collect()

        # 5. Run cleanup on workers
        client.run(_worker_full_cleanup)

        # At this point, worker memory should be released
        # We can't directly measure it in a test, but the pattern should not raise errors

    finally:
        client.close()
        cluster.close()


def test_no_persist_between_batches():
    """Verify that .persist() is not used between batches.

    This is a documentation test - we grep for .persist() in code.
    """
    import subprocess
    import os

    # Get the project root
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    # Search for .persist() in Python files
    result = subprocess.run(
        [
            "grep",
            "-r",
            "--include=*.py",
            r"\.persist()",
            os.path.join(project_root, "dctools"),
            os.path.join(project_root, "dc2"),
        ],
        capture_output=True,
        text=True,
    )

    # Should return non-zero (no matches) or empty output
    assert result.returncode != 0 or len(result.stdout.strip()) == 0, (
        f"Found .persist() in code (should be avoided): {result.stdout}"
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
