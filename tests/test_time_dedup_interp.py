"""Tests for time deduplication and interpolation."""

import numpy as np
import xarray as xr


def test_time_dedup_then_interp_does_not_raise_invalidindexerror():
    """Test that deduplication followed by interpolation works.

    Verifies that datasets with duplicate timestamps can be processed.
    """
    # Create a dataset with duplicated time stamps
    t0 = np.datetime64("2024-01-03T00:00:00")
    times = np.array([t0, t0, t0 + np.timedelta64(12, "h")], dtype="datetime64[ns]")

    ds = xr.Dataset(
        {
            "ssha": ("time", np.array([1.0, 3.0, 5.0], dtype=np.float32)),
        },
        coords={"time": times},
    )

    # The intended behaviour for dc-tools: average duplicates then interpolate.
    # This must not raise pandas.errors.InvalidIndexError.
    ds2 = ds.groupby("time").mean(skipna=True).sortby("time")

    vt = np.datetime64("2024-01-03T06:00:00")
    out = ds2.interp(time=[vt], method="linear", assume_sorted=True)

    assert "time" in out.dims
    assert out.sizes["time"] == 1
    # Mean at t0 is (1+3)/2 = 2, linear to 5 at +12h => at +6h should be 3.5
    assert np.isclose(out["ssha"].values.item(), 3.5)
