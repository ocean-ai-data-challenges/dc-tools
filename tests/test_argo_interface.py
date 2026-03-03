#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

"""Test ArgoInterface with the new configuration system."""

import os
import pytest
import pandas as pd
import numpy as np
import ujson
import zstandard as zstd

# Set environment variables before importing
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"
os.environ["NETCDF4_DEACTIVATE_MPI"] = "1"
os.environ["NETCDF4_USE_FILE_LOCKING"] = "FALSE"


def test_argo_interface_from_config():
    """Test creating ArgoInterface from ARGOConnectionConfig."""
    from dctools.data.connection.config import ARGOConnectionConfig
    from dctools.data.connection.argo_data import ArgoInterface

    # Configuration locale pour les tests
    params = {
        "init_type": "from_scratch",
        "local_root": "/tmp/argo_test",
        "base_path": "/tmp/argo_test/argo_index",
        "variables": ["TEMP", "PSAL"],
        "depth_values": [0, 10, 20, 50, 100],
        "chunks": {"N_PROF": 1000},
        "file_cache": None,
        "dataset_processor": None,
        "filter_values": {
            "start_time": "2024-01-01",
            "end_time": "2024-02-01",
            "min_lon": -180,
            "max_lon": 180,
            "min_lat": -90,
            "max_lat": 90,
        },
    }

    config = ARGOConnectionConfig(params)
    argo_interface = ArgoInterface.from_config(config)

    assert argo_interface.base_path == "/tmp/argo_test/argo_index"
    assert argo_interface.variables == ["TEMP", "PSAL"]
    assert argo_interface.chunks == {"N_PROF": 1000}


def test_argo_interface_with_s3_config():
    """Test creating ArgoInterface with S3/Wasabi configuration."""
    from dctools.data.connection.config import ARGOConnectionConfig
    from dctools.data.connection.argo_data import ArgoInterface

    # Configuration S3/Wasabi
    params = {
        "init_type": "from_scratch",
        "local_root": "/tmp/argo_test",
        "s3_bucket": "test-bucket",
        "s3_folder": "argo_data",
        "endpoint_url": "https://s3.eu-west-2.wasabisys.com",
        "s3_key": "test_key",
        "s3_secret_key": "test_secret",
        "variables": ["TEMP", "PSAL"],
        "depth_values": [0, 10, 20, 50, 100],
        "file_cache": None,
        "dataset_processor": None,
        "filter_values": {
            "start_time": "2024-01-01",
            "end_time": "2024-02-01",
            "min_lon": -180,
            "max_lon": 180,
            "min_lat": -90,
            "max_lat": 90,
        },
    }

    config = ARGOConnectionConfig(params)
    argo_interface = ArgoInterface.from_config(config)

    # Vérifier que base_path est construit correctement
    assert argo_interface.base_path == "s3://test-bucket/argo_data/argo_index"

    # Vérifier que storage_options sont corrects
    assert "client_kwargs" in argo_interface.s3_storage_options
    assert (
        argo_interface.s3_storage_options["client_kwargs"]["endpoint_url"]
        == "https://s3.eu-west-2.wasabisys.com"
    )
    assert argo_interface.s3_storage_options["key"] == "test_key"
    assert argo_interface.s3_storage_options["secret"] == "test_secret"


def test_argo_interface_retry_settings_from_config():
    """Retry settings should be propagated from config into ArgoInterface."""
    from dctools.data.connection.config import ARGOConnectionConfig
    from dctools.data.connection.argo_data import ArgoInterface

    params = {
        "init_type": "from_scratch",
        "local_root": "/tmp/argo_test",
        "base_path": "/tmp/argo_test/argo_index",
        "variables": ["TEMP"],
        "max_fetch_retries": 7,
        "retry_backoff_seconds": 0.25,
        "file_cache": None,
        "dataset_processor": None,
        "filter_values": {"start_time": "2024-01-01", "end_time": "2024-02-01"},
    }

    config = ARGOConnectionConfig(params)
    argo_interface = ArgoInterface.from_config(config)

    assert argo_interface.max_fetch_retries == 7
    assert argo_interface.retry_backoff_seconds == 0.25


def test_argo_interface_retry_settings_none_fallback_to_defaults():
    """None retry settings in config should safely fallback to defaults."""
    from dctools.data.connection.config import ARGOConnectionConfig
    from dctools.data.connection.argo_data import ArgoInterface

    params = {
        "init_type": "from_scratch",
        "local_root": "/tmp/argo_test",
        "base_path": "/tmp/argo_test/argo_index",
        "variables": ["TEMP"],
        "max_fetch_retries": None,
        "retry_backoff_seconds": None,
        "file_cache": None,
        "dataset_processor": None,
        "filter_values": {"start_time": "2024-01-01", "end_time": "2024-02-01"},
    }

    config = ARGOConnectionConfig(params)
    argo_interface = ArgoInterface.from_config(config)

    assert argo_interface.max_fetch_retries == 4
    assert argo_interface.retry_backoff_seconds == 0.8


def test_get_files_for_month_prefers_6_element_region(monkeypatch):
    """Use 6-element region box first for argopy compatibility."""
    from dctools.data.connection.argo_data import ArgoInterface
    import dctools.data.connection.argo_data as argo_mod

    calls = []

    class FakeFetcher:
        def __init__(self, src=None):
            self.src = src

        def region(self, box):
            calls.append(len(box))
            if len(box) != 6:
                raise ValueError("index box must be a list with 4 or 6 elements")
            return self

        def to_dataframe(self):
            return pd.DataFrame({"file": ["a.nc", "b.nc"]})

    monkeypatch.setattr(argo_mod, "IndexFetcher", FakeFetcher)

    argo_interface = ArgoInterface(base_path="/tmp/argo_index")
    files = argo_interface._get_files_for_month(2024, 1)

    assert calls == [6]
    assert set(files) == {"a.nc", "b.nc"}


def test_get_files_for_month_falls_back_to_8_element_region(monkeypatch):
    """Fallback to 8-element region box for older argopy signatures."""
    from dctools.data.connection.argo_data import ArgoInterface
    import dctools.data.connection.argo_data as argo_mod

    calls = []

    class FakeFetcher:
        def __init__(self, src=None):
            self.src = src

        def region(self, box):
            calls.append(len(box))
            if len(box) == 6:
                raise ValueError("simulated 6-element failure")
            if len(box) == 8:
                return self
            raise ValueError("unexpected region box")

        def to_dataframe(self):
            return pd.DataFrame({"file": ["c.nc"]})

    monkeypatch.setattr(argo_mod, "IndexFetcher", FakeFetcher)

    argo_interface = ArgoInterface(base_path="/tmp/argo_index")
    files = argo_interface._get_files_for_month(2024, 1)

    assert calls == [6, 8]
    assert files == ["c.nc"]


def test_update_master_index_keeps_existing_entries(tmp_path):
    """Updating master index must preserve previously indexed months."""
    from dctools.data.connection.argo_data import ArgoInterface

    argo_interface = ArgoInterface(base_path=str(tmp_path))

    argo_interface._update_master_index(
        2024,
        1,
        {
            "start": int(pd.Timestamp("2024-01-01").value),
            "end": int(pd.Timestamp("2024-01-31").value),
        },
    )
    argo_interface._update_master_index(
        2024,
        2,
        {
            "start": int(pd.Timestamp("2024-02-01").value),
            "end": int(pd.Timestamp("2024-02-29").value),
        },
    )

    master_path = tmp_path / "master_index.json"
    with open(master_path, "r") as f:
        content = ujson.loads(f.read())

    assert "2024_01" in content
    assert "2024_02" in content


def test_permanent_errors_not_retried(monkeypatch, tmp_path):
    """FileNotFoundError and HTTP 404 should never be retried."""
    from dctools.data.connection.argo_data import ArgoInterface
    import dctools.data.connection.argo_data as argo_mod

    call_count = {"init": 0}

    class FakeNetCDF3ToZarrFail:
        def __init__(self, url, **kwargs):
            call_count["init"] += 1
            raise FileNotFoundError(url)

    monkeypatch.setattr(argo_mod, "NetCDF3ToZarr", FakeNetCDF3ToZarrFail)

    argo = ArgoInterface(
        base_path=str(tmp_path),
        variables=["TEMP"],
        max_fetch_retries=4,
        retry_backoff_seconds=0.0,
    )

    # For a relative path, two GDAC candidates are tried, each fails once (no retry)
    out = argo._build_single_ref("jma/4902986/profiles/D4902986_162.nc", tmp_path)
    assert out is None
    # Each candidate should be tried exactly once (no retries for permanent errors)
    assert call_count["init"] == 2


def test_build_single_ref_tries_gdac_fallback_urls(monkeypatch, tmp_path):
    """Relative GDAC paths should fallback to known base URLs."""
    from dctools.data.connection.argo_data import ArgoInterface
    import dctools.data.connection.argo_data as argo_mod

    opened_urls = []

    class FakeNetCDF3ToZarr:
        def __init__(self, url, **kwargs):
            opened_urls.append(url)
            if not str(url).startswith("https://data-argo.ifremer.fr/dac/"):
                raise FileNotFoundError(url)

        def translate(self):
            return {"refs": {"TIME": ["x"], "PRES": ["y"], "TEMP": ["z"]}}

    monkeypatch.setattr(argo_mod, "NetCDF3ToZarr", FakeNetCDF3ToZarr)

    argo_interface = ArgoInterface(base_path=str(tmp_path), variables=["TEMP"])
    out = argo_interface._build_single_ref("jma/4902986/profiles/D4902986_162.nc", tmp_path)

    assert out is not None
    assert any(str(u).startswith("https://data-argo.ifremer.fr/dac/") for u in opened_urls)


def test_build_single_ref_returns_none_when_all_candidates_fail(monkeypatch, tmp_path):
    """Unresolvable profile paths should be skipped instead of raising."""
    from dctools.data.connection.argo_data import ArgoInterface
    import dctools.data.connection.argo_data as argo_mod

    class FakeNetCDF3ToZarrFail:
        def __init__(self, url, **kwargs):
            raise FileNotFoundError(url)

    monkeypatch.setattr(argo_mod, "NetCDF3ToZarr", FakeNetCDF3ToZarrFail)

    argo_interface = ArgoInterface(base_path=str(tmp_path), variables=["TEMP"])
    out = argo_interface._build_single_ref("jma/4902986/profiles/D4902986_162.nc", tmp_path)

    assert out is None


def test_open_profile_with_retries_eventual_success(monkeypatch, tmp_path):
    """Profile fetch should retry transient failures and eventually succeed."""
    from dctools.data.connection.argo_data import ArgoInterface
    import dctools.data.connection.argo_data as argo_mod

    call_count = {"init": 0, "sleep": 0}

    class FakeNetCDF3ToZarr:
        """Mock that fails twice (transient) then succeeds."""

        def __init__(self, url, **kwargs):
            call_count["init"] += 1
            if call_count["init"] <= 2:
                raise RuntimeError("421 There are too many connections from your internet address.")

        def translate(self):
            return {"refs": {"TIME": ["x"], "PRES": ["y"], "TEMP": ["z"]}}

    monkeypatch.setattr(argo_mod, "NetCDF3ToZarr", FakeNetCDF3ToZarr)
    monkeypatch.setattr(
        argo_mod.time, "sleep", lambda _: call_count.__setitem__("sleep", call_count["sleep"] + 1)
    )

    argo_interface = ArgoInterface(
        base_path=str(tmp_path),
        variables=["TEMP"],
        max_fetch_retries=3,
        retry_backoff_seconds=0.0,
    )
    out = argo_interface._build_single_ref("jma/4902986/profiles/D4902986_162.nc", tmp_path)

    assert out is not None
    assert call_count["init"] == 3
    assert call_count["sleep"] == 2


def test_patch_ref_urls_replaces_local_with_remote():
    """_patch_ref_urls should replace local paths with GDAC URLs in byte-range refs."""
    from dctools.data.connection.argo_data import ArgoInterface

    ref = {
        "version": 1,
        "refs": {
            ".zattrs": "{}",
            ".zgroup": '{"zarr_format":2}',
            "TEMP/.zarray": '{"chunks":[1],"dtype":"<f4"}',
            "TEMP/0": ["/tmp/nc_cache/D123.nc", 100, 200],
            "PSAL/0": ["/tmp/nc_cache/D123.nc", 300, 150],
            "TIME/0": ["/tmp/nc_cache/D123.nc", 50, 30],
        },
    }

    ArgoInterface._patch_ref_urls(
        ref,
        "/tmp/nc_cache/D123.nc",
        "https://data-argo.ifremer.fr/dac/csiro/123/profiles/D123.nc",
    )

    for k in ("TEMP/0", "PSAL/0", "TIME/0"):
        assert ref["refs"][k][0] == "https://data-argo.ifremer.fr/dac/csiro/123/profiles/D123.nc"
        assert len(ref["refs"][k]) == 3

    # Inline metadata must NOT be touched
    assert ref["refs"][".zattrs"] == "{}"
    assert ref["refs"][".zgroup"] == '{"zarr_format":2}'
    assert ref["refs"]["TEMP/.zarray"] == '{"chunks":[1],"dtype":"<f4"}'


def test_get_gdac_url_for_profile_relative():
    """Relative argopy paths should be mapped to the first GDAC mirror."""
    from dctools.data.connection.argo_data import ArgoInterface

    argo = ArgoInterface(base_path="/tmp/idx")
    url = argo._get_gdac_url_for_profile("csiro/2901861/profiles/D2901861_228.nc")
    assert url == "https://data-argo.ifremer.fr/dac/csiro/2901861/profiles/D2901861_228.nc"


def test_get_gdac_url_for_profile_already_remote():
    """Already-remote URLs should pass through unchanged."""
    from dctools.data.connection.argo_data import ArgoInterface

    argo = ArgoInterface(base_path="/tmp/idx")
    url = argo._get_gdac_url_for_profile("https://example.com/foo.nc")
    assert url == "https://example.com/foo.nc"


def test_get_gdac_url_for_profile_strips_dac_prefix():
    """If argopy includes 'dac/' prefix, it must be stripped."""
    from dctools.data.connection.argo_data import ArgoInterface

    argo = ArgoInterface(base_path="/tmp/idx")
    url = argo._get_gdac_url_for_profile("dac/jma/123/profiles/R123.nc")
    assert url == "https://data-argo.ifremer.fr/dac/jma/123/profiles/R123.nc"


def test_build_single_ref_with_download_info_uses_local_file(monkeypatch, tmp_path):
    """When download_info is provided, should use local file and patch URLs."""
    from dctools.data.connection.argo_data import ArgoInterface
    import dctools.data.connection.argo_data as argo_mod

    opened_paths = []

    class FakeNetCDF3ToZarr:
        def __init__(self, path, **kwargs):
            opened_paths.append(path)

        def translate(self):
            return {
                "version": 1,
                "refs": {
                    ".zattrs": "{}",
                    ".zgroup": '{"zarr_format":2}',
                    "TEMP/0": [opened_paths[-1], 100, 200],
                    "TIME/0": [opened_paths[-1], 50, 30],
                },
            }

    monkeypatch.setattr(argo_mod, "NetCDF3ToZarr", FakeNetCDF3ToZarr)

    argo = ArgoInterface(base_path=str(tmp_path), variables=["TEMP"])

    local_nc = str(tmp_path / "D123.nc")
    gdac_url = "https://data-argo.ifremer.fr/dac/csiro/123/profiles/D123.nc"

    ref = argo._build_single_ref(
        "csiro/123/profiles/D123.nc",
        tmp_path,
        download_info=(local_nc, gdac_url),
    )

    assert ref is not None
    # Should have opened the local file, not GDAC
    assert opened_paths == [local_nc]
    # Byte-range refs should point to GDAC URL
    assert ref["refs"]["TEMP/0"][0] == gdac_url
    assert ref["refs"]["TIME/0"][0] == gdac_url


@pytest.mark.skip(reason="Requires actual ARGO data and network access")
def test_build_monthly_index():
    """
    Example showing how to build monthly ARGO indexes.

    This test is skipped by default as it requires:
    1. Network access to ARGO ERDDAP/GDAC servers
    2. S3 write permissions (or local storage)
    3. Significant computation time

    To run manually:
    ```python
    from dctools.data.connection.config import ARGOConnectionConfig
    from dctools.data.connection.argo_data import ArgoInterface

    # Load your configuration
    params = {
        'init_type': 'from_scratch',
        'local_root': '/path/to/local/storage',
        's3_bucket': 'your-bucket',
        's3_folder': 'ARGO',
        'endpoint_url': 'https://s3.eu-west-2.wasabisys.com',
        's3_key': 'your_key',
        's3_secret_key': 'your_secret',
        'variables': ['TEMP', 'PSAL', 'PRES'],
        'file_cache': None,
        'dataset_processor': None,
        'filter_values': {...}
    }

    config = ARGOConnectionConfig(params)
    argo_interface = ArgoInterface.from_config(config)

    # Build indexes for 2024
    argo_interface.build_multi_year_monthly(
        start_year=2024,
        end_year=2024,
        temp_dir='/tmp/argo_refs',
        n_workers=8
    )
    ```
    """
    pass


@pytest.mark.skip(reason="Requires pre-built ARGO index")
def test_open_time_window():
    """
    Example showing how to open a time window from ARGO data.

    This test is skipped by default as it requires a pre-built index.

    To run manually:
    ```python
    from dctools.data.connection.config import ARGOConnectionConfig
    from dctools.data.connection.argo_data import ArgoInterface

    # Load your configuration
    params = {...}  # Same as above

    config = ARGOConnectionConfig(params)
    argo_interface = ArgoInterface.from_config(config)

    # Open data for January 2024
    ds = argo_interface.open_time_window(
        start='2024-01-01',
        end='2024-01-31',
        depth_levels=[0, 10, 20, 50, 100, 200],
        variables=['TEMP', 'PSAL']
    )

    print(f"Dataset dimensions: {ds.dims}")
    print(f"Dataset variables: {list(ds.data_vars)}")
    ```
    """
    pass


def test_build_month_stores_profile_refs_not_combined(monkeypatch, tmp_path):
    """build_month must store individual profile_refs, not MultiZarrToZarr combined."""
    from dctools.data.connection.argo_data import ArgoInterface
    import dctools.data.connection.argo_data as argo_mod

    built_refs = []

    class FakeFetcher:
        def __init__(self, src=None):
            pass

        def region(self, box):
            return self

        def to_dataframe(self):
            return pd.DataFrame({"file": ["a.nc", "b.nc"]})

    # Each fake profile has N_PROF=1, TIME, and a DATA variable.
    def fake_netcdf3(path, **kw):
        class _H:
            def __init__(self):
                built_refs.append(path)

            def translate(self_inner):
                import base64

                # Inline TIME as 8-byte float64 (days since 1950-01-01)
                t_val = 27029.5 if "a.nc" in str(path) else 27030.0  # ~2024-01-01 / 02
                t_bytes = np.array([t_val], dtype="<f8").tobytes()
                d_bytes = np.array([1.0, 2.0, 3.0], dtype="<f4").tobytes()
                # fsspec ReferenceFileSystem expects "base64:" prefix
                return {
                    "version": 1,
                    "refs": {
                        ".zattrs": ujson.dumps({"Conventions": "Argo"}),
                        ".zgroup": '{"zarr_format":2}',
                        "TIME/.zarray": ujson.dumps(
                            {
                                "chunks": [1],
                                "compressor": None,
                                "dtype": "<f8",
                                "fill_value": None,
                                "filters": None,
                                "order": "C",
                                "shape": [1],
                                "zarr_format": 2,
                            }
                        ),
                        "TIME/.zattrs": ujson.dumps(
                            {
                                "_ARRAY_DIMENSIONS": ["N_PROF"],
                                "units": "days since 1950-01-01 00:00:00",
                                "calendar": "standard",
                            }
                        ),
                        "TIME/0": "base64:" + base64.b64encode(t_bytes).decode(),
                        "DATA/.zarray": ujson.dumps(
                            {
                                "chunks": [1, 3],
                                "compressor": None,
                                "dtype": "<f4",
                                "fill_value": "NaN",
                                "filters": None,
                                "order": "C",
                                "shape": [1, 3],
                                "zarr_format": 2,
                            }
                        ),
                        "DATA/.zattrs": ujson.dumps(
                            {
                                "_ARRAY_DIMENSIONS": ["N_PROF", "N_LEVELS"],
                            }
                        ),
                        "DATA/0.0": "base64:" + base64.b64encode(d_bytes).decode(),
                    },
                }

        return _H()

    monkeypatch.setattr(argo_mod, "IndexFetcher", FakeFetcher)
    monkeypatch.setattr(argo_mod, "NetCDF3ToZarr", fake_netcdf3)

    argo = ArgoInterface(base_path=str(tmp_path))
    argo.build_month(2024, 1, temp_dir=str(tmp_path / "refs"), n_workers=2)

    # Verify the monthly JSON was written
    monthly_path = tmp_path / "2024_01.json.zst"
    assert monthly_path.exists()

    with open(monthly_path, "rb") as f:
        dctx = zstd.ZstdDecompressor()
        monthly = ujson.loads(dctx.decompress(f.read()))

    # Must use new format: profile_refs (list), NOT kerchunk_refs (single dict)
    assert "profile_refs" in monthly
    assert "kerchunk_refs" not in monthly
    assert isinstance(monthly["profile_refs"], list)
    assert len(monthly["profile_refs"]) == 2

    # Temporal index must be sorted
    assert "temporal_index" in monthly
    epochs = monthly["temporal_index"]["sorted_times_epoch"]
    assert epochs == sorted(epochs)

    # Each profile ref must be a valid Kerchunk ref dict
    for ref in monthly["profile_refs"]:
        assert "refs" in ref
        assert "TIME/0" in ref["refs"]


if __name__ == "__main__":
    # Run basic tests
    test_argo_interface_from_config()
    test_argo_interface_with_s3_config()
    print("✓ All basic tests passed!")
