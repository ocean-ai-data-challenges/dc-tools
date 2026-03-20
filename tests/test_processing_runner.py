"""Unit tests for the lightweight CLI runner orchestration helpers."""

from __future__ import annotations

import sys
import types
from argparse import Namespace
from pathlib import Path

import dctools.processing.runner as runner


class _PerfReportCtx:
    def __init__(self, filename: str, calls: list[str]):
        self.filename = filename
        self.calls = calls

    def __enter__(self):
        self.calls.append(f"enter:{self.filename}")
        return self

    def __exit__(self, exc_type, exc, tb):
        self.calls.append("exit")
        return False


def _install_runtime_imports(monkeypatch, load_args_and_config):
    """Install fake modules imported inside run_from_config/run_from_cli."""
    perf_calls: list[str] = []

    fake_distributed = types.ModuleType("dask.distributed")
    fake_distributed.performance_report = (
        lambda filename: _PerfReportCtx(filename=filename, calls=perf_calls)
    )
    monkeypatch.setitem(sys.modules, "dask.distributed", fake_distributed)

    fake_args_cfg = types.ModuleType("dctools.utilities.args_config")
    fake_args_cfg.load_args_and_config = load_args_and_config
    fake_args_cfg.parse_arguments = lambda: Namespace(
        data_directory="/tmp/default",
        config_name=None,
    )
    monkeypatch.setitem(sys.modules, "dctools.utilities.args_config", fake_args_cfg)

    return perf_calls


def test_ensure_project_on_syspath_inserts_once(monkeypatch):
    """Project root should be inserted at most once in sys.path."""
    fake_root = Path("/tmp/fake-root")
    monkeypatch.setattr(runner, "_project_root", lambda: fake_root)

    original = list(sys.path)
    try:
        if str(fake_root) in sys.path:
            sys.path.remove(str(fake_root))
        runner._ensure_project_on_syspath()
        runner._ensure_project_on_syspath()
        assert sys.path.count(str(fake_root)) == 1
    finally:
        sys.path[:] = original


def test_resolve_config_path_uses_cli_override(monkeypatch):
    """resolve_config_path should prefer cli args config_name when provided."""
    monkeypatch.setattr(runner, "_challenge_config_dir", lambda: Path("/tmp/dc2/config"))
    cli_args = Namespace(config_name="dc3")

    out = runner.resolve_config_path("dc2", cli_args)

    assert out == Path("/tmp/dc2/config/dc3.yaml")


def test_configure_hdf5_env_uses_setdefault(monkeypatch):
    """Environment setup must not overwrite an existing value."""
    monkeypatch.setenv("HDF5_USE_FILE_LOCKING", "CUSTOM")

    runner._configure_hdf5_netcdf_env()

    assert sys.modules is not None
    assert runner.os.environ["HDF5_USE_FILE_LOCKING"] == "CUSTOM"
    assert runner.os.environ["NETCDF4_DEACTIVATE_MPI"] == "1"


def test_run_from_config_success_and_close(tmp_path, monkeypatch):
    """Successful run should execute run_eval inside performance_report and call close()."""
    run_calls: list[str] = []
    close_calls: list[str] = []

    def fake_load(config_path: str, args=None):
        del config_path
        del args
        # Keep a file so run_from_config executes its file-removal branch.
        weights_file = tmp_path / "weights"
        weights_file.write_text("w", encoding="utf-8")
        return Namespace(data_directory=str(tmp_path), logfile=None, config_name=None)

    perf_calls = _install_runtime_imports(monkeypatch, load_args_and_config=fake_load)

    class _Eval:
        def __init__(self, args):
            del args

        def run_eval(self):
            run_calls.append("run_eval")

        def close(self):
            close_calls.append("close")

    rc = runner.run_from_config(Path("/tmp/dc2.yaml"), evaluation_cls=_Eval, cli_args=None)

    assert rc == 0
    assert run_calls == ["run_eval"]
    assert close_calls == ["close"]
    assert any(item.startswith("enter:") for item in perf_calls)
    assert "exit" in perf_calls
    assert (tmp_path / "catalogs").is_dir()
    assert (tmp_path / "results").is_dir()


def test_run_from_config_returns_1_when_config_loading_fails(monkeypatch):
    """If load_args_and_config returns None, run should fail early."""

    def fake_load(config_path: str, args=None):
        del config_path
        del args
        return None

    _install_runtime_imports(monkeypatch, load_args_and_config=fake_load)
    rc = runner.run_from_config(Path("/tmp/dc2.yaml"), evaluation_cls=object, cli_args=None)
    assert rc == 1


def test_run_from_config_uses_dataset_processor_close_when_no_close(tmp_path, monkeypatch):
    """Fallback cleanup path should close dataset_processor when close() is absent."""
    closed: list[str] = []

    def fake_load(config_path: str, args=None):
        del config_path
        del args
        return Namespace(data_directory=str(tmp_path), logfile=None, config_name=None)

    _install_runtime_imports(monkeypatch, load_args_and_config=fake_load)

    class _Proc:
        def close(self):
            closed.append("proc")

    class _Eval:
        def __init__(self, args):
            del args
            self.dataset_processor = _Proc()

        def run_eval(self):
            return None

    rc = runner.run_from_config(Path("/tmp/dc2.yaml"), evaluation_cls=_Eval, cli_args=None)
    assert rc == 0
    assert closed == ["proc"]


def test_run_from_config_handles_runtime_exception(tmp_path, monkeypatch):
    """Unexpected exceptions should return code 1."""

    def fake_load(config_path: str, args=None):
        del config_path
        del args
        return Namespace(data_directory=str(tmp_path), logfile=None, config_name=None)

    _install_runtime_imports(monkeypatch, load_args_and_config=fake_load)

    class _Eval:
        def __init__(self, args):
            del args

        def run_eval(self):
            raise RuntimeError("boom")

    rc = runner.run_from_config(Path("/tmp/dc2.yaml"), evaluation_cls=_Eval, cli_args=None)
    assert rc == 1


def test_run_from_cli_parses_and_delegates(monkeypatch):
    """run_from_cli should parse args, resolve path and delegate to run_from_config."""
    captured: dict[str, object] = {}

    fake_args_mod = types.ModuleType("dctools.utilities.args_config")
    fake_args_mod.parse_arguments = lambda: Namespace(data_directory="/tmp/x", config_name="dc9")
    monkeypatch.setitem(sys.modules, "dctools.utilities.args_config", fake_args_mod)

    monkeypatch.setattr(
        runner,
        "resolve_config_path",
        lambda default_config_name, cli_args: Path("/tmp/f.yaml"),
    )

    def fake_run_from_config(config_path, cli_args=None, evaluation_cls=None):
        captured["config_path"] = config_path
        captured["cli_args"] = cli_args
        captured["evaluation_cls"] = evaluation_cls
        return 7

    monkeypatch.setattr(runner, "run_from_config", fake_run_from_config)

    rc = runner.run_from_cli(default_config_name="dc2")
    assert rc == 7
    assert captured["config_path"] == Path("/tmp/f.yaml")
    assert isinstance(captured["cli_args"], Namespace)
    assert captured["evaluation_cls"] is None
