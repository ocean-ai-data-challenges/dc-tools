"""Unit tests for args/config parsing and logging helpers."""

from argparse import Namespace

from dctools.utilities import args_config as ac


def test_normalize_loguru_level():
    """Loguru level normalization should handle common input types."""
    assert ac._normalize_loguru_level(None) is None
    assert ac._normalize_loguru_level(10) == 10
    assert ac._normalize_loguru_level(" info ") == "INFO"


def test_normalize_python_logging_level():
    """Python logging level normalization should map names to numeric levels."""
    assert ac._normalize_python_logging_level(None) is None
    assert ac._normalize_python_logging_level(15) == 15
    assert ac._normalize_python_logging_level("warning") == 30
    assert ac._normalize_python_logging_level("not_a_level") == 30


def test_parse_arguments_minimal_required():
    """CLI parser should accept required data directory argument."""
    args = ac.parse_arguments(["--data_directory", "/tmp/data"])
    assert args.data_directory == "/tmp/data"
    assert args.metric == "rmse"


def test_load_configs_reads_yaml(tmp_path):
    """YAML config file should be loaded into a dictionary."""
    cfg = tmp_path / "conf.yaml"
    cfg.write_text("foo: 1\nbar: baz\n", encoding="utf-8")

    loaded = ac.load_configs(str(cfg))
    assert loaded["foo"] == 1
    assert loaded["bar"] == "baz"


def test_load_args_and_config_merges_values(tmp_path):
    """Config values should be merged into parsed args namespace."""
    cfg = tmp_path / "conf.yaml"
    cfg.write_text("batch_size: 7\nlog_level: INFO\n", encoding="utf-8")

    base_args = Namespace(data_directory="/tmp/data", logfile=None)
    merged = ac.load_args_and_config(str(cfg), args=base_args)

    assert merged is not None
    assert merged.batch_size == 7
    assert hasattr(merged, "device")


def test_configure_logging_adds_console_and_file_sink(monkeypatch, tmp_path):
    """Logger configuration should register sinks for console and logfile."""
    calls = []

    def fake_remove(*_args, **_kwargs):
        calls.append(("remove", None, None))

    def fake_add(sink, **kwargs):
        calls.append(("add", sink, kwargs))
        return 1

    def fake_info(message):
        calls.append(("info", message, None))

    monkeypatch.setattr(ac.logger, "remove", fake_remove)
    monkeypatch.setattr(ac.logger, "add", fake_add)
    monkeypatch.setattr(ac.logger, "info", fake_info)

    args = Namespace(
        logfile=str(tmp_path / "run.log"),
        log_level="debug",
        logging={"console": True, "format": "{message}"},
    )

    ac.configure_logging_from_args(args)

    assert any(c[0] == "remove" for c in calls)
    add_calls = [c for c in calls if c[0] == "add"]
    assert len(add_calls) >= 2
