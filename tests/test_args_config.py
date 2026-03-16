"""Unit tests for args/config parsing and logging helpers."""

from argparse import Namespace

from dctools.utilities import args_config as ac
from dctools.utilities.parallelism import ParallelismConfig


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
    cfg.write_text("obs_batch_size: 7\nlog_level: INFO\n", encoding="utf-8")

    base_args = Namespace(data_directory="/tmp/data", logfile=None)
    merged = ac.load_args_and_config(str(cfg), args=base_args)

    assert merged is not None
    assert merged.obs_batch_size == 7
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


# ── ParallelismConfig tests ──────────────────────────────────────────────

class TestParallelismConfig:
    """Tests for the centralised ParallelismConfig dataclass."""

    def test_defaults(self):
        cfg = ParallelismConfig()
        assert cfg.obs_batch_size == 30
        assert cfg.stall_timeout == 1200
        assert cfg.memory_target == 0.60
        assert cfg.reduce_precision is True  # R4: enabled by default for 50% RAM saving

    def test_from_dict_basic(self):
        cfg = ParallelismConfig.from_dict({"obs_batch_size": 50, "stall_timeout": 600})
        assert cfg.obs_batch_size == 50
        assert cfg.stall_timeout == 600

    def test_from_dict_compat_renames(self):
        """Old YAML keys (max_p_memory_increase) should be accepted."""
        cfg = ParallelismConfig.from_dict({
            "max_p_memory_increase": 0.35,
            "max_worker_memory_fraction": 0.75,
        })
        assert cfg.max_memory_increase == 0.35
        assert cfg.max_memory_fraction == 0.75

    def test_from_dict_env_override(self, monkeypatch):
        monkeypatch.setenv("DCTOOLS_EVAL_STALL_TIMEOUT", "999")
        cfg = ParallelismConfig.from_dict({"stall_timeout": 100})
        assert cfg.stall_timeout == 999  # env overrides YAML

    def test_from_dict_no_env_override(self, monkeypatch):
        monkeypatch.setenv("DCTOOLS_EVAL_STALL_TIMEOUT", "999")
        cfg = ParallelismConfig.from_dict({"stall_timeout": 100}, env_override=False)
        assert cfg.stall_timeout == 100  # YAML wins

    def test_from_args_new_format(self):
        """When args.parallelism is a dict, it is used."""
        args = Namespace(parallelism={"obs_batch_size": 64, "prep_workers": 8})
        cfg = ParallelismConfig.from_args(args)
        assert cfg.obs_batch_size == 64
        assert cfg.prep_workers == 8

    def test_from_args_legacy_fallback(self):
        """Scattered top-level keys are collected as fallback."""
        args = Namespace(
            obs_batch_size=50,
            max_p_memory_increase=0.40,
            reduce_precision=False,
        )
        cfg = ParallelismConfig.from_args(args)
        assert cfg.obs_batch_size == 50
        assert cfg.max_memory_increase == 0.40
        assert cfg.reduce_precision is False

    def test_from_args_already_built(self):
        """If args.parallelism is already a ParallelismConfig, return it."""
        original = ParallelismConfig(obs_batch_size=99)
        args = Namespace(parallelism=original)
        cfg = ParallelismConfig.from_args(args)
        assert cfg is original

    def test_worker_env_vars(self):
        cfg = ParallelismConfig(
            compute_timeout=120,
            blosc_threads=4,
            obs_viewer_threads=6,
        )
        env = cfg.worker_env_vars()
        assert env["DCTOOLS_S3_COMPUTE_TIMEOUT"] == "120"
        assert env["DCTOOLS_OBS_VIEWER_THREADS"] == "6"
        assert env["BLOSC_NTHREADS"] == "4"

    def test_dask_memory_config(self):
        cfg = ParallelismConfig(memory_target=0.50, memory_terminate=0.99)
        mc = cfg.dask_memory_config()
        assert mc["distributed.worker.memory.target"] == 0.50
        assert mc["distributed.worker.memory.terminate"] == 0.99

    def test_load_args_builds_parallelism(self, tmp_path):
        """load_args_and_config should build args.parallelism from YAML."""
        cfg = tmp_path / "conf.yaml"
        cfg.write_text(
            "parallelism:\n"
            "  obs_batch_size: 64\n"
            "  stall_timeout: 120\n",
            encoding="utf-8",
        )
        base = Namespace(data_directory="/tmp", logfile=None)
        merged = ac.load_args_and_config(str(cfg), args=base)
        assert merged is not None
        assert isinstance(merged.parallelism, ParallelismConfig)
        assert merged.parallelism.obs_batch_size == 64
        assert merged.parallelism.stall_timeout == 120
