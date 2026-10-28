"""The forecast horizon comes from n_days_forecast; a challenge config is found from its name."""

from argparse import Namespace

import pytest

from dctools.data.coordinates import get_target_time_values
from dctools.submission import ModelSubmission, SubmissionValidator  # noqa: F401  (importable)
from dctools.submission.validator import CONFIG_DIR_ENV, resolve_dc_config


def test_auto_lead_times_follow_the_horizon():
    assert get_target_time_values({"n_days_forecast": 15, "target_time_values": "auto"}) == list(range(15))
    assert get_target_time_values(Namespace(n_days_forecast=3, target_time_values="auto")) == [0, 1, 2]


def test_explicit_lead_times_must_match_the_horizon():
    assert get_target_time_values({"n_days_forecast": 2, "target_time_values": [0, 1]}) == [0, 1]
    with pytest.raises(ValueError, match="n_days_forecast is 15"):
        get_target_time_values({"n_days_forecast": 15, "target_time_values": list(range(10))})


def test_no_lead_times_stays_none():
    # DC3 sets n_days_forecast only: its behaviour is unchanged
    assert get_target_time_values({"n_days_forecast": 181}) is None


def test_config_found_in_the_challenge_repository(tmp_path, monkeypatch):
    (tmp_path / "dc2" / "config").mkdir(parents=True)
    cfg = tmp_path / "dc2" / "config" / "dc2_edito.yaml"
    cfg.write_text("n_days_forecast: 10\n")
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv(CONFIG_DIR_ENV, raising=False)
    assert resolve_dc_config("dc2_edito") == cfg.resolve()
    assert resolve_dc_config(str(cfg)) == cfg.resolve()
    with pytest.raises(FileNotFoundError, match="dc2_edito"):
        resolve_dc_config("dc2")


def test_config_dir_from_the_environment(tmp_path, monkeypatch):
    cfg = tmp_path / "elsewhere.yaml"
    cfg.write_text("n_days_forecast: 10\n")
    monkeypatch.setenv(CONFIG_DIR_ENV, str(tmp_path))
    monkeypatch.chdir(tmp_path.parent)
    assert resolve_dc_config("elsewhere") == cfg.resolve()
