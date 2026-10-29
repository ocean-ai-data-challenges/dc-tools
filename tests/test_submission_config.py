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


def test_required_variables_are_the_stored_ones(tmp_path, monkeypatch):
    # sst/sss are derived by DC2 from thetao/so: a submission does not have to hold them
    cfg = tmp_path / "dc.yaml"
    cfg.write_text(
        "n_days_forecast: 2\n"
        "sources:\n"
        "  - dataset: model\n"
        "    observation_dataset: false\n"
        "    keep_variables: [so, thetao, zos]\n"
        "    eval_variables: [so, thetao, zos, sst, sss]\n"
    )
    monkeypatch.delenv(CONFIG_DIR_ENV, raising=False)
    v = SubmissionValidator.from_dc_config(str(cfg))
    assert v.required_variables == ["so", "thetao", "zos"]


def test_a_forecast_as_long_as_the_horizon_is_indexed():
    import pandas as pd

    from dctools.data.datasets.forecast import build_forecast_index_from_catalog
    # daily means stamped 2024-01-03 .. 2024-01-12: ten lead days
    cat = pd.DataFrame({"path": ["f"], "date_start": [pd.Timestamp("2024-01-03")],
                        "date_end": [pd.Timestamp("2024-01-12")]})
    idx = build_forecast_index_from_catalog(cat, "2024-01-01", "2024-01-31", n_days_forecast=10)
    assert list(idx["lead_time"]) == list(range(10))
    assert build_forecast_index_from_catalog(
        cat, "2024-01-01", "2024-01-31", n_days_forecast=11).empty
