"""Argo standard mode on the arrays of a GDAC profile file (two profiles, three levels)."""

import numpy as np
import pytest

from dctools_core.argo_qc import decode_qc, depth_from_pressure, standard_mode


def _profile_file(**override):
    arrays = {
        "DATA_MODE": np.array([b"D", b"R"]),
        "LATITUDE": np.array([45.0, 10.0]), "LONGITUDE": np.array([-30.0, 60.0]),
        "POSITION_QC": np.array([b"1", b"1"]), "JULD_QC": np.array([b"1", b"1"]),
        "PRES": np.array([[5.0, 500.0, 2000.0], [5.0, 500.0, 2000.0]]),
        "PRES_QC": np.array([[b"1", b"1", b"1"], [b"1", b"4", b"1"]]),
        "TEMP": np.array([[20.0, 10.0, 3.0], [25.0, 12.0, 4.0]]),
        "TEMP_QC": np.array([[b"1", b"1", b"1"], [b"1", b"1", b"3"]]),
        "PSAL": np.array([[36.0, 35.0, 34.9], [36.5, 35.2, 34.8]]),
        "PSAL_QC": np.array([[b"1", b"1", b"1"], [b"4", b"1", b"1"]]),
        "PRES_ADJUSTED": np.array([[5.5, 500.5, 2000.5], [np.nan] * 3]),
        "TEMP_ADJUSTED": np.array([[20.1, 10.1, 3.1], [np.nan] * 3]),
        "PSAL_ADJUSTED": np.array([[36.1, 35.1, 34.95], [np.nan] * 3]),
        "PRES_ADJUSTED_QC": np.array([[b"1", b"1", b"1"], [b" "] * 3]),
        "TEMP_ADJUSTED_QC": np.array([[b"1", b"2", b"4"], [b" "] * 3]),
        "PSAL_ADJUSTED_QC": np.array([[b"1", b"1", b"1"], [b" "] * 3]),
    }
    arrays.update(override)
    return arrays


def test_flags_decode_whatever_their_type():
    assert decode_qc(np.array([b"1", b"4", b" "])).tolist() == [1, 4, 9]
    assert decode_qc(np.array(["2", "9"])).tolist() == [2, 9]
    assert decode_qc(np.array([1.0, np.nan])).tolist() == [1, 9]


def test_delayed_mode_profiles_take_their_adjusted_values_and_flags():
    out = standard_mode(_profile_file(), temperature="in_situ")
    np.testing.assert_allclose(out["PRES"][0], [5.5, 500.5, 2000.5])
    # adjusted temperature flag 4 at 2000 dbar blanks it; the raw flag (1) is not used
    np.testing.assert_allclose(out["TEMP"][0, :2], [20.1, 10.1])
    assert np.isnan(out["TEMP"][0, 2])
    np.testing.assert_allclose(out["PSAL"][0], [36.1, 35.1, 34.95])


def test_bad_flags_blank_their_variable_or_their_level():
    out = standard_mode(_profile_file(), temperature="in_situ")
    # real-time profile: bad pressure at 500 dbar removes the whole level
    assert np.isnan(out["PRES"][1, 1]) and np.isnan(out["TEMP"][1, 1]) and np.isnan(out["PSAL"][1, 1])
    # bad salinity at 5 dbar blanks the salinity only
    assert np.isnan(out["PSAL"][1, 0]) and out["TEMP"][1, 0] == 25.0
    # temperature flag 3 (probably bad) is not good
    assert np.isnan(out["TEMP"][1, 2]) and out["PSAL"][1, 2] == pytest.approx(34.8)


def test_a_bad_position_drops_the_profile():
    out = standard_mode(_profile_file(POSITION_QC=np.array([b"1", b"4"])), temperature="in_situ")
    assert np.isnan(out["TEMP"][1]).all() and np.isfinite(out["TEMP"][0, 0])


def test_data_mode_as_one_string():
    out = standard_mode(_profile_file(DATA_MODE=np.array(b"DR")), temperature="in_situ")
    np.testing.assert_allclose(out["PRES"][0, 0], 5.5)


def test_depth_is_not_pressure():
    d = depth_from_pressure(np.array([2000.0]), np.array([45.0]))
    assert d[0] == pytest.approx(1977.0, abs=3.0)        # ~23 m shallower than 2000
    assert depth_from_pressure(np.array([0.0]), np.array([0.0]))[0] == pytest.approx(0.0, abs=1e-6)


def test_potential_temperature_is_colder_at_depth():
    pytest.importorskip("gsw")
    insitu = standard_mode(_profile_file(), temperature="in_situ")
    pot = standard_mode(_profile_file(), temperature="potential")
    assert pot["TEMP"][0, 0] == pytest.approx(insitu["TEMP"][0, 0], abs=0.01)   # 5 dbar: the same
    assert insitu["TEMP"][0, 1] - pot["TEMP"][0, 1] > 0.05                      # 500 dbar: colder
