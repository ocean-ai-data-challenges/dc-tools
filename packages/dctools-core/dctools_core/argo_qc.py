"""Argo core profiles in the "standard" mode: adjusted values, good flags, depth, potential temperature.

The rules of ``oceanml3d-eval`` (``oceanml3d_eval/argo_gdac.py``, ``reference/profiles.py``), shared so
that DC2 and oceanml3d-eval score against the same Argo measurements:

- where ``DATA_MODE`` is ``A`` or ``D`` the ``*_ADJUSTED`` values and flags replace the raw ones;
- a profile is dropped when its position or date flag is not 1 or 2; a level is dropped when its
  pressure flag is not; a bad temperature (salinity) flag blanks that level's temperature (salinity)
  only;
- depth is computed from pressure and latitude (TEOS-10 ``gsw.z_from_p``; Saunders 1981 without gsw):
  a pressure in dbar is not a depth in metres (about 1 % apart, 20 m at 2000 dbar);
- temperature is converted from in-situ to potential (``gsw.pt0_from_t``, practical salinity 35 where
  salinity is missing: below 0.01 degC over 0-2000 dbar), the kind of a model's ``thetao``.

Inputs are the arrays of a GDAC profile file (``N_PROF x N_LEVELS``); QC flags may be bytes, strings
or numbers.
"""
from __future__ import annotations

from typing import Dict, Optional

import numpy as np

GOOD_FLAGS = (1, 2)
ADJUSTED_MODES = ("A", "D")
TEMPERATURES = ("in_situ", "potential")


def decode_qc(arr) -> np.ndarray:
    """QC flags ('0'..'9', blank = missing) to int8; anything but a digit becomes 9."""
    a = np.asarray(arr)
    if a.dtype.kind in "iuf":
        return np.where(np.isfinite(a.astype(float)), a, 9).astype(np.int8)
    if a.dtype.kind == "U":
        a = np.char.encode(a, "ascii", "ignore")
    elif a.dtype.kind == "O":
        a = np.array([v if isinstance(v, bytes) else str(v).encode("ascii", "ignore") for v in a.ravel()],
                     dtype="S").reshape(a.shape)
    if a.dtype.itemsize != 1:
        a = np.char.strip(a).astype("S1")          # '' stays '' -> byte 0 -> 9 below
    code = a.view(np.uint8).astype(np.int16) - ord("0")
    return np.where((code >= 0) & (code <= 9), code, 9).astype(np.int8)


def qc_levels(arr, n_prof: int, n_lev: int) -> np.ndarray:
    """A per-level QC variable as ``(n_prof, n_lev)`` single characters, whatever the reader made of
    it: ``char(N_PROF, N_LEVELS)`` may come back as ``(N_PROF,)`` strings of length ``N_LEVELS``."""
    a = np.asarray(arr)
    if a.shape == (n_prof, n_lev):
        return a
    if a.dtype.kind == "S" and a.shape == (n_prof,):
        raw = np.frombuffer(a.astype(f"S{max(a.dtype.itemsize, 1)}").tobytes(),
                            dtype=np.uint8).reshape(n_prof, -1)
        out = np.full((n_prof, n_lev), ord(" "), np.uint8)
        w = min(n_lev, raw.shape[1])
        out[:, :w] = raw[:, :w]
        out[out == 0] = ord(" ")
        return out.view("S1")
    return a.reshape(n_prof, -1)[:, :n_lev]


def text(values) -> np.ndarray:
    """Character arrays (bytes, str or objects) as stripped str."""
    a = np.asarray(values)
    if a.dtype.kind == "O":
        a = np.array([v if isinstance(v, bytes) else str(v).encode("ascii", "ignore") for v in a.ravel()],
                     dtype="S")
    if a.dtype.kind == "S":
        return np.char.strip(np.char.decode(a, "ascii", "ignore"))
    return np.char.strip(a.astype(str))


def data_mode(arr, n_prof: int) -> np.ndarray:
    """``DATA_MODE`` as one character per profile ('R' when absent); ``char(N_PROF)`` may come back as
    one string of N_PROF characters."""
    if arr is None:
        return np.full(n_prof, "R")
    a = np.asarray(arr)
    if a.dtype.kind == "S" and a.size == 1 and n_prof > 1:
        a = np.frombuffer(a.tobytes(), dtype="S1")[:n_prof]
    mode = text(a).reshape(-1)
    if mode.size == 1 and n_prof > 1 and len(str(mode[0])) == n_prof:
        mode = np.array(list(str(mode[0])))
    return mode.reshape(n_prof) if mode.size == n_prof else np.full(n_prof, "R")


def depth_from_pressure(pres, lat) -> np.ndarray:
    """Depth (m, positive down) of a pressure (dbar) at a latitude: ``-gsw.z_from_p``; without gsw,
    Saunders (1981), within 0.1 m of it over 0-2000 dbar."""
    p = np.asarray(pres, dtype=float)
    la = np.broadcast_to(np.asarray(lat, dtype=float), p.shape)
    try:
        import gsw
    except ImportError:
        c1 = (5.92 + 5.25 * np.sin(np.deg2rad(np.abs(la))) ** 2) * 1e-3
        return (1 - c1) * p - 2.21e-6 * p**2
    return -np.asarray(gsw.z_from_p(p, la), dtype=float)


def potential_temperature(temp, psal, pres, lat, lon) -> np.ndarray:
    """Potential temperature referenced to the surface (TEOS-10 ``gsw.pt0_from_t``)."""
    try:
        import gsw
    except ImportError as exc:
        raise ImportError("potential temperature from Argo needs gsw (pip install gsw), or "
                          "temperature='in_situ'") from exc
    t = np.asarray(temp, dtype=float)
    sp = np.where(np.isfinite(psal), psal, 35.0)
    p = np.asarray(pres, dtype=float)
    la = np.broadcast_to(np.asarray(lat, dtype=float), p.shape)
    lo = np.broadcast_to(np.asarray(lon, dtype=float), p.shape)
    sa = gsw.SA_from_SP(sp, p, lo, la)
    return np.asarray(gsw.pt0_from_t(sa, t, p), dtype=float)


def standard_mode(arrays: Dict[str, np.ndarray], temperature: str = "potential",
                  max_pres: Optional[float] = None) -> Dict[str, np.ndarray]:
    """``PRES, TEMP, PSAL`` of a profile file in the standard mode, plus ``DEPTH``.

    ``arrays`` holds the file's variables by name (``PRES``, ``TEMP``, ``PSAL``, their ``_ADJUSTED``
    and ``_QC`` / ``_ADJUSTED_QC``, ``POSITION_QC``, ``JULD_QC``, ``DATA_MODE``, ``LATITUDE``,
    ``LONGITUDE``); the absent ones count as missing (flags as bad, except a file with no flag at
    all for a variable, whose values are kept). Returns float arrays ``(n_prof, n_lev)`` with NaN
    where a value is dropped; ``TEMP`` is potential temperature unless ``temperature='in_situ'``.
    """
    if temperature not in TEMPERATURES:
        raise ValueError(f"temperature must be one of {TEMPERATURES}, not {temperature!r}")
    pres_raw = np.asarray(arrays["PRES"], dtype=float)
    if pres_raw.ndim == 1:
        pres_raw = pres_raw[None, :]
    n_prof, n_lev = pres_raw.shape
    adjusted = np.isin(data_mode(arrays.get("DATA_MODE"), n_prof), ADJUSTED_MODES)

    def values(name: str) -> np.ndarray:
        if name not in arrays:
            return np.full((n_prof, n_lev), np.nan)
        raw = np.asarray(arrays[name], dtype=float).reshape(n_prof, n_lev)
        adj = arrays.get(f"{name}_ADJUSTED")
        if adjusted.any() and adj is not None:
            raw = np.where(adjusted[:, None], np.asarray(adj, dtype=float).reshape(n_prof, n_lev), raw)
        return raw

    def good(name: str) -> np.ndarray:
        if f"{name}_QC" not in arrays and f"{name}_ADJUSTED_QC" not in arrays:
            return np.ones((n_prof, n_lev), bool)
        flags = (decode_qc(qc_levels(arrays[f"{name}_QC"], n_prof, n_lev))
                 if f"{name}_QC" in arrays else np.full((n_prof, n_lev), 9, np.int8))
        adj = arrays.get(f"{name}_ADJUSTED_QC")
        if adjusted.any() and adj is not None:
            flags = np.where(adjusted[:, None], decode_qc(qc_levels(adj, n_prof, n_lev)), flags)
        return np.isin(flags, GOOD_FLAGS)

    def profile_good(name: str) -> np.ndarray:
        if name not in arrays:
            return np.ones(n_prof, bool)
        a = np.asarray(arrays[name]).reshape(-1)
        if a.size != n_prof:                      # one string of n_prof characters
            a = qc_levels(a[:1], 1, n_prof)[0]
        return np.isin(decode_qc(a), GOOD_FLAGS)

    pres, temp, psal = values("PRES"), values("TEMP"), values("PSAL")
    level_ok = good("PRES") & np.isfinite(pres) & (pres >= 0)
    if max_pres is not None:
        level_ok &= pres <= max_pres
    level_ok &= (profile_good("POSITION_QC") & profile_good("JULD_QC"))[:, None]
    pres = np.where(level_ok, pres, np.nan)
    temp = np.where(level_ok & good("TEMP"), temp, np.nan)
    psal = np.where(level_ok & good("PSAL"), psal, np.nan)

    lat = np.asarray(arrays.get("LATITUDE", np.full(n_prof, np.nan)), dtype=float).reshape(n_prof, 1)
    lon = np.asarray(arrays.get("LONGITUDE", np.full(n_prof, np.nan)), dtype=float).reshape(n_prof, 1)
    lat_for_depth = np.where(np.isfinite(lat), lat, 0.0)
    depth = depth_from_pressure(pres, lat_for_depth)
    if temperature == "potential":
        ok = np.isfinite(temp) & np.isfinite(lat) & np.isfinite(lon)
        pt = np.full_like(temp, np.nan)
        if ok.any():
            b = np.broadcast_to
            pt[ok] = potential_temperature(temp[ok], psal[ok], pres[ok],
                                           b(lat, temp.shape)[ok], b(lon, temp.shape)[ok])
        temp = pt
    return {"PRES": pres, "TEMP": temp, "PSAL": psal, "DEPTH": depth}
