"""Along-track altimetry granules as stored for the data challenges (``DC2/ZARR/<Mission>/*.zarr``).

One zarr store per pass segment, the period in the file name:

* CNES/AVISO GDR naming (SARAL, Jason-3): ``SRL_GPN_2PfP188_0004_20241209_232245_20241210_001302.CNES.zarr``
* SWOT L3 naming: ``SWOT_L3_LR_SSH_Basic_009_..._20240111T232120_20240112T001246_....zarr``

The name is the one piece of metadata that has always been right: the ``time`` arrays of some
granules were written as float32 nanoseconds since a wrong epoch (see DC2
``scripts/fix_altimetry_time_corruption.py``), so a reader filters granules by name first and only
then opens the ones it needs.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Any

import fsspec
import pandas as pd

log = logging.getLogger(__name__)

# `_YYYYMMDD_HHMMSS_YYYYMMDD_HHMMSS` (GDR) and `_YYYYMMDDTHHMMSS_YYYYMMDDTHHMMSS` (SWOT)
_GDR = re.compile(r"_(\d{8})_(\d{6})_(\d{8})_(\d{6})")
_ISO = re.compile(r"_(\d{8})T(\d{6})_(\d{8})T(\d{6})")


@dataclass(frozen=True)
class Granule:
    path: str
    start: pd.Timestamp
    end: pd.Timestamp


def parse_granule_period(name: str) -> tuple[pd.Timestamp, pd.Timestamp] | None:
    """Start and end of a granule from its file name, or None when the name carries no period."""
    for rx in (_GDR, _ISO):
        m = rx.search(name)
        if m:
            d0, t0, d1, t1 = m.groups()
            return pd.Timestamp(f"{d0}T{t0}"), pd.Timestamp(f"{d1}T{t1}")
    return None


def list_granules(url: str, fs: fsspec.AbstractFileSystem, first: str | pd.Timestamp | None = None,
                  last: str | pd.Timestamp | None = None, suffix: str = ".zarr",
                  recursive: bool = False) -> list[Granule]:
    """Granules under ``url`` overlapping ``[first, last]`` (``last`` inclusive to the end of that day).

    Listing 20 000 keys is one paginated ``ListObjectsV2``, a few seconds; opening them is not, hence
    the name-based filter. ``recursive=True`` for stores nested in cycle sub-folders (SWOT).
    """
    t0 = pd.Timestamp(first) if first is not None else None
    t1 = pd.Timestamp(last) + pd.Timedelta(days=1) if last is not None else None
    pattern = f"{url.rstrip('/')}/{'**/' if recursive else ''}*{suffix}"
    out: list[Granule] = []
    skipped = 0
    for path in sorted(fs.glob(pattern)):
        period = parse_granule_period(path.rsplit("/", 1)[-1])
        if period is None:
            skipped += 1
            continue
        start, end = period
        if (t0 is not None and end < t0) or (t1 is not None and start >= t1):
            continue
        out.append(Granule(path, start, end))
    if skipped:
        log.warning("%d entries under %s have no period in their name and were ignored", skipped, url)
    return out


def granules_from_catalog(catalog: dict[str, Any] | str, fs: fsspec.AbstractFileSystem | None = None,
                          first: str | None = None, last: str | None = None) -> list[Granule]:
    """Same, from a DC GeoJSON catalog (``catalogs/<dataset>.json`` on S3, or an already-loaded dict):
    features carry ``properties.path/date_start/date_end``."""
    if isinstance(catalog, str):
        import json
        opener = fs.open if fs is not None else fsspec.open
        with opener(catalog, "r") as f:
            catalog = json.load(f)
    t0 = pd.Timestamp(first) if first else None
    t1 = pd.Timestamp(last) + pd.Timedelta(days=1) if last else None
    out = []
    for feat in catalog.get("features", []):
        p = feat.get("properties", {})
        if not p.get("date_start") or not p.get("date_end"):
            continue
        start, end = pd.Timestamp(p["date_start"]), pd.Timestamp(p["date_end"])
        if (t0 is not None and end < t0) or (t1 is not None and start >= t1):
            continue
        out.append(Granule(p["path"], start, end))
    return sorted(out, key=lambda g: g.start)
