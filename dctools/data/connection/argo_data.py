"""ARGO data interface: monthly partitioning, Kerchunk references, and S3/local access."""
import shutil
import tempfile
import xarray as xr
import fsspec
import pandas as pd
import numpy as np
import ujson
import zstandard as zstd
import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    from kerchunk.netCDF3 import NetCDF3ToZarr  # type: ignore[import-untyped]
except Exception:  # pragma: no cover
    NetCDF3ToZarr = None

try:
    from argopy import IndexFetcher
except Exception:  # pragma: no cover
    IndexFetcher = None
from loguru import logger

# Transient HTTP errors worth retrying (connection drops, rate limits, server errors)
_TRANSIENT_EXCEPTIONS = (ConnectionError, TimeoutError)
try:
    import urllib3

    _TRANSIENT_EXCEPTIONS = (*_TRANSIENT_EXCEPTIONS, urllib3.exceptions.HTTPError)  # type: ignore[assignment]
except ImportError:
    pass
try:
    import requests

    _TRANSIENT_EXCEPTIONS = (  # type: ignore[assignment]
        *_TRANSIENT_EXCEPTIONS,
        requests.exceptions.ConnectionError,
        requests.exceptions.Timeout,
    )
except ImportError:
    pass


# ---- Helper: per-profile 2D pressure -> standard-depth interpolation ------
def _interpolate_profiles_to_depth(
    ds: xr.Dataset,
    depth_levels: np.ndarray,
    pres_var: str,
) -> xr.Dataset:
    """Interpolate 2D ARGO profiles from pressure levels to standard depths.

    For each profile along N_PROF, the 1-D column of *pres_var* values is
    used as the source abscissa and ``depth_levels`` as the target.
    All data variables that share the vertical dimension (N_LEVELS) are
    interpolated; variables without that dimension are kept untouched.

    Parameters
    ----------
    ds : xr.Dataset
        Concatenated ARGO profiles with dims ``(N_PROF, N_LEVELS)``.
    depth_levels : np.ndarray
        1-D array of target depth/pressure levels.
    pres_var : str
        Name of the pressure variable (``PRES`` or ``PRES_ADJUSTED``).

    Returns
    -------
    xr.Dataset
        New dataset with vertical dim replaced by ``depth`` coordinate.
    """
    # Identify the vertical dimension ----------------------------------
    pres_da = ds[pres_var]
    vert_dim: Optional[str] = None
    prof_dim = "N_PROF"

    if pres_da.ndim == 2:
        # 2D pressure: (N_PROF, N_LEVELS) — typical for profile_refs data
        for d in pres_da.dims:
            if d != prof_dim:
                vert_dim = str(d)
                break
    elif pres_da.ndim == 1:
        # 1D pressure: simple case
        vert_dim = str(pres_da.dims[0])
        if vert_dim == prof_dim:
            # Pressure is only indexed on profiles, nothing to interpolate
            logger.debug("Pressure variable has only profile dim — cannot interpolate")
            return ds
    else:
        logger.debug(f"Unexpected ndim={pres_da.ndim} for {pres_var}")
        return ds

    if vert_dim is None:
        logger.debug("Could not determine vertical dimension for interpolation")
        return ds

    # Load pressure array (small, already in memory for profile_refs) ----
    pres = np.asarray(pres_da.values, dtype=np.float64)
    if pres.ndim == 1:
        # Broadcast to 2D for uniform handling
        pres = pres[np.newaxis, :]

    n_prof = ds.sizes.get(prof_dim, pres.shape[0])
    n_depth = len(depth_levels)

    # ---- Pre-sort pressure profiles once (shared across all variables) ----
    # For each profile, compute the argsort of its valid pressures so that
    # the costly per-profile sort is done once instead of once-per-variable.
    # Also pre-compute the finite-pressure mask per profile.
    _pres_sort_cache: Dict[int, tuple] = {}  # i -> (sorted_p, sort_order, mask)
    for i in range(n_prof):
        p_col = pres[i, :] if pres.ndim == 2 else pres[0, :]
        fin_mask = np.isfinite(p_col)
        if fin_mask.sum() >= 2:
            p_valid = p_col[fin_mask]
            order = np.argsort(p_valid)
            _pres_sort_cache[i] = (p_valid[order], order, fin_mask)

    # Interpolate each data variable that has vert_dim --------------------
    new_data_vars: Dict[str, tuple] = {}
    for var_name in ds.data_vars:
        if var_name == pres_var:
            continue  # drop the source pressure variable

        var = ds[var_name]
        if vert_dim in var.dims:
            # Needs per-profile interpolation
            data = np.asarray(var.values, dtype=np.float64)
            if data.ndim == 1:
                data = data[np.newaxis, :]

            result = np.full((n_prof, n_depth), np.nan, dtype=np.float64)

            # Vectorised inner loop: reuse pre-sorted pressure indices
            for i, cached in _pres_sort_cache.items():
                sorted_p, order, fin_mask = cached
                v_col = data[i, :]
                # Intersect the pressure-finite mask with value-finite mask
                v_fin = np.isfinite(v_col)
                joint = fin_mask & v_fin
                if joint.sum() < 2:
                    continue
                if np.array_equal(joint, fin_mask):
                    # All pressure-valid points also have valid values:
                    # reuse the pre-computed sort order directly (fast path)
                    v_sorted = v_col[fin_mask][order]
                    result[i, :] = np.interp(
                        depth_levels,
                        sorted_p,
                        v_sorted,
                        left=np.nan,
                        right=np.nan,
                    )
                else:
                    # Some values are NaN where pressure is valid:
                    # must re-sort on the joint subset (rare path)
                    p_j = pres[i, :][joint] if pres.ndim == 2 else pres[0, :][joint]
                    v_j = v_col[joint]
                    j_order = np.argsort(p_j)
                    result[i, :] = np.interp(
                        depth_levels,
                        p_j[j_order],
                        v_j[j_order],
                        left=np.nan,
                        right=np.nan,
                    )

            new_dims = tuple(d if d != vert_dim else "depth" for d in var.dims)
            new_data_vars[var_name] = xr.Variable(new_dims, result, attrs=var.attrs)  # type: ignore[index,assignment]
        else:
            # e.g. TIME(N_PROF), LATITUDE(N_PROF) — keep as-is
            new_data_vars[var_name] = var  # type: ignore[index,assignment]

    # Build new coordinates -----------------------------------------------
    new_coords: Dict[str, xr.Variable] = {
        "depth": xr.Variable(
            ("depth",),
            depth_levels,
            attrs={"units": "dbar", "long_name": "Depth / pressure level"},
        ),
    }
    for cname, cvar in ds.coords.items():
        if cname == pres_var:
            continue  # replaced by "depth"
        if vert_dim not in cvar.dims:
            new_coords[cname] = cvar  # type: ignore[index]

    return xr.Dataset(new_data_vars, coords=new_coords, attrs=ds.attrs)


class ArgoInterface:
    """Ultra-scalable ARGO interface with monthly partitioning and S3/local access.

    Classe ultra-scalable ARGO avec :
    - Partition mensuelle
    - JSON compressé Zstd
    - Index int64 epoch pour O(log n)
    - Lecture lazy Dask / S3
    - Multi-variables et interpolation sur profondeur

    Peut utiliser une configuration S3/Wasabi ou un stockage local.
    """

    def __init__(
        self,
        base_path: str,
        variables: Optional[List[str]] = None,
        s3_storage_options: Optional[Dict] = None,
        chunks: Optional[Dict[str, int]] = None,
        max_fetch_retries: int = 4,
        retry_backoff_seconds: float = 0.8,
    ):
        """
        Initialise l'interface ARGO.

        Args:
            base_path: Chemin de base pour les fichiers index (S3 ou local)
            variables: Liste des variables à extraire
            s3_storage_options: Options pour fsspec S3 (key, secret, endpoint_url, etc.)
            chunks: Configuration des chunks Dask (par défaut {"N_PROF": 2000})
        """
        self.base_path = base_path.rstrip("/")
        self.variables = variables
        self.s3_storage_options = s3_storage_options or {}
        self.chunks = chunks or {"N_PROF": 2000}
        try:
            normalized_retries = int(max_fetch_retries)
        except (TypeError, ValueError):
            normalized_retries = 4
        self.max_fetch_retries = max(0, normalized_retries)

        try:
            normalized_backoff = float(retry_backoff_seconds)
        except (TypeError, ValueError):
            normalized_backoff = 0.8
        self.retry_backoff_seconds = max(0.0, normalized_backoff)
        self._gdac_base_urls = [
            "https://data-argo.ifremer.fr/dac",
            "https://usgodae.org/pub/outgoing/argo/dac",
        ]

    @classmethod
    def from_config(cls, config):
        """
        Crée une instance ArgoInterface à partir d'une ARGOConnectionConfig.

        Args:
            config: Instance de ARGOConnectionConfig ou SimpleNamespace avec les paramètres

        Returns:
            ArgoInterface: Instance configurée
        """
        from types import SimpleNamespace

        # Si config est un objet avec params attribute
        if hasattr(config, "params"):
            params = config.params
        # Si config est déjà un SimpleNamespace ou dict-like
        elif isinstance(config, (SimpleNamespace, dict)):
            params = config if isinstance(config, SimpleNamespace) else SimpleNamespace(**config)
        else:
            params = config

        # Récupérer le base_path
        # Priority: explicit base_path > S3 bucket/folder > local fallback.
        # local_catalog_path is for the DatasetCatalog JSON, NOT for the
        # Kerchunk master_index.  The master_index always lives on S3
        # alongside the monthly .json.zst files when an S3 config is given.
        base_path = getattr(params, "base_path", None)
        has_s3 = (
            hasattr(params, "s3_bucket")
            and params.s3_bucket
            and hasattr(params, "s3_folder")
            and params.s3_folder
        )
        if not base_path and has_s3:
            base_path = f"s3://{params.s3_bucket}/{params.s3_folder}/argo_index"
        elif not base_path:
            local_catalog_path = getattr(params, "local_catalog_path", None)
            if local_catalog_path:
                base_path = str(Path(local_catalog_path).parent / "argo_index")
            else:
                base_path = str(Path(params.local_root) / "argo_index")

        # Récupérer storage_options
        if has_s3:
            # Build S3 credentials from config even when local_catalog_path
            # is set — the master_index & monthly references live on S3.
            if hasattr(config, "get_storage_options"):
                s3_storage_options = config.get_storage_options()
            else:
                s3_storage_options = {}
                if hasattr(params, "endpoint_url") and params.endpoint_url:
                    s3_storage_options["client_kwargs"] = {"endpoint_url": params.endpoint_url}
                if hasattr(params, "s3_key") and params.s3_key:
                    s3_storage_options["key"] = params.s3_key
                if hasattr(params, "s3_secret_key") and params.s3_secret_key:
                    s3_storage_options["secret"] = params.s3_secret_key
        elif hasattr(config, "get_storage_options"):
            s3_storage_options = config.get_storage_options()
        else:
            s3_storage_options = {}

        return cls(
            base_path=base_path,
            variables=getattr(params, "variables", None),
            s3_storage_options=s3_storage_options,
            chunks=getattr(params, "chunks", {"N_PROF": 2000}),
            max_fetch_retries=getattr(params, "max_fetch_retries", 4),
            retry_backoff_seconds=getattr(params, "retry_backoff_seconds", 0.8),
        )

    @staticmethod
    def _is_permanent_error(exc: Exception) -> bool:
        """Return True if *exc* is a permanent (non-retryable) error."""
        if isinstance(exc, FileNotFoundError):
            return True
        if isinstance(exc, OSError) and getattr(exc, "errno", None) == 2:
            return True
        # HTTP 404 wrapped by fsspec / aiohttp
        msg = str(exc).lower()
        if "404" in msg or "not found" in msg:
            return True
        return False

    def _open_profile_with_retries(self, candidate_path: str):
        """Open and translate one ARGO profile with retry/backoff on transient failures.

        Permanent errors (FileNotFoundError, HTTP 404) are raised immediately
        without wasting time on retries.
        """
        if NetCDF3ToZarr is None:  # pragma: no cover
            raise ImportError(
                "kerchunk is required to build ARGO references"
                " (NetCDF3ToZarr), but it failed to import."
            )
        last_exc = None
        for attempt in range(self.max_fetch_retries + 1):
            try:
                # Let NetCDF3ToZarr open the file itself so it records the
                # correct URL inside Zarr chunk references.
                h = NetCDF3ToZarr(candidate_path)
                return h.translate()
            except Exception as exc:
                last_exc = exc

                # Never retry permanent errors
                if self._is_permanent_error(exc):
                    raise

                is_last_attempt = attempt >= self.max_fetch_retries
                if is_last_attempt:
                    break

                # Conservative backoff for network/rate-limit issues (e.g. HTTP/FTP 421)
                wait_s = self.retry_backoff_seconds * (2**attempt) + random.uniform(0.0, 0.35)
                logger.debug(
                    f"ARGO profile fetch retry {attempt + 1}/{self.max_fetch_retries} "
                    f"for '{candidate_path}' after error: {exc} (sleep={wait_s:.2f}s)"
                )
                time.sleep(wait_s)

        raise RuntimeError(last_exc)

    # ---------------- GDAC URL HELPERS ----------------
    def _get_gdac_url_for_profile(self, nc_path_str: str) -> str:
        """Return the canonical GDAC URL for a profile path."""
        if nc_path_str.startswith(("http://", "https://", "ftp://", "s3://")):
            return nc_path_str
        rel_path = nc_path_str.lstrip("/")
        if rel_path.startswith("dac/"):
            rel_path = rel_path[4:]
        return f"{self._gdac_base_urls[0]}/{rel_path}"

    @staticmethod
    def _patch_ref_urls(ref: dict, local_path: str, remote_url: str) -> None:
        """Replace *local_path* with *remote_url* in Kerchunk refs in-place.

        Kerchunk refs map keys to either:
        - a JSON string (inline metadata like .zattrs)
        - a list ``[path, offset, length]`` pointing to a byte range
        Only the latter needs patching.
        """
        refs = ref.get("refs", {})
        for k, v in refs.items():
            if isinstance(v, list) and len(v) >= 1 and v[0] == local_path:
                refs[k] = [remote_url, *v[1:]]

    # ---- Persistent profile download cache (shared across calls) ---------
    _profile_cache_dir: Optional[Path] = None

    @classmethod
    def _get_profile_cache_dir(cls) -> Path:
        """Return (and lazily create) a persistent cache directory for ARGO profiles.

        The directory persists for the lifetime of the process so that
        overlapping time-window requests never re-download the same profile.
        """
        if cls._profile_cache_dir is None or not cls._profile_cache_dir.exists():
            cls._profile_cache_dir = Path(tempfile.gettempdir()) / "argo_profile_dl_cache"
            cls._profile_cache_dir.mkdir(parents=True, exist_ok=True)
        return cls._profile_cache_dir

    @staticmethod
    def _extract_urls_from_refs(refs_list: List[dict]) -> Dict[int, str]:
        """Extract the source URL for each kerchunk ref dict.

        Returns a mapping ``{ref_index: url}`` for every ref whose
        byte-range entries point to a remote HTTP(S) URL.
        """
        result: Dict[int, str] = {}
        for idx, ref_dict in enumerate(refs_list):
            refs = ref_dict.get("refs", {})
            for _k, v in refs.items():
                if (
                    isinstance(v, list)
                    and len(v) >= 1
                    and isinstance(v[0], str)
                    and v[0].startswith(("http://", "https://"))
                ):
                    result[idx] = v[0]
                    break
        return result

    def _batch_download_and_open_profiles(
        self,
        selected_refs: List[dict],
        variables: Optional[List[str]],
        n_download_workers: int = 16,
    ) -> List[xr.Dataset]:
        """Download ARGO profiles in batch, then open locally.

        Uses a single ``requests.Session`` with connection pooling to
        download all needed NetCDF files in parallel.  This avoids the
        per-profile fsspec overhead and shares TCP+TLS handshakes across
        all profiles, which is **dramatically** faster than opening
        individual Kerchunk refs via ``fsspec.filesystem("reference")``.

        Parameters
        ----------
        selected_refs : list[dict]
            Kerchunk ref dicts for the selected profiles.
        variables : list[str] or None
            Variables to keep from each profile.
        n_download_workers : int
            Number of parallel download threads (default 16).

        Returns
        -------
        list[xr.Dataset]
            Successfully loaded profile datasets.
        """
        import requests
        from requests.adapters import HTTPAdapter

        cache_dir = self._get_profile_cache_dir()

        # --- 1. Extract GDAC URLs from kerchunk refs -------------------------
        url_map = self._extract_urls_from_refs(selected_refs)
        if not url_map:
            logger.warning(
                "Could not extract any HTTP URLs from profile refs — "
                "falling back to legacy per-profile fsspec loading"
            )
            return self._legacy_fetch_profiles(selected_refs, variables)

        unique_urls: Dict[str, List[int]] = {}
        for idx, url in url_map.items():
            unique_urls.setdefault(url, []).append(idx)

        # --- 2. Batch download with requests.Session (connection pooling) -----
        session = requests.Session()
        pool_sz = min(n_download_workers + 2, 30)
        retry_strategy: Any
        try:
            from urllib3.util.retry import Retry

            retry_strategy = Retry(
                total=self.max_fetch_retries,
                backoff_factor=self.retry_backoff_seconds,
                status_forcelist=[429, 500, 502, 503, 504],
            )
        except ImportError:
            retry_strategy = self.max_fetch_retries
        adapter = HTTPAdapter(
            pool_connections=pool_sz,
            pool_maxsize=pool_sz,
            max_retries=retry_strategy,
        )
        session.mount("https://", adapter)
        session.mount("http://", adapter)

        downloaded: Dict[str, str] = {}  # url -> local_path
        n_fail = 0
        n_cached = 0

        def _is_valid_cached_profile(local_path: Path) -> bool:
            ds = None
            try:
                if not local_path.exists() or local_path.stat().st_size <= 0:
                    return False
                ds = xr.open_dataset(
                    local_path,
                    engine="scipy",
                    backend_kwargs={"mmap": False},
                )
                return True
            except Exception:
                return False
            finally:
                try:
                    if ds is not None:
                        ds.close()
                except Exception:
                    pass

        def _dl_one(url: str) -> tuple:
            stem = Path(url).stem or "profile"
            local_path = cache_dir / f"{stem}.nc"
            if local_path.exists() and local_path.stat().st_size > 0:
                if _is_valid_cached_profile(local_path):
                    return (url, str(local_path), True)  # cache hit
                local_path.unlink(missing_ok=True)
            try:
                resp = session.get(url, timeout=60)
                resp.raise_for_status()
                local_path.write_bytes(resp.content)
                if not _is_valid_cached_profile(local_path):
                    local_path.unlink(missing_ok=True)
                    raise OSError("downloaded ARGO profile cache is unreadable")
                return (url, str(local_path), False)
            except Exception as exc:
                # Try alternate GDAC mirror
                for alt_base in self._gdac_base_urls[1:]:
                    # Reconstruct URL with alternate mirror
                    for base in self._gdac_base_urls[:1]:
                        if url.startswith(base):
                            alt_url = url.replace(base, alt_base, 1)
                            try:
                                resp = session.get(alt_url, timeout=60)
                                resp.raise_for_status()
                                local_path.write_bytes(resp.content)
                                if not _is_valid_cached_profile(local_path):
                                    local_path.unlink(missing_ok=True)
                                    raise OSError(
                                        "downloaded ARGO profile cache is unreadable"
                                    )
                                return (url, str(local_path), False)
                            except Exception:
                                pass
                return (url, None, exc)

        with ThreadPoolExecutor(max_workers=n_download_workers) as pool:
            futures = {pool.submit(_dl_one, url): url for url in unique_urls}
            for fut in as_completed(futures):
                try:
                    url, local_path, flag = fut.result()
                    if local_path is not None:
                        downloaded[url] = local_path
                        if flag is True:
                            n_cached += 1
                    else:
                        n_fail += 1
                except Exception:
                    n_fail += 1

        session.close()

        n_downloaded = len(downloaded) - n_cached
        if n_downloaded > 0 or n_cached > 0:
            logger.debug(
                f"ARGO batch download: {n_downloaded} new, "
                f"{n_cached} cached, {n_fail} failed "
                f"({len(unique_urls)} unique URLs)"
            )

        # --- 3. Open each profile locally (no HTTP, fast) --------------------
        keep_vars_set = set(variables) if variables else None
        always_keep = {"PRES", "PRES_ADJUSTED", "TIME"}

        per_profile: List[xr.Dataset] = []
        open_fail = 0

        def _open_local(local_path: str) -> Optional[xr.Dataset]:
            try:
                ds = xr.open_dataset(
                    local_path,
                    engine="scipy",
                    backend_kwargs={"mmap": False},
                )
                if "TIME" not in ds and "JULD" in ds:
                    ds = ds.rename({"JULD": "TIME"})
                if keep_vars_set:
                    keep = [v for v in keep_vars_set if v in ds]
                    keep += [c for c in always_keep if c in ds and c not in keep]
                    ds = ds[keep]
                ds.load()
                ds.close()
                return ds
            except Exception:
                return None

        # Open in parallel (local I/O, very fast)
        ordered_local_paths: List[Optional[str]] = []
        for idx in range(len(selected_refs)):
            url = url_map.get(idx)  # type: ignore[assignment]
            if url and url in downloaded:
                ordered_local_paths.append(downloaded[url])
            else:
                ordered_local_paths.append(None)

        with ThreadPoolExecutor(max_workers=min(16, len(selected_refs))) as pool:
            futures_open = {
                pool.submit(_open_local, lp): i
                for i, lp in enumerate(ordered_local_paths)
                if lp is not None
            }
            results_map: Dict[int, xr.Dataset] = {}
            for fut in as_completed(futures_open):  # type: ignore[assignment]
                idx = futures_open[fut]  # type: ignore[index]
                ds = fut.result()
                if ds is not None:
                    results_map[idx] = ds  # type: ignore[assignment]
                else:
                    open_fail += 1

        # Preserve order
        for idx in sorted(results_map.keys()):
            per_profile.append(results_map[idx])

        if open_fail > 0:
            logger.debug(f"ARGO local open: {open_fail} files failed to parse")

        return per_profile

    def _legacy_fetch_profiles(
        self,
        selected_refs: List[dict],
        variables: Optional[List[str]],
    ) -> List[xr.Dataset]:
        """Fallback: load profiles one by one via fsspec (slow, original path)."""
        _MAX_RETRIES = 4
        _BACKOFF_BASE = 0.5

        def _fetch_one(ref: dict) -> Optional[xr.Dataset]:
            last_exc = None
            for attempt in range(_MAX_RETRIES + 1):
                try:
                    fs_ref = fsspec.filesystem(
                        "reference",
                        fo=ref,
                        target_options={"timeout": 30},
                    )
                    mapper = fs_ref.get_mapper("")
                    ds_p = xr.open_dataset(
                        mapper,
                        engine="zarr",
                        consolidated=False,
                    )
                    if "TIME" not in ds_p and "JULD" in ds_p:
                        ds_p = ds_p.rename({"JULD": "TIME"})
                    if variables:
                        keep = [v for v in variables if v in ds_p]
                        keep += [
                            c
                            for c in ("PRES", "PRES_ADJUSTED", "TIME")
                            if c in ds_p and c not in keep
                        ]
                        ds_p = ds_p[keep]
                    ds_p.load()
                    return ds_p
                except Exception as exc:
                    last_exc = exc
                    msg = str(exc).lower()
                    if isinstance(exc, FileNotFoundError) or "404" in msg or "not found" in msg:
                        return None
                    if attempt >= _MAX_RETRIES:
                        break
                    time.sleep(_BACKOFF_BASE * (2**attempt) + random.uniform(0.0, 0.3))
            logger.debug(f"Legacy fetch failed: {last_exc}")
            return None

        results: List[xr.Dataset] = []
        with ThreadPoolExecutor(max_workers=4) as tp:
            futs = {tp.submit(_fetch_one, ref): i for i, ref in enumerate(selected_refs)}
            for fut in as_completed(futs):
                ds = fut.result()
                if ds is not None:
                    results.append(ds)
        return results

    # ---------------- BATCH DOWNLOAD ----------------
    def _batch_download_profiles(
        self,
        files: List[str],
        download_dir: Path,
        n_workers: int = 8,
        desc: str = "Downloading",
    ) -> Dict[str, Tuple[str, str]]:
        """Download ARGO NetCDF profiles in batch with HTTP connection pooling.

        Using a single ``requests.Session`` avoids repeating TCP+TLS
        handshakes for every profile, which is the main cause of slowness
        when downloading thousands of small files.

        Returns:
            dict mapping *nc_path* -> ``(local_file_path, gdac_url)``
            for every successfully downloaded profile.
        """
        import requests
        from requests.adapters import HTTPAdapter

        download_dir = Path(download_dir)
        download_dir.mkdir(parents=True, exist_ok=True)

        # --- session with connection pooling + automatic retries ----------
        session = requests.Session()
        pool_size = min(n_workers + 2, 20)
        retry_strategy: Any
        try:
            from urllib3.util.retry import Retry

            retry_strategy = Retry(
                total=self.max_fetch_retries,
                backoff_factor=self.retry_backoff_seconds,
                status_forcelist=[429, 500, 502, 503, 504],
            )
        except ImportError:
            retry_strategy = self.max_fetch_retries
        adapter = HTTPAdapter(
            pool_connections=pool_size,
            pool_maxsize=pool_size,
            max_retries=retry_strategy,
        )
        session.mount("https://", adapter)
        session.mount("http://", adapter)

        def _download_one(nc_path: str):
            """Download a single profile; returns (nc_path, local, url) or (nc_path, None, exc)."""
            nc_path_str = str(nc_path)
            stem = Path(nc_path_str).stem or Path(nc_path_str.rstrip("/")).name
            local_file = download_dir / f"{stem}.nc"

            # Already on disk from a previous (possibly interrupted) run
            if local_file.exists() and local_file.stat().st_size > 0:
                return (nc_path_str, str(local_file), self._get_gdac_url_for_profile(nc_path_str))

            # Build candidate URLs
            is_remote = nc_path_str.startswith(("http://", "https://"))
            if is_remote:
                urls = [nc_path_str]
            else:
                rel_path = nc_path_str.lstrip("/")
                if rel_path.startswith("dac/"):
                    rel_path = rel_path[4:]
                urls = [f"{base}/{rel_path}" for base in self._gdac_base_urls]

            last_exc: Optional[Exception] = None
            for url in urls:
                try:
                    resp = session.get(url, timeout=60)
                    resp.raise_for_status()
                    local_file.write_bytes(resp.content)
                    return (nc_path_str, str(local_file), url)
                except Exception as exc:
                    last_exc = exc
                    # Don't try other mirrors for 404
                    msg = str(exc).lower()
                    if "404" in msg or "not found" in msg:
                        break

            return (nc_path_str, None, last_exc)

        results: Dict[str, Tuple[str, str]] = {}
        n_fail = 0

        with ThreadPoolExecutor(max_workers=n_workers) as pool:
            futures = {pool.submit(_download_one, f): f for f in files}
            for future in as_completed(futures):
                try:
                    nc_path_str, local_path, url_or_exc = future.result()
                    if local_path is not None:
                        results[nc_path_str] = (local_path, url_or_exc)
                    else:
                        n_fail += 1
                except Exception:
                    n_fail += 1

        logger.debug(f"{desc}: {len(results)}/{len(files)} downloaded, {n_fail} failed")

        if n_fail:
            logger.warning(f"{desc}: {n_fail}/{len(files)} profiles failed to download")

        session.close()
        return results

    # ---------------- ARGOPY ----------------
    def _get_files_for_month(self, year, month):
        if IndexFetcher is None:  # pragma: no cover
            raise ImportError(
                "argopy is required for ARGO index queries, but it failed to import. "
                "Install/fix argopy (and its optional deps) to enable this feature."
            )
        start = f"{year}-{month:02d}-01"
        end_date = pd.Timestamp(start) + pd.offsets.MonthEnd(0)
        end = end_date.strftime("%Y-%m-%d")

        # argopy accepts region boxes with either:
        # - 6 elements: [lon_min, lon_max, lat_min, lat_max, datim_min, datim_max]
        # - 8 elements (older patterns): + [pres_min, pres_max]
        candidate_boxes = [
            [-180, 180, -90, 90, start, end],
            [-180, 180, -90, 90, 0, 6000, start, end],
        ]

        last_exc = None
        idx = None
        for box in candidate_boxes:
            try:
                idx = IndexFetcher(src="gdac").region(box).to_dataframe()
                break
            except Exception as exc:
                last_exc = exc
                logger.debug(f"ARGO IndexFetcher.region failed with box size {len(box)}: {exc}")

        if idx is None:
            raise RuntimeError(
                "Could not query ARGO index for month window "
                f"{start}..{end}. Last error: {last_exc}"
            )

        if "file" not in idx.columns:
            logger.warning(
                f"ARGO monthly index query returned no 'file' column for window {start}..{end}"
            )
            return []

        return idx["file"].dropna().unique().tolist()

    # ---------------- BUILD SINGLE REF ----------------
    def _build_single_ref(self, nc_path, out_dir, download_info=None):
        """Build a Kerchunk reference dict for one ARGO profile.

        Returns the reference *dict* (not a file path) so that
        ``MultiZarrToZarr`` can consume it directly.  A Zstd-compressed
        cache file is kept on disk so that repeated runs skip the GDAC
        download.

        Args:
            nc_path: Original profile path (from argopy).
            out_dir: Directory for ``.json.zst`` cache files.
            download_info: Optional ``(local_path, gdac_url)`` tuple from
                :meth:`_batch_download_profiles`.  When provided the local
                file is used for indexation (no network) and refs are patched
                to point to *gdac_url*.
        """
        nc_path_str = str(nc_path)
        stem = Path(nc_path_str).stem
        if not stem:
            stem = Path(nc_path_str.rstrip("/")).name
        out_file = Path(out_dir) / f"{stem}.json.zst"

        # Cache hit — read back the compressed ref dict
        if out_file.exists():
            try:
                with open(out_file, "rb") as f:
                    dctx = zstd.ZstdDecompressor()
                    return ujson.loads(dctx.decompress(f.read()))
            except Exception:
                # Corrupted cache: delete and re-download
                out_file.unlink(missing_ok=True)

        if NetCDF3ToZarr is None:  # pragma: no cover
            raise ImportError(
                "kerchunk is required to build ARGO references"
                " (NetCDF3ToZarr), but it failed to import."
            )

        ref = None

        if download_info is not None:
            # ---- Fast path: build ref from pre-downloaded local file ----
            local_path, gdac_url = download_info
            try:
                h = NetCDF3ToZarr(local_path)
                ref = h.translate()
                # Replace local paths with the GDAC URL in byte-range refs
                self._patch_ref_urls(ref, local_path, gdac_url)
            except Exception as exc:
                logger.warning(
                    f"Skipping ARGO profile '{nc_path_str}': local indexation failed ({exc})."
                )
                return None
        else:
            # ---- Fallback: download directly via NetCDF3ToZarr (old path) ----
            is_remote = nc_path_str.startswith(("http://", "https://", "ftp://", "s3://"))
            is_absolute_local = nc_path_str.startswith("/")

            if is_remote or is_absolute_local:
                candidate_paths = [nc_path_str]
            else:
                rel_path = nc_path_str.lstrip("/")
                if rel_path.startswith("dac/"):
                    rel_path = rel_path[4:]
                candidate_paths = [f"{base}/{rel_path}" for base in self._gdac_base_urls]

            last_exc = None
            for candidate in candidate_paths:
                try:
                    ref = self._open_profile_with_retries(candidate)
                    break
                except Exception as exc:
                    last_exc = exc

            if ref is None:
                logger.warning(
                    f"Skipping ARGO profile '{nc_path_str}': could not open source "
                    f"from any GDAC path candidate ({last_exc})."
                )
                return None

        if self.variables:
            # Keep requested variables, but also preserve Zarr metadata keys
            # (.zattrs, .zgroup), dimension variables (N_PROF, N_LEVELS, N_PARAM),
            # and coordinate variables (LATITUDE, LONGITUDE) — without these
            # MultiZarrToZarr cannot find concat/identical dims.
            _always_keep = frozenset(
                {
                    "TIME",
                    "JULD",
                    "PRES",
                    "N_PROF",
                    "N_LEVELS",
                    "N_PARAM",
                    "LATITUDE",
                    "LONGITUDE",
                }
            )
            keep_names = _always_keep | set(self.variables)
            ref["refs"] = {
                k: v
                for k, v in ref["refs"].items()
                if k.startswith(".")  # .zattrs, .zgroup
                or k.split("/")[0] in keep_names
            }

        # Persist to disk cache (Zstd-compressed)
        cctx = zstd.ZstdCompressor()
        with open(out_file, "wb") as f:
            f.write(cctx.compress(ujson.dumps(ref).encode()))

        return ref

    # ---------------- BUILD MONTH ----------------
    def build_month(self, year, month, temp_dir="tmp_refs", n_workers=8):
        """Build the compressed Kerchunk JSON for one month.

        Uses a **two-phase** approach for speed:

        1. **Batch download** — all missing profiles are fetched in
           parallel using ``requests.Session`` (HTTP connection pooling:
           a single TCP+TLS handshake per GDAC mirror instead of one
           per profile).
        2. **Local indexation** — ``NetCDF3ToZarr`` runs on the locally
           cached ``.nc`` files (no network latency).  Refs are patched
           so they still point to the GDAC URLs.

        After indexation the raw ``.nc`` cache is deleted to save disk.
        """
        if "://" not in self.base_path:
            Path(self.base_path).mkdir(parents=True, exist_ok=True)

        files = self._get_files_for_month(year, month)
        if not files:
            logger.warning(f"No ARGO profile files found for {year}-{month:02d}; skipping month.")
            return

        ref_dir = Path(temp_dir) / f"{year}_{month:02d}"
        ref_dir.mkdir(parents=True, exist_ok=True)
        nc_cache_dir = ref_dir / "nc_cache"

        # --- Identify profiles that already have a .json.zst ref cache ----
        to_download: List[str] = []
        for f in files:
            stem = Path(str(f)).stem or Path(str(f).rstrip("/")).name
            if not (ref_dir / f"{stem}.json.zst").exists():
                to_download.append(f)

        logger.info(
            f"ARGO {year}-{month:02d}: {len(files)} profiles, "
            f"{len(files) - len(to_download)} cached, "
            f"{len(to_download)} to download"
        )

        # --- Phase 1: batch download missing profiles ---------------------
        download_map: Dict[str, Tuple[str, str]] = {}
        if to_download:
            download_map = self._batch_download_profiles(
                to_download,
                nc_cache_dir,
                n_workers=n_workers,
                desc=f"Download ARGO {year}-{month:02d}",
            )

        # --- Phase 2: build Kerchunk refs (local I/O only) ----------------
        ref_dicts: List[Dict] = []
        n_failed = 0
        desc = f"Indexing ARGO {year}-{month:02d}"

        with ThreadPoolExecutor(max_workers=n_workers) as pool:
            futures = {
                pool.submit(
                    self._build_single_ref,
                    f,
                    ref_dir,
                    download_map.get(str(f)),
                ): f
                for f in files
            }
            for future in as_completed(futures):
                try:
                    result = future.result()
                    if result is not None:
                        ref_dicts.append(result)
                    else:
                        n_failed += 1
                except Exception:
                    n_failed += 1

        logger.debug(f"{desc}: {len(ref_dicts)} refs built, {n_failed} failed")

        # --- Phase 2.5: clean up raw .nc cache to save disk ---------------
        if nc_cache_dir.exists():
            shutil.rmtree(nc_cache_dir, ignore_errors=True)

        if n_failed:
            logger.warning(
                f"{desc}: {n_failed}/{len(files)} profiles failed ({len(ref_dicts)} succeeded)"
            )

        if not ref_dicts:
            logger.warning(
                f"No valid Kerchunk refs were produced for {year}-{month:02d}; skipping month."
            )
            return

        # --- Phase 3: extract TIME from each ref and build temporal index --
        # MultiZarrToZarr cannot combine ARGO profiles because:
        #   1. N_PROF is a dimension without a coordinate variable -> KeyError
        #   2. N_LEVELS varies across profiles (different floats measure
        #      different depth levels) -> chunk size mismatch
        # Instead we store individual profile refs and combine lazily at
        # query time (open_time_window) using xr.concat.
        profile_times: List[Optional[pd.Timestamp]] = [None] * len(ref_dicts)

        def _extract_time(idx_ref):
            idx, ref = idx_ref
            try:
                fs_ref = fsspec.filesystem("reference", fo=ref)
                mapper = fs_ref.get_mapper("")
                ds_tmp = xr.open_dataset(mapper, engine="zarr", consolidated=False)
                time_var = None
                for name in ("TIME", "JULD", "time", "juld"):
                    if name in ds_tmp.variables:
                        time_var = ds_tmp[name]
                        break
                if time_var is None:
                    for var in ds_tmp.variables.values():
                        standard_name = str(var.attrs.get("standard_name", "")).lower()
                        if standard_name == "time":
                            time_var = var  # type: ignore[assignment]
                            break
                if time_var is None:
                    ds_tmp.close()
                    return (idx, pd.NaT)

                if idx < 3:
                    standard_name = time_var.attrs.get("standard_name")  # type: ignore[assignment]
                    units = time_var.attrs.get("units")
                    logger.debug(
                        f"ARGO time variable detected: {getattr(time_var, 'name', None)} "
                        f"(standard_name={standard_name}, units={units})"
                    )

                values = time_var.values
                if np.issubdtype(getattr(values, "dtype", None), np.datetime64):
                    t = pd.to_datetime(values)
                else:
                    units = time_var.attrs.get("units")
                    calendar = time_var.attrs.get("calendar", "standard")
                    if units:
                        try:
                            decoded = xr.coding.times.decode_cf_datetime(values, units, calendar)
                            t = pd.to_datetime(decoded)
                        except Exception:
                            t = pd.to_datetime(values, errors="coerce")
                    else:
                        t = pd.to_datetime(values, errors="coerce")

                ds_tmp.close()
                t_flat = np.ravel(t)
                if t_flat.size == 0:
                    return (idx, pd.NaT)
                return (idx, t_flat[0])
            except Exception:
                return (idx, pd.NaT)

        desc_t = f"Extracting TIME {year}-{month:02d}"
        with ThreadPoolExecutor(max_workers=n_workers) as pool:
            time_futures = {
                pool.submit(_extract_time, (i, ref)): i for i, ref in enumerate(ref_dicts)
            }
            for future in as_completed(time_futures):
                idx, t_val = future.result()
                profile_times[idx] = t_val

        logger.debug(
            f"{desc_t}: extracted {sum(1 for t in profile_times if t is not None)}"
            f"/{len(ref_dicts)} profiles"
        )

        # Drop profiles where TIME extraction failed
        valid_pairs = [
            (ref_dicts[i], profile_times[i])
            for i in range(len(ref_dicts))
            if profile_times[i] is not None and not pd.isna(profile_times[i])
        ]
        if not valid_pairs:
            logger.warning(
                f"Could not extract TIME from any profile for {year}-{month:02d}; skipping month."
            )
            return

        valid_refs, valid_times = zip(*valid_pairs, strict=False)
        valid_refs = list(valid_refs)
        times_index = pd.DatetimeIndex(valid_times)

        # Sort by time
        order = np.argsort(times_index)
        sorted_refs = [valid_refs[i] for i in order]
        sorted_times_epoch = times_index[order].view(np.int64)

        monthly_json = {
            "profile_refs": sorted_refs,
            "temporal_index": {
                "sorted_times_epoch": sorted_times_epoch.tolist(),
            },
            "metadata": {
                "year": year,
                "month": month,
                "n_prof": len(sorted_refs),
                "variables": self.variables,
                "start": int(sorted_times_epoch[0]),
                "end": int(sorted_times_epoch[-1]),
            },
        }

        out_path = f"{self.base_path}/{year}_{month:02d}.json.zst"
        with fsspec.open(out_path, "wb", **self.s3_storage_options) as f:
            cctx = zstd.ZstdCompressor()
            f.write(cctx.compress(ujson.dumps(monthly_json).encode()))

        logger.info(f"ARGO {year}-{month:02d}: monthly index written ({len(sorted_refs)} profiles)")
        self._update_master_index(year, month, monthly_json["metadata"])

    # ---------------- BUILD MULTI-YEAR / MONTH ----------------
    def build_multi_year_monthly(self, start_year, end_year, temp_dir="tmp_refs", n_workers=8):
        """Construit tous les JSON mensuels pour plusieurs années."""
        total_months = (end_year - start_year + 1) * 12
        logger.info(
            f"Building ARGO monthly index for {start_year}-{end_year} "
            f"({total_months} months, {n_workers} workers)"
        )
        for year in range(start_year, end_year + 1):
            for month in range(1, 13):
                try:
                    self.build_month(year, month, temp_dir=temp_dir, n_workers=n_workers)
                except Exception as exc:
                    logger.warning(
                        f"ARGO monthly index build failed for {year}-{month:02d}; "
                        f"continuing with next month ({exc})"
                    )
        logger.info("Multi-year ARGO index build complete.")

    def build_time_window_monthly(self, start, end, temp_dir="tmp_refs", n_workers=8):
        """Build monthly ARGO index only for months intersecting [start, end]."""
        start_ts = pd.Timestamp(start)
        end_ts = pd.Timestamp(end)
        if start_ts > end_ts:
            start_ts, end_ts = end_ts, start_ts

        month_periods = pd.period_range(
            start=start_ts.to_period("M"), end=end_ts.to_period("M"), freq="M"
        )
        logger.info(
            f"Building ARGO monthly index for {len(month_periods)} months "
            f"({start_ts.date()} -> {end_ts.date()}, {n_workers} workers)"
        )
        for period in month_periods:
            year = int(period.year)
            month = int(period.month)
            try:
                self.build_month(year, month, temp_dir=temp_dir, n_workers=n_workers)
            except Exception as exc:
                logger.warning(
                    f"ARGO monthly index build failed for {year}-{month:02d}; "
                    f"continuing with next month ({exc})"
                )

    # ---------------- MASTER INDEX ----------------
    def _update_master_index(self, year, month, metadata):
        master_path = f"{self.base_path}/master_index.json"
        try:
            with fsspec.open(master_path, "r", **self.s3_storage_options) as f:
                master = ujson.loads(f.read())
        except Exception:
            master = {}

        key = f"{year}_{month:02d}"
        master[key] = {
            "start": metadata["start"],
            "end": metadata["end"],
            "path": f"{key}.json.zst",
        }

        with fsspec.open(master_path, "w", **self.s3_storage_options) as f:
            f.write(ujson.dumps(master, indent=2))

    # ---------------- OPEN TIME WINDOW ----------------
    def open_time_window(
        self,
        start,
        end,
        depth_levels,
        variables=None,
        master_index=None,
        max_profiles: Optional[int] = None,
    ):
        """Open ARGO data for a time window, loading monthly indexes in parallel.

        Monthly Kerchunk JSON indexes are read from S3/Wasabi (not GDAC),
        so parallel loading is safe and does not add pressure on ARGO GDAC
        servers.  Actual GDAC byte-range requests only happen later when
        Dask materialises the lazy dataset.

        Parameters
        ----------
        start, end : str or pd.Timestamp
            Time window boundaries.
        depth_levels : array-like
            Target depth levels for interpolation.
        variables : list[str] or None
            Subset of data variables to keep.
        master_index : dict or None
            Pre-loaded master index dict (from ``ArgoManager._master_index``).
            If *None* the master index is read from S3/local on every call.
        max_profiles : int or None
            Maximum number of profiles to load across all months.
            When set, loading stops once the cap is reached.
            Useful for metadata-only access to avoid loading thousands of
            profiles.
        """
        # --- resolve cache key ------------------------------------------------
        _depth_key = tuple(np.asarray(depth_levels).tolist()) if depth_levels is not None else ()
        _var_key = tuple(sorted(variables)) if variables else ()
        cache_key = (
            str(pd.Timestamp(start)),
            str(pd.Timestamp(end)),
            _depth_key,
            _var_key,
        )
        if not hasattr(self, "_tw_cache"):
            self._tw_cache: Dict[tuple, xr.Dataset] = {}
        if cache_key in self._tw_cache:
            logger.debug("ARGO open_time_window cache hit")
            return self._tw_cache[cache_key]

        # --- master index: prefer caller-provided, avoid S3 round-trip -------
        if master_index is not None:
            master = master_index
        else:
            master_path = f"{self.base_path}/master_index.json"
            with fsspec.open(master_path, "r", **self.s3_storage_options) as f:
                raw = f.read()
            # Support both plain JSON and legacy zstd-compressed format
            try:
                master = ujson.loads(raw)
            except (ValueError, TypeError):
                dctx = zstd.ZstdDecompressor()
                master = ujson.loads(
                    dctx.decompress(raw if isinstance(raw, bytes) else raw.encode())
                )

        start_epoch = int(pd.Timestamp(start).value)
        end_epoch = int(pd.Timestamp(end).value)

        # Filter months whose time range overlaps the requested window
        relevant_months = [
            (key, info)
            for key, info in master.items()
            if not (info["end"] < start_epoch or info["start"] > end_epoch)
        ]

        if not relevant_months:
            raise ValueError("Aucune donnée trouvée pour cette période")

        def _concat_profiles_safe(profile_datasets: List[xr.Dataset]) -> xr.Dataset:
            """Concat ARGO profiles while tolerating heterogeneous N_LEVELS sizes."""
            try:
                return xr.concat(profile_datasets, dim="N_PROF", join="outer")
            except Exception as exc:
                if "N_LEVELS" not in str(exc):
                    raise

                max_n_levels = max(
                    (int(ds.sizes.get("N_LEVELS", 0)) for ds in profile_datasets),
                    default=0,
                )
                if max_n_levels <= 0:
                    raise

                padded: List[xr.Dataset] = []
                target_levels = np.arange(max_n_levels)
                for ds in profile_datasets:
                    if "N_LEVELS" in ds.dims:
                        if "N_LEVELS" not in ds.coords:
                            ds = ds.assign_coords(N_LEVELS=np.arange(ds.sizes["N_LEVELS"]))
                        ds = ds.reindex(N_LEVELS=target_levels)
                    padded.append(ds)

                # logger.debug(
                #    "ARGO concat fallback applied: padded profiles to "
                #    f"N_LEVELS={max_n_levels} after alignment error: {exc}"
                # )
                return xr.concat(padded, dim="N_PROF", join="outer")

        # Track total profiles loaded across months (for max_profiles cap)
        _profiles_loaded_counter = [0]  # mutable wrapper for closure access

        # -- helper: load one month (I/O to S3/Wasabi only) -----------------
        def _load_month(key_info):
            key, info = key_info
            # Resolve the monthly file relative to base_path.
            # master_index may contain stale absolute local paths when
            # it was built then uploaded to S3 — always reconstruct
            # from the key.
            month_path = f"{self.base_path}/{key}.json.zst"
            with fsspec.open(month_path, "rb", **self.s3_storage_options) as f:
                dctx_local = zstd.ZstdDecompressor()
                monthly_json = ujson.loads(dctx_local.decompress(f.read()))

            times_epoch = np.array(
                monthly_json["temporal_index"]["sorted_times_epoch"], dtype=np.int64
            )

            left = np.searchsorted(times_epoch, start_epoch, side="left")
            right = np.searchsorted(times_epoch, end_epoch, side="right")

            n_selected = right - left

            # Apply max_profiles cap if set
            if max_profiles is not None:
                remaining = max_profiles - _profiles_loaded_counter[0]
                if remaining <= 0:
                    logger.debug(
                        f"Month {key}: skipping — max_profiles cap ({max_profiles}) already reached"
                    )
                    return None
                if n_selected > remaining:
                    logger.debug(
                        f"Month {key}: capping selection from {n_selected} to {remaining} profiles "
                        f"(max_profiles={max_profiles})"
                    )
                    right = left + remaining
                    n_selected = remaining  # type: ignore[assignment]
            """if n_selected > 0:
                logger.debug(
                    f"Month {key}: selecting {n_selected}/{n_total} profiles ({pct_selected:.1f}%) "
                    f"for window {pd.Timestamp(start_epoch)} - {pd.Timestamp(end_epoch)}"
                )"""

            if left >= right:
                return None

            # --- New format: individual profile refs -----------------------
            # Optimized path: avoid parsing all refs or loading heavy JSONs if possible
            if "profile_refs" in monthly_json:
                n_total_month = len(monthly_json["profile_refs"])
                # Log only if significantly larger than expected for 12h
                if (right - left) > (n_total_month * 0.5) and n_total_month > 100:
                    logger.debug(
                        f"Accessing >50% of month {key} ({right - left}/{n_total_month} profiles). "
                        "This might be slow."
                    )

                selected_refs = monthly_json["profile_refs"][left:right]
                n_refs = len(selected_refs)
                if n_refs > 2000:
                    logger.info(
                        f"Month {key}: loading {n_refs} profiles in parallel. "
                        "This may take a moment."
                    )

                # --- BATCH DOWNLOAD + LOCAL OPEN (fast path) ---------------
                # Instead of loading each profile via fsspec HTTP (slow,
                # creates per-profile TCP connections, causes GDAC
                # rate-limiting with multiple Dask workers), we:
                #   1. Extract GDAC URLs from kerchunk refs
                #   2. Batch-download all profiles using requests.Session
                #      (HTTP connection pooling -> single TCP+TLS handshake)
                #   3. Open each profile from local disk (no HTTP)
                per_profile = self._batch_download_and_open_profiles(
                    selected_refs,
                    variables,
                    n_download_workers=16,
                )
                failed = n_refs - len(per_profile)

                if failed > 0:
                    logger.info(f"Month {key}: {failed}/{n_refs} profiles failed to load")
                if not per_profile:
                    return None
                _profiles_loaded_counter[0] += len(per_profile)
                combined = _concat_profiles_safe(per_profile)
                # Release individual dataset handles
                for _ds in per_profile:
                    _ds.close()
                return combined

            # --- Legacy format: single combined kerchunk_refs + indices ----
            indices = np.array(
                monthly_json["temporal_index"]["sorted_prof_indices"], dtype=np.int64
            )
            selected = indices[left:right]

            if len(selected) == 0:
                return None

            # target_options/remote_options must be empty:
            # the Kerchunk refs point to public HTTP URLs
            # on data-argo.ifremer.fr, NOT to S3.
            fs_ref = fsspec.filesystem(
                "reference",
                fo=monthly_json["kerchunk_refs"],
            )
            mapper = fs_ref.get_mapper("")
            ds = xr.open_dataset(mapper, engine="zarr", consolidated=False, chunks=self.chunks)

            if "TIME" not in ds and "JULD" in ds:
                ds = ds.rename({"JULD": "TIME"})

            if variables:
                ds = ds[variables + ["PRES", "TIME"]]

            ds = ds.isel(N_PROF=selected)
            _profiles_loaded_counter[0] += len(selected)
            return ds

        # -- parallel month loading ------------------------------------------
        datasets: List[xr.Dataset] = []
        n_months = len(relevant_months)

        if n_months == 1:
            # Single month — skip thread-pool overhead
            ds = _load_month(relevant_months[0])
            if ds is not None:
                datasets.append(ds)
        else:
            max_workers = min(n_months, 12)
            # Duration in days (approx)
            duration_ns = end_epoch - start_epoch
            duration_days = duration_ns / (24 * 3600 * 1_000_000_000)

            logger.info(
                f"Loading {n_months} ARGO monthly indexes"
                f" (coverage: {duration_days:.2f} days) in parallel "
                f"({max_workers} threads, S3/Wasabi only — no GDAC pressure)"
            )
            with ThreadPoolExecutor(max_workers=max_workers) as pool:
                futures = {pool.submit(_load_month, ki): ki[0] for ki in relevant_months}
                for future in as_completed(futures):
                    month_key = futures[future]
                    try:
                        ds = future.result()
                        if ds is not None:
                            datasets.append(ds)
                    except Exception as exc:
                        logger.warning(f"Failed to load ARGO month '{month_key}': {exc}")

        if not datasets:
            raise ValueError("Aucune donnée trouvée pour cette période")

        ds_final = _concat_profiles_safe(datasets)
        # Release per-month datasets (data is now in ds_final)
        for _ds in datasets:
            _ds.close()
        del datasets

        # ------------------------------------------------------------------
        # Depth interpolation (optional, guarded)
        # ------------------------------------------------------------------
        renames: Dict[str, str] = {"N_PROF": "obs"}

        if depth_levels is not None and len(depth_levels) > 0:
            depth_levels_arr = np.asarray(depth_levels, dtype=float)

            # --- Find the best pressure variable --------------------------
            # Prefer PRES_ADJUSTED (quality-controlled), fall back to PRES.
            pres_var: Optional[str] = None
            for candidate in ("PRES_ADJUSTED", "PRES"):
                if candidate in ds_final.data_vars or candidate in ds_final.coords:
                    pres_var = candidate
                    break

            if pres_var is not None:
                ds_final = _interpolate_profiles_to_depth(ds_final, depth_levels_arr, pres_var)
                # After interpolation ds_final has dims (N_PROF, depth).
                # 'depth' is already a coordinate.
                """logger.debug(
                    f"Depth interpolation done using {pres_var} -> "
                    f"{len(depth_levels_arr)} levels"
                )"""
            else:
                logger.warning(
                    "No pressure variable (PRES_ADJUSTED / PRES) found in "
                    "ARGO data — depth interpolation skipped."
                )

            ds_final = ds_final.rename(renames)
        else:
            # No depth levels requested (e.g. metadata-only sampling)
            ds_final = ds_final.rename(renames)

        # Re-chunk only when the data is dask-backed (legacy Kerchunk
        # path).  The modern profile_refs path already loaded everything
        # into NumPy via .load(); wrapping in-memory arrays in dask chunks
        # adds graph-scheduling overhead for no benefit — the caller
        # (prefetch_batch_shared_zarr) will write straight to Zarr.
        _has_dask_vars = any(hasattr(ds_final[v].data, "dask") for v in ds_final.variables)
        if _has_dask_vars:
            raw_chunks = self.chunks or {"N_PROF": 500}
            final_dims = set(ds_final.dims)
            mapped_chunks: Dict[str, int] = {}
            for k, v in raw_chunks.items():
                target = renames.get(k, k)  # apply same rename mapping
                if target in final_dims:
                    mapped_chunks[target] = v
            if not mapped_chunks:
                mapped_chunks = {"obs": 500} if "obs" in final_dims else {}
            if mapped_chunks:
                ds_final = ds_final.chunk(mapped_chunks)

        # Keep cache bounded to avoid graph/object accumulation across many windows.
        if len(self._tw_cache) >= 1:
            self._tw_cache.clear()
        self._tw_cache[cache_key] = ds_final
        return ds_final
