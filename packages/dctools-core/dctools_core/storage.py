"""Filesystems for remote object stores, and retries around them.

Moved from ``dctools.data.connection.config.S3ConnectionConfig.create_fs`` and
``dctools.data.connection.connection_manager._is_transient_remote_error``.

The EDITO MinIO gateway (``https://minio.dive.edito.eu``) is the default endpoint because every
dataset curated for the data challenges lives there, readable anonymously; credentials are only
used when both a key and a secret are given (arguments, then ``AWS_ACCESS_KEY_ID`` /
``AWS_SECRET_ACCESS_KEY``).
"""
from __future__ import annotations

import logging
import os
import random
import time
from collections.abc import Callable
from typing import Any, TypeVar

import fsspec

log = logging.getLogger(__name__)

EDITO_ENDPOINT = "https://minio.dive.edito.eu"
ENDPOINT_ENV = "DCTOOLS_S3_ENDPOINT"

T = TypeVar("T")


def make_filesystem(protocol: str = "s3", *, endpoint_url: str | None = None,
                    key: str | None = None, secret_key: str | None = None,
                    anon: bool | None = None, **kwargs: Any) -> fsspec.AbstractFileSystem:
    """Return an fsspec filesystem; for ``s3`` resolve endpoint and credentials.

    ``endpoint_url`` defaults to ``$DCTOOLS_S3_ENDPOINT`` then the EDITO gateway. ``anon`` defaults to
    "no full credential pair available". Timeouts are set through ``config_kwargs`` (passing a
    botocore ``Config`` object in ``client_kwargs`` breaks with recent aiobotocore).
    """
    if protocol != "s3":
        return fsspec.filesystem(protocol, **kwargs)
    key = key or os.environ.get("AWS_ACCESS_KEY_ID")
    secret_key = secret_key or os.environ.get("AWS_SECRET_ACCESS_KEY")
    if anon is None:
        anon = not (key and secret_key)
    endpoint_url = endpoint_url or os.environ.get(ENDPOINT_ENV) or EDITO_ENDPOINT
    client_kwargs = {"endpoint_url": endpoint_url, **kwargs.pop("client_kwargs", {})}
    config_kwargs = {"connect_timeout": 30, "read_timeout": 60, **kwargs.pop("config_kwargs", {})}
    if anon:
        return fsspec.filesystem("s3", anon=True, client_kwargs=client_kwargs,
                                 config_kwargs=config_kwargs, **kwargs)
    return fsspec.filesystem("s3", key=key, secret=secret_key, client_kwargs=client_kwargs,
                             config_kwargs=config_kwargs, **kwargs)


def split_url(url: str) -> tuple[str, str]:
    """``s3://bucket/prefix`` -> ``("s3", "bucket/prefix")``; a bare path -> ``("file", path)``."""
    if "://" in url:
        protocol, _, path = url.partition("://")
        return protocol, path
    return "file", url


def is_transient_remote_error(exc: BaseException) -> bool:
    """Heuristic for S3/network errors worth retrying.

    Covers gateway hiccups such as HTTP 499 (nginx "client closed request", emitted when the
    client is too slow to respond, e.g. under RAM/swap pressure), 500/502/503/504, and generic
    connection resets/timeouts. String-matching on the message is deliberate: the concrete type
    varies between botocore ``ClientError``, ``OSError`` and aiohttp errors.
    """
    msg = str(exc)
    return any(token in msg for token in (
        "499", "500", "502", "503", "504", "SlowDown", "RequestTimeout",
        "Connection reset", "Connection aborted", "Timeout", "timed out",
    ))


def retry_remote(fn: Callable[[], T], *, attempts: int = 3, base_delay: float = 3.0,
                 fs: fsspec.AbstractFileSystem | None = None, what: str = "remote call") -> T:
    """Call ``fn`` with exponential backoff on transient errors; invalidate ``fs``'s listing cache
    between attempts (a broken/stale listing is often what the next attempt would reuse)."""
    last: BaseException | None = None
    for attempt in range(1, attempts + 1):
        try:
            return fn()
        except Exception as exc:  # noqa: BLE001 - the filter is is_transient_remote_error
            last = exc
            if attempt >= attempts or not is_transient_remote_error(exc):
                raise
            if fs is not None:
                try:
                    fs.invalidate_cache()
                except Exception:  # noqa: BLE001
                    pass
            wait = base_delay * 2 ** (attempt - 1) + random.uniform(0, 0.5)
            log.debug("transient error on %s (attempt %d/%d): %r; retrying in %.1fs",
                      what, attempt, attempts, exc, wait)
            time.sleep(wait)
    assert last is not None
    raise last
