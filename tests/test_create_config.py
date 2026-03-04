"""Unit tests for worker connection config factory."""

from types import SimpleNamespace

from dctools.data.connection import create_config as cc


class _FakeConfigClass:
    def __init__(self, params):
        self.params = SimpleNamespace(**params)


class _FakeConnectionManager:
    def __init__(self, config, call_list_files=False, do_logging=False):
        self.config = config
        self.call_list_files = call_list_files
        self.do_logging = do_logging

    def open(self, path, *args, **kwargs):
        return (path, args, kwargs)


class _FakeSession:
    def __init__(self):
        self.closed = False

    def close(self):
        self.closed = True


class _FakeFS:
    def __init__(self):
        self._session = _FakeSession()


def test_create_worker_connect_config_local(monkeypatch):
    """Factory should rebuild config/manager and return callable open function."""
    monkeypatch.setitem(cc.CONNECTION_CONFIG_REGISTRY, "local", _FakeConfigClass)
    monkeypatch.setitem(cc.CONNECTION_MANAGER_REGISTRY, "local", _FakeConnectionManager)

    src = SimpleNamespace(protocol="local", foo="bar", dataset_processor=object())

    open_func = cc.create_worker_connect_config(src)

    assert callable(open_func)
    result = open_func("file.nc")
    assert result[0] == "file.nc"


def test_create_worker_connect_config_cmems_closes_sessions(monkeypatch):
    """CMEMS protocol should close previous fs sessions before rebuilding config."""
    monkeypatch.setitem(cc.CONNECTION_CONFIG_REGISTRY, "cmems", _FakeConfigClass)
    monkeypatch.setitem(cc.CONNECTION_MANAGER_REGISTRY, "cmems", _FakeConnectionManager)

    fs = _FakeFS()
    src = SimpleNamespace(protocol="cmems", fs=fs, dataset_processor=object())

    open_func = cc.create_worker_connect_config(src)

    assert fs._session.closed is True
    assert callable(open_func)
