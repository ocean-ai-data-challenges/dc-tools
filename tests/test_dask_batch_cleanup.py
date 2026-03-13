"""Tests for Dask batch cleanup functionality."""

from types import SimpleNamespace


class _FakeClient:
    def __init__(self):
        self.calls = []
        self._submitted = []

    def ncores(self):
        # Mimic distributed.Client.ncores(): {worker_addr: nthreads}
        return {"tcp://worker": 1}

    def submit(self, fn, *args, **kwargs):
        fut = _FakeFuture({"duration_s": 0.1, "n_points": 1})
        self._submitted.append(fut)
        return fut

    def scheduler_info(self):
        return {
            "workers": {
                "w": {"name": "w", "status": "running", "memory_limit": 1, "metrics": {"memory": 0}}
            }
        }

    def cancel(self, futures, force=False):
        self.calls.append(("cancel", force, len(futures) if futures is not None else None))

    def run(self, fn):
        # we don't execute fn here; we just assert it's called
        self.calls.append(("run", getattr(fn, "__name__", str(fn))))


class _FakeDatasetProcessor:
    def __init__(self, client):
        self.client = client


class _FakeFuture:
    def __init__(self, value=None):
        self._value = value

    def result(self, timeout=None):
        return self._value

    def cancel(self):
        return True


class _FakeAsCompleted:
    def __init__(self, futures=None):
        self._futures = list(futures or [])

    def add(self, fut):
        self._futures.append(fut)

    def __iter__(self):
        # Yield futures in submission order.
        yield from list(self._futures)


class _FakeTqdmBar:
    def __init__(self, *args, **kwargs):
        pass

    def set_postfix_str(self, *args, **kwargs):
        return None

    def update(self, *args, **kwargs):
        return None

    def write(self, *args, **kwargs):
        return None

    def close(self):
        return None


def test_batch_cleanup_cancels_with_force_and_runs_worker_cleanup(monkeypatch):
    """Ensure Evaluator's batch loop uses cancel(force=True) and runs worker cleanup."""
    # Import here to ensure we test the installed/module version under pytest.
    import dctools.metrics.evaluator as ev

    fake_client = _FakeClient()

    # Patch wait() to be a no-op in this unit test.
    monkeypatch.setattr(ev, "wait", lambda futures: None)

    # Patch as_completed() to support .add() + iteration.
    monkeypatch.setattr(ev, "as_completed", lambda futures=None: _FakeAsCompleted(futures))

    # Patch tqdm() to avoid progress bar overhead.
    monkeypatch.setattr(ev, "tqdm", lambda *a, **k: _FakeTqdmBar())

    # Build a minimal Evaluator instance without invoking its heavy __init__.
    evaluator = ev.Evaluator.__new__(ev.Evaluator)
    evaluator.dataset_processor = _FakeDatasetProcessor(fake_client)
    evaluator.metrics = {"ref": [object()]}
    evaluator.reduce_precision = False
    evaluator.results_dir = "/tmp"
    evaluator.dataloader = SimpleNamespace(ref_managers={}, pred_manager=None, pred_alias="pred")

    # ParallelismConfig is required by _evaluate_batch (pcfg.reduce_precision, etc.)
    from dctools.utilities.parallelism import ParallelismConfig
    evaluator.pcfg = ParallelismConfig()

    # Avoid actually running compute_metric in this unit test.
    monkeypatch.setattr(ev, "compute_metric", lambda *a, **k: {"duration_s": 0.1, "n_points": 1})

    # Minimal inputs
    batch = [
        {
            "ref_alias": "ref",
            "pred_data": "x",
            "ref_data": "y",
            "forecast_reference_time": "2024-01-01",
            "lead_time": 0,
            "valid_time": "2024-01-01",
            "pred_coords": None,
            "ref_coords": None,
            "ref_is_observation": False,
        }
    ]

    # We don't use inside _evaluate_batch besides deletions/attrs
    pred_params = SimpleNamespace(protocol="local")
    ref_params = SimpleNamespace(protocol="local")
    pred_transform = SimpleNamespace()
    ref_transform = SimpleNamespace()

    out = evaluator._evaluate_batch(
        batch=batch,
        pred_alias="pred",
        ref_alias="ref",
        pred_connection_params=pred_params,
        ref_connection_params=ref_params,
        pred_transform=pred_transform,
        ref_transform=ref_transform,
        argo_index=None,
    )

    assert isinstance(out, list)
    assert out and out[0] is not None

    # Validate cleanup semantics
    assert any(c[0] == "cancel" and c[1] is True for c in fake_client.calls)
    assert any(c[0] == "run" for c in fake_client.calls)
