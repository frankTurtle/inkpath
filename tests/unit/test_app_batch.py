"""Batch-mode orchestration: reconcile-then-submit, and the no-double-commit rule."""

from __future__ import annotations

import pytest

from rmsync import app as app_mod
from rmsync import state as state_mod
from rmsync.app import Runner
from rmsync.config import Config
from rmsync.fetch import PageRef
from rmsync.providers import ProviderResult


def _cfg(**kw):
    base = dict(
        state_bucket="b", github_repo="o/r", watch_folder="R",
        batch_mode="bedrock-batch", batch_min_records=100, batch_max_wait_days=14,
        batch_role_arn="arn:role", ai_model_id="anthropic.claude-x",
    )
    base.update(kw)
    return Config(**base)


class StubProvider:
    def __init__(self):
        self.calls = 0

    def extract_and_tag(self, png, tags, titles=None):
        self.calls += 1
        return ProviderResult(text="y" * 60, tags=["t"], title="T")


class StubVault:
    def __init__(self):
        self.committed: list[str] = []

    def put_file(self, path, content, message):
        self.committed.append(path)
        return "sha"


@pytest.fixture
def staged(monkeypatch):
    """In-memory stand-in for the S3 staging prefix."""
    store: dict[str, bytes] = {}
    monkeypatch.setattr(app_mod, "stage_png", lambda b, k, p: store.__setitem__(k, p))

    def _load(bucket, key):
        if key not in store:
            raise KeyError(key)
        return store[key]

    monkeypatch.setattr(app_mod, "load_png", _load)
    return store


def _runner(cfg, provider=None, vault=None):
    r = Runner(cfg)
    r._provider = provider or StubProvider()
    r._vault = vault or StubVault()
    return r


def _page(data, page_id="p1"):
    return PageRef(
        doc_id="d1", doc_hash="dh", notebook="Reading", page_id=page_id,
        page_hash=f"h-{page_id}", page_index=0, data=data,
    )


def test_process_batch_queues_and_marks_pending(page_rm, staged, monkeypatch):
    st = state_mod.empty_state()
    r = _runner(_cfg())
    r.process_batch(st, [_page(page_rm)])

    assert len(st["batch"]["queue"]) == 1
    assert state_mod.is_page_pending_batch(st, "d1", "p1", "h-p1")
    assert not state_mod.is_page_current(st, "d1", "p1", "h-p1")
    assert staged  # the render was parked for a later poll
    assert r.stats["batchQueued"] == 1


def test_process_batch_does_not_submit_below_minimum(page_rm, staged, monkeypatch):
    """Bedrock rejects a sub-100 job outright, so it must never be attempted."""
    called = {"n": 0}

    class NeverSubmit:
        def submit(self, *a, **kw):
            called["n"] += 1
            raise AssertionError("must not submit")

    r = _runner(_cfg())
    monkeypatch.setattr(r, "_batch", lambda: NeverSubmit())
    r.process_batch(state_mod.empty_state(), [_page(page_rm)])
    assert called["n"] == 0


def test_process_batch_submits_once_minimum_reached(page_rm, staged, monkeypatch):
    st = state_mod.empty_state()
    r = _runner(_cfg(batch_min_records=1))

    class FakeBatch:
        def __init__(self):
            self.submitted = None

        def submit(self, state, pages, *, min_records, build_record):
            self.submitted = pages
            return "arn:1"

    fake = FakeBatch()
    monkeypatch.setattr(r, "_batch", lambda: fake)
    r.process_batch(st, [_page(page_rm)])

    assert fake.submitted is not None and len(fake.submitted) == 1
    assert st["batch"]["queue"] == []          # drained on successful submit


def test_failed_submit_retains_queue(page_rm, staged, monkeypatch):
    st = state_mod.empty_state()
    r = _runner(_cfg(batch_min_records=1))

    class Boom:
        def submit(self, *a, **kw):
            raise RuntimeError("throttled")

    monkeypatch.setattr(r, "_batch", lambda: Boom())
    r.process_batch(st, [_page(page_rm)])
    assert len(st["batch"]["queue"]) == 1      # survives for the next poll
    assert r.stats["errors"] == 1


def test_blank_page_never_enters_the_batch_queue(blank_rm, staged):
    st = state_mod.empty_state()
    r = _runner(_cfg())
    r.process_batch(st, [_page(blank_rm)])
    assert st["batch"]["queue"] == []
    assert r.stats["blankSkipped"] == 1


def test_reconcile_commits_completed_results(staged, monkeypatch):
    st = state_mod.empty_state()
    record = {
        "doc_id": "d1", "doc_hash": "dh", "notebook": "Reading", "page_id": "p1",
        "page_hash": "h-p1", "page_index": 0, "s3_key": "staging/d1/p1.png",
        "queued_at": "2026-08-01T00:00:00Z",
    }
    st["batch"]["pendingJobs"] = [
        {"jobArn": "arn:1", "outputPrefix": "batch/x/", "submittedAt": "2026-08-01T00:00:00Z",
         "records": [record]}
    ]
    staged["staging/d1/p1.png"] = b"PNG"

    class FakeBatch:
        def check_pending(self, state):
            return list(state["batch"]["pendingJobs"]), [], []

        def retrieve_results(self, job):
            return {"p1": '{"text":"' + "z" * 60 + '","tags":["a"],"title":"T","links":[]}'}

        def clear_job(self, state, job):
            state["batch"]["pendingJobs"] = []

    vault = StubVault()
    r = _runner(_cfg(), vault=vault)
    monkeypatch.setattr(r, "_batch", lambda: FakeBatch())
    r.reconcile_batches(st)

    assert r.stats["batchReconciled"] == 1
    assert any(p.endswith(".md") for p in vault.committed)
    assert state_mod.is_page_current(st, "d1", "p1", "h-p1")
    assert st["batch"]["pendingJobs"] == []


def test_reconcile_failed_job_falls_back_to_sync(staged, monkeypatch):
    st = state_mod.empty_state()
    record = {
        "doc_id": "d1", "doc_hash": "dh", "notebook": "Reading", "page_id": "p1",
        "page_hash": "h-p1", "page_index": 0, "s3_key": "staging/d1/p1.png",
        "queued_at": "2026-08-01T00:00:00Z",
    }
    st["batch"]["pendingJobs"] = [
        {"jobArn": "arn:1", "outputPrefix": "b/", "submittedAt": "2026-08-01T00:00:00Z",
         "records": [record]}
    ]
    staged["staging/d1/p1.png"] = b"PNG"

    class FakeBatch:
        def check_pending(self, state):
            return [], list(state["batch"]["pendingJobs"]), []

        def clear_job(self, state, job):
            state["batch"]["pendingJobs"] = []

    provider, vault = StubProvider(), StubVault()
    r = _runner(_cfg(), provider=provider, vault=vault)
    monkeypatch.setattr(r, "_batch", lambda: FakeBatch())
    r.reconcile_batches(st)

    assert provider.calls == 1                 # re-run on demand, not lost
    assert any(p.endswith(".md") for p in vault.committed)


def test_no_double_commit_when_batch_and_fallback_overlap(staged, monkeypatch):
    """A page already committed by the batch path is not committed again."""
    st = state_mod.empty_state()
    state_mod.record_page(
        st, doc_id="d1", doc_hash="dh", notebook="Reading", page_id="p1", page_hash="h-p1",
        note_path="Inbox/Reading/T.md", status=state_mod.STATUS_COMMITTED, timestamp="t",
    )
    staged["staging/d1/p1.png"] = b"PNG"
    record = {
        "doc_id": "d1", "doc_hash": "dh", "notebook": "Reading", "page_id": "p1",
        "page_hash": "h-p1", "page_index": 0, "s3_key": "staging/d1/p1.png",
        "queued_at": "2026-08-01T00:00:00Z",
    }
    vault = StubVault()
    r = _runner(_cfg(), vault=vault)
    r._process_records_synchronously(st, [record])
    assert vault.committed == []


def test_missing_staged_png_clears_state_so_page_resyncs(staged, monkeypatch):
    """If the staged render expired, drop the record so it is re-fetched."""
    st = state_mod.empty_state()
    state_mod.record_page(
        st, doc_id="d1", doc_hash="dh", notebook="Reading", page_id="p1", page_hash="h-p1",
        note_path="", status=state_mod.STATUS_PENDING_BATCH, timestamp="t",
    )
    record = {
        "doc_id": "d1", "doc_hash": "dh", "notebook": "Reading", "page_id": "p1",
        "page_hash": "h-p1", "page_index": 0, "s3_key": "staging/gone.png",
        "queued_at": "2026-08-01T00:00:00Z",
    }
    r = _runner(_cfg())
    r._process_records_synchronously(st, [record])
    assert state_mod.page_record(st, "d1", "p1") is None
    assert r.stats["errors"] == 1


def test_overdue_pages_force_flush_to_sync(page_rm, staged, monkeypatch):
    """Past BatchMaxWaitDays a page must never sit waiting for a quorum."""
    st = state_mod.empty_state()
    staged["staging/d1/old.png"] = b"PNG"
    st["batch"]["queue"] = [{
        "doc_id": "d1", "doc_hash": "dh", "notebook": "Reading", "page_id": "old",
        "page_hash": "h-old", "page_index": 0, "s3_key": "staging/d1/old.png",
        "queued_at": "2020-01-01T00:00:00Z",
    }]
    provider, vault = StubProvider(), StubVault()
    r = _runner(_cfg(), provider=provider, vault=vault)

    class NoSubmit:
        def submit(self, *a, **kw):
            return None

    monkeypatch.setattr(r, "_batch", lambda: NoSubmit())
    r.process_batch(st, [])

    assert provider.calls == 1                       # flushed through sync
    assert st["batch"]["queue"] == []                # and removed from the queue
