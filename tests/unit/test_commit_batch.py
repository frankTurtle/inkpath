from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from rmsync import batch as batch_mod
from rmsync import state as state_mod
from rmsync.batch import QueuedPage, drop_from_queue, overdue_pages, queue_pages, should_submit
from rmsync.commit import CommitError, GitHubVault, attachment_path, commit_message, note_path

# ----------------------------------------------------------------- commit ---


class FakeResp:
    def __init__(self, status=200, payload=None, text=""):
        self.status_code = status
        self.ok = 200 <= status < 300
        self._payload = payload if payload is not None else {}
        self.text = text

    def json(self):
        return self._payload


class FakeSession:
    def __init__(self, gets, puts):
        self.headers: dict = {}
        self._gets = list(gets)
        self._puts = list(puts)
        self.put_calls: list[dict] = []

    def get(self, url, **kw):
        return self._gets.pop(0)

    def put(self, url, json=None, **kw):
        self.put_calls.append(json or {})
        return self._puts.pop(0)


def _vault(session):
    v = GitHubVault("owner/repo", "token", "main")
    v._session = session
    return v


def test_commit_new_file_sends_no_sha():
    s = FakeSession(gets=[FakeResp(404)], puts=[FakeResp(201, {"commit": {"sha": "abc123"}})])
    assert _vault(s).put_file("a/b.md", b"body", "msg") == "abc123"
    assert "sha" not in s.put_calls[0]


def test_commit_existing_file_includes_sha():
    s = FakeSession(
        gets=[FakeResp(200, {"sha": "deadbeef"})], puts=[FakeResp(200, {"commit": {"sha": "c1"}})]
    )
    _vault(s).put_file("a/b.md", b"body", "msg")
    assert s.put_calls[0]["sha"] == "deadbeef"


def test_commit_retries_on_409_rereading_sha(monkeypatch):
    monkeypatch.setattr("rmsync.commit.time.sleep", lambda *_: None)
    s = FakeSession(
        gets=[FakeResp(200, {"sha": "stale"}), FakeResp(200, {"sha": "fresh"})],
        puts=[FakeResp(409, text="conflict"), FakeResp(200, {"commit": {"sha": "ok"}})],
    )
    assert _vault(s).put_file("a/b.md", b"x", "m") == "ok"
    assert s.put_calls[0]["sha"] == "stale"
    assert s.put_calls[1]["sha"] == "fresh"   # SHA re-read, not reused


def test_commit_gives_up_after_max_retries(monkeypatch):
    monkeypatch.setattr("rmsync.commit.time.sleep", lambda *_: None)
    s = FakeSession(
        gets=[FakeResp(200, {"sha": "s"})] * 3, puts=[FakeResp(409, text="conflict")] * 3
    )
    with pytest.raises(CommitError, match="after 3 attempts"):
        _vault(s).put_file("a/b.md", b"x", "m")


def test_commit_raises_on_auth_failure():
    s = FakeSession(gets=[FakeResp(404)], puts=[FakeResp(401, text="Bad credentials")])
    with pytest.raises(CommitError, match="401"):
        _vault(s).put_file("a/b.md", b"x", "m")


def test_bad_repo_format_rejected():
    with pytest.raises(CommitError, match="owner/repo"):
        GitHubVault("notarepo", "t")


def test_note_paths_are_sanitized():
    p = note_path("Inbox/reMarkable", "Reading: Antifragile", "Via negativa")
    assert p == "Inbox/reMarkable/Reading Antifragile/Via negativa.md"
    a = attachment_path("Inbox/reMarkable", "Reading: Antifragile", "Via negativa")
    assert a.endswith("/attachments/Via negativa.png")


def test_commit_message_format():
    assert commit_message("Reading", "b083f079-c45e") == "rm-sync: Reading (b083f079)"


# ------------------------------------------------------------------ batch ---


def _page(pid="p1", days_old=0):
    ts = (datetime.now(UTC) - timedelta(days=days_old)).strftime("%Y-%m-%dT%H:%M:%SZ")
    return QueuedPage(
        doc_id="d1", doc_hash="dh", notebook="N", page_id=pid, page_hash=f"h-{pid}",
        page_index=0, s3_key=f"staging/{pid}.png", queued_at=ts,
    )


def test_batch_rejects_submit_below_minimum():
    """Bedrock rejects a sub-100 job at submission - never call it speculatively."""
    st = state_mod.empty_state()
    queue_pages(st, [_page(f"p{i}") for i in range(5)])
    assert should_submit(st, 100, "bedrock-batch") is False


def test_batch_submits_at_minimum():
    st = state_mod.empty_state()
    queue_pages(st, [_page(f"p{i}") for i in range(100)])
    assert should_submit(st, 100, "bedrock-batch") is True


def test_direct_batch_has_no_minimum():
    st = state_mod.empty_state()
    queue_pages(st, [_page("p1")])
    assert should_submit(st, 100, "direct-batch") is True


def test_batch_submit_call_refuses_below_minimum():
    """The submit path itself must refuse, not just the predicate."""

    class NeverCalled:
        def create_model_invocation_job(self, **kw):
            raise AssertionError("must not submit below the minimum")

    b = batch_mod.BedrockBatch(
        model_id="m", bucket="b", role_arn="r", client=NeverCalled(), s3_client=object()
    )
    st = state_mod.empty_state()
    assert b.submit(st, [_page()], min_records=100, build_record=lambda p: {}) is None


def test_batch_force_flushes_past_max_wait():
    st = state_mod.empty_state()
    queue_pages(st, [_page("old", days_old=20), _page("new", days_old=1)])
    assert [p.page_id for p in overdue_pages(st, 14)] == ["old"]


def test_batch_nothing_overdue_within_window():
    st = state_mod.empty_state()
    queue_pages(st, [_page("recent", days_old=2)])
    assert overdue_pages(st, 14) == []


def test_batch_pending_state_not_refetched():
    """A page already queued is not re-added on the next poll."""
    st = state_mod.empty_state()
    page = _page("p1")
    queue_pages(st, [page])
    queue_pages(st, [page])
    assert len(st["batch"]["queue"]) == 1


def test_drop_from_queue_removes_only_named_pages():
    st = state_mod.empty_state()
    queue_pages(st, [_page("a"), _page("b")])
    drop_from_queue(st, [_page("a")])
    assert [i["page_id"] for i in st["batch"]["queue"]] == ["b"]


def test_batch_failed_job_falls_back_to_sync():
    class FakeBedrock:
        def get_model_invocation_job(self, jobIdentifier):  # noqa: N803
            return {"status": "Failed"}

    st = state_mod.empty_state()
    st["batch"]["pendingJobs"] = [
        {"jobArn": "arn:1", "submittedAt": "2026-08-01T00:00:00Z",
         "outputPrefix": "batch/x/", "records": [_page().to_dict()]}
    ]
    b = batch_mod.BedrockBatch(
        model_id="m", bucket="b", role_arn="r", client=FakeBedrock(), s3_client=object()
    )
    completed, failed, running = b.check_pending(st)
    assert not completed and not running and len(failed) == 1


def test_batch_poll_error_keeps_job_pending():
    """A transient poll failure must not silently drop the job."""

    class Boom:
        def get_model_invocation_job(self, jobIdentifier):  # noqa: N803
            raise RuntimeError("throttled")

    st = state_mod.empty_state()
    st["batch"]["pendingJobs"] = [
        {"jobArn": "arn:1", "submittedAt": "2026-08-01T00:00:00Z", "outputPrefix": "p/", "records": []}
    ]
    b = batch_mod.BedrockBatch(
        model_id="m", bucket="b", role_arn="r", client=Boom(), s3_client=object()
    )
    completed, failed, running = b.check_pending(st)
    assert len(running) == 1 and not failed


def test_batch_submit_writes_manifest_and_records_job():
    class FakeS3:
        def __init__(self):
            self.puts = []

        def put_object(self, **kw):
            self.puts.append(kw)

    class FakeBedrock:
        def create_model_invocation_job(self, **kw):
            self.kwargs = kw
            return {"jobArn": "arn:job:1"}

    s3, br = FakeS3(), FakeBedrock()
    b = batch_mod.BedrockBatch(model_id="m", bucket="bk", role_arn="role", client=br, s3_client=s3)
    st = state_mod.empty_state()
    pages = [_page(f"p{i}") for i in range(3)]
    arn = b.submit(st, pages, min_records=3, build_record=lambda p: {"recordId": p.page_id})

    assert arn == "arn:job:1"
    assert len(s3.puts) == 1
    assert s3.puts[0]["Body"].decode().count("\n") == 3      # one JSONL line per page
    assert br.kwargs["timeoutDurationInHours"] == 24         # Bedrock's minimum
    assert st["batch"]["pendingJobs"][0]["jobArn"] == "arn:job:1"


def test_clear_job_removes_only_that_job():
    st = state_mod.empty_state()
    st["batch"]["pendingJobs"] = [{"jobArn": "a"}, {"jobArn": "b"}]
    b = batch_mod.BedrockBatch(
        model_id="m", bucket="bk", role_arn="r", client=object(), s3_client=object()
    )
    b.clear_job(st, {"jobArn": "a"})
    assert [j["jobArn"] for j in st["batch"]["pendingJobs"]] == ["b"]


def test_no_double_commit_across_batch_and_sync_paths():
    """Whichever path produced the content, an already-committed page is skipped."""
    st = state_mod.empty_state()
    state_mod.record_page(
        st, doc_id="d1", doc_hash="dh", notebook="N", page_id="p1", page_hash="h-p1",
        note_path="Inbox/N/Note.md", status=state_mod.STATUS_COMMITTED, timestamp="t",
    )
    assert state_mod.existing_note_path(st, "d1", "p1") == "Inbox/N/Note.md"
    assert state_mod.is_page_current(st, "d1", "p1", "h-p1")
