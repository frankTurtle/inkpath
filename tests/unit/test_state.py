from __future__ import annotations

import pytest
from botocore.exceptions import ClientError

from rmsync import state as state_mod


class FakeS3:
    def __init__(self) -> None:
        self.store: dict[str, bytes] = {}

    def get_object(self, Bucket: str, Key: str):  # noqa: N803
        if Key not in self.store:
            raise ClientError({"Error": {"Code": "NoSuchKey"}}, "GetObject")
        body = self.store[Key]

        class B:
            @staticmethod
            def read() -> bytes:
                return body

        return {"Body": B()}

    def put_object(self, Bucket: str, Key: str, Body: bytes, **kw):  # noqa: N803
        self.store[Key] = Body


@pytest.fixture(autouse=True)
def fake_s3(monkeypatch):
    fake = FakeS3()
    monkeypatch.setattr(state_mod, "_client", lambda: fake)
    return fake


def test_state_first_run_empty():
    st = state_mod.load_state("test-bucket")
    assert st["docs"] == {}
    assert st["batch"] == {"queue": [], "pendingJobs": []}


def test_state_round_trip_preserves_shape():
    st = state_mod.empty_state()
    state_mod.record_page(
        st,
        doc_id="doc1",
        doc_hash="h1",
        notebook="Reading",
        page_id="p1",
        page_hash="ph1",
        note_path="Inbox/a.md",
        status=state_mod.STATUS_COMMITTED,
        timestamp="2026-08-29T00:00:00Z",
    )
    state_mod.save_state(st, "test-bucket")
    back = state_mod.load_state("test-bucket")
    assert back["docs"]["doc1"]["pages"]["p1"]["notePath"] == "Inbox/a.md"
    assert back["docs"]["doc1"]["hash"] == "h1"


def test_is_page_current_only_for_matching_hash():
    st = state_mod.empty_state()
    state_mod.record_page(
        st, doc_id="d", doc_hash="dh", notebook="N", page_id="p", page_hash="v1",
        note_path="a.md", status=state_mod.STATUS_COMMITTED, timestamp="t",
    )
    assert state_mod.is_page_current(st, "d", "p", "v1")
    assert not state_mod.is_page_current(st, "d", "p", "v2")


def test_pending_batch_is_a_third_bucket():
    """A pending page is neither committed nor safe to re-queue."""
    st = state_mod.empty_state()
    state_mod.record_page(
        st, doc_id="d", doc_hash="dh", notebook="N", page_id="p", page_hash="v1",
        note_path="", status=state_mod.STATUS_PENDING_BATCH, timestamp="t",
    )
    assert state_mod.is_page_pending_batch(st, "d", "p", "v1")
    assert not state_mod.is_page_current(st, "d", "p", "v1")


def test_doc_hash_advances_only_when_all_pages_committed():
    st = state_mod.empty_state()
    state_mod.record_page(
        st, doc_id="d", doc_hash="dh", notebook="N", page_id="p1", page_hash="a",
        note_path="a.md", status=state_mod.STATUS_COMMITTED, timestamp="t",
    )
    state_mod.record_page(
        st, doc_id="d", doc_hash="dh", notebook="N", page_id="p2", page_hash="b",
        note_path="", status=state_mod.STATUS_PENDING_BATCH, timestamp="t",
    )
    # One page still pending, so the document must not look finished.
    assert st["docs"]["d"].get("hash") != "dh"


def test_vocabulary_dedupes():
    st = state_mod.empty_state()
    state_mod.learn_vocabulary(st, ["a", "b", "a"], "Title")
    state_mod.learn_vocabulary(st, ["b", "c"], "Title")
    assert st["tagVocabulary"] == ["a", "b", "c"]
    assert st["noteTitles"] == ["Title"]


def test_corrupt_state_raises_rather_than_resyncing_everything(fake_s3):
    fake_s3.store["state.json"] = b"{not json"
    with pytest.raises(ValueError):
        state_mod.load_state("test-bucket")
