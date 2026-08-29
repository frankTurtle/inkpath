"""Diff and download logic - the part that decides what costs money."""

from __future__ import annotations

from rmsync import state as state_mod
from rmsync.config import Config
from rmsync.fetch import diff_pages, download_pages
from rmsync.remarkable import Item, RawEntry


def _cfg(**kw):
    base = dict(state_bucket="b", github_repo="o/r", watch_folder="R", max_pages_per_run=20)
    base.update(kw)
    return Config(**base)


def _doc(doc_id="d1", hash_="dh1", name="Reading"):
    return Item(id=doc_id, hash=hash_, visible_name=name, type="DocumentType", parent="F1")


class FakeClient:
    """Serves a document whose collection holds N `.rm` page blobs."""

    def __init__(self, pages: dict[str, str], content: dict | None = None):
        # pages: pageId -> blobHash
        self.pages = pages
        self.content = content
        self.entry_calls = 0
        self.blob_calls: list[str] = []

    def get_entries(self, entry_id, hash_):
        self.entry_calls += 1
        entries = [
            RawEntry(hash=h, type=0, id=f"{pid}.rm", subfiles=0, size=10)
            for pid, h in self.pages.items()
        ]
        entries.append(RawEntry(hash="ch", type=0, id=f"{entry_id}.content", subfiles=0, size=5))
        return entries

    def get_content(self, doc_id, entries):
        return self.content

    def get_blob(self, file_name, hash_):
        self.blob_calls.append(file_name)
        return b"BLOB:" + file_name.encode()


def test_unchanged_document_costs_zero_http_calls():
    """The fast path: an unchanged collection hash means nothing inside changed."""
    st = state_mod.empty_state()
    st["docs"]["d1"] = {"hash": "dh1", "pages": {}}
    client = FakeClient({"p1": "h1"})
    pending, skipped = diff_pages(client, [_doc()], st, _cfg())

    assert pending == []
    assert client.entry_calls == 0     # never even opened the document


def test_new_document_yields_all_pages():
    st = state_mod.empty_state()
    client = FakeClient({"p1": "h1", "p2": "h2"})
    pending, _ = diff_pages(client, [_doc()], st, _cfg())
    assert {p.page_id for p in pending} == {"p1", "p2"}


def test_fetch_skips_unchanged_version():
    """Only the edited page is re-processed, not the whole notebook."""
    st = state_mod.empty_state()
    state_mod.record_page(
        st, doc_id="d1", doc_hash="old", notebook="Reading", page_id="p1", page_hash="h1",
        note_path="a.md", status=state_mod.STATUS_COMMITTED, timestamp="t",
    )
    # p1 unchanged, p2 is new -> document hash differs, so the doc is opened.
    client = FakeClient({"p1": "h1", "p2": "h2"})
    pending, _ = diff_pages(client, [_doc(hash_="dh-new")], st, _cfg())
    assert [p.page_id for p in pending] == ["p2"]


def test_edited_page_is_reprocessed():
    st = state_mod.empty_state()
    state_mod.record_page(
        st, doc_id="d1", doc_hash="old", notebook="Reading", page_id="p1", page_hash="h1",
        note_path="a.md", status=state_mod.STATUS_COMMITTED, timestamp="t",
    )
    client = FakeClient({"p1": "h1-EDITED"})
    pending, _ = diff_pages(client, [_doc(hash_="dh-new")], st, _cfg())
    assert [p.page_hash for p in pending] == ["h1-EDITED"]


def test_pending_batch_page_is_not_requeued():
    st = state_mod.empty_state()
    state_mod.record_page(
        st, doc_id="d1", doc_hash="old", notebook="Reading", page_id="p1", page_hash="h1",
        note_path="", status=state_mod.STATUS_PENDING_BATCH, timestamp="t",
    )
    client = FakeClient({"p1": "h1"})
    pending, skipped = diff_pages(client, [_doc(hash_="dh-new")], st, _cfg())
    assert pending == [] and skipped == 1


def test_page_order_follows_content_blob():
    """cPages order decides page numbering, not blob-hash sort order."""
    st = state_mod.empty_state()
    content = {"cPages": {"pages": [{"id": "pB"}, {"id": "pA"}]}}
    client = FakeClient({"pA": "hA", "pB": "hB"}, content=content)
    pending, _ = diff_pages(client, [_doc()], st, _cfg())
    assert [p.page_id for p in pending] == ["pB", "pA"]
    assert [p.page_index for p in pending] == [0, 1]


def test_page_order_falls_back_to_sorted_when_content_missing():
    st = state_mod.empty_state()
    client = FakeClient({"pB": "hB", "pA": "hA"}, content=None)
    pending, _ = diff_pages(client, [_doc()], st, _cfg())
    assert [p.page_id for p in pending] == ["pA", "pB"]


def test_fetch_caps_at_max_pages():
    st = state_mod.empty_state()
    client = FakeClient({f"p{i}": f"h{i}" for i in range(10)})
    pending, _ = diff_pages(client, [_doc()], st, _cfg())
    capped = download_pages(client, pending, _cfg(max_pages_per_run=3))
    assert len(capped) == 3
    assert len(client.blob_calls) == 3      # uncapped pages are never downloaded


def test_download_attaches_blob_data():
    st = state_mod.empty_state()
    client = FakeClient({"p1": "h1"})
    pending, _ = diff_pages(client, [_doc()], st, _cfg())
    pages = download_pages(client, pending, _cfg())
    assert pages[0].data == b"BLOB:p1.rm"
