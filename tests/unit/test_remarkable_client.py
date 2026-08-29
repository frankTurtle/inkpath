"""reMarkable Cloud client: the reverse-engineered surface most likely to break."""

from __future__ import annotations

import json

import pytest

from rmsync.remarkable import RemarkableClient, RemarkableError

HASH_A = "a" * 64
HASH_B = "b" * 64
HASH_C = "c" * 64


class FakeResp:
    def __init__(self, status=200, body=b"", payload=None):
        self.status_code = status
        self.ok = 200 <= status < 300
        self.reason = "OK" if self.ok else "Error"
        self.content = body
        self.text = body.decode() if isinstance(body, bytes) else str(body)
        self._payload = payload

    def json(self):
        if self._payload is None:
            raise ValueError("no json")
        return self._payload


class FakeSession:
    """Routes by URL and records the rm-filename header the server validates."""

    def __init__(self, root=None, blobs=None):
        self.headers: dict = {}
        self._root = root
        self._blobs = blobs or {}
        self.filenames: list[str] = []

    def mount(self, *a, **kw):
        pass

    def get(self, url, headers=None, timeout=None, **kw):
        if url.endswith("/sync/v4/root"):
            if self._root is None:
                return FakeResp(500, b"boom")
            return FakeResp(200, payload=self._root)
        hash_ = url.rsplit("/", 1)[-1]
        self.filenames.append((headers or {}).get("rm-filename", ""))
        if hash_ not in self._blobs:
            return FakeResp(404, b"missing")
        return FakeResp(200, body=self._blobs[hash_])


def _client(session):
    c = RemarkableClient("token")
    c._session = session
    return c


def test_get_root_parses_hash_and_generation():
    s = FakeSession(root={"hash": HASH_A, "generation": 42, "schemaVersion": 4})
    assert _client(s).get_root() == (HASH_A, 42, 4)


def test_get_root_rejects_unsupported_schema():
    s = FakeSession(root={"hash": HASH_A, "generation": 1, "schemaVersion": 9})
    with pytest.raises(RemarkableError, match="Unsupported root schema"):
        _client(s).get_root()


def test_get_root_failure_is_loud():
    """A silent fetch failure is indistinguishable from 'no new pages'."""
    s = FakeSession(root=None)
    with pytest.raises(RemarkableError, match="Root fetch failed"):
        _client(s).get_root()


def test_get_blob_sends_required_rm_filename_header():
    """reMarkable validates rm-filename against the hash and rejects without it."""
    s = FakeSession(blobs={HASH_A: b"data"})
    assert _client(s).get_blob("doc.metadata", HASH_A) == b"data"
    assert s.filenames == ["doc.metadata"]


def test_get_blob_rejects_malformed_hash():
    with pytest.raises(RemarkableError, match="valid content hash"):
        _client(FakeSession()).get_blob("x.rm", "not-a-hash")


def test_list_items_resolves_metadata():
    meta = {
        "visibleName": "Antifragile",
        "type": "DocumentType",
        "parent": "F1",
        "lastModified": "123",
    }
    root_index = f"4\n0:root:1:10\n{HASH_B}:0:doc1:2:20\n".encode()
    doc_index = f"3\n{HASH_C}:0:doc1.metadata:0:5\n".encode()
    s = FakeSession(
        root={"hash": HASH_A, "generation": 1, "schemaVersion": 4},
        blobs={HASH_A: root_index, HASH_B: doc_index, HASH_C: json.dumps(meta).encode()},
    )
    items, cache = _client(s).list_items()
    assert len(items) == 1
    assert items[0].visible_name == "Antifragile"
    assert items[0].parent == "F1"
    assert cache["doc1"]["hash"] == HASH_B


def test_list_items_accepts_legacy_double_s_spelling():
    """The retired storage API spelled it `vissibleName`; read either."""
    meta = {"vissibleName": "Legacy", "type": "DocumentType", "parent": ""}
    root_index = f"4\n0:root:1:10\n{HASH_B}:0:doc1:2:20\n".encode()
    doc_index = f"3\n{HASH_C}:0:doc1.metadata:0:5\n".encode()
    s = FakeSession(
        root={"hash": HASH_A, "generation": 1, "schemaVersion": 4},
        blobs={HASH_A: root_index, HASH_B: doc_index, HASH_C: json.dumps(meta).encode()},
    )
    items, _ = _client(s).list_items()
    assert items[0].visible_name == "Legacy"


def test_list_items_uses_cache_to_avoid_refetch():
    """An unchanged document needs no metadata refetch - two HTTP calls, not 2N."""
    root_index = f"4\n0:root:1:10\n{HASH_B}:0:doc1:2:20\n".encode()
    s = FakeSession(
        root={"hash": HASH_A, "generation": 1, "schemaVersion": 4},
        blobs={HASH_A: root_index},   # doc blobs absent: a refetch would 404
    )
    cache = {
        "doc1": {"hash": HASH_B, "visibleName": "Cached", "type": "DocumentType", "parent": "F1"}
    }
    items, _ = _client(s).list_items(cache)
    assert items[0].visible_name == "Cached"
    assert s.filenames == ["root.docSchema"]   # only the root index was fetched


def test_list_items_skips_documents_without_metadata():
    root_index = f"4\n0:root:1:10\n{HASH_B}:0:doc1:2:20\n".encode()
    doc_index = b"3\n"                      # no .metadata entry
    s = FakeSession(
        root={"hash": HASH_A, "generation": 1, "schemaVersion": 4},
        blobs={HASH_A: root_index, HASH_B: doc_index},
    )
    items, _ = _client(s).list_items()
    assert items == []


def test_get_metadata_raises_on_bad_json():
    doc_index_entries = f"3\n{HASH_C}:0:doc1.metadata:0:5\n"
    s = FakeSession(blobs={HASH_B: doc_index_entries.encode(), HASH_C: b"{bad"})
    c = _client(s)
    entries = c.get_entries("doc1", HASH_B)
    with pytest.raises(RemarkableError, match="Unparseable metadata"):
        c.get_metadata("doc1", entries)
