"""End-to-end run() with every boundary faked: no AWS, no GitHub, no reMarkable."""

from __future__ import annotations

import json

import pytest

from rmsync import app as app_mod
from rmsync import state as state_mod
from rmsync.config import Config
from rmsync.providers import ProviderResult
from rmsync.remarkable import Item, RawEntry

HASH = "a" * 64


def _cfg(**kw):
    base = dict(
        state_bucket="bucket", github_repo="o/vault", watch_folder="Reading",
        vault_note_path="Inbox/reMarkable", max_pages_per_run=20,
        blank_page_threshold=3, render_width=1400, min_text_length=20,
    )
    base.update(kw)
    return Config(**base)


class FakeRmClient:
    def __init__(self, page_bytes, pages=("p1",)):
        self._page_bytes = page_bytes
        self._pages = pages

    def list_items(self, cache=None):
        items = [
            Item(id="F1", hash="fh", visible_name="Reading", type="CollectionType", parent=""),
            Item(id="d1", hash="dh", visible_name="Antifragile", type="DocumentType", parent="F1"),
        ]
        return items, {}

    def get_entries(self, entry_id, hash_):
        return [
            RawEntry(hash=f"h-{p}", type=0, id=f"{p}.rm", subfiles=0, size=10)
            for p in self._pages
        ]

    def get_content(self, doc_id, entries):
        return {"cPages": {"pages": [{"id": p} for p in self._pages]}}

    def get_blob(self, file_name, hash_):
        return self._page_bytes


class FakeVault:
    def __init__(self):
        self.committed: dict[str, bytes] = {}

    def put_file(self, path, content, message):
        self.committed[path] = content
        return "sha"


@pytest.fixture
def wired(monkeypatch, page_rm):
    """Patch every external boundary and capture what run() persists."""
    saved: dict = {}
    store: dict[str, bytes] = {}

    monkeypatch.setattr(state_mod, "load_state", lambda b, k="state.json": saved.get("state") or state_mod.empty_state())
    monkeypatch.setattr(state_mod, "save_state", lambda st, b, k="state.json": saved.__setitem__("state", st))
    monkeypatch.setattr(app_mod.state_mod, "load_state", lambda b, k="state.json": saved.get("state") or state_mod.empty_state())
    monkeypatch.setattr(app_mod.state_mod, "save_state", lambda st, b, k="state.json": saved.__setitem__("state", st))
    monkeypatch.setattr(app_mod, "get_secret", lambda name, prefix="/rmsync", required=True: "secret")
    monkeypatch.setattr(app_mod, "get_user_token", lambda t: "usertoken")
    monkeypatch.setattr(app_mod, "stage_png", lambda b, k, p: store.__setitem__(k, p))
    monkeypatch.setattr(app_mod, "load_png", lambda b, k: store[k])

    client = FakeRmClient(page_rm)
    monkeypatch.setattr(app_mod, "RemarkableClient", lambda token: client)

    vault = FakeVault()
    provider = type(
        "P", (), {"calls": 0,
                  "extract_and_tag": lambda self, png, tags, titles=None: ProviderResult(
                      text="Antifragility is not resilience." * 3,
                      tags=["book-notes", "Epistemology"], title="Via negativa")}
    )()

    original_init = app_mod.Runner.__init__

    def patched_init(self, cfg):
        original_init(self, cfg)
        self._vault = vault
        self._provider = provider

    monkeypatch.setattr(app_mod.Runner, "__init__", patched_init)
    return {"saved": saved, "vault": vault, "store": store}


def test_run_commits_a_note_and_saves_state(wired):
    stats = app_mod.run({}, _cfg())

    assert stats["commits"] == 1
    assert stats["modelCalls"] == 1
    assert stats["errors"] == 0

    (path, body), = wired["vault"].committed.items()
    assert path == "Inbox/reMarkable/Antifragile/Via negativa.md"
    text = body.decode()
    assert text.startswith("---\n")
    assert "source: reMarkable" in text
    assert "rm_doc_id: d1" in text
    # Tags were sanitized on the way through.
    assert "epistemology" in text and "Epistemology" not in text

    st = wired["saved"]["state"]
    assert st["docs"]["d1"]["pages"]["p1"]["status"] == "committed"


def test_second_run_is_a_no_op(wired):
    """Unchanged documents produce no work, no model call, and no commit."""
    app_mod.run({}, _cfg())
    first = dict(wired["vault"].committed)

    stats = app_mod.run({}, _cfg())
    assert stats["modelCalls"] == 0
    assert stats["commits"] == 0
    assert stats["pagesConsidered"] == 0
    assert wired["vault"].committed == first     # nothing re-committed


def test_run_returns_summary_json_serialisable(wired):
    stats = app_mod.run({}, _cfg())
    json.dumps(stats)          # RUN_SUMMARY must be loggable
    assert "durationMs" in stats


def test_lambda_handler_wraps_run(wired, monkeypatch):
    for k, v in {
        "STATE_BUCKET": "bucket", "GITHUB_REPO": "o/vault",
        "WATCH_FOLDER": "Reading", "VAULT_NOTE_PATH": "Inbox/reMarkable",
    }.items():
        monkeypatch.setenv(k, v)
    resp = app_mod.lambda_handler({}, None)
    assert resp["statusCode"] == 200
    assert json.loads(resp["body"])["commits"] == 1
