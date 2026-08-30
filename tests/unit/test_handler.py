"""Handler-level tests, including the invariant that loses notes if broken:
state must never advance past a failed commit.
"""

from __future__ import annotations

from rmsync import state as state_mod
from rmsync.app import Runner
from rmsync.commit import CommitError
from rmsync.config import Config
from rmsync.fetch import PageRef
from rmsync.providers import ProviderResult


def _cfg(**kw):
    base = dict(
        state_bucket="b", github_repo="o/r", watch_folder="Reading",
        min_text_length=20, blank_page_threshold=3, render_width=1400,
    )
    base.update(kw)
    return Config(**base)


class StubProvider:
    def __init__(self, text="x" * 60):
        self.calls = 0
        self._text = text

    def extract_and_tag(self, image_bytes, existing_tags, existing_titles=None):
        self.calls += 1
        return ProviderResult(text=self._text, tags=["book-notes"], title="A Title")


class StubVault:
    def __init__(self, fail=False):
        self.fail = fail
        self.committed: list[str] = []

    def put_file(self, path, content, message):
        if self.fail:
            raise CommitError("GitHub is down")
        self.committed.append(path)
        return "sha"


def _runner(cfg, provider, vault):
    r = Runner(cfg)
    r._provider = provider
    r._vault = vault
    return r


def _page(data, page_id="p1"):
    return PageRef(
        doc_id="d1", doc_hash="dh", notebook="Reading", page_id=page_id,
        page_hash=f"h-{page_id}", page_index=0, data=data,
    )


def test_happy_path_commits_and_records_state(page_rm):
    st = state_mod.empty_state()
    vault = StubVault()
    r = _runner(_cfg(), StubProvider(), vault)
    r.process_sync(st, [_page(page_rm)])

    assert len(vault.committed) == 1
    assert vault.committed[0].endswith(".md")
    rec = state_mod.page_record(st, "d1", "p1")
    assert rec["status"] == state_mod.STATUS_COMMITTED
    assert r.stats["commits"] == 1 and r.stats["modelCalls"] == 1
    # Vocabulary grows so later pages reuse tags instead of inventing them.
    assert "book-notes" in st["tagVocabulary"]


def test_state_not_written_on_commit_failure(page_rm):
    """The PRP's most important ordering rule."""
    st = state_mod.empty_state()
    r = _runner(_cfg(), StubProvider(), StubVault(fail=True))
    r.process_sync(st, [_page(page_rm)])

    assert state_mod.page_record(st, "d1", "p1") is None
    assert st["docs"] == {}
    assert r.stats["errors"] == 1 and r.stats["commits"] == 0


def test_blank_page_costs_no_model_call(blank_rm):
    st = state_mod.empty_state()
    provider, vault = StubProvider(), StubVault()
    r = _runner(_cfg(), provider, vault)
    r.process_sync(st, [_page(blank_rm)])

    assert provider.calls == 0        # cheapest optimisation, fires often
    assert vault.committed == []
    assert r.stats["blankSkipped"] == 1
    # Recorded as done so it is never re-rendered.
    assert state_mod.is_page_current(st, "d1", "p1", "h-p1")


def test_one_bad_page_does_not_lose_the_others(page_rm):
    st = state_mod.empty_state()

    class FlakyProvider(StubProvider):
        def extract_and_tag(self, image_bytes, existing_tags, existing_titles=None):
            self.calls += 1
            if self.calls == 1:
                raise RuntimeError("model blew up")
            return ProviderResult(text="y" * 60, tags=["t"], title="T2")

    r = _runner(_cfg(), FlakyProvider(), StubVault())
    r.process_sync(st, [_page(page_rm, "p1"), _page(page_rm, "p2")])

    assert r.stats["errors"] == 1
    assert r.stats["commits"] == 1
    assert state_mod.page_record(st, "d1", "p1") is None   # retried next poll
    assert state_mod.page_record(st, "d1", "p2") is not None


def test_dry_run_makes_no_commits(page_rm):
    st = state_mod.empty_state()
    vault = StubVault()
    r = _runner(_cfg(dry_run=True), StubProvider(), vault)
    r.process_sync(st, [_page(page_rm)])
    assert vault.committed == []
    assert r.stats["commits"] == 1     # recorded as processed, nothing pushed


def test_low_text_page_attaches_png(page_rm):
    st = state_mod.empty_state()
    vault = StubVault()
    r = _runner(_cfg(), StubProvider(text="hi"), vault)
    r.process_sync(st, [_page(page_rm)])
    # Attachment committed alongside the note rather than an empty note alone.
    assert any(p.endswith(".png") for p in vault.committed)
    assert any(p.endswith(".md") for p in vault.committed)


def test_already_committed_page_is_not_recommitted(page_rm):
    st = state_mod.empty_state()
    state_mod.record_page(
        st, doc_id="d1", doc_hash="dh", notebook="Reading", page_id="p1", page_hash="h-p1",
        note_path="Inbox/x.md", status=state_mod.STATUS_COMMITTED, timestamp="t",
    )
    vault = StubVault()
    r = _runner(_cfg(), StubProvider(), vault)
    r.process_sync(st, [_page(page_rm)])
    assert vault.committed == []


def test_two_pages_with_the_same_title_do_not_overwrite(page_rm):
    """Real bug: two Walden pages both titled "Walden Thoreau Quotes" resolved
    to one path, and the second silently overwrote the first."""
    st = state_mod.empty_state()
    vault = StubVault()
    r = _runner(_cfg(), StubProvider(), vault)
    r.process_sync(st, [_page(page_rm, "p1"), _page(page_rm, "p2")])

    paths = [p for p in vault.committed if p.endswith(".md")]
    assert len(paths) == 2
    assert len(set(paths)) == 2, f"pages overwrote each other: {paths}"
    assert any("(p2)" in p for p in paths)


def test_recommitting_the_same_page_reuses_its_path(page_rm):
    """An edited page must update its own note, not spawn a duplicate."""
    st = state_mod.empty_state()
    vault = StubVault()
    r = _runner(_cfg(), StubProvider(), vault)
    r.process_sync(st, [_page(page_rm, "p1")])
    first = list(vault.committed)

    edited = _page(page_rm, "p1")
    edited.page_hash = "h-p1-EDITED"
    r.process_sync(st, [edited])
    assert vault.committed[len(first):] == first  # same path rewritten
