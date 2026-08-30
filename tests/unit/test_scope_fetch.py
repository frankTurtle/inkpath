from __future__ import annotations

import pytest

from rmsync.config import Config, ConfigError
from rmsync.fetch import select_documents
from rmsync.remarkable import Item, RemarkableError, parse_index
from rmsync.scope import ScopeError, resolve_scope, resolve_watch_folder, valid_parent_ids


def folder(id_, name, parent=""):
    return Item(id=id_, hash="h", visible_name=name, type="CollectionType", parent=parent)


def doc(id_, name, parent):
    return Item(id=id_, hash="h", visible_name=name, type="DocumentType", parent=parent)


LIBRARY = [
    folder("F1", "Reading"),
    folder("F2", "Archive"),
    folder("F3", "2026", parent="F1"),   # nested inside the watch folder
    folder("F4", "Deep", parent="F3"),   # nested two levels
    doc("D1", "Antifragile", "F1"),
    doc("D2", "Seneca", "F3"),
    doc("D3", "Deep Note", "F4"),
    doc("D4", "Tax Receipts", "F2"),     # outside the watch folder
    doc("D5", "Scratch", "F1"),
]


def _cfg(**kw):
    base = dict(state_bucket="b", github_repo="o/r", watch_folder="Reading")
    base.update(kw)
    return Config(**base)


def test_scope_resolves_folder_by_name():
    assert resolve_watch_folder(LIBRARY, folder_name="reading") == "F1"


def test_scope_walks_nested_subfolders():
    assert valid_parent_ids(LIBRARY, "F1") == {"F1", "F3", "F4"}


def test_scope_duplicate_names_raise_rather_than_guess():
    dupes = [*LIBRARY, folder("F9", "Reading", parent="F2")]
    with pytest.raises(ScopeError, match="WatchFolderId"):
        resolve_watch_folder(dupes, folder_name="Reading")


def test_scope_folder_id_takes_precedence():
    assert resolve_watch_folder(LIBRARY, folder_name="Reading", folder_id="F2") == "F2"


def test_scope_missing_folder_raises():
    with pytest.raises(ScopeError, match="No folder named"):
        resolve_watch_folder(LIBRARY, folder_name="Nope")


def test_scope_unknown_folder_id_raises():
    with pytest.raises(ScopeError, match="not found"):
        resolve_watch_folder(LIBRARY, folder_id="nope")


def test_fetch_respects_watch_folder():
    parents = resolve_scope(LIBRARY, folder_name="Reading")
    got = {d.id for d in select_documents(LIBRARY, parents, _cfg())}
    assert got == {"D1", "D2", "D3", "D5"}
    assert "D4" not in got  # outside the folder


def test_fetch_include_notebooks_restricts_to_named_set():
    parents = resolve_scope(LIBRARY, folder_name="Reading")
    cfg = _cfg(include_notebooks=["Antifragile", "Seneca"])
    assert {d.id for d in select_documents(LIBRARY, parents, cfg)} == {"D1", "D2"}


def test_fetch_include_does_not_escape_watch_folder():
    """An include name living outside the folder must stay excluded."""
    parents = resolve_scope(LIBRARY, folder_name="Reading")
    cfg = _cfg(include_notebooks=["Tax Receipts"])
    assert select_documents(LIBRARY, parents, cfg) == []


def test_fetch_exclude_notebooks_drops_named_set():
    parents = resolve_scope(LIBRARY, folder_name="Reading")
    cfg = _cfg(exclude_notebooks=["Scratch"])
    assert {d.id for d in select_documents(LIBRARY, parents, cfg)} == {"D1", "D2", "D3"}


def test_fetch_include_and_exclude_together_raises():
    parents = resolve_scope(LIBRARY, folder_name="Reading")
    cfg = _cfg(include_notebooks=["A"], exclude_notebooks=["B"])
    with pytest.raises(ConfigError, match="mutually exclusive"):
        select_documents(LIBRARY, parents, cfg)


def test_fetch_skips_trashed_and_deleted():
    lib = [*LIBRARY, doc("D6", "Gone", "trash")]
    lib.append(Item(id="D7", hash="h", visible_name="Del", type="DocumentType",
                    parent="F1", deleted=True))
    parents = resolve_scope(lib, folder_name="Reading")
    got = {d.id for d in select_documents(lib, parents, _cfg())}
    assert "D6" not in got and "D7" not in got


# ------------------------------------------------------------------ config --


def test_config_validate_rejects_include_and_exclude():
    with pytest.raises(ConfigError, match="mutually exclusive"):
        Config(
            state_bucket="b", github_repo="o/r", watch_folder="R",
            include_notebooks=["a"], exclude_notebooks=["b"],
        ).validate()


def test_config_requires_a_watch_scope():
    """Running unscoped would OCR the entire library on the first run."""
    with pytest.raises(ConfigError, match="entire reMarkable library"):
        Config(state_bucket="b", github_repo="o/r").validate()


def test_config_rejects_unknown_batch_mode():
    with pytest.raises(ConfigError, match="BatchMode"):
        Config(state_bucket="b", github_repo="o/r", watch_folder="R",
               batch_mode="nonsense").validate()


# ----------------------------------------------------------- index parsing --


def test_parse_index_schema_3():
    blob = "3\n" + "a" * 64 + ":80000000:doc1:2:100\n"
    entries = parse_index(blob)
    assert entries[0].id == "doc1" and entries[0].subfiles == 2


def test_parse_index_schema_4():
    blob = "4\n0:root:1:100\n" + "b" * 64 + ":0:doc2:3:200\n"
    entries = parse_index(blob)
    assert len(entries) == 1 and entries[0].id == "doc2"


def test_parse_index_schema_4_count_mismatch_raises():
    blob = "4\n0:root:5:100\n" + "b" * 64 + ":0:doc2:3:200\n"
    with pytest.raises(RemarkableError, match="declared 5 entries"):
        parse_index(blob)


def test_parse_index_rejects_unknown_schema():
    with pytest.raises(RemarkableError, match="Unsupported index schema"):
        parse_index("9\nwhatever\n")


def test_parse_index_rejects_malformed_line():
    with pytest.raises(RemarkableError, match="Malformed"):
        parse_index("3\ntoo:few:fields\n")


# ------------------------------------------------------- multiple folders --


def test_scope_unions_several_folders_by_id():
    """Reading notes and journals are different subtrees, watched together."""
    parents = resolve_scope(LIBRARY, folder_ids=["F1", "F2"])
    assert parents == {"F1", "F3", "F4", "F2"}


def test_scope_unions_folders_by_name():
    parents = resolve_scope(LIBRARY, folder_names=["Reading", "Archive"])
    assert parents == {"F1", "F3", "F4", "F2"}


def test_ids_take_precedence_over_names():
    """A stale placeholder WatchFolder must not fail a run that uses ids."""
    parents = resolve_scope(LIBRARY, folder_names=["Nonexistent"], folder_ids=["F1"])
    assert parents == {"F1", "F3", "F4"}


def test_scope_single_value_still_works():
    assert resolve_scope(LIBRARY, folder_id="F1") == {"F1", "F3", "F4"}
    assert resolve_scope(LIBRARY, folder_name="Reading") == {"F1", "F3", "F4"}


def test_scope_requires_at_least_one_folder():
    with pytest.raises(ScopeError, match="must be set"):
        resolve_scope(LIBRARY)


def test_scope_one_bad_folder_fails_the_whole_run():
    """Better to fail loudly than silently sync a subset."""
    with pytest.raises(ScopeError, match="No folder named"):
        resolve_scope(LIBRARY, folder_names=["Reading", "Nope"])


def test_documents_from_both_folders_are_selected():
    parents = resolve_scope(LIBRARY, folder_ids=["F1", "F2"])
    got = {d.id for d in select_documents(LIBRARY, parents, _cfg())}
    assert got == {"D1", "D2", "D3", "D5", "D4"}


def test_config_accepts_comma_separated_folder_ids(monkeypatch):
    monkeypatch.setenv("STATE_BUCKET", "b")
    monkeypatch.setenv("GITHUB_REPO", "o/r")
    monkeypatch.setenv("WATCH_FOLDER_ID", "aaa , bbb")
    monkeypatch.setenv("WATCH_FOLDER", "")
    cfg = Config.from_env()
    assert cfg.watch_folder_ids == ["aaa", "bbb"]
