"""Resolve which reMarkable folders are in scope.

reMarkable has no server-side folder query - `list_items()` returns the entire
flat library and every item carries a `parent` pointing at its containing
folder's id. Scoping is therefore done client-side against that flat list.
"""

from __future__ import annotations

import logging

from .remarkable import TYPE_COLLECTION, Item

logger = logging.getLogger(__name__)


class ScopeError(RuntimeError):
    """The configured watch folder could not be resolved."""


def resolve_watch_folder(
    items: list[Item], *, folder_name: str = "", folder_id: str = ""
) -> str:
    """Return the id of the watched folder.

    `folder_id` wins when set. Matching by name is case-insensitive, and
    ambiguity is an error rather than a coin flip: reMarkable permits duplicate
    folder names under different parents, and silently picking one would sync
    the wrong notebooks with no visible failure.
    """
    if folder_id:
        match = next((i for i in items if i.id == folder_id), None)
        if match is None:
            raise ScopeError(
                f"WatchFolderId {folder_id!r} not found in the library. "
                "Check the id, or clear it to match by name instead."
            )
        if not match.is_folder:
            raise ScopeError(f"WatchFolderId {folder_id!r} is a {match.type}, not a folder")
        return match.id

    if not folder_name:
        raise ScopeError("One of WatchFolder or WatchFolderId must be set")

    wanted = folder_name.strip().lower()
    matches = [
        i
        for i in items
        if i.type == TYPE_COLLECTION and i.visible_name.strip().lower() == wanted and not i.in_trash
    ]
    if not matches:
        raise ScopeError(
            f"No folder named {folder_name!r} found on reMarkable. Folder names are "
            "matched exactly (case-insensitively); check for a rename or a typo."
        )
    if len(matches) > 1:
        raise ScopeError(
            f"{len(matches)} folders are named {folder_name!r} "
            f"(ids: {', '.join(m.id for m in matches)}). reMarkable allows duplicate "
            "folder names, so set WatchFolderId to the one you want instead."
        )
    return matches[0].id


def valid_parent_ids(items: list[Item], root_folder_id: str) -> set[str]:
    """The watched folder plus every folder nested inside it, transitively."""
    valid = {root_folder_id}
    folders = [i for i in items if i.type == TYPE_COLLECTION and not i.in_trash]
    changed = True
    while changed:
        changed = False
        for folder in folders:
            if folder.id not in valid and folder.parent in valid:
                valid.add(folder.id)
                changed = True
    return valid


def resolve_scope(
    items: list[Item],
    *,
    folder_name: str = "",
    folder_id: str = "",
    folder_names: list[str] | None = None,
    folder_ids: list[str] | None = None,
) -> set[str]:
    """Resolve one or more watch folders to the union of their in-scope parents.

    Several folders can be watched at once (say reading notes and journals) and
    they need not be siblings; each is expanded to its own subtree and the
    results are unioned.
    """
    names = list(folder_names or ([folder_name] if folder_name else []))
    ids = list(folder_ids or ([folder_id] if folder_id else []))
    if not names and not ids:
        raise ScopeError("One of WatchFolder or WatchFolderId must be set")

    # Ids win outright when present, rather than being unioned with names. A
    # deployment that has moved to ids often still carries a leftover
    # placeholder name, and unioning would make that stale value fail the run.
    if ids:
        if names:
            logger.info("WatchFolderId is set; ignoring WatchFolder %s", names)
        resolved = [(resolve_watch_folder(items, folder_id=f), f) for f in ids]
    else:
        resolved = [(resolve_watch_folder(items, folder_name=n), n) for n in names]

    parents: set[str] = set()
    roots: list[str] = []
    for root, _requested in resolved:
        roots.append(root)
        parents |= valid_parent_ids(items, root)

    logger.info(
        "%d watch folder(s) %s resolve to %d in-scope folder(s)",
        len(roots),
        [r[:8] for r in roots],
        len(parents),
    )
    return parents


def resolve_scope_map(
    items: list[Item],
    *,
    folder_names: list[str] | None = None,
    folder_ids: list[str] | None = None,
) -> dict[str, str]:
    """Map every in-scope folder id to the watch root it descends from.

    Routing a document to a vault destination needs to know *which* watched
    folder it came from, which a flat set of parent ids cannot answer.
    """
    names = list(folder_names or [])
    ids = list(folder_ids or [])
    if not names and not ids:
        raise ScopeError("One of WatchFolder or WatchFolderId must be set")

    if ids:
        if names:
            logger.info("WatchFolderId is set; ignoring WatchFolder %s", names)
        roots = [resolve_watch_folder(items, folder_id=f) for f in ids]
    else:
        roots = [resolve_watch_folder(items, folder_name=n) for n in names]

    mapping: dict[str, str] = {}
    for root in roots:
        for parent in valid_parent_ids(items, root):
            # First root wins if subtrees somehow overlap, so a document is
            # never routed to two different destinations.
            mapping.setdefault(parent, root)
    logger.info(
        "%d watch folder(s) resolve to %d in-scope folder(s)", len(roots), len(mapping)
    )
    return mapping
