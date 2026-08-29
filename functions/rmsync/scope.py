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
    items: list[Item], *, folder_name: str = "", folder_id: str = ""
) -> set[str]:
    """Convenience wrapper: name/id -> full set of in-scope parent ids."""
    root = resolve_watch_folder(items, folder_name=folder_name, folder_id=folder_id)
    parents = valid_parent_ids(items, root)
    logger.info("Watch folder %s resolves to %d in-scope folder(s)", root[:8], len(parents))
    return parents
