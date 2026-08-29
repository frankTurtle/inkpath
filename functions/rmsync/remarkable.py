"""Minimal reMarkable Cloud client (sync v3/v4, root-hash protocol).

reMarkable publishes no official third-party API. Everything here is
reverse-engineered from the actively-maintained rmapi-js client, so failures are
raised loudly rather than swallowed - a silent fetch failure is indistinguishable
from "no new pages", which is exactly the bug that loses notes quietly.

Protocol shape::

    GET /sync/v4/root                -> {"hash", "generation", "schemaVersion"}
    GET /sync/v3/files/<hash>        -> blob bytes  (rm-filename header REQUIRED)

An index blob is newline-delimited text::

    <schemaVersion>            # "3" or "4"
    0:<id>:<count>:<size>      # schema 4 only - info line
    <hash>:<type>:<id>:<subfiles>:<size>
    ...

The root index lists documents; each document's index lists its own files
(`<id>.metadata`, `<id>.content`, `<pageUuid>.rm`, ...).
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from typing import Any

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

logger = logging.getLogger(__name__)

RAW_HOST = "https://eu.tectonic.remarkable.com"
HTTP_TIMEOUT = 30
_HASH_RE = re.compile(r"^[0-9a-f]{64}$")

TYPE_DOCUMENT = "DocumentType"
TYPE_COLLECTION = "CollectionType"
TYPE_TEMPLATE = "TemplateType"

TRASH_PARENT = "trash"


class RemarkableError(RuntimeError):
    """A reMarkable Cloud call failed or returned something unparseable."""


@dataclass(frozen=True)
class RawEntry:
    """One line of an index blob."""

    hash: str
    type: int
    id: str
    subfiles: int
    size: int


@dataclass
class Item:
    """A document or folder, with its metadata resolved."""

    id: str
    hash: str
    visible_name: str
    type: str
    parent: str
    last_modified: str = ""
    deleted: bool = False

    @property
    def is_folder(self) -> bool:
        return self.type == TYPE_COLLECTION

    @property
    def is_document(self) -> bool:
        return self.type == TYPE_DOCUMENT

    @property
    def in_trash(self) -> bool:
        return self.parent == TRASH_PARENT


def _parse_entry_line(line: str) -> RawEntry:
    parts = line.split(":")
    if len(parts) != 5:
        raise RemarkableError(f"Malformed index entry line: {line!r}")
    hash_, type_, id_, subfiles, size = parts
    if type_ not in {"0", "80000000"}:
        raise RemarkableError(f"Unknown entry type {type_!r} in line {line!r}")
    try:
        return RawEntry(
            hash=hash_,
            type=int(type_),
            id=id_,
            subfiles=int(subfiles),
            size=int(size),
        )
    except ValueError as exc:
        raise RemarkableError(f"Malformed index entry line: {line!r}") from exc


def parse_index(text: str) -> list[RawEntry]:
    """Parse an index blob into entries. Handles schema 3 and 4."""
    lines = text.strip("\n").split("\n")
    if not lines or not lines[0]:
        raise RemarkableError("Empty index blob")
    version, rest = lines[0], lines[1:]
    if version == "3":
        return [_parse_entry_line(line) for line in rest if line]
    if version == "4":
        if not rest:
            raise RemarkableError("Missing info line for schema version 4")
        info, entry_lines = rest[0], [line for line in rest[1:] if line]
        info_parts = info.split(":")
        if len(info_parts) != 4 or info_parts[0] != "0":
            raise RemarkableError(f"Malformed schema 4 info line: {info!r}")
        expected = int(info_parts[2])
        entries = [_parse_entry_line(line) for line in entry_lines]
        if expected != len(entries):
            raise RemarkableError(
                f"Schema 4 index declared {expected} entries but contained {len(entries)}"
            )
        return entries
    raise RemarkableError(f"Unsupported index schema version {version!r}")


class RemarkableClient:
    """Read-only client. This pipeline never writes to reMarkable."""

    def __init__(self, user_token: str, *, raw_host: str = RAW_HOST) -> None:
        self._host = raw_host.rstrip("/")
        self._session = requests.Session()
        self._session.headers.update({"Authorization": f"Bearer {user_token}"})
        retry = Retry(
            total=3,
            backoff_factor=0.5,
            status_forcelist=(429, 500, 502, 503, 504),
            allowed_methods=frozenset({"GET"}),
        )
        self._session.mount("https://", HTTPAdapter(max_retries=retry))

    # ------------------------------------------------------------ transport --

    def get_root(self) -> tuple[str, int, int]:
        """Return (hash, generation, schemaVersion) for the vault root."""
        resp = self._session.get(f"{self._host}/sync/v4/root", timeout=HTTP_TIMEOUT)
        if not resp.ok:
            raise RemarkableError(
                f"Root fetch failed ({resp.status_code} {resp.reason}): {resp.text[:200]}"
            )
        try:
            data = resp.json()
            schema = int(data["schemaVersion"])
        except (ValueError, KeyError) as exc:
            raise RemarkableError(f"Unparseable root response: {resp.text[:200]}") from exc
        if schema not in (3, 4):
            raise RemarkableError(f"Unsupported root schema version {schema}")
        return str(data["hash"]), int(data["generation"]), schema

    def get_blob(self, file_name: str, hash_: str) -> bytes:
        """Fetch a blob by hash.

        The `rm-filename` header is required - reMarkable validates it against
        the hash and rejects the request without it.
        """
        if not _HASH_RE.match(hash_):
            raise RemarkableError(f"Not a valid content hash: {hash_!r}")
        resp = self._session.get(
            f"{self._host}/sync/v3/files/{hash_}",
            headers={"rm-filename": file_name},
            timeout=HTTP_TIMEOUT,
        )
        if not resp.ok:
            raise RemarkableError(
                f"Blob fetch failed for {file_name} ({resp.status_code} {resp.reason})"
            )
        return resp.content

    def get_text(self, file_name: str, hash_: str) -> str:
        return self.get_blob(file_name, hash_).decode("utf-8")

    # --------------------------------------------------------------- reads --

    def get_entries(self, entry_id: str, hash_: str) -> list[RawEntry]:
        """Entries of an index blob (the root index, or a document's file list)."""
        return parse_index(self.get_text(f"{entry_id}.docSchema", hash_))

    def get_metadata(self, doc_id: str, entries: list[RawEntry]) -> dict[str, Any] | None:
        """Fetch and parse a document's `.metadata` blob."""
        target = f"{doc_id}.metadata"
        for entry in entries:
            if entry.id == target:
                raw = self.get_text(target, entry.hash)
                try:
                    return json.loads(raw)
                except ValueError as exc:
                    raise RemarkableError(f"Unparseable metadata for {doc_id}") from exc
        return None

    def get_content(self, doc_id: str, entries: list[RawEntry]) -> dict[str, Any] | None:
        """Fetch and parse a document's `.content` blob (page order lives here)."""
        target = f"{doc_id}.content"
        for entry in entries:
            if entry.id == target:
                try:
                    return json.loads(self.get_text(target, entry.hash))
                except ValueError:
                    logger.warning("Unparseable .content for %s; falling back to file order", doc_id)
                    return None
        return None

    def list_items(
        self, metadata_cache: dict[str, dict[str, Any]] | None = None
    ) -> tuple[list[Item], dict[str, dict[str, Any]]]:
        """List every item in the library, resolving metadata.

        reMarkable has no server-side "list documents under this folder" call:
        the root index is the whole flat library and scoping is done client-side.

        `metadata_cache` maps docId -> {hash, visibleName, type, parent,
        lastModified}. An entry whose hash is unchanged since the last run needs
        no metadata refetch, which keeps a steady-state poll down to two HTTP
        calls instead of two per document.
        """
        cache = dict(metadata_cache or {})
        root_hash, generation, _schema = self.get_root()
        logger.info("reMarkable root hash=%s generation=%s", root_hash[:12], generation)

        items: list[Item] = []
        fresh_cache: dict[str, dict[str, Any]] = {}
        refetched = 0

        for entry in self.get_entries("root", root_hash):
            cached = cache.get(entry.id)
            if cached and cached.get("hash") == entry.hash:
                meta = cached
            else:
                doc_entries = self.get_entries(entry.id, entry.hash)
                raw_meta = self.get_metadata(entry.id, doc_entries)
                refetched += 1
                if raw_meta is None:
                    logger.debug("No .metadata for %s; skipping", entry.id)
                    continue
                meta = {
                    "hash": entry.hash,
                    # The current sync API spells this `visibleName`. The
                    # double-s `vissibleName` belongs to the retired
                    # document-storage API; read either so a protocol change in
                    # that direction does not silently drop every notebook.
                    "visibleName": raw_meta.get("visibleName")
                    or raw_meta.get("vissibleName")
                    or "",
                    "type": raw_meta.get("type", ""),
                    "parent": raw_meta.get("parent", "") or "",
                    "lastModified": raw_meta.get("lastModified", "") or "",
                    "deleted": bool(raw_meta.get("deleted") or False),
                }
            fresh_cache[entry.id] = meta
            items.append(
                Item(
                    id=entry.id,
                    hash=entry.hash,
                    visible_name=meta.get("visibleName", ""),
                    type=meta.get("type", ""),
                    parent=meta.get("parent", "") or "",
                    last_modified=meta.get("lastModified", "") or "",
                    deleted=bool(meta.get("deleted") or False),
                )
            )

        logger.info(
            "Listed %d items (%d metadata refetched, %d served from cache)",
            len(items),
            refetched,
            len(items) - refetched,
        )
        return items, fresh_cache
