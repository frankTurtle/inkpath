"""Sync state, stored as a single JSON object in S3.

DynamoDB would work but is not free; this is one small map read and written once
per run, so S3 GET+PUT costs fractions of a cent.

State shape::

    {
      "version": 1,
      "docs": {
        "<docId>": {
          "hash": "<collection hash>",
          "notebook": "Reading - Antifragile",
          "processedAt": "2026-08-29T12:00:00Z",
          "pages": {
            "<pageId>": {
              "hash": "<blob hash>",
              "notePath": "Inbox/reMarkable/.../note.md",
              "status": "committed" | "pending-batch",
              "committedAt": "..."
            }
          }
        }
      },
      "tagVocabulary": ["book-notes", "epistemology"],
      "noteTitles": ["Antifragility and optionality"],
      "batch": {"queue": [...], "pendingJobs": [...]}
    }

Page-level hashes matter: editing one page of a notebook changes the notebook's
collection hash, but only that page's blob hash. Diffing per page means a
30-page notebook with one new page costs exactly one model call, not thirty.
"""

from __future__ import annotations

import json
import logging
from typing import Any

import boto3
from botocore.exceptions import ClientError

logger = logging.getLogger(__name__)

STATE_KEY = "state.json"
STATE_VERSION = 1

STATUS_COMMITTED = "committed"
STATUS_PENDING_BATCH = "pending-batch"

# Keep the model's tag suggestions anchored without unbounded prompt growth.
MAX_TAG_VOCABULARY = 200
MAX_NOTE_TITLES = 200

_s3 = None


def _client() -> Any:
    global _s3
    if _s3 is None:
        _s3 = boto3.client("s3")
    return _s3


def empty_state() -> dict[str, Any]:
    return {
        "version": STATE_VERSION,
        "docs": {},
        "tagVocabulary": {},
        "noteTitles": [],
        "batch": {"queue": [], "pendingJobs": []},
    }


def _normalise(state: dict[str, Any]) -> dict[str, Any]:
    """Fill in any keys a older/partial state object is missing."""
    base = empty_state()
    base.update(state or {})
    base.setdefault("docs", {})
    batch = base.get("batch") or {}
    batch.setdefault("queue", [])
    batch.setdefault("pendingJobs", [])
    base["batch"] = batch
    for doc in base["docs"].values():
        doc.setdefault("pages", {})
    return base


def load_state(bucket: str, key: str = STATE_KEY) -> dict[str, Any]:
    """Load state from S3. A first run (404) is an empty state, not an error."""
    try:
        body = _client().get_object(Bucket=bucket, Key=key)["Body"].read()
    except ClientError as exc:
        code = exc.response.get("Error", {}).get("Code", "")
        if code in {"NoSuchKey", "404", "NoSuchBucket"}:
            logger.info("No existing state at s3://%s/%s - first run", bucket, key)
            return empty_state()
        raise
    try:
        return _normalise(json.loads(body.decode("utf-8")))
    except (ValueError, UnicodeDecodeError):
        # Corrupt state would otherwise re-OCR the whole library. Refuse loudly.
        logger.exception("state.json at s3://%s/%s is not valid JSON", bucket, key)
        raise


def save_state(state: dict[str, Any], bucket: str, key: str = STATE_KEY) -> None:
    """Persist state. Call ONLY after the work it records actually succeeded."""
    state = _normalise(state)
    state["version"] = STATE_VERSION
    _client().put_object(
        Bucket=bucket,
        Key=key,
        Body=json.dumps(state, indent=2, sort_keys=True).encode("utf-8"),
        ContentType="application/json",
    )
    logger.info("Saved state: %d documents tracked", len(state["docs"]))


# --------------------------------------------------------------- accessors ---


def page_record(state: dict[str, Any], doc_id: str, page_id: str) -> dict[str, Any] | None:
    return state.get("docs", {}).get(doc_id, {}).get("pages", {}).get(page_id)


def is_page_current(state: dict[str, Any], doc_id: str, page_id: str, page_hash: str) -> bool:
    """True if this exact page version was already committed."""
    rec = page_record(state, doc_id, page_id)
    return bool(
        rec
        and rec.get("hash") == page_hash
        and rec.get("status") == STATUS_COMMITTED
    )


def is_page_pending_batch(state: dict[str, Any], doc_id: str, page_id: str, page_hash: str) -> bool:
    """True if this page version is already awaiting a batch result.

    A pending page is neither done nor safe to re-queue, so the diff needs this
    third bucket - new / pending-batch / committed.
    """
    rec = page_record(state, doc_id, page_id)
    return bool(
        rec
        and rec.get("hash") == page_hash
        and rec.get("status") == STATUS_PENDING_BATCH
    )


def record_page(
    state: dict[str, Any],
    *,
    doc_id: str,
    doc_hash: str,
    notebook: str,
    page_id: str,
    page_hash: str,
    note_path: str,
    status: str,
    timestamp: str,
) -> None:
    doc = state.setdefault("docs", {}).setdefault(doc_id, {"pages": {}})
    doc["notebook"] = notebook
    doc["processedAt"] = timestamp
    doc.setdefault("pages", {})[page_id] = {
        "hash": page_hash,
        "notePath": note_path,
        "status": status,
        "committedAt": timestamp if status == STATUS_COMMITTED else None,
    }
    # Only advance the document hash once EVERY page of it is committed, so an
    # interrupted run resumes the remaining pages on the next poll. Setting it
    # early would make the diff fast-path skip the whole document and silently
    # lose every page that had not been processed yet.
    if all(p.get("status") == STATUS_COMMITTED for p in doc["pages"].values()):
        doc["hash"] = doc_hash
    else:
        doc.pop("hash", None)


def existing_note_path(state: dict[str, Any], doc_id: str, page_id: str) -> str | None:
    """Note path already recorded for this page, if any.

    Checked before every commit regardless of which path (batch or sync)
    produced the content, so a batch fallback cannot double-commit.
    """
    rec = page_record(state, doc_id, page_id)
    return rec.get("notePath") if rec else None


def claimed_paths(state: dict[str, Any], *, excluding: tuple[str, str] | None = None) -> set[str]:
    """Every note path already recorded, optionally excluding one (docId, pageId).

    Used to keep two pages from resolving to the same file.
    """
    taken: set[str] = set()
    for doc_id, doc in state.get("docs", {}).items():
        for page_id, page in doc.get("pages", {}).items():
            if excluding and (doc_id, page_id) == excluding:
                continue
            path = page.get("notePath")
            if path:
                taken.add(path)
    return taken


def tags_for(state: dict[str, Any], collection: str = "") -> list[str]:
    """Tag vocabulary for one collection.

    Scoped per collection because a single shared vocabulary cross-contaminates:
    offered a book's marketing tags while transcribing a diary page, the model
    dutifully reuses them, and journal entries end up tagged "psychographics".
    """
    vocab = state.get("tagVocabulary") or {}
    if isinstance(vocab, list):        # legacy flat vocabulary
        return list(vocab)
    scoped = list(vocab.get(collection, []))
    # Fall back to the legacy bucket so an upgraded deployment keeps its history.
    return scoped or list(vocab.get("", []))


def learn_vocabulary(
    state: dict[str, Any],
    tags: list[str],
    title: str | None,
    collection: str = "",
) -> None:
    """Grow the tag/title vocabulary so the model reuses tags instead of
    inventing near-duplicates."""
    vocab = state.get("tagVocabulary") or {}
    if isinstance(vocab, list):        # migrate legacy flat list
        vocab = {"": vocab}
    bucket: list[str] = list(vocab.get(collection, []))
    for tag in tags:
        if tag and tag not in bucket:
            bucket.append(tag)
    vocab[collection] = bucket[-MAX_TAG_VOCABULARY:]
    state["tagVocabulary"] = vocab

    titles: list[str] = state.setdefault("noteTitles", [])
    if title and title not in titles:
        titles.append(title)
    state["noteTitles"] = titles[-MAX_NOTE_TITLES:]
