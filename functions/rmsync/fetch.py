"""Select in-scope documents, diff against state, and download changed pages."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from . import state as state_mod
from .config import Config, ConfigError
from .remarkable import Item, RemarkableClient

logger = logging.getLogger(__name__)


@dataclass
class PageRef:
    """One page of one notebook, pending processing."""

    doc_id: str
    doc_hash: str
    notebook: str
    page_id: str
    page_hash: str
    page_index: int
    # The stored file name, which is NOT "<page_id>.rm": reMarkable nests page
    # blobs as "<docId>/<pageId>.rm". The rm-filename header must match exactly.
    file_name: str = ""
    # Vault directory this page's note belongs in, decided by the watch
    # folder it came from (see routes.py).
    vault_dir: str = ""
    link_mode: str = ""
    doc_page_count: int = 0
    data: bytes = field(default=b"", repr=False)

    def __post_init__(self) -> None:
        if not self.file_name:
            self.file_name = f"{self.page_id}.rm"


def select_documents(
    items: list[Item], parents: set[str] | dict[str, str], cfg: Config
) -> list[Item]:
    """Apply folder scope, then include/exclude notebook filters."""
    if cfg.include_notebooks and cfg.exclude_notebooks:
        raise ConfigError("IncludeNotebooks and ExcludeNotebooks are mutually exclusive")

    in_folder = [
        i
        for i in items
        if i.is_document and i.parent in parents and not i.in_trash and not i.deleted
    ]

    if cfg.include_notebooks:
        wanted = {n.strip().lower() for n in cfg.include_notebooks}
        selected = [i for i in in_folder if i.visible_name.strip().lower() in wanted]
        missing = wanted - {i.visible_name.strip().lower() for i in in_folder}
        if missing:
            # Names are matched by visibleName, so a rename on the tablet silently
            # drops a notebook out of scope. Log it rather than let it be
            # discovered as a missing note weeks later.
            logger.warning(
                "IncludeNotebooks names not found in the watch folder: %s", sorted(missing)
            )
    elif cfg.exclude_notebooks:
        unwanted = {n.strip().lower() for n in cfg.exclude_notebooks}
        selected = [i for i in in_folder if i.visible_name.strip().lower() not in unwanted]
    else:
        selected = in_folder

    # Log the resolved set every run so scope drift is visible in CloudWatch.
    logger.info(
        "Resolved notebook set (%d of %d in-folder): %s",
        len(selected),
        len(in_folder),
        sorted(i.visible_name for i in selected),
    )
    return selected


def _dedupe(ids: list[str]) -> list[str]:
    """Preserve first-seen order, drop repeats.

    A page id can legitimately appear more than once in cPages (copied or
    redirected pages). Without this the same page is rendered and sent to the
    model several times in one run - duplicated cost, and duplicated commits.
    """
    seen: set[str] = set()
    out: list[str] = []
    for page_id in ids:
        if page_id not in seen:
            seen.add(page_id)
            out.append(page_id)
    return out


def _ordered_page_ids(content: dict | None) -> list[str]:
    """Page ids in reading order, from the `.content` blob when available."""
    if not content:
        return []
    pages = content.get("cPages", {}).get("pages")
    if isinstance(pages, list):
        return _dedupe(
            [
                p["id"]
                for p in pages
                if isinstance(p, dict) and p.get("id") and not p.get("deleted")
            ]
        )
    legacy = content.get("pages")
    if isinstance(legacy, list) and all(isinstance(p, str) for p in legacy):
        return _dedupe(legacy)
    return []


def diff_pages(
    client: RemarkableClient,
    docs: list[Item],
    st: dict,
    cfg: Config,
    scope_map: dict[str, str] | None = None,
) -> tuple[list[PageRef], int]:
    """Return pages needing work, plus a count skipped as already pending batch.

    Three buckets, not two: a page can be new, awaiting a batch result, or
    committed. A pending page is neither done nor safe to re-queue.
    """
    from .routes import parse_routes, vault_dir

    routes = parse_routes(cfg.vault_routes)
    scope_map = scope_map or {}
    pending: list[PageRef] = []
    pending_batch_skipped = 0

    for doc in docs:
        # Fast path: the document's collection hash is unchanged, so no page
        # inside it can have changed either. Costs zero HTTP calls.
        recorded = st.get("docs", {}).get(doc.id, {})
        if recorded.get("hash") == doc.hash:
            continue

        route = routes.get(scope_map.get(doc.parent, ""))
        dest = vault_dir(doc.visible_name, route=route, default_path=cfg.vault_note_path)
        page_link_mode = (route.link_mode if route and route.link_mode else cfg.link_mode)

        entries = client.get_entries(doc.id, doc.hash)
        content = client.get_content(doc.id, entries)
        order = _ordered_page_ids(content)

        # Page blobs are stored as "<docId>/<pageId>.rm", so the bare page uuid
        # is the basename. Keying on the full path would never match the page
        # ids in .content, silently dropping the reading order and leaving
        # pages sorted by uuid instead.
        rm_entries = {}
        for entry in entries:
            if entry.id.endswith(".rm"):
                rm_entries[entry.id.rsplit("/", 1)[-1][:-3]] = entry

        ordered = _dedupe([pid for pid in order if pid in rm_entries])
        if order and not ordered:
            logger.warning(
                "None of the %d page ids in .content matched a .rm blob for %s; "
                "falling back to uuid order",
                len(order),
                doc.visible_name,
            )
        page_ids = ordered or sorted(rm_entries)

        for index, page_id in enumerate(page_ids):
            entry = rm_entries[page_id]
            if state_mod.is_page_current(st, doc.id, page_id, entry.hash):
                continue
            if state_mod.is_page_pending_batch(st, doc.id, page_id, entry.hash):
                pending_batch_skipped += 1
                continue
            pending.append(
                PageRef(
                    doc_id=doc.id,
                    doc_hash=doc.hash,
                    notebook=doc.visible_name,
                    page_id=page_id,
                    page_hash=entry.hash,
                    page_index=index,
                    file_name=entry.id,
                    vault_dir=dest,
                    link_mode=page_link_mode,
                    doc_page_count=len(page_ids),
                )
            )

    logger.info(
        "Diff: %d page(s) need work, %d already pending batch",
        len(pending),
        pending_batch_skipped,
    )
    return pending, pending_batch_skipped


def download_pages(client: RemarkableClient, pages: list[PageRef], cfg: Config) -> list[PageRef]:
    """Cap to MAX_PAGES_PER_RUN and download those pages' `.rm` blobs.

    Anything over the cap is simply picked up on the next poll - the diff makes
    that automatically correct.
    """
    capped = pages[: cfg.max_pages_per_run]
    if len(pages) > cfg.max_pages_per_run:
        logger.warning(
            "Capping this run at %d of %d pending page(s); remainder follows next poll",
            cfg.max_pages_per_run,
            len(pages),
        )
    for page in capped:
        page.data = client.get_blob(page.file_name, page.page_hash)
    return capped
