"""Turn a rendered page into note content: transcription, tags, links, frontmatter."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from datetime import UTC, datetime

import yaml

from . import providers
from .providers import ProviderResult, sanitize_tags

logger = logging.getLogger(__name__)

NEEDS_REVIEW_TAG = "needs-review"
SOURCE = "reMarkable"
_UNSAFE_PATH_RE = re.compile(r"[^A-Za-z0-9 _.\-]")


@dataclass
class Note:
    """A rendered note, ready to commit."""

    title: str
    body: str
    tags: list[str] = field(default_factory=list)
    links: list[str] = field(default_factory=list)
    needs_review: bool = False
    attach_png: bool = False
    input_tokens: int = 0
    output_tokens: int = 0


def sanitize_path_component(raw: str, *, fallback: str = "untitled") -> str:
    """Make a string safe for a file path without mangling it beyond recognition."""
    cleaned = _UNSAFE_PATH_RE.sub(" ", (raw or "").strip())
    cleaned = re.sub(r"\s+", " ", cleaned).strip(" .")
    cleaned = cleaned[:80].strip(" .")
    return cleaned or fallback


def build_frontmatter(
    *,
    tags: list[str],
    doc_id: str,
    notebook: str,
    created: str | None = None,
) -> str:
    """YAML frontmatter Obsidian renders as properties."""
    data = {
        "tags": tags,
        "source": SOURCE,
        "rm_doc_id": doc_id,
        "rm_notebook": notebook,
        "created": created or datetime.now(UTC).strftime("%Y-%m-%d"),
    }
    body = yaml.safe_dump(
        data, sort_keys=False, allow_unicode=True, default_flow_style=None
    )
    return f"---\n{body}---\n"


def compose_note(
    result: ProviderResult,
    *,
    doc_id: str,
    notebook: str,
    page_index: int,
    min_text_length: int = 20,
    attachment_name: str | None = None,
    created: str | None = None,
) -> Note:
    """Assemble frontmatter + body from a provider result."""
    # Sanitize here too, not only in parse_response: composition is the last
    # gate before frontmatter, and an Obsidian tag containing a space is invalid.
    tags = sanitize_tags(result.tags)
    needs_review = result.needs_review
    attach = False

    # Vision models read handwriting but not diagrams. An almost-empty note is
    # worse than a flagged one with the image attached.
    if len(result.text.strip()) < min_text_length:
        needs_review = True
        attach = True
        logger.info(
            "Transcription below MIN_TEXT_LENGTH (%d < %d); flagging needs-review",
            len(result.text.strip()),
            min_text_length,
        )

    if needs_review and NEEDS_REVIEW_TAG not in tags:
        tags.append(NEEDS_REVIEW_TAG)
    if not tags:
        tags = [NEEDS_REVIEW_TAG]

    title = result.title.strip() or f"{notebook} p{page_index + 1}"

    parts = [
        build_frontmatter(tags=tags, doc_id=doc_id, notebook=notebook, created=created)
    ]
    parts.append(f"\n# {title}\n\n")
    if result.text.strip():
        parts.append(result.text.strip() + "\n")
    if attach and attachment_name:
        parts.append(f"\n![[{attachment_name}]]\n")
    if result.links:
        links = " ".join(f"[[{link}]]" for link in result.links)
        parts.append(f"\n## Related\n\n{links}\n")

    return Note(
        title=title,
        body="".join(parts),
        tags=tags,
        links=result.links,
        needs_review=needs_review,
        attach_png=attach,
        input_tokens=result.input_tokens,
        output_tokens=result.output_tokens,
    )


def enrich_page(
    provider: providers.VisionProvider,
    png: bytes,
    *,
    doc_id: str,
    notebook: str,
    page_index: int,
    existing_tags: list[str],
    existing_titles: list[str],
    min_text_length: int = 20,
    attachment_name: str | None = None,
) -> Note:
    """One vision call -> a complete note."""
    result = provider.extract_and_tag(png, existing_tags, existing_titles)
    return compose_note(
        result,
        doc_id=doc_id,
        notebook=notebook,
        page_index=page_index,
        min_text_length=min_text_length,
        attachment_name=attachment_name,
    )
