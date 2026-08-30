"""Turn a rendered page into note content: transcription, tags, links, frontmatter."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from datetime import UTC, date, datetime

import yaml

from . import providers
from .providers import ProviderResult, sanitize_tags

logger = logging.getLogger(__name__)

NEEDS_REVIEW_TAG = "needs-review"
# Below this a title is too generic to autolink safely ("ERE", "p.11").
MIN_AUTOLINK_LEN = 5
MAX_AUTOLINKS = 6
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


def autolink(text: str, titles: list[str], *, max_links: int = MAX_AUTOLINKS) -> str:
    """Wrap the first mention of each existing note title in [[wikilinks]].

    Done in code rather than by the model: asked to wrap specific words, a small
    model simply ignores the instruction, whereas exact matching is free,
    deterministic, and can only ever produce links that actually resolve.

    Longest titles match first, spans already inside [[...]] are left alone, and
    a casing difference becomes an alias so the sentence still reads naturally.
    """
    if not text or not titles:
        return text

    linked = 0
    for title in sorted(set(titles), key=len, reverse=True):
        if linked >= max_links or len(title) < MIN_AUTOLINK_LEN:
            continue
        # Skip anything already inside a wikilink.
        spans = [m.span() for m in re.finditer(r"\[\[[^\]]*\]\]", text)]
        pattern = re.compile(rf"(?<![\w\[]){re.escape(title)}(?![\w\]])", re.IGNORECASE)
        for match in pattern.finditer(text):
            if any(start <= match.start() < end for start, end in spans):
                continue
            found = match.group(0)
            replacement = f"[[{title}]]" if found == title else f"[[{title}|{found}]]"
            text = text[: match.start()] + replacement + text[match.end() :]
            linked += 1
            break
    return text


def build_frontmatter(
    *,
    tags: list[str],
    doc_id: str,
    notebook: str,
    created: str | None = None,
) -> str:
    """YAML frontmatter Obsidian renders as properties."""
    # A real date object, not a string: yaml.safe_dump quotes strings, and a
    # quoted 'created' reads as a plain string in Obsidian rather than a date.
    if created:
        try:
            created_value: object = date.fromisoformat(created)
        except ValueError:
            created_value = created
    else:
        created_value = datetime.now(UTC).date()

    data = {
        "tags": tags,
        "source": SOURCE,
        "rm_doc_id": doc_id,
        "rm_notebook": notebook,
        "created": created_value,
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
    link_mode: str = "related",
    known_titles: list[str] | None = None,
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

    body_text = result.text.strip()
    if link_mode in ("inline", "both") and known_titles:
        # Never link a note to itself.
        others = [t for t in known_titles if t != title]
        body_text = autolink(body_text, others)

    parts = [
        build_frontmatter(tags=tags, doc_id=doc_id, notebook=notebook, created=created)
    ]
    parts.append(f"\n# {title}\n\n")
    if body_text:
        parts.append(body_text + "\n")
    if attach and attachment_name:
        parts.append(f"\n![[{attachment_name}]]\n")
    if result.links and link_mode in ("related", "both"):
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
