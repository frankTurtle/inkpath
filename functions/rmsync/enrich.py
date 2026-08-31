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


def strip_title(title: str, pattern: str) -> str:
    """Remove a boilerplate prefix from a model-generated title.

    "Journal Entry January 1 2019" -> "January 1 2019". The folder already says
    these are journals, so the prefix only makes titles long and repetitive -
    and long labels are what clutter the graph view.

    Done in code rather than by prompt: the model reintroduces a prefix
    whenever it feels descriptive, and a title that changes shape moves the
    file it names.
    """
    if not pattern or not title:
        return title
    try:
        cleaned = re.sub(pattern, "", title, count=1, flags=re.IGNORECASE).strip()
    except re.error:
        logger.warning("TitleStripPattern %r is not a valid regex; ignoring", pattern)
        return title
    cleaned = cleaned.strip(" -\u2013:\u2014")
    # Never strip a title down to nothing.
    return cleaned or title

# Link targets that are references, not concepts. Left alone, a page of book
# quotes turns every "p.5" into its own note and floods the graph with stubs.
_JUNK_LINK_RE = re.compile(
    r"""^(
          \d+(\.\d+)?              # 5, 10.2
        | [ivxlcdm]+                # roman numerals
        | (p|pg|pp|page|ch|chap|chapter|fig|figure|sec|section)
          [\s.]*\d+(\s*[-\u2013]\s*\d+)?
        )$""",
    re.IGNORECASE | re.VERBOSE,
)
MIN_LINK_TARGET_LEN = 4
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


def is_junk_link(target: str) -> bool:
    """True for a link target that is a citation or number rather than an idea."""
    cleaned = target.split("|", 1)[0].strip()
    if len(cleaned) < MIN_LINK_TARGET_LEN:
        return True
    return bool(_JUNK_LINK_RE.match(cleaned))


def clean_inline_links(text: str) -> str:
    """Unwrap [[links]] the model created around page references.

    The brackets are removed but the words stay, so the sentence is unchanged -
    only the spurious link disappears.
    """

    def _replace(match: re.Match[str]) -> str:
        target = match.group(1)
        if is_junk_link(target):
            # Keep the display text, drop the link.
            return target.split("|", 1)[-1]
        return match.group(0)

    return re.sub(r"\[\[([^\]]+)\]\]", _replace, text)


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
    link_notebook: bool = False,
) -> str:
    """YAML frontmatter Obsidian renders as properties.

    With `link_notebook`, rm_notebook becomes a [[wikilink]], which makes each
    notebook a hub note that its pages point at. Without it, a folder of
    unlinked entries shows up in the graph as scattered silos.
    """
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
        # Quoted by safe_dump: a bare [[x]] is a nested YAML list, not a string.
        "rm_notebook": f"[[{notebook}]]" if link_notebook and notebook else notebook,
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
    link_notebook: bool = False,
    title_strip_pattern: str = "",
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

    title = strip_title(result.title.strip(), title_strip_pattern) or (
        f"{notebook} p{page_index + 1}"
    )

    body_text = clean_inline_links(result.text.strip())
    if link_mode in ("inline", "both") and known_titles:
        # Never link a note to itself.
        others = [t for t in known_titles if t != title]
        body_text = autolink(body_text, others)

    parts = [
        build_frontmatter(
            tags=tags,
            doc_id=doc_id,
            notebook=notebook,
            created=created,
            link_notebook=link_notebook,
        )
    ]
    parts.append(f"\n# {title}\n\n")
    if body_text:
        parts.append(body_text + "\n")
    if attach and attachment_name:
        parts.append(f"\n![[{attachment_name}]]\n")
    # A note must never appear in its own Related list.
    related = [
        link
        for link in result.links
        if link.strip().lower() != title.strip().lower() and not is_junk_link(link)
    ]
    if related and link_mode in ("related", "both"):
        links = " ".join(f"[[{link}]]" for link in related)
        parts.append(f"\n## Related\n\n{links}\n")

    return Note(
        title=title,
        body="".join(parts),
        tags=tags,
        links=related,
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
