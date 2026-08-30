"""Provider-agnostic contract for the combined OCR + tagging call.

One vision call does both jobs, which removes an entire external OCR vendor from
the pipeline. The provider is a config value, so swapping models is a
`sam deploy --parameter-overrides AiModelId=...`, not a code change.

Prompt construction and response parsing live here so the synchronous and batch
paths share them exactly - only the submit/poll/retrieve mechanics differ.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Protocol

logger = logging.getLogger(__name__)

MAX_OUTPUT_TOKENS = 4096
TEMPERATURE = 0.0

_FENCE_RE = re.compile(r"^\s*```(?:json)?\s*|\s*```\s*$", re.MULTILINE)
_TAG_STRIP_RE = re.compile(r"[^a-z0-9_/-]")


@dataclass
class ProviderResult:
    """The common shape every provider returns."""

    text: str = ""
    tags: list[str] = field(default_factory=list)
    title: str = ""
    links: list[str] = field(default_factory=list)
    needs_review: bool = False
    input_tokens: int = 0
    output_tokens: int = 0


class VisionProvider(Protocol):
    def extract_and_tag(
        self,
        image_bytes: bytes,
        existing_tags: list[str],
        existing_titles: list[str] | None = None,
    ) -> ProviderResult:
        """Return {text, tags, title, links} for one rendered page."""
        ...


def sanitize_tag(raw: str) -> str:
    """Obsidian tags cannot contain spaces.

    Strip '#', collapse whitespace to '_', lowercase, drop anything outside
    [a-z0-9_/-].
    """
    if not raw:
        return ""
    tag = raw.strip().lstrip("#").strip()
    tag = re.sub(r"\s+", "_", tag).lower()
    tag = _TAG_STRIP_RE.sub("", tag)
    tag = re.sub(r"_{2,}", "_", tag).strip("_-/")
    return tag


def sanitize_tags(raw_tags: list[str] | None) -> list[str]:
    seen: list[str] = []
    for raw in raw_tags or []:
        tag = sanitize_tag(str(raw))
        if tag and tag not in seen:
            seen.append(tag)
    return seen


LINK_MODES = {"related", "inline", "both"}

_INLINE_RULE = """
  INLINE LINKS: inside "text", wrap the page's key concepts in [[double
  brackets]] so they become Obsidian links. Rules:
    - Link concepts and proper nouns worth their own note (ideas, people,
      books, techniques). Never link ordinary words.
    - At most 6 per page. Fewer is better than more; a page where everything
      is a link is a page where nothing is.
    - When a concept matches an existing note title below, use that title
      EXACTLY so the link resolves instead of creating a near-duplicate.
    - Link a given concept only on its first occurrence.
    - Keep the surrounding sentence unchanged - you are wrapping words that are
      already there, never inserting new ones."""


def build_prompt(
    existing_tags: list[str],
    existing_titles: list[str] | None = None,
    link_mode: str = "related",
) -> str:
    """Prompt for the combined transcribe + tag + link call."""
    vocab = ", ".join(existing_tags[-80:]) if existing_tags else "(none yet)"
    titles = ", ".join(f'"{t}"' for t in (existing_titles or [])[-60:]) or "(none yet)"
    inline_rule = _INLINE_RULE if link_mode in ("inline", "both") else ""
    return f"""You are transcribing one handwritten page from a reMarkable tablet into an Obsidian note.

Return ONLY a single JSON object. No preamble, no explanation, no markdown code fences.

The JSON object must have exactly these keys:
  "text":  string. The transcribed handwriting as Markdown. Preserve list and
           heading structure. Use "" if the page has no legible handwriting.
           If the writer drew [[double brackets]] around anything by hand,
           keep them exactly - that is a deliberate Obsidian link.{inline_rule}
  "tags":  array of 2-5 short topical tags, lowercase, no spaces, no "#".
           STRONGLY prefer reusing tags from the existing vocabulary below over
           inventing near-duplicates.
  "title": string. A short, specific note title derived from the content.
  "links": array of up to 3 existing note titles this page genuinely relates to,
           for [[wikilinks]]. Use [] if none clearly apply. Only use titles from
           the existing notes list; never invent one.

Existing tag vocabulary: {vocab}
Existing note titles: {titles}

Transcribe only what is actually written. Do not summarise, expand, or invent
content. If the page is a diagram with little text, return what text there is
and leave "text" short rather than describing the drawing."""


RETRY_SUFFIX = (
    "\n\nYour previous response could not be parsed as JSON. "
    "Respond with ONLY the raw JSON object, starting with { and ending with }. "
    "No code fences. No commentary."
)


def parse_response(raw: str) -> ProviderResult:
    """Strictly parse a provider response into a ProviderResult.

    Raises ValueError so the caller can retry once with a stricter instruction
    before falling back - tagging must never fail the whole run.
    """
    if not raw or not raw.strip():
        raise ValueError("Empty response")
    cleaned = _FENCE_RE.sub("", raw).strip()
    if not cleaned.startswith("{"):
        start, end = cleaned.find("{"), cleaned.rfind("}")
        if start == -1 or end <= start:
            raise ValueError(f"No JSON object found in response: {raw[:120]!r}")
        cleaned = cleaned[start : end + 1]
    try:
        data = json.loads(cleaned)
    except ValueError as exc:
        raise ValueError(f"Response was not valid JSON: {raw[:120]!r}") from exc
    if not isinstance(data, dict):
        raise ValueError("Response JSON was not an object")

    links = [
        str(link).strip().strip("[]") for link in (data.get("links") or []) if str(link).strip()
    ]
    return ProviderResult(
        text=str(data.get("text") or "").strip(),
        tags=sanitize_tags(data.get("tags")),
        title=str(data.get("title") or "").strip(),
        links=links[:3],
    )


def fallback_result(raw_text: str) -> ProviderResult:
    """Last resort after a failed retry: keep whatever came back, flag for review.

    Never fail the run over tagging - a note that needs review beats a lost page.
    """
    logger.warning("Falling back to needs-review result; response was unparseable")
    return ProviderResult(
        text=(raw_text or "").strip(),
        tags=["needs-review"],
        title="",
        links=[],
        needs_review=True,
    )
