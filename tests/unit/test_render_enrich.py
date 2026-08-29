from __future__ import annotations

import io

import pytest
import yaml
from PIL import Image

from rmsync.enrich import build_frontmatter, compose_note, sanitize_path_component
from rmsync.providers import ProviderResult, parse_response, sanitize_tag, sanitize_tags
from rmsync.providers.base import fallback_result
from rmsync.providers.bedrock import BedrockProvider
from rmsync.providers.direct_api import DirectApiProvider
from rmsync.render import render_page, stroke_count

# --------------------------------------------------------------- rendering --


def test_render_skips_blank_page(blank_rm):
    assert render_page(blank_rm, width=1400, blank_threshold=3) is None


def test_render_produces_nonempty_png(page_rm):
    png = render_page(page_rm, width=1400, blank_threshold=3)
    assert png is not None and png.startswith(b"\x89PNG")
    img = Image.open(io.BytesIO(png))
    assert img.mode == "L"           # grayscale keeps vision cost down
    assert img.width == 1400
    assert img.getextrema()[0] == 0  # actual ink, not a blank canvas


def test_stroke_count_matches_fixture(page_rm, blank_rm):
    assert stroke_count(page_rm) == 5
    assert stroke_count(blank_rm) == 1


def test_render_threshold_is_configurable(blank_rm):
    assert render_page(blank_rm, width=800, blank_threshold=1) is not None


# ------------------------------------------------------------- tag hygiene --


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("AI API", "ai_api"),
        ("#tag", "tag"),
        ("  spaced  ", "spaced"),
        ("Zettelkasten", "zettelkasten"),
        ("book/notes", "book/notes"),
        ("multi   space", "multi_space"),
        ("!!!", ""),
        ("", ""),
    ],
)
def test_tag_sanitizer(raw, expected):
    assert sanitize_tag(raw) == expected


def test_tag_sanitizer_drops_non_ascii():
    assert sanitize_tag("café") == "caf"


def test_tag_sanitizer_dedupes_and_drops_empties():
    assert sanitize_tags(["AI API", "ai api", "#", "Zettel"]) == ["ai_api", "zettel"]


def test_tags_never_contain_spaces():
    """Obsidian tags cannot contain spaces - the invariant that matters."""
    for tag in sanitize_tags(["a b c", "  x  y  ", "A B"]):
        assert " " not in tag


# ---------------------------------------------------------- response parse --


def test_parse_response_plain_json():
    r = parse_response('{"text":"hi","tags":["A B"],"title":"T","links":["N"]}')
    assert r.text == "hi" and r.tags == ["a_b"] and r.title == "T" and r.links == ["N"]


def test_parse_response_strips_code_fences():
    r = parse_response('```json\n{"text":"x","tags":[],"title":"","links":[]}\n```')
    assert r.text == "x"


def test_parse_response_recovers_object_from_prose():
    r = parse_response('Sure!\n{"text":"y","tags":[],"title":"","links":[]}\nHope that helps')
    assert r.text == "y"


def test_parse_response_caps_links_at_three():
    r = parse_response('{"text":"t","tags":[],"title":"","links":["a","b","c","d"]}')
    assert len(r.links) == 3


@pytest.mark.parametrize("bad", ["", "   ", "no json here", "[1,2,3]"])
def test_parse_failure_raises(bad):
    with pytest.raises(ValueError):
        parse_response(bad)


def test_parse_failure_falls_back():
    """Tagging must never fail the run."""
    r = fallback_result("garbage output")
    assert r.tags == ["needs-review"] and r.needs_review and r.text == "garbage output"


# ------------------------------------------------------------- frontmatter --


def test_frontmatter_valid_yaml():
    fm = build_frontmatter(
        tags=["book-notes", "epistemology"],
        doc_id="b083f079-c45e-4b1f-81ea-32c36a672142",
        notebook="Reading - Antifragile",
        created="2026-08-29",
    )
    assert fm.startswith("---\n") and fm.endswith("---\n")
    data = yaml.safe_load(fm.split("---")[1])
    assert data["tags"] == ["book-notes", "epistemology"]
    assert data["source"] == "reMarkable"
    assert data["rm_doc_id"] == "b083f079-c45e-4b1f-81ea-32c36a672142"


def test_frontmatter_quotes_names_with_specials():
    """A colon or '#' in a notebook name must not break the YAML."""
    fm = build_frontmatter(tags=["a"], doc_id="d", notebook="Reading: Antifragile #1")
    data = yaml.safe_load(fm.split("---")[1])
    assert data["rm_notebook"] == "Reading: Antifragile #1"


# ------------------------------------------------------------ note assembly --


def test_low_text_flags_needs_review():
    note = compose_note(
        ProviderResult(text="hi", tags=["x"], title="T"),
        doc_id="d", notebook="N", page_index=0, min_text_length=20,
        attachment_name="p.png",
    )
    assert note.needs_review and note.attach_png
    assert "needs-review" in note.tags
    assert "![[p.png]]" in note.body


def test_sufficient_text_does_not_flag():
    note = compose_note(
        ProviderResult(text="x" * 50, tags=["ok"], title="T"),
        doc_id="d", notebook="N", page_index=0, min_text_length=20,
    )
    assert not note.needs_review and note.tags == ["ok"]


def test_note_falls_back_to_generated_title():
    note = compose_note(
        ProviderResult(text="x" * 50, tags=["t"], title=""),
        doc_id="d", notebook="Reading", page_index=2, min_text_length=20,
    )
    assert note.title == "Reading p3"


def test_links_render_as_wikilinks():
    note = compose_note(
        ProviderResult(text="x" * 50, tags=["t"], title="T", links=["Other Note"]),
        doc_id="d", notebook="N", page_index=0,
    )
    assert "[[Other Note]]" in note.body


def test_note_body_frontmatter_parses(page_rm):
    note = compose_note(
        ProviderResult(text="x" * 50, tags=["a"], title="T"),
        doc_id="d", notebook="N", page_index=0,
    )
    assert yaml.safe_load(note.body.split("---")[1])["tags"] == ["a"]


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("Reading: Antifragile", "Reading Antifragile"),
        ("a/b\\c", "a b c"),
        ("...", "untitled"),
        ("", "untitled"),
    ],
)
def test_path_component_sanitized(raw, expected):
    assert sanitize_path_component(raw) == expected


# --------------------------------------------------------------- providers --

PAYLOAD = '{"text":"hello world","tags":["a b"],"title":"T","links":[]}'


class FakeBedrockClient:
    def __init__(self, text=PAYLOAD):
        self.text = text
        self.calls = 0

    def converse(self, **kw):
        self.calls += 1
        return {
            "output": {"message": {"content": [{"text": self.text}]}},
            "usage": {"inputTokens": 10, "outputTokens": 5},
            "stopReason": "end_turn",
        }


class FakeResp:
    ok = True
    status_code = 200

    def __init__(self, text=PAYLOAD):
        self._text = text

    def json(self):
        return {
            "content": [{"type": "text", "text": self._text}],
            "usage": {"input_tokens": 10, "output_tokens": 5},
            "stop_reason": "end_turn",
        }


class FakeSession:
    def __init__(self, text=PAYLOAD):
        self.headers: dict = {}
        self._text = text

    def post(self, *a, **kw):
        return FakeResp(self._text)


def test_provider_bedrock_and_direct_share_result_shape():
    """Swapping AiProvider must never require a change to enrich.py."""
    b = BedrockProvider("some.model", client=FakeBedrockClient()).extract_and_tag(b"png", [])
    d = DirectApiProvider("some.model", "key", session=FakeSession()).extract_and_tag(b"png", [])

    assert isinstance(b, ProviderResult) and isinstance(d, ProviderResult)
    for field in ("text", "tags", "title", "links", "input_tokens", "output_tokens"):
        assert getattr(b, field) == getattr(d, field), field
    assert b.tags == ["a_b"]


def test_bedrock_retries_once_then_falls_back():
    client = FakeBedrockClient(text="not json at all")
    r = BedrockProvider("some.model", client=client).extract_and_tag(b"png", [])
    assert client.calls == 2          # one retry, then give up
    assert r.tags == ["needs-review"]
    assert r.input_tokens == 20       # both attempts counted


def test_direct_provider_requires_api_key():
    with pytest.raises(ValueError, match="ai-api-key"):
        DirectApiProvider("m", "")


def test_bedrock_requires_model_id():
    with pytest.raises(ValueError, match="AiModelId"):
        BedrockProvider("")


def test_grok_model_gets_low_reasoning_effort():
    """Reasoning tiers add cost for a task that is OCR plus tag selection."""
    from rmsync.providers.bedrock import _additional_fields

    assert _additional_fields("xai.grok-4.6") == {"reasoning_effort": "low"}
    assert _additional_fields("anthropic.claude-haiku-4-5") == {}
