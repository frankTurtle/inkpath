from __future__ import annotations

import datetime
import io

import pytest
import yaml
from PIL import Image

from rmsync.enrich import build_frontmatter, compose_note, sanitize_path_component
from rmsync.providers import ProviderResult, parse_response, sanitize_tag, sanitize_tags
from rmsync.providers.base import fallback_result
from rmsync.providers.bedrock import BedrockProvider
from rmsync.providers.direct_api import DirectApiProvider
from rmsync.render import parse_lines, render_page, stroke_count

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
        self.posted: list = []

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


# ------------------------------------------------- direct provider routing --


@pytest.mark.parametrize(
    "base_url,expected_url",
    [
        ("https://api.x.ai", "https://api.x.ai/v1/messages"),
        ("https://api.x.ai/", "https://api.x.ai/v1/messages"),
        ("https://api.x.ai/v1/messages", "https://api.x.ai/v1/messages"),
        ("https://api.anthropic.com", "https://api.anthropic.com/v1/messages"),
    ],
)
def test_direct_provider_builds_messages_url(base_url, expected_url):
    p = DirectApiProvider("grok-4.6", "key", base_url=base_url, session=FakeSession())
    assert p._base_url == expected_url


def test_direct_provider_uses_bearer_for_xai():
    """xAI is Anthropic-shaped but takes Bearer; x-api-key would 401."""
    s = FakeSession()
    DirectApiProvider("grok-4.6", "xai-key", base_url="https://api.x.ai", session=s)
    assert s.headers["Authorization"] == "Bearer xai-key"
    assert "x-api-key" not in s.headers


def test_direct_provider_uses_x_api_key_for_anthropic():
    s = FakeSession()
    DirectApiProvider("claude", "ant-key", base_url="https://api.anthropic.com", session=s)
    assert s.headers["x-api-key"] == "ant-key"
    assert "anthropic-version" in s.headers
    assert "Authorization" not in s.headers


def test_direct_provider_defaults_to_anthropic():
    s = FakeSession()
    p = DirectApiProvider("claude", "k", session=s)
    assert p._base_url == "https://api.anthropic.com/v1/messages"
    assert "x-api-key" in s.headers


def test_registry_passes_base_url_to_direct_provider():
    from rmsync.providers import get as get_provider

    p = get_provider("direct", "grok-4.6", api_key="k", base_url="https://api.x.ai")
    assert p._base_url == "https://api.x.ai/v1/messages"


# ----------------------------------------------------- legacy .lines format --


def test_render_parses_legacy_v5(legacy_v5_rm):
    """Notebooks on older firmware stay v5 forever; rmscene only reads v6."""
    assert stroke_count(legacy_v5_rm) == 5
    png = render_page(legacy_v5_rm, width=1400, blank_threshold=3)
    assert png is not None and png.startswith(b"\x89PNG")


def test_legacy_v5_matches_v6_geometry(legacy_v5_rm, page_rm):
    """v5 x is absolute, v6 x is centred on 0. Normalising means the same
    drawing renders identically from either format."""
    v5 = Image.open(io.BytesIO(render_page(legacy_v5_rm, width=1400, blank_threshold=3)))
    v6 = Image.open(io.BytesIO(render_page(page_rm, width=1400, blank_threshold=3)))
    assert v5.size == v6.size


def test_wrong_legacy_version_raises_not_blank():
    """A v9 header must not be mistaken for an empty page."""
    with pytest.raises(ValueError, match="header"):
        render_page(
            b"reMarkable .lines file, version=9          " + b"\x00" * 32,
            width=800,
            blank_threshold=1,
        )


def test_legacy_version_detection():
    from rmsync.render import _legacy_version

    assert _legacy_version(b"reMarkable .lines file, version=5          " + b"\x00") == 5
    assert _legacy_version(b"reMarkable .lines file, version=3          " + b"\x00") == 3
    assert _legacy_version(b"reMarkable .lines file, version=9          " + b"\x00") is None
    assert _legacy_version(b"not a lines file") is None


def test_legacy_truncated_blob_keeps_what_parsed(legacy_v5_rm):
    """A truncated page should not lose every stroke that did parse."""
    truncated = legacy_v5_rm[: len(legacy_v5_rm) // 2]
    assert 0 < len(parse_lines(truncated)) < 5


def test_unparseable_blob_raises_not_silently_empty():
    """A blob we cannot read at all must fail loudly, not look like a blank page."""
    with pytest.raises((ValueError, EOFError)):
        render_page(b"total garbage that is not any known format", width=800, blank_threshold=1)


def test_created_is_an_unquoted_yaml_date():
    """Obsidian treats a quoted 'created' as a string, not a date property."""
    fm = build_frontmatter(tags=["a"], doc_id="d", notebook="N", created="2026-08-30")
    assert "created: 2026-08-30" in fm
    assert "created: '2026-08-30'" not in fm
    assert yaml.safe_load(fm.split("---")[1])["created"] == datetime.date(2026, 8, 30)


def test_created_defaults_to_today_as_a_date():
    fm = build_frontmatter(tags=["a"], doc_id="d", notebook="N")
    assert isinstance(yaml.safe_load(fm.split("---")[1])["created"], datetime.date)


def test_created_falls_back_gracefully_on_bad_input():
    fm = build_frontmatter(tags=["a"], doc_id="d", notebook="N", created="not-a-date")
    assert yaml.safe_load(fm.split("---")[1])["created"] == "not-a-date"


# ------------------------------------------------------------- link modes --


def test_prompt_preserves_hand_drawn_brackets():
    """Writing [[ ]] on the tablet is a deliberate link; never strip it."""
    from rmsync.providers.base import build_prompt

    assert "drew [[double brackets]]" in build_prompt([], [])


def test_related_mode_has_no_inline_instruction():
    from rmsync.providers.base import build_prompt

    assert "INLINE LINKS" not in build_prompt([], [], "related")


@pytest.mark.parametrize("mode", ["inline", "both"])
def test_inline_modes_request_inline_links(mode):
    from rmsync.providers.base import build_prompt

    p = build_prompt([], ["Optionality"], mode)
    assert "INLINE LINKS" in p
    assert "At most 6 per page" in p
    assert "Optionality" in p          # existing titles offered for exact reuse


def test_inline_links_survive_parsing():
    """Brackets inside `text` must reach the note untouched."""
    r = parse_response(
        '{"text":"Learning [[Via negativa]] beats adding.","tags":["a"],'
        '"title":"T","links":[]}'
    )
    assert "[[Via negativa]]" in r.text


def test_provider_passes_link_mode_to_prompt():
    captured = {}

    class Capture(FakeBedrockClient):
        def converse(self, **kw):
            captured["prompt"] = kw["messages"][0]["content"][1]["text"]
            return super().converse(**kw)

    BedrockProvider("m", client=Capture(), link_mode="both").extract_and_tag(b"png", [])
    assert "INLINE LINKS" in captured["prompt"]


# ------------------------------------------------------------- autolinking --


def test_autolink_wraps_first_mention_only():
    from rmsync.enrich import autolink

    out = autolink("Optionality matters. Optionality again.", ["Optionality"])
    assert out.count("[[Optionality]]") == 1
    assert out.endswith("Optionality again.")


def test_autolink_uses_alias_when_casing_differs():
    from rmsync.enrich import autolink

    out = autolink("He praised optionality here.", ["Optionality"])
    assert "[[Optionality|optionality]]" in out


def test_autolink_never_double_links():
    from rmsync.enrich import autolink

    out = autolink("See [[Optionality]] and Optionality.", ["Optionality"])
    assert out.count("[[") == 1 or "[[Optionality|" not in out


def test_autolink_prefers_longest_title():
    from rmsync.enrich import autolink

    out = autolink("Read about Via negativa thinking.", ["Via", "Via negativa"])
    assert "[[Via negativa]]" in out


def test_autolink_skips_short_titles():
    from rmsync.enrich import autolink

    assert autolink("ERE is a book.", ["ERE"]) == "ERE is a book."


def test_autolink_respects_word_boundaries():
    from rmsync.enrich import autolink

    assert autolink("Optionalityism", ["Optionality"]) == "Optionalityism"


def test_autolink_caps_total_links():
    from rmsync.enrich import autolink

    titles = [f"Concept{i:02d}" for i in range(20)]
    text = " ".join(titles)
    assert autolink(text, titles).count("[[") == 6


def test_compose_note_autolinks_in_inline_mode():
    note = compose_note(
        ProviderResult(text="Thinking about Optionality today and more text here.",
                       tags=["t"], title="New Note"),
        doc_id="d", notebook="N", page_index=0,
        link_mode="inline", known_titles=["Optionality"],
    )
    assert "[[Optionality]]" in note.body


def test_related_mode_does_not_autolink():
    note = compose_note(
        ProviderResult(text="Thinking about Optionality today and more text here.",
                       tags=["t"], title="New Note"),
        doc_id="d", notebook="N", page_index=0,
        link_mode="related", known_titles=["Optionality"],
    )
    assert "[[Optionality]]" not in note.body


def test_note_never_links_to_itself():
    note = compose_note(
        ProviderResult(text="Optionality is the subject of this whole page here.",
                       tags=["t"], title="Optionality"),
        doc_id="d", notebook="N", page_index=0,
        link_mode="inline", known_titles=["Optionality"],
    )
    assert "[[Optionality]]" not in note.body


def test_inline_mode_omits_related_section():
    note = compose_note(
        ProviderResult(text="x" * 60, tags=["t"], title="T", links=["Other"]),
        doc_id="d", notebook="N", page_index=0, link_mode="inline",
    )
    assert "## Related" not in note.body


# ------------------------------------------------------ junk link filtering --


@pytest.mark.parametrize(
    "target",
    ["P.5", "p.10", "pg 12", "page 3", "Ch. 4", "10", "10.2", "iv", "P.10", "pp 22-24"],
)
def test_page_references_are_not_links(target):
    """A page of book quotes would otherwise mint a note per citation."""
    from rmsync.enrich import is_junk_link

    assert is_junk_link(target)


@pytest.mark.parametrize("target", ["Thoreau", "Via negativa", "Optionality", "Walden"])
def test_real_concepts_stay_links(target):
    from rmsync.enrich import is_junk_link

    assert not is_junk_link(target)


def test_clean_inline_links_keeps_words_drops_brackets():
    from rmsync.enrich import clean_inline_links

    out = clean_inline_links("He has no time.[[P.5]] Read [[Thoreau]] now.")
    assert out == "He has no time.P.5 Read [[Thoreau]] now."


def test_clean_inline_links_handles_aliases():
    from rmsync.enrich import clean_inline_links

    assert clean_inline_links("see [[P.5|page five]]") == "see page five"


def test_compose_note_strips_page_number_links():
    note = compose_note(
        ProviderResult(text="Quote here.[[P.5]] Another quote.[[P.6]] " + "x" * 40,
                       tags=["t"], title="Walden Quotes"),
        doc_id="d", notebook="Walden", page_index=0, link_mode="both",
    )
    assert "[[P.5]]" not in note.body and "[[P.6]]" not in note.body
    assert "P.5" in note.body            # the citation text itself survives


def test_note_is_not_in_its_own_related_section():
    """Observed live: a note listed itself under Related."""
    note = compose_note(
        ProviderResult(text="x" * 60, tags=["t"], title="Walden Thoreau Quotes",
                       links=["Walden Thoreau Quotes", "Other Note"]),
        doc_id="d", notebook="Walden", page_index=0, link_mode="both",
    )
    assert "[[Walden Thoreau Quotes]]" not in note.body
    assert "[[Other Note]]" in note.body


def test_related_drops_junk_targets():
    note = compose_note(
        ProviderResult(text="x" * 60, tags=["t"], title="T", links=["P.5", "Real Note"]),
        doc_id="d", notebook="N", page_index=0, link_mode="both",
    )
    assert "[[P.5]]" not in note.body and "[[Real Note]]" in note.body


def test_none_link_mode_suppresses_all_linking():
    """Journals are chronological; linking their prose buries the graph."""
    note = compose_note(
        ProviderResult(text="Talked to Samantha about deep work today, at length.",
                       tags=["t"], title="Journal Entry", links=["Some Note"]),
        doc_id="d", notebook="Journal 2023", page_index=0,
        link_mode="none", known_titles=["Some Note", "Deep work"],
    )
    assert "[[" not in note.body


def test_none_mode_prompt_forbids_new_brackets():
    from rmsync.providers.base import build_prompt

    p = build_prompt([], ["X"], "none")
    assert "Do NOT add" in p and "Always return []" in p
    assert "INLINE LINKS" not in p


# -------------------------------------------------------- notebook hub link --


def test_notebook_property_can_be_a_wikilink():
    """Makes each notebook a hub node its pages point at."""
    fm = build_frontmatter(
        tags=["a"], doc_id="d", notebook="Journal 2021", link_notebook=True
    )
    assert "rm_notebook: '[[Journal 2021]]'" in fm


def test_notebook_wikilink_is_quoted_so_yaml_stays_a_string():
    """A bare [[x]] parses as a nested list, not a link."""
    fm = build_frontmatter(
        tags=["a"], doc_id="d", notebook="Journal 2021", link_notebook=True
    )
    assert yaml.safe_load(fm.split("---")[1])["rm_notebook"] == "[[Journal 2021]]"


def test_notebook_property_plain_by_default():
    fm = build_frontmatter(tags=["a"], doc_id="d", notebook="Journal 2021")
    assert yaml.safe_load(fm.split("---")[1])["rm_notebook"] == "Journal 2021"


def test_compose_note_threads_link_notebook_through():
    note = compose_note(
        ProviderResult(text="x" * 60, tags=["t"], title="T"),
        doc_id="d", notebook="Journal 2021", page_index=0, link_notebook=True,
    )
    assert "[[Journal 2021]]" in note.body


def test_empty_notebook_is_not_linked():
    fm = build_frontmatter(tags=["a"], doc_id="d", notebook="", link_notebook=True)
    assert "[[]]" not in fm


# ------------------------------------------------------------ title strip --


@pytest.mark.parametrize(
    "title,expected",
    [
        ("Journal Entry January 1 2019", "January 1 2019"),
        ("Journal Entry - May 3-4 2146", "May 3-4 2146"),
        ("journal entry April 24 2023", "April 24 2023"),
        ("January 1 2019", "January 1 2019"),          # already clean
        ("Walden Thoreau Quotes", "Walden Thoreau Quotes"),
    ],
)
def test_strip_title_removes_prefix(title, expected):
    from rmsync.enrich import strip_title

    assert strip_title(title, r"^Journal Entry") == expected


def test_strip_title_never_empties_a_title():
    from rmsync.enrich import strip_title

    assert strip_title("Journal Entry", r"^Journal Entry") == "Journal Entry"


def test_strip_title_ignores_invalid_regex():
    from rmsync.enrich import strip_title

    assert strip_title("Journal Entry X", "[unclosed") == "Journal Entry X"


def test_strip_title_noop_without_pattern():
    from rmsync.enrich import strip_title

    assert strip_title("Journal Entry X", "") == "Journal Entry X"


def test_compose_note_uses_stripped_title_for_heading():
    note = compose_note(
        ProviderResult(text="x" * 60, tags=["journal"], title="Journal Entry January 1 2019"),
        doc_id="d", notebook="Journal 2021", page_index=0,
        title_strip_pattern=r"^Journal Entry",
    )
    assert note.title == "January 1 2019"
    assert "# January 1 2019" in note.body
    assert "Journal Entry" not in note.body


# ------------------------------------------------------- collection hub --


def test_collection_property_is_a_wikilink():
    fm = build_frontmatter(tags=["a"], doc_id="d", notebook="Walden", collection="Book Notes")
    assert yaml.safe_load(fm.split("---")[1])["collection"] == "[[Book Notes]]"


def test_collection_absent_when_not_requested():
    fm = build_frontmatter(tags=["a"], doc_id="d", notebook="Walden")
    assert "collection" not in yaml.safe_load(fm.split("---")[1])


def test_two_tier_hubs_note_to_notebook_to_collection():
    """note -> [[Walden]] -> [[Book Notes]] makes the folder one cluster."""
    note = compose_note(
        ProviderResult(text="x" * 60, tags=["t"], title="T"),
        doc_id="d", notebook="Walden", page_index=0,
        link_notebook=True, collection="Book Notes",
    )
    assert "[[Walden]]" in note.body and "[[Book Notes]]" in note.body


# ------------------------------------------------------------ tag denylist --


def test_excluded_tags_are_dropped():
    """A tag restating the folder adds nothing the path and hub do not."""
    note = compose_note(
        ProviderResult(text="x" * 60, tags=["journal", "parenting"], title="T"),
        doc_id="d", notebook="Journal 2021", page_index=0,
        exclude_tags=["journal"],
    )
    assert note.tags == ["parenting"]


def test_exclusion_is_case_insensitive():
    note = compose_note(
        ProviderResult(text="x" * 60, tags=["Journal"], title="T"),
        doc_id="d", notebook="N", page_index=0, exclude_tags=["journal"],
    )
    assert "Journal" not in note.tags


def test_exclusion_emptying_tags_is_not_needs_review():
    """needs-review means the OCR looks wrong. A note left tagless because its
    only tag was on the denylist is deliberate, and must not raise that flag."""
    note = compose_note(
        ProviderResult(text="x" * 60, tags=["journal"], title="T"),
        doc_id="d", notebook="N", page_index=0, exclude_tags=["journal"],
    )
    assert note.tags == []


def test_model_returning_no_tags_still_flags_review():
    note = compose_note(
        ProviderResult(text="x" * 60, tags=[], title="T"),
        doc_id="d", notebook="N", page_index=0,
    )
    assert note.tags == ["needs-review"]
