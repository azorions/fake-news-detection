from pathlib import Path

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from fnd_api.schemas import SpanKind
from fnd_api.signals.rules import RawSpan, RuleEngine
from fnd_api.signals.spans import finalize

LEXICONS = Path(__file__).parents[3] / "data" / "lexicons"


@pytest.fixture(scope="module")
def engine() -> RuleEngine:
    return RuleEngine.load(LEXICONS)


def found(engine: RuleEngine, text: str) -> list[tuple[SpanKind, str]]:
    return [(span.kind, text[span.start : span.end]) for span in engine.find(text)]


# ---------------------------------------------------------------- detectors


def test_headline_with_every_kind_of_signal(engine: RuleEngine) -> None:
    text = "SHOCKING: You won't believe what experts say!!!"
    spans = found(engine, text)
    assert (SpanKind.SENSATIONAL, "SHOCKING") in spans
    assert (SpanKind.CLICKBAIT, "You won't believe") in spans
    assert (SpanKind.VAGUE_ATTRIBUTION, "experts say") in spans
    assert (SpanKind.SENSATIONAL, "!!!") in spans
    for span in engine.find(text):
        assert span.label and span.explanation
        assert 0 < span.score <= 1


def test_neutral_reporting_has_no_signals(engine: RuleEngine) -> None:
    text = (
        "The finance ministry said on Tuesday that the annual budget was approved after a "
        "committee review, according to a published statement. Is the plan funded? Yes."
    )
    assert engine.find(text) == []


@pytest.mark.parametrize(
    "text",
    [
        "NASA and the FBI said",
        "NASA FBI CIA officials met",
        "The USA and the EU agreed. OK.",
        "I AM",
    ],
)
def test_acronyms_are_not_shouting(engine: RuleEngine, text: str) -> None:
    assert [s for s in found(engine, text) if s[1].isupper()] == []


@pytest.mark.parametrize(
    ("text", "shouted"),
    [
        ("THIS IS HUGE news for everyone", "THIS IS HUGE"),
        ("They said the FBI IS LYING to you", "FBI IS LYING"),
        ("WAKE UP\nPEOPLE now", "WAKE UP\nPEOPLE"),
        ("ÉNORME SCANDALE ici", "ÉNORME SCANDALE"),
    ],
)
def test_runs_of_capitalised_words_are_flagged(engine: RuleEngine, text: str, shouted: str) -> None:
    assert (SpanKind.SENSATIONAL, shouted) in found(engine, text)


def test_repeated_punctuation(engine: RuleEngine) -> None:
    assert (SpanKind.SENSATIONAL, "?!") in found(engine, "Really?! Yes.")
    assert (SpanKind.SENSATIONAL, "!!!!") in found(engine, "Wake up!!!! Now")
    assert found(engine, "Really? Yes! Fine.") == []


def test_terms_match_whole_words_in_any_case(engine: RuleEngine) -> None:
    assert (SpanKind.SENSATIONAL, "Bombshell") in found(engine, "Bombshell report released")
    assert (SpanKind.SENSATIONAL, "bombshell") in found(engine, "a bombshell report")
    assert found(engine, "The bombshells were defused") == []


def test_phrases_match_across_line_breaks_and_curly_apostrophes(engine: RuleEngine) -> None:
    assert (SpanKind.VAGUE_ATTRIBUTION, "experts\nsay") in found(
        engine, "But experts\nsay otherwise"
    )
    assert (SpanKind.CLICKBAIT, "You won’t believe") in found(engine, "You won’t believe this")
    assert (SpanKind.LOADED_LANGUAGE, "so-called") in found(engine, "the so-called experts")


def test_offsets_are_code_points_after_emoji_and_combining_marks(engine: RuleEngine) -> None:
    text = "😱😱 éà İstanbul\r\nSHOCKING news!!"
    spans = engine.find(text)
    assert {text[s.start : s.end] for s in spans} == {"SHOCKING", "!!"}


@given(st.text(max_size=300))
@settings(max_examples=200, deadline=None)
def test_spans_are_always_in_bounds(engine: RuleEngine, text: str) -> None:
    for span in engine.find(text):
        assert 0 <= span.start < span.end <= len(text)
        assert text[span.start : span.end].strip() != ""


@given(
    st.lists(
        st.sampled_from(
            ["SHOCKING", "you won't believe", "experts say", "!!!", "😱", "é", "\r\n", " ",
             "İ", "ǅ", "ß", "THE TRUTH", "sources say", "a", "?!", "​", "🇺🇸"]
        ),
        max_size=25,
    ).map("".join)
)  # fmt: skip
@settings(max_examples=200, deadline=None)
def test_spans_stay_exact_in_awkward_unicode(engine: RuleEngine, text: str) -> None:
    for span in finalize(engine.find(text), text):
        assert 0 <= span.start < span.end <= len(text)
        assert text[span.start : span.end] == text[span.start : span.end].strip()


# ---------------------------------------------------------------- lexicons


def test_missing_lexicon_directory_fails_loudly(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        RuleEngine.load(tmp_path / "nowhere")


def test_invalid_clickbait_pattern_names_the_culprit(tmp_path: Path) -> None:
    for name in ("sensational.txt", "loaded_language.txt", "vague_attribution.txt", "acronyms.txt"):
        (tmp_path / name).write_text("# empty\n", encoding="utf-8")
    (tmp_path / "clickbait.yaml").write_text(
        "- pattern: 'you won(t believe'\n  label: L\n  explanation: E\n  score: 0.5\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="you won"):
        RuleEngine.load(tmp_path)


def test_empty_lexicons_give_an_engine_that_still_checks_form(tmp_path: Path) -> None:
    for name in ("sensational.txt", "loaded_language.txt", "vague_attribution.txt", "acronyms.txt"):
        (tmp_path / name).write_text("# nothing\n\n", encoding="utf-8")
    (tmp_path / "clickbait.yaml").write_text("[]\n", encoding="utf-8")
    bare = RuleEngine.load(tmp_path)
    assert found(bare, "SHOCKING: experts say WAKE UP NOW!!!") == [
        (SpanKind.SENSATIONAL, "WAKE UP NOW"),
        (SpanKind.SENSATIONAL, "!!!"),
    ]


# ---------------------------------------------------------------- finalize


def raw(start: int, end: int, kind: SpanKind = SpanKind.SENSATIONAL, score: float = 0.5) -> RawSpan:
    return RawSpan(start, end, kind, "label", "explanation", score)


def test_finalize_sorts_numbers_and_marks_rule_origin() -> None:
    text = "0123456789" * 3
    spans = finalize([raw(20, 25), raw(0, 5, SpanKind.CLICKBAIT), raw(10, 12)], text)
    assert [(s.start, s.end) for s in spans] == [(0, 5), (10, 12), (20, 25)]
    assert [s.id for s in spans] == ["s1", "s2", "s3"]
    assert {s.origin for s in spans} == {"rule"}
    assert spans[0].kind == SpanKind.CLICKBAIT


def test_finalize_drops_empty_and_out_of_bounds_spans() -> None:
    text = "short text"
    spans = finalize([raw(3, 3), raw(5, 4), raw(-1, 4), raw(8, 11), raw(0, 5), raw(10, 12)], text)
    assert [(s.start, s.end) for s in spans] == [(0, 5)]


def test_finalize_removes_spans_nested_in_a_span_of_the_same_kind() -> None:
    text = "x" * 40
    spans = finalize(
        [
            raw(0, 20),
            raw(5, 10),  # inside the first, same kind: dropped
            raw(5, 10, SpanKind.CLICKBAIT),  # inside the first, other kind: kept
            raw(15, 25),  # overlaps but is not nested: kept
            raw(30, 35, score=0.2),
            raw(30, 35, score=0.9),  # duplicate: the stronger one is kept
        ],
        text,
    )
    assert [(s.start, s.end, s.kind) for s in spans] == [
        (0, 20, SpanKind.SENSATIONAL),
        (5, 10, SpanKind.CLICKBAIT),
        (15, 25, SpanKind.SENSATIONAL),
        (30, 35, SpanKind.SENSATIONAL),
    ]
    assert spans[-1].score == 0.9
    assert len({s.id for s in spans}) == len(spans)


def test_finalize_of_nothing_is_nothing() -> None:
    assert finalize([], "text") == []
