import pytest
from pydantic import ValidationError

from fnd_api.schemas import (
    AnalysisResponse,
    AnalyzeOptions,
    AnalyzeRequest,
    Depth,
    FeedbackCreate,
    StageEvent,
    Verdict,
)
from tests.fixtures.responses import SAMPLE_TEXT, make_response

# ---------------------------------------------------------------- request


def test_request_accepts_text_or_url_with_default_options() -> None:
    from_text = AnalyzeRequest.model_validate({"text": "  Some claim about the budget.  "})
    assert from_text.text == "Some claim about the budget."
    assert from_text.options == AnalyzeOptions(
        depth=Depth.STANDARD, max_claims=5, allow_external=True, use_cache=True, store=True
    )
    from_url = AnalyzeRequest.model_validate({"url": "https://example.com/news/1"})
    assert str(from_url.url) == "https://example.com/news/1"


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"text": "a claim", "url": "https://example.com/a"},
        {"text": "   "},
        {"text": "x" * 100_001},
        {"url": "ftp://example.com/file"},
        {"url": "not a url"},
        {"text": "a claim", "title": "t" * 501},
        {"text": "a claim", "unknown": 1},
        {"text": "a claim", "options": {"max_claims": 11}},
        {"text": "a claim", "options": {"depth": "extreme"}},
        {"text": "a claim", "options": {"depht": "quick"}},
    ],
)
def test_request_rejects_invalid_input(payload: dict[str, object]) -> None:
    with pytest.raises(ValidationError):
        AnalyzeRequest.model_validate(payload)


def test_text_at_the_limit_is_accepted() -> None:
    assert len(AnalyzeRequest(text="x" * 100_000).text or "") == 100_000


# ---------------------------------------------------------------- response


def test_response_survives_a_json_round_trip() -> None:
    response = make_response()
    restored = AnalysisResponse.model_validate_json(response.model_dump_json())
    assert restored == response
    assert restored.schema_version == "1.0"
    span = restored.highlights[0]
    assert restored.article.text[span.start : span.end] == "SHOCKING"
    assert restored.article.text == SAMPLE_TEXT


def test_satire_score_is_absent_rather_than_a_placeholder() -> None:
    assert make_response().classifier.satire_score is None
    assert '"satire_score":null' in make_response().model_dump_json()


def test_enums_serialize_as_their_wire_values() -> None:
    dumped = make_response(verdict=Verdict.MISLEADING).model_dump(mode="json")
    assert dumped["verdict"] == "potentially_misleading"
    assert dumped["article"]["input_type"] == "text"


@pytest.mark.parametrize(
    "overrides",
    [
        {"risk_score": 1.2},
        {"confidence": -0.1},
        {"verdict": "fake"},
        {"tags": ["made_up_tag"]},
        {"classifier": {"p_unreliable": 1.5}},
    ],
)
def test_response_rejects_out_of_range_values(overrides: dict[str, object]) -> None:
    with pytest.raises(ValidationError):
        make_response(**overrides)


# ---------------------------------------------------------------- feedback and events


def test_feedback_defaults_to_the_verdict_and_limits_the_comment() -> None:
    feedback = FeedbackCreate(rating="disagree", corrected_verdict=Verdict.RELIABLE)
    assert feedback.target_type == "verdict"
    assert feedback.target_id is None
    with pytest.raises(ValidationError):
        FeedbackCreate(rating="disagree", comment="x" * 2001)
    with pytest.raises(ValidationError):
        FeedbackCreate.model_validate({"rating": "maybe"})


def test_stage_event_names_are_restricted_to_the_pipeline_stages() -> None:
    event = StageEvent(stage="classifier", status="ok", data={"p_unreliable": 0.9})  # type: ignore[arg-type]
    assert event.model_dump(mode="json")["status"] == "ok"
    with pytest.raises(ValidationError):
        StageEvent.model_validate({"stage": "magic", "status": "ok", "data": {}})
