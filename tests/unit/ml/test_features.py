import numpy as np
import pytest

from fnd_ml.common.features import FEATURE_NAMES, stylometric


def _named(text: str) -> dict[str, float]:
    return dict(zip(FEATURE_NAMES, stylometric(text).tolist(), strict=True))


def test_stylometric_counts() -> None:
    f = _named("WOW!!! You will not believe this. Really?")
    assert f["exclaim_per_100w"] == pytest.approx(300 / 7)
    assert f["question_per_100w"] == pytest.approx(100 / 7)
    assert f["caps_share"] == pytest.approx(1 / 7)
    assert f["mean_sentence_words"] == pytest.approx(7 / 3)
    assert f["second_person_share"] == pytest.approx(1 / 7)


def test_word_shape_features() -> None:
    f = _named("The Senate voted 52 to 48.")
    assert f["log_words"] == pytest.approx(np.log1p(6))
    assert f["mean_word_len"] == pytest.approx(20 / 6)
    assert f["title_case_share"] == pytest.approx(2 / 6)
    assert f["digit_share"] == pytest.approx(4 / 21)
    assert f["caps_share"] == 0.0
    assert f["exclaim_per_100w"] == 0.0


def test_a_single_capital_letter_is_not_a_word_in_capitals() -> None:
    assert _named("I saw A USA flag")["caps_share"] == pytest.approx(1 / 5)


def test_straight_and_curly_quotes_are_counted() -> None:
    assert _named('He said "no" and “never” again')["quote_per_100w"] == pytest.approx(400 / 6)


def test_sentence_lengths_have_a_mean_and_a_spread() -> None:
    f = _named("One two three. Four.")
    assert f["mean_sentence_words"] == pytest.approx(2.0)
    assert f["std_sentence_words"] == pytest.approx(1.0)


def test_a_decimal_point_does_not_end_a_sentence() -> None:
    assert _named("Prices rose 3.5 percent.")["mean_sentence_words"] == pytest.approx(4.0)


def test_type_token_ratio_ignores_case() -> None:
    assert _named("The cat and the dog")["type_token_ratio"] == pytest.approx(4 / 5)


def test_type_token_ratio_reads_only_the_first_600_words() -> None:
    text = " ".join(["same"] * 600 + [f"w{i}" for i in range(100)])
    assert _named(text)["type_token_ratio"] == pytest.approx(1 / 600)


def test_second_person_includes_possessives_and_contractions() -> None:
    assert _named("Your vote is yours; you're right")["second_person_share"] == pytest.approx(3 / 6)


@pytest.mark.parametrize("text", ["", "   ", "!!!", "..."])
def test_stylometric_handles_text_without_words(text: str) -> None:
    features = stylometric(text)
    assert features.shape == (len(FEATURE_NAMES),)
    assert features.dtype == np.float32
    assert np.isfinite(features).all()
