import itertools

import pytest
from hypothesis import given
from hypothesis import strategies as st

from fnd_ml.common.labels import LABEL_NAMES, RELIABLE, UNRELIABLE
from fnd_ml.common.text import (
    find_artifact_spans,
    model_input,
    normalize_text,
    remove_spans,
    scrub_text,
    text_hash,
)

# ---------------------------------------------------------------- normalize_text


def test_normalize_unifies_newlines_quotes_dashes_and_spaces() -> None:
    assert normalize_text("a\r\nb “q” — c d") == 'a\nb "q" - c d'


def test_normalize_collapses_blank_lines_and_trims_each_line() -> None:
    assert normalize_text("  one  \n\n\n\n two\t three  ") == "one\n\ntwo three"


def test_normalize_removes_control_and_zero_width_characters() -> None:
    assert normalize_text("fa​ke\x00 ne﻿ws") == "fake news"


@given(st.text())
def test_normalize_is_idempotent(text: str) -> None:
    once = normalize_text(text)
    assert normalize_text(once) == once


# ---------------------------------------------------------------- artifact scrubbing


def test_wire_dateline_is_removed() -> None:
    text = "WASHINGTON (Reuters) - The Senate voted on Tuesday."
    assert scrub_text(text) == "The Senate voted on Tuesday."


def test_multi_city_dateline_is_removed() -> None:
    assert scrub_text("NEW YORK/LONDON (Reuters) - Oil prices rose.") == "Oil prices rose."


def test_publisher_names_links_handles_and_image_credits_are_removed() -> None:
    text = (
        "He told Reuters it was fine. Featured image via Getty Images\n"
        "See https://t.co/abc123 and pic.twitter.com/XyZ by @someuser today."
    )
    scrubbed = scrub_text(text)
    for leaked in (
        "Reuters",
        "Featured image",
        "Getty",
        "https://",
        "pic.twitter.com",
        "@someuser",
    ):
        assert leaked not in scrubbed
    for kept in ("He told", "it was fine.", "today."):
        assert kept in scrubbed


def test_plain_text_has_no_artifacts() -> None:
    assert find_artifact_spans("The Senate voted on Tuesday to pass the bill.") == []


def test_artifact_spans_are_sorted_and_disjoint() -> None:
    spans = find_artifact_spans("WASHINGTON (Reuters) - A told Reuters. https://x.co/a @bob")
    assert len(spans) >= 3
    assert spans == sorted(spans)
    assert all(end <= nxt for (_, end), (nxt, _) in itertools.pairwise(spans))


# ---------------------------------------------------------------- remove_spans / model_input


@given(text=st.text(max_size=200), data=st.data())
def test_remove_spans_mapping_points_at_original_characters(text: str, data: st.DataObject) -> None:
    cuts = sorted(data.draw(st.lists(st.integers(0, len(text)), max_size=6)))
    spans = [(cuts[i], cuts[i + 1]) for i in range(0, len(cuts) - 1, 2)]

    reduced, mapping = remove_spans(text, spans)

    assert len(mapping) == len(reduced)
    assert all(text[mapping[i]] == reduced[i] for i in range(len(reduced)))
    assert len(reduced) == len(text) - sum(end - start for start, end in spans)


def test_model_input_maps_body_offsets_and_marks_title_as_unmapped() -> None:
    title = "Senate passes bill"
    text = "WASHINGTON (Reuters) - The Senate voted."

    model_text, mapping = model_input(title, text)

    assert model_text == "Senate passes bill\n\nThe Senate voted."
    body_start = len(title) + 2
    assert all(index == -1 for index in mapping[:body_start])
    assert all(text[mapping[i]] == model_text[i] for i in range(body_start, len(model_text)))


def test_model_input_without_title_is_the_scrubbed_body() -> None:
    model_text, mapping = model_input(None, "The Senate voted.")
    assert model_text == "The Senate voted."
    assert mapping == list(range(len("The Senate voted.")))


def test_model_input_scrubs_the_title_too() -> None:
    model_text, _ = model_input("Oil rises - Reuters", "Prices went up.")
    assert "Reuters" not in model_text
    assert model_text.endswith("\n\nPrices went up.")


# ---------------------------------------------------------------- hashing and labels


def test_text_hash_ignores_case_punctuation_and_spacing() -> None:
    assert text_hash("Hello,   World!") == text_hash("hello world")
    assert text_hash("hello world") != text_hash("hello there")


def test_label_convention_is_one_for_unreliable() -> None:
    assert (UNRELIABLE, RELIABLE) == (1, 0)
    assert LABEL_NAMES == {0: "reliable", 1: "unreliable"}


# ---------------------------------------------------------------- dataset-specific artifacts
# Each case is a pattern that the leak probe found among the baseline's strongest features.


@pytest.mark.parametrize(
    ("text", "leaked"),
    [
        (
            "Great news https: . pic. twitter. - Us Weekly () January 11, 2017, Jenner fought back",
            ["https", "pic", "twitter", "Us Weekly", "2017"],
        ),
        (
            "He writes books. Follow him on Twitter at @x. He can be reached at joe@mail. com",
            ["Follow him", "Twitter", "reached", "mail"],
        ),
        ("That is correct. Via: cns news", ["Via", "cns"]),
        ("the coach stated.Read more: Daily Caller", ["Read more", "Daily Caller"]),
        ("according to the Times.For entire story: The Wrap", ["For entire story", "Wrap"]),
        ("the plaque (Source: TMZ ) / The real kicker", ["Source", "TMZ"]),
        ("Next Prev Swipe left/right 5 vegan alternatives", ["Swipe", "Prev"]),
        (
            "Posted on October 27, 2016 by Paul Joseph Watson Bureaucrat said",
            ["Posted on", "October", "2016", "Watson"],
        ),
        ("TRUNEWS 11/02/16 Jim Rickards", ["11/02/16", "TRUNEWS"]),
        ("the conservative Breitbart News stated", ["Breitbart"]),
        ("Giving States Option - The New York Times", ["New York Times"]),
        ("MOSCOW (Sputnik) - Putin arrived", ["Sputnik", "MOSCOW"]),
        ("in a TV interview with RT , that he knows", ["RT"]),
        ("It happened in 2016 and again in 1998.", ["2016", "1998"]),
    ],
)
def test_dataset_specific_artifacts_are_removed(text: str, leaked: list[str]) -> None:
    scrubbed = scrub_text(text)
    for fragment in leaked:
        assert fragment not in scrubbed


@pytest.mark.parametrize(
    "text",
    [
        "Follow Trump's impact on energy and healthcare.",
        "Water arrives via the main pipeline.",
        "The source of the river is unknown.",
        "They react badly to the news and read more books.",
        "Around 2500 people and 1200 police attended.",
    ],
)
def test_ordinary_sentences_survive_scrubbing(text: str) -> None:
    assert scrub_text(text) == text
