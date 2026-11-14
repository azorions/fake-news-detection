import pytest

from fnd_api.ingest.language import detect_language

ENGLISH = (
    "The city council voted on Thursday evening to approve a flood barrier along the eastern bank "
    "of the river, ending a review that lasted a little more than two years."
)
SPANISH = (
    "El ayuntamiento votó el jueves por la noche a favor de construir una barrera contra las "
    "inundaciones en la orilla oriental del río, después de una revisión de más de dos años."
)


def test_english_paragraph() -> None:
    assert detect_language(ENGLISH) == "en"


@pytest.mark.parametrize(
    ("text", "code"),
    [
        (SPANISH, "es"),
        ("Der Stadtrat hat am Donnerstagabend dem Bau einer Hochwasserschutzmauer zugestimmt.", "de"),
        ("Le conseil municipal a approuvé jeudi soir la construction d'une digue le long du fleuve.", "fr"),
        ("Городской совет в четверг вечером одобрил строительство защитной дамбы вдоль реки.", "ru"),
    ],
)  # fmt: skip
def test_other_languages(text: str, code: str) -> None:
    assert detect_language(text) == code


@pytest.mark.parametrize(
    "text",
    ["Senate passes bill", "BREAKING NEWS", "Obama", "", "   ", "12345 67890 !!! ??? ... ---"],
)
def test_too_short_to_tell_is_none(text: str) -> None:
    assert detect_language(text) is None


@pytest.mark.parametrize(
    "text",
    [
        "SHOCKING: You won't believe what the senator said about vaccines!!!",
        "Trump says Obama wiretapped Trump Tower during the election",
        "Senate passes sweeping tax bill after marathon overnight session",
    ],
)
def test_english_headlines_are_never_taken_for_another_language(text: str) -> None:
    assert detect_language(text) in ("en", None)


def test_english_text_quoting_another_language_is_still_english() -> None:
    mixed = ENGLISH + ' The mayor ended with the words "muchas gracias a todos".'
    assert detect_language(mixed) == "en"
