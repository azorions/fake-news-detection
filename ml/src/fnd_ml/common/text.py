"""Text normalization and artifact scrubbing, shared by training and serving.

Two different things happen to text, and they must not be confused:

* ``normalize_text`` produces the text the system stores, displays and reports offsets against.
* Scrubbing (``find_artifact_spans`` + ``remove_spans``) produces the text the *models* read. It removes
  strings that identify the publisher instead of describing the content, such as wire datelines. Those
  strings predict the label in the public datasets without saying anything about reliability.

``model_input`` returns the scrubbed text together with a mapping back to the normalized text, so that
explanations computed on the model's input can be reported as offsets the user can see.
"""

from __future__ import annotations

import hashlib
import re
import unicodedata

# ---------------------------------------------------------------- normalization

_EXPLICIT_CHARS: dict[str, str] = {
    # single quotes and primes
    "‘": "'", "’": "'", "‚": "'", "‛": "'", "′": "'",
    # double quotes and guillemets
    "“": '"', "”": '"', "„": '"', "‟": '"', "″": '"',
    "«": '"', "»": '"',
    # hyphens and dashes
    "‐": "-", "‑": "-", "‒": "-", "–": "-", "—": "-", "―": "-",
    "−": "-",
    # everything that ends a line becomes a newline
    "\n": "\n", "\r": "\n", "\x0b": "\n", "\x0c": "\n", "\x85": "\n", " ": "\n",
    " ": "\n",
    "\t": " ",
}  # fmt: skip

_REMOVED_CATEGORIES = frozenset({"Cc", "Cf", "Co", "Cs"})  # control, format, private use, surrogate


class _CharTable(dict[int, int | str | None]):
    """Translation table that classifies each character once, the first time it is seen."""

    def __missing__(self, codepoint: int) -> int | str | None:
        category = unicodedata.category(chr(codepoint))
        result: int | str | None
        if category in _REMOVED_CATEGORIES:
            result = None
        elif category == "Zs":
            result = " "
        else:
            result = codepoint
        self[codepoint] = result
        return result


_CHAR_TABLE = _CharTable({ord(char): replacement for char, replacement in _EXPLICIT_CHARS.items()})
_SPACES_RE = re.compile(r" {2,}")
_SPACE_AROUND_NEWLINE_RE = re.compile(r" ?\n ?")
_BLANK_LINES_RE = re.compile(r"\n{3,}")


def _normalize_once(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("\r\n", "\n").translate(_CHAR_TABLE)
    text = _SPACES_RE.sub(" ", text)
    text = _SPACE_AROUND_NEWLINE_RE.sub("\n", text)
    text = _BLANK_LINES_RE.sub("\n\n", text)
    return text.strip(" \n")


def normalize_text(text: str) -> str:
    """Return the canonical form of ``text``. Idempotent.

    Applies NFKC, turns typographic quotes and dashes into ASCII, removes control and zero-width
    characters, collapses runs of spaces, trims every line and keeps at most one blank line.
    """
    # Deleting a character can put two characters next to each other that NFKC then combines, so a
    # single pass is not always a fixed point. Two passes almost always are; the bound is a safety net.
    for _ in range(5):
        normalized = _normalize_once(text)
        if normalized == text:
            break
        text = normalized
    return text


# ---------------------------------------------------------------- artifact scrubbing

# Every pattern below removes something that identifies where or when a text was published instead of
# what it says. Each one was added because the leak probe (the baseline's strongest features) showed
# the models using it as a shortcut for the label. Removing a year or an outlet name costs a little
# content; keeping it teaches the model "written in late 2016" or "mentions Breitbart" instead of style.

_MONTH = (
    r"(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|June?|July?|Aug(?:ust)?"
    r"|Sep(?:t(?:ember)?)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\.?"
)
_DATE = rf"{_MONTH} \d{{1,2}}(?:st|nd|rd|th)?,? (?:19|20)\d{{2}}"
_NAME = r"[A-Z][\w.'-]+(?: [A-Z][\w.'-]+){0,2}"
_OUTLETS = (
    r"Reuters|Breitbart(?: News| Texas| London)?|(?:The )?New York Times"
    r"|(?i:sputnik)(?: International| news)?|RT(?: International)?|21st Century Wire|21WIRE"
    r"|Getty Images|Daily Caller|TRUNEWS|Infowars|WND"
)

_ARTIFACT_PATTERNS: tuple[re.Pattern[str], ...] = (
    # Wire dateline at the start of the text or of a line: "WASHINGTON (Reuters) - "
    re.compile(
        r"(?m)^[A-Z][A-Za-z .,'/\-]{0,60}\((?:Reuters|AP|AFP|(?i:sputnik)|TASS|Xinhua)\)\s*-\s*"
    ),
    # Attribution left behind by an embedded tweet: "- Us Weekly () January 11, 2017,"
    re.compile(rf"-\s[^()\n]{{2,50}}\(@?\w{{0,20}}\)\s?{_DATE},?"),
    re.compile(rf"(?i:posted on) {_DATE}(?: by {_NAME})?"),
    # Image credits, up to the end of the sentence or line
    re.compile(
        r"(?i)\b(?:featured image|images?|photos?)\s+(?:via|by|credit:?|courtesy of)\b[^\n.]{0,80}\.?"
    ),
    # Source credits and "read on" teasers
    re.compile(r"\((?:Image )?Source:[^)\n]{0,60}\)"),
    re.compile(r"\b(?:Via|Source|Read more|For entire story|Related articles?)\s?:[^\n.]{0,40}"),
    re.compile(r"\b(?:READ MORE|Continue this story at)\b[^\n.:]{0,40}:?"),
    re.compile(r"(?:Next )?(?:Prev )?Swipe left/right"),
    # Author follow and contact lines
    re.compile(
        rf"\bFollow (?:him|her|them|us|{_NAME}) on (?:Twitter|Facebook|Instagram)[^\n.]{{0,30}}\.?"
    ),
    re.compile(r"\b(?:or )?e-?mail (?:the author|him|her|them)(?: at)?\b"),
    re.compile(r"\b(?:He|She) can be reached at\b"),
    re.compile(r"[\w.+-]+@[\w-]+\.\s?[A-Za-z]{2,}"),
    # Links, including the mangled remains of stripped ones ("https: .", "pic. twitter.")
    re.compile(r"https?://\S+|www\.\S+|\bhttps?:(?:\s?\.)?"),
    re.compile(r"\bpic\.\s?twitter\.(?:\s?com/\w+)?"),
    re.compile(r"(?<![\w@])@\w{2,15}\b"),
    # Dates and years: the datasets were collected in narrow time windows that differ by label
    re.compile(_DATE),
    re.compile(r"\b\d{1,2}/\d{1,2}/\d{2,4}\b"),
    re.compile(r"\b(?:19|20)\d{2}\b"),
    re.compile(rf"\b(?:{_OUTLETS})\b"),
)


def find_artifact_spans(text: str) -> list[tuple[int, int]]:
    """Return sorted, non-overlapping ``(start, end)`` spans of publisher-identifying artifacts."""
    found = sorted(
        match.span() for pattern in _ARTIFACT_PATTERNS for match in pattern.finditer(text)
    )
    merged: list[tuple[int, int]] = []
    for start, end in found:
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged


def remove_spans(text: str, spans: list[tuple[int, int]]) -> tuple[str, list[int]]:
    """Delete ``spans`` (sorted, non-overlapping) from ``text``.

    Returns the reduced text and ``mapping``, where ``mapping[i]`` is the index in ``text`` of the
    ``i``-th character of the reduced text.
    """
    pieces: list[str] = []
    mapping: list[int] = []
    cursor = 0
    for start, end in spans:
        if start > cursor:
            pieces.append(text[cursor:start])
            mapping.extend(range(cursor, start))
        cursor = max(cursor, end)
    if cursor < len(text):
        pieces.append(text[cursor:])
        mapping.extend(range(cursor, len(text)))
    return "".join(pieces), mapping


def scrub_text(text: str) -> str:
    """Return ``text`` with artifacts removed. Equal to ``model_input(None, text)[0]``."""
    return remove_spans(text, find_artifact_spans(text))[0]


def model_input(title: str | None, text: str) -> tuple[str, list[int]]:
    """Build the string the classifiers read, plus a mapping back into ``text``.

    The result is ``"<scrubbed title>\\n\\n<scrubbed text>"`` (or just the scrubbed text when there is
    no title). ``mapping[i]`` is the index in ``text`` of character ``i`` of the result, or ``-1`` for
    characters that come from the title or the separator.
    """
    body, mapping = remove_spans(text, find_artifact_spans(text))
    clean_title = scrub_text(title).strip(" -|:\n") if title else ""
    if not clean_title:
        return body, mapping
    prefix = clean_title + "\n\n"
    return prefix + body, [-1] * len(prefix) + mapping


# ---------------------------------------------------------------- hashing

_WORD_RE = re.compile(r"\w+")


def text_hash(text: str) -> str:
    """Hash that ignores case, punctuation and spacing. Used for exact-duplicate detection."""
    canonical = " ".join(_WORD_RE.findall(text.lower()))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()
