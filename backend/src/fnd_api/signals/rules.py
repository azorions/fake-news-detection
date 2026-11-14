"""Rule-based rhetoric signals: wording and form that are typical of unreliable articles.

Every detector searches the text as given and reports code point offsets into it. The text is
never lowercased or otherwise transformed first, because that can change its length.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Self

import yaml

from fnd_api.schemas import SpanKind

_WORD = re.compile(r"[^\W\d_]+(?:['’][^\W\d_]+)*")
_REPEATED_PUNCTUATION = re.compile(r"[!?]{2,}")
_MIN_SHOUTED_WORDS = 2


@dataclass(frozen=True)
class RawSpan:
    start: int
    end: int
    kind: SpanKind
    label: str
    explanation: str
    score: float


@dataclass(frozen=True)
class _Rule:
    pattern: re.Pattern[str]
    kind: SpanKind
    label: str
    explanation: str
    score: float


_TERM_LISTS = (
    (
        "sensational.txt",
        SpanKind.SENSATIONAL,
        "Sensational wording",
        "A word chosen to provoke a reaction rather than to describe.",
        0.6,
    ),
    (
        "loaded_language.txt",
        SpanKind.LOADED_LANGUAGE,
        "Loaded language",
        "A word that passes judgement instead of describing.",
        0.5,
    ),
    (
        "vague_attribution.txt",
        SpanKind.VAGUE_ATTRIBUTION,
        "Vague attribution",
        "The claim is credited to nobody who can be checked.",
        0.5,
    ),
)


def _flexible(pattern: str) -> str:
    """Let a plain apostrophe match a curly one and a space match any run of whitespace."""
    return pattern.replace("'", "['’]").replace(" ", r"\s+")


def _bounded(pattern: str) -> re.Pattern[str]:
    return re.compile(rf"(?<!\w)(?:{pattern})(?!\w)", re.IGNORECASE)


def _read_lines(path: Path) -> list[str]:
    lines = (line.strip() for line in path.read_text(encoding="utf-8").splitlines())
    return [line for line in lines if line and not line.startswith("#")]


def _term_rule(
    path: Path, kind: SpanKind, label: str, explanation: str, score: float
) -> _Rule | None:
    terms = sorted(set(_read_lines(path)), key=len, reverse=True)  # longest first
    if not terms:
        return None
    alternatives = "|".join(_flexible(re.escape(term).replace(r"\ ", " ")) for term in terms)
    return _Rule(_bounded(alternatives), kind, label, explanation, score)


def _clickbait_rules(path: Path) -> list[_Rule]:
    entries = yaml.safe_load(path.read_text(encoding="utf-8")) or []
    rules = []
    for entry in entries:
        source = str(entry["pattern"])
        try:
            pattern = _bounded(_flexible(source))
        except re.error as error:
            raise ValueError(f"invalid clickbait pattern {source!r}: {error}") from error
        rules.append(
            _Rule(
                pattern,
                SpanKind.CLICKBAIT,
                str(entry["label"]),
                str(entry["explanation"]),
                float(entry["score"]),
            )
        )
    return rules


class RuleEngine:
    def __init__(self, rules: list[_Rule], acronyms: frozenset[str]) -> None:
        self._rules = rules
        self._acronyms = acronyms

    @classmethod
    def load(cls, lexicon_dir: Path) -> Self:
        """Read the lexicons. A missing file or an invalid pattern is an error at startup."""
        rules = _clickbait_rules(lexicon_dir / "clickbait.yaml")
        for name, kind, label, explanation, score in _TERM_LISTS:
            rule = _term_rule(lexicon_dir / name, kind, label, explanation, score)
            if rule is not None:
                rules.append(rule)
        acronyms = frozenset(line.upper() for line in _read_lines(lexicon_dir / "acronyms.txt"))
        return cls(rules, acronyms)

    def find(self, text: str) -> list[RawSpan]:
        """All signals in ``text``, in no particular order; spans may overlap."""
        spans = [
            RawSpan(match.start(), match.end(), rule.kind, rule.label, rule.explanation, rule.score)
            for rule in self._rules
            for match in rule.pattern.finditer(text)
        ]
        spans.extend(self._shouting(text))
        spans.extend(
            RawSpan(
                match.start(),
                match.end(),
                SpanKind.SENSATIONAL,
                "Repeated punctuation",
                "Stacked exclamation or question marks add emphasis, not information.",
                0.4,
            )
            for match in _REPEATED_PUNCTUATION.finditer(text)
        )
        return spans

    def _shouting(self, text: str) -> list[RawSpan]:
        """Runs of two or more words in capitals, not counting known acronyms."""
        found: list[RawSpan] = []
        run: list[re.Match[str]] = []

        def close() -> None:
            shouted = [word for word in run if word.group() not in self._acronyms]
            if len(shouted) >= _MIN_SHOUTED_WORDS:
                found.append(
                    RawSpan(
                        run[0].start(),
                        run[-1].end(),
                        SpanKind.SENSATIONAL,
                        "Shouting in capitals",
                        "Whole words in capitals are used for emphasis in place of evidence.",
                        0.5,
                    )
                )
            run.clear()

        for word in _WORD.finditer(text):
            capitals = len(word.group()) >= 2 and word.group().isupper()
            if run and not (capitals and text[run[-1].end() : word.start()].isspace()):
                close()
            if capitals:
                run.append(word)
        if run:
            close()
        return found
