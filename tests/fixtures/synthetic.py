"""A small, clearly separable corpus for fast tests: sensational texts (label 1) vs neutral ones (0)."""

from __future__ import annotations

import random

import pandas as pd

_OPENERS = ["SHOCKING:", "BREAKING:", "EXPOSED:", "BOMBSHELL:", "You won't believe this:"]
_SUBJECTS = [
    "the government",
    "a top doctor",
    "the mainstream media",
    "a secret group",
    "big pharma",
]
_CLAIMS = [
    "is hiding the truth about vaccines",
    "admits the election was rigged",
    "covers up a miracle cure",
    "plans to control your mind",
    "destroys the evidence of the hoax",
]
_CLOSERS = [
    "They don't want you to know!!!",
    "Share before it gets deleted!",
    "Wake up, people!!!",
    "What happened next will shock you!",
]
_OFFICIALS = [
    "The finance ministry",
    "A city spokesperson",
    "The central bank",
    "The health agency",
]
_DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
_TOPICS = ["the annual budget", "quarterly inflation", "the transit plan", "hospital funding"]
_DEVELOPMENTS = [
    "rose by two percent compared with last year",
    "was approved after a committee review",
    "will be discussed at the next session",
    "remained unchanged from the previous quarter",
]
_SOURCES = ["a published statement", "official figures", "the meeting minutes", "a press briefing"]


def make_corpus(n: int = 200, seed: int = 0) -> pd.DataFrame:
    rng = random.Random(seed)
    rows = []
    for i in range(n):
        label = i % 2
        if label:
            text = (
                f"{rng.choice(_OPENERS)} {rng.choice(_SUBJECTS)} {rng.choice(_CLAIMS)}! "
                f"{rng.choice(_CLOSERS)}"
            )
        else:
            text = (
                f"{rng.choice(_OFFICIALS)} said on {rng.choice(_DAYS)} that {rng.choice(_TOPICS)} "
                f"{rng.choice(_DEVELOPMENTS)}, according to {rng.choice(_SOURCES)}."
            )
        rows.append(
            {
                "id": f"synthetic:{i}",
                "dataset": "synthetic",
                "title": None,
                "text": text,
                "label": label,
                "publisher": None,
                "url": None,
                "date": None,
                "model_text": text,
                "length_bucket": "headline",
            }
        )
    return pd.DataFrame(rows)


SENSATIONAL_TEXT = (
    "SHOCKING: the government is hiding the truth about vaccines! They don't want you to know!!! "
    "BOMBSHELL: a secret group plans to control your mind! Wake up, people!!!"
)
NEUTRAL_TEXT = (
    "The finance ministry said on Tuesday that the annual budget was approved after a committee "
    "review, according to a published statement. The central bank said on Friday that quarterly "
    "inflation remained unchanged from the previous quarter, according to official figures."
)
