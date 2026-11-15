"""One loader per dataset, each returning rows in the common schema (``fnd_ml.data.schema``)."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pandas as pd

from fnd_ml.data.loaders import (
    buzzfeed_webis,
    constraint,
    fakenewsnet,
    isot,
    liar,
    satire_headlines,
    welfake,
)

LOADERS: dict[str, Callable[[Path], pd.DataFrame]] = {
    "welfake": welfake.load,
    "isot": isot.load,
    "fakenewsnet": fakenewsnet.load,
    "liar": liar.load,
}

# Evaluation only. These are deliberately absent from LOADERS, which is all that ``fnd data build``
# reads for training data: external test sets and the satire headlines can never be trained on.
EXTERNAL_LOADERS: dict[str, Callable[[Path], pd.DataFrame]] = {
    "constraint": constraint.load,
    "buzzfeed_webis": buzzfeed_webis.load,
    "satire_headlines": satire_headlines.load,
}

_WIRE_DATELINE = r"\((?:Reuters|AP)\) -"


def load(name: str, raw_dir: Path) -> pd.DataFrame:
    if name not in LOADERS:
        raise KeyError(f"no loader for {name!r}; known: {', '.join(LOADERS)}")
    return LOADERS[name](raw_dir / name)


def load_external(name: str, raw_dir: Path) -> pd.DataFrame:
    if name not in EXTERNAL_LOADERS:
        raise KeyError(f"no evaluation loader for {name!r}; known: {', '.join(EXTERNAL_LOADERS)}")
    return EXTERNAL_LOADERS[name](raw_dir / name)


def load_liar_middle(raw_dir: Path) -> pd.DataFrame:
    """LIAR's half-true and barely-true statements (evaluation only), see ``liar.load_middle``."""
    return liar.load_middle(raw_dir / "liar")


def wire_label_share(frame: pd.DataFrame) -> float:
    """Share of wire-service rows (with a "(Reuters) -" style dateline) that are labelled reliable.

    Wire copy is reliable reporting in every dataset used here, so a value far below 1.0 means the
    loader's label map is inverted. Returns NaN when the frame has no wire rows.
    """
    wire = frame["text"].str.contains(_WIRE_DATELINE, regex=True, na=False)
    if not wire.any():
        return float("nan")
    return float((frame.loc[wire, "label"] == 0).mean())


__all__ = [
    "EXTERNAL_LOADERS",
    "LOADERS",
    "load",
    "load_external",
    "load_liar_middle",
    "wire_label_share",
]
