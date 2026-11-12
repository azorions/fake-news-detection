"""One loader per dataset, each returning rows in the common schema (``fnd_ml.data.schema``)."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pandas as pd

from fnd_ml.data.loaders import fakenewsnet, isot, liar, welfake

LOADERS: dict[str, Callable[[Path], pd.DataFrame]] = {
    "welfake": welfake.load,
    "isot": isot.load,
    "fakenewsnet": fakenewsnet.load,
    "liar": liar.load,
}

_WIRE_DATELINE = r"\((?:Reuters|AP)\) -"


def load(name: str, raw_dir: Path) -> pd.DataFrame:
    if name not in LOADERS:
        raise KeyError(f"no loader for {name!r}; known: {', '.join(LOADERS)}")
    return LOADERS[name](raw_dir / name)


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


__all__ = ["LOADERS", "load", "load_liar_middle", "wire_label_share"]
