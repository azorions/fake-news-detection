"""Grouped, deterministic train/val/calib/test assignment.

Rows are grouped by who published them, so the test set measures performance on publishers the model
never saw: by source domain when there is a URL, by speaker for LIAR, otherwise each row is its own group.
A group's split is a hash of the group name and the seed, so it never depends on row order.
"""

from __future__ import annotations

import hashlib

import pandas as pd

SPLITS: tuple[str, ...] = ("train", "val", "calib", "test")
_CUTOFFS = ((70, "train"), (80, "val"), (90, "calib"), (100, "test"))


def assign_group(frame: pd.DataFrame) -> pd.Series:
    has_publisher = frame["publisher"].notna()
    has_domain = has_publisher & frame["url"].notna()
    groups = "row:" + frame["text_hash"].astype(str)
    groups = groups.where(~has_publisher, frame["dataset"] + ":" + frame["publisher"].astype(str))
    return groups.where(~has_domain, "domain:" + frame["publisher"].astype(str))


def _bucket(group: str, seed: int) -> int:
    digest = hashlib.sha256(f"{seed}:{group}".encode()).digest()
    return int.from_bytes(digest[:8], "big") % 100


def _split_for(bucket: int) -> str:
    return next(name for cutoff, name in _CUTOFFS if bucket < cutoff)


def assign_split(groups: pd.Series, seed: int = 13) -> pd.Series:
    unique = groups.unique()
    lookup = {group: _split_for(_bucket(str(group), seed)) for group in unique}
    return groups.map(lookup)
