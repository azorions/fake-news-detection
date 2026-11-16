"""Fit a tokenized article into the transformer's input window. Shared by training and serving.

A news article is often longer than the 512 tokens the model reads. The start of an article carries
the claim and the end often carries the conclusion or the call to action, so the body keeps both
ends and drops the middle. The title gets its own, bounded segment in front.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


def head_tail(
    body_ids: Sequence[int], budget: int, head_share: float = 0.75
) -> tuple[list[int], bool]:
    """Keep at most ``budget`` tokens: the first ``head_share`` of the budget and the rest from the end.

    Returns the kept tokens and whether any token was dropped.
    """
    if len(body_ids) <= budget:
        return list(body_ids), False
    if budget <= 0:
        return [], True
    head = min(budget, max(1, int(budget * head_share)))
    tail = budget - head
    kept = list(body_ids[:head])
    if tail:
        kept.extend(body_ids[-tail:])
    return kept, True


def build_input(
    title_ids: Sequence[int],
    body_ids: Sequence[int],
    *,
    cls_id: int,
    sep_id: int,
    max_length: int = 512,
    max_title: int = 64,
    head_share: float = 0.75,
) -> tuple[list[int], bool]:
    """Return ``[CLS] title [SEP] body [SEP]`` (``[CLS] body [SEP]`` without a title) and a cut flag.

    The title keeps its first ``max_title`` tokens; the body gets the room that is left and keeps
    its head and tail. The flag is true when any title or body token was dropped.
    """
    title = list(title_ids[:max_title])
    prefix = [cls_id, *title, sep_id] if title else [cls_id]
    body, body_cut = head_tail(body_ids, max_length - len(prefix) - 1, head_share)
    return [*prefix, *body, sep_id], body_cut or len(title_ids) > max_title


@dataclass(frozen=True)
class InputSpec:
    """How an article becomes input ids. Saved with the model, so serving builds the same input."""

    max_length: int
    max_title_tokens: int
    head_share: float
    cls_id: int
    sep_id: int
    pad_id: int

    def build(self, title_ids: Sequence[int], body_ids: Sequence[int]) -> tuple[list[int], bool]:
        """``build_input`` with this model's settings."""
        return build_input(
            title_ids,
            body_ids,
            cls_id=self.cls_id,
            sep_id=self.sep_id,
            max_length=self.max_length,
            max_title=self.max_title_tokens,
            head_share=self.head_share,
        )


def pad_batch(
    rows: Sequence[Sequence[int] | NDArray[np.int32]], pad_id: int
) -> tuple[NDArray[np.int64], NDArray[np.int64]]:
    """``input_ids`` and ``attention_mask`` of the rows, padded to the longest of them."""
    width = max(len(row) for row in rows)
    input_ids = np.full((len(rows), width), pad_id, dtype=np.int64)
    mask = np.zeros((len(rows), width), dtype=np.int64)
    for index, row in enumerate(rows):
        input_ids[index, : len(row)] = row
        mask[index, : len(row)] = 1
    return input_ids, mask
