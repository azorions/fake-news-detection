"""Fit a tokenized article into the transformer's input window. Shared by training and serving.

A news article is often longer than the 512 tokens the model reads. The start of an article carries
the claim and the end often carries the conclusion or the call to action, so the body keeps both
ends and drops the middle. The title gets its own, bounded segment in front.
"""

from __future__ import annotations

from collections.abc import Sequence


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
