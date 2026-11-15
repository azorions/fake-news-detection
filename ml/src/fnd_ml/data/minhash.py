"""MinHash signatures and LSH clustering for near-duplicate detection.

Two texts are near-duplicates when the Jaccard similarity of their 5-word shingle sets is at least
the threshold. The similarity is estimated from MinHash signatures, and candidate pairs are found
by banding, so no pair of unrelated texts is ever compared.
"""

from __future__ import annotations

import zlib
from collections.abc import Iterable

import numpy as np
from numpy.typing import NDArray

from fnd_ml.common.tokens import word_tokens

_PRIME = 4294967291  # the largest prime below 2**32, so every signature value fits in uint32
_MAX_ALL_PAIRS = 50  # buckets up to this size are compared pair by pair


def signatures(
    texts: Iterable[str],
    *,
    num_perm: int = 128,
    shingle: int = 5,
    max_tokens: int = 1000,
    seed: int = 13,
) -> tuple[NDArray[np.uint32], NDArray[np.bool_]]:
    """One MinHash signature per text, and a mask that is False where a text has fewer than
    ``shingle`` words (its signature is all zeros and it must not be compared)."""
    rng = np.random.default_rng(seed)
    a = rng.integers(1, 2**31, size=num_perm, dtype=np.uint64)
    b = rng.integers(0, 2**31, size=num_perm, dtype=np.uint64)

    rows: list[NDArray[np.uint32]] = []
    valid: list[bool] = []
    for text in texts:
        tokens = [token for token, _, _ in word_tokens(text, max_tokens)]
        if len(tokens) < shingle:
            rows.append(np.zeros(num_perm, dtype=np.uint32))
            valid.append(False)
            continue
        hashes = np.fromiter(
            {
                zlib.crc32(" ".join(tokens[i : i + shingle]).encode("utf-8"))
                for i in range(len(tokens) - shingle + 1)
            },
            dtype=np.uint64,
        )
        # a < 2**31 and hash < 2**32, so the product and the sum stay below 2**64.
        permuted = (hashes[:, None] * a[None, :] + b[None, :]) % np.uint64(_PRIME)
        rows.append(permuted.min(axis=0).astype(np.uint32))
        valid.append(True)

    matrix = np.vstack(rows) if rows else np.zeros((0, num_perm), dtype=np.uint32)
    return matrix, np.array(valid, dtype=np.bool_)


def clusters(
    sig: NDArray[np.uint32],
    valid: NDArray[np.bool_],
    *,
    bands: int = 16,
    rows: int = 8,
    threshold: float = 0.8,
) -> NDArray[np.int64]:
    """Cluster id per row: the smallest row index of its near-duplicate cluster.

    Rows that share all ``rows`` values of any band are candidates; candidates join when the share
    of equal signature positions (the Jaccard estimate) is at least ``threshold``. Clusters are the
    connected components, so a chain of small edits ends up in one cluster.
    """
    n = len(sig)
    parent = np.arange(n, dtype=np.int64)

    def find(i: int) -> int:
        root = i
        while parent[root] != root:
            root = int(parent[root])
        while parent[i] != root:
            parent[i], i = root, int(parent[i])
        return root

    def union(i: int, j: int) -> None:
        first, second = find(i), find(j)
        if first != second:
            parent[max(first, second)] = min(first, second)

    index = np.flatnonzero(valid)
    if len(index) > 1:
        multipliers = np.random.default_rng(0).integers(1, 2**63, size=rows, dtype=np.uint64) | 1
        for band in range(bands):
            block = sig[index, band * rows : (band + 1) * rows].astype(np.uint64)
            keys = (block * multipliers[None, :]).sum(axis=1, dtype=np.uint64)  # wraps on purpose
            order = np.argsort(keys, kind="stable")
            sorted_keys = keys[order]
            starts = np.flatnonzero(np.r_[True, sorted_keys[1:] != sorted_keys[:-1]])
            ends = np.r_[starts[1:], len(order)]
            for start, end in zip(starts, ends, strict=True):
                if end - start < 2:
                    continue
                members = index[order[start:end]]  # ascending row indices (stable sort)
                rows_sig = sig[members]
                if len(members) <= _MAX_ALL_PAIRS:
                    same = (rows_sig[:, None, :] == rows_sig[None, :, :]).mean(axis=2)
                    for i, j in zip(*np.nonzero(np.triu(same >= threshold, k=1)), strict=True):
                        union(int(members[i]), int(members[j]))
                else:
                    same_as_first = (rows_sig == rows_sig[0]).mean(axis=1)
                    for j in np.flatnonzero(same_as_first[1:] >= threshold) + 1:
                        union(int(members[0]), int(members[j]))

    return np.array([find(i) for i in range(n)], dtype=np.int64)
