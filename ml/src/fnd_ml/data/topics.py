"""``fnd data topics``: give every corpus text a topic group.

A model can look strong because its test split repeats the subjects of its train split. The topic
groups let the evaluation hold whole subjects out: each text is embedded with a small sentence
model and the embeddings are clustered with k-means. The result is written next to the frozen
splits; the split files and their manifest are not changed.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from sklearn.cluster import KMeans

from fnd_ml.data.build import MANIFEST_NAME
from fnd_ml.data.download import sha256_file

EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
CACHE_NAME = "embeddings_bge_small.npy"
SAMPLE_TITLES = 10
SAMPLE_CHARS = 100


def embed_texts(
    texts: Sequence[str],
    *,
    model_name: str = EMBEDDING_MODEL,
    max_length: int = 256,
    batch_size: int = 64,
    device: str | None = None,
) -> NDArray[np.float32]:
    """Return one L2-normalized CLS vector per text (CUDA when there is a device, else CPU)."""
    import torch
    from transformers import AutoModel, AutoTokenizer

    target = device or ("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(target).eval()

    # Longest first: a batch is padded only to its own longest text, and a batch that does not
    # fit in memory fails at the start instead of after most of the work.
    order = sorted(range(len(texts)), key=lambda i: len(texts[i]), reverse=True)
    vectors: NDArray[np.float32] | None = None
    with torch.inference_mode():
        for start in range(0, len(order), batch_size):
            rows = order[start : start + batch_size]
            encoded = tokenizer(
                [texts[i] for i in rows],
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            ).to(target)
            cls = model(**encoded).last_hidden_state[:, 0]
            batch = torch.nn.functional.normalize(cls, dim=-1).float().cpu().numpy()
            if vectors is None:
                vectors = np.empty((len(texts), batch.shape[1]), dtype=np.float32)
            vectors[rows] = batch
    if vectors is None:
        return np.empty((0, int(model.config.hidden_size)), dtype=np.float32)
    return vectors


def assign_topics(
    embeddings: NDArray[np.float32], *, k: int = 50, seed: int = 13
) -> NDArray[np.int32]:
    """Cluster the embeddings into ``k`` topic groups; the same seed gives the same groups."""
    labels = KMeans(n_clusters=k, n_init=4, random_state=seed).fit_predict(embeddings)
    return np.asarray(labels, dtype=np.int32)


def _embeddings(processed_dir: Path, texts: Sequence[str]) -> NDArray[np.float32]:
    """Embed the texts, or load the vectors a previous run cached for this exact build."""
    interim = processed_dir.parent / "interim"
    cache = interim / CACHE_NAME
    stamp_path = cache.with_suffix(".json")
    stamp = {
        "manifest_sha256": sha256_file(processed_dir / MANIFEST_NAME),
        "model": EMBEDDING_MODEL,
        "rows": len(texts),
    }
    reusable = cache.exists() and stamp_path.exists()
    if reusable and json.loads(stamp_path.read_text(encoding="utf-8")) == stamp:
        cached: NDArray[np.float32] = np.load(cache)
        return cached

    vectors = embed_texts(texts, model_name=EMBEDDING_MODEL)
    interim.mkdir(parents=True, exist_ok=True)
    np.save(cache, vectors)
    stamp_path.write_text(json.dumps(stamp, indent=2) + "\n", encoding="utf-8")
    return vectors


def _sample_titles(frame: pd.DataFrame, rng: np.random.Generator) -> list[str]:
    """Up to ten titles of one topic; a row without a title is shown by the start of its text."""
    count = min(SAMPLE_TITLES, len(frame))
    chosen = frame.iloc[np.sort(rng.choice(len(frame), size=count, replace=False))]
    samples = []
    for title, text in zip(chosen["title"], chosen["model_text"], strict=True):
        shown = title if isinstance(title, str) and title.strip() else text
        samples.append(" ".join(str(shown).split())[:SAMPLE_CHARS])
    return samples


def write_topics(processed_dir: Path, *, k: int = 50, seed: int = 13) -> dict[str, Any]:
    """Write ``topics.parquet`` and ``topics.json`` into ``processed_dir`` and return the summary.

    ``topics.parquet`` has one row per corpus text (``text_hash``, ``topic_group``), in corpus
    order. ``topics.json`` lists each topic's size and sample titles, so a person can check what
    the groups mean, and the largest topic's share, which shows a degenerate clustering.
    """
    corpus = pd.read_parquet(
        processed_dir / "corpus.parquet", columns=["text_hash", "title", "model_text"]
    )
    vectors = _embeddings(processed_dir, corpus["model_text"].tolist())
    labels = assign_topics(vectors, k=k, seed=seed)
    pd.DataFrame({"text_hash": corpus["text_hash"], "topic_group": labels}).to_parquet(
        processed_dir / "topics.parquet", index=False
    )

    rng = np.random.default_rng(seed)
    topics: list[dict[str, Any]] = []
    largest = 0
    for topic in np.unique(labels).tolist():
        members = corpus[labels == topic]
        largest = max(largest, len(members))
        topics.append(
            {
                "topic_group": topic,
                "size": len(members),
                "sample_titles": _sample_titles(members, rng),
            }
        )
    summary: dict[str, Any] = {
        "k": k,
        "seed": seed,
        "model": EMBEDDING_MODEL,
        "rows": len(corpus),
        "largest_topic_share": round(largest / len(corpus), 4),
        "topics": topics,
    }
    (processed_dir / "topics.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return summary
