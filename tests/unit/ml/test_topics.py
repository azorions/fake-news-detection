import json
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from numpy.typing import NDArray

from fnd_ml.common.text import text_hash
from fnd_ml.data import topics
from fnd_ml.data.build import MANIFEST_NAME
from fnd_ml.data.topics import assign_topics, embed_texts, write_topics

CENTERS = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
SUBJECTS = ("budget", "football", "vaccine")


def _blobs(per_blob: int = 40, seed: int = 0) -> NDArray[np.float32]:
    """Three tight clouds around three orthogonal directions; rows 0-39, 40-79, 80-119."""
    rng = np.random.default_rng(seed)
    points = np.vstack([center + 0.05 * rng.normal(size=(per_blob, 3)) for center in CENTERS])
    return points.astype(np.float32)


def test_rows_of_one_blob_share_a_topic_and_blobs_differ() -> None:
    assigned = assign_topics(_blobs(), k=3, seed=13)

    assert assigned.dtype == np.int32
    assert assigned.shape == (120,)
    per_blob = [set(assigned[start : start + 40].tolist()) for start in (0, 40, 80)]
    assert all(len(found) == 1 for found in per_blob)
    assert len(set.union(*per_blob)) == 3


def test_the_same_seed_gives_the_same_assignment() -> None:
    blobs = _blobs()
    np.testing.assert_array_equal(
        assign_topics(blobs, k=3, seed=13), assign_topics(blobs, k=3, seed=13)
    )


def _write_corpus(root: Path, per_subject: int = 40) -> Path:
    """A processed folder whose corpus has three subjects; the subject is the first word."""
    processed = root / "data" / "processed"
    processed.mkdir(parents=True)
    rows = [
        {"title": f"{subject} story {i}" if i % 2 else None, "model_text": f"{subject} text {i}"}
        for subject in SUBJECTS
        for i in range(per_subject)
    ]
    frame = pd.DataFrame(rows)
    frame["text_hash"] = [text_hash(text) for text in frame["model_text"]]
    frame.to_parquet(processed / "corpus.parquet", index=False)
    (processed / MANIFEST_NAME).write_text(json.dumps({"rows": {"total": len(frame)}}))
    return processed


class FakeEmbedder:
    """Stands in for the real model: a text is placed by its first word."""

    def __init__(self) -> None:
        self.calls = 0

    def __call__(self, texts: Sequence[str], **_: object) -> NDArray[np.float32]:
        self.calls += 1
        rng = np.random.default_rng(1)
        rows = [CENTERS[SUBJECTS.index(text.split()[0])] for text in texts]
        return (np.array(rows) + 0.05 * rng.normal(size=(len(texts), 3))).astype(np.float32)


@pytest.fixture
def embedder(monkeypatch: pytest.MonkeyPatch) -> FakeEmbedder:
    fake = FakeEmbedder()
    monkeypatch.setattr(topics, "embed_texts", fake)
    return fake


def test_write_topics_gives_every_text_a_topic_and_leaves_the_manifest_alone(
    embedder: FakeEmbedder, tmp_path: Path
) -> None:
    processed = _write_corpus(tmp_path)
    manifest_before = (processed / MANIFEST_NAME).read_bytes()
    corpus_before = (processed / "corpus.parquet").read_bytes()

    summary = write_topics(processed, k=3, seed=13)

    assert (processed / MANIFEST_NAME).read_bytes() == manifest_before
    assert (processed / "corpus.parquet").read_bytes() == corpus_before
    written = pd.read_parquet(processed / "topics.parquet")
    corpus = pd.read_parquet(processed / "corpus.parquet")
    assert list(written.columns) == ["text_hash", "topic_group"]
    assert written["text_hash"].tolist() == corpus["text_hash"].tolist()
    subject = corpus["model_text"].str.split().str[0]
    assert written.groupby(subject.to_numpy())["topic_group"].nunique().tolist() == [1, 1, 1]
    assert written["topic_group"].nunique() == 3
    assert summary["rows"] == 120
    assert summary["largest_topic_share"] == pytest.approx(1 / 3, abs=1e-4)
    assert json.loads((processed / "topics.json").read_text(encoding="utf-8")) == summary


def test_the_summary_lists_size_and_at_most_ten_sample_titles_per_topic(
    embedder: FakeEmbedder, tmp_path: Path
) -> None:
    processed = _write_corpus(tmp_path)

    summary = write_topics(processed, k=3, seed=13)

    assert [topic["topic_group"] for topic in summary["topics"]] == [0, 1, 2]
    assert [topic["size"] for topic in summary["topics"]] == [40, 40, 40]
    for topic in summary["topics"]:
        samples = topic["sample_titles"]
        assert len(samples) == 10
        # All ten come from one subject; a row without a title is shown by the start of its text.
        assert len({sample.split()[0] for sample in samples}) == 1
        assert all(sample.split()[1] in {"story", "text"} for sample in samples)


def test_embeddings_are_cached_next_to_the_processed_folder(
    embedder: FakeEmbedder, tmp_path: Path
) -> None:
    processed = _write_corpus(tmp_path)

    first = write_topics(processed, k=3, seed=13)
    second = write_topics(processed, k=3, seed=13)

    assert embedder.calls == 1
    assert second == first
    assert (tmp_path / "data" / "interim" / "embeddings_bge_small.npy").exists()


def test_a_cache_from_another_build_is_not_reused(embedder: FakeEmbedder, tmp_path: Path) -> None:
    processed = _write_corpus(tmp_path)
    write_topics(processed, k=3, seed=13)
    (processed / MANIFEST_NAME).write_text(json.dumps({"rows": {"total": 120}, "seed": 14}))

    write_topics(processed, k=3, seed=13)

    assert embedder.calls == 2


@pytest.mark.gpu
def test_paraphrases_are_closer_than_an_unrelated_sentence() -> None:
    vectors = embed_texts(
        [
            "The central bank raised interest rates by half a point on Tuesday.",
            "On Tuesday the central bank increased its interest rate by 0.5 points.",
            "The football team won the championship after a penalty shootout.",
        ]
    )

    assert vectors.shape == (3, 384)
    np.testing.assert_allclose(np.linalg.norm(vectors, axis=1), 1.0, atol=1e-4)
    assert vectors[0] @ vectors[1] > vectors[0] @ vectors[2] + 0.1
