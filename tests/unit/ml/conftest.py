"""Fixtures shared by the slow tests of training and export. PyTorch is loaded only when used."""

from __future__ import annotations

from pathlib import Path

import pytest

from tests.fixtures.processed import write_processed
from tests.fixtures.tiny_deberta import corpus, write_tiny_model


@pytest.fixture(scope="session")
def tiny_model(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A folder that looks like a pretrained checkpoint: encoder weights and a tokenizer."""
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    return write_tiny_model(tmp_path_factory.mktemp("tiny-deberta"))


@pytest.fixture(scope="session")
def tiny_data(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, Path]:
    """The toy corpus as a processed folder, and its lock."""
    return write_processed(tmp_path_factory.mktemp("tiny-data"), corpus())
