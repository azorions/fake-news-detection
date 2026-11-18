"""A toy DeBERTa and the corpus it is trained on, for tests of training, export and scoring.

The model has the DeBERTa-v3 layout (relative attention, no token types) at a size that trains on
a CPU in a second. Its tokenizer knows the words of the synthetic corpus and nothing else.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
from tokenizers import Tokenizer, models, normalizers, pre_tokenizers

from fnd_ml.models.transformer import TrainConfig
from tests.fixtures.processed import BOUNDS, two_dataset_corpus

STEPS_PER_EPOCH = 30  # 240 train rows in batches of 8


def corpus() -> pd.DataFrame:
    """The synthetic two-dataset corpus. Every third row has a title, the others have none."""
    frame = two_dataset_corpus()
    titles = [
        " ".join(text.split()[:3]) if index % 3 == 0 else None
        for index, text in enumerate(frame["text"])
    ]
    return frame.assign(title=titles)


def split(name: str) -> pd.DataFrame:
    start, end = BOUNDS[name]
    return corpus().iloc[start:end]


def write_tiny_model(directory: Path) -> Path:
    """Write what a pretrained checkpoint has: encoder weights and a tokenizer. Needs PyTorch."""
    import torch
    import transformers

    split_words = pre_tokenizers.Whitespace()
    words = sorted(
        {
            word
            for text in corpus()["text"]
            for word, _ in split_words.pre_tokenize_str(text.lower())
        }
    )
    vocab = {
        token: index for index, token in enumerate(["[PAD]", "[UNK]", "[CLS]", "[SEP]", *words])
    }
    backend = Tokenizer(models.WordLevel(vocab, unk_token="[UNK]"))
    backend.normalizer = normalizers.Lowercase()
    backend.pre_tokenizer = split_words
    transformers.PreTrainedTokenizerFast(
        tokenizer_object=backend,
        unk_token="[UNK]",
        pad_token="[PAD]",
        cls_token="[CLS]",
        sep_token="[SEP]",
    ).save_pretrained(directory)

    config = transformers.DebertaV2Config(
        vocab_size=len(vocab),
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=2,
        intermediate_size=64,
        max_position_embeddings=64,
        type_vocab_size=0,
        relative_attention=True,
        position_buckets=16,
        norm_rel_ebd="layer_norm",
        share_att_key=True,
        pos_att_type=["p2c", "c2p"],
        position_biased_input=False,
        pooler_hidden_size=32,
    )
    torch.manual_seed(0)
    transformers.DebertaV2Model(config).save_pretrained(directory)
    return directory


def tiny_config(model_dir: Path, **changes: Any) -> TrainConfig:
    """Settings that train the toy model to a clear result in 90 steps."""
    settings: dict[str, Any] = {
        "model_name": str(model_dir),
        "max_length": 48,
        "max_title_tokens": 8,
        "lr": 3e-3,
        "weight_decay": 0.0,
        "epochs": 3,
        "batch_size": 8,
        "eval_every_steps": 30,
        "checkpoint_every_steps": 20,
        "log_every_steps": 5,
    }
    return TrainConfig(**(settings | changes))
