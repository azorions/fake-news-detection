# Fake News Detection

Analyses a news article, headline or URL and returns a verdict with calibrated confidence, highlighted
explanations, source checks and (from milestone M3) evidence for the article's key claims.

The design is in [docs/design.md](docs/design.md). The build is in progress, milestone by milestone; plans
and measured results are in [docs/plans/](docs/plans/).

## Requirements

- Python 3.12 and [uv](https://docs.astral.sh/uv/)
- Optional, for training the transformer: an NVIDIA GPU. The pinned PyTorch build targets CUDA 12.6.

## Setup

```
uv sync                 # API, data pipeline, baseline model, dev tools
uv sync --extra train   # adds PyTorch for transformer training
```

## Common tasks

```
uv run poe check        # lint, type-check, tests
uv run poe test         # tests only
uv run poe cov          # tests with coverage
```
