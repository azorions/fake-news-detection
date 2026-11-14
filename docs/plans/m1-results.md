# M1 results (measured 2026-11-14)

Branch `m1-mvp`, baseline artifact `v1-69c6a3df`, Windows 11, Python 3.12.10.

## Status

M1 is verified except for one item: the GPU smoke test (design: "GPU smoke test first") has not run.
`uv sync --extra train` (PyTorch, 2.4 GB) was killed for lack of memory on 2026-11-12 and has not been
restarted; on 2026-11-14 the machine had 0.4 GB of 15.9 GB RAM free and 7.6 GB free on C:. Nothing in M1
needs PyTorch, but M2 does. The merge to `main` and the `m1` tag wait for that test and for the
whole-branch review.

## Verification list

| Check | Result | Measured |
|---|---|---|
| `uv run poe check` | pass | ruff clean, mypy clean (80 source files), 381 tests passed in 68 s |
| Line coverage ≥ 85% | pass | `fnd_api` 97.3% (1570/1614), `fnd_ml.common` 100% (112/112), all packages 97.1% |
| Baseline macro-F1 ≥ 0.80 on the grouped test | pass | 0.855 (ROC-AUC 0.941, ECE 0.016, 8,794 rows) |
| Split hygiene | pass | 91,588 rows, 67,285 groups, 0 groups and 0 text hashes shared between splits |
| Pasted article → verdict and spans | pass | `high_risk`, risk 0.94; 14 highlight spans and 40 word weights, every offset points at its text |
| Real article URL → ≥ 200 words, title, date | pass | bbc.com article: 1,002 words, title, date 2019-04-10 from JSON-LD, extraction `full` |
| Same request again → cached, < 100 ms | pass | `cached: true`, same `scan_id`; 2.3 to 3.0 ms round trip |
| 31st analysis in a minute → 429 | pass | 30 × 200, then 429 `rate_limited` with `Retry-After: 60` |
| `http://127.0.0.1/`, `http://169.254.169.254/` | pass | both 422 `fetch_blocked`, `application/problem+json` |
| Scan in `GET /scans`; feedback persists | pass | scan listed; feedback 201, `feedback_count` 1, row present in the SQLite file after the server stopped |
| GPU smoke test (`scripts/smoke_gpu.py`) | **not run** | see Status |

The live checks ran against `uv run poe serve` on 127.0.0.1:8000 with the real baseline artifact and a
temporary data directory.

## Latency (one machine, one client, warm server)

| Request | Time |
|---|---|
| Pasted text, 120 words, not cached | 7 ms in the service, 13 ms round trip |
| 30 different pasted texts in a row | 13 ms each on average |
| URL, not cached (fetch + extract + score) | 432 ms; 1.9 s on the first fetch after start |
| Cache hit | under 1 ms in the service, 2.3 to 3.0 ms round trip |
| First request after start | 767 ms |

## Baseline quality on the test split (publishers never seen in training)

| Slice | Rows | Macro-F1 |
|---|---:|---:|
| All | 8,794 | 0.855 |
| WELFake | 6,301 | 0.921 |
| FakeNewsNet | 1,521 | 0.723 |
| LIAR | 972 | 0.575 |
| Articles | 5,262 | 0.919 |
| Short texts | 795 | 0.946 |
| Headlines | 2,737 | 0.699 |

Verdict thresholds 0.35 / 0.65. Full tables: `ml/reports/baseline_v1.md`.

## Findings for M2

1. **A genuine BBC science article is rated High risk (0.82, medium confidence).** The article is "First
   ever black hole image released" (2019). No rhetoric signals fired; the score comes from many small word
   weights (`this`, `today`, `huge`, `located` up; `said` down). The baseline has learned the style of its
   training sources (wire copy as "reliable"), so explanatory or feature writing from a reputable outlet
   reads as unreliable. In M1 the risk score is the classifier alone; the source term (M3) and the
   transformer with topic-grouped splits (M2) are what should correct this. It belongs in the M2 comparison
   as a named test case.
2. Short statements and headlines are weak (LIAR 0.575, headlines 0.699), as the baseline report says.
3. The model reads only the first 600 words; the BBC article was truncated.

## Notes

- Health endpoints are at `/healthz` and `/readyz`; everything else is under `/api/v1`.
- In the word weights, `token` is the lower-cased form the model saw; `start`/`end` point at the original
  word.
- Rate-limit counters are per process and reset on restart.
