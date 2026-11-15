# M2 — ML Upgrade Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan
> task-by-task (inline, no subagents: the user's standing instruction). Steps use checkbox (`- [ ]`) syntax.

**Goal:** Frozen, leak-checked splits v2; a like-for-like comparison of the logistic-regression baseline,
XGBoost and a fine-tuned DeBERTa-v3; the transformer served through ONNX Runtime with the baseline as fallback.

**Architecture:** `fnd-ml` gains data hygiene (near-duplicates, probes, topics), two more model families behind
one `Scorer` interface, calibration, export and an evaluation command that enforces the design's gates.
`fnd-api` loads whichever artifacts exist and never imports PyTorch: serving needs only `onnxruntime` and
`tokenizers`.

**Tech Stack:** numpy MinHash, scikit-learn, XGBoost, PyTorch 2.14 (cu126) + transformers, ONNX + ONNX
Runtime, tokenizers, matplotlib, psutil; FastAPI side unchanged.

**Spec:** `docs/design.md` (section 4 in full, the M2 checklist in section 5, the GPU and training-time rows
of section 6). Starting point: `docs/plans/m1-results.md`.

## Global Constraints

- Python `>=3.12,<3.13`. PyTorch pinned to `2.14.*` from the `cu126` index, in the `train` extra only.
- `fnd_api` and everything it imports must load without `torch`, `transformers`, `xgboost` or `matplotlib`.
- Label convention: `1 = unreliable`, `0 = reliable`. `p_unreliable = P(label = 1)`. Seed `13` everywhere.
- Split roles: **val** = model selection and early stopping; **calib** = temperature, isotonic and thresholds
  only; **test** = touched once per model version; **external sets** = never used for training, tuning or
  thresholds.
- Calibrated probabilities are clipped to `[0.01, 0.99]` (M1 ruling).
- Gates, verbatim from the design: 0 publisher groups and 0 near-duplicate clusters shared between splits;
  no source identifier among the top 50 baseline features; baseline macro-F1 ≥ 0.80 on the grouped test;
  transformer ≥ +0.03 macro-F1 over the baseline in distribution and not worse on the mean of external sets;
  ECE ≤ 0.05 (15 bins) and Brier lower than uncalibrated, in distribution; ONNX fp32 max probability
  difference ≤ 0.001 on 500 samples; int8 macro-F1 drop ≤ 0.01, else ship fp32; classifier p95 ≤ 500 ms on
  CPU at 512 tokens.
- A failed gate is reported with its measured value and cause. Nothing is tuned on the test split.
- Unit and integration tests make no network calls and need no GPU or PyTorch. Markers: `slow`, `network`,
  `gpu`. Coverage ≥ 85% on `fnd_api` and `fnd_ml.common`.
- This machine: 16 GB RAM, Quadro P4000 (8 GB), Windows. Every long command checks free memory first and
  stops with a clear message instead of being killed. DataLoader workers: 0.
- Git: branch `m2-ml`, small commits, nothing pushed.

## Decisions made here

- **Task order follows what is unblocked.** Tasks 1–6 need no PyTorch. Task 7 (install + GPU smoke test) needs
  about 4 GB of free RAM and may be run earlier, as soon as the machine has it.
- **Extras:** `classic = [xgboost, matplotlib]`; `train = [fnd-ml[classic], torch, transformers, sentencepiece,
  protobuf, onnx, onnxscript]`. Base gains `onnxruntime`, `tokenizers`, `psutil`. The development environment
  is `uv sync --extra classic` until Task 7, then `uv sync --extra train`.
- **uv cache on D:** `[tool.uv] cache-dir = ".uv-cache"` (git-ignored). C: has 7.6 GB free and the PyTorch
  download alone is 2.4 GB.
- **MinHash is implemented with numpy**, not a library: signatures for ~92k documents take ~50 MB instead of
  several hundred MB of Python dictionaries.
- **Exact and near-duplicate removal keep metadata.** ISOT's 44,898 rows are exact duplicates of WELFake rows
  and were dropped whole in M1, which is why the corpus has no dates. The kept row now takes the first
  non-null `date`, `publisher` and `url` of its duplicates and lists their datasets in `also_in`.
- **Topic groups are a tag, not a split constraint** (design step 7 groups by publisher and near-duplicate
  cluster). They are written to `topics.parquet` beside the frozen splits. Topic-held-out, temporal,
  adversarial and robustness protocols stay in M5; M2 writes the columns they need.
- **Frozen means locked:** the manifest's sha256 is committed in `ml/configs/data_v2.lock`; training and
  evaluation refuse to run against a different manifest.
- **Truncation:** `[CLS] title [SEP] body [SEP]`, title capped at 64 tokens, body keeps the first 75% and the
  last 25% of its budget. The design says head+tail without a ratio; news front-loads its content.
- **Sampling:** within each dataset both labels are drawn equally often; dataset shares stay as they are
  (`w = 0.5 · n_dataset / n_dataset,label`). Loss is then unweighted. With `balance: none` the loss is
  class-weighted instead.
- **int8 and parity are judged on val**, so the test split is still touched once, by the shipped variant.
- **Word highlights stay lexical in M2.** They come from the baseline and say so (`Attribution.model`);
  occlusion and Integrated Gradients for the transformer are M3.
- **Satire:** score ≥ 0.80 adds the `satire` tag and one summary sentence; the verdict does not change.
- **Leave-one-dataset-out for DeBERTa:** `deberta-v3-small`, 20,000 training rows per fold, 2 epochs, 256
  tokens (the design says "capped folds" without numbers).

## Review Focus

1. A headline shorter than 5 words reaches near-duplicate removal: it must be kept and never merged with
   another short text. (Task 1)
2. The transformer artifact exists but is damaged (truncated `model.onnx`): the service must start, serve the
   baseline and say so. (Task 13)
3. A 100,000-character article on the ONNX path: truncated to 512 tokens, `truncated: true`, no error.
   (Task 13)
4. No title, or a 3-word text: the transformer input is `[CLS] body [SEP]` and still valid. (Tasks 9, 13)
5. Training is killed while a checkpoint is being written: `--resume` must start from the last complete
   checkpoint. (Task 9)

---

### Task 1: Dedup v2 — metadata survives, near-duplicates go

**Files:**
- Modify: `ml/src/fnd_ml/data/dedup.py`, `ml/src/fnd_ml/data/schema.py` (optional column `also_in`)
- Create: `ml/src/fnd_ml/data/minhash.py`
- Test: `tests/unit/ml/test_near_dedup.py`

**Interfaces:**
- Produces: `drop_exact_duplicates(frame) -> tuple[pd.DataFrame, dict[str, int]]` (signature unchanged; kept
  row coalesces `date`, `publisher`, `url`; new column `also_in`: sorted, comma-joined other datasets, `""`
  when none).
- Produces: `minhash.signatures(texts: Iterable[str], *, num_perm: int = 128, shingle: int = 5,
  max_tokens: int = 1000, seed: int = 13) -> tuple[NDArray[np.uint32], NDArray[np.bool_]]` (signature matrix
  and a `valid` mask, False for texts with fewer than `shingle` tokens).
- Produces: `minhash.clusters(sig, valid, *, bands: int = 16, rows: int = 8, threshold: float = 0.8)
  -> NDArray[np.int64]` (cluster id per row = smallest row index in its cluster).
- Produces: `drop_near_duplicates(frame, *, threshold: float = 0.8, seed: int = 13)
  -> tuple[pd.DataFrame, dict[str, int]]` on column `model_text`; stats keys `input`,
  `clusters_with_duplicates`, `near_duplicates_removed`, `conflicting_removed`, `output`.
- Produces: `count_cross_split_near_duplicates(frame: pd.DataFrame) -> int`.

- [ ] **Step 1: Write the failing tests**

```python
BASE = " ".join(f"w{i}" for i in range(200))

def test_exact_duplicate_keeps_metadata_from_the_dropped_copy():
    frame = rows(("welfake", BASE, 0, None), ("isot", BASE, 0, "2017-03-01"))   # dataset, text, label, date
    kept, _ = drop_exact_duplicates(frame)
    assert len(kept) == 1 and kept.loc[0, "dataset"] == "welfake"
    assert str(kept.loc[0, "date"])[:10] == "2017-03-01" and kept.loc[0, "also_in"] == "isot"

def test_near_duplicates_form_one_cluster():
    sig, valid = signatures([BASE, BASE.replace("w17", "x17"), " ".join(f"z{i}" for i in range(200))])
    ids = clusters(sig, valid)
    assert ids[0] == ids[1] != ids[2]

def test_half_rewritten_text_stays_apart():      # every second word replaced
    ...  assert ids[0] != ids[1]

def test_short_texts_are_never_merged():
    sig, valid = signatures(["breaking news today", "breaking news today"])
    assert not valid.any() and len(set(clusters(sig, valid))) == 2

def test_signature_is_the_same_in_any_position():
    assert (signatures([BASE])[0][0] == signatures(["other words " * 9, BASE])[0][1]).all()

def test_cluster_with_both_labels_is_dropped_whole(): ...   # stats["conflicting_removed"] == 2
def test_nothing_crosses_splits_after_removal(): ...        # count_cross_split_near_duplicates(...) == 0
```

- [ ] **Step 2: Run** `uv run pytest tests/unit/ml/test_near_dedup.py` — Expected: FAIL (import errors).
- [ ] **Step 3: Implement.** Algorithm (not fixed by the tests):
  - Shingle = 5 consecutive lower-cased tokens from `word_tokens(text, max_tokens)`, joined by a space; shingle
    hash `h = zlib.crc32(...)`.
  - Permutation `i`: `(a_i · h + b_i) mod 4294967291`, `a_i ∈ [1, 2³¹)`, `b_i ∈ [0, 2³¹)` from
    `np.random.default_rng(seed)`, computed in `uint64`; signature = column minimum.
  - 16 bands of 8 columns; rows that share a band key are candidates. In a bucket of ≤ 50 rows compare all
    pairs, in a larger one compare each row with the bucket's first row. Two rows join when the share of
    equal signature positions is ≥ `threshold`. Union-find gives the clusters.
- [ ] **Step 4: Run** the same command — Expected: PASS. Then `uv run pytest tests/unit/ml` — Expected: PASS.
- [ ] **Step 5: Commit** `feat(ml): near-duplicate removal with MinHash; duplicates keep date, publisher and url`.

### Task 2: Leak probes

**Files:**
- Create: `ml/src/fnd_ml/data/leak_probe.py`, `data/registry/source_identifiers.txt`
- Modify: `ml/src/fnd_ml/cli.py` (`fnd data probe`)
- Test: `tests/unit/ml/test_leak_probe.py`

**Interfaces:**
- Consumes: `BaselineModel.top_features(sample_texts, k=50) -> dict[str, list[str]]`.
- Produces: `load_identifiers(path: Path) -> frozenset[str]` (one lower-case term per line, `#` comments).
- Produces: `label_probe(top_features: dict[str, list[str]], identifiers: frozenset[str]) -> list[str]`
  (features, in input order, where a token equals an identifier or the feature equals a two-word identifier).
- Produces: `origin_probe(frame: pd.DataFrame, *, seed: int = 13, max_rows_per_dataset: int = 8000)
  -> dict[str, Any]` with `accuracy`, `majority_baseline`, `macro_f1`, `rows`, and `per_label`
  (`{"0": {...}, "1": {...}}`, the same numbers within one label). Fit on `split == "train"`, scored on `val`.

- [ ] **Step 1: Write the failing tests**

```python
def test_label_probe_flags_identifier_tokens():
    top = {"unreliable": ["shocking", "breitbart news", "via"], "reliable": ["said on", "reuters"]}
    assert label_probe(top, frozenset({"breitbart", "reuters"})) == ["breitbart news", "reuters"]


def test_label_probe_matches_whole_tokens_only():
    assert label_probe({"unreliable": ["report", "part of"]}, frozenset({"rt"})) == []


def test_two_word_identifier():
    assert label_probe({"reliable": ["york times"]}, frozenset({"york times"})) == ["york times"]


def test_origin_probe_separates_distinct_vocabularies():  # dataset A uses words a*, B uses b*
    assert origin_probe(frame)["accuracy"] > 0.95


def test_origin_probe_is_near_chance_for_one_vocabulary():
    r = origin_probe(frame)
    assert r["accuracy"] < r["majority_baseline"] + 0.15
```

- [ ] **Step 2: Run** `uv run pytest tests/unit/ml/test_leak_probe.py` — Expected: FAIL.
- [ ] **Step 3: Implement.** Origin probe: `HashingVectorizer` (word 1–2-grams, 2¹⁸ buckets) + multinomial
  logistic regression. Seed `source_identifiers.txt` with the names found in M1 (reuters, breitbart, sputnik,
  rt, infowars, "york times", siriusxm, …) plus every FakeNewsNet publisher with ≥ 50 rows that is not an
  ordinary English word.
- [ ] **Step 4: Run** — Expected: PASS.
- [ ] **Step 5: Commit** `feat(ml): label and dataset-of-origin leak probes`.

### Task 3: External and satire datasets

**Files:**
- Modify: `data/registry/datasets.yaml`, `data/registry/checksums.json`, `ml/src/fnd_ml/data/loaders/__init__.py`
- Create: `ml/src/fnd_ml/data/loaders/{constraint,buzzfeed_webis,satire_headlines}.py`,
  `tests/fixtures/raw/{constraint,buzzfeed_webis,satire_headlines}/…` (3–4 rows each)
- Test: `tests/unit/ml/test_external_loaders.py`

**Interfaces:**
- Produces: `loaders.EXTERNAL_LOADERS` and `loaders.load_external(name: str, raw_dir: Path) -> pd.DataFrame`
  (common schema). These names are **not** in `loaders.LOADERS`, so `build_corpus` cannot train on them.
- Registry entries carry `role: external` or `role: satire`.

Label maps: Constraint `fake → 1`, `real → 0`. BuzzFeed-Webis `mostly false` and `mixture of true and false
→ 1`, `mostly true → 0`, `no factual content` dropped; `publisher` = the portal. Satire headlines: `label = 1`
means satire (not "unreliable"); `publisher` from the article link.

Source candidates, verified at download: Constraint — GitHub `diptamath/covid_fake_news` (`data/*.csv`);
BuzzFeed-Webis 2016 — Zenodo record 239675 (`articles.zip`); satire — GitHub
`rishabhmisra/News-Headlines-Dataset-For-Sarcasm-Detection` (`Sarcasm_Headlines_Dataset_v2.json`).

- [ ] **Step 1: Write the failing tests** — one per loader on its fixture: row count, label map, dropped
  `no factual content` row, schema passes `validate_frame`; and
  `assert not set(loaders.EXTERNAL_LOADERS) & set(loaders.LOADERS)`.
- [ ] **Step 2: Run** `uv run pytest tests/unit/ml/test_external_loaders.py` — Expected: FAIL.
- [ ] **Step 3: Implement** the loaders and registry entries (URL, license, sha256 recorded at first download).
- [ ] **Step 4: Run** — Expected: PASS.
- [ ] **Step 5: Download for real:** `uv run fnd data download --only constraint --only buzzfeed_webis --only
  satire_headlines` — Expected: three `ok` lines. A source that cannot be fetched without logging in is
  dropped and ledgered (design rule). If both external sets fail, stop and ask.
- [ ] **Step 6: Commit** `feat(ml): external test sets and satire headlines (never used for training)`.

### Task 4: `fnd data build` v2 and the frozen manifest

**Files:**
- Create: `ml/src/fnd_ml/common/resources.py`, `ml/configs/data_v2.lock`
- Modify: `ml/src/fnd_ml/data/build.py`, `ml/src/fnd_ml/data/clean.py`, `ml/src/fnd_ml/cli.py`,
  `ml/pyproject.toml` (`psutil`), `pyproject.toml` (`[tool.uv] cache-dir`), `.gitignore` (`.uv-cache/`)
- Test: `tests/unit/ml/test_build_v2.py`, `tests/unit/ml/test_resources.py`

**Interfaces:**
- Consumes: Task 1 (`drop_near_duplicates`, `count_cross_split_near_duplicates`), Task 3 (`load_external`).
- Produces: `resources.require_free_ram(gb: float, what: str) -> None` (raises `InsufficientMemory` with the
  measured and required amounts); CLI commands call it unless `--force`.
- Produces: `build_corpus(raw_dir, out_dir, seed=13) -> dict[str, Any]`; order: load → clean → exact dedup →
  near dedup → group → split. New column `year` (nullable `Int16`). New manifest keys: `splits_version: 2`,
  `near_dedup`, `shared_near_duplicate_clusters_between_splits`, `dates` (`rows_with_year`, `min_year`,
  `max_year`), `external` (per set: `rows`, `unreliable_share`, `dropped_overlap_with_corpus`).
- Produces: files `eval/{constraint,buzzfeed_webis,satire_headlines}.parquet` (cleaned; rows whose
  `text_hash` is in the corpus removed).
- Produces: `assert_frozen(processed_dir: Path, lock_path: Path = Path("ml/configs/data_v2.lock")) -> str`
  (returns the manifest sha256; raises `ManifestMismatch` when it differs from the lock).

- [ ] **Step 1: Write the failing tests** on a synthetic raw directory: near-duplicate pair leaves one row;
  manifest has `splits_version == 2` and all three `shared_*` counts `== 0`; `year` is set for a dated row;
  an external row whose text is in the corpus is dropped and counted; `assert_frozen` passes on a matching
  lock and raises after the manifest changes; `require_free_ram(10**6, "x")` raises `InsufficientMemory`.
- [ ] **Step 2: Run** `uv run pytest tests/unit/ml/test_build_v2.py tests/unit/ml/test_resources.py` —
  Expected: FAIL.
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run** — Expected: PASS; `uv run pytest tests/unit/ml` — Expected: PASS.
- [ ] **Step 5: Real build** (needs 3 GB free RAM): `uv run fnd data build` — Expected: `shared groups /
  hashes / near-duplicate clusters between splits: 0 / 0 / 0`; `rows_with_year > 0`. Ledger the row counts
  and how many near-duplicates were removed.
- [ ] **Step 6: Freeze:** write the manifest sha256 into `ml/configs/data_v2.lock`.
- [ ] **Step 7: Commit** `feat(ml): data build v2 with near-duplicate removal, years, external sets; splits frozen`.

### Task 5: Baseline v2, character n-gram ablation, XGBoost

**Files:**
- Create: `ml/src/fnd_ml/common/features.py`, `ml/src/fnd_ml/models/xgb.py`,
  `ml/src/fnd_ml/models/scorer.py`, `ml/configs/xgb.yaml`
- Modify: `ml/src/fnd_ml/models/train_baseline.py`, `ml/src/fnd_ml/models/baseline.py` (optional char
  n-grams), `ml/src/fnd_ml/cli.py` (`fnd train xgb`, `fnd train baseline --ablation char`),
  `ml/pyproject.toml` (extra `classic`)
- Test: `tests/unit/ml/test_features.py`, `tests/unit/ml/test_xgb.py`, `tests/unit/ml/test_scorer.py`

**Interfaces:**
- Consumes: Task 2 (`label_probe`, `origin_probe`), Task 4 (`assert_frozen`).
- Produces: `scorer.Scorer = Callable[[pd.DataFrame], NDArray[np.float64]]` (frame with `title`, `text`,
  `model_text` → calibrated `p_unreliable`), `scorer.Fit = Callable[[pd.DataFrame, pd.DataFrame,
  pd.DataFrame], Scorer]` (train, val, calib).
- Produces: `baseline.load_scorer(artifact_dir: Path) -> Scorer`, `baseline.fit_scorer: Fit`;
  `xgb.load_scorer(artifact_dir: Path) -> Scorer`, `xgb.fit_scorer: Fit`.
- Produces: `features.FEATURE_NAMES: tuple[str, ...]` and `features.stylometric(text: str)
  -> NDArray[np.float32]`: `log_words`, `mean_word_len`, `type_token_ratio` (first 600 words), `caps_share`
  (words of ≥ 2 letters in capitals), `title_case_share`, `exclaim_per_100w`, `question_per_100w`,
  `quote_per_100w`, `digit_share`, `mean_sentence_words`, `std_sentence_words`, `second_person_share`.
- Produces: `train_xgb(processed_dir, artifacts_dir, reports_dir, config) -> dict[str, Any]` (report in the
  baseline's shape; artifact kind `xgb`; features = stylometric + `TruncatedSVD(256)` of the hashed TF-IDF,
  fit on train; `XGBClassifier(n_estimators=600, max_depth=6, learning_rate=0.08, subsample=0.8,
  colsample_bytree=0.8, tree_method="hist", early_stopping_rounds=30)` on val; isotonic on calib).
- `train_baseline` now: calls `assert_frozen`, names versions `v2-<hash>`, stores `leak_probe` and
  `origin_probe` in the report, and never scores the test split when the label probe is not empty.

- [ ] **Step 1: Write the failing tests**

```python
def test_stylometric_counts():
    f = dict(zip(FEATURE_NAMES, stylometric("WOW!!! You will not believe this. Really?")))
    assert f["exclaim_per_100w"] == pytest.approx(300 / 7)
    assert f["question_per_100w"] == pytest.approx(100 / 7)
    assert f["caps_share"] == pytest.approx(1 / 7)
    assert f["mean_sentence_words"] == pytest.approx(7 / 3)
    assert f["second_person_share"] == pytest.approx(1 / 7)


def test_stylometric_handles_empty_text():
    assert np.isfinite(stylometric("")).all()


def test_xgb_learns_a_stylistic_rule(tmp_path):  # label 1 = text in capitals with "!!!"
    scorer = xgb.fit_scorer(train, val, calib)
    assert macro_f1(test.label, scorer(test) >= 0.5) > 0.9


def test_scorers_return_one_probability_per_row(): ...  # both families, values in [0.01, 0.99]
def test_training_refuses_a_changed_manifest(tmp_path): ...  # ManifestMismatch
```

- [ ] **Step 2: Run** `uv sync --extra classic`, then `uv run pytest tests/unit/ml/test_features.py
  tests/unit/ml/test_xgb.py tests/unit/ml/test_scorer.py` — Expected: FAIL.
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run** — Expected: PASS.
- [ ] **Step 5: Ablation:** `uv run fnd train baseline --ablation char` on a 20,000-row stratified sample of
  train: word 1–2-grams against word 1–2 + `char_wb` 3–5. Writes `ml/reports/ablation_char_ngrams.md` with val
  macro-F1, ROC-AUC and peak memory. Adopt character n-grams only if val macro-F1 rises by ≥ 0.01 and the
  projected full-data memory is under 6 GB; ledger the decision.
- [ ] **Step 6: Real runs:** `uv run fnd train baseline` — Expected: version `v2-…`, label probe `[]`, test
  macro-F1 ≥ 0.80. A flagged identifier goes on the scrub list, then rebuild and re-freeze (at most two
  rounds, as in M1). `uv run fnd train xgb` — Expected: a report with the same tables.
- [ ] **Step 7: Commit** `feat(ml): baseline v2, stylometric features, XGBoost comparison model`.

### Task 6: Satire scorer

**Files:**
- Create: `ml/src/fnd_ml/models/satire.py`, `ml/src/fnd_ml/inference/satire_scorer.py`, `ml/configs/satire.yaml`
- Modify: `ml/src/fnd_ml/cli.py` (`fnd train satire`)
- Test: `tests/unit/ml/test_satire.py`

**Interfaces:**
- Consumes: `BaselineModel` (same hashed TF-IDF + logistic regression + isotonic; config `n_features: 262144,
  max_tokens: 64, min_df: 2, C: 2.0`), `eval/satire_headlines.parquet` from Task 4.
- Produces: `train_satire(processed_dir, artifacts_dir, reports_dir) -> dict[str, Any]`; artifact kind
  `satire`; split 80/10/10 by `text_hash` bucket (train/calib/test); writes `eval/satire_slice.parquet`
  (the test tenth), which no model trains on.
- Produces: `SatireScorer.load(artifact_dir: Path) -> Self`, `.version: str`,
  `.score(title: str | None, text: str) -> float` (the title when there is one, otherwise the first 40 words).

- [ ] **Step 1: Write the failing tests:** a toy set where satire headlines start with "Area man" scores
  > 0.8 for `"Area man wins argument"` and < 0.2 for a plain headline; `score(None, long_text)` reads only
  the first 40 words (changing word 41 does not change the score); the test tenth is disjoint from train.
- [ ] **Step 2: Run** `uv run pytest tests/unit/ml/test_satire.py` — Expected: FAIL.
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run** — Expected: PASS.
- [ ] **Step 5: Real run:** `uv run fnd train satire` — Expected: ROC-AUC on its test tenth in the report. The
  data has two publishers, so the report states that the score partly reflects their house styles.
- [ ] **Step 6: Commit** `feat(ml): satire headline scorer`.

### Task 7: Training environment and GPU smoke test (closes M1 Task 1)

**Files:**
- Modify: `ml/pyproject.toml` (extra `train`; base `onnxruntime`, `tokenizers`), `uv.lock`, `README.md`,
  `tests/unit/test_imports.py`

- [ ] **Step 1: Write the failing test** `test_api_imports_without_training_packages`: a subprocess imports
  `fnd_api.main` and asserts none of `torch`, `transformers`, `xgboost`, `matplotlib` is in `sys.modules`.
  Run it — Expected: PASS already (it pins the rule for the tasks that follow).
- [ ] **Step 2: Check memory:** at least 4 GB of RAM free. If not, stop and tell the user what to close.
- [ ] **Step 3: Install in the background:** `uv sync --extra train` — Expected: exit 0.
- [ ] **Step 4: Run** `uv run python scripts/smoke_gpu.py` — Expected: exit 0; the architecture list contains
  `sm_61`; one training step prints a finite loss.
- [ ] **Step 5: If it fails:** the design's fallbacks, in order — an older pinned PyTorch, `deberta-v3-small`
  as the final model, a cloud notebook. A driver or other system change needs the user's agreement first.
- [ ] **Step 6: Close M1:** `git branch main <M1 head>` and `git tag m1 <M1 head>` (the last commit of
  `m1-mvp`); nothing is pushed. Update `docs/plans/m1-results.md` (GPU row).
- [ ] **Step 7: Commit** `build: training extra (PyTorch, transformers, ONNX); GPU smoke test passes`.

### Task 8: Topic groups

**Files:**
- Create: `ml/src/fnd_ml/data/topics.py`
- Modify: `ml/src/fnd_ml/cli.py` (`fnd data topics`)
- Test: `tests/unit/ml/test_topics.py`

**Interfaces:**
- Produces: `embed_texts(texts: Sequence[str], *, model_name: str = "BAAI/bge-small-en-v1.5",
  max_length: int = 256, batch_size: int = 64, device: str | None = None) -> NDArray[np.float32]`
  (CLS vector, L2-normalized; imports PyTorch inside the function).
- Produces: `assign_topics(embeddings: NDArray[np.float32], *, k: int = 50, seed: int = 13)
  -> NDArray[np.int32]` (`KMeans(n_init=4)`).
- Produces: `write_topics(processed_dir: Path, *, k: int = 50, seed: int = 13) -> dict[str, Any]`: writes
  `topics.parquet` (`text_hash`, `topic_group`) and `topics.json` (size and ten sample titles per topic);
  caches embeddings in `data/interim/embeddings_bge_small.npy`; leaves the frozen split files untouched.

- [ ] **Step 1: Write the failing tests:** three separated Gaussian blobs with `k=3` → rows of one blob share
  a topic and blobs differ; same seed → same assignment; `write_topics` with a stubbed `embed_texts` leaves
  `splits_manifest.json` byte-identical. One `@pytest.mark.gpu` test: two paraphrases are closer than an
  unrelated sentence.
- [ ] **Step 2: Run** `uv run pytest tests/unit/ml/test_topics.py` — Expected: FAIL.
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run** — Expected: PASS; `uv run pytest -m gpu tests/unit/ml/test_topics.py` — Expected: PASS.
- [ ] **Step 5: Real run:** `uv run fnd data topics` — Expected: 50 topics; ledger the largest topic's share.
- [ ] **Step 6: Commit** `feat(ml): topic groups from bge-small embeddings`.

### Task 9: Transformer fine-tuning code

**Files:**
- Create: `ml/src/fnd_ml/common/truncation.py`, `ml/src/fnd_ml/models/transformer.py`,
  `ml/configs/deberta_small.yaml`, `ml/configs/deberta_base.yaml`
- Modify: `ml/src/fnd_ml/common/text.py`, `ml/src/fnd_ml/cli.py` (`fnd train transformer`)
- Test: `tests/unit/ml/test_truncation.py`, `tests/unit/ml/test_transformer_sampling.py`,
  `tests/unit/ml/test_transformer_train.py` (`slow`)

**Interfaces:**
- Produces: `text.transformer_parts(title: str | None, text: str) -> tuple[str, str]` (scrubbed title, `""`
  when absent, and scrubbed body); used by training and by serving.
- Produces: `truncation.head_tail(body_ids: Sequence[int], budget: int, head_share: float = 0.75)
  -> tuple[list[int], bool]`; `truncation.build_input(title_ids, body_ids, *, cls_id: int, sep_id: int,
  max_length: int = 512, max_title: int = 64, head_share: float = 0.75) -> tuple[list[int], bool]`.
- Produces: `TrainConfig` (yaml): `model_name`, `max_length`, `max_title_tokens=64`, `head_share=0.75`,
  `lr=2e-5`, `weight_decay=0.01`, `epochs`, `batch_size`, `grad_accum`, `warmup_share=0.10`,
  `eval_every_steps`, `checkpoint_every_steps`, `patience=3`, `balance="dataset_label"`, `amp="auto"`,
  `seed=13`, `max_train_rows=None`.
  - `deberta_small.yaml`: `microsoft/deberta-v3-small`, 256 tokens, batch 16, accum 1, 2 epochs, eval 500,
    checkpoint 500.
  - `deberta_base.yaml`: `microsoft/deberta-v3-base`, 512 tokens, batch 8, accum 2, 3 epochs, eval 1000,
    checkpoint 500.
- Produces: `sampling_weights(dataset: Sequence[str], label: Sequence[int]) -> NDArray[np.float64]`;
  `epoch_order(weights, n: int, seed: int, epoch: int) -> NDArray[np.int64]`.
- Produces: `train(config: TrainConfig, processed_dir: Path, run_dir: Path, *, resume: bool = False,
  max_steps: int | None = None) -> dict[str, Any]` (`best_step`, `best_val_macro_f1`, `steps`,
  `seconds_per_step`, `stopped_early`). Writes `run_dir/checkpoint.pt` (written to `checkpoint.pt.tmp`, then
  renamed), `run_dir/best/` (model + tokenizer, with `tokenizer.json`), `run_dir/log.jsonl`,
  `run_dir/config.yaml`.
- Produces: `benchmark_precision(config, processed_dir, steps: int = 50) -> dict[str, Any]` (`fp32` and `amp`:
  `s_per_step`, `peak_mem_mb`, `loss_finite`). `amp: auto` uses mixed precision only when it is ≥ 15% faster
  or fp32 does not fit, and its loss stays finite.
- Produces: `predict_logits(model_dir: Path, frame: pd.DataFrame, *, batch_size: int = 32,
  device: str | None = None) -> NDArray[np.float64]` (margin `z₁ − z₀`), `transformer.load_scorer`,
  `transformer.fit_scorer(config) -> Fit`.

- [ ] **Step 1: Write the failing tests**

```python
def test_head_tail_keeps_both_ends():
    ids, cut = head_tail(list(range(1000)), 100)
    assert cut and ids[:75] == list(range(75)) and ids[75:] == list(range(975, 1000))


def test_short_body_is_untouched():
    assert head_tail([1, 2, 3], 100) == ([1, 2, 3], False)


def test_build_input_layout_and_length():
    ids, cut = build_input([7] * 100, list(range(100, 2000)), cls_id=1, sep_id=2)
    assert (
        len(ids) == 512
        and cut
        and ids[0] == 1
        and ids[65] == 2
        and ids[-1] == 2
        and ids[1:65] == [7] * 64
    )


def test_no_title_gives_cls_body_sep():
    assert build_input([], [5, 6, 7], cls_id=1, sep_id=2) == ([1, 5, 6, 7, 2], False)


def test_weights_balance_labels_inside_each_dataset():
    w = sampling_weights(["a"] * 4 + ["b"] * 6, [1, 0, 0, 0, 1, 1, 1, 1, 1, 0])
    assert w[0] == pytest.approx(w[1:4].sum()) and w[:4].sum() / w.sum() == pytest.approx(0.4)


def test_epoch_order_is_repeatable_and_changes_per_epoch(): ...


@pytest.mark.slow  # tiny random DeBERTa-v2 and a word-level tokenizer built in the test; CPU
def test_training_lowers_the_loss(tmp_path): ...  # last logged loss < 0.5 × first
@pytest.mark.slow
def test_resume_matches_an_uninterrupted_run(
    tmp_path,
): ...  # 40 steps vs 20 + resume to 40: loss within 1e-4
@pytest.mark.slow
def test_resume_ignores_a_half_written_checkpoint(tmp_path): ...  # stray checkpoint.pt.tmp
```

- [ ] **Step 2: Run** `uv run pytest tests/unit/ml/test_truncation.py tests/unit/ml/test_transformer_sampling.py`
  and `uv run pytest -m slow tests/unit/ml/test_transformer_train.py` — Expected: FAIL.
- [ ] **Step 3: Implement** a plain PyTorch loop: tokens encoded once and cached in `run_dir`, AdamW, linear
  warmup then linear decay, gradient accumulation, evaluation on val every `eval_every_steps`, early stop
  after `patience` evaluations without a better val macro-F1. The checkpoint holds model, optimizer,
  scheduler, scaler, step, epoch, best metric and random-generator states.
- [ ] **Step 4: Run** both commands — Expected: PASS.
- [ ] **Step 5: Commit** `feat(ml): resumable DeBERTa fine-tuning with head+tail truncation and balanced sampling`.

### Task 10: Pilot and final training runs

- [ ] **Step 1:** `uv run fnd train transformer --config ml/configs/deberta_small.yaml --benchmark` — Expected:
  fp32 and mixed-precision seconds per step and peak memory; ledger them and the mode chosen.
- [ ] **Step 2: Pilot** in the background: `uv run fnd train transformer --config ml/configs/deberta_small.yaml`.
  After 100 steps read `log.jsonl` and ledger seconds per step and the projected duration. Expected: it
  finishes and reports val macro-F1. If that is more than 0.02 below baseline v2 on val, stop and debug
  before the long run.
- [ ] **Step 3:** `uv run fnd train transformer --config ml/configs/deberta_base.yaml --benchmark`, then project
  the 3-epoch duration. Over 14 hours: 384 tokens, then a stratified 40,000-row training cap (design
  fallbacks); ledger the ruling.
- [ ] **Step 4: Final run** in the background (restart with `--resume` after any interruption). Tasks 11–13
  are developed against the pilot's `best/` while it runs.
- [ ] **Step 5:** Ledger best step, val macro-F1 and wall-clock time. No commit (runs are git-ignored).

### Task 11: Temperature scaling and thresholds

**Files:**
- Create: `ml/src/fnd_ml/calibration/temperature.py`
- Modify: `ml/src/fnd_ml/cli.py` (`fnd calibrate --run <run_dir>`)
- Test: `tests/unit/ml/test_calibration.py`

**Interfaces:**
- Consumes: `predict_logits` (Task 9), `pick_thresholds`, `expected_calibration_error`, `brier`.
- Produces: `fit_temperature(logits: ArrayLike, y: ArrayLike) -> float` (minimizes the negative
  log-likelihood of `sigmoid(z / T)`, `T ∈ [0.05, 20]`); `apply_temperature(logits: ArrayLike, t: float)
  -> NDArray[np.float64]` (clipped to `[0.01, 0.99]`).
- Produces: `run_dir/calibration.json` (`method: "temperature"`, `temperature`, `ece_before`, `ece_after`,
  `brier_before`, `brier_after`) and `run_dir/thresholds.json`, both from the calib split only.

- [ ] **Step 1: Write the failing tests:** 5,000 synthetic rows with labels drawn from `sigmoid(z)` and logits
  reported as `3z` → `fit_temperature` within 10% of 3 and ECE lower after than before; already calibrated
  logits → within 10% of 1; `apply_temperature` never leaves `[0.01, 0.99]`.
- [ ] **Step 2: Run** `uv run pytest tests/unit/ml/test_calibration.py` — Expected: FAIL.
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run** — Expected: PASS.
- [ ] **Step 5: Commit** `feat(ml): temperature scaling and thresholds for the transformer`.

### Task 12: ONNX export, quantization, parity, latency

**Files:**
- Create: `ml/src/fnd_ml/export/{__init__,onnx_export,quantize,parity,benchmark}.py`,
  `scripts/make_tiny_onnx_fixture.py`, `tests/fixtures/tiny_transformer/` (`model.onnx`,
  `tokenizer/tokenizer.json`, `calibration.json`, `thresholds.json`, `metadata.json`)
- Modify: `ml/src/fnd_ml/cli.py` (`fnd export --run <run_dir>`)
- Test: `tests/unit/ml/test_export.py` (`slow`)

**Interfaces:**
- Produces: `export_onnx(model_dir: Path, out_path: Path, *, opset: int = 17) -> Path` (inputs `input_ids`,
  `attention_mask`, int64, dynamic batch and sequence axes; output `logits`).
- Produces: `quantize_int8(fp32_path: Path, out_path: Path) -> Path` (dynamic quantization, `QInt8` weights).
- Produces: `parity(model_dir: Path, onnx_path: Path, frame: pd.DataFrame, *, n: int = 500, seed: int = 13)
  -> dict[str, float]` (`max_abs_probability_diff`, `n`) on val rows.
- Produces: `benchmark(onnx_path: Path, tokenizer_dir: Path, *, lengths: tuple[int, ...] = (128, 256, 512),
  runs: int = 100, warmup: int = 10, threads: int | None = None) -> dict[int, dict[str, float]]`
  (`p50_ms`, `p95_ms`).
- Produces: artifact `ml/artifacts/transformer/<version>/` with `model.onnx` (the shipped variant),
  `tokenizer/tokenizer.json`, `calibration.json`, `thresholds.json`, `metadata.json` (`base_model`, `variant`,
  `max_length`, `max_title_tokens`, `head_share`, git commit, data manifest sha256, val metrics, parity,
  latency, sha256 of `model.onnx`); `version = "v1-" + first 8 hex of that sha256`; registered as kind
  `transformer`.

Shipping rule: int8 ships when its val macro-F1 is within 0.01 of fp32 and its p95 at 512 tokens is lower;
otherwise fp32. A p95 above 500 ms for the shipped variant is reported as a gate exception.

- [ ] **Step 1: Write the failing tests** (tiny random model): ONNX Runtime logits equal PyTorch's within
  `1e-4` for batch sizes 1 and 3 and two sequence lengths; the int8 file is smaller than the fp32 file;
  `parity(...)["max_abs_probability_diff"] <= 0.001`; `benchmark` returns `p95_ms >= p50_ms > 0` per length.
- [ ] **Step 2: Run** `uv run pytest -m slow tests/unit/ml/test_export.py` — Expected: FAIL.
- [ ] **Step 3: Implement.** Read the installed PyTorch's `torch.onnx.export` documentation first; its
  defaults have changed between releases.
- [ ] **Step 4: Run** — Expected: PASS. Then `uv run python scripts/make_tiny_onnx_fixture.py` writes the
  fixture (an embedding → mean → linear graph of a few kilobytes, version `v1-tinyfix`,
  `base_model: "tiny-fixture"`).
- [ ] **Step 5: Real export** once Task 10's final run and Task 11 are done: `uv run fnd calibrate --run …`,
  `uv run fnd export --run …` — Expected: parity ≤ 0.001; latency table; the variant chosen; ledger all three.
- [ ] **Step 6: Commit** `feat(ml): ONNX export, int8 quantization, parity check and latency benchmark`.

### Task 13: Serve the transformer, fall back to the baseline, report satire

**Files:**
- Create: `ml/src/fnd_ml/inference/onnx_classifier.py`
- Modify: `backend/src/fnd_api/models_registry.py`, `backend/src/fnd_api/pipeline/{stages,orchestrator,context}.py`,
  `backend/src/fnd_api/fusion/summary.py`, `backend/src/fnd_api/config.py` (`onnx_threads: int = 4`),
  `backend/src/fnd_api/main.py`, `README.md`
- Test: `tests/unit/ml/test_onnx_classifier.py`, `tests/integration/test_transformer_serving.py`

**Interfaces:**
- Consumes: `transformer_parts`, `build_input` (Task 9), `apply_temperature` (Task 11), the fixture and
  artifact layout (Task 12), `SatireScorer` (Task 6), `ClassifierOutput` (M1).
- Produces: `OnnxClassifier.load(artifact_dir: Path, *, threads: int = 4) -> Self`; `.version: str`;
  `.name: str` (metadata `base_model`); `.thresholds: tuple[float, float]`;
  `.predict(title: str | None, text: str) -> ClassifierOutput`.
- Produces: `ModelRegistry(baseline, transformer, satire)`; `ModelRegistry.load(artifacts_dir: Path, *,
  onnx_threads: int = 4)` (an artifact that fails to load is logged and left `None`);
  `.primary -> BaselineClassifier | OnnxClassifier | None`; `.ready` (a primary exists); `.versions` (every
  loaded model).
- Produces: `stages.classify(models: ModelRegistry, title: str | None, text: str, limiter:
  anyio.CapacityLimiter) -> tuple[ClassifierResult, StageStatus, str | None]`. Transformer absent → baseline,
  `DEGRADED`, detail `"transformer model not available; baseline used"`. Transformer raises → logged,
  baseline, `DEGRADED`, detail `"transformer model failed; baseline used"`.
- `ClassifierResult.model` is the model's name, `calibration` is `"temperature"` or `"isotonic"`,
  `satire_score` is set when the scorer is loaded. Thresholds come from the model that produced the
  probability. `PipelineContext.inference_limiter = anyio.CapacityLimiter(2)`.
- `summary.build(..., satire: bool = False)` appends `"The headline reads like satire."`; the tag `satire`
  is added at `satire_score >= 0.80`.
- Word weights: from the baseline when it is loaded; otherwise `attributions == []` and the `explanations`
  stage is `SKIPPED`.

- [ ] **Step 1: Write the failing tests**

```python
# unit, on tests/fixtures/tiny_transformer
def test_predict_returns_a_calibrated_probability(): ...  # 0.01 <= p <= 0.99, version "v1-tinyfix"
def test_long_text_is_truncated_not_rejected():
    out = clf.predict("A title", "word " * 20000)  # about 100,000 characters
    assert out.truncated and 0.01 <= out.p_unreliable <= 0.99


def test_three_words_without_a_title(): ...  # predict(None, "markets fell today") works
def test_damaged_model_file_raises_on_load(tmp_path): ...  # model.onnx cut to 100 bytes


# integration
async def test_meta_reports_the_transformer(client): ...  # models["transformer"] == "v1-tinyfix"
async def test_transformer_gives_the_verdict(client):
    ...  # classifier.model == "tiny-fixture",
    # calibration == "temperature", stage ok


async def test_without_the_artifact_the_baseline_answers(client):
    ...
    # classifier.model == "baseline"; classifier stage status "degraded", detail contains "baseline used"


async def test_damaged_artifact_does_not_stop_the_service(
    client,
): ...  # /readyz 200, baseline answers
async def test_runtime_failure_falls_back(
    client, monkeypatch
): ...  # 200, "transformer model failed"
async def test_result_is_not_served_from_the_cache_after_a_model_change(
    make_client,
): ...  # cached is False
async def test_satire_adds_a_tag_and_keeps_the_verdict(client): ...
@pytest.mark.slow
async def test_real_artifact_answers(
    real_client,
): ...  # skipped when ml/artifacts has no transformer
```

- [ ] **Step 2: Run** `uv run pytest tests/unit/ml/test_onnx_classifier.py
  tests/integration/test_transformer_serving.py` — Expected: FAIL.
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run** — Expected: PASS; `uv run poe check` — Expected: PASS.
- [ ] **Step 5: Commit** `feat(api): serve the transformer through ONNX Runtime with baseline fallback; satire tag`.

### Task 14: Evaluation protocols, gates and the comparison report

**Files:**
- Create: `ml/src/fnd_ml/eval/{protocols,gates,plots}.py`, `ml/configs/eval_gates.yaml`
- Modify: `ml/src/fnd_ml/eval/report.py`, `ml/src/fnd_ml/cli.py` (`fnd eval`)
- Test: `tests/unit/ml/test_protocols.py`, `tests/unit/ml/test_gates.py`

**Interfaces:**
- Consumes: `Scorer`, `Fit` and the three families' `load_scorer` / `fit_scorer` (Tasks 5, 9),
  `classification_report`, `assert_frozen`.
- Produces: `protocols.in_distribution(scorers: dict[str, Scorer], test: pd.DataFrame) -> dict[str, Any]`
  (per model: overall, by dataset, by length bucket);
  `protocols.leave_one_dataset_out(fit: Fit, corpus: pd.DataFrame) -> dict[str, dict[str, Any]]` (train, val
  and calib rows of the other datasets; scored on the held-out dataset's test rows);
  `protocols.external(scorers, sets: dict[str, pd.DataFrame]) -> dict[str, Any]`.
- Produces: `gates.GateResult(name: str, passed: bool, measured: float | str, threshold: str, note: str)`;
  `gates.evaluate(results: dict[str, Any], gates_path: Path) -> list[GateResult]`.
- Produces: `plots.reliability_diagram(y, p_before, p_after, path: Path, title: str) -> None` (15 bins).
- Produces: `fnd eval --out ml/reports/eval_v1 [--with-transformer-lodo] [--allow-retest "<reason>"]`: writes
  `eval_v1.json`, `eval_v1.md`, `figures/reliability_<model>_<set>.png`; prints the gate table; exits 1 when
  a gate fails (the report is still written). Each model version's test scoring is logged in
  `ml/reports/test_touch_log.json`; scoring it again needs `--allow-retest`, and the reason goes in the report.

- [ ] **Step 1: Write the failing tests:** every gate passes and fails on synthetic results at its exact
  threshold (`+0.03` passes, `+0.029` fails; external mean equal passes, lower fails; ECE `0.05` passes;
  parity `0.001` passes; int8 drop `0.011` → note "ship fp32"; p95 `500` passes); leave-one-dataset-out with
  a recording stub `fit` never receives a row of the held-out dataset; by-dataset row counts add up to the
  total; the markdown has one table per protocol and the gate table; a second scoring of the same version
  raises without `--allow-retest`.
- [ ] **Step 2: Run** `uv run pytest tests/unit/ml/test_protocols.py tests/unit/ml/test_gates.py` — Expected:
  FAIL.
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run** — Expected: PASS.
- [ ] **Step 5: Commit** `feat(ml): evaluation protocols, quality gates and the model comparison report`.

### Task 15: M2 verification

- [ ] `uv run poe check` passes; coverage ≥ 85% on `fnd_api` and `fnd_ml.common`.
- [ ] `uv run fnd eval --out ml/reports/eval_v1 --with-transformer-lodo`: `eval_v1.md` compares LR, XGBoost and
  DeBERTa on macro-F1, ROC-AUC, PR-AUC, Brier and ECE in distribution, leave-one-dataset-out and on the
  external sets, with reliability diagrams. Every gate passes or is reported with its measured value and
  cause.
- [ ] Start `uv run poe serve` against the real artifacts: `/api/v1/meta` reports the transformer version;
  `uv run pytest -m slow tests/integration/test_transformer_serving.py` passes.
- [ ] Spot checks through the running service, baseline against transformer: the pasted text from M1's
  verification and the BBC article that M1 rated High risk (0.82). Record both scores.
- [ ] Move the transformer artifact folder aside and restart: the answer comes from the baseline and the
  `classifier` stage says so. Move it back.
- [ ] Stop the server; write `docs/plans/m2-results.md` with the measured numbers; merge `m2-ml` into `main`,
  tag `m2`. Nothing is pushed.
