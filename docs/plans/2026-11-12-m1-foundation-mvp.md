# M1 — Foundation and MVP Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan
> task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A working vertical slice: datasets in, a calibrated TF-IDF baseline out, and a FastAPI service that
analyses pasted text or a URL and returns a verdict with exact highlight offsets, history and feedback.

**Architecture:** A uv workspace with two packages. `fnd-ml` owns data, training and the inference/explain
code shared with serving. `fnd-api` owns the HTTP service and imports `fnd-ml` without training dependencies.
The pipeline is one async generator of stage events so the SSE endpoint in M3 needs no rewrite.

**Tech Stack:** Python 3.12, uv, scikit-learn, pandas/pyarrow, Typer, FastAPI, Pydantic v2, SQLAlchemy 2 async
+ aiosqlite, Alembic, structlog, slowapi, diskcache/redis, httpx, trafilatura, extruct, tldextract, lingua,
pytest, respx, hypothesis, ruff, mypy, poethepoet.

**Spec:** `docs/design.md` (sections 2, 3, 4 steps 1–8 and 11–12, and the M1 checklist in section 5).

## Global Constraints

- Python `>=3.12,<3.13`. PyTorch pinned to `2.14.*` from the `cu126` index (training extra only).
- Label convention everywhere: `1 = unreliable`, `0 = reliable`. `p_unreliable = P(label = 1)`.
- All offsets are **Unicode code point** indices into `article.text`, end exclusive, and
  `article.text[start:end]` must equal the highlighted string.
- `article.text` is the normalized text. Scrubbing for the model never changes the stored text.
- Limits: 30 analyses/min, 120 reads/min, 100,000 characters of text, 5 MB per fetched page, 8 s fetch timeout.
- Scan-result cache TTL 24 h. Cache key covers canonical input, options and model versions.
- Unit and integration tests make no network calls and need no GPU. Markers: `slow`, `network`, `gpu`.
- Gates: line coverage ≥ 85% on `fnd_api` and `fnd_ml.common`; baseline macro-F1 ≥ 0.80 on the grouped test.
- English only: other languages return `unsupported_language` (HTTP 422).
- Errors are `application/problem+json` with `code` from the design's list.
- Git: small commits on `m1-mvp`, nothing pushed.

## Deviations from the design, decided here

- **Baseline features are word 1–2-grams only in M1.** Character n-grams on ~100k long documents risk
  exhausting 16 GB RAM and their benefit is unmeasured; they become an M2 ablation on a subsample.
- **`classifier.satire_score` is `float | None`**, `None` until the satire scorer exists in M2, so no
  placeholder number looks like a measurement.
- **Publisher grouping** uses the URL domain (FakeNewsNet) or speaker (LIAR). WELFake and ISOT carry no
  publisher field, so their rows group by text hash; M2 adds topic groups for them.
- **`confidence` in M1** is `max(p, 1 - p)` of the calibrated risk, capped by degradations. The fused
  measurement described in the design arrives with fusion in M3.

## Review Focus

1. **Emoji, combining marks and CRLF in input.** Offsets must still satisfy `text[start:end]`; tested by a
   hypothesis property test in Task 10 and an API test in Task 11.
2. **Public URL that redirects to a private address.** Must be blocked at the hop, not followed; Task 9.
3. **Page that yields almost no text** (JS-rendered or paywalled). Must return a `partial_text` result with
   capped confidence or `content_too_short`, never a confident verdict; Tasks 9 and 11.
4. **Very short or mixed-language text.** Language detection must not reject a 3-word English headline;
   Task 9.
5. **Two identical requests at the same moment.** Both must succeed without a database error; Task 11.

---

### Task 1: Workspace scaffold and GPU smoke test

**Files:**
- Create: `pyproject.toml`, `.python-version`, `.gitignore`, `.env.example`, `.pre-commit-config.yaml`,
  `README.md`, `ml/pyproject.toml`, `ml/src/fnd_ml/__init__.py`, `backend/pyproject.toml`,
  `backend/src/fnd_api/__init__.py`, `scripts/smoke_gpu.py`, `tests/conftest.py`,
  `tests/unit/test_imports.py`

**Interfaces:**
- Produces: packages `fnd_ml` and `fnd_api` importable; poe tasks `lint`, `types`, `test`, `check`, `serve`;
  extra `train` on `fnd-ml`; `scripts/smoke_gpu.py` exits 0 only if CUDA works.

- [ ] Write `tests/unit/test_imports.py`: `import fnd_ml, fnd_api`; assert both expose `__version__ == "0.1.0"`.
- [ ] Create the workspace files. Root `pyproject.toml` holds `[tool.uv.workspace] members = ["ml", "backend"]`,
      the `pytorch-cu126` index (`explicit = true`), ruff (line length 100), mypy (`disallow_untyped_defs`),
      pytest (`asyncio_mode = "auto"`, markers), coverage and poe config.
- [ ] `uv sync --all-packages`; run `uv run poe check`. Expected: lint, types and 1 test pass.
- [ ] `scripts/smoke_gpu.py`: print torch version, `torch.cuda.get_arch_list()`, device name; assert
      `torch.cuda.is_available()`, assert an `sm_61` entry is in the arch list, run one forward/backward/step
      of a small `nn.Linear` on GPU, print peak memory.
- [ ] `uv sync --all-packages --extra train`; run `uv run python scripts/smoke_gpu.py`. Expected: exit 0.
      If it fails, record the error and apply the design's fallbacks before continuing.
- [ ] Commit.

### Task 2: Text utilities

**Files:**
- Create: `ml/src/fnd_ml/common/__init__.py`, `ml/src/fnd_ml/common/text.py`,
  `ml/src/fnd_ml/common/labels.py`
- Test: `tests/unit/ml/test_text.py`

**Interfaces:**
- Produces:
  - `normalize_text(text: str) -> str` — NFKC, CRLF→LF, curly quotes/dashes to ASCII, control chars removed,
    runs of spaces collapsed, paragraphs kept as `\n\n`, stripped.
  - `find_artifact_spans(text: str) -> list[tuple[int, int]]` — datelines (`CITY (Reuters) -`), bylines,
    "Featured image via…", `pic.twitter.com/…`, `@handles`, URLs, known publisher names.
  - `remove_spans(text: str, spans: list[tuple[int, int]]) -> tuple[str, list[int]]` — returns the reduced
    text and `mapping` where `mapping[i]` is the original index of reduced character `i`.
  - `model_input(title: str | None, text: str) -> tuple[str, list[int]]` — scrubbed model text plus mapping
    into `text` (title characters map to `-1`).
  - `text_hash(text: str) -> str` — sha256 of lowercase, punctuation-stripped, whitespace-collapsed text.
  - `labels.py`: `UNRELIABLE = 1`, `RELIABLE = 0`, `LABEL_NAMES`.

- [ ] Tests: `normalize_text("a\r\nb “q” — c d")` equals `'a\nb "q" - c d'`; idempotence
      (`normalize(normalize(x)) == normalize(x)`, hypothesis); `find_artifact_spans` catches
      `"WASHINGTON (Reuters) - The Senate"` and leaves `"The Senate"`; `remove_spans` mapping satisfies
      `original[mapping[i]] == reduced[i]` for all `i` (hypothesis); `text_hash("Hello, World!") ==
      text_hash("hello world")`.
- [ ] Run tests, expect failures; implement; run, expect pass; commit.

### Task 3: Dataset registry, download and loaders

**Files:**
- Create: `data/registry/datasets.yaml`, `ml/src/fnd_ml/data/__init__.py`, `ml/src/fnd_ml/data/download.py`,
  `ml/src/fnd_ml/data/schema.py`, `ml/src/fnd_ml/data/loaders/{__init__,welfake,isot,fakenewsnet,liar}.py`,
  `ml/src/fnd_ml/cli.py`
- Test: `tests/unit/ml/test_loaders.py`, `tests/unit/ml/test_download.py`

**Interfaces:**
- Produces:
  - `schema.COLUMNS = ["id", "dataset", "title", "text", "label", "publisher", "url", "date"]`;
    `validate_frame(df) -> None` raises on wrong columns, nulls in `text`/`label`, or labels outside `{0, 1}`.
  - `download.fetch_dataset(name: str, raw_dir: Path, registry: Path) -> list[Path]` — streams files, writes
    sha256 to `data/registry/checksums.json` on first fetch and verifies on later ones.
  - `loaders.load(name: str, raw_dir: Path) -> pd.DataFrame` returning `schema.COLUMNS`.
  - CLI: `fnd data download [--only NAME]`.
- Registry candidates (verified by the first download; unreachable ones are dropped and logged):
  WELFake (Zenodo record 4561253), ISOT, FakeNewsNet CSVs (KaiDMML/FakeNewsNet `dataset/`), LIAR
  (`liar_dataset.zip`, UCSB).
- LIAR mapping: `true`, `mostly-true` → 0; `false`, `pants-fire` → 1; `half-true`, `barely-true` → excluded
  here and written to `data/interim/liar_middle.parquet`.

- [ ] Tests on tiny in-test files written to `tmp_path` in each dataset's real layout: each loader returns
      `schema.COLUMNS`; LIAR maps the six labels as above; FakeNewsNet derives `publisher` from `news_url`;
      `validate_frame` rejects label `2`; `fetch_dataset` (respx-mocked) writes a checksum and raises
      `ChecksumMismatch` when the bytes change.
- [ ] Implement; run tests; `uv run fnd data download`.
- [ ] **Label sanity check on real data:** in WELFake and ISOT, rows containing `(Reuters) -` must map to
      label 0 in ≥ 95% of cases. WELFake's published label description is known to be disputed, so the loader's
      map is set from this check, and the check is kept as a `network`-free assertion inside `fnd data build`.
- [ ] Commit (code and registry only; `data/raw/` is ignored).

### Task 4: Clean, deduplicate, split

**Files:**
- Create: `ml/src/fnd_ml/data/clean.py`, `ml/src/fnd_ml/data/dedup.py`, `ml/src/fnd_ml/data/splits.py`,
  `ml/src/fnd_ml/data/build.py`
- Test: `tests/unit/ml/test_clean_dedup_splits.py`

**Interfaces:**
- Consumes: `normalize_text`, `find_artifact_spans`, `remove_spans`, `text_hash`, `loaders.load`.
- Produces:
  - `clean.clean_frame(df) -> pd.DataFrame` — adds `text_clean` (normalized + scrubbed), `n_words`,
    `length_bucket` (`headline` ≤ 40 words, `short` ≤ 150, else `article`), `text_hash`; drops rows with
    fewer than 3 words.
  - `dedup.drop_exact_duplicates(df) -> tuple[pd.DataFrame, dict]` — one row per `text_hash`; hashes with
    conflicting labels are dropped entirely; returns counts.
  - `splits.assign_group(df) -> pd.Series` — URL domain, else `liar:<speaker>`, else `row:<text_hash>`.
  - `splits.assign_split(groups: pd.Series, seed: int = 13) -> pd.Series` — deterministic hash of
    `group + seed` into train 70 / val 10 / calib 10 / test 10.
  - `build.build_corpus(raw_dir, out_dir, seed=13) -> dict` — writes `corpus.parquet`,
    `splits/{train,val,calib,test}.parquet`, `splits_manifest.json` (row counts, label balance per split and
    dataset, file sha256, seed, dedup counts). CLI: `fnd data build`.

- [ ] Tests: `clean_frame` buckets a 10-word row as `headline`; duplicates differing only in case and
      punctuation collapse to one; a duplicate pair with labels 0 and 1 disappears; `assign_split` is
      deterministic and no group appears in two splits (hypothesis over random group lists); the manifest
      lists all four splits with both labels present.
- [ ] Implement; run tests; `uv run fnd data build`; confirm the manifest shows 0 shared groups and 0 shared
      hashes across splits.
- [ ] Commit.

### Task 5: Baseline training, calibration, thresholds, report

**Files:**
- Create: `ml/src/fnd_ml/common/tokens.py`, `ml/src/fnd_ml/common/artifacts.py`,
  `ml/src/fnd_ml/models/baseline.py`, `ml/src/fnd_ml/calibration/{__init__,metrics,isotonic,thresholds}.py`,
  `ml/src/fnd_ml/eval/{__init__,metrics,report}.py`, `ml/configs/baseline.yaml`
- Test: `tests/unit/ml/test_baseline.py`, `tests/unit/ml/test_calibration.py`, `tests/fixtures/synthetic.py`

**Interfaces:**
- Produces:
  - `tokens.word_tokens(text: str) -> list[tuple[str, int, int]]` — lowercase tokens matching
    `(?u)\b\w\w+\b` with offsets. `tokens.analyze(text: str) -> list[str]` — unigrams and bigrams built from
    `word_tokens`, capped at the first 1000 tokens. Used as the vectorizer's `analyzer`, so training and
    attribution share one tokenizer.
  - `BaselineModel.fit(texts, y)`, `.decision_function(texts) -> np.ndarray`,
    `.predict_raw(texts) -> np.ndarray`, `.calibrate(texts, y)`, `.predict_proba(texts) -> np.ndarray`
    (calibrated `p_unreliable`), `.save(dir: Path, metadata: dict)`, `BaselineModel.load(dir: Path)`.
    Config: `min_df=3`, `max_features=300_000`, `sublinear_tf=True`, LR `C=4.0`,
    `class_weight="balanced"`, `max_iter=1000`.
  - `metrics.expected_calibration_error(y, p, n_bins=15) -> float`, `metrics.brier(y, p) -> float`.
  - `thresholds.pick_thresholds(y, p, precision_target=0.90, npv_target=0.90) -> tuple[float, float]` —
    smallest `t_high` whose band precision meets the target and largest `t_low` whose band NPV meets it;
    falls back to `(0.35, 0.65)` and flags `met_targets=False` when no threshold qualifies.
  - `eval.metrics.classification_report(y, p) -> dict` with `macro_f1`, `roc_auc`, `pr_auc`, `brier`, `ece`.
  - `artifacts.read_manifest(dir) / write_manifest(dir, data) / register(dir, kind, version, path)`.
  - CLI: `fnd train baseline` → `ml/artifacts/baseline/<version>/{model.joblib, calibration.joblib,
    thresholds.json, metadata.json}`, updates `ml/artifacts/manifest.json`, writes
    `ml/reports/baseline_v1.{json,md}` with metrics per split and per dataset, before and after calibration.
  - `tests/fixtures/synthetic.make_corpus(n: int = 200, seed: int = 0) -> pd.DataFrame` — templated
    sensational vs neutral texts for fast tests.

- [ ] Tests: ECE of perfectly calibrated toy data is `< 0.02` and of `p = 0.9` on all-negative labels is
      `> 0.8`; `pick_thresholds` on separable data returns `t_low < t_high`; a baseline trained on
      `make_corpus()` reaches macro-F1 `> 0.95` on a held-out synthetic split; `save` then `load` gives
      identical `predict_proba`; calibrated probabilities are within `[0, 1]`.
- [ ] Implement; run tests; `uv run fnd train baseline`.
- [ ] Gate: macro-F1 ≥ 0.80 on the grouped test split. Inspect the top 50 features per class and add any
      source identifier to `find_artifact_spans`, then rebuild and retrain.
- [ ] Commit code, config and the report.

### Task 6: Baseline inference and lexical attribution

**Files:**
- Create: `ml/src/fnd_ml/inference/{__init__,baseline_classifier}.py`,
  `ml/src/fnd_ml/explain/{__init__,lexical}.py`
- Test: `tests/unit/ml/test_lexical.py`

**Interfaces:**
- Consumes: `BaselineModel`, `tokens.word_tokens`, `model_input`.
- Produces:
  - `ClassifierOutput(p_unreliable: float, p_raw: float, version: str, truncated: bool)`.
  - `BaselineClassifier.load(artifact_dir: Path)`, `.predict(title: str | None, text: str) -> ClassifierOutput`,
    `.thresholds -> tuple[float, float]`.
  - `lexical.attribute(clf, title, text, top_k=40) -> list[TokenWeight]` where
    `TokenWeight(start: int, end: int, token: str, weight: float)` with offsets into `text`. A bigram's
    contribution is split equally between its two tokens. Title-only tokens are omitted.
  - `lexical.total_logit(clf, title, text) -> float` (for the exactness test).

- [ ] Tests: the sum of **all** token weights (title included, `top_k=None`) plus the intercept equals the
      model's decision function within `1e-6`; every returned span satisfies
      `text[start:end].lower() == token`; text with a dateline artifact still yields correct offsets into the
      original text; an out-of-vocabulary text returns an empty list.
- [ ] Implement; run tests; commit.

### Task 7: Backend core

**Files:**
- Create: `backend/src/fnd_api/{config,logging,main}.py`, `backend/src/fnd_api/api/{__init__,deps,errors}.py`,
  `backend/src/fnd_api/api/v1/{__init__,health,meta}.py`, `backend/src/fnd_api/models_registry.py`
- Test: `tests/integration/conftest.py`, `tests/integration/test_health_meta.py`

**Interfaces:**
- Produces:
  - `Settings` (env prefix `FND_`): `var_dir`, `database_url`, `cache_dir`, `redis_url`, `artifacts_dir`,
    `rate_limit_analyze="30/minute"`, `rate_limit_read="120/minute"`, `max_text_chars=100_000`,
    `max_fetch_bytes=5_000_000`, `fetch_timeout_s=8.0`, `allow_private_urls=False`,
    `store_input_text=True`, `log_json=True`, `log_level="INFO"`.
  - `create_app(settings: Settings | None = None) -> FastAPI` with a lifespan that loads models, opens the
    DB, cache and a shared `httpx.AsyncClient`, and stores them on `app.state`.
  - `ApiError(code: str, status: int, detail: str)` plus handlers producing problem+json with `request_id`.
  - `ModelRegistry.load(artifacts_dir) -> ModelRegistry` with `.baseline`, `.versions -> dict[str, str]`.
  - `GET /healthz` → `{"status": "ok"}`; `GET /readyz` → 200 when models and DB are ready, else 503;
    `GET /api/v1/meta` → model versions, enabled providers, limits.
  - Integration fixtures: `settings` (tmp dirs, tiny synthetic baseline artifact), `client`
    (`httpx.AsyncClient` over `ASGITransport`).

- [ ] Tests: `/healthz` 200; `/readyz` 200 with an artifact and 503 without; `/api/v1/meta` lists the
      baseline version and `"external": false`; an unknown route returns problem+json with a `request_id`
      that also appears in the `X-Request-ID` response header; a validation error returns `invalid_input`.
- [ ] Implement; run tests; commit.

### Task 8: Database and cache

**Files:**
- Create: `backend/src/fnd_api/db/{__init__,models,session,repositories}.py`, `backend/alembic.ini`,
  `backend/alembic/{env.py,script.py.mako}`, `backend/alembic/versions/0001_initial.py`,
  `backend/src/fnd_api/cache/{__init__,base,disk_cache,redis_cache,keys}.py`
- Test: `tests/unit/backend/test_repositories.py`, `tests/unit/backend/test_cache.py`

**Interfaces:**
- Produces:
  - ORM models `Scan`, `Feedback`, `SourceReputation` matching the design's DDL exactly.
  - `session.init_db(settings) -> AsyncEngine` (runs `alembic upgrade head`, enables WAL and foreign keys);
    `session_factory(engine)`.
  - `ScanRepository.add(response: AnalysisResponse, *, input_hash, depth, external_used, status) -> None`,
    `.get(scan_id) -> AnalysisResponse | None`,
    `.list(cursor, limit, verdict, domain, q) -> tuple[list[ScanSummary], str | None]`,
    `.delete(scan_id) -> bool`; `FeedbackRepository.add(scan_id, data) -> FeedbackOut`,
    `.count(scan_id) -> int`.
  - `Cache` protocol: `async get(key) -> bytes | None`, `async set(key, value: bytes, ttl_s: int)`,
    `async delete(key)`, `async close()`; `make_cache(settings) -> Cache`.
  - `keys.scan_key(canonical_input: str, options: AnalyzeOptions, model_versions: dict[str, str]) -> str`.

- [ ] Tests: migrations create the three tables and the four `scans` indexes; add → get round-trips a
      response; list paginates newest-first with a working cursor; deleting a scan cascades to feedback;
      disk cache honours TTL (`ttl_s=1`, read after expiry returns `None`); the same contract suite runs
      against `fakeredis`; `scan_key` changes when `depth` or a model version changes and ignores
      `use_cache`/`store`.
- [ ] Implement; run tests; commit.

### Task 9: Ingest

**Files:**
- Create: `backend/src/fnd_api/ingest/{__init__,url_guard,fetcher,extractor,metadata,paywall,language}.py`,
  `data/fixtures/html/{news_jsonld.html,blog_plain.html,paywalled.html,js_shell.html}`
- Test: `tests/unit/backend/test_url_guard.py`, `test_fetcher.py`, `test_extractor.py`, `test_language.py`

**Interfaces:**
- Produces:
  - `url_guard.is_public_ip(ip: str) -> bool`; `async url_guard.validate_url(url, *, resolver,
    allow_private=False) -> None` raising `FetchBlocked` for non-http(s) schemes, credentials in the URL,
    and hosts resolving to loopback, private, link-local, multicast, reserved or unspecified addresses.
  - `async fetcher.fetch_html(url, client, settings, resolver=...) -> FetchedPage` — follows at most 5
    redirects manually and validates every hop; verifies the connected peer address when the transport
    exposes it; streams with the byte cap; accepts `text/html`, `application/xhtml+xml`, `text/plain`.
    Raises `FetchBlocked`, `FetchFailed` (status, timeout, content type) or `PayloadTooLarge`.
    `FetchedPage(url, final_url, status, html, fetched_at)`.
  - `extractor.extract_article(html: str, url: str) -> ExtractedArticle` with `title`, `text`, `authors`,
    `site_name`, `published_at`, `modified_at`, `date_source`, `canonical_url`, `paywall_hint: bool`.
  - `paywall.looks_paywalled(article: ExtractedArticle, html: str) -> bool` — JSON-LD
    `isAccessibleForFree: false`, or fewer than 120 words together with a paywall phrase.
  - `language.detect_language(text: str) -> str | None` — returns `None` (treated as English) for texts under
    5 words or low-confidence detections.

- [ ] Tests: `validate_url` blocks `http://127.0.0.1/`, `http://169.254.169.254/latest/meta-data`,
      `http://[::1]/`, `ftp://example.com`, `http://user:pw@example.com`, and a hostname the fake resolver
      maps to `10.0.0.5`; allows one mapped to `93.184.216.34`. `fetch_html` (respx): follows a public
      redirect; raises `FetchBlocked` when a public URL redirects to `http://127.0.0.1/`; raises
      `PayloadTooLarge` on an over-cap body; raises `FetchFailed` for `application/pdf` and for HTTP 500.
      `extract_article` on `news_jsonld.html` returns title, ≥ 200 words, an author and the JSON-LD date with
      `date_source == "jsonld"`; `paywalled.html` sets `paywall_hint`; `js_shell.html` yields fewer than 40
      words. `detect_language`: English paragraph → `"en"`, Spanish paragraph → `"es"`,
      `"Senate passes bill"` → `None`.
- [ ] Implement; run tests; commit.

### Task 10: Rule-based rhetoric signals

**Files:**
- Create: `backend/src/fnd_api/signals/{__init__,rules,spans}.py`,
  `data/lexicons/{clickbait.yaml,sensational.txt,loaded_language.txt,vague_attribution.txt,acronyms.txt}`
- Test: `tests/unit/backend/test_signals.py`

**Interfaces:**
- Produces:
  - `rules.RuleEngine.load(lexicon_dir: Path) -> RuleEngine`; `.find(text: str) -> list[RawSpan]` with
    `RawSpan(start, end, kind: SpanKind, label: str, explanation: str, score: float)`.
    Detectors: clickbait patterns, sensational terms, loaded language, vague attribution, runs of two or more
    ALL-CAPS words (acronym list excluded), repeated `!`/`?`.
  - `spans.finalize(raw: list[RawSpan], text: str) -> list[Span]` — drops out-of-bounds and empty spans,
    removes a span nested inside a same-kind span, sorts by `start`, assigns ids `s1…`, sets `origin="rule"`.

- [ ] Tests: `"SHOCKING: You won't believe what experts say!!!"` yields `sensational`, `clickbait`,
      `vague_attribution` and punctuation spans, each with `text[start:end]` equal to the matched string;
      `"NASA and the FBI said"` yields no ALL-CAPS span; hypothesis property over arbitrary unicode text
      (emoji, combining marks, CRLF included): every span is in bounds, non-empty and `start < end`;
      `finalize` output is sorted with unique ids.
- [ ] Implement; run tests; commit.

### Task 11: Schemas, pipeline, `/analyze`, rate limiting

**Files:**
- Create: `backend/src/fnd_api/schemas/{__init__,enums,request,response,events,feedback}.py`,
  `backend/src/fnd_api/pipeline/{__init__,context,stages,orchestrator}.py`,
  `backend/src/fnd_api/fusion/{__init__,scorer,confidence,summary}.py`,
  `backend/src/fnd_api/ratelimit.py`, `backend/src/fnd_api/api/v1/analyze.py`
- Test: `tests/unit/backend/test_schemas.py`, `test_fusion.py`, `tests/integration/test_analyze.py`

**Interfaces:**
- Consumes: Tasks 6–10.
- Produces:
  - Schemas exactly as in design section 3, with `satire_score: float | None = None`.
  - `PipelineContext(settings, models, cache, http, rules, scans, resolver)`.
  - `async orchestrator.run(req: AnalyzeRequest, ctx: PipelineContext) -> AsyncIterator[StageEvent |
    AnalysisResponse]` — stages `ingest`, `source`, `classifier`, `signals`, `explanations`,
    `final_verdict`; the last item is the `AnalysisResponse`.
    `async orchestrator.analyze(req, ctx) -> AnalysisResponse` consumes it.
  - `scorer.band(risk: float, t_low: float, t_high: float) -> Verdict`;
    `confidence.compute(risk: float, *, partial_text: bool, word_count: int) -> tuple[float,
    ConfidenceLevel]` — `max(risk, 1 - risk)`, capped at 0.6 for partial text or fewer than 40 words;
    levels: high ≥ 0.85, medium ≥ 0.65, else low.
  - `summary.build(response_parts) -> str` — deterministic template text.
  - Behaviour: text under 3 words → `content_too_short` (422); non-English → `unsupported_language` (422);
    fewer than 40 extracted words from a URL → verdict `unverified` with tag `partial_text`; cache hit →
    same body with `cached=True` and a fresh `latency_ms`; `store=False` skips the DB write.
  - `POST /api/v1/analyze` limited by `settings.rate_limit_analyze`; 429 as problem+json `rate_limited`.

- [ ] Tests (unit): `AnalyzeRequest` rejects both and neither of `url`/`text`, extra fields, and text over
      100,000 characters; `band` boundaries (`t_low` → misleading, `t_high` → high risk);
      `confidence.compute(0.95, partial_text=True, word_count=500)` is `(0.6, low)`.
- [ ] Tests (integration, tiny model, respx): sensational text → `high_risk`, every highlight and token
      attribution satisfies `text[start:end]`; the same test passes for text containing emoji and CRLF; a
      second identical call returns `cached: true`; a URL request against a mocked page returns title, domain
      and `extraction == "full"`; a mocked JS-shell page returns `unverified` + `partial_text`;
      `http://127.0.0.1/` → 422 `fetch_blocked`; Spanish text → 422 `unsupported_language`; with the limit
      set to `3/minute` the 4th call returns 429 `rate_limited`; two concurrent identical requests
      (`asyncio.gather`) both return 200.
- [ ] Implement; run tests; commit.

### Task 12: History and feedback endpoints

**Files:**
- Create: `backend/src/fnd_api/api/v1/{scans,feedback}.py`
- Test: `tests/integration/test_history_feedback.py`

**Interfaces:**
- Consumes: `ScanRepository`, `FeedbackRepository`, schemas.
- Produces: `GET /api/v1/scans` (`cursor`, `limit` 1–100 default 20, `verdict`, `domain`, `q`),
  `GET /api/v1/scans/{scan_id}`, `DELETE /api/v1/scans/{scan_id}` (204),
  `POST /api/v1/scans/{scan_id}/feedback` (201, `FeedbackOut`). Unknown ids → 404 problem+json `not_found`.

- [ ] Tests: an analysed scan appears first in the list and round-trips through `GET /scans/{id}`;
      pagination with `limit=1` walks two scans via `next_cursor`; `verdict` and `domain` filters work;
      feedback returns 201 and increments `feedback_count`; feedback on an unknown scan → 404;
      `corrected_verdict` outside the enum → 422; delete → 204, then 404.
- [ ] Implement; run tests; commit.

### Task 13: M1 verification

- [ ] `uv run poe check` passes; coverage report shows ≥ 85% for `fnd_api` and `fnd_ml.common`.
- [ ] `ml/reports/baseline_v1.md` exists with macro-F1 ≥ 0.80 on the grouped test; the split manifest shows
      zero shared groups and hashes.
- [ ] Start `uv run poe serve` in the background against the real baseline artifact and verify by HTTP:
  - pasted article → verdict plus spans whose offsets match the text;
  - a real public article URL → ≥ 200 extracted words, with title and date;
  - the same request again → `cached: true`, under 100 ms;
  - 31 analyses inside a minute → the 31st is 429;
  - `http://127.0.0.1/` and `http://169.254.169.254/` → `fetch_blocked`;
  - the scan is in `GET /scans`, and feedback on it persists.
- [ ] Stop the server; write measured numbers into `docs/plans/m1-results.md`; merge `m1-mvp` into `main`,
      tag `m1`.
