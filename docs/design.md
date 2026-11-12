# Fake News Detection System — Design & Implementation Plan

## Context

**What and why.** `D:\sources\fake-news-detection` is an empty repo (only `.git`). The goal is an end-to-end
system that takes a news article, headline, or URL and returns a three-level verdict with calibrated
confidence, shows *why* (highlighted phrasing, flagged claims, token/sentence attribution), checks the source
and metadata, and grounds the verdict in retrieved evidence instead of writing style alone.

**Constraints you locked in.**

| Question | Your answer | Consequence |
|---|---|---|
| External APIs vs offline | Hybrid, pluggable | Every LLM/search dependency sits behind an interface with a local fallback; the system works with zero keys |
| Usage / deployment | Local, single user | One process, SQLite, disk cache, no accounts; Docker files written but not runnable here |
| Languages | English only | DeBERTa-v3 (English); non-English input returns `unsupported_language` |
| Training compute | Local Quadro P4000 (8 GB) | Pinned PyTorch build, small batches, hours-long background runs with resume |

**What I measured on this machine.** Python 3.12.10, uv 0.13.1, Node 24.14 / npm 11.16, Quadro P4000 8 GB
(driver 582.53), 16 GB RAM, 12 logical CPUs, 898 GB free. Not installed: Docker, Redis, Ollama, pnpm, `make`.
`ANTHROPIC_API_KEY` is not set.

**Goals.** Honest generalization (measured on unseen publishers and unseen datasets), calibrated scores,
explanations that map to verbatim text, evidence with real URLs, graceful degradation, one-command local run.

**Non-goals (this build).** Accounts/auth, multilingual, image/video verification, social-propagation
features, headless-browser rendering, cloud deployment, Prometheus/Grafana.

**Anti-goals (would make it a failure even if it "works").**
- A classifier that scores ~99% on ISOT and collapses on new publishers.
- Presenting a writing-style score as a truth verdict.
- Any fabricated highlight or evidence item (every span must match the text; every evidence card must come
  from a provider with a real URL or a dated local snapshot).

---

## Approach and stack (my calls, with reasons)

**Three layers, not one model.** You asked me to compare them; the answer is that each covers a different failure.

| Layer | Strength | Weakness | Role in the system |
|---|---|---|---|
| TF-IDF + Logistic Regression (and XGBoost on stylometric features) | Milliseconds, tiny, exact per-token attributions | Learns topic and publisher shortcuts; weak out of distribution | Sanity baseline, runtime fallback, lexical explainer |
| Fine-tuned DeBERTa-v3-base | Best style/discourse modelling of the three | Still measures style, not truth; slow to train on 8 GB | Primary style-risk signal |
| Claim extraction + retrieval + verification (LLM, or local NLI) | Grounds in facts; can handle events newer than training data | Seconds of latency, API cost, prompt-injection surface | Evidence path; can override the style signal through guards |

| Area | Choice | Why this, not the alternative |
|---|---|---|
| Python tooling | uv workspace, Python 3.12, `poe` tasks | uv is installed; 3.12 has wheels for everything; `make` does not exist on Windows |
| Classifier | `microsoft/deberta-v3-base`, binary head, head+tail truncation to 512 tokens | Stronger than RoBERTa-base on classification at the same size; fallback is `deberta-v3-small` |
| Serving runtime | ONNX Runtime (fp32 + int8), no PyTorch in the default API path | Low CPU latency, small install; PyTorch only loads for opt-in deep attribution |
| Backend | FastAPI, Pydantic v2, SQLAlchemy 2 async + aiosqlite, Alembic, structlog, slowapi | As requested; SQLite fits single-user and needs no server |
| Cache | `Cache` protocol: Redis when `REDIS_URL` is set, else `diskcache` | Redis is not installed here, so disk is the tested default |
| Scraping | httpx + trafilatura + extruct (JSON-LD/OpenGraph) | trafilatura bundles readability/jusText fallbacks and date extraction |
| Evidence index | LanceDB (embedded, hybrid BM25 + vector), `bge-small-en-v1.5` via fastembed | Works like SQLite, no server, no Docker; pgvector/Qdrant would need a service |
| Local verifier | FEVER-trained NLI cross-encoder exported to ONNX (candidate `MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli`, confirmed in M3) | Offline stance detection without an LLM |
| LLM | Claude via the official `anthropic` async SDK, default `claude-opus-5-5`, model per task set by env var | Default follows Anthropic's current guidance; see cost note in section 6 |
| Web search | Claude's server-side web search tool restricted with `allowed_domains` | One key covers LLM and search; domain allow-list keeps evidence to trusted sources |
| Frontend | Vite + React + TypeScript SPA, Tailwind, TanStack Query, types generated from OpenAPI | No SSR needed; the static build is served by FastAPI, so it stays one process. Next.js would add a Node server for no benefit |
| Tests | pytest, respx, hypothesis, Vitest + Testing Library + MSW, Playwright | Offsets get property tests; outbound HTTP is always mocked in unit tests |

---

## 1. System Architecture

```
 User (browser)
      |
      v
+-----------------------------------------------------------------------------------------+
| FRONTEND  React + Vite SPA (static build served by the API)                             |
|   URL / text / headline input -> gauge, highlighted text, evidence cards, history       |
+-----------------------------------------------------------------------------------------+
      |  POST /api/v1/analyze (JSON)   or   POST /api/v1/analyze/stream (SSE stage events)
      v
+-----------------------------------------------------------------------------------------+
| BACKEND  FastAPI                                                                        |
|  request-id -> structured log -> rate limit -> size limit -> validation (Pydantic)      |
|                                                                                         |
|  [0] CACHE LOOKUP  key = sha256(canonical URL | normalized text + options + model vers) |
|        | miss                                            hit -> stored response, <100 ms|
|        v                                                                                |
|  [1] INGEST & PREPROCESS                                                                |
|      URL : SSRF guard -> async fetch (timeout, size cap) -> trafilatura main text       |
|            -> JSON-LD/OpenGraph metadata -> paywall check -> Wayback / lede fallback    |
|      text / headline : used as given                                                    |
|      then: normalize -> language check (en) -> sentence split with char offsets         |
|        |                                                                                |
|        +--------------------------+-----------------------------+                       |
|        v                          v                             v                       |
|  [2A] STYLE PATH (local)    [2B] SOURCE & METADATA       [2C] CLAIM & EVIDENCE PATH     |
|   DeBERTa-v3 (ONNX)          domain reputation table       claim extraction             |
|     -> calibrated p          lookalike / TLD heuristics      LLM | heuristic fallback   |
|   TF-IDF + LR baseline       domain age (RDAP, cached)     retrieval (parallel)         |
|     (fallback, lexical)      date anomalies                  Google Fact Check API      |
|   rhetoric signals           satire / opinion flags          Claude web search          |
|     rules (+ LLM spans)                                        (trusted domains only)   |
|   satire score                                               local hybrid index         |
|                                                                (BM25 + embeddings)      |
|                                                            verification per claim       |
|                                                              LLM verifier | local NLI   |
|        |                          |                             |                       |
|        +--------------------------+-----------------------------+                       |
|                                   v                                                     |
|  [3] EXPLAINABILITY ENGINE                                                              |
|      lexical token weights (baseline, exact) + sentence occlusion (transformer)         |
|      + Integrated Gradients (deep mode) + rule/LLM spans -> one merged, offset-checked  |
|      span list; each claim linked to its evidence                                       |
|                                   v                                                     |
|  [4] FUSION & VERDICT                                                                   |
|      log-odds: classifier + source prior + evidence adjustments -> risk score           |
|      guards: refuted key claim, satire, too short, partial text -> band + confidence    |
|                                   v                                                     |
|  [5] PERSIST + CACHE SET -> AnalysisResponse (JSON) / final SSE event                   |
+-----------------------------------------------------------------------------------------+
      |                 |                    |                         |
      v                 v                    v                         v
  SQLite            Cache                LanceDB                 ml/artifacts/
  scans, feedback,  Redis if REDIS_URL,  local evidence index    ONNX models, tokenizers,
  source_reputation else diskcache       (offline provider)      calibration, thresholds

 OFFLINE TRAINING (ml/):
  raw datasets -> clean + scrub -> dedup -> grouped splits -> train (LR, XGB, DeBERTa)
     -> calibrate -> thresholds -> export ONNX (+ int8) -> eval report + quality gates
```

**Three depths, one pipeline.** The orchestrator is a single async generator of stage events; the JSON
endpoint collects them, the SSE endpoint streams them.

| `depth` | Stages | Target p95 on this machine |
|---|---|---|
| `quick` | ingest, source, classifier, rule signals, lexical attribution | 2 s |
| `standard` (default) | + sentence attribution, claims, evidence, verification | 8 s offline / 25 s hybrid; first verdict event within 2.5 s |
| `deep` | + Integrated Gradients token attribution, up to 10 claims | best effort |

**Explanation tiers.**

| Tier | Method | Runs on | Cost |
|---|---|---|---|
| Lexical tokens | Exact TF-IDF × coefficient contributions | Baseline model | Milliseconds, always on |
| Sentences | Leave-one-sentence-out occlusion, capped at 16 segments | Transformer (ONNX) | Up to 16 extra forward passes |
| Deep tokens | Integrated Gradients (Captum) | Transformer (PyTorch, GPU if present) | Seconds, opt-in |
| Rhetoric spans | Rule lexicons and patterns; LLM adds fallacies when a key is set | Rules / LLM | Milliseconds / one LLM call |

**Fusion.** `z = logit(p_classifier) + w_src·source_term + Σ evidence_adjustments`, `risk = sigmoid(z)`.
- `p_classifier` is temperature-scaled. `w_src` is fit by logistic regression on validation rows with a known
  domain. Evidence adjustments come from a small documented policy table (trust tier × stance × claim
  centrality), bounded so no single item can swing the score alone.
- Bands: `risk < t_low` → Reliable, `risk ≥ t_high` → High Risk, otherwise Potentially Misleading.
- Guards: a key claim refuted by a fact-checker means never Reliable; text too short with no evidence means
  Unverified; satire gets a `satire` tag instead of a "fake" label; partial text caps confidence.
- `confidence` is the measured accuracy of the assigned band at that risk level on held-out data (isotonic
  fit), then capped by degradations. Every term is returned in `score_breakdown`.

**Evidence rules.** The analysed article's own URL, domain and near-identical copies are excluded from
evidence. No evidence means `not_enough_evidence`, never `refuted`. "Trusted domains" is an editable policy
file (`data/sources/trusted_domains.yaml`), and each card shows its source so you can judge it.

---

## 2. Directory & File Structure

```
fake-news-detection/
├── README.md  pyproject.toml (uv workspace root; ruff/mypy/pytest/poe config)  uv.lock
├── .python-version (3.12)  .env.example  .gitignore  .pre-commit-config.yaml
├── docker-compose.yml  backend/Dockerfile          # written, not runnable here (no Docker)
├── .github/workflows/ci.yml
├── docs/        design.md (this plan)  architecture.md  model_card.md  data_statement.md  runbook.md
│   └── plans/   one detailed task plan per milestone, written when the milestone starts
├── data/
│   ├── registry/datasets.yaml      # per dataset: source URL, license, sha256, loader, label map
│   ├── lexicons/                   # clickbait, sensational, hedges, fallacy_patterns.yaml
│   ├── sources/                    # domain_reputation.csv + PROVENANCE.md, satire_domains.txt,
│   │                               # trusted_domains.yaml
│   ├── fixtures/                   # tracked: saved HTML pages, 200-row sample, golden_set.jsonl
│   ├── raw/  interim/  processed/  # gitignored; processed/ holds corpus + splits + manifest
│   └── kb/                         # gitignored; LanceDB evidence index
├── ml/
│   ├── pyproject.toml              # package fnd-ml; heavy deps in the "train" group
│   ├── configs/                    # baseline, xgb, deberta_small, deberta_base, eval_gates (yaml)
│   ├── src/fnd_ml/
│   │   ├── common/     text.py (normalize, scrub)  segment.py (sentences + offsets)
│   │   │               truncation.py  features.py (stylometric)  artifacts.py  labels.py
│   │   ├── data/       download.py  loaders/  clean.py  dedup.py  leak_probe.py  topics.py
│   │   │               splits.py  build.py
│   │   ├── models/     baseline.py  xgb.py  transformer.py  satire.py
│   │   ├── calibration/ temperature.py  isotonic.py  thresholds.py  metrics.py (ECE, Brier)
│   │   ├── export/     onnx_export.py  quantize.py  parity.py  benchmark.py
│   │   ├── inference/  onnx_classifier.py  baseline_classifier.py  nli.py  embedder.py
│   │   ├── explain/    lexical.py  occlusion.py  integrated_gradients.py  faithfulness.py
│   │   ├── eval/       metrics.py  protocols.py  adversarial.py  report.py  gates.py
│   │   └── cli.py      # fnd data|train|calibrate|export|eval|index
│   ├── artifacts/      # gitignored except manifest.json
│   │                   # <model>/<version>/{model.onnx, tokenizer/, calibration.json,
│   │                   #                    thresholds.json, metadata.json}
│   ├── runs/           # gitignored training runs and checkpoints
│   └── reports/        # eval_<version>.{json,md}, reliability plots
├── backend/
│   ├── pyproject.toml  alembic.ini  alembic/versions/
│   └── src/fnd_api/
│       ├── main.py (app factory, lifespan, static SPA mount)  config.py  logging.py  ratelimit.py
│       ├── api/v1/     analyze.py  scans.py  feedback.py  sources.py  meta.py  health.py
│       ├── api/        deps.py  errors.py
│       ├── schemas/    request.py  response.py  events.py  feedback.py
│       ├── db/         models.py  session.py  repositories.py
│       ├── cache/      base.py  redis_cache.py  disk_cache.py  keys.py
│       ├── ingest/     url_guard.py (SSRF)  fetcher.py  extractor.py  metadata.py  paywall.py
│       │               wayback.py  language.py
│       ├── source/     reputation.py  domain_heuristics.py  rdap.py  date_anomalies.py
│       ├── signals/    rules.py  fallacies.py  spans.py
│       ├── claims/     extractor_llm.py  extractor_heuristic.py  checkworthiness.py
│       ├── evidence/   base.py  google_factcheck.py  claude_websearch.py  local_index.py
│       │               ranker.py  trust.py
│       ├── verify/     llm_verifier.py  nli_verifier.py  aggregate.py
│       ├── llm/        client.py (parse, refusal handling, usage logging)  prompts/
│       ├── fusion/     scorer.py  guards.py  confidence.py
│       ├── pipeline/   orchestrator.py  stages.py  budget.py
│       └── security/   sanitize.py (prompt-injection hardening)  limits.py
├── frontend/
│   ├── package.json  vite.config.ts  tsconfig.json  index.html
│   └── src/
│       ├── main.tsx  App.tsx  routes/ (Analyze, History, ScanDetail)
│       ├── api/        schema.d.ts (generated)  client.ts  sse.ts  queries.ts
│       ├── components/ InputPanel  VerdictHeader  RiskGauge  ScoreBreakdown  HighlightedText
│       │               SpanTooltip  ClaimList  EvidenceCard  SourceCard  MetadataPanel
│       │               StageProgress  FeedbackBar  HistoryTable   (tests colocated: *.test.tsx)
│       └── lib/        spans.ts (overlap resolution)  format.ts
├── tests/
│   ├── unit/ml/  unit/backend/     # no network, no GPU
│   ├── integration/                # API with fake providers and tiny models
│   ├── ml_eval/                    # gates, adversarial, golden set (markers: slow, gpu, network)
│   ├── e2e/                        # Playwright
│   └── load/                       # latency and concurrency script
└── scripts/      smoke_gpu.py  dev.py (API + Vite)  build_reputation.py  build_kb.py
```

`fnd_ml.common` and `fnd_ml.inference` are imported by both training and the API, so preprocessing cannot
drift between them. The API depends on `fnd-ml` without the `train` group, so it installs without PyTorch.

---

## 3. Database & API Schema

### Endpoints

| Method | Path | Purpose |
|---|---|---|
| POST | `/api/v1/analyze` | Full analysis, JSON response |
| POST | `/api/v1/analyze/stream` | Same pipeline as SSE: `stage` events, then `result` or `error` |
| GET | `/api/v1/scans?cursor&limit&verdict&domain&q` | History, newest first, cursor-paginated |
| GET | `/api/v1/scans/{scan_id}` | Stored result |
| DELETE | `/api/v1/scans/{scan_id}` | Delete a scan and its feedback |
| POST | `/api/v1/scans/{scan_id}/feedback` | Add feedback |
| GET | `/api/v1/sources/{domain}` | Domain reputation lookup |
| GET | `/api/v1/meta` | Model versions, enabled providers, limits |
| GET | `/healthz`, `/readyz` | Liveness, readiness (models loaded, DB reachable) |

Errors use `application/problem+json` with a stable `code`: `invalid_input`, `unsupported_language`,
`content_too_short`, `fetch_failed`, `fetch_blocked`, `payload_too_large`, `rate_limited`, `internal`.
Default limits: 30 analyses per minute, 120 reads per minute, 100,000 characters of text, 5 MB per fetched page.

### Pydantic models (`backend/src/fnd_api/schemas/`)

```python
class InputType(StrEnum):       URL = "url"; TEXT = "text"; HEADLINE = "headline"
class Depth(StrEnum):           QUICK = "quick"; STANDARD = "standard"; DEEP = "deep"
class Verdict(StrEnum):         RELIABLE = "reliable"; MISLEADING = "potentially_misleading"
                                HIGH_RISK = "high_risk"; UNVERIFIED = "unverified"
class ConfidenceLevel(StrEnum): LOW = "low"; MEDIUM = "medium"; HIGH = "high"
class ClaimStatus(StrEnum):     SUPPORTED = "supported"; REFUTED = "refuted"; MIXED = "mixed"
                                NOT_ENOUGH_EVIDENCE = "not_enough_evidence"; NOT_CHECKED = "not_checked"
class Stance(StrEnum):          SUPPORTS = "supports"; REFUTES = "refutes"; NEUTRAL = "neutral"
class SpanKind(StrEnum):        CLICKBAIT = "clickbait"; SENSATIONAL = "sensational"
                                LOADED_LANGUAGE = "loaded_language"
                                VAGUE_ATTRIBUTION = "vague_attribution"; FALLACY = "fallacy"; CLAIM = "claim"
class StageStatus(StrEnum):     OK = "ok"; DEGRADED = "degraded"; SKIPPED = "skipped"
                                TIMED_OUT = "timed_out"; FAILED = "failed"

# ---- request ----
class AnalyzeOptions(BaseModel):
    depth: Depth = Depth.STANDARD
    max_claims: int = Field(5, ge=0, le=10)
    allow_external: bool = True      # False forces offline providers even when keys exist
    use_cache: bool = True
    store: bool = True               # persist to history

class AnalyzeRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)
    url: HttpUrl | None = None
    text: str | None = Field(None, min_length=1, max_length=100_000)
    title: str | None = Field(None, max_length=500)
    options: AnalyzeOptions = AnalyzeOptions()

    @model_validator(mode="after")
    def exactly_one_input(self) -> Self:
        if (self.url is None) == (self.text is None):
            raise ValueError("provide exactly one of 'url' or 'text'")
        return self
    # input_type is inferred: url -> URL; text of <= 40 words -> HEADLINE; otherwise TEXT

# ---- response ----
class Flag(BaseModel):
    code: str                        # e.g. "future_publish_date", "lookalike_domain"
    severity: Literal["info", "warn", "alert"]
    message: str

class Span(BaseModel):
    id: str
    start: int = Field(ge=0)         # char offsets into article.text, end exclusive
    end: int = Field(gt=0)
    kind: SpanKind
    label: str                       # e.g. "False dilemma", "Curiosity-gap headline"
    explanation: str
    score: float = Field(ge=0, le=1)
    origin: Literal["rule", "llm", "model"]
    claim_id: str | None = None

class TokenAttribution(BaseModel):
    start: int; end: int; token: str
    weight: float                    # signed; positive pushes toward high risk

class SentenceAttribution(BaseModel):
    index: int; start: int; end: int
    weight: float                    # change in log-odds when the sentence is removed

class Attribution(BaseModel):
    method: Literal["lexical_linear", "sentence_occlusion", "integrated_gradients"]
    model: str
    sentences: list[SentenceAttribution] = []
    tokens: list[TokenAttribution] = []

class ClassifierResult(BaseModel):
    model: str; version: str
    p_unreliable: float = Field(ge=0, le=1)        # calibrated
    p_unreliable_raw: float = Field(ge=0, le=1)
    calibration: Literal["temperature", "isotonic", "none"]
    satire_score: float = Field(ge=0, le=1)
    truncated: bool

class SourceInfo(BaseModel):
    domain: str | None
    known: bool
    credibility_score: float | None = Field(None, ge=0, le=1)
    category: Literal["reliable", "mixed", "unreliable", "satire", "unknown"]
    domain_age_days: int | None
    flags: list[Flag]
    provenance: list[str]            # which reputation lists the rating came from

class MetadataInfo(BaseModel):
    title: str | None; authors: list[str]; site_name: str | None
    published_at: datetime | None; modified_at: datetime | None
    date_source: Literal["jsonld", "meta", "url", "text", "none"]
    anomalies: list[Flag]

class Evidence(BaseModel):
    id: str
    provider: Literal["google_factcheck", "web_search", "local_index"]
    title: str; url: str | None; publisher: str | None
    published_at: datetime | None
    snippet: str
    rating: str | None               # fact-checker's own rating text, e.g. "False"
    stance: Stance
    stance_score: float = Field(ge=0, le=1)
    trust_tier: Literal["fact_checker", "high", "reference", "unknown"]
    snapshot_date: date | None       # set for local_index items

class Claim(BaseModel):
    id: str; text: str
    start: int | None; end: int | None             # location in article.text when found verbatim
    checkworthiness: float = Field(ge=0, le=1)
    status: ClaimStatus
    confidence: float = Field(ge=0, le=1)
    rationale: str
    method: Literal["llm", "nli", "none"]
    evidence: list[Evidence]

class ScoreComponent(BaseModel):
    name: Literal["classifier", "source", "evidence", "guard"]
    log_odds: float
    detail: str

class StageReport(BaseModel):
    name: str; status: StageStatus; duration_ms: int
    provider: str | None = None; detail: str | None = None

class ArticleInfo(BaseModel):
    input_type: InputType
    url: str | None; canonical_url: str | None
    text: str                        # the analysed text; all offsets refer to this string
    word_count: int; language: str
    extraction: Literal["full", "partial", "headline_only", "user_text"]

class AnalysisResponse(BaseModel):
    schema_version: Literal["1.0"] = "1.0"
    scan_id: str; created_at: datetime
    verdict: Verdict
    risk_score: float = Field(ge=0, le=1)
    confidence: float = Field(ge=0, le=1)
    confidence_level: ConfidenceLevel
    summary: str                     # deterministic, template-built explanation
    tags: list[Literal["satire", "opinion", "partial_text", "no_evidence_found",
                       "evidence_conflict", "old_article", "external_disabled"]]
    article: ArticleInfo
    classifier: ClassifierResult
    source: SourceInfo
    metadata: MetadataInfo
    claims: list[Claim]
    highlights: list[Span]
    attributions: list[Attribution]
    score_breakdown: list[ScoreComponent]
    stages: list[StageReport]
    models: dict[str, str]
    cached: bool; latency_ms: int

# ---- feedback and history ----
class FeedbackCreate(BaseModel):
    rating: Literal["agree", "disagree"]
    corrected_verdict: Verdict | None = None
    target_type: Literal["verdict", "claim", "highlight", "source"] = "verdict"
    target_id: str | None = None     # claim or span id inside the stored result
    comment: str | None = Field(None, max_length=2000)

class FeedbackOut(FeedbackCreate):
    id: str; scan_id: str; created_at: datetime

class ScanSummary(BaseModel):
    scan_id: str; created_at: datetime; input_type: InputType
    title: str | None; domain: str | None
    verdict: Verdict; risk_score: float; confidence: float; feedback_count: int

class ScanPage(BaseModel):
    items: list[ScanSummary]; next_cursor: str | None

# ---- SSE ----
class StageEvent(BaseModel):         # event: stage
    stage: Literal["ingest", "source", "classifier", "signals", "preliminary_verdict",
                   "claims", "evidence", "explanations", "final_verdict"]
    status: StageStatus
    data: dict[str, Any]             # the part of AnalysisResponse that this stage produced
# event: result -> AnalysisResponse        event: error -> problem+json body
```

### Database (SQLite, WAL mode, Alembic-managed)

```sql
CREATE TABLE scans (
  id               TEXT PRIMARY KEY,                 -- ULID
  created_at       TEXT NOT NULL,                    -- ISO-8601 UTC
  input_type       TEXT NOT NULL CHECK (input_type IN ('url','text','headline')),
  input_hash       TEXT NOT NULL,                    -- sha256(canonical URL or normalized text)
  url              TEXT,
  canonical_url    TEXT,
  domain           TEXT,                             -- registrable domain
  title            TEXT,
  input_text       TEXT,                             -- NULL when STORE_INPUT_TEXT=false
  word_count       INTEGER NOT NULL,
  verdict          TEXT NOT NULL CHECK (verdict IN
                     ('reliable','potentially_misleading','high_risk','unverified')),
  risk_score       REAL NOT NULL,
  confidence       REAL NOT NULL,
  classifier_prob  REAL,
  depth            TEXT NOT NULL,
  external_used    INTEGER NOT NULL DEFAULT 0,
  status           TEXT NOT NULL CHECK (status IN ('complete','partial','failed')),
  error_code       TEXT,
  latency_ms       INTEGER NOT NULL,
  llm_input_tokens  INTEGER,
  llm_output_tokens INTEGER,
  model_versions   TEXT NOT NULL,                    -- JSON
  schema_version   TEXT NOT NULL,
  result_json      TEXT NOT NULL                     -- full AnalysisResponse snapshot
);
CREATE INDEX ix_scans_created ON scans (created_at DESC);
CREATE INDEX ix_scans_hash    ON scans (input_hash);
CREATE INDEX ix_scans_domain  ON scans (domain);
CREATE INDEX ix_scans_verdict ON scans (verdict);

CREATE TABLE feedback (
  id                TEXT PRIMARY KEY,
  scan_id           TEXT NOT NULL REFERENCES scans(id) ON DELETE CASCADE,
  created_at        TEXT NOT NULL,
  rating            TEXT NOT NULL CHECK (rating IN ('agree','disagree')),
  corrected_verdict TEXT,
  target_type       TEXT NOT NULL DEFAULT 'verdict'
                      CHECK (target_type IN ('verdict','claim','highlight','source')),
  target_id         TEXT,
  comment           TEXT
);
CREATE INDEX ix_feedback_scan ON feedback (scan_id);

CREATE TABLE source_reputation (
  domain            TEXT PRIMARY KEY,                -- registrable domain, lowercase
  credibility_score REAL,                            -- 0..1, NULL if only categorical
  category          TEXT NOT NULL CHECK (category IN
                      ('reliable','mixed','unreliable','satire','unknown')),
  is_satire         INTEGER NOT NULL DEFAULT 0,
  provenance        TEXT NOT NULL,                   -- JSON: [{list, rating, as_of}]
  updated_at        TEXT NOT NULL
);
```

Claims and evidence live inside `result_json` with stable ids, and feedback points at them through
`target_type` + `target_id`, so there is one source of truth per scan. The cache is a separate key-value store.
Cache TTLs: scan results 24 h, evidence queries 12 h, fetched pages 1 h, domain age 30 days.

---

## 4. ML Training & Evaluation Strategy

### Data

| Role | Dataset (expected source, verified at download) | Notes |
|---|---|---|
| Train / in-distribution | WELFake (~72k articles, Zenodo) | Merges several corpora; overlaps ISOT, so dedup matters |
| Train / in-distribution | ISOT (~45k articles) | Real class is Reuters only; scrubbed and capped so it cannot dominate |
| Train / in-distribution | FakeNewsNet headlines (~23k, GitHub CSVs with URLs) | Gives headline-length inputs and publisher domains |
| Train / in-distribution | LIAR (12.8k PolitiFact statements) | true/mostly-true vs false/pants-fire; middle labels held out as a "mixed" slice |
| Satire scorer | Satire-vs-news headline set | Separate small TF-IDF model, not mixed into the main label |
| External test only | Constraint COVID-19 (2020 posts), BuzzFeed-Webis 2016 (9 publishers), satire slice | Never used for training, tuning or thresholds |
| Adversarial test only | Generated sets (see evaluation) | Built once, cached to disk for reproducibility |
| Evidence index | LIAR-PLUS justifications, Simple English Wikipedia snapshot | Offline provider; every item carries its snapshot date |

Each source is recorded in `data/registry/datasets.yaml` with URL, license and sha256. A dataset that cannot
be fetched non-interactively is dropped and logged; the minimum viable core is WELFake + LIAR + FakeNewsNet
headlines.

### Pipeline, step by step

1. **Ingest.** One loader per dataset maps to a common schema: `id, dataset, title, text, label, publisher,
   url, date, length_bucket`.
2. **Clean.** Unicode and whitespace normalization, drop empty and non-English rows, strip HTML remnants.
3. **Scrub label-leaking artifacts.** Remove datelines (`WASHINGTON (Reuters) -`), publisher names, bylines,
   "Featured image via…", Twitter handles and URLs.
4. **Leak probes (repeat step 3 until clean).**
   - Label probe: inspect the top 50 TF-IDF features of a quick LR; any source identifier goes on the scrub list.
   - Origin probe: train a classifier to predict dataset-of-origin from scrubbed text and report its accuracy,
     so the remaining shortcut is measured rather than assumed away.
5. **Deduplicate across all datasets.** Exact hash on normalized text, then MinHash LSH (5-word shingles,
   Jaccard ≥ 0.8). One row kept per cluster; clusters with conflicting labels are dropped.
6. **Group.** Assign `publisher_group` (registrable domain, or dataset+source where no URL exists) and
   `topic_group` (KMeans over `bge-small` embeddings, k≈50).
7. **Split once, with a fixed seed.** Train / val / calib / test, stratified by label and dataset, grouped so
   that no publisher group and no near-duplicate cluster crosses a split. Written to
   `data/processed/splits_manifest.json` with hashes.
   - val: model selection and early stopping.
   - calib: temperature and thresholds only.
   - test: touched once per model version.
8. **Train the baseline.** TF-IDF (word 1–2-grams + char 3–5-grams, fit on train only) + Logistic Regression;
   XGBoost on stylometric features + TF-IDF-SVD as the second baseline.
9. **Train the transformer.**
   - Pilot: `deberta-v3-small`, 256 tokens, to prove the pipeline end to end.
   - Final: `deberta-v3-base`, `title [SEP] body`, head+tail truncation to 512 tokens, lr 2e-5, effective batch
     16–32 through gradient accumulation, 3 epochs, 10% warmup, early stopping on val macro-F1.
   - fp32 vs mixed precision is benchmarked over 50 steps first, because this GPU generation gains memory but
     little speed from mixed precision.
   - Checkpoints every N steps with `--resume`, run in the background.
10. **Class imbalance.** Class-weighted loss; sampling balanced per dataset × label so "which dataset" cannot
    stand in for the label; thresholds tuned on calib instead of assuming 0.5; PR-AUC and macro-F1 reported.
11. **Calibrate.** Temperature scaling for the transformer, isotonic for the baseline, both on calib. Report
    ECE (15 bins), Brier and reliability diagrams before and after, in distribution and on external sets.
12. **Thresholds.** Choose `t_low` and `t_high` on calib so the High Risk band has precision ≥ 0.90 and the
    Reliable band has negative predictive value ≥ 0.90; everything between is Potentially Misleading.
13. **Export.** ONNX fp32, then dynamic int8 quantization; parity check against PyTorch; CPU latency benchmark
    at 128/256/512 tokens. Artifact folder holds model, tokenizer, `calibration.json`, `thresholds.json` and
    `metadata.json` (git commit, data manifest hash, metrics, sha256).

### Evaluation protocols

| Protocol | What it answers |
|---|---|
| In-distribution grouped test | Performance on unseen publishers from the same datasets |
| Leave-one-dataset-out (baseline on all folds; `deberta-v3-small` on capped folds) | Does it transfer to a different corpus? |
| External test sets | Generalization to data never seen in any form |
| Topic-held-out and temporal (train ≤ 2017, test later) | Topic memorization and drift |
| Adversarial | Typos/homoglyphs; true articles rewritten sensationally and false ones rewritten in wire-service tone; clickbait headline on a true body; neutral-tone false claims; satire |
| Robustness | Empty, very short, very long and non-English input must abstain or error cleanly |
| Explanation faithfulness | Removing the top-attributed sentences must move the score more than removing random ones |
| End-to-end verdict | Fused verdict on a labelled claim set, reported with and without the originating fact-checker's domain so the score is not inflated by finding the answer key |

### Quality gates (`ml/configs/eval_gates.yaml`, enforced by `fnd eval`)

| Gate | Threshold |
|---|---|
| Split hygiene | 0 publisher groups and 0 near-duplicate clusters shared between splits |
| Leak probe | No source identifier among the top 50 baseline features |
| Baseline sanity floor | Macro-F1 ≥ 0.80 on the grouped in-distribution test |
| Transformer vs baseline | ≥ +0.03 macro-F1 in distribution, and not worse on the mean of external sets |
| Calibration | ECE ≤ 0.05 and Brier lower than uncalibrated, in distribution |
| ONNX parity | fp32: max probability difference ≤ 0.001 on 500 samples; int8: macro-F1 drop ≤ 0.01, else ship fp32 |
| Latency | Classifier p95 ≤ 500 ms on CPU at 512 tokens |

These are sanity floors, not targets tuned on test data. If a gate fails I report the measured number and the
cause; I do not adjust anything using the test split.

---

## 5. Phased Execution Roadmap

Each milestone ends with its verification list passing and a short status report with measured numbers.

### M1 — Foundation and MVP (baseline, end to end through the API)

- GPU smoke test first: install the pinned PyTorch build, check the GPU architecture list, run one training
  step. This settles the local-training decision before anything depends on it.
- Scaffold the uv workspace, linting, typing, test config, `poe` tasks, pre-commit.
- Dataset registry, download, loaders, cleaning, scrubbing, exact dedup, publisher-grouped split.
- Baseline TF-IDF + LR with isotonic calibration, artifact export, lexical attribution.
- Backend: settings, structured logging with request ids, SQLite + Alembic, cache (disk and Redis
  implementations), rate limiting, `/analyze` for text and URL, scraping pipeline with SSRF guard, metadata
  extraction, rule-based rhetoric spans, history, feedback, health and meta endpoints.

**Verified when:**
- [ ] `uv run poe check` (ruff, mypy, pytest) passes; line coverage ≥ 85% on `fnd_api` and `fnd_ml.common`.
- [ ] Baseline report exists; macro-F1 ≥ 0.80 on the grouped test; split-hygiene gate passes.
- [ ] Live: a pasted article returns a verdict and spans where `text[start:end]` matches every span.
- [ ] Live: a real public article URL extracts ≥ 200 words with title and date.
- [ ] Live: the same request again returns `cached: true` in under 100 ms.
- [ ] Live: the 31st analysis in a minute returns 429; `http://127.0.0.1/` and `http://169.254.169.254/`
      return `fetch_blocked`.
- [ ] The scan appears in `GET /scans` and feedback on it persists.

### M2 — ML upgrade

- Near-duplicate dedup, leak probes, topic groups, temporal tags; freeze splits v2; retrain the baseline on
  v2 so the comparison is like for like; XGBoost comparison; satire scorer.
- DeBERTa-v3-small pilot, then the DeBERTa-v3-base final run (background, resumable).
- Temperature scaling, thresholds, ONNX export, quantization, parity and latency benchmarks.
- Backend serves the transformer through ONNX Runtime in a thread pool, with the baseline as fallback.

**Verified when:**
- [ ] All section 4 gates pass, or each exception is reported with its measured value.
- [ ] `ml/reports/eval_v1.md` compares LR, XGBoost and DeBERTa on macro-F1, ROC-AUC, PR-AUC, Brier and ECE
      across in-distribution, leave-one-dataset-out and external sets, with reliability diagrams.
- [ ] `/api/v1/meta` reports the transformer version; integration tests pass against the real artifact.
- [ ] Deleting the transformer artifact makes the API fall back to the baseline and say so in `stages`.

### M3 — Explainability and evidence

- Sentence occlusion, Integrated Gradients deep mode, faithfulness metric.
- Source reputation table built from open lists with provenance; lookalike/TLD heuristics; domain age; date
  anomalies (future date, modified before published, URL date mismatch, old article resurfacing).
- Claim extraction: LLM with structured output, heuristic fallback (entities, numbers, quotes, attribution).
- Evidence providers behind one interface: Google Fact Check API, Claude web search on trusted domains, local
  LanceDB hybrid index. Verification: LLM verifier, local NLI fallback.
- Fusion, guards, confidence; SSE endpoint; per-stage time budgets with graceful degradation.
- Prompt-injection hardening: article text is passed as delimited data, LLM output is schema-validated, quoted
  spans must be found verbatim, and the LLM never sets the final verdict directly.

**Verified when:**
- [ ] Property tests: every span and attribution offset is in bounds and matches the text; LLM quotes that
      cannot be located are dropped.
- [ ] Faithfulness: on 200 high-risk test articles, removing the top 3 attributed sentences lowers the score
      more than removing 3 random sentences in ≥ 70% of cases.
- [ ] Offline mode (no keys): the full pipeline returns claims, local evidence and NLI verdicts with the
      `external_disabled` tag, and tests assert zero outbound requests.
- [ ] Guards: a refuted key claim never yields Reliable; injection strings in the article do not change the
      verdict; a provider timeout yields `timed_out` in `stages` and a valid response inside the budget.
- [ ] Local verifier macro-F1 on a 300-claim labelled sample is reported.
- [ ] With keys: 10 curated claims that have published fact-checks; at least 8 retrieve relevant evidence
      and get the matching status. Token usage and cost per scan are logged and reported.

### M4 — UI and polish

```
+----------------------------------------------------------------------------------------+
| Fake News Detector              [Analyze]  [History]                           (theme) |
+----------------------------------------------------------------------------------------+
| ( URL | Text | Headline )  [ https://example.news/story...               ] [ Analyze ] |
| depth: (quick) (standard) (deep)            [x] allow external lookups                 |
+----------------------------------------------------------------------------------------+
|  ( gauge 0.82 )   HIGH RISK    confidence 0.88 (high)       tags: partial_text         |
|  why:  classifier +1.9  |  source +0.8  |  evidence +1.2  |  guards 0                  |
+--------------------------------------------+-------------------------------------------+
| ARTICLE (highlighted)                      | CLAIMS & EVIDENCE                         |
| "SHOCKING: what doctors [don't want you    | 1. "Vaccine X causes ..."      REFUTED    |
|  to know] ..."  [claim 1]                  |    [Snopes - False - 2024]  [WHO - refutes]|
| legend: clickbait / sensational / fallacy  | 2. "Senator Y said ..."        SUPPORTED  |
|  / claim / sentence heat                   |    [AP - supports]                        |
+--------------------------------------------+-------------------------------------------+
| SOURCE  example.news: unknown, 41 days old, lookalike of example.com                   |
| METADATA  published 2031-01-01 (future date)          [ Agree ] [ Disagree ] feedback  |
+----------------------------------------------------------------------------------------+
```

- Vite + React + TypeScript app with API types generated from the OpenAPI schema.
- Results render progressively from the SSE stream; history list and scan detail; feedback.
- Responsive from 375 px up, keyboard navigable, highlights carry text labels as well as colour, dark mode.
- Production build served by FastAPI at `/`, so `uv run poe serve` starts everything.

**Verified when:**
- [ ] `npm run lint`, `npm run typecheck` and `npm test` pass; `npm run build` output is served by the API.
- [ ] Playwright: paste text → verdict, at least one highlight and evidence cards appear; URL flow works;
      the scan shows in History; feedback is stored (checked through the API).
- [ ] No horizontal scroll at 375, 768 and 1440 px; axe-core reports no serious or critical violations.
- [ ] I run it in a browser and show you screenshots.

### M5 — Evaluation, hardening, deployment

- Full evaluation suite (adversarial, temporal, robustness, end-to-end verdict) and `eval_final.md`.
- Latency and concurrency test; feedback export into an evaluation set.
- Dockerfile and compose file (API + Redis), GitHub Actions workflow.
- Docs: README quickstart, model card, data statement with licenses and known biases, runbook, limitations.

**Verified when:**
- [ ] `fnd eval --suite full` passes the gates or lists exceptions with numbers.
- [ ] Measured on this machine: quick p95 ≤ 2 s, standard offline p95 ≤ 8 s, cached ≤ 100 ms, 20 concurrent
      quick requests with no errors.
- [ ] Clean environment: delete `.venv`, `uv sync`, `uv run poe check` passes, `uv run poe serve` works by
      following the README only.
- [ ] Docker and CI files are stated as written but not run here.

---

## 6. Edge Cases & Risks

| Area | Risk | Handling |
|---|---|---|
| Paywalled URLs | Only a teaser is extracted | Detect via JSON-LD `isAccessibleForFree`, paywall phrases and length. Fall back to lede and metadata, then a public Wayback snapshot, then ask for pasted text. Tag `partial_text` and cap confidence. No paywall circumvention |
| Blocked or JS-only pages | 403s, bot walls, empty HTML | Same degradation path; a headless browser is deliberately out of scope |
| Satire | Satire labelled as "fake" | Satire domain list, page self-labels, satire score; shown as a `satire` tag with its own wording. Satire copied to another site is caught only by the evidence path, and that limit is documented |
| Opinion pieces | Opinions treated as false claims | `opinion` tag from metadata or section URL; only checkable claims are verified |
| Temporal drift | Training data is mostly 2015–2020 | Evidence path carries new events; temporal split quantifies decay; local index items show snapshot dates; newer events yield `not_enough_evidence`; feedback export feeds retraining |
| Breaking news | Nothing published yet to check against | Abstain as Unverified; never infer falsity from missing evidence |
| Sober-toned misinformation | Style model cannot see it, including LLM-written text | Evidence path and guards; the adversarial rewrite set measures the gap; stated as a limitation |
| True facts, misleading framing | Claims check out but the article misleads | Middle band plus rhetoric spans; stated as a limitation |
| Dataset bias | US-political skew, publisher-as-label shortcut | Scrubbing, probes, grouped splits, external tests; data statement; UI wording is "risk signals", not accusations |
| Latency | LLM and search take seconds | Stage budgets, parallel claims, SSE with an early verdict, caching, int8 ONNX, claim cap |
| Prompt injection | Article text tries to steer the LLM | Delimited data, schema-validated output, verbatim quote check, verdict computed in code, injection tests |
| SSRF and hostile pages | URL input reaches internal addresses; huge responses | Scheme and IP allow-rules checked on every redirect hop, size and time caps, content-type allow-list |
| XSS | Scraped text rendered in the UI | Text nodes only, links restricted to http/https |
| GPU toolchain | Per PyTorch's notices, kernels for this GPU generation ship only in the CUDA 12.6 wheels, which end with PyTorch 2.14 | Pin torch to that line; smoke test in M1. Fallbacks in order: older pinned torch, `deberta-v3-small`, a cloud notebook |
| Training time | Estimated 8–14 GPU-hours across M2 | Measured in the first 100 steps and reported before the long run; checkpoints and resume; fallbacks are 384 tokens or a capped training set |
| LLM cost | Default `claude-opus-5-5` costs more than I implied earlier: my rough estimate is $0.05–$0.50 per standard scan depending on length, claims and searches | Models are env-configurable per task (Sonnet 5.5 or Haiku 5.5 cut token cost a lot); claims and searches are capped; actual usage is logged per scan and reported in M3 |
| LLM refusals and errors | A call is declined or times out | Check `stop_reason`, fall back to the local NLI verifier for that claim |
| Google Fact Check API | It is a `v1alpha1` API and I could not confirm its current status | Optional provider, verified with a live call once a key exists; nothing depends on it |
| Dataset and list licensing | Some lists may not be redistributable | Only redistributable data is committed; everything else is fetched by script with provenance |
| Unverifiable here | Docker, real Redis, CI | Redis path tested with a fake; Docker and CI files are delivered as unrun |
| Windows | No `make`, no uvloop, DataLoader worker quirks | `poe` tasks, default event loop, conservative worker counts |

Sources for the GPU toolchain row:
[CUDA 12.6 wheels end with PyTorch 2.14](https://dev-discuss.pytorch.org/t/notice-cuda-12-6-wheels-will-no-longer-be-published-from-pytorch-2-15-drops-maxwell-pascal-volta/3432),
[Pascal removed from CUDA 12.8/12.9 builds](https://dev-discuss.pytorch.org/t/cuda-toolkit-version-and-architecture-support-update-maxwell-and-pascal-architecture-support-removed-in-cuda-12-8-and-12-9-builds/3128).

---

## How I will execute

- **Git.** Work on milestone branches (`m1-mvp`, `m2-ml`, …) with small local commits per verified task,
  merged into `main` and tagged when the milestone's checklist passes. Nothing is pushed unless you ask.
- **First commit.** This plan goes into the repo as `docs/design.md`.
- **Per milestone.** I write a detailed task plan into `docs/plans/` at the start, build test-first, and
  finish with the verification list and measured numbers. I do not stop between milestones except for the
  items below.
- **LLM code.** Written against the official Anthropic Python SDK docs: structured outputs through
  `messages.parse`, the current web search tool with `allowed_domains`, refusal handling, prompt caching of
  the stable instructions.
- **Long GPU runs.** Background, checkpointed and resumable; I keep building code that does not depend on the
  model while they run.

## What I need from you, and when

| When | What | If you don't |
|---|---|---|
| M3 | An Anthropic API key in `.env`; optionally a Google Fact Check API key | Hybrid mode is built and tested against fakes and offline mode is fully verified; live checks stay unrun and I say so |
| M3 | Your choice of LLM models if you want lower cost than the default | Default stays `claude-opus-5-5` |
| M3 / M5 | A spending cap for live LLM evaluation (I propose 40 items) | Only the free offline evaluation runs |
| Only if the smoke test fails | Agreement before any system-level change such as a driver update | I use the fallbacks in section 6 |

## End-to-end verification of the finished system

1. `uv sync` then `uv run poe check` — lint, types, all unit and integration tests.
2. `uv run fnd eval --suite full` — writes `ml/reports/eval_final.md` and enforces the gates.
3. `uv run poe serve`, open `http://localhost:8000`:
   - paste a sensational fabricated article → High Risk with highlights and a score breakdown;
   - analyse a wire-service article by URL → Reliable or Unverified, with source card and metadata;
   - analyse a satire site URL → `satire` tag;
   - repeat a request → served from cache;
   - leave feedback → visible in History.
4. `npx playwright test` — the same flows automated.
5. With keys set, run the 10 curated claims and compare retrieved evidence against the known fact-checks.
