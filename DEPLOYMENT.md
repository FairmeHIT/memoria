# memoria academic-track deployment notes

This document is the deployment and method-disclosure package for the
academic-track submission. It explains how to run the repository, the Docker
command, the Add/Search wrapper, authorship and technical-report location, and
all method changes implemented in this repository.

## 1. Repository

Public repository:

```text
https://github.com/FairmeHIT/memoria
```

The repository is self-contained for platform deployment. It does not include
benchmark data, local SQLite databases, local virtual environments, real
credentials, or run artifacts.

Main files:

- `Dockerfile` — production image.
- `README.md` — project overview, configuration, and smoke checks.
- `SPEC.md` — technical report and full Add/Search contract rationale.
- `BENCHMARK_BASELINES.md` — local retrieval-only benchmark baselines.
- `.env.example` — safe example environment variables.
- `.env.none` — lexical-only formal deployment profile.
- `.env.gpt4o` — optional `gpt-4o-mini` reranking profile.
- `.env.bge` — local self-hosted BGE ablation profile, not recommended for
  formal academic-track submission unless the rules explicitly allow it.

## 2. Original authors and technical report

`memoria` is an original implementation by FairmeHIT for the Agent Memory
Leaderboard academic track. It is not a fork of an existing memory-system
implementation.

The technical report for this repository is `SPEC.md`. It describes:

- product positioning;
- external Add/Search contract constraints;
- storage schema;
- retrieval architecture;
- ranking requirements;
- security and retention policy;
- deployment contract;
- quality gates.

The repository also uses public benchmark/evaluation contracts and benchmark
data sources documented in `README.md`, `SPEC.md`, and
`BENCHMARK_BASELINES.md`. Public upstream sources include LoCoMo-Refined,
LongMemEval, CL-bench, PersonaMem-v2, BEAM, and SWEContextBench. Those
benchmark copies are not required to build or deploy this repository.

## 3. Runtime contract

The deployed service exposes three HTTP endpoints:

| Method | Path | Purpose |
| --- | --- | --- |
| `GET` | `/health` | Liveness/readiness. No authentication required. |
| `POST` | `/v1/add` | Synchronously ingest ordered conversation messages. |
| `POST` | `/v1/search` | Return ranked memory evidence for one `user_id`. |

Authentication is configured by `MEMORIA_AUTH_SCHEME`. Formal deployments
should use:

```text
MEMORIA_AUTH_SCHEME=bearer
MEMORIA_API_KEY=<memory-system-key>
```

The platform should call Add/Search with:

```http
Authorization: Bearer <memory-system-key>
```

## 4. Build

Build the image from the repository root:

```bash
docker build --tag memoria:0.4.0 .
```

The image uses Python 3.11, installs the package from source, exposes port
`8080`, and runs as a non-root user.

Container entrypoint:

```bash
uvicorn memoria.app:create_app --factory --host 0.0.0.0 --port 8080
```

## 5. Recommended formal run command

For the academic-track code-submission route, the safest reproducible
configuration is lexical-only retrieval. Add/Search call no external model,
which avoids remote-model dependency, cost, and model-rule ambiguity.

```bash
docker volume create memoria-data

docker run --rm --publish 8080:8080 \
  --env MEMORIA_API_KEY='<memory-system-key>' \
  --env MEMORIA_AUTH_SCHEME=bearer \
  --env MEMORIA_DATA_DIR=/var/lib/memoria \
  --env MEMORIA_RETENTION_DAYS=30 \
  --env MEMORIA_MAX_TOP_K=1000 \
  --env MEMORIA_EMBEDDING_BACKEND=none \
  --env MEMORIA_RERANKER_BACKEND=none \
  --mount source=memoria-data,target=/var/lib/memoria \
  memoria:0.4.0
```

Register these URLs with the evaluation platform:

```text
Add URL:    https://<host>/v1/add
Search URL: https://<host>/v1/search
Health:     https://<host>/health
```

If the platform accesses the container directly, use port `8080`.

## 6. Optional `gpt-4o-mini` reranking profile

If the academic-track review requires or allows a model during Search and the
model must be `gpt-4o-mini`, use this profile:

```bash
docker run --rm --publish 8080:8080 \
  --env MEMORIA_API_KEY='<memory-system-key>' \
  --env MEMORIA_AUTH_SCHEME=bearer \
  --env MEMORIA_DATA_DIR=/var/lib/memoria \
  --env MEMORIA_RETENTION_DAYS=30 \
  --env MEMORIA_MAX_TOP_K=1000 \
  --env MEMORIA_EMBEDDING_BACKEND=none \
  --env MEMORIA_RERANKER_BACKEND=gpt4o \
  --env MEMORIA_GPT4O_API_KEY='<openai-compatible-api-key>' \
  --env MEMORIA_GPT4O_BASE_URL='https://api.openai.com/v1' \
  --env MEMORIA_GPT4O_MODEL=gpt-4o-mini \
  --env MEMORIA_RERANKER_CANDIDATE_LIMIT=200 \
  --mount source=memoria-data,target=/var/lib/memoria \
  memoria:0.4.0
```

This mode does not call an embedding model. It retrieves candidates locally and
sends only already-`user_id`-scoped evidence candidates to a chat-completions
reranker. It should be used only after validating cost, latency, and
competition-rule fit.

## 7. Add wrapper

Add accepts one JSON object per synchronous request. A successful HTTP 200
means all supplied messages have been committed and are immediately searchable.

```python
import requests


def memoria_add(base_url: str, api_key: str, request: dict) -> dict:
    response = requests.post(
        base_url.rstrip("/") + "/v1/add",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        json=request,
        timeout=1200,
    )
    response.raise_for_status()
    return response.json()


memoria_add(
    "https://<host>",
    "<memory-system-key>",
    {
        "request_id": "example-add-001",
        "messages": [
            {
                "role": "user",
                "timestamp": 1704067200000,
                "content": "I prefer vegetarian meals when traveling.",
            }
        ],
        "user_id": "example-user-001",
        "session_id": "example-session-001",
    },
)
```

Required Add fields:

- `request_id`: idempotency key.
- `messages`: non-empty ordered list of source messages.
- `messages[].role`: `user` or `assistant`.
- `messages[].content`: non-empty original text.
- `messages[].timestamp`: optional Unix timestamp in milliseconds.
- `user_id`: required isolation namespace.
- `session_id`: source conversation grouping only; not used as a Search filter.

## 8. Search wrapper

Search returns ordered evidence records only. It never generates the final
answer.

```python
import requests


def memoria_search(
    base_url: str,
    api_key: str,
    *,
    query: str,
    user_id: str,
    top_k: int = 100,
    options: list[str] | None = None,
) -> list[dict]:
    payload: dict = {
        "query": query,
        "user_id": user_id,
        "top_k": top_k,
    }
    if options is not None:
        payload["options"] = options
    response = requests.post(
        base_url.rstrip("/") + "/v1/search",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        json=payload,
        timeout=1200,
    )
    response.raise_for_status()
    return response.json()["data"]


hits = memoria_search(
    "https://<host>",
    "<memory-system-key>",
    query="What meals do I prefer?",
    user_id="example-user-001",
    top_k=100,
)
```

Search response shape:

```json
{
  "data": [
    {
      "id": "mem_...",
      "content": "user: I prefer vegetarian meals when traveling.",
      "score": 1.5,
      "created_at": "2026-08-07T00:00:00+00:00"
    }
  ]
}
```

The answer-generation layer should pass `data[].content` as memory evidence to
the shared answer pipeline. It should not treat `score` or `created_at` as an
answer.

## 9. Smoke check

After deployment:

```bash
curl --fail https://<host>/health
```

Add:

```bash
curl --fail https://<host>/v1/add \
  --header 'Content-Type: application/json' \
  --header 'Authorization: Bearer <memory-system-key>' \
  --data '{
    "request_id": "smoke-001",
    "messages": [{"role": "user", "content": "I prefer tea."}],
    "user_id": "smoke-user-001",
    "session_id": "smoke-session-001"
  }'
```

Search:

```bash
curl --fail https://<host>/v1/search \
  --header 'Content-Type: application/json' \
  --header 'Authorization: Bearer <memory-system-key>' \
  --data '{
    "query": "What do I prefer?",
    "user_id": "smoke-user-001",
    "top_k": 100
  }'
```

Expected result: `data` contains an evidence record whose `content` includes
`user: I prefer tea.`

## 10. Method changes and implementation choices

This section discloses every material method change implemented in the
repository.

### 10.1 Core baseline

- Implemented a FastAPI Add/Search service for the Agent Memory Leaderboard
  contract.
- Added strict Pydantic request/response schemas with `extra="forbid"` so
  undocumented fields are rejected instead of silently changing behavior.
- Implemented synchronous Add semantics: source messages, evidence rows, FTS
  rows, optional vector rows, and idempotency metadata are committed before
  returning success.
- Implemented `request_id` idempotency using a canonical request hash.
- Implemented strict `user_id` isolation. `session_id` is stored as provenance
  only and is not used as a Search filter.
- Implemented deterministic stable IDs for message-derived memories.

### 10.2 Lexical retrieval

- Added SQLite FTS5 candidate retrieval over role-prefixed original evidence.
- Added exact phrase and token scoring components.
- Added conservative English inflection expansion for plural, past-tense, and
  `-ing` forms before FTS matching.
- Added multiple-choice option text as a lower-weight auxiliary candidate and
  scoring channel. Options are never returned as evidence and are never used to
  construct a final answer.
- Added duplicate suppression by evidence content hash.
- Added stable tie-breaking by score, source timestamp, and stable memory ID.

### 10.3 Memory evolution

- Added a narrow deterministic claim extractor for explicit first-person facts
  and preferences, such as current location, workplace, and preference changes.
- Added immutable supersession relations between newer and older conflicting
  source messages.
- Current-state searches suppress superseded evidence by default.
- Historical searches, such as questions containing `used to` or `in the past`,
  retain both older and newer evidence.
- The system still returns the original source message, never a generated
  summary or derived claim.

### 10.4 Temporal ranking

- Added a bounded recency boost for current-state queries containing cues such
  as `currently`, `latest`, or `now`.
- The boost uses source timestamps when supplied and never changes the public
  response contract.

### 10.5 Diagnostics and retention

- Added private `search_audit` diagnostics containing hashed `user_id`, hashed
  query, candidate IDs, selected IDs, counts, elapsed time, and index version.
- Audit rows do not store raw query text, memory content, gold answers, or
  rubrics.
- Fixed SQLite audit-write contention so audit lock failures do not fail an
  otherwise successful Search.
- Added retention cleanup for source data, FTS rows, embeddings, model-audit
  rows, and search-audit rows.

### 10.6 Optional vector and reranking backends

- Added an offline deterministic hashing-vector backend for infrastructure
  validation. It is local and deterministic; it is not claimed to be semantic
  retrieval.
- Added Qwen-compatible embedding/reranking support as a legacy optional
  backend.
- Added self-hosted BGE embedding and reranking support for local ablations.
  BGE payloads are bounded for 8,192-token embedding inputs and 512-token
  rerank query-document pairs.
- Added BGE chunking and embedding fingerprinting. Existing BGE vectors can be
  rebuilt with `memoria-reindex-vectors` after fingerprint changes.
- Added `gpt4o` reranking support. It uses an OpenAI-compatible
  chat-completions endpoint and defaults to `gpt-4o-mini`. It reranks only
  already-scoped candidate evidence and sends no choices, gold answers, or
  memory records from other users.

### 10.7 Offline benchmark tooling

- Added `memoria-prepare-benchmark` adapters for LoCoMo-Refined and
  LongMemEval-S.
- Added `memoria-load` to import Add-shaped JSONL through the same production
  transactional store path.
- Added `memoria-evaluate` to compute retrieval-only Recall@K, MRR, and
  nDCG@K using the same Search implementation as the HTTP service.
- Added `memoria-smoke-bge` for opt-in live BGE smoke checks.
- Added `BENCHMARK_BASELINES.md` with local retrieval-only baseline numbers.

### 10.8 Packaging and deployment

- Added a Dockerfile with Python 3.11, non-root runtime user, `/var/lib/memoria`
  storage, and port `8080`.
- Added Docker Compose for local smoke testing.
- Added deployment profiles for lexical-only, `gpt-4o-mini` reranking, and
  local BGE ablation modes.
- Ensured `.dockerignore` excludes local `.env`, virtualenvs, tests, cache, and
  local data from the Docker build context.

## 11. Validation evidence

Latest local verification:

```text
pytest: 68 passed
coverage: 81.46%
docker build: passed
container smoke: /health, /v1/add, and /v1/search passed with Bearer auth
secret scan: no real credential pattern found
```

Latest local retrieval-only baselines:

| Benchmark | Samples | Recall@100 | MRR | nDCG@100 |
| --- | ---: | ---: | ---: | ---: |
| LoCoMo-Refined lexical | 1,376 | 0.101914 | 0.034524 | 0.046419 |
| LongMemEval-S lexical | 500 | 0.523047 | 0.697894 | 0.463042 |
| LoCoMo-Refined BGE hybrid, historical cache | 1,376 | 0.126950 | 0.064069 | 0.074929 |

These are retrieval-evidence metrics only. They are not final answer accuracy.

## 12. Data handling

- Evaluation data is stored only in the configured SQLite data directory.
- No source memory body, query body, gold answer, rubric, or credential is
  written to application logs or public diagnostics.
- Cleanup command:

```bash
docker exec <container> memoria-cleanup --data-dir /var/lib/memoria
```

The default retention window is 30 days.
