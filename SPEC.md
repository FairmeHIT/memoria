# memoria Project Specification

**Status:** Design baseline

**Target:** Agent Memory Leaderboard code submission, deployed by the platform

**Version:** 0.4.0

**Last updated:** 2026-08-06

## 1. Product Positioning

`memoria` is an independent, locally runnable memory engine built for the
Agent Memory Leaderboard. It accepts conversational memories through the
Leaderboard's Add contract, stores and indexes them by evaluation user, then
returns ranked memory evidence through the Search contract.

It is a memory system, not an answer system. The Leaderboard owns answer
generation and scoring. `memoria` must never generate a final answer, use a
gold answer or rubric, or turn a multiple-choice option into an answer-like
memory record.

The intended submission route is **code submission, platform deployment**:

1. Publish a public GitHub repository.
2. Provide a Docker build and startup command.
3. Expose the Add and Search HTTP APIs in this specification.
4. Include complete deployment, configuration, and API-wrapper instructions.
5. Let the Leaderboard build, deploy, smoke-test, and evaluate the submitted
   version.

`memoria` has no dependency on an existing memory project, product, workspace,
or agent framework.

## 2. External Contract Constraints

The implementation must conform to the current Agent Memory Leaderboard
Add/Search contract documented at:

- https://agentmemories.ai/api-guide
- https://agentmemories.ai/zh-cn/docs

The following are non-negotiable:

- Add is synchronous. A successful response means all supplied messages are
  durable and immediately searchable.
- `user_id` is the only Search isolation boundary. A Search must never return
  another user's memories.
- `session_id` groups source conversations only. It must not be used as a
  Search filter.
- Search receives the original question, optional choices, `user_id`, and
  `top_k`. The formal external evaluation uses `top_k = 100`.
- Search preserves the implementation's returned order. Higher optional
  `score` values must mean greater relevance.
- Search returns evidence records only. It never calls an answer-generation
  path or returns an inferred final answer.
- The platform may retry transient Add and Search failures. Add must therefore
  be idempotent by `request_id`.
- No evaluation data or derived copy may be used for training, fine-tuning,
  product analytics, dataset reconstruction, or disclosure. Evaluation data
  must be deleted within 30 days unless the platform gives written permission
  to retain it.

If a model is used during Add or Search in a submitted evaluation version, it
must be configured to use `gpt-4o-mini`, as required by the current formal
evaluation checklist. The baseline engine must remain usable without a model
provider.

## 3. Scope

### In scope for the first implementation

- A Dockerized HTTP service with Add, Search, and health endpoints.
- SQLite-backed durable storage and FTS5 lexical retrieval.
- Per-`user_id` isolation, per-`request_id` idempotency, and deterministic
  ranking for identical stored state and request input.
- Conversation-aware records: message role, order, optional timestamp, source
  session, and source Add request.
- Search evidence containing the original memory text or an explicitly stored
  evidence-preserving representation.
- Structured minimal audit metadata without storing complete evaluation bodies
  in application logs.
- Contract, isolation, storage, retrieval, and container smoke tests.

### Explicitly out of scope for the first implementation

- A web UI, account system, hosted control plane, or multi-tenant SaaS layer.
- Answer generation, LLM judging, or changes to the Leaderboard evaluator.
- Benchmark-specific hardcoding, gold-answer access, rubric access, or
  cross-sample sharing.
- A required external vector database or network dependency.
- Cross-device synchronization, third-party sharing, or policy workflows not
  needed by the Add/Search contract.

## 4. System Architecture

```text
Leaderboard Add request
  -> request validation and authentication
  -> idempotency check
  -> immutable message and memory persistence
  -> FTS index update in the same transaction
  -> synchronous success response

Leaderboard Search request
  -> request validation and authentication
  -> user_id-scoped FTS and optional vector candidate retrieval
  -> reciprocal-rank fusion, deterministic relevance, recency, and evidence ranking
  -> top_k truncation
  -> ordered evidence response
```

The initial retrieval path is local and reproducible:

```text
SQLite FTS5 candidate retrieval
  -> exact phrase, token, and conservative inflection matching
  -> optional timestamp/recency adjustment
  -> duplicate suppression
  -> deterministic tie-breaking by stable memory ID
```

The optional vector channel persists an L2-normalized float32 vector for
each source message in SQLite. Vector scans query only rows matching the exact
`user_id` and provider fingerprint, then fuse their ranking with FTS through
reciprocal-rank fusion. The built-in hashing provider is deterministic and
offline; the self-hosted BGE provider calls `/v1/embeddings` during Add and
Search and `/v1/rerank` for final ordering. The legacy Qwen-compatible provider
remains available but is not required. Reranking receives only the original
query and already-scoped candidate evidence, never options, gold labels, or
another user's records.

When BGE chunking or input budgeting rules change, the embedding fingerprint
must also change so stored vectors can be reindexed under the new semantics.

The baseline also maintains a narrow, deterministic structured channel for
explicit first-person profile facts and preference corrections (for example,
"I no longer prefer coffee" or "I live in Paris"). It records an immutable
supersession relation between conflicting source messages; ordinary Search
suppresses superseded evidence, while historical questions retain it. Search
always returns the original message evidence, never the derived claim. Optional
learned semantic embeddings, reranking, and LLM-assisted extraction may be added later.
They must not change the wire contract, bypass `user_id` isolation, or make the
system unavailable when their provider is unset.

## 5. Planned Repository Skeleton

All project files live under `memoria/`.

```text
memoria/
  SPEC.md                     # This specification
  README.md                    # Submission-oriented overview and run guide
  pyproject.toml               # Python package, lint, test, coverage settings
  Dockerfile                   # Platform deployment image
  docker-compose.yml           # Local development convenience only
  .dockerignore
  .gitignore
  .env.example                 # Names only; never secrets
  src/memoria/
    __init__.py
    app.py                     # ASGI application composition
    config.py                  # Immutable environment configuration
    api/
      add.py                   # POST Add endpoint
      search.py                # POST Search endpoint
      health.py                # GET health endpoint
      schemas.py               # Strict request and response schemas
      auth.py                  # Token, Bearer, X-Api-Key verification
      errors.py                # Public-safe error envelopes
    domain/
      models.py                # Immutable domain data structures
      ranking.py               # Score components and deterministic ordering
    store/
      sqlite.py                # Connection lifecycle, WAL and transactions
      schema.sql               # Tables, indexes and FTS5 virtual table
      repositories.py          # Parameterized persistence operations
      migrations.py            # Schema version upgrades
    ingest/
      normalizer.py            # Message validation and canonicalization
      memories.py              # Message-to-memory evidence construction
      idempotency.py           # Request hash and duplicate behavior
    retrieval/
      lexical.py               # FTS5 retrieval
      ranker.py                # Evidence scoring and top_k selection
      dedupe.py                # Exact duplicate suppression
    retention/
      cleanup.py               # Evaluation-data expiry process
    observability/
      metrics.py               # Aggregate, body-free metrics
      logging.py               # Secret- and body-safe structured logs
  tests/
    contract/
    unit/
    integration/
    fixtures/
  docs/
    deployment.md
    operations.md
    security.md
```

No empty placeholder modules are required at specification time. Each module
will be added with a test and implementation in the same change.

## 6. Public HTTP Interface

The platform configures Add and Search URLs independently. The submitted
service will expose the following default paths:

| Method | Path | Purpose | Authentication |
| --- | --- | --- | --- |
| `POST` | `/v1/add` | Synchronously ingest a message chunk | configured scheme |
| `POST` | `/v1/search` | Retrieve ranked evidence | configured scheme |
| `GET` | `/health` | Liveness and readiness check | none |

`/health` returns a 2xx status only when SQLite is writable and the configured
schema version is ready to serve Add and Search. It does not expose keys,
stored data, or configuration values.

### 6.1 Authentication

The service supports the platform's accepted schemes:

- `Authorization: Token <secret>`
- `Authorization: Bearer <secret>`
- `X-Api-Key: <secret>`
- `none` only for a deliberately configured public smoke environment

Configuration selects exactly one scheme. The expected secret is loaded from
`MEMORIA_API_KEY`; it is never committed, logged, placed in an error body, or
returned by health checks. Authentication failures return HTTP 401. Requests
with valid credentials but forbidden scope return HTTP 403.

## 7. Add API Contract

### 7.1 Request

```http
POST /v1/add
Content-Type: application/json
Authorization: Bearer <memory-system-key>
```

```json
{
  "request_id": "eval:run_abc123:locomo_refined:conv-0:chunk-0",
  "messages": [
    {
      "role": "user",
      "timestamp": 1704067200000,
      "content": "memory text"
    },
    {
      "role": "assistant",
      "content": "assistant response"
    }
  ],
  "user_id": "eval:run_abc123:locomo:conv-0",
  "session_id": "eval:run_abc123:sample:0"
}
```

| Field | Required | Rules |
| --- | --- | --- |
| `request_id` | yes | Non-empty string. Unique idempotency key for this Add request. |
| `messages` | yes | Non-empty ordered array. Each message is stored in the supplied order. |
| `messages[].role` | yes | `user` or `assistant`. |
| `messages[].content` | yes | Non-empty string containing the original supplied message text. |
| `messages[].timestamp` | no | Integer Unix timestamp in milliseconds. |
| `user_id` | yes | Non-empty retrieval namespace. It is the required isolation key. |
| `session_id` | yes | Non-empty source conversation identifier; organization only. |

Only these public contract fields are accepted. Unknown fields are rejected
with HTTP 422 instead of silently changing storage semantics.

### 7.2 Required Add behavior

1. Validate authentication and the entire request before writing any record.
2. Compute a canonical hash of the request body.
3. If `request_id` was committed with the same canonical hash, return the
   original successful response without writing duplicate memories.
4. If `request_id` was committed with a different canonical hash, return HTTP
   409 with a body-safe conflict reason.
5. Persist messages, derived memory evidence, and FTS index entries within one
   SQLite transaction.
6. Commit the transaction before returning HTTP 200.
7. Ensure a Search on the request's `user_id` can observe the newly committed
   records before the response is sent.

### 7.3 Success response

```http
HTTP/1.1 200 OK
Content-Type: application/json
```

```json
{
  "success": true,
  "request_id": "eval:run_abc123:locomo_refined:conv-0:chunk-0",
  "user_id": "eval:run_abc123:locomo:conv-0",
  "session_id": "eval:run_abc123:sample:0"
}
```

All three identifiers must exactly match the request. The endpoint must not
return HTTP 202, an asynchronous task ID, polling URL, or memory IDs.

## 8. Search API Contract

### 8.1 Request

```http
POST /v1/search
Content-Type: application/json
Authorization: Bearer <memory-system-key>
```

```json
{
  "query": "Which answer best matches the memory?",
  "options": ["A. First answer", "B. Second answer"],
  "user_id": "eval:run_abc123:locomo:conv-0",
  "top_k": 100
}
```

| Field | Required | Rules |
| --- | --- | --- |
| `query` | yes | Non-empty original benchmark question. It is searched verbatim. |
| `options` | no | Array of non-empty option strings for choice questions. |
| `user_id` | yes | Non-empty namespace. Every candidate and returned record must match it exactly. |
| `top_k` | yes | Positive integer. The response contains at most this many records. |

`filters`, `rerank`, `keyword_search`, answer fields, rubric fields, and gold
answer fields are not part of the contract and must be rejected with HTTP 422.

### 8.2 Required Search behavior

1. Validate authentication and request fields.
2. Query only records whose persisted `user_id` equals the requested value.
3. Use `query` and optional `options` solely to rank memory evidence.
4. Do not access evaluator labels, rubric text, hidden data, or memories from
   other `user_id` namespaces.
5. Return evidence in final descending relevance order. Tie-breaking must be
   stable and documented.
6. Return `{"data": []}` when the namespace has no relevant evidence.
7. Return at most `top_k` results, including when `top_k` exceeds the number
   of stored memories.

### 8.3 Success response

```http
HTTP/1.1 200 OK
Content-Type: application/json
```

```json
{
  "data": [
    {
      "id": "mem_01J5J7JX7Z4S1Y3C3ZKQ6YQ2F6",
      "content": "user: I prefer vegetarian meals when traveling.",
      "score": 0.87,
      "created_at": "2026-07-01T12:00:00Z"
    }
  ]
}
```

| Field | Required | Rules |
| --- | --- | --- |
| `data` | yes | Array. It is always present, including for zero results. |
| `data[].id` | yes | Stable, non-empty memory identifier. |
| `data[].content` | yes | Non-empty memory evidence supplied directly to the shared answer pipeline. |
| `data[].score` | no | Numeric relevance score; greater means more relevant. |
| `data[].created_at` | no | Source or persistence timestamp in ISO 8601 UTC form. |

The endpoint must return an object with `data`; it must not return a bare array
or wrap results in `items`. Unspecified fields are not relied on by the
platform and should not carry answer-like hints.

The service may record an internal Search trace containing hashes of `user_id`
and `query`, candidate and selected memory IDs, counts, elapsed milliseconds,
and an index-version string. It must not store the query or memory content in
that trace, expose the trace through the public API, or allow trace persistence
failures to fail an otherwise successful Search.

## 9. Persistence and Data Model

SQLite is the source of truth for the first release. It runs with WAL enabled,
foreign keys enabled, bounded transactions, and parameterized SQL only.

The logical records are immutable. Corrections or state changes create a new
row or version relation; they never mutate the original message body.

| Record | Essential fields | Purpose |
| --- | --- | --- |
| `add_requests` | `request_id`, `request_hash`, `user_id`, `session_id`, `committed_at` | Idempotency and minimal request lifecycle. |
| `messages` | `id`, `user_id`, `session_id`, `sequence`, `role`, `content`, `observed_at`, `add_request_id` | Original ordered conversation evidence. |
| `memories` | `id`, `user_id`, `session_id`, `message_id`, `content`, `created_at`, `content_hash` | Searchable evidence unit. |
| `memory_claims` | `memory_id`, `user_id`, `predicate`, `value_key`, `polarity` | Narrow, rule-derived retrieval metadata with message provenance. |
| `memory_supersessions` | `superseding_memory_id`, `superseded_memory_id` | Append-only relation used to avoid returning obsolete explicit claims by default. |
| `memory_fts` | FTS5 index over `memories.content` | Local lexical candidate retrieval. |
| `memory_embeddings` | `memory_id`, `user_id`, provider fingerprint, dimension, normalized float32 BLOB | Optional local vector candidate retrieval, atomically committed with its source. |
| `search_audit` | `trace_id`, hashed identifiers, candidate/selection IDs, timing, index version | Body-free retrieval diagnostics, retained only within the evaluation window. |
| `retention_jobs` | `user_id`, `expires_at`, `status` | Scheduled deletion proof without keeping request bodies in logs. |

The initial evidence unit is one normalized source message. This favors exact
names, timestamps, roles, and ordered facts required by memory benchmarks.
Later versions may add derived facts or summaries as separate records with
clear provenance, never as replacements for the original evidence.

## 10. Ranking Requirements

The first ranker must be deterministic and explainable. Its minimum inputs are
lexical relevance, optional choice text, explicit supersession state, exact
phrase matches, message timestamp when present, and stable IDs for tie-breaking.

Ranking rules:

1. Limit candidates to the requested `user_id` before scoring.
2. Rank evidence against the question and optional option text; options may
   improve retrieval but must not produce a selected option as output.
3. Use conservative inflection variants as a candidate channel and score input;
   never replace the original message text.
4. Use optional choices only as an auxiliary candidate channel and score input;
   never return a choice as evidence or infer an answer from it.
5. For a narrow, explicit correction relation, prefer the newer source message
   for ordinary current-state questions. Historical questions retain both
   source messages.
6. For current-state cues such as `currently`, `latest`, or `now`, apply a
   bounded candidate-relative recency boost using source event timestamps.
7. Keep original evidence content, role prefix, and timestamp information when
   they are available and helpful to the answer pipeline.
8. Suppress exact duplicate evidence while preserving the best-ranked record.
9. Never replace evidence with generated prose in the Search response.
10. Never exceed `top_k`.

## 11. Errors, Safety, and Retention

Error bodies use a safe, machine-readable envelope:

```json
{
  "detail": {
    "reason": "human-readable reason without secrets or memory content"
  }
}
```

| Status | Meaning |
| --- | --- |
| `400` | Malformed JSON or semantically invalid request not represented by a field error. |
| `401` | Missing or invalid configured API credential. |
| `403` | Valid credential is not authorized for the requested operation. |
| `409` | Conflicting duplicate `request_id` or a transient write conflict. |
| `422` | Missing, extra, empty, or incorrectly typed API field. |
| `429` | Locally applied capacity limit. |
| `500` | Unexpected internal failure; never expose a traceback or secret. |
| `503` | Service not ready, database unavailable, or required provider unavailable. |

Security requirements:

- Validate all untrusted request data at the API boundary.
- Use parameterized SQL. Never interpolate a user field into SQL, FTS syntax,
  a filesystem path, or a shell command.
- Escape FTS query syntax so input cannot alter the intended retrieval query.
- Do not log authorization headers, API keys, full request bodies, returned
  memory contents, gold answers, or rubric material.
- Store operational logs and aggregate metrics separately from evaluation data.
- Search traces contain no raw query or memory body and are deleted by the same
  retention process as source and derived records.
- Implement a scheduled retention process that removes source data, indexes,
  and derived records no later than 30 days after the evaluation run.
- Ensure cleanup is idempotent and leaves only non-sensitive aggregate
  operational counters needed to prove execution health.

## 12. Deployment Contract

The repository must be deployable by the platform without local machine state.

The completed repository must include:

- `Dockerfile` with a reproducible Python runtime and non-root runtime user.
- A documented container command, for example
  `uvicorn memoria.app:app --host 0.0.0.0 --port 8080`.
- A writable persistent data directory configured by `MEMORIA_DATA_DIR`.
- `MEMORIA_API_KEY` and `MEMORIA_AUTH_SCHEME` configuration, documented without
  real credentials.
- Self-hosted BGE mode uses `API_KEY` or `MEMORIA_BGE_API_KEY`,
  `MEMORIA_BGE_BASE_URL`, and `MEMORIA_BGE_AUTH_SCHEME`; these are runtime
  environment variables and must never be committed or logged.
- A documented `GET /health` endpoint.
- Startup migration execution before the service accepts traffic.
- `README.md` instructions for build, run, configuration, endpoint registration,
  smoke verification, tests, and data cleanup.

The container must not download private datasets, read host paths, require a
desktop GUI, use embedded credentials, or rely on a locally running service.

## 13. Quality Gates

Implementation is complete for the first release only when all of the
following pass:

1. Contract tests verify exact Add and Search success bodies, required fields,
   prohibited extra fields, empty-result response, and `top_k` truncation.
2. Add idempotency tests prove a repeated identical request writes no duplicate
   memories and returns the same success payload.
3. Isolation tests prove that querying one `user_id` cannot return any evidence
   stored under another `user_id`, regardless of matching query terms.
4. Transaction tests prove an Add success response is followed immediately by
   searchable records.
5. Ranking tests prove stable ordering, descending score behavior, duplicate
   suppression, and timestamp preservation.
6. Security tests cover invalid authentication, FTS special characters,
   malformed JSON, oversized input limits, duplicate IDs, and secret-free error
   responses.
7. Retention tests prove the cleanup job removes expired source and derived
   data.
8. Integration tests run against the built Docker image, call `/health`, Add,
   and Search, and verify the public contract end to end.
9. Unit and integration coverage is at least 80% for `src/memoria/`.

## 14. Delivery Sequence

1. Create package metadata, strict API schemas, configuration, and contract
   tests before endpoint implementations.
2. Implement SQLite schema, migrations, transactional repositories, and Add
   idempotency.
3. Implement FTS5 retrieval and deterministic Search ordering.
4. Add authentication, health checks, safe logging, retention, and container
   deployment assets.
5. Run the Docker contract suite locally.
6. Write the public submission README, Docker command, method disclosure, and
   deployment notes required for platform review.
7. Run the platform smoke compatibility test before requesting a formal Full
   evaluation.

## 15. Definition of Success

`memoria` is successful when the platform can build the public repository,
start the container, call Add and Search concurrently through the documented
URLs, receive strictly conformant synchronous responses, and evaluate only
ordered memory evidence scoped to the supplied `user_id`.

## 16. Offline Evaluation Harness

For authorized local data preparation, `memoria-load` imports one exact
Leaderboard Add request per JSONL line into the normal transactional store.
It prints only aggregate request/message counts and retains Add idempotency.

The repository includes `memoria-evaluate`, an offline command that reuses the
same store Search path and accepts one JSON object per line:

```json
{"query":"...","options":["..."],"user_id":"...","relevant_ids":["mem_..."],"top_k":100}
```

It reports aggregate Recall@K, mean reciprocal rank, and nDCG@K without
printing query text, memory content, or labels. This harness is for local,
authorized benchmark analysis and is not part of the public Add/Search API.
It can use `--workers 1..16` to run independent local evaluation queries in
parallel; the production service semantics remain unchanged.

`memoria-prepare-benchmark` currently supports the downloaded LoCoMo-Refined
and LongMemEval-S layouts. It emits only Add-shaped source records and
evidence-ID evaluation cases. LoCoMo uses public turn-level evidence IDs;
LongMemEval expands official answer session IDs to source message IDs. The
mapping and baseline results are documented in `BENCHMARK_BASELINES.md`.

## 17. Vector Reindexing and Model Diagnostics

`memoria-reindex-vectors --data-dir <path> --batch-size <1..500>` backfills
only source messages missing vectors for the currently configured embedding
provider fingerprint. It does not replay Add requests, change source evidence,
or bypass the `user_id` stored with each vector. A repeated run with the same
provider is a no-op for already indexed messages.

When Qwen providers are configured, the service records body-free per-call
diagnostics in the local `model_audit` table: operation, provider, model,
input count, provider token counts when returned, retry attempts, elapsed time,
success state, and a generic error kind. It never records API keys, request
headers, queries, options, memory bodies, embeddings, or reranking results.
Diagnostic persistence is best effort and cannot turn a successful Add or
Search operation into a failure. Retention cleanup removes these records with
the evaluation data.
