# memoria

`memoria` is a local, Docker-deployable memory engine for the Agent Memory
Leaderboard. It accepts synchronous Add requests, stores evidence in a
`user_id`-isolated SQLite/FTS5 database, and returns ranked memory evidence
through Search. The Leaderboard owns answer generation and scoring.

The design baseline and full contract rationale are in [SPEC.md](SPEC.md).
Academic-track deployment and method disclosure are in
[DEPLOYMENT.md](DEPLOYMENT.md).

## Leaderboard Submission

This repository is designed for the **code submission, platform deployment**
route. Submit a public GitHub repository with this Docker build, the startup
command below, the Add/Search URLs, and an exact version or commit identifier.

The platform configures the Add and Search URLs independently. The default
paths are:

| Method | Path | Purpose |
| --- | --- | --- |
| `POST` | `/v1/add` | Synchronously ingest ordered conversation messages. |
| `POST` | `/v1/search` | Return ordered memory evidence. |
| `GET` | `/health` | Unauthenticated liveness and readiness check. |

`memoria` never generates final answers, accesses gold answers or rubrics, or
returns memory belonging to another `user_id`.

## Retrieval Method

The baseline is fully local and deterministic. SQLite FTS5 retrieves candidates
from the original question; for multiple-choice requests, option text provides
an auxiliary candidate channel with lower ranking weight. A conservative
inflection channel expands common plural, past-tense, and `-ing` forms before
FTS matching. The service stores the original role-prefixed message as the
returned evidence.

For a deliberately narrow set of explicit first-person facts and preference
corrections, such as `I no longer prefer coffee`, `I live in Paris`, or `I work
at Example Labs`, memoria appends a relation from the newer source message to
the older conflicting source message. Current-state searches prefer the newer
evidence, while questions containing historical cues such as `used to` retain
both. No generated summary or derived claim is ever returned.

Each Search also writes a private, body-free trace with hashed request
identifiers, candidate/selected IDs, counts, elapsed time, and the index
version. It is for recall and ranking diagnostics only, is not exposed by HTTP,
and is removed by `memoria-cleanup` with the evaluation data.

Queries with current-state cues such as `currently`, `latest`, or `now` receive
a bounded boost for newer source event timestamps. This does not replace the
original evidence or alter the public Search contract.

`MEMORIA_EMBEDDING_BACKEND=hashing` enables a second, entirely local dense
candidate channel. It stores normalized vectors in SQLite under the same
`user_id` boundary as source evidence, performs an exact cosine scan only
inside that namespace, and fuses vector and FTS ranks with reciprocal-rank
fusion. Add writes source, FTS, and vector rows in one transaction; a vector
failure returns an error rather than exposing a partially searchable request.
The default `none` preserves the lexical baseline. For the self-hosted BGE
profile, set `MEMORIA_EMBEDDING_BACKEND=bge` and
`MEMORIA_RERANKER_BACKEND=bge`. Add sends source message text to
`POST /v1/embeddings`; Search embeds only the original query, retrieves strictly
within `user_id`, then sends at most 500 already-scoped evidence candidates to
`POST /v1/rerank`. The client accepts either `Authorization: Bearer` or
`X-API-Key`, and sends `return_documents=false` so reranking never returns
duplicate document bodies. Optional choices are never sent to either model API.
Before calling BGE, it conservatively bounds embedding payloads to the 8,192
token limit and each rerank query-document pair to the 512-token limit. It uses
UTF-8 byte length as a safe upper bound for byte-fallback tokenization and
reserves eight tokens for model special tokens. Long embedding inputs are
split and length-weighted back into one vector; long rerank queries and
documents are split into bounded pairs, with the highest chunk score retained
per original document. Because these chunking semantics affect persisted
vectors, the BGE embedding fingerprint is versioned; after changing this logic,
run `memoria-reindex-vectors` to rebuild existing BGE rows under the active
fingerprint. A model API failure returns a safe 503 response; neither API key
nor memory content is written to application logs. The Qwen-compatible backend
remains available by explicitly selecting `qwen`.

## Requirements

- Python 3.11 or newer for local development.
- Docker for the platform-equivalent local smoke test.
- No external database or model provider is required for the baseline.

## Local Development

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -e '.[dev]'
.venv/bin/python -m pytest
```

The test suite verifies the external contract, synchronous visibility after
Add, idempotency, `user_id` isolation, authentication, retention cleanup, and
at least 80% source coverage.

## Offline Retrieval Evaluation

For authorized local benchmark fixtures, first import Add-shaped JSONL through
the same transactional store path:

```bash
memoria-load --data adds.jsonl --data-dir ./data
```

Each line in `adds.jsonl` is exactly one Leaderboard Add request. The command
prints only aggregate request/message counts and preserves Add idempotency.

`memoria-evaluate` reuses the production Search implementation against a local
JSONL file. Each line contains a query, namespace, and manually or benchmark
annotated relevant memory IDs:

```json
{"query":"What meals do I prefer?","user_id":"user-a","relevant_ids":["mem_..."]}
```

Run it against the same data directory used by a local service:

```bash
memoria-evaluate --data eval-cases.jsonl --data-dir ./data --workers 4
```

The command emits aggregate `recall_at_k`, `mrr`, and `ndcg_at_k` values. It
does not print queries, memory content, or labels. `--workers` is a bounded
local-only evaluation accelerator (default `1`, maximum `16`); it does not
change the deployed HTTP service's request handling.
Pass `--report-out metrics.json` to persist only the final aggregate JSON for a
long-running local benchmark.

When enabling or changing an embedding provider for an existing database, do
not replay Add requests. Backfill only source messages that lack vectors for
the active model fingerprint:

```bash
memoria-reindex-vectors --data-dir ./data --batch-size 100
```

The command requires an enabled embedding backend and prints only
`scanned`/`updated` totals. It is idempotent for an unchanged provider
fingerprint. Qwen calls are recorded in the local `model_audit` SQLite table as
aggregate operation, model, count, token, retry, latency, and success fields;
it contains no API keys, queries, document bodies, or result content. Retention
cleanup deletes these diagnostics with evaluation data.

To verify a live self-hosted BGE service after configuring `.env`, run the
opt-in smoke check:

```bash
MEMORIA_EMBEDDING_BACKEND=bge MEMORIA_RERANKER_BACKEND=bge memoria-smoke-bge --run
```

The smoke sends long embedding, query, and document inputs through the BGE
client so the client-side 8,192-token embedding and 512-token rerank-pair
budgeting is exercised against the real API. Its output is a body-free JSON
summary and does not print API keys, queries, documents, or memory content.

Adapters for the downloaded LoCoMo-Refined and LongMemEval-S layouts are
available through `memoria-prepare-benchmark`. Their local retrieval-only
results and exact evidence mapping are recorded in
[BENCHMARK_BASELINES.md](BENCHMARK_BASELINES.md). These figures are not final
answer scores; they are used to compare retrieval changes before running an
official judge.

## Container Deployment

Build the image:

```bash
docker build --tag memoria:0.4.0 .
```

Create a persistent Docker volume, then start the service with a Memory System
Key:

```bash
docker volume create memoria-data

docker run --rm --publish 8080:8080 \
  --env-file .env \
  --mount source=memoria-data,target=/var/lib/memoria \
  --env MEMORIA_DATA_DIR=/var/lib/memoria \
  memoria:0.4.0
```

The Docker entrypoint is:

```bash
uvicorn memoria.app:create_app --factory --host 0.0.0.0 --port 8080
```

`docker compose up --build` offers the same local service. Keep real `.env`
files out of source control. For formal evaluation, start from `.env.none` for
the lexical baseline or `.env.gpt4o` for `gpt-4o-mini` reranking; use `.env.bge`
only for local ablations unless the current competition rules explicitly allow
self-hosted non-`gpt-4o-mini` model calls during Add/Search.

Ready-made deployment profiles are included and mirror the retrieval choices
from the Leaderboard Full gate:

- `.env.none` — pure lexical retrieval, no external model. Fully satisfies the
  `gpt-4o-mini` Full-gate item because Add/Search call no model.
- `.env.gpt4o` — lexical retrieval plus a `gpt-4o-mini` chat reranker. The only
  remote model used during Search is `gpt-4o-mini`; supply your own
  `MEMORIA_GPT4O_API_KEY` (OpenAI key) and pay the token cost.
- `.env.bge` — self-hosted BGE embedding and reranking. Strong retrieval, but
  uses models other than `gpt-4o-mini`, so the Full-gate `gpt-4o-mini` item
  cannot be confirmed honestly.

Copy the chosen profile to `.env`, fill in real secrets, then deploy.

## Configuration

| Variable | Required | Default | Meaning |
| --- | --- | --- | --- |
| `MEMORIA_API_KEY` | when auth is enabled | none | Memory System Key supplied to the Leaderboard. |
| `MEMORIA_AUTH_SCHEME` | no | `bearer` | `bearer`, `token`, `x_api_key`, or `none`. |
| `MEMORIA_DATA_DIR` | no | `./data` | Writable SQLite data directory. |
| `MEMORIA_RETENTION_DAYS` | no | `30` | Maximum local lifetime for evaluation data. |
| `MEMORIA_MAX_TOP_K` | no | `1000` | Highest accepted Search `top_k`. |
| `MEMORIA_EMBEDDING_BACKEND` | no | `none` | `none`, `hashing`, `qwen`, or self-hosted `bge`. |
| `MEMORIA_EMBEDDING_MODEL` | for `qwen` | `text-embedding-v4` | Legacy Qwen embedding model. |
| `MEMORIA_BGE_EMBEDDING_MODEL` | for `bge` | `BAAI/bge-m3` | Self-hosted embedding model name. |
| `MEMORIA_EMBEDDING_DIMENSIONS` | no | `384` | Stored vector dimension; BGE-M3 normally uses `1024`. |
| `MEMORIA_EMBEDDING_BASE_URL` | for `qwen` | none | Legacy Qwen compatible-mode base URL. |
| `MEMORIA_BGE_BASE_URL` | for `bge` | `http://127.0.0.1:8000` | Model API host; client appends `/v1/embeddings` and `/v1/rerank`. |
| `MEMORIA_BGE_API_KEY` | for `bge` | `API_KEY` alias | Self-hosted model API key. |
| `MEMORIA_BGE_AUTH_SCHEME` | no | `bearer` | `bearer` or `x_api_key`. |
| `MEMORIA_VECTOR_CANDIDATE_LIMIT` | no | `500` | Per-user vector candidates inspected before fusion. |
| `MEMORIA_RRF_K` | no | `60` | Reciprocal-rank fusion smoothing constant. |
| `MEMORIA_RERANKER_BACKEND` | no | `none` | `none`, `qwen`, `bge`, or `gpt4o`. |
| `MEMORIA_RERANKER_MODEL` | for `qwen` | `qwen3-rerank` | Legacy Qwen compatible reranker model. |
| `MEMORIA_RERANKER_BASE_URL` | for `qwen` | none | Legacy Qwen compatible-api base URL. |
| `MEMORIA_BGE_RERANKER_MODEL` | for `bge` | `BAAI/bge-reranker-v2-m3` | Self-hosted reranker model name. |
| `MEMORIA_RERANKER_CANDIDATE_LIMIT` | no | `200` | Maximum already-scoped candidates sent to reranking; 200 is the validated BGE default. |
| `MEMORIA_GPT4O_API_KEY` | for `gpt4o` | `OPENAI_API_KEY` alias | Your OpenAI key for the gpt-4o-mini reranker. |
| `MEMORIA_GPT4O_BASE_URL` | no | `https://api.openai.com/v1` | OpenAI-compatible base URL for chat completions. |
| `MEMORIA_GPT4O_MODEL` | no | `gpt-4o-mini` | Chat model used to score rerank candidates. |
| `MEMORIA_QWEN_API_KEY` | for Qwen | none | DashScope key. `DASHSCOPE_API_KEY` is accepted as an alias. |
| `MEMORIA_QWEN_TIMEOUT_SECONDS` | no | `15` | Per-request Qwen timeout. |
| `MEMORIA_QWEN_RETRIES` | no | `2` | Retry count for transient Qwen transport failures. |

`none` is only appropriate for an intentionally public smoke environment.
Production and formal evaluation deployments must configure a secret using one
of `bearer`, `token`, or `x_api_key`.

## Smoke Check

```bash
curl --fail http://localhost:8080/health

curl --fail http://localhost:8080/v1/add \
  --header 'Content-Type: application/json' \
  --header 'Authorization: Bearer replace-with-a-long-random-secret' \
  --data '{
    "request_id": "local-001",
    "messages": [{"role": "user", "content": "I prefer vegetarian meals."}],
    "user_id": "local-user-001",
    "session_id": "local-session-001"
  }'

curl --fail http://localhost:8080/v1/search \
  --header 'Content-Type: application/json' \
  --header 'Authorization: Bearer replace-with-a-long-random-secret' \
  --data '{"query": "What meals do I prefer?", "user_id": "local-user-001", "top_k": 100}'
```

The Add response must echo `request_id`, `user_id`, and `session_id`, with
`success: true`. The Search response must have a top-level `data` array whose
records contain non-empty `id` and `content` fields.

## Data Handling

Evaluation data and derived records are used only to execute the active
evaluation. They are never used for model training, analytics, dataset
reconstruction, or external sharing. The retention cleanup API is internal to
the service and removes expired source and derived records. The deployment
operator must schedule the following command at least daily:

```bash
memoria-cleanup
```
