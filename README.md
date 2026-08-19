# memoria

Deployment and method disclosure:

- [DEPLOYMENT.md](DEPLOYMENT.md)
- [SPEC.md](SPEC.md)
- [BENCHMARK_BASELINES.md](BENCHMARK_BASELINES.md)

Public endpoints:

| Method | Path | Purpose |
| --- | --- | --- |
| `GET` | `/health` | Liveness and readiness |
| `POST` | `/v1/add` | Synchronous memory ingestion |
| `POST` | `/v1/search` | Ranked evidence retrieval |

Quick start:

```bash
# 1. Clone and install
git clone https://github.com/FairmeHIT/memoria.git
cd memoria

# 2. Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# 3. Install core dependencies
pip install -e .

# 4. (Optional) Install local BGE embedding for hybrid retrieval (+10% recall)
#     The model (bge-small-en-v1.5, 130 MB) is downloaded automatically
#     from the Hugging Face mirror (hf-mirror.com) on first use.
pip install -e ".[bge]"

# 5. (Optional) Install dev tools for running tests
pip install -e ".[dev]"

# 6. Run with default settings (pure lexical, no external models)
export MEMORIA_API_KEY='<memory-system-key>'
export MEMORIA_AUTH_SCHEME=bearer
export MEMORIA_DATA_DIR=./data
export MEMORIA_EMBEDDING_BACKEND=none
curl --fail http://localhost:8080/health
```

Or with Docker (includes BGE dependencies):

```bash
docker build --tag memoria:0.4.0 .

docker volume create memoria-data

docker run --rm --publish 8080:8080 \
  --env MEMORIA_API_KEY='<memory-system-key>' \
  --env MEMORIA_AUTH_SCHEME=bearer \
  --env MEMORIA_DATA_DIR=/var/lib/memoria \
  --env MEMORIA_EMBEDDING_BACKEND=none \
  --env MEMORIA_RERANKER_BACKEND=none \
  --mount source=memoria-data,target=/var/lib/memoria \
  memoria:0.4.0
```

Smoke check:

```bash
curl --fail http://localhost:8080/health
```

For the full wrapper examples, deployment profiles, authorship, and method
changes, see [DEPLOYMENT.md](DEPLOYMENT.md).

## Retrieval enhancements (v0.4.1)

Code-level optimizations informed by the AML Top-10 source-code analysis:

1. **Chinese support** — CJK 2/3-gram indexing and query expansion plus a
   Chinese stopword list, so Chinese conversations are recallable without an
   external segmenter. FTS5 uses `remove_diacritics 2`.
2. **Dual FTS5 indexes** — an additional `porter unicode61` index handles
   English inflection (`preferences` → `prefer`, `running` → `run`) and is
   queried in parallel with the raw index.
3. **Temporal-intent-aware ranking** — queries are classified into
   `latest` / `earliest` / `sequence` / `point` / `none` (English + Chinese
   markers). Newer evidence is boosted for `latest`, older for `earliest`,
   and superseded memories are retained for historical questions.
4. **Relative-time resolution** — `yesterday`, `last week`, `next month` etc.
   are resolved to absolute dates and appended to returned evidence, giving the
   answer model exact temporal anchors.
5. **Semantic concept expansion** — 20 concept groups (e.g. `like/love/enjoy`
   ↔ `prefer/favorite`) widen lexical recall for paraphrase queries.
6. **Conversation context window** — each returned hit is expanded with up to
   `CONTEXT_RADIUS=2` adjacent messages from the same session (`>>>` marks the
   anchor hit), so downstream evidence covers multi-turn exchanges.

All tuning constants live at the top of `src/memoria/store.py` for ablation.

## Embedding backends

| Backend | Env var | Description | Dependencies |
|---------|---------|-------------|-------------|
| `none` | `MEMORIA_EMBEDDING_BACKEND=none` | Pure lexical (FTS5 + CJK n-gram) | None (default) |
| `hashing` | `MEMORIA_EMBEDDING_BACKEND=hashing` | Dense lexical vectors via feature hashing | None |
| `local` | `MEMORIA_EMBEDDING_BACKEND=local` | **Local BGE via sentence-transformers (CPU)** | `pip install -e ".[bge]"` |
| `bge` | `MEMORIA_EMBEDDING_BACKEND=bge` | Self-hosted BGE HTTP API | External service |
| `qwen` | `MEMORIA_EMBEDDING_BACKEND=qwen` | Alibaba Cloud Qwen API | `DASHSCOPE_API_KEY` |

The `local` backend mirrors InvMem's approach: loads `BAAI/bge-small-en-v1.5`
(384 dim, ~130 MB RAM, CPU-only) via `sentence-transformers` with the BGE
query prefix `"Represent this sentence for searching relevant passages: "`.
The model is **downloaded automatically** from the Hugging Face mirror
(hf-mirror.com) on first use — no manual download step is needed.
To use a different model, set `MEMORIA_BGE_EMBEDDING_MODEL` (e.g. `BAAI/bge-m3`).

## Dimension regression testing

The `memoria-dimension-eval` CLI targets each of the 7 AML evaluation
dimensions with synthetic scenarios. After any code change, run:

```bash
# Pure lexical (no embeddings)
MEMORIA_AUTH_SCHEME=none memoria-dimension-eval

# With local BGE
MEMORIA_AUTH_SCHEME=none MEMORIA_EMBEDDING_BACKEND=local memoria-dimension-eval

# Save a JSON report for comparison
MEMORIA_AUTH_SCHEME=none memoria-dimension-eval --report-out baseline.json
```

### One-click evaluation script

The repo ships `eval.sh` with all reproduction commands:

```bash
./eval.sh              # 维度评测（纯词法）
./eval.sh local        # 维度评测（本地 BGE 混合）
./eval.sh full         # LoCoMo 完整基准（纯词法，自动下载数据）
./eval.sh full local   # LoCoMo 完整基准（本地 BGE 混合）
```

The report shows per-dimension Recall@100, MRR, and nDCG, plus an overall
aggregate. To integrate into CI, the same tests run via pytest:

```bash
pytest tests/ -k dimension -v
```

The pytest tests assert each dimension's Recall@100 stays above a baseline
threshold (defined in `tests/integration/test_dimension_eval.py`), acting as
a regression guard.
