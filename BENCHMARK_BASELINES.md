# Benchmark Baselines

These are local retrieval-only baselines produced on 2026-08-05 from the
authorized files under `../benchmarks`. They measure whether the evidence IDs
used to build a Search evaluation case are retrieved; they are not final answer
accuracy and do not run any benchmark judge.

## LoCoMo-Refined

Source: [mem-eval-suite/LoCoMo_refined](https://github.com/mem-eval-suite/LoCoMo_refined)

- Add chunks: `399`
- Messages: `5,882`
- Evaluation questions: `1,376`
- Recall@100: `0.101914`
- MRR: `0.034524`
- nDCG@100: `0.046419`

Relevant IDs come from each question's public `evidence` turn IDs. The
multimodal image itself is not sent to memoria; only the available text turn is
ingested.

### Local Vector Ablation

The optional deterministic hashing-vector channel was evaluated on the same
LoCoMo fixture on 2026-08-06. It uses SQLite-scoped exact cosine retrieval and
RRF with FTS, but no learned model or network dependency:

- Recall@100: `0.101914`
- MRR: `0.034524`
- nDCG@100: `0.046419`

It preserves the FTS baseline ranking while supplying a vector-only candidate
when lexical retrieval has no match. This is an infrastructure validation, not
evidence that hashing is a semantic retrieval model.

### Self-Hosted BGE Hybrid Retrieval

The self-hosted `BAAI/bge-m3` embedding service and
`BAAI/bge-reranker-v2-m3` reranker were evaluated through the production
`memoria-load` and `memoria-evaluate` paths. The run used a fresh SQLite store,
`top_k=100`, and two independent local evaluation workers.

- Add chunks: `399`
- Messages and persisted vectors: `5,882`
- Evaluation questions: `1,376`
- Recall@100: `0.126950` (`+24.57%` vs. lexical baseline)
- MRR: `0.064069` (`+85.58%` vs. lexical baseline)
- nDCG@100: `0.074929` (`+61.42%` vs. lexical baseline)
- Model calls: `1,775` embeddings and `1,376` reranks, all successful

This is the current semantic-retrieval baseline for LoCoMo. It remains a
retrieval-evidence measurement rather than final answer accuracy.

#### Rerank Candidate Ablation

The same fresh BGE store and 1,376 questions were re-evaluated with only the
rerank candidate limit changed:

| Candidate limit | Recall@100 | MRR | nDCG@100 | Mean rerank latency | Rerank failures |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 500 | 0.126950 | 0.064069 | 0.074929 | 1,183 ms | 0 |
| 200 | 0.127071 | 0.063950 | 0.074813 | 820 ms | 0 |
| 100 | 0.104155 | 0.061802 | 0.068602 | 608 ms | 0 |

`200` is the recommended submission default: it reduces mean rerank latency
by about 31% versus 500 while preserving the retrieval metrics. A limit of
100 is faster but loses most of the Recall@100 gain over the lexical baseline.

## LongMemEval-S

Source: [xiaowu0162/LongMemEval](https://github.com/xiaowu0162/LongMemEval)

- Add chunks: `24,129`
- Messages: `246,738`
- Evaluation questions: `500`
- Recall@100: `0.523047`
- MRR: `0.697894`
- nDCG@100: `0.463042`

Relevant IDs are expanded from each question's official `answer_session_ids`
to all source messages in those sessions. This is a session-level retrieval
metric and is intentionally not compared directly with LoCoMo turn-level
evidence metrics.

## Reproduction

The adapters write only Add-shaped records and evaluation cases; answer text,
rubrics, and other gold fields are excluded from generated files.

```bash
memoria-prepare-benchmark \
  --benchmark locomo_refined \
  --input ../benchmarks/locomo_refined/data/public/conversations.jsonl \
  --questions ../benchmarks/locomo_refined/data/public/questions.jsonl \
  --adds-out /tmp/memoria-locomo/adds.jsonl \
  --eval-out /tmp/memoria-locomo/eval.jsonl

memoria-load --data /tmp/memoria-locomo/adds.jsonl --data-dir /tmp/memoria-locomo-db
memoria-evaluate --data /tmp/memoria-locomo/eval.jsonl --data-dir /tmp/memoria-locomo-db
```

The LongMemEval command uses `--benchmark longmemeval` and takes the downloaded
`longmemeval_s_cleaned.json` as `--input`. LongMemEval-M is not included in the
local baseline.
