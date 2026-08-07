# BGE Smoke TDD Evidence

## Source

Derived during this TDD run. The project needs a repeatable way to verify the
self-hosted BGE API after enabling the 8,192-token embedding and 512-token
rerank-pair client limits, without making default tests depend on a networked
model service.

## User Journeys

- As a maintainer, I want a manually triggered BGE smoke check, so I can verify
  the configured live model service before deployment.
- As a maintainer, I want default tests and command execution to avoid external
  network calls, so local and CI runs remain deterministic.
- As an operator, I want smoke output to omit secrets and request bodies, so
  diagnostics can be shared safely.

## Evidence

| Guarantee | Test | RED evidence | GREEN evidence |
| --- | --- | --- | --- |
| Smoke exercises long embedding, query, and document inputs through the BGE client | `tests/unit/test_bge_smoke.py:test_bge_smoke_runs_long_inputs_and_returns_body_free_summary` | Failed with `ModuleNotFoundError: No module named 'memoria.bge_smoke'` | Passed after adding `memoria.bge_smoke.run_bge_smoke` |
| Smoke requires both BGE embedding and BGE reranker backends | `tests/unit/test_bge_smoke.py:test_bge_smoke_requires_bge_backends` | Same missing-module RED | Passed with a clear `ValueError` on non-BGE settings |
| CLI skips by default unless explicitly enabled | `tests/unit/test_bge_smoke.py:test_bge_smoke_cli_skips_without_explicit_opt_in` | Same missing-module RED | Passed with `{"status": "skipped"}` and no network setup |

The RED command was `.venv/bin/pytest --no-cov -q tests/unit/test_bge_smoke.py`.
It failed during collection because the smoke module did not exist.

The focused GREEN command was `.venv/bin/pytest --no-cov -q tests/unit/test_bge_smoke.py`:
`3 passed`.

The default CLI command was `.venv/bin/memoria-smoke-bge`:
`{"status": "skipped", "reason": "set MEMORIA_BGE_SMOKE=1 or pass --run"}`.

The live BGE smoke command was
`MEMORIA_EMBEDDING_BACKEND=bge MEMORIA_RERANKER_BACKEND=bge MEMORIA_QWEN_TIMEOUT_SECONDS=10 MEMORIA_QWEN_RETRIES=0 .venv/bin/memoria-smoke-bge --run`:
`{"embedding_dimensions":1024,"embedding_fingerprint":"bge-embedding-v2:BAAI/bge-m3:1024","rerank_results":2,"status":"passed"}`.

The complete verification command was `.venv/bin/pytest`:
`61 passed`, with total coverage of `81.03%`.

## Coverage and Known Gaps

The default test suite verifies the smoke tool's behavior with fake providers.
It intentionally does not call a live BGE service. Run
`MEMORIA_EMBEDDING_BACKEND=bge MEMORIA_RERANKER_BACKEND=bge memoria-smoke-bge --run`
in an environment with `MEMORIA_BGE_BASE_URL` and a model API key to perform the
external smoke.
