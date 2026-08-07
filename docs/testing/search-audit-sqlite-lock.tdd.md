# Search audit SQLite lock TDD evidence

## Source

No external plan file was used. The journey was derived from the observed
benchmark failure where `memoria-evaluate --workers 4` returned
`sqlite3.OperationalError: database is locked` from Search audit connection
setup.

## User journey

As a benchmark operator, I want concurrent Search evaluation to keep returning
retrieval results even if private audit diagnostics cannot be written, so that
diagnostic SQLite contention does not invalidate benchmark runs.

## Task report

| # | Guarantee | Test or command | Type | Result | Evidence |
|---|---|---|---|---|---|
| 1 | Search still returns evidence when audit connection setup raises `database is locked` | `tests/integration/test_search_audit.py::test_search_succeeds_when_audit_connection_is_locked` | integration | RED before fix | `assert 503 == 200` |
| 2 | The same lock condition no longer turns Search into a 503 response | `python -m pytest tests/integration/test_search_audit.py -q --no-cov` | integration | PASS | `2 passed` |
| 3 | Real LoCoMo concurrent evaluation works with four local workers | `memoria-evaluate ... --workers 4` | benchmark smoke | PASS | `{"samples":1376,"recall_at_k":0.101914,"mrr":0.034524,"ndcg_at_k":0.046419}` |
| 4 | Full project test suite and coverage remain above the configured threshold | `python -m pytest` | regression | PASS | `62 passed`; total coverage `81.21%` |

## Implementation summary

SQLite WAL mode is now enabled during store initialization instead of being
re-applied on every new connection. Search audit writes are serialized with the
store write lock and the whole audit connection/write path is protected by the
existing diagnostic-error suppression policy, so audit failures do not affect
successful retrieval.

## Known gaps

This fix protects in-process concurrent benchmark evaluation and suppresses
diagnostic failures caused by external SQLite contention. It does not guarantee
that every audit row is persisted under heavy cross-process write pressure,
which is intentional because audit data is private diagnostics and must not
block Search.
