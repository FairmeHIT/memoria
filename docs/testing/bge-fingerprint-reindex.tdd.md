# BGE Fingerprint Reindex TDD Evidence

## Source

Derived during this TDD run. The trigger was the change to BGE input handling:
long inputs are now split to respect the API limits, so stored vectors must be
versioned and reindexed when those semantics change.

## User Journeys

- As a maintainer, I want the BGE embedding fingerprint to change when BGE
  chunking semantics change, so old vectors do not mix with new ones.
- As a maintainer, I want `memoria-reindex-vectors` to replace stale BGE rows
  with the active fingerprint, so search uses only the current vector space.

## Evidence

| Guarantee | Test | RED evidence | GREEN evidence |
| --- | --- | --- | --- |
| BGE provider fingerprint is versioned for the new chunking semantics | `tests/unit/test_bge.py:test_bge_embedding_uses_self_hosted_contract_and_x_api_key` | Failed while asserting the old `bge-embedding-v1` fingerprint | Passed after bumping the provider fingerprint to `bge-embedding-v2` |
| Reindex replaces stale vectors after a fingerprint change and search uses the rebuilt rows | `tests/integration/test_reindex_vectors.py:test_reindex_replaces_stale_vectors_after_fingerprint_change` | Existing store logic had no regression for the v1-to-v2 migration path | Passed with `{"scanned": 2, "updated": 2}` and search returning the expected memory |

The RED command was `.venv/bin/pytest --no-cov -q tests/unit/test_bge.py`.
It failed on the fingerprint assertion before the code change.

The focused GREEN commands were:

- `.venv/bin/pytest --no-cov -q tests/unit/test_bge.py`
- `.venv/bin/pytest --no-cov -q tests/integration/test_reindex_vectors.py`

The complete verification command was `.venv/bin/pytest`:
`61 passed`, with total coverage of `81.03%`.

`python -m pip check` reported no broken requirements, and
`python -m compileall -q src tests` completed cleanly.

## Coverage and Known Gaps

This change is intentionally narrow. It does not alter vector storage format or
search scoring. It only versions the BGE provider fingerprint and documents the
required reindex step when the chunking rules change again.
