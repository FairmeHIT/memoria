from __future__ import annotations

import json
import sqlite3

from tests.conftest import add_payload


def test_search_writes_body_free_trace_with_candidate_and_selection_counts(client) -> None:
    assert client.post("/v1/add", json=add_payload()).status_code == 200

    response = client.post(
        "/v1/search",
        json={"query": "What meals do I prefer?", "user_id": "user-a", "top_k": 100},
    )

    assert response.status_code == 200
    connection = client.app.state.store._connect()
    try:
        audit = connection.execute(
            """
            SELECT user_id_hash, query_hash, requested_top_k, candidate_count,
                   selected_count, candidate_ids, selected_ids, elapsed_ms, index_version
            FROM search_audit
            """
        ).fetchone()
    finally:
        connection.close()

    assert audit is not None
    assert audit["user_id_hash"] != "user-a"
    assert audit["query_hash"] != "What meals do I prefer?"
    assert audit["requested_top_k"] == 100
    assert audit["candidate_count"] >= audit["selected_count"] >= 1
    assert json.loads(audit["candidate_ids"])
    assert json.loads(audit["selected_ids"])
    assert audit["elapsed_ms"] >= 0
    assert audit["index_version"] == "fts5-v3-vectors-v1"


def test_search_succeeds_when_audit_connection_is_locked(client, monkeypatch) -> None:
    assert client.post("/v1/add", json=add_payload()).status_code == 200
    store = client.app.state.store
    original_connect = store._connect

    def connect_with_locked_audit(*, timeout: float = 10):
        if timeout == 0.25:
            raise sqlite3.OperationalError("database is locked")
        return original_connect(timeout=timeout)

    monkeypatch.setattr(store, "_connect", connect_with_locked_audit)

    response = client.post(
        "/v1/search",
        json={"query": "What meals do I prefer?", "user_id": "user-a", "top_k": 100},
    )

    assert response.status_code == 200
    assert response.json()["data"]
