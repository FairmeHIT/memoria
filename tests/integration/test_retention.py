from __future__ import annotations

from datetime import UTC, datetime, timedelta

from tests.conftest import add_payload


def test_retention_cleanup_removes_expired_evaluation_data(client) -> None:
    client.post("/v1/add", json=add_payload())

    deleted = client.app.state.store.cleanup_expired(
        now=datetime.now(UTC) + timedelta(days=31)
    )

    assert deleted == 1
    response = client.post(
        "/v1/search",
        json={"query": "vegetarian", "user_id": "user-a", "top_k": 100},
    )
    assert response.status_code == 200
    assert response.json() == {"data": []}


def test_retention_cleanup_removes_derived_claims_and_supersessions(client) -> None:
    first = add_payload(request_id="expiry-claim-1")
    first["messages"] = [{"role": "user", "content": "I prefer coffee."}]
    second = add_payload(request_id="expiry-claim-2")
    second["messages"] = [{"role": "user", "content": "I no longer prefer coffee."}]
    assert client.post("/v1/add", json=first).status_code == 200
    assert client.post("/v1/add", json=second).status_code == 200
    assert client.post(
        "/v1/search",
        json={"query": "coffee", "user_id": "user-a", "top_k": 10},
    ).status_code == 200

    client.app.state.store.cleanup_expired(now=datetime.now(UTC) + timedelta(days=31))

    connection = client.app.state.store._connect()
    try:
        assert connection.execute("SELECT COUNT(*) FROM memory_claims").fetchone()[0] == 0
        assert connection.execute("SELECT COUNT(*) FROM memory_supersessions").fetchone()[0] == 0
        assert connection.execute("SELECT COUNT(*) FROM search_audit").fetchone()[0] == 0
    finally:
        connection.close()
