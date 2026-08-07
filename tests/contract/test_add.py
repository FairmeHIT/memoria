from __future__ import annotations

from tests.conftest import add_payload


def test_add_echoes_contract_fields_and_makes_memory_immediately_searchable(client) -> None:
    payload = add_payload()

    add_response = client.post("/v1/add", json=payload)

    assert add_response.status_code == 200
    assert add_response.json() == {
        "success": True,
        "request_id": "req-1",
        "user_id": "user-a",
        "session_id": "session-a",
    }

    search_response = client.post(
        "/v1/search",
        json={"query": "What meals do I prefer?", "user_id": "user-a", "top_k": 100},
    )

    assert search_response.status_code == 200
    data = search_response.json()["data"]
    assert data
    assert "vegetarian meals" in data[0]["content"]
    assert data[0]["id"]


def test_add_rejects_unknown_contract_fields(client) -> None:
    payload = add_payload()
    payload["async_mode"] = True

    response = client.post("/v1/add", json=payload)

    assert response.status_code == 422
    assert response.json() == {"detail": {"reason": "invalid request"}}


def test_add_is_idempotent_and_rejects_conflicting_request_id(client) -> None:
    payload = add_payload()

    first = client.post("/v1/add", json=payload)
    repeated = client.post("/v1/add", json=payload)

    assert first.status_code == repeated.status_code == 200
    assert first.json() == repeated.json()

    data = client.post(
        "/v1/search",
        json={"query": "vegetarian", "user_id": "user-a", "top_k": 100},
    ).json()["data"]
    assert len(data) == 1

    conflict_payload = add_payload()
    conflict_payload["messages"][0]["content"] = "I prefer seafood."
    conflict = client.post("/v1/add", json=conflict_payload)

    assert conflict.status_code == 409
    assert conflict.json() == {"detail": {"reason": "request_id already exists with different content"}}

