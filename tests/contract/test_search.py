from __future__ import annotations

from tests.conftest import add_payload


def test_search_isolated_by_user_id_and_respects_top_k(client) -> None:
    client.post("/v1/add", json=add_payload(request_id="a-1", user_id="user-a"))
    client.post("/v1/add", json=add_payload(request_id="b-1", user_id="user-b"))

    isolated = client.post(
        "/v1/search",
        json={"query": "vegetarian", "user_id": "user-b", "top_k": 1},
    )

    assert isolated.status_code == 200
    data = isolated.json()["data"]
    assert len(data) == 1
    assert "vegetarian" in data[0]["content"]

    missing_namespace = client.post(
        "/v1/search",
        json={"query": "vegetarian", "user_id": "unknown-user", "top_k": 100},
    )
    assert missing_namespace.status_code == 200
    assert missing_namespace.json() == {"data": []}


def test_search_rejects_extra_fields_and_returns_safe_validation_error(client) -> None:
    response = client.post(
        "/v1/search",
        json={
            "query": "vegetarian",
            "user_id": "user-a",
            "top_k": 100,
            "gold_answer": "A",
        },
    )

    assert response.status_code == 422
    assert response.json() == {"detail": {"reason": "invalid request"}}

