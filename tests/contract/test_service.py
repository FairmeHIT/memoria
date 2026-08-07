from __future__ import annotations


def test_health_reports_ready(client) -> None:
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_bearer_auth_does_not_expose_the_configured_secret(bearer_client) -> None:
    unauthenticated = bearer_client.get("/health")
    assert unauthenticated.status_code == 200

    missing = bearer_client.post(
        "/v1/search",
        json={"query": "memory", "user_id": "user-a", "top_k": 1},
    )
    assert missing.status_code == 401
    assert "test-token" not in missing.text

    authorized = bearer_client.post(
        "/v1/search",
        headers={"Authorization": "Bearer test-token"},
        json={"query": "memory", "user_id": "user-a", "top_k": 1},
    )
    assert authorized.status_code == 200
