from __future__ import annotations

from tests.conftest import add_payload


def test_current_query_prefers_newer_evidence_when_relevance_is_tied(client) -> None:
    older = add_payload(request_id="home-older")
    older["messages"] = [
        {"role": "user", "timestamp": 1_700_000_000_000, "content": "I live in Berlin."}
    ]
    newer = add_payload(request_id="home-newer")
    newer["messages"] = [
        {"role": "user", "timestamp": 1_730_000_000_000, "content": "I live in Paris."}
    ]
    assert client.post("/v1/add", json=older).status_code == 200
    assert client.post("/v1/add", json=newer).status_code == 200

    response = client.post(
        "/v1/search",
        json={"query": "Where do I currently live?", "user_id": "user-a", "top_k": 2},
    )

    assert response.status_code == 200
    first_content = response.json()["data"][0]["content"]
    assert "Paris" in first_content, f"Expected Paris in first result, got: {first_content}"
