from __future__ import annotations


def test_search_matches_common_inflection_variants_without_external_models(client) -> None:
    payload = {
        "request_id": "morphology-1",
        "messages": [
            {
                "role": "user",
                "content": "Researchers analyze datasets for science.",
            }
        ],
        "user_id": "morphology-user",
        "session_id": "morphology-session",
    }
    assert client.post("/v1/add", json=payload).status_code == 200

    response = client.post(
        "/v1/search",
        json={
            "query": "researcher analyzing dataset",
            "user_id": "morphology-user",
            "top_k": 10,
        },
    )

    assert response.status_code == 200
    assert "Researchers analyze datasets for science." in response.json()["data"][0]["content"]
