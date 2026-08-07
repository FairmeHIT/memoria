from __future__ import annotations


def test_new_exclusive_location_supersedes_old_location_for_current_search(client) -> None:
    older = {
        "request_id": "location-older",
        "messages": [{"role": "user", "content": "I live in Berlin."}],
        "user_id": "location-user",
        "session_id": "location-session",
    }
    newer = {
        "request_id": "location-newer",
        "messages": [{"role": "user", "content": "I live in Paris."}],
        "user_id": "location-user",
        "session_id": "location-session",
    }
    assert client.post("/v1/add", json=older).status_code == 200
    assert client.post("/v1/add", json=newer).status_code == 200

    current = client.post(
        "/v1/search",
        json={"query": "Where do I live?", "user_id": "location-user", "top_k": 10},
    )
    historical = client.post(
        "/v1/search",
        json={"query": "Where did I live in the past?", "user_id": "location-user", "top_k": 10},
    )

    assert [item["content"] for item in current.json()["data"]] == [
        "user: I live in Paris."
    ]
    assert {item["content"] for item in historical.json()["data"]} == {
        "user: I live in Berlin.",
        "user: I live in Paris.",
    }
