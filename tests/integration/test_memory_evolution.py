from __future__ import annotations

from tests.conftest import add_payload


def test_search_uses_options_as_an_auxiliary_retrieval_channel(client) -> None:
    payload = add_payload()
    payload["messages"] = [
        {"role": "user", "content": "I always order soba noodles when I travel."}
    ]
    assert client.post("/v1/add", json=payload).status_code == 200

    response = client.post(
        "/v1/search",
        json={
            "query": "What is my preferred cuisine?",
            "options": ["A. Soba noodles", "B. Tacos"],
            "user_id": "user-a",
            "top_k": 10,
        },
    )

    assert response.status_code == 200
    assert any("soba noodles" in item["content"] for item in response.json()["data"]), (
        f"Expected soba noodles in results, got {[i['content'] for i in response.json()['data']]}"
    )


def test_latest_explicit_preference_correction_supersedes_old_evidence(client) -> None:
    first = add_payload(request_id="preference-1")
    first["messages"] = [
        {
            "role": "user",
            "timestamp": 1_700_000_000_000,
            "content": "I prefer coffee in the morning.",
        }
    ]
    second = add_payload(request_id="preference-2")
    second["messages"] = [
        {
            "role": "user",
            "timestamp": 1_700_000_100_000,
            "content": "I no longer prefer coffee in the morning.",
        }
    ]
    assert client.post("/v1/add", json=first).status_code == 200
    assert client.post("/v1/add", json=second).status_code == 200

    current = client.post(
        "/v1/search",
        json={"query": "What does the user prefer in the morning?", "user_id": "user-a", "top_k": 10},
    )
    historical = client.post(
        "/v1/search",
        json={"query": "What did the user used to prefer in the morning?", "user_id": "user-a", "top_k": 10},
    )

    assert current.status_code == historical.status_code == 200
    current_contents = [item["content"] for item in current.json()["data"]]
    assert any("no longer prefer coffee" in c for c in current_contents), (
        f"Expected 'no longer prefer coffee' in current, got {current_contents}"
    )
    historical_contents = {item["content"] for item in historical.json()["data"]}
    assert any("prefer coffee" in c for c in historical_contents), (
        f"Expected 'prefer coffee' in historical, got {historical_contents}"
    )
    assert any("no longer prefer coffee" in c for c in historical_contents), (
        f"Expected 'no longer prefer coffee' in historical, got {historical_contents}"
    )
