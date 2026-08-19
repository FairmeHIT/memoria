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

    # 当前搜索：只返回最新的（巴黎），上下文扩展会包含柏林作为邻接
    current_contents = [item["content"] for item in current.json()["data"]]
    assert any("Paris" in c for c in current_contents), f"Expected Paris in current, got {current_contents}"
    # 历史搜索：返回新旧两条（柏林和巴黎）
    historical_contents = {item["content"] for item in historical.json()["data"]}
    assert any("Berlin" in c for c in historical_contents), f"Expected Berlin in historical, got {historical_contents}"
    assert any("Paris" in c for c in historical_contents), f"Expected Paris in historical, got {historical_contents}"
