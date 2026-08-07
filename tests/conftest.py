from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from memoria.app import create_app
from memoria.config import Settings


@pytest.fixture
def client(tmp_path: Path) -> TestClient:
    settings = Settings(
        data_dir=tmp_path,
        auth_scheme="none",
        api_key=None,
        retention_days=30,
        max_top_k=1_000,
    )
    with TestClient(create_app(settings)) as test_client:
        yield test_client


@pytest.fixture
def bearer_client(tmp_path: Path) -> TestClient:
    settings = Settings(
        data_dir=tmp_path,
        auth_scheme="bearer",
        api_key="test-token",
        retention_days=30,
        max_top_k=1_000,
    )
    with TestClient(create_app(settings)) as test_client:
        yield test_client


def add_payload(*, request_id: str = "req-1", user_id: str = "user-a", session_id: str = "session-a") -> dict:
    return {
        "request_id": request_id,
        "messages": [
            {
                "role": "user",
                "timestamp": 1_704_067_200_000,
                "content": "I prefer vegetarian meals when traveling.",
            },
            {
                "role": "assistant",
                "content": "I will remember that preference.",
            },
        ],
        "user_id": user_id,
        "session_id": session_id,
    }
