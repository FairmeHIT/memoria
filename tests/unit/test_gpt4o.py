from __future__ import annotations

import json
from unittest.mock import patch

import pytest

from memoria.config import Settings
from memoria.gpt4o import Gpt4oReranker, _parse_chat_scores
from memoria.qwen import create_reranker
from memoria.runtime import create_runtime_store


class FakeResponse:
    def __init__(self, payload: dict) -> None:
        self._payload = json.dumps(payload).encode("utf-8")

    def __enter__(self) -> "FakeResponse":
        return self

    def __exit__(self, *_args: object) -> None:
        return None

    def read(self) -> bytes:
        return self._payload


def test_gpt4o_reranker_calls_chat_completions_and_returns_indexes() -> None:
    response = FakeResponse(
        {
            "model": "gpt-4o-mini",
            "choices": [
                {"message": {"role": "assistant", "content": '{"scores": [0.9, 0.3]}'}}
            ],
            "usage": {"prompt_tokens": 10, "total_tokens": 12},
        }
    )
    with patch("memoria.qwen.urllib.request.urlopen", return_value=response) as urlopen:
        reranker = Gpt4oReranker(
            api_key="x",
            base_url="https://api.openai.com/v1",
            model="gpt-4o-mini",
            timeout_seconds=3,
            retries=0,
        )
        results = reranker.rerank("question", ["first", "second"], top_n=2)

    request = urlopen.call_args.args[0]
    assert request.full_url == "https://api.openai.com/v1/chat/completions"
    assert request.get_header("Authorization") == "Bearer x"
    body = json.loads(request.data)
    assert body["model"] == "gpt-4o-mini"
    assert body["temperature"] == 0.0
    assert body["response_format"] == {"type": "json_object"}
    assert results == ((0, 0.9), (1, 0.3))


def test_gpt4o_reranker_strips_json_fence_and_ignores_unused_fields() -> None:
    scores = _parse_chat_scores(
        {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": "```json\n{\"scores\": [1, 0.5]}\n```",
                    }
                }
            ]
        },
        model="gpt-4o-mini",
        expected_count=2,
    )
    assert scores == [1.0, 0.5]


def test_gpt4o_reranker_batches_many_documents_into_separate_calls() -> None:
    documents = [f"doc-{index}-" + "x" * 9_000 for index in range(3)]

    responses = [
        FakeResponse(
            {
                "choice": [{"message": {"content": '{"scores": [0.9, 0.8]}'}}],
                "choices": [{"message": {"content": '{"scores": [0.9, 0.8]}'}}],
                "usage": {},
            }
        ),
        FakeResponse(
            {
                "choices": [{"message": {"content": '{"scores": [0.7]}'}}],
                "usage": {},
            }
        ),
    ]

    with patch("memoria.qwen.urllib.request.urlopen", side_effect=responses) as urlopen:
        reranker = Gpt4oReranker(
            api_key="x",
            base_url="https://api.openai.com/v1",
            model="gpt-4o-mini",
            timeout_seconds=3,
            retries=0,
        )
        results = reranker.rerank("question", documents, top_n=3)

    assert urlopen.call_count == 2
    assert results == ((0, 0.9), (1, 0.8), (2, 0.7))


def test_gpt4o_reranker_validates_blank_query_and_input() -> None:
    reranker = Gpt4oReranker(
        api_key="x",
        base_url="https://api.openai.com/v1",
        model="gpt-4o-mini",
        timeout_seconds=3,
        retries=0,
    )
    with pytest.raises(ValueError, match="query must not be blank"):
        reranker.rerank("   ", ["document"])
    with pytest.raises(ValueError, match="must be positive"):
        reranker.rerank("query", ["document"], top_n=0)
    assert reranker.rerank("query", [], top_n=2) == ()


def test_gpt4o_backend_requires_its_key_and_base_url(tmp_path) -> None:
    settings = Settings(
        data_dir=tmp_path,
        auth_scheme="none",
        api_key=None,
        retention_days=30,
        max_top_k=100,
        embedding_backend="none",
        reranker_backend="gpt4o",
        gpt4o_api_key="test-openai-key-not-real",
        gpt4o_base_url="https://api.openai.com/v1",
        reranker_model="gpt-4o-mini",
    )

    store = create_runtime_store(settings)

    assert isinstance(store._reranker, Gpt4oReranker)
    assert store._reranker._url == "https://api.openai.com/v1/chat/completions"


def test_gpt4o_factory_wires_from_settings(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("MEMORIA_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("MEMORIA_AUTH_SCHEME", "none")
    monkeypatch.setenv("MEMORIA_RERANKER_BACKEND", "gpt4o")
    monkeypatch.setenv("OPENAI_API_KEY", "test-openai-key-not-real")
    monkeypatch.setenv("MEMORIA_GPT4O_BASE_URL", "https://api.openai.com/v1")
    monkeypatch.setenv("MEMORIA_GPT4O_MODEL", "gpt-4o-mini")

    settings = Settings.from_env()
    reranker = create_reranker(
        backend=settings.reranker_backend,
        api_key=settings.gpt4o_api_key,
        base_url=settings.gpt4o_base_url,
        model=settings.reranker_model,
        timeout_seconds=settings.qwen_timeout_seconds,
        retries=settings.qwen_retries,
        instruct=None,
    )

    assert settings.reranker_model == "gpt-4o-mini"
    assert isinstance(reranker, Gpt4oReranker)
