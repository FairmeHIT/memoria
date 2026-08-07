from __future__ import annotations

import json
import urllib.error
from dataclasses import dataclass, field
from unittest.mock import patch

import pytest

from memoria.embeddings import EmbeddingUnavailable, create_embedder
from memoria.config import Settings
from memoria.qwen import QwenEmbeddingProvider, QwenReranker, create_reranker


class FakeResponse:
    def __init__(self, payload: dict) -> None:
        self._payload = json.dumps(payload).encode("utf-8")

    def __enter__(self) -> "FakeResponse":
        return self

    def __exit__(self, *_args: object) -> None:
        return None

    def read(self) -> bytes:
        return self._payload


@dataclass
class RecordingSink:
    calls: list = field(default_factory=list)

    def record(self, audit) -> None:
        self.calls.append(audit)


def test_qwen_embedding_batches_and_sends_dimensions() -> None:
    response = FakeResponse(
        {
            "data": [
                {"index": 1, "embedding": [0.0, 1.0]},
                {"index": 0, "embedding": [1.0, 0.0]},
            ],
            "usage": {"prompt_tokens": 12, "total_tokens": 12},
        }
    )
    recorder = RecordingSink()
    with patch("memoria.qwen.urllib.request.urlopen", return_value=response) as urlopen:
        provider = QwenEmbeddingProvider(
            api_key="x",
            base_url="https://workspace.example/compatible-mode/v1",
            model="text-embedding-v4",
            dimensions=2,
            timeout_seconds=3,
            retries=0,
            batch_size=10,
            recorder=recorder,
        )
        vectors = provider.embed(["first", "second"])

    assert vectors == ((1.0, 0.0), (0.0, 1.0))
    request = urlopen.call_args.args[0]
    assert request.full_url.endswith("/embeddings")
    body = json.loads(request.data)
    assert body["model"] == "text-embedding-v4"
    assert body["dimensions"] == 2
    assert body["encoding_format"] == "float"
    assert recorder.calls[0].input_count == 2
    assert recorder.calls[0].prompt_tokens == 12
    assert recorder.calls[0].success is True


def test_qwen_reranker_returns_valid_index_and_score_order() -> None:
    response = FakeResponse(
        {
            "results": [
                {"index": 1, "relevance_score": 0.9},
                {"index": 0, "relevance_score": 0.2},
            ]
        }
    )
    with patch("memoria.qwen.urllib.request.urlopen", return_value=response) as urlopen:
        reranker = QwenReranker(
            api_key="x",
            base_url="https://workspace.example/compatible-api/v1",
            model="qwen3-rerank",
            timeout_seconds=3,
            retries=0,
            instruct="Retrieve relevant passages.",
        )
        results = reranker.rerank("question", ["first", "second"])

    assert results == ((1, 0.9), (0, 0.2))
    request = urlopen.call_args.args[0]
    assert request.full_url.endswith("/reranks")
    assert json.loads(request.data)["instruct"] == "Retrieve relevant passages."


def test_qwen_provider_rejects_malformed_embedding_response() -> None:
    with patch(
        "memoria.qwen.urllib.request.urlopen",
        return_value=FakeResponse({"data": [{"index": 0, "embedding": [1.0]}]}),
    ):
        provider = QwenEmbeddingProvider(
            api_key="x",
            base_url="https://workspace.example/compatible-mode/v1",
            model="text-embedding-v4",
            dimensions=2,
            timeout_seconds=3,
            retries=0,
            batch_size=10,
        )
        with pytest.raises(EmbeddingUnavailable):
            provider.embed(["first"])


def test_qwen_backends_require_a_secret_and_their_own_base_urls(tmp_path) -> None:
    with pytest.raises(ValueError, match="Qwen embedding requires"):
        Settings(
            data_dir=tmp_path,
            auth_scheme="none",
            api_key=None,
            retention_days=30,
            max_top_k=100,
            embedding_backend="qwen",
        )

    settings = Settings(
        data_dir=tmp_path,
        auth_scheme="none",
        api_key=None,
        retention_days=30,
        max_top_k=100,
        embedding_backend="qwen",
        embedding_dimensions=1024,
        embedding_base_url="https://example/compatible-mode/v1",
        reranker_backend="qwen",
        reranker_base_url="https://example/compatible-api/v1",
        qwen_api_key="test-token",
    )

    assert settings.reranker_model == "qwen3-rerank"


def test_qwen_factories_create_configured_providers_without_a_network_call() -> None:
    test_key = "x"
    embedder = create_embedder(
        backend="qwen",
        dimensions=1024,
        api_key=test_key,
        base_url="https://workspace.example/compatible-mode/v1",
        model="text-embedding-v4",
    )
    reranker = create_reranker(
        backend="qwen",
        api_key=test_key,
        base_url="https://workspace.example/compatible-api/v1",
        model="qwen3-rerank",
        timeout_seconds=3,
        retries=0,
        instruct=None,
    )

    assert embedder is not None and embedder.dimensions == 1024
    assert reranker is not None


def test_qwen_provider_hides_transport_failures() -> None:
    provider = QwenEmbeddingProvider(
        api_key="x",
        base_url="https://workspace.example/compatible-mode/v1",
        model="text-embedding-v4",
        dimensions=2,
        timeout_seconds=3,
        retries=0,
        batch_size=10,
    )
    with patch("memoria.qwen.urllib.request.urlopen", side_effect=urllib.error.URLError("offline")):
        with pytest.raises(EmbeddingUnavailable, match="Qwen API request failed"):
            provider.embed(["first"])


def test_qwen_provider_does_not_retry_forbidden_responses() -> None:
    provider = QwenEmbeddingProvider(
        api_key="x",
        base_url="https://workspace.example/compatible-mode/v1",
        model="text-embedding-v4",
        dimensions=2,
        timeout_seconds=3,
        retries=2,
        batch_size=10,
    )
    forbidden = urllib.error.HTTPError(
        "https://workspace.example/compatible-mode/v1/embeddings",
        403,
        "Forbidden",
        hdrs=None,
        fp=None,
    )
    with patch("memoria.qwen.urllib.request.urlopen", side_effect=forbidden) as urlopen:
        with pytest.raises(EmbeddingUnavailable, match="Qwen API request failed"):
            provider.embed(["first"])

    assert urlopen.call_count == 1
