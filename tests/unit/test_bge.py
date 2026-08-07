from __future__ import annotations

import json
from unittest.mock import patch

import pytest

from memoria.bge import BgeEmbeddingProvider, BgeReranker, _endpoint
from memoria.config import Settings
from memoria.embeddings import create_embedder
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


def test_bge_embedding_uses_self_hosted_contract_and_x_api_key() -> None:
    response = FakeResponse(
        {
            "model": "BAAI/bge-m3",
            "data": [
                {"index": 0, "embedding": [3.0, 4.0]},
                {"index": 1, "embedding": [4.0, 3.0]},
            ],
            "usage": {"prompt_tokens": 4, "total_tokens": 4},
        }
    )
    with patch("memoria.qwen.urllib.request.urlopen", return_value=response) as urlopen:
        provider = BgeEmbeddingProvider(
            api_key="x",
            base_url="http://model.local:8000",
            model="BAAI/bge-m3",
            dimensions=2,
            timeout_seconds=3,
            retries=0,
            auth_scheme="x_api_key",
        )
        vectors = provider.embed(["hello", "world"])

    assert provider.fingerprint == "bge-embedding-v2:BAAI/bge-m3:2"
    request = urlopen.call_args.args[0]
    assert request.full_url == "http://model.local:8000/v1/embeddings"
    assert request.get_header("X-api-key") == "x"
    assert request.get_header("Authorization") is None
    body = json.loads(request.data)
    assert body == {"input": ["hello", "world"], "model": "BAAI/bge-m3", "encoding_format": "float"}
    assert vectors[0] == (3.0, 4.0)


def test_bge_embedding_chunks_long_text_and_averages_chunk_vectors() -> None:
    long_text = "a" * 8_250
    responses = [
        FakeResponse(
            {
                "model": "BAAI/bge-m3",
                "data": [{"index": 0, "embedding": [1.0, 0.0]}],
                "usage": {"prompt_tokens": 8_192, "total_tokens": 8_192},
            }
        ),
        FakeResponse(
            {
                "model": "BAAI/bge-m3",
                "data": [{"index": 0, "embedding": [0.0, 1.0]}],
                "usage": {"prompt_tokens": 66, "total_tokens": 66},
            }
        ),
    ]
    with patch("memoria.qwen.urllib.request.urlopen", side_effect=responses) as urlopen:
        provider = BgeEmbeddingProvider(
            api_key="x",
            base_url="http://model.local:8000",
            model="BAAI/bge-m3",
            dimensions=2,
            timeout_seconds=3,
            retries=0,
        )
        vectors = provider.embed([long_text])

    assert urlopen.call_count == 2
    payloads = [json.loads(call.args[0].data) for call in urlopen.call_args_list]
    assert all(len(body["input"]) == 1 for body in payloads)
    assert all(
        sum(len(chunk.encode("utf-8")) + 8 for chunk in body["input"]) <= 8_192
        for body in payloads
    )
    assert "".join(chunk for body in payloads for chunk in body["input"]) == long_text
    assert vectors == ((pytest.approx(8_184 / 8_250), pytest.approx(66 / 8_250)),)


def test_bge_reranker_uses_v1_endpoint_and_returns_indexes() -> None:
    response = FakeResponse(
        {
            "model": "BAAI/bge-reranker-v2-m3",
            "results": [
                {"index": 1, "relevance_score": 0.3, "document": None},
                {"index": 0, "relevance_score": 0.9, "document": None},
            ],
        }
    )
    with patch("memoria.qwen.urllib.request.urlopen", return_value=response) as urlopen:
        reranker = BgeReranker(
            api_key="x",
            base_url="http://model.local:8000",
            model="BAAI/bge-reranker-v2-m3",
            timeout_seconds=3,
            retries=0,
        )
        results = reranker.rerank("question", ["first", "second"], top_n=2)

    request = urlopen.call_args.args[0]
    assert request.full_url == "http://model.local:8000/v1/rerank"
    assert json.loads(request.data)["return_documents"] is False
    assert results == ((0, 0.9), (1, 0.3))


def test_bge_reranker_chunks_long_queries_and_documents() -> None:
    long_query = "问" * 100
    long_document = "文" * 600

    def response_for_request(request, **_kwargs) -> FakeResponse:
        body = json.loads(request.data)
        scores = (
            [0.10, 0.80, 0.20, 0.90, 0.40, 0.30, 0.20, 0.60, 0.40]
            if len(body["query"].encode("utf-8")) > 100
            else [0.20, 0.10, 0.10, 0.20, 0.10]
        )
        return FakeResponse(
            {
                "model": "BAAI/bge-reranker-v2-m3",
                "results": [
                    {"index": index, "relevance_score": score, "document": None}
                    for index, score in enumerate(scores)
                ],
            }
        )

    with patch("memoria.qwen.urllib.request.urlopen", side_effect=response_for_request) as urlopen:
        reranker = BgeReranker(
            api_key="x",
            base_url="http://model.local:8000",
            model="BAAI/bge-reranker-v2-m3",
            timeout_seconds=3,
            retries=0,
        )
        results = reranker.rerank(long_query, [long_document, "short"], top_n=2)

    assert urlopen.call_count == 2
    payloads = [json.loads(call.args[0].data) for call in urlopen.call_args_list]
    assert "".join(body["query"] for body in payloads) == long_query
    assert [len(body["documents"]) for body in payloads] == [9, 5]
    for body in payloads:
        assert all(
            len(body["query"].encode("utf-8")) + len(chunk.encode("utf-8")) <= 504
            for chunk in body["documents"]
        )
        assert "".join(body["documents"][:-1]) == long_document
        assert body["top_n"] == len(body["documents"])
    assert results == ((0, 0.9), (1, 0.4))


def test_bge_reranker_batches_expanded_documents_within_api_limit() -> None:
    query = "query"
    long_document = "x" * (504 * 501)

    def response_for_request(request, **_kwargs) -> FakeResponse:
        body = json.loads(request.data)
        return FakeResponse(
            {
                "model": "BAAI/bge-reranker-v2-m3",
                "results": [
                    {"index": index, "relevance_score": float(index), "document": None}
                    for index in range(len(body["documents"]))
                ],
            }
        )

    with patch("memoria.qwen.urllib.request.urlopen", side_effect=response_for_request) as urlopen:
        reranker = BgeReranker(
            api_key="x",
            base_url="http://model.local:8000",
            model="BAAI/bge-reranker-v2-m3",
            timeout_seconds=3,
            retries=0,
        )
        results = reranker.rerank(query, [long_document])

    assert urlopen.call_count == 2
    for call in urlopen.call_args_list:
        body = json.loads(call.args[0].data)
        assert len(body["documents"]) <= 500
        assert all(
            len(body["query"].encode("utf-8")) + len(chunk.encode("utf-8")) <= 504
            for chunk in body["documents"]
        )
    assert results[0][0] == 0


def test_bge_backend_requires_its_key_and_base_url(tmp_path) -> None:
    settings = Settings(
        data_dir=tmp_path,
        auth_scheme="none",
        api_key=None,
        retention_days=30,
        max_top_k=100,
        embedding_backend="bge",
        reranker_backend="bge",
        embedding_dimensions=1024,
        embedding_model="BAAI/bge-m3",
        embedding_base_url="http://model.local:8000",
        reranker_model="BAAI/bge-reranker-v2-m3",
        reranker_base_url="http://model.local:8000",
        model_api_key="test-token",
        model_auth_scheme="x_api_key",
    )

    assert settings.embedding_backend == "bge"
    assert settings.model_auth_scheme == "x_api_key"


def test_bge_provider_rejects_incomplete_or_blank_input() -> None:
    with pytest.raises(ValueError, match="configuration is incomplete"):
        BgeEmbeddingProvider(
            api_key="",
            base_url="http://model.local:8000",
            model="BAAI/bge-m3",
            dimensions=2,
            timeout_seconds=3,
            retries=0,
        )

    provider = BgeEmbeddingProvider(
        api_key="x",
        base_url="http://model.local:8000",
        model="BAAI/bge-m3",
        dimensions=2,
        timeout_seconds=3,
        retries=0,
    )
    with pytest.raises(ValueError, match="blank text"):
        provider.embed(["   "])


def test_bge_reranker_validates_limits_and_v1_base_url() -> None:
    reranker = BgeReranker(
        api_key="x",
        base_url="http://model.local:8000/v1",
        model="BAAI/bge-reranker-v2-m3",
        timeout_seconds=3,
        retries=0,
    )

    with pytest.raises(ValueError, match="not be blank"):
        reranker.rerank(" ", ["document"])
    with pytest.raises(ValueError, match="must be positive"):
        reranker.rerank("query", ["document"], top_n=0)
    assert _endpoint("http://model.local:8000/v1", "/v1/rerank") == "http://model.local:8000/v1/rerank"


def test_bge_settings_load_from_environment_and_factories(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("MEMORIA_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("MEMORIA_AUTH_SCHEME", "none")
    monkeypatch.setenv("MEMORIA_EMBEDDING_BACKEND", "bge")
    monkeypatch.setenv("MEMORIA_RERANKER_BACKEND", "bge")
    monkeypatch.setenv("API_KEY", "test-token")
    monkeypatch.setenv("MEMORIA_BGE_API_KEY", "test-token")
    monkeypatch.setenv("MEMORIA_BGE_BASE_URL", "http://model.local:8000")
    monkeypatch.setenv("MEMORIA_BGE_AUTH_SCHEME", "x_api_key")

    settings = Settings.from_env()
    embedder = create_embedder(
        backend=settings.embedding_backend,
        dimensions=settings.embedding_dimensions,
        api_key=settings.model_api_key,
        base_url=settings.embedding_base_url,
        model=settings.embedding_model,
        auth_scheme=settings.model_auth_scheme,
    )
    reranker = create_reranker(
        backend=settings.reranker_backend,
        api_key=settings.model_api_key,
        base_url=settings.reranker_base_url,
        model=settings.reranker_model,
        timeout_seconds=settings.qwen_timeout_seconds,
        retries=settings.qwen_retries,
        instruct=None,
        auth_scheme=settings.model_auth_scheme,
    )

    assert settings.embedding_model == "BAAI/bge-m3"
    assert isinstance(embedder, BgeEmbeddingProvider)
    assert isinstance(reranker, BgeReranker)

    store = create_runtime_store(settings)
    assert isinstance(store._embedder, BgeEmbeddingProvider)
    assert store._embedder._api_key == "test-token"
