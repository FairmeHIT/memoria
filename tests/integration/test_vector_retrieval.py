from __future__ import annotations

from collections.abc import Sequence

import pytest
from fastapi.testclient import TestClient

from memoria.app import create_app
from memoria.config import Settings
from memoria.embeddings import EmbeddingUnavailable
from memoria.schemas import AddRequest
from memoria.store import MemoryStore


class StaticEmbedder:
    fingerprint = "test-static-v1"
    dimensions = 2

    def __init__(self, vectors: dict[str, tuple[float, float]]) -> None:
        self._vectors = vectors

    def embed(self, texts: Sequence[str]) -> tuple[tuple[float, ...], ...]:
        return tuple(self._vectors[text] for text in texts)


def _settings(tmp_path) -> Settings:
    return Settings(
        data_dir=tmp_path,
        auth_scheme="none",
        api_key=None,
        retention_days=30,
        max_top_k=100,
        embedding_backend="hashing",
        embedding_dimensions=8,
    )


def _request(*, request_id: str, user_id: str, content: str) -> AddRequest:
    return AddRequest.model_validate(
        {
            "request_id": request_id,
            "user_id": user_id,
            "session_id": "session-1",
            "messages": [{"role": "user", "content": content}],
        }
    )


def test_vector_retrieval_finds_semantic_evidence_without_shared_terms(tmp_path) -> None:
    vectors = {
        "I commute by bicycle every morning.": (1.0, 0.0),
        "The release checklist is in the repository.": (0.0, 1.0),
        "How does the user travel to work?": (1.0, 0.0),
    }
    store = MemoryStore(_settings(tmp_path), embedder=StaticEmbedder(vectors))
    store.initialize()
    store.add(_request(request_id="bike", user_id="user-a", content="I commute by bicycle every morning."))
    store.add(_request(request_id="release", user_id="user-a", content="The release checklist is in the repository."))

    hits = store.search(
        query="How does the user travel to work?", options=None, user_id="user-a", top_k=1
    )

    assert [hit.content for hit in hits] == ["user: I commute by bicycle every morning."]


def test_vector_rows_are_never_recalled_across_users(tmp_path) -> None:
    vectors = {
        "I commute by bicycle every morning.": (1.0, 0.0),
        "How does the user travel to work?": (1.0, 0.0),
    }
    store = MemoryStore(_settings(tmp_path), embedder=StaticEmbedder(vectors))
    store.initialize()
    store.add(_request(request_id="bike", user_id="user-a", content="I commute by bicycle every morning."))

    assert store.search(
        query="How does the user travel to work?", options=None, user_id="user-b", top_k=10
    ) == []


def test_embedding_failure_does_not_commit_partial_add_request(tmp_path) -> None:
    class FailingEmbedder:
        fingerprint = "failing-v1"
        dimensions = 2

        def embed(self, texts: Sequence[str]) -> tuple[tuple[float, ...], ...]:
            raise EmbeddingUnavailable("embedding unavailable")

    store = MemoryStore(_settings(tmp_path), embedder=FailingEmbedder())
    store.initialize()

    with pytest.raises(EmbeddingUnavailable):
        store.add(_request(request_id="failed", user_id="user-a", content="Never persist this."))

    connection = store._connect()
    try:
        assert connection.execute("SELECT COUNT(*) FROM add_requests").fetchone()[0] == 0
        assert connection.execute("SELECT COUNT(*) FROM memory_embeddings").fetchone()[0] == 0
    finally:
        connection.close()


def test_configured_hashing_backend_persists_vectors_through_http(tmp_path) -> None:
    settings = _settings(tmp_path)
    with TestClient(create_app(settings)) as client:
        response = client.post(
            "/v1/add",
            json={
                "request_id": "http-vector",
                "user_id": "user-a",
                "session_id": "session-a",
                "messages": [{"role": "user", "content": "I prefer quiet libraries."}],
            },
        )
        assert response.status_code == 200
        search = client.post(
            "/v1/search",
            json={"query": "Which libraries do I prefer?", "user_id": "user-a", "top_k": 10},
        )

        assert search.status_code == 200
        assert search.json()["data"][0]["content"] == "user: I prefer quiet libraries."
        connection = client.app.state.store._connect()
        try:
            assert connection.execute("SELECT COUNT(*) FROM memory_embeddings").fetchone()[0] == 1
        finally:
            connection.close()


def test_reranker_controls_return_order_and_scores(tmp_path) -> None:
    class ReverseReranker:
        def rerank(
            self, query: str, documents: Sequence[str], *, top_n: int | None = None
        ) -> tuple[tuple[int, float], ...]:
            assert query == "Which item is relevant?"
            assert top_n == 2
            assert list(documents) == ["user: relevant item", "user: another item"]
            return ((1, 0.9), (0, 0.2))

    store = MemoryStore(_settings(tmp_path), reranker=ReverseReranker())
    store.initialize()
    store.add(_request(request_id="first", user_id="user-a", content="relevant item"))
    store.add(_request(request_id="second", user_id="user-a", content="another item"))

    hits = store.search(query="Which item is relevant?", options=None, user_id="user-a", top_k=2)

    assert [hit.content for hit in hits] == ["user: another item", "user: relevant item"]
    assert [hit.score for hit in hits] == [0.9, 0.2]


def test_partial_reranker_response_keeps_scores_in_return_order(tmp_path) -> None:
    class PartialReranker:
        def rerank(
            self, _query: str, _documents: Sequence[str], *, top_n: int | None = None
        ) -> tuple[tuple[int, float], ...]:
            return ((1, 0.9),)

    store = MemoryStore(_settings(tmp_path), reranker=PartialReranker())
    store.initialize()
    store.add(_request(request_id="first", user_id="user-a", content="relevant item"))
    store.add(_request(request_id="second", user_id="user-a", content="another item"))

    hits = store.search(query="Which item is relevant?", options=None, user_id="user-a", top_k=2)

    assert [hit.score for hit in hits] == sorted(
        [hit.score for hit in hits], reverse=True
    )
