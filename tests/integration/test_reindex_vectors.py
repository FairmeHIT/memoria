from __future__ import annotations

from collections.abc import Sequence

from memoria.config import Settings
from memoria.schemas import AddRequest
from memoria.store import MemoryStore


class StaticEmbedder:
    dimensions = 2

    def __init__(
        self,
        vectors: dict[str, tuple[float, float]],
        *,
        fingerprint: str = "reindex-test-v1",
    ) -> None:
        self._vectors = vectors
        self.fingerprint = fingerprint

    def embed(self, texts: Sequence[str]) -> tuple[tuple[float, ...], ...]:
        return tuple(self._vectors[text] for text in texts)


def _settings(tmp_path) -> Settings:
    return Settings(
        data_dir=tmp_path,
        auth_scheme="none",
        api_key=None,
        retention_days=30,
        max_top_k=100,
        embedding_dimensions=8,
    )


def test_reindex_backfills_only_missing_current_model_vectors(tmp_path) -> None:
    settings = _settings(tmp_path)
    source_store = MemoryStore(settings)
    source_store.initialize()
    source_store.add(
        AddRequest.model_validate(
            {
                "request_id": "legacy-request",
                "user_id": "user-a",
                "session_id": "session-a",
                "messages": [
                    {"role": "user", "content": "I commute by bicycle."},
                    {"role": "assistant", "content": "Noted."},
                ],
            }
        )
    )
    embedder = StaticEmbedder(
        {
            "I commute by bicycle.": (1.0, 0.0),
            "Noted.": (0.0, 1.0),
            "How does the user travel?": (1.0, 0.0),
        }
    )
    store = MemoryStore(settings, embedder=embedder)
    store.initialize()

    result = store.reindex_embeddings(batch_size=10)

    assert result == {"scanned": 2, "updated": 2}
    assert store.reindex_embeddings(batch_size=10) == {"scanned": 0, "updated": 0}
    assert "I commute by bicycle." in store.search(
        query="How does the user travel?", options=None, user_id="user-a", top_k=1
    )[0].content


def test_reindex_replaces_stale_vectors_after_fingerprint_change(tmp_path) -> None:
    settings = _settings(tmp_path)
    legacy_store = MemoryStore(
        settings,
        embedder=StaticEmbedder(
            {
                "I commute by bicycle.": (1.0, 0.0),
                "Noted.": (0.0, 1.0),
            },
            fingerprint="bge-embedding-v1:BAAI/bge-m3:2",
        ),
    )
    legacy_store.initialize()
    legacy_store.add(
        AddRequest.model_validate(
            {
                "request_id": "legacy-request",
                "user_id": "user-a",
                "session_id": "session-a",
                "messages": [
                    {"role": "user", "content": "I commute by bicycle."},
                    {"role": "assistant", "content": "Noted."},
                ],
            }
        )
    )

    current_store = MemoryStore(
        settings,
        embedder=StaticEmbedder(
            {
                "I commute by bicycle.": (1.0, 0.0),
                "Noted.": (0.0, 1.0),
                "How does the user travel?": (1.0, 0.0),
            },
            fingerprint="bge-embedding-v2:BAAI/bge-m3:2",
        ),
    )
    current_store.initialize()

    result = current_store.reindex_embeddings(batch_size=10)

    assert result == {"scanned": 2, "updated": 2}
    connection = current_store._connect()
    try:
        rows = connection.execute(
            "SELECT DISTINCT model_fingerprint FROM memory_embeddings ORDER BY model_fingerprint"
        ).fetchall()
        assert [row[0] for row in rows] == ["bge-embedding-v2:BAAI/bge-m3:2"]
    finally:
        connection.close()
    assert "I commute by bicycle." in current_store.search(
        query="How does the user travel?", options=None, user_id="user-a", top_k=1
    )[0].content
