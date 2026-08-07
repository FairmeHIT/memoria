from __future__ import annotations

import json
from collections.abc import Sequence

import pytest

from memoria.bge_smoke import main, run_bge_smoke
from memoria.config import Settings


class FakeEmbedder:
    fingerprint = "bge-embedding-v2:BAAI/bge-m3:8"
    dimensions = 8

    def embed(self, texts: Sequence[str]) -> tuple[tuple[float, ...], ...]:
        assert len(texts) == 1
        assert len(texts[0].encode("utf-8")) > 8_192
        return ((1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),)


class FakeReranker:
    def rerank(
        self, query: str, documents: Sequence[str], *, top_n: int | None = None
    ) -> tuple[tuple[int, float], ...]:
        assert len(query.encode("utf-8")) > 512
        assert any(len(document.encode("utf-8")) > 512 for document in documents)
        assert top_n == 2
        return ((0, 0.9), (1, 0.1))


def _settings(tmp_path) -> Settings:
    return Settings(
        data_dir=tmp_path,
        auth_scheme="none",
        api_key=None,
        retention_days=30,
        max_top_k=100,
        embedding_backend="bge",
        reranker_backend="bge",
        embedding_dimensions=8,
        embedding_model="BAAI/bge-m3",
        embedding_base_url="http://model.local:8000",
        reranker_model="BAAI/bge-reranker-v2-m3",
        reranker_base_url="http://model.local:8000",
        model_api_key="test-token",
        model_auth_scheme="x_api_key",
    )


def test_bge_smoke_runs_long_inputs_and_returns_body_free_summary(tmp_path) -> None:
    result = run_bge_smoke(
        _settings(tmp_path),
        embedder_factory=lambda **_kwargs: FakeEmbedder(),
        reranker_factory=lambda **_kwargs: FakeReranker(),
    )

    assert result == {
        "embedding_dimensions": 8,
        "embedding_fingerprint": "bge-embedding-v2:BAAI/bge-m3:8",
        "rerank_results": 2,
        "status": "passed",
    }
    assert "test-token" not in json.dumps(result)


def test_bge_smoke_requires_bge_backends(tmp_path) -> None:
    settings = Settings(
        data_dir=tmp_path,
        auth_scheme="none",
        api_key=None,
        retention_days=30,
        max_top_k=100,
    )

    with pytest.raises(ValueError, match="requires BGE embedding and reranker backends"):
        run_bge_smoke(settings)


def test_bge_smoke_cli_skips_without_explicit_opt_in(capsys, monkeypatch) -> None:
    monkeypatch.delenv("MEMORIA_BGE_SMOKE", raising=False)

    assert main([]) == 0

    output = json.loads(capsys.readouterr().out)
    assert output["status"] == "skipped"
