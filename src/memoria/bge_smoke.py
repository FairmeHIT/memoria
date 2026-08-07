"""Opt-in smoke checks for the self-hosted BGE API."""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections.abc import Callable, Mapping, Sequence
from typing import Any, Protocol

from memoria.bge import BgeEmbeddingProvider, BgeReranker
from memoria.config import Settings


_SMOKE_ENV = "MEMORIA_BGE_SMOKE"
_LONG_EMBEDDING_TEXT = "e" * 8_250
_LONG_RERANK_QUERY = "q" * 600
_LONG_RERANK_DOCUMENT = "d" * 900


class _Embedder(Protocol):
    fingerprint: str
    dimensions: int

    def embed(self, texts: Sequence[str]) -> tuple[tuple[float, ...], ...]: ...


class _Reranker(Protocol):
    def rerank(
        self, query: str, documents: Sequence[str], *, top_n: int | None = None
    ) -> tuple[tuple[int, float], ...]: ...


def run_bge_smoke(
    settings: Settings,
    *,
    embedder_factory: Callable[..., _Embedder] = BgeEmbeddingProvider,
    reranker_factory: Callable[..., _Reranker] = BgeReranker,
) -> dict[str, Any]:
    """Exercise client-side BGE chunking against an opt-in remote service."""

    if settings.embedding_backend != "bge" or settings.reranker_backend != "bge":
        raise ValueError("BGE smoke requires BGE embedding and reranker backends")
    if settings.model_api_key is None:
        raise ValueError("BGE smoke requires a configured model API key")

    embedder = embedder_factory(
        api_key=settings.model_api_key,
        base_url=settings.embedding_base_url,
        model=settings.embedding_model,
        dimensions=settings.embedding_dimensions,
        timeout_seconds=settings.qwen_timeout_seconds,
        retries=settings.qwen_retries,
        auth_scheme=settings.model_auth_scheme,
    )
    embedding = embedder.embed((_LONG_EMBEDDING_TEXT,))
    if len(embedding) != 1 or len(embedding[0]) != embedder.dimensions:
        raise RuntimeError("BGE embedding smoke returned an unexpected vector shape")

    reranker = reranker_factory(
        api_key=settings.model_api_key,
        base_url=settings.reranker_base_url,
        model=settings.reranker_model,
        timeout_seconds=settings.qwen_timeout_seconds,
        retries=settings.qwen_retries,
        auth_scheme=settings.model_auth_scheme,
    )
    rerank_results = reranker.rerank(
        _LONG_RERANK_QUERY,
        (_LONG_RERANK_DOCUMENT, "short candidate"),
        top_n=2,
    )
    if not rerank_results:
        raise RuntimeError("BGE rerank smoke returned no results")

    return {
        "embedding_dimensions": embedder.dimensions,
        "embedding_fingerprint": embedder.fingerprint,
        "rerank_results": len(rerank_results),
        "status": "passed",
    }


def main(argv: Sequence[str] | None = None, environ: Mapping[str, str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run opt-in BGE input-limit smoke checks")
    parser.add_argument("--run", action="store_true", help="perform remote BGE API calls")
    args = parser.parse_args(argv)
    environment = os.environ if environ is None else environ

    if not args.run and environment.get(_SMOKE_ENV) != "1":
        print(json.dumps({"status": "skipped", "reason": f"set {_SMOKE_ENV}=1 or pass --run"}))
        return 0

    try:
        result = run_bge_smoke(Settings.from_env())
    except Exception as error:
        print(json.dumps({"status": "failed", "error": type(error).__name__}), file=sys.stderr)
        return 1
    print(json.dumps(result, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
