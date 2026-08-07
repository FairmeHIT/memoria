"""Construction of production-equivalent memoria stores for HTTP and CLI use."""
from __future__ import annotations

from memoria.config import Settings
from memoria.embeddings import create_embedder
from memoria.model_audit import SqliteModelAuditRecorder
from memoria.qwen import create_reranker
from memoria.store import MemoryStore


def create_runtime_store(settings: Settings) -> MemoryStore:
    """Create one configured store with body-free remote-model call auditing."""

    recorder = SqliteModelAuditRecorder(settings.data_dir)
    model_api_key = settings.model_api_key if settings.embedding_backend == "bge" else settings.qwen_api_key
    reranker_api_key = (
        settings.gpt4o_api_key
        if settings.reranker_backend == "gpt4o"
        else (settings.model_api_key if settings.reranker_backend == "bge" else settings.qwen_api_key)
    )
    reranker_base_url = (
        settings.gpt4o_base_url
        if settings.reranker_backend == "gpt4o"
        else settings.reranker_base_url
    )
    store = MemoryStore(
        settings,
        embedder=create_embedder(
            backend=settings.embedding_backend,
            dimensions=settings.embedding_dimensions,
            api_key=model_api_key,
            base_url=settings.embedding_base_url,
            model=settings.embedding_model,
            timeout_seconds=settings.qwen_timeout_seconds,
            retries=settings.qwen_retries,
            recorder=recorder,
            auth_scheme=settings.model_auth_scheme if settings.embedding_backend == "bge" else "bearer",
        ),
        reranker=create_reranker(
            backend=settings.reranker_backend,
            api_key=reranker_api_key,
            base_url=reranker_base_url,
            model=settings.reranker_model,
            timeout_seconds=settings.qwen_timeout_seconds,
            retries=settings.qwen_retries,
            instruct=settings.reranker_instruct,
            recorder=recorder,
            auth_scheme=settings.model_auth_scheme if settings.reranker_backend == "bge" else "bearer",
        ),
    )
    store.initialize()
    return store
