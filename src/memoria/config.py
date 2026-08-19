"""Runtime configuration loaded from explicit environment variables."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv

AuthScheme = Literal["none", "token", "bearer", "x_api_key"]
VALID_AUTH_SCHEMES: frozenset[str] = frozenset({"none", "token", "bearer", "x_api_key"})
EmbeddingBackend = Literal["none", "hashing", "qwen", "bge", "local"]
RerankerBackend = Literal["none", "qwen", "bge", "gpt4o", "local"]
VALID_EMBEDDING_BACKENDS: frozenset[str] = frozenset({"none", "hashing", "qwen", "bge", "local"})
VALID_RERANKER_BACKENDS: frozenset[str] = frozenset({"none", "qwen", "bge", "gpt4o", "local"})


@dataclass(frozen=True, slots=True)
class Settings:
    """Configuration shared by the HTTP handlers and persistence layer."""

    data_dir: Path
    auth_scheme: AuthScheme
    api_key: str | None
    retention_days: int
    max_top_k: int
    embedding_backend: EmbeddingBackend = "none"
    embedding_dimensions: int = 384
    vector_candidate_limit: int = 500
    rrf_k: int = 60
    embedding_model: str = "text-embedding-v4"
    embedding_base_url: str = ""
    reranker_backend: RerankerBackend = "none"
    reranker_model: str = "qwen3-rerank"
    reranker_multilingual_model: str = ""
    reranker_base_url: str = ""
    reranker_candidate_limit: int = 500
    reranker_instruct: str | None = "Given a web search query, retrieve relevant passages that answer the query."
    qwen_api_key: str | None = None
    qwen_timeout_seconds: float = 15.0
    qwen_retries: int = 2
    gpt4o_api_key: str | None = None
    gpt4o_base_url: str = "https://api.openai.com/v1"
    model_api_key: str | None = None
    model_auth_scheme: Literal["bearer", "x_api_key"] = "bearer"

    def __post_init__(self) -> None:
        if self.auth_scheme not in VALID_AUTH_SCHEMES:
            raise ValueError("MEMORIA_AUTH_SCHEME must be none, token, bearer, or x_api_key")
        if self.auth_scheme != "none" and not self.api_key:
            raise ValueError("MEMORIA_API_KEY is required when authentication is enabled")
        if self.retention_days < 1:
            raise ValueError("MEMORIA_RETENTION_DAYS must be at least 1")
        if not 1 <= self.max_top_k <= 1_000:
            raise ValueError("MEMORIA_MAX_TOP_K must be between 1 and 1000")
        if self.embedding_backend not in VALID_EMBEDDING_BACKENDS:
            raise ValueError("MEMORIA_EMBEDDING_BACKEND must be none, hashing, qwen, or bge")
        if self.embedding_dimensions < 8:
            raise ValueError("MEMORIA_EMBEDDING_DIMENSIONS must be at least 8")
        if not 1 <= self.vector_candidate_limit <= 5_000:
            raise ValueError("MEMORIA_VECTOR_CANDIDATE_LIMIT must be between 1 and 5000")
        if not 1 <= self.rrf_k <= 1_000:
            raise ValueError("MEMORIA_RRF_K must be between 1 and 1000")
        if self.reranker_backend not in VALID_RERANKER_BACKENDS:
            raise ValueError("MEMORIA_RERANKER_BACKEND must be none, qwen, bge, gpt4o, or local")
        if self.embedding_backend == "qwen" and (
            not self.qwen_api_key or not self.embedding_base_url.strip() or not self.embedding_model.strip()
        ):
            raise ValueError("Qwen embedding requires API key, model, and base URL")
        if self.reranker_backend == "qwen" and (
            not self.qwen_api_key or not self.reranker_base_url.strip() or not self.reranker_model.strip()
        ):
            raise ValueError("Qwen reranker requires API key, model, and base URL")
        if self.embedding_backend == "bge" and (
            not self.model_api_key or not self.embedding_base_url.strip() or not self.embedding_model.strip()
        ):
            raise ValueError("BGE embedding requires API key, model, and base URL")
        if self.reranker_backend == "bge" and (
            not self.model_api_key or not self.reranker_base_url.strip() or not self.reranker_model.strip()
        ):
            raise ValueError("BGE reranker requires API key, model, and base URL")
        if self.reranker_backend == "gpt4o" and (
            not self.gpt4o_api_key or not self.gpt4o_base_url.strip() or not self.reranker_model.strip()
        ):
            raise ValueError("gpt-4o-mini reranker requires an OpenAI API key, base URL, and model")
        if self.model_auth_scheme not in {"bearer", "x_api_key"}:
            raise ValueError("MEMORIA_MODEL_API_AUTH_SCHEME must be bearer or x_api_key")
        if not 1 <= self.reranker_candidate_limit <= 500:
            raise ValueError("MEMORIA_RERANKER_CANDIDATE_LIMIT must be between 1 and 500")
        if self.qwen_timeout_seconds <= 0:
            raise ValueError("MEMORIA_QWEN_TIMEOUT_SECONDS must be positive")
        if not 0 <= self.qwen_retries <= 5:
            raise ValueError("MEMORIA_QWEN_RETRIES must be between 0 and 5")

    @classmethod
    def from_env(cls) -> "Settings":
        load_dotenv(override=False)
        data_dir = Path(os.environ.get("MEMORIA_DATA_DIR", "./data"))
        auth_scheme = os.environ.get("MEMORIA_AUTH_SCHEME", "bearer").lower()
        embedding_backend = os.environ.get("MEMORIA_EMBEDDING_BACKEND", "none").lower()
        reranker_backend = os.environ.get("MEMORIA_RERANKER_BACKEND", "none").lower()
        bge_base_url = os.environ.get("MEMORIA_BGE_BASE_URL") or os.environ.get(
            "MEMORIA_MODEL_API_BASE_URL", "http://127.0.0.1:8000"
        )
        model_api_key = (
            os.environ.get("MEMORIA_BGE_API_KEY")
            or os.environ.get("MEMORIA_MODEL_API_KEY")
            or os.environ.get("API_KEY")
        )
        return cls(
            data_dir=data_dir,
            auth_scheme=auth_scheme,  # type: ignore[arg-type]
            api_key=os.environ.get("MEMORIA_API_KEY"),
            retention_days=int(os.environ.get("MEMORIA_RETENTION_DAYS", "30")),
            max_top_k=int(os.environ.get("MEMORIA_MAX_TOP_K", "1000")),
            embedding_backend=embedding_backend,  # type: ignore[arg-type]
            embedding_dimensions=int(os.environ.get("MEMORIA_EMBEDDING_DIMENSIONS", "384")),
            vector_candidate_limit=int(os.environ.get("MEMORIA_VECTOR_CANDIDATE_LIMIT", "500")),
            rrf_k=int(os.environ.get("MEMORIA_RRF_K", "60")),
            embedding_model=os.environ.get(
                "MEMORIA_BGE_EMBEDDING_MODEL" if embedding_backend in ("bge", "local") else "MEMORIA_EMBEDDING_MODEL",
                "BAAI/bge-small-en-v1.5" if embedding_backend == "local" else ("BAAI/bge-m3" if embedding_backend == "bge" else "text-embedding-v4"),
            ),
            embedding_base_url=(
                bge_base_url if embedding_backend == "bge" else os.environ.get("MEMORIA_EMBEDDING_BASE_URL", "")
            ),
            reranker_backend=reranker_backend,  # type: ignore[arg-type]
            reranker_model=os.environ.get(
                "MEMORIA_BGE_RERANKER_MODEL"
                if reranker_backend == "bge"
                else ("MEMORIA_GPT4O_MODEL" if reranker_backend == "gpt4o" else "MEMORIA_RERANKER_MODEL"),
                "BAAI/bge-reranker-v2-m3"
                if reranker_backend == "bge"
                else ("gpt-4o-mini" if reranker_backend == "gpt4o" else (
                    "models/bge-reranker-v2-m3" if reranker_backend == "local" else "qwen3-rerank"
                )),
            ),
            reranker_base_url=(
                bge_base_url if reranker_backend == "bge" else os.environ.get("MEMORIA_RERANKER_BASE_URL", "")
            ),
            reranker_multilingual_model=os.environ.get("MEMORIA_RERANKER_MULTILINGUAL_MODEL", ""),
            reranker_candidate_limit=int(os.environ.get("MEMORIA_RERANKER_CANDIDATE_LIMIT", "500")),
            reranker_instruct=os.environ.get(
                "MEMORIA_RERANKER_INSTRUCT",
                "Given a web search query, retrieve relevant passages that answer the query.",
            ),
            qwen_api_key=os.environ.get("MEMORIA_QWEN_API_KEY")
            or os.environ.get("DASHSCOPE_API_KEY"),
            qwen_timeout_seconds=float(os.environ.get("MEMORIA_QWEN_TIMEOUT_SECONDS", "15")),
            qwen_retries=int(os.environ.get("MEMORIA_QWEN_RETRIES", "2")),
            gpt4o_api_key=os.environ.get("MEMORIA_GPT4O_API_KEY")
            or os.environ.get("OPENAI_API_KEY"),
            gpt4o_base_url=os.environ.get("MEMORIA_GPT4O_BASE_URL", "https://api.openai.com/v1"),
            model_api_key=model_api_key,
            model_auth_scheme=(
                os.environ.get("MEMORIA_BGE_AUTH_SCHEME")
                or os.environ.get("MEMORIA_MODEL_API_AUTH_SCHEME", "bearer")
            ).lower(),  # type: ignore[arg-type]
        )
