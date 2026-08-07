"""Local embedding interfaces and safe vector serialization."""
from __future__ import annotations

import hashlib
import math
import re
import struct
from collections.abc import Sequence
from typing import Protocol

from memoria.model_audit import ModelCallAudit


TOKEN_RE = re.compile(r"[A-Za-z0-9_]+|[\u4e00-\u9fff]+")


class EmbeddingUnavailable(RuntimeError):
    """An enabled embedding channel could not produce a valid vector."""


class EmbeddingProvider(Protocol):
    """A deterministic local text-to-vector implementation."""

    fingerprint: str
    dimensions: int

    def embed(self, texts: Sequence[str]) -> tuple[tuple[float, ...], ...]: ...


class ModelCallRecorder(Protocol):
    def record(self, audit: ModelCallAudit) -> None: ...


class HashingEmbedder:
    """Offline dense lexical vectors used when no learned model is configured."""

    fingerprint: str

    def __init__(self, *, dimensions: int) -> None:
        if dimensions < 8:
            raise ValueError("embedding dimensions must be at least 8")
        self.dimensions = dimensions
        self.fingerprint = f"hashing-charword-v1-{dimensions}"

    def embed(self, texts: Sequence[str]) -> tuple[tuple[float, ...], ...]:
        return tuple(self._embed_one(text) for text in texts)

    def _embed_one(self, text: str) -> tuple[float, ...]:
        values = [0.0] * self.dimensions
        tokens = [token.casefold() for token in TOKEN_RE.findall(text)]
        features = list(tokens)
        features.extend(f"{left}|{right}" for left, right in zip(tokens, tokens[1:], strict=False))
        for token in tokens:
            padded = f"^{token}$"
            features.extend(padded[index : index + 3] for index in range(max(len(padded) - 2, 0)))
        if not features:
            features.append(f"raw:{text.casefold()}")
        for feature in features:
            digest = hashlib.blake2b(feature.encode("utf-8"), digest_size=8).digest()
            index = int.from_bytes(digest[:4], "little") % self.dimensions
            values[index] += 1.0 if digest[4] & 1 else -1.0
        magnitude = math.sqrt(sum(value * value for value in values))
        if magnitude == 0.0:
            return tuple(values)
        return tuple(value / magnitude for value in values)


def create_embedder(
    *,
    backend: str,
    dimensions: int,
    api_key: str | None = None,
    base_url: str = "",
    model: str = "text-embedding-v4",
    timeout_seconds: float = 15.0,
    retries: int = 2,
    recorder: ModelCallRecorder | None = None,
    auth_scheme: str = "bearer",
) -> EmbeddingProvider | None:
    """Create a configured local, Qwen, or self-hosted BGE provider."""

    if backend == "none":
        return None
    if backend == "hashing":
        return HashingEmbedder(dimensions=dimensions)
    if backend == "qwen":
        from memoria.qwen import QwenEmbeddingProvider

        if api_key is None:
            raise ValueError("Qwen embedding API key is required")
        return QwenEmbeddingProvider(
            api_key=api_key,
            base_url=base_url,
            model=model,
            dimensions=dimensions,
            timeout_seconds=timeout_seconds,
            retries=retries,
            recorder=recorder,
        )
    if backend == "bge":
        from memoria.bge import BgeEmbeddingProvider

        if api_key is None:
            raise ValueError("BGE embedding API key is required")
        return BgeEmbeddingProvider(
            api_key=api_key,
            base_url=base_url,
            model=model,
            dimensions=dimensions,
            timeout_seconds=timeout_seconds,
            retries=retries,
            auth_scheme=auth_scheme,
            recorder=recorder,
        )
    raise ValueError("MEMORIA_EMBEDDING_BACKEND must be none, hashing, qwen, or bge")


def serialize_vector(vector: Sequence[float]) -> bytes:
    """Encode an L2-normalized vector as explicit little-endian float32 values."""

    values = tuple(float(value) for value in vector)
    if not values or any(not math.isfinite(value) for value in values):
        raise EmbeddingUnavailable("embedding provider returned an invalid vector")
    magnitude = math.sqrt(sum(value * value for value in values))
    if magnitude == 0.0:
        raise EmbeddingUnavailable("embedding provider returned a zero vector")
    normalized = tuple(value / magnitude for value in values)
    return struct.pack(f"<{len(normalized)}f", *normalized)


def deserialize_vector(payload: bytes, *, dimensions: int) -> tuple[float, ...]:
    """Decode a stored vector after validating its immutable dimensions."""

    expected_size = dimensions * 4
    if dimensions < 1 or len(payload) != expected_size:
        raise EmbeddingUnavailable("stored embedding has an invalid dimension")
    values = struct.unpack(f"<{dimensions}f", payload)
    if any(not math.isfinite(value) for value in values):
        raise EmbeddingUnavailable("stored embedding contains a non-finite value")
    return tuple(values)
