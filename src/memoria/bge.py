"""HTTP clients for the self-hosted BGE embedding and rerank API."""
from __future__ import annotations

import math
from collections.abc import Sequence

from memoria.qwen import (
    ModelCallRecorder,
    _call_qwen,
    _parse_embeddings,
    _parse_rerank,
)


# UTF-8 byte counts conservatively bound BGE's byte-fallback tokenization.
# Reserve room for the model's pair and sequence special tokens.
_BGE_EMBEDDING_TOKEN_LIMIT = 8_192
_BGE_RERANK_PAIR_TOKEN_LIMIT = 512
_BGE_SPECIAL_TOKEN_RESERVE = 8
_BGE_EMBEDDING_CONTENT_BYTE_BUDGET = _BGE_EMBEDDING_TOKEN_LIMIT - _BGE_SPECIAL_TOKEN_RESERVE
_BGE_RERANK_PAIR_CONTENT_BYTE_BUDGET = _BGE_RERANK_PAIR_TOKEN_LIMIT - _BGE_SPECIAL_TOKEN_RESERVE
_BGE_RERANK_QUERY_BYTE_BUDGET = 256
_BGE_EMBEDDING_BATCH_SIZE = 32
_BGE_RERANK_BATCH_SIZE = 500


class BgeEmbeddingProvider:
    """Call ``POST /v1/embeddings`` on the self-hosted BGE service."""

    def __init__(
        self,
        *,
        api_key: str,
        base_url: str,
        model: str,
        dimensions: int,
        timeout_seconds: float,
        retries: int,
        auth_scheme: str = "bearer",
        recorder: ModelCallRecorder | None = None,
    ) -> None:
        if not api_key.strip() or not base_url.strip() or not model.strip():
            raise ValueError("BGE embedding configuration is incomplete")
        if dimensions < 1:
            raise ValueError("BGE embedding dimensions must be positive")
        if auth_scheme not in {"bearer", "x_api_key"}:
            raise ValueError("BGE API auth scheme must be bearer or x_api_key")
        self.fingerprint = f"bge-embedding-v2:{model}:{dimensions}"
        self.dimensions = dimensions
        self._api_key = api_key
        self._url = _endpoint(base_url, "/v1/embeddings")
        self._model = model
        self._timeout = timeout_seconds
        self._retries = retries
        self._auth_scheme = auth_scheme
        self._recorder = recorder

    def embed(self, texts: Sequence[str]) -> tuple[tuple[float, ...], ...]:
        if any(not text.strip() for text in texts):
            raise ValueError("BGE embedding input must not contain blank text")
        chunked_inputs = tuple(
            (source_index, chunk, _utf8_size(chunk))
            for source_index, text in enumerate(texts)
            for chunk in _split_utf8(text, _BGE_EMBEDDING_CONTENT_BYTE_BUDGET)
        )
        vectors_by_input: list[list[tuple[tuple[float, ...], int]]] = [
            [] for _ in texts
        ]
        for batch in _embedding_batches(chunked_inputs):
            batch_texts = [chunk for _, chunk, _ in batch]
            body = _call_qwen(
                url=self._url,
                api_key=self._api_key,
                payload={"input": batch_texts, "model": self._model, "encoding_format": "float"},
                timeout_seconds=self._timeout,
                retries=self._retries,
                operation="embedding",
                provider="bge",
                model=self._model,
                input_count=len(batch_texts),
                recorder=self._recorder,
                auth_scheme=self._auth_scheme,
            )
            vectors = _parse_embeddings(body, expected_count=len(batch), dimensions=self.dimensions)
            for (source_index, _chunk, byte_count), vector in zip(batch, vectors, strict=True):
                vectors_by_input[source_index].append((vector, byte_count))
        return tuple(_weighted_average(vectors) for vectors in vectors_by_input)


class BgeReranker:
    """Call ``POST /v1/rerank`` on the self-hosted BGE service."""

    def __init__(
        self,
        *,
        api_key: str,
        base_url: str,
        model: str,
        timeout_seconds: float,
        retries: int,
        auth_scheme: str = "bearer",
        recorder: ModelCallRecorder | None = None,
    ) -> None:
        if not api_key.strip() or not base_url.strip() or not model.strip():
            raise ValueError("BGE reranker configuration is incomplete")
        if auth_scheme not in {"bearer", "x_api_key"}:
            raise ValueError("BGE API auth scheme must be bearer or x_api_key")
        self._api_key = api_key
        self._url = _endpoint(base_url, "/v1/rerank")
        self._model = model
        self._timeout = timeout_seconds
        self._retries = retries
        self._auth_scheme = auth_scheme
        self._recorder = recorder

    def rerank(
        self, query: str, documents: Sequence[str], *, top_n: int | None = None
    ) -> tuple[tuple[int, float], ...]:
        if not query.strip():
            raise ValueError("BGE rerank query must not be blank")
        if not documents:
            return ()
        if top_n is not None and top_n < 1:
            raise ValueError("BGE rerank top_n must be positive")
        scores_by_document: dict[int, float] = {}
        for query_chunk in _split_utf8(query, _BGE_RERANK_QUERY_BYTE_BUDGET):
            document_byte_budget = _BGE_RERANK_PAIR_CONTENT_BYTE_BUDGET - _utf8_size(query_chunk)
            chunked_documents = tuple(
                (source_index, chunk)
                for source_index, document in enumerate(documents)
                for chunk in _split_utf8(document, document_byte_budget)
            )
            for start in range(0, len(chunked_documents), _BGE_RERANK_BATCH_SIZE):
                batch = chunked_documents[start : start + _BGE_RERANK_BATCH_SIZE]
                body = _call_qwen(
                    url=self._url,
                    api_key=self._api_key,
                    payload={
                        "query": query_chunk,
                        "documents": [chunk for _, chunk in batch],
                        "model": self._model,
                        "top_n": len(batch),
                        "return_documents": False,
                        "normalize": True,
                    },
                    timeout_seconds=self._timeout,
                    retries=self._retries,
                    operation="rerank",
                    provider="bge",
                    model=self._model,
                    input_count=len(batch),
                    recorder=self._recorder,
                    auth_scheme=self._auth_scheme,
                )
                for chunk_index, score in _parse_rerank(body, document_count=len(batch)):
                    source_index = batch[chunk_index][0]
                    previous_score = scores_by_document.get(source_index)
                    if previous_score is None or score > previous_score:
                        scores_by_document[source_index] = score
        ranked = tuple(sorted(scores_by_document.items(), key=lambda item: (-item[1], item[0])))
        return ranked if top_n is None else ranked[:top_n]


def _endpoint(base_url: str, suffix: str) -> str:
    value = base_url.rstrip("/")
    if value.endswith("/v1"):
        return f"{value}{suffix[3:]}"
    return f"{value}{suffix}"


def _embedding_batches(
    chunks: Sequence[tuple[int, str, int]],
) -> tuple[tuple[tuple[int, str, int], ...], ...]:
    batches: list[tuple[tuple[int, str, int], ...]] = []
    current: list[tuple[int, str, int]] = []
    current_size = 0
    for chunk in chunks:
        chunk_size = chunk[2] + _BGE_SPECIAL_TOKEN_RESERVE
        if current and (
            len(current) == _BGE_EMBEDDING_BATCH_SIZE
            or current_size + chunk_size > _BGE_EMBEDDING_TOKEN_LIMIT
        ):
            batches.append(tuple(current))
            current = []
            current_size = 0
        current.append(chunk)
        current_size += chunk_size
    if current:
        batches.append(tuple(current))
    return tuple(batches)


def _split_utf8(text: str, byte_budget: int) -> tuple[str, ...]:
    if byte_budget < 1:
        raise ValueError("BGE text byte budget must be positive")
    chunks: list[str] = []
    current: list[str] = []
    current_size = 0
    for character in text:
        character_size = _utf8_size(character)
        if current and current_size + character_size > byte_budget:
            chunks.append("".join(current))
            current = []
            current_size = 0
        current.append(character)
        current_size += character_size
    if current:
        chunks.append("".join(current))
    return tuple(chunks) if chunks else ("",)


def _utf8_size(text: str) -> int:
    return len(text.encode("utf-8"))


def _weighted_average(vectors: Sequence[tuple[tuple[float, ...], int]]) -> tuple[float, ...]:
    if not vectors:
        raise ValueError("BGE embedding input must not be empty")
    total_weight = sum(weight for _, weight in vectors)
    dimensions = len(vectors[0][0])
    return tuple(
        math.fsum(vector[dimension] * weight for vector, weight in vectors) / total_weight
        for dimension in range(dimensions)
    )
