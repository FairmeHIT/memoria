"""Qwen-compatible embedding and reranking HTTP clients."""
from __future__ import annotations

import json
import math
import time
import urllib.error
import urllib.request
from collections.abc import Sequence
from typing import Any, Protocol

from memoria.embeddings import EmbeddingUnavailable
from memoria.model_audit import ModelCallAudit


class ModelCallRecorder(Protocol):
    def record(self, audit: ModelCallAudit) -> None: ...


class QwenEmbeddingProvider:
    """Call the OpenAI-compatible DashScope embedding endpoint."""

    def __init__(
        self,
        *,
        api_key: str,
        base_url: str,
        model: str,
        dimensions: int,
        timeout_seconds: float,
        retries: int,
        batch_size: int = 10,
        recorder: ModelCallRecorder | None = None,
    ) -> None:
        if not api_key.strip():
            raise ValueError("Qwen embedding API key must not be blank")
        if dimensions < 1 or not model.strip() or not base_url.strip():
            raise ValueError("Qwen embedding configuration is incomplete")
        self.fingerprint = f"qwen-embedding-v1:{model}:{dimensions}"
        self.dimensions = dimensions
        self._api_key = api_key
        self._url = _join_url(base_url, "embeddings")
        self._model = model
        self._timeout = timeout_seconds
        self._retries = retries
        self._batch_size = batch_size
        self._recorder = recorder

    def embed(self, texts: Sequence[str]) -> tuple[tuple[float, ...], ...]:
        outputs: list[tuple[float, ...]] = []
        for start in range(0, len(texts), self._batch_size):
            batch = list(texts[start : start + self._batch_size])
            payload = {
                "model": self._model,
                "input": batch,
                "dimensions": self.dimensions,
                "encoding_format": "float",
            }
            body = _call_qwen(
                url=self._url,
                api_key=self._api_key,
                payload=payload,
                timeout_seconds=self._timeout,
                retries=self._retries,
                operation="embedding",
                provider="qwen",
                model=self._model,
                input_count=len(batch),
                recorder=self._recorder,
            )
            outputs.extend(_parse_embeddings(body, expected_count=len(batch), dimensions=self.dimensions))
        return tuple(outputs)


class QwenReranker:
    """Call the Qwen3 reranker endpoint and return ranked candidate indexes."""

    def __init__(
        self,
        *,
        api_key: str,
        base_url: str,
        model: str,
        timeout_seconds: float,
        retries: int,
        instruct: str | None = None,
        recorder: ModelCallRecorder | None = None,
    ) -> None:
        if not api_key.strip():
            raise ValueError("Qwen reranker API key must not be blank")
        if not model.strip() or not base_url.strip():
            raise ValueError("Qwen reranker configuration is incomplete")
        self._api_key = api_key
        self._url = _join_url(base_url, "reranks")
        self._model = model
        self._timeout = timeout_seconds
        self._retries = retries
        self._instruct = instruct
        self._recorder = recorder

    def rerank(
        self, query: str, documents: Sequence[str], *, top_n: int | None = None
    ) -> tuple[tuple[int, float], ...]:
        if not documents:
            return ()
        if len(documents) > 500:
            raise ValueError("Qwen reranker accepts at most 500 documents")
        payload: dict[str, Any] = {
            "model": self._model,
            "query": query,
            "documents": list(documents),
        }
        if top_n is not None:
            payload["top_n"] = top_n
        if self._instruct:
            payload["instruct"] = self._instruct
        body = _call_qwen(
            url=self._url,
            api_key=self._api_key,
            payload=payload,
            timeout_seconds=self._timeout,
            retries=self._retries,
            operation="rerank",
            provider="qwen",
            model=self._model,
            input_count=len(documents),
            recorder=self._recorder,
        )
        return _parse_rerank(body, document_count=len(documents))


def _call_qwen(
    *,
    url: str,
    api_key: str,
    payload: dict[str, Any],
    timeout_seconds: float,
    retries: int,
    operation: str,
    provider: str,
    model: str,
    input_count: int,
    recorder: ModelCallRecorder | None,
    auth_scheme: str = "bearer",
) -> dict[str, Any]:
    started = time.perf_counter()
    try:
        body, attempts = _post_json(
            url=url,
            api_key=api_key,
            payload=payload,
            timeout_seconds=timeout_seconds,
            retries=retries,
            auth_scheme=auth_scheme,
        )
    except EmbeddingUnavailable as error:
        _record_call(
            recorder=recorder,
            operation=operation,
            provider=provider,
            model=model,
            input_count=input_count,
            attempts=retries + 1,
            elapsed_ms=_elapsed_ms(started),
            success=False,
            error_kind="request_failed",
        )
        raise error
    usage = body.get("usage")
    _record_call(
        recorder=recorder,
        operation=operation,
        provider=provider,
        model=model,
        input_count=input_count,
        prompt_tokens=_usage_int(usage, "prompt_tokens"),
        total_tokens=_usage_int(usage, "total_tokens"),
        attempts=attempts,
        elapsed_ms=_elapsed_ms(started),
        success=True,
        error_kind=None,
    )
    return body


def _post_json(
    *,
    url: str,
    api_key: str,
    payload: dict[str, Any],
    timeout_seconds: float,
    retries: int,
    auth_scheme: str = "bearer",
) -> tuple[dict[str, Any], int]:
    if auth_scheme not in {"bearer", "x_api_key"}:
        raise ValueError("remote API auth scheme must be bearer or x_api_key")
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
        if auth_scheme == "bearer"
        else "",
        "X-API-Key": api_key if auth_scheme == "x_api_key" else "",
    }
    headers = {key: value for key, value in headers.items() if value}
    request = urllib.request.Request(
        url,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers=headers,
        method="POST",
    )
    last_error: Exception | None = None
    for attempt in range(retries + 1):
        try:
            with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
                decoded = json.loads(response.read().decode("utf-8"))
            if not isinstance(decoded, dict):
                raise EmbeddingUnavailable("Qwen API returned an invalid response")
            return decoded, attempt + 1
        except (EmbeddingUnavailable, json.JSONDecodeError, TypeError) as error:
            last_error = error
            break
        except urllib.error.HTTPError as error:
            last_error = error
            if error.code in {408, 429} or 500 <= error.code <= 599:
                if attempt < retries:
                    time.sleep(min(0.25 * (2**attempt), 2.0))
                    continue
            break
        except (urllib.error.URLError, TimeoutError, OSError) as error:
            last_error = error
            if attempt < retries:
                time.sleep(min(0.25 * (2**attempt), 2.0))
    raise EmbeddingUnavailable("Qwen API request failed") from last_error


def _record_call(
    *,
    recorder: ModelCallRecorder | None,
    operation: str,
    provider: str,
    model: str,
    input_count: int,
    attempts: int,
    elapsed_ms: float,
    success: bool,
    error_kind: str | None,
    prompt_tokens: int | None = None,
    total_tokens: int | None = None,
) -> None:
    if recorder is None:
        return
    try:
        recorder.record(
            ModelCallAudit(
                operation=operation,
                provider=provider,
                model=model,
                input_count=input_count,
                prompt_tokens=prompt_tokens,
                total_tokens=total_tokens,
                attempts=attempts,
                elapsed_ms=elapsed_ms,
                success=success,
                error_kind=error_kind,
            )
        )
    except Exception:
        # Diagnostics are best effort and must not change API results.
        return


def _usage_int(usage: object, key: str) -> int | None:
    if not isinstance(usage, dict):
        return None
    value = usage.get(key)
    return value if isinstance(value, int) and value >= 0 else None


def _elapsed_ms(started: float) -> float:
    return (time.perf_counter() - started) * 1_000


def _parse_embeddings(
    body: dict[str, Any], *, expected_count: int, dimensions: int
) -> list[tuple[float, ...]]:
    data = body.get("data")
    if not isinstance(data, list) or len(data) != expected_count:
        raise EmbeddingUnavailable("Qwen embedding response has an invalid item count")
    ordered: list[tuple[float, ...] | None] = [None] * expected_count
    for position, item in enumerate(data):
        if not isinstance(item, dict):
            raise EmbeddingUnavailable("Qwen embedding response has an invalid item")
        index = item.get("index", position)
        vector = item.get("embedding")
        if not isinstance(index, int) or not 0 <= index < expected_count:
            raise EmbeddingUnavailable("Qwen embedding response has an invalid index")
        if not isinstance(vector, list) or len(vector) != dimensions:
            raise EmbeddingUnavailable("Qwen embedding response has an invalid dimension")
        values = tuple(float(value) for value in vector)
        if any(not math.isfinite(value) for value in values):
            raise EmbeddingUnavailable("Qwen embedding response has an invalid value")
        if ordered[index] is not None:
            raise EmbeddingUnavailable("Qwen embedding response has duplicate indexes")
        ordered[index] = values
    if any(value is None for value in ordered):
        raise EmbeddingUnavailable("Qwen embedding response is missing an item")
    return [value for value in ordered if value is not None]


def _parse_rerank(body: dict[str, Any], *, document_count: int) -> tuple[tuple[int, float], ...]:
    results = body.get("results")
    if not isinstance(results, list):
        raise EmbeddingUnavailable("Qwen reranker response has no results")
    parsed: list[tuple[int, float]] = []
    seen: set[int] = set()
    for result in results:
        if not isinstance(result, dict):
            raise EmbeddingUnavailable("Qwen reranker response has an invalid result")
        index = result.get("index")
        score = result.get("relevance_score")
        if (
            not isinstance(index, int)
            or not 0 <= index < document_count
            or index in seen
            or not isinstance(score, (int, float))
            or not math.isfinite(float(score))
        ):
            raise EmbeddingUnavailable("Qwen reranker response has an invalid result")
        seen.add(index)
        parsed.append((index, float(score)))
    return tuple(sorted(parsed, key=lambda item: (-item[1], item[0])))


def create_reranker(
    *,
    backend: str,
    api_key: str | None,
    base_url: str,
    model: str,
    timeout_seconds: float,
    retries: int,
    instruct: str | None,
    recorder: ModelCallRecorder | None = None,
    auth_scheme: str = "bearer",
) -> QwenReranker | None:
    if backend == "none":
        return None
    if api_key is None:
        raise ValueError("remote reranker API key is required")
    if backend == "qwen":
        return QwenReranker(
            api_key=api_key,
            base_url=base_url,
            model=model,
            timeout_seconds=timeout_seconds,
            retries=retries,
            instruct=instruct,
            recorder=recorder,
        )
    if backend == "bge":
        from memoria.bge import BgeReranker

        return BgeReranker(
            api_key=api_key,
            base_url=base_url,
            model=model,
            timeout_seconds=timeout_seconds,
            retries=retries,
            auth_scheme=auth_scheme,
            recorder=recorder,
        )
    if backend == "gpt4o":
        from memoria.gpt4o import Gpt4oReranker

        return Gpt4oReranker(
            api_key=api_key,
            base_url=base_url,
            model=model,
            timeout_seconds=timeout_seconds,
            retries=retries,
            recorder=recorder,
        )
    raise ValueError("MEMORIA_RERANKER_BACKEND must be none, qwen, bge, or gpt4o")


def _join_url(base_url: str, suffix: str) -> str:
    return f"{base_url.rstrip('/')}/{suffix}"
