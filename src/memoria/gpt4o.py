"""OpenAI-compatible gpt-4o-mini reranking client.

The Leaderboard's Full gate requires any model used by the submitted memory
system during Add or Search to be ``gpt-4o-mini``. This module provides a
reranker that scores already-retrieved evidence candidates with ``gpt-4o-mini``
through the OpenAI chat completions endpoint, keeping lexical retrieval local.
"""
from __future__ import annotations

import json
import math
import re
from collections.abc import Sequence

from memoria.embeddings import EmbeddingUnavailable
from memoria.qwen import ModelCallRecorder, _call_qwen

_SYSTEM_PROMPT = (
    "You are a relevance judge for a memory retrieval system. Score how relevant "
    "each numbered memory record is to the given query. Return ONLY a JSON object "
    'with a "scores" array of exactly one float per record, in the same order as '
    "the records, each in [0, 1] where 1 means highly relevant and 0 means "
    "irrelevant. Do not add comments or text outside the JSON."
)

_MAX_DOCS_PER_CALL = 100
_MAX_BYTES_PER_CALL = 24_000


class Gpt4oReranker:
    """Score candidate evidence with gpt-4o-mini via chat completions."""

    def __init__(
        self,
        *,
        api_key: str,
        base_url: str,
        model: str,
        timeout_seconds: float,
        retries: int,
        recorder: ModelCallRecorder | None = None,
    ) -> None:
        if not api_key.strip() or not model.strip() or not base_url.strip():
            raise ValueError("gpt-4o-mini reranker configuration is incomplete")
        self._api_key = api_key
        self._url = _join_url(base_url, "chat/completions")
        self._model = model
        self._timeout = timeout_seconds
        self._retries = retries
        self._recorder = recorder

    def rerank(
        self, query: str, documents: Sequence[str], *, top_n: int | None = None
    ) -> tuple[tuple[int, float], ...]:
        if not query.strip():
            raise ValueError("gpt-4o-mini rerank query must not be blank")
        if not documents:
            return ()
        if top_n is not None and top_n < 1:
            raise ValueError("gpt-4o-mini rerank top_n must be positive")

        scores_by_document: dict[int, float] = {}
        batch: list[int] = []
        batch_bytes = 0
        for index, document in enumerate(documents):
            size = _utf8_size(f"{index}|{document}")
            if batch and (len(batch) == _MAX_DOCS_PER_CALL or batch_bytes + size > _MAX_BYTES_PER_CALL):
                self._score_batch(query, batch, documents, scores_by_document)
                batch = []
                batch_bytes = 0
            batch.append(index)
            batch_bytes += size
        if batch:
            self._score_batch(query, batch, documents, scores_by_document)

        ranked = tuple(sorted(scores_by_document.items(), key=lambda item: (-item[1], item[0])))
        return ranked if top_n is None else ranked[:top_n]

    def _score_batch(
        self,
        query: str,
        batch: list[int],
        documents: Sequence[str],
        scores_by_document: dict[int, float],
    ) -> None:
        user_message = "Query: " + query + "\n\n" + "\n".join(
            f"{index}| " + documents[index].replace("\n", " ") for index in batch
        )
        body = _call_qwen(
            url=self._url,
            api_key=self._api_key,
            payload={
                "model": self._model,
                "messages": [
                    {"role": "system", "content": _SYSTEM_PROMPT},
                    {"role": "user", "content": user_message},
                ],
                "temperature": 0.0,
                "response_format": {"type": "json_object"},
            },
            timeout_seconds=self._timeout,
            retries=self._retries,
            operation="rerank",
            provider="gpt4o",
            model=self._model,
            input_count=len(batch),
            recorder=self._recorder,
        )
        scores = _parse_chat_scores(body, model=self._model, expected_count=len(batch))
        for index, score in zip(batch, scores, strict=True):
            scores_by_document[index] = score


def _parse_chat_scores(body: dict, *, model: str, expected_count: int) -> list[float]:
    choices = body.get("choices")
    if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
        raise EmbeddingUnavailable("gpt-4o-mini reranker response has no choices")
    messages = choices[0].get("message")
    content = messages.get("content") if isinstance(messages, dict) else None
    if not isinstance(content, str) or not content.strip():
        raise EmbeddingUnavailable("gpt-4o-mini reranker response has no message content")

    text = content.strip()
    match = re.match(r"^```(?:json)?\s*(.*?)\s*```$", text, flags=re.DOTALL)
    if match:
        text = match.group(1).strip()
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError as error:
        raise EmbeddingUnavailable("gpt-4o-mini reranker returned invalid JSON") from error
    if not isinstance(parsed, dict):
        raise EmbeddingUnavailable("gpt-4o-mini reranker response is not a JSON object")
    scores = parsed.get("scores")
    if not isinstance(scores, list) or len(scores) != expected_count:
        raise EmbeddingUnavailable("gpt-4o-mini reranker returned an invalid score count")

    out: list[float] = []
    for value in scores:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise EmbeddingUnavailable("gpt-4o-mini reranker returned a non-numeric score")
        score = float(value)
        if not math.isfinite(score):
            raise EmbeddingUnavailable("gpt-4o-mini reranker returned a non-finite score")
        out.append(max(0.0, min(1.0, score)))
    return out


def _utf8_size(text: str) -> int:
    return len(text.encode("utf-8"))


def _join_url(base_url: str, suffix: str) -> str:
    return f"{base_url.rstrip('/')}/{suffix}"