"""SQLite persistence, idempotent ingestion, FTS retrieval, and retention."""
from __future__ import annotations

import hashlib
import json
import re
import sqlite3
import threading
import time
from collections.abc import Sequence
from dataclasses import dataclass, replace
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Protocol

from memoria.claims import Claim, extract_claims
from memoria.config import Settings
from memoria.embeddings import EmbeddingProvider, EmbeddingUnavailable, deserialize_vector, serialize_vector
from memoria.schemas import AddRequest, AddResponse, SearchHit


SCHEMA = """
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS add_requests (
    request_id TEXT PRIMARY KEY,
    request_hash TEXT NOT NULL,
    user_id TEXT NOT NULL,
    session_id TEXT NOT NULL,
    committed_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS messages (
    id TEXT PRIMARY KEY,
    add_request_id TEXT NOT NULL REFERENCES add_requests(request_id) ON DELETE CASCADE,
    user_id TEXT NOT NULL,
    session_id TEXT NOT NULL,
    sequence INTEGER NOT NULL,
    role TEXT NOT NULL CHECK(role IN ('user', 'assistant')),
    content TEXT NOT NULL,
    observed_at TEXT,
    created_at TEXT NOT NULL,
    content_hash TEXT NOT NULL,
    UNIQUE(add_request_id, sequence)
);

CREATE INDEX IF NOT EXISTS messages_user_id_idx ON messages(user_id);
CREATE INDEX IF NOT EXISTS messages_add_request_id_idx ON messages(add_request_id);

CREATE TABLE IF NOT EXISTS memories (
    id TEXT PRIMARY KEY,
    message_id TEXT NOT NULL UNIQUE REFERENCES messages(id) ON DELETE CASCADE,
    user_id TEXT NOT NULL,
    session_id TEXT NOT NULL,
    content TEXT NOT NULL,
    created_at TEXT NOT NULL,
    content_hash TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS memories_user_id_idx ON memories(user_id);

CREATE TABLE IF NOT EXISTS memory_embeddings (
    memory_id TEXT PRIMARY KEY REFERENCES memories(id) ON DELETE CASCADE,
    user_id TEXT NOT NULL,
    model_fingerprint TEXT NOT NULL,
    dimensions INTEGER NOT NULL CHECK(dimensions > 0),
    vector BLOB NOT NULL,
    created_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS memory_embeddings_user_model_idx
    ON memory_embeddings(user_id, model_fingerprint);

CREATE TABLE IF NOT EXISTS memory_claims (
    memory_id TEXT NOT NULL REFERENCES memories(id) ON DELETE CASCADE,
    user_id TEXT NOT NULL,
    predicate TEXT NOT NULL,
    value_key TEXT NOT NULL,
    polarity INTEGER NOT NULL CHECK(polarity IN (-1, 1)),
    created_at TEXT NOT NULL,
    PRIMARY KEY (memory_id, predicate, value_key)
);

CREATE INDEX IF NOT EXISTS memory_claims_lookup_idx
    ON memory_claims(user_id, predicate, value_key, polarity);

CREATE TABLE IF NOT EXISTS memory_supersessions (
    superseding_memory_id TEXT NOT NULL REFERENCES memories(id) ON DELETE CASCADE,
    superseded_memory_id TEXT NOT NULL REFERENCES memories(id) ON DELETE CASCADE,
    PRIMARY KEY (superseding_memory_id, superseded_memory_id),
    CHECK(superseding_memory_id != superseded_memory_id)
);

CREATE TABLE IF NOT EXISTS search_audit (
    trace_id TEXT PRIMARY KEY,
    user_id_hash TEXT NOT NULL,
    query_hash TEXT NOT NULL,
    requested_top_k INTEGER NOT NULL,
    candidate_count INTEGER NOT NULL,
    selected_count INTEGER NOT NULL,
    candidate_ids TEXT NOT NULL,
    selected_ids TEXT NOT NULL,
    elapsed_ms REAL NOT NULL,
    index_version TEXT NOT NULL,
    created_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS search_audit_created_at_idx ON search_audit(created_at);

CREATE TABLE IF NOT EXISTS model_audit (
    audit_id TEXT PRIMARY KEY,
    operation TEXT NOT NULL CHECK(operation IN ('embedding', 'rerank')),
    provider TEXT NOT NULL,
    model TEXT NOT NULL,
    input_count INTEGER NOT NULL CHECK(input_count >= 0),
    prompt_tokens INTEGER,
    total_tokens INTEGER,
    attempts INTEGER NOT NULL CHECK(attempts >= 1),
    elapsed_ms REAL NOT NULL CHECK(elapsed_ms >= 0),
    success INTEGER NOT NULL CHECK(success IN (0, 1)),
    error_kind TEXT,
    created_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS model_audit_created_at_idx ON model_audit(created_at);

CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts USING fts5(
    memory_id UNINDEXED,
    content,
    tokenize='unicode61'
);
"""

TOKEN_RE = re.compile(r"[A-Za-z0-9_]+|[\u4e00-\u9fff]+")


class IdempotencyConflict(Exception):
    """A request identifier was reused for different submitted content."""


@dataclass(frozen=True, slots=True)
class RankedMemory:
    id: str
    content: str
    score: float
    created_at: str
    content_hash: str


class RerankerProvider(Protocol):
    def rerank(
        self, query: str, documents: Sequence[str], *, top_n: int | None = None
    ) -> tuple[tuple[int, float], ...]: ...


class MemoryStore:
    """A small, thread-safe SQLite store with synchronous write visibility."""

    def __init__(
        self,
        settings: Settings,
        *,
        embedder: EmbeddingProvider | None = None,
        reranker: RerankerProvider | None = None,
    ) -> None:
        self._settings = settings
        self._database_path = settings.data_dir / "memoria.sqlite3"
        self._write_lock = threading.RLock()
        self._embedder = embedder
        self._reranker = reranker

    def initialize(self) -> None:
        self._settings.data_dir.mkdir(parents=True, exist_ok=True)
        with self._write_lock:
            connection = self._connect(enable_wal=True)
            try:
                connection.executescript(SCHEMA)
            finally:
                connection.close()

    def healthcheck(self) -> bool:
        try:
            connection = self._connect()
            try:
                return connection.execute("SELECT 1").fetchone() is not None
            finally:
                connection.close()
        except sqlite3.Error:
            return False

    def add(self, request: AddRequest) -> AddResponse:
        """Commit a request once, including its FTS entries, before responding."""

        request_hash = _canonical_hash(request.model_dump(mode="json"))
        now = _utc_now()
        response = AddResponse(
            success=True,
            request_id=request.request_id,
            user_id=request.user_id,
            session_id=request.session_id,
        )
        existing_hash = self._request_hash(request.request_id)
        if existing_hash is not None:
            if existing_hash == request_hash:
                return response
            raise IdempotencyConflict
        vectors = self._embed_messages([message.content for message in request.messages])

        with self._write_lock:
            connection = self._connect()
            try:
                connection.execute("BEGIN IMMEDIATE")
                existing = connection.execute(
                    "SELECT request_hash FROM add_requests WHERE request_id = ?",
                    (request.request_id,),
                ).fetchone()
                if existing is not None:
                    connection.execute("COMMIT")
                    if existing["request_hash"] == request_hash:
                        return response
                    raise IdempotencyConflict

                connection.execute(
                    """
                    INSERT INTO add_requests (request_id, request_hash, user_id, session_id, committed_at)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (request.request_id, request_hash, request.user_id, request.session_id, now),
                )
                for sequence, message in enumerate(request.messages):
                    message_id = _stable_id("msg", request.request_id, str(sequence))
                    memory_id = _stable_id("mem", request.request_id, str(sequence))
                    observed_at = _timestamp_to_iso(message.timestamp)
                    created_at = observed_at or now
                    content_hash = _sha256(message.content)
                    evidence = f"{message.role}: {message.content}"
                    connection.execute(
                        """
                        INSERT INTO messages (
                            id, add_request_id, user_id, session_id, sequence, role, content,
                            observed_at, created_at, content_hash
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            message_id,
                            request.request_id,
                            request.user_id,
                            request.session_id,
                            sequence,
                            message.role,
                            message.content,
                            observed_at,
                            created_at,
                            content_hash,
                        ),
                    )
                    connection.execute(
                        """
                        INSERT INTO memories (id, message_id, user_id, session_id, content, created_at, content_hash)
                        VALUES (?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            memory_id,
                            message_id,
                            request.user_id,
                            request.session_id,
                            evidence,
                            created_at,
                            content_hash,
                        ),
                    )
                    connection.execute(
                        "INSERT INTO memories_fts (memory_id, content) VALUES (?, ?)",
                        (memory_id, evidence),
                    )
                    if vectors is not None:
                        connection.execute(
                            """
                            INSERT INTO memory_embeddings (
                                memory_id, user_id, model_fingerprint, dimensions, vector, created_at
                            ) VALUES (?, ?, ?, ?, ?, ?)
                            """,
                            (
                                memory_id,
                                request.user_id,
                                self._embedder.fingerprint,
                                self._embedder.dimensions,
                                vectors[sequence],
                                created_at,
                            ),
                        )
                    if message.role == "user":
                        self._record_claims(
                            connection=connection,
                            memory_id=memory_id,
                            user_id=request.user_id,
                            created_at=created_at,
                            claims=extract_claims(message.content),
                        )
                connection.execute("COMMIT")
                return response
            except Exception:
                if connection.in_transaction:
                    connection.execute("ROLLBACK")
                raise
            finally:
                connection.close()

    def _request_hash(self, request_id: str) -> str | None:
        connection = self._connect()
        try:
            row = connection.execute(
                "SELECT request_hash FROM add_requests WHERE request_id = ?", (request_id,)
            ).fetchone()
            return None if row is None else str(row["request_hash"])
        finally:
            connection.close()

    def _embed_messages(self, contents: list[str]) -> tuple[bytes, ...] | None:
        if self._embedder is None:
            return None
        vectors = self._embedder.embed(contents)
        if len(vectors) != len(contents):
            raise EmbeddingUnavailable("embedding provider returned an unexpected vector count")
        encoded = tuple(serialize_vector(vector) for vector in vectors)
        expected_size = self._embedder.dimensions * 4
        if any(len(vector) != expected_size for vector in encoded):
            raise EmbeddingUnavailable("embedding provider returned an unexpected vector dimension")
        return encoded

    def reindex_embeddings(self, *, batch_size: int = 100) -> dict[str, int]:
        """Backfill or replace vectors whose provider fingerprint is stale."""

        if self._embedder is None:
            raise EmbeddingUnavailable("vector reindex requires an embedding provider")
        if not 1 <= batch_size <= 500:
            raise ValueError("batch_size must be between 1 and 500")
        connection = self._connect()
        try:
            rows = connection.execute(
                """
                SELECT memories.id, memories.user_id, memories.created_at, messages.content
                FROM memories
                JOIN messages ON messages.id = memories.message_id
                LEFT JOIN memory_embeddings
                  ON memory_embeddings.memory_id = memories.id
                 AND memory_embeddings.model_fingerprint = ?
                 AND memory_embeddings.dimensions = ?
                WHERE memory_embeddings.memory_id IS NULL
                ORDER BY memories.id
                """,
                (self._embedder.fingerprint, self._embedder.dimensions),
            ).fetchall()
        finally:
            connection.close()

        updated = 0
        for start in range(0, len(rows), batch_size):
            batch = rows[start : start + batch_size]
            vectors = self._embed_messages([str(row["content"]) for row in batch])
            if vectors is None:
                raise EmbeddingUnavailable("vector reindex provider is unavailable")
            with self._write_lock:
                connection = self._connect()
                try:
                    connection.execute("BEGIN IMMEDIATE")
                    for row, vector in zip(batch, vectors, strict=True):
                        connection.execute(
                            """
                            INSERT INTO memory_embeddings (
                                memory_id, user_id, model_fingerprint, dimensions, vector, created_at
                            ) VALUES (?, ?, ?, ?, ?, ?)
                            ON CONFLICT(memory_id) DO UPDATE SET
                                user_id = excluded.user_id,
                                model_fingerprint = excluded.model_fingerprint,
                                dimensions = excluded.dimensions,
                                vector = excluded.vector,
                                created_at = excluded.created_at
                            """,
                            (
                                str(row["id"]),
                                str(row["user_id"]),
                                self._embedder.fingerprint,
                                self._embedder.dimensions,
                                vector,
                                str(row["created_at"]),
                            ),
                        )
                        updated += 1
                    connection.execute("COMMIT")
                except Exception:
                    if connection.in_transaction:
                        connection.execute("ROLLBACK")
                    raise
                finally:
                    connection.close()
        return {"scanned": len(rows), "updated": updated}

    def search(
        self,
        *,
        query: str,
        options: list[str] | None,
        user_id: str,
        top_k: int,
    ) -> list[SearchHit]:
        """Return only user-scoped, deduplicated evidence in stable score order."""

        started = time.perf_counter()
        primary_expression = _fts_expression(query)
        option_expression = _fts_expression(" ".join(options or []))
        query_vector = self._embed_query(query)
        if not primary_expression and not option_expression and query_vector is None:
            self._record_search_audit(
                user_id=user_id,
                query=query,
                top_k=top_k,
                candidate_ids=[],
                selected_ids=[],
                elapsed_ms=_elapsed_ms(started),
            )
            return []

        candidate_limit = min(max(top_k * 5, 100), 5_000)
        connection = self._connect()
        try:
            user_count = int(
                connection.execute(
                    "SELECT COUNT(*) FROM memories WHERE user_id = ?", (user_id,)
                ).fetchone()[0]
            )
            if primary_expression or option_expression:
                if user_count <= 5_000:
                    lexical_rows = self._scoped_candidate_rows(
                        connection=connection,
                        query=query,
                        options=options or [],
                        user_id=user_id,
                        candidate_limit=candidate_limit,
                    )
                else:
                    lexical_rows = self._candidate_rows(
                        connection=connection,
                        expressions=(primary_expression, option_expression),
                        user_id=user_id,
                        candidate_limit=candidate_limit,
                    )
            else:
                lexical_rows = []
            vector_rows, vector_similarities = self._vector_candidate_rows(
                connection=connection,
                user_id=user_id,
                query_vector=query_vector,
                candidate_limit=min(
                    max(top_k * 5, 100), self._settings.vector_candidate_limit
                ),
            )
            rows_by_id = {str(row["id"]): row for row in lexical_rows}
            rows_by_id.update(
                {str(row["id"]): row for row in vector_rows if str(row["id"]) not in rows_by_id}
            )
            rows = list(rows_by_id.values())
            superseded_ids = (
                set()
                if _is_historical_query(query)
                else self._superseded_memory_ids(connection, [str(row["id"]) for row in rows])
            )
        finally:
            connection.close()

        recency_scores = _recency_scores(rows) if _is_current_query(query) else {}
        lexical_ranks = _candidate_ranks(lexical_rows, query, options or [])
        vector_ranks = {
            str(row["id"]): rank for rank, row in enumerate(vector_rows, start=1)
        }
        ranked = sorted(
            (
                _rank_memory(
                    row,
                    query,
                    options or [],
                    lexical_rank=lexical_ranks.get(str(row["id"])),
                    vector_rank=vector_ranks.get(str(row["id"])),
                    vector_similarity=vector_similarities.get(str(row["id"]), 0.0),
                    rrf_k=self._settings.rrf_k,
                    hybrid=self._embedder is not None,
                    recency_score=recency_scores.get(str(row["id"]), 0.0),
                )
                for row in rows
                if str(row["id"]) not in superseded_ids
            ),
            key=lambda item: (-item.score, item.created_at, item.id),
            reverse=False,
        )
        if self._reranker is not None and ranked:
            ranked = self._apply_reranker(query, ranked)
        selected: list[SearchHit] = []
        seen_content_hashes: set[str] = set()
        for item in ranked:
            if item.content_hash in seen_content_hashes:
                continue
            seen_content_hashes.add(item.content_hash)
            selected.append(
                SearchHit(
                    id=item.id,
                    content=item.content,
                    score=round(item.score, 6),
                    created_at=item.created_at,
                )
            )
            if len(selected) == top_k:
                break
        self._record_search_audit(
            user_id=user_id,
            query=query,
            top_k=top_k,
            candidate_ids=[str(row["id"]) for row in rows],
            selected_ids=[item.id for item in selected],
            elapsed_ms=_elapsed_ms(started),
        )
        return selected

    def _apply_reranker(self, query: str, ranked: list[RankedMemory]) -> list[RankedMemory]:
        candidates = ranked[: self._settings.reranker_candidate_limit]
        results = self._reranker.rerank(
            query,
            [item.content for item in candidates],
            top_n=len(candidates),
        )
        reranked: list[RankedMemory] = []
        seen_indexes: set[int] = set()
        for index, score in results:
            if not 0 <= index < len(candidates) or index in seen_indexes:
                continue
            seen_indexes.add(index)
            reranked.append(replace(candidates[index], score=score))
        fallback_score = min((item.score for item in reranked), default=0.0)
        for item in [
            *[candidates[index] for index in range(len(candidates)) if index not in seen_indexes],
            *ranked[len(candidates) :],
        ]:
            fallback_score -= 0.000001
            reranked.append(replace(item, score=fallback_score))
        return reranked

    def _embed_query(self, query: str) -> tuple[float, ...] | None:
        if self._embedder is None:
            return None
        vectors = self._embedder.embed([query])
        if len(vectors) != 1:
            raise EmbeddingUnavailable("embedding provider returned an unexpected vector count")
        return deserialize_vector(
            serialize_vector(vectors[0]), dimensions=self._embedder.dimensions
        )

    def _scoped_candidate_rows(
        self,
        *,
        connection: sqlite3.Connection,
        query: str,
        options: list[str],
        user_id: str,
        candidate_limit: int,
    ) -> list[sqlite3.Row]:
        query_terms = _canonical_terms(query)
        option_terms = _canonical_terms(" ".join(options))
        if not query_terms and not option_terms:
            return []
        rows = connection.execute(
            """
            SELECT id, content, created_at, content_hash, 0.0 AS bm25_score
            FROM memories
            WHERE user_id = ?
            LIMIT ?
            """,
            (user_id, max(candidate_limit, 5_000)),
        ).fetchall()
        matching = []
        for row in rows:
            content_terms = _canonical_terms(str(row["content"]))
            if query_terms & content_terms or option_terms & content_terms:
                matching.append(row)
        return matching[:candidate_limit]

    def _candidate_rows(
        self,
        *,
        connection: sqlite3.Connection,
        expressions: tuple[str, str],
        user_id: str,
        candidate_limit: int,
    ) -> list[sqlite3.Row]:
        rows_by_id: dict[str, sqlite3.Row] = {}
        for expression in expressions:
            if not expression:
                continue
            rows = connection.execute(
                """
                SELECT memories.id, memories.content, memories.created_at, memories.content_hash,
                       bm25(memories_fts) AS bm25_score
                FROM memories_fts
                JOIN memories ON memories.id = memories_fts.memory_id
                WHERE memories_fts MATCH ? AND memories.user_id = ?
                LIMIT ?
                """,
                (expression, user_id, candidate_limit),
            ).fetchall()
            for row in rows:
                rows_by_id.setdefault(str(row["id"]), row)
        return list(rows_by_id.values())

    def _vector_candidate_rows(
        self,
        *,
        connection: sqlite3.Connection,
        user_id: str,
        query_vector: tuple[float, ...] | None,
        candidate_limit: int,
    ) -> tuple[list[sqlite3.Row], dict[str, float]]:
        if query_vector is None or self._embedder is None:
            return [], {}
        rows = connection.execute(
            """
            SELECT memories.id, memories.content, memories.created_at, memories.content_hash,
                   0.0 AS bm25_score, memory_embeddings.vector
            FROM memory_embeddings
            JOIN memories ON memories.id = memory_embeddings.memory_id
            WHERE memory_embeddings.user_id = ?
              AND memories.user_id = ?
              AND memory_embeddings.model_fingerprint = ?
              AND memory_embeddings.dimensions = ?
            """,
            (
                user_id,
                user_id,
                self._embedder.fingerprint,
                self._embedder.dimensions,
            ),
        ).fetchall()
        similarities = {
            str(row["id"]): max(
                0.0,
                sum(
                    left * right
                    for left, right in zip(
                        query_vector,
                        deserialize_vector(
                            bytes(row["vector"]), dimensions=self._embedder.dimensions
                        ),
                        strict=True,
                    )
                ),
            )
            for row in rows
        }
        ranked_rows = sorted(
            rows,
            key=lambda row: (
                -similarities[str(row["id"])],
                str(row["created_at"]),
                str(row["id"]),
            ),
        )[:candidate_limit]
        return ranked_rows, similarities

    def _record_claims(
        self,
        *,
        connection: sqlite3.Connection,
        memory_id: str,
        user_id: str,
        created_at: str,
        claims: tuple[Claim, ...],
    ) -> None:
        for claim in claims:
            if claim.exclusive:
                prior_rows = connection.execute(
                    """
                    SELECT memory_id
                    FROM memory_claims
                    WHERE user_id = ? AND predicate = ? AND polarity = ?
                    """,
                    (user_id, claim.predicate, claim.polarity),
                ).fetchall()
            else:
                prior_rows = connection.execute(
                    """
                    SELECT memory_id
                    FROM memory_claims
                    WHERE user_id = ? AND predicate = ? AND value_key = ? AND polarity != ?
                    """,
                    (user_id, claim.predicate, claim.value_key, claim.polarity),
                ).fetchall()
            connection.execute(
                """
                INSERT INTO memory_claims (memory_id, user_id, predicate, value_key, polarity, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (memory_id, user_id, claim.predicate, claim.value_key, claim.polarity, created_at),
            )
            connection.executemany(
                """
                INSERT OR IGNORE INTO memory_supersessions (
                    superseding_memory_id, superseded_memory_id
                ) VALUES (?, ?)
                """,
                ((memory_id, str(row["memory_id"])) for row in prior_rows),
            )

    def _superseded_memory_ids(
        self, connection: sqlite3.Connection, memory_ids: list[str]
    ) -> set[str]:
        if not memory_ids:
            return set()
        placeholders = ", ".join("?" for _ in memory_ids)
        rows = connection.execute(
            f"""
            SELECT superseded_memory_id
            FROM memory_supersessions
            WHERE superseded_memory_id IN ({placeholders})
            """,
            memory_ids,
        ).fetchall()
        return {str(row["superseded_memory_id"]) for row in rows}

    def _record_search_audit(
        self,
        *,
        user_id: str,
        query: str,
        top_k: int,
        candidate_ids: list[str],
        selected_ids: list[str],
        elapsed_ms: float,
    ) -> None:
        """Persist body-free retrieval diagnostics without affecting Search."""

        trace_id = _stable_id("trace", f"{user_id}:{query}:{time.time_ns()}", "0")
        try:
            with self._write_lock:
                connection = self._connect(timeout=0.25)
                try:
                    connection.execute(
                        """
                        INSERT INTO search_audit (
                            trace_id, user_id_hash, query_hash, requested_top_k,
                            candidate_count, selected_count, candidate_ids, selected_ids,
                            elapsed_ms, index_version, created_at
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            trace_id,
                            _sha256(user_id),
                            _sha256(query),
                            top_k,
                            len(candidate_ids),
                            len(selected_ids),
                            json.dumps(candidate_ids, separators=(",", ":")),
                            json.dumps(selected_ids, separators=(",", ":")),
                            round(elapsed_ms, 3),
                            "hybrid-fts5-v3-hashing-v1"
                            if self._embedder is not None
                            else "fts5-v3-vectors-v1",
                            _utc_now(),
                        ),
                    )
                finally:
                    connection.close()
        except sqlite3.Error:
            # Diagnostics must never turn a successful retrieval into a failure.
            return

    def cleanup_expired(self, *, now: datetime | None = None) -> int:
        """Delete expired source rows, derivatives, and FTS entries atomically."""

        cutoff = (now or datetime.now(UTC)) - timedelta(days=self._settings.retention_days)
        cutoff_text = cutoff.astimezone(UTC).isoformat()
        with self._write_lock:
            connection = self._connect()
            try:
                connection.execute("BEGIN IMMEDIATE")
                rows = connection.execute(
                    """
                    SELECT memories.id
                    FROM memories
                    JOIN messages ON messages.id = memories.message_id
                    JOIN add_requests ON add_requests.request_id = messages.add_request_id
                    WHERE add_requests.committed_at < ?
                    """,
                    (cutoff_text,),
                ).fetchall()
                memory_ids = [row["id"] for row in rows]
                if memory_ids:
                    connection.executemany(
                        "DELETE FROM memories_fts WHERE memory_id = ?",
                        ((memory_id,) for memory_id in memory_ids),
                    )
                deleted = connection.execute(
                    "DELETE FROM add_requests WHERE committed_at < ?",
                    (cutoff_text,),
                ).rowcount
                connection.execute("DELETE FROM search_audit WHERE created_at < ?", (cutoff_text,))
                connection.execute("DELETE FROM model_audit WHERE created_at < ?", (cutoff_text,))
                connection.execute("COMMIT")
                return deleted
            except Exception:
                if connection.in_transaction:
                    connection.execute("ROLLBACK")
                raise
            finally:
                connection.close()

    def _connect(self, *, timeout: float = 10, enable_wal: bool = False) -> sqlite3.Connection:
        connection = sqlite3.connect(self._database_path, timeout=timeout, isolation_level=None)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA busy_timeout = 10000")
        if enable_wal:
            connection.execute("PRAGMA journal_mode = WAL")
        return connection


def _canonical_hash(payload: dict[str, object]) -> str:
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return _sha256(encoded)


def _sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _stable_id(prefix: str, request_id: str, sequence: str) -> str:
    digest = _sha256(f"{request_id}:{sequence}")[:26]
    return f"{prefix}_{digest}"


def stable_memory_id(request_id: str, sequence: int) -> str:
    """Return the public deterministic memory ID for a source message."""

    return _stable_id("mem", request_id, str(sequence))


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _timestamp_to_iso(timestamp: int | None) -> str | None:
    if timestamp is None:
        return None
    return datetime.fromtimestamp(timestamp / 1000, tz=UTC).isoformat()


def _fts_expression(query: str) -> str:
    tokens = {
        variant
        for token in TOKEN_RE.findall(query)
        for variant in _term_variants(token)
    }
    return " OR ".join('"' + token.replace('"', '""') + '"' for token in sorted(tokens))


def _term_variants(token: str) -> set[str]:
    """Return a conservative set of query/index-compatible inflection forms."""

    normalized = token.casefold()
    variants = {normalized}
    if len(normalized) > 3 and not normalized.endswith("s"):
        variants.add(normalized + "s")
    if normalized.endswith("ies") and len(normalized) > 4:
        variants.add(normalized[:-3] + "y")
    if normalized.endswith("ing") and len(normalized) > 5:
        root = normalized[:-3]
        if len(root) > 2 and root[-1] == root[-2]:
            root = root[:-1]
        if root.endswith("z"):
            variants.add(root + "e")
        variants.add(root)
    if normalized.endswith("ed") and len(normalized) > 4:
        root = normalized[:-2]
        variants.add(root + "e" if root.endswith("z") else root)
    if normalized.endswith("es") and len(normalized) > 4:
        variants.add(normalized[:-2])
    if normalized.endswith("s") and len(normalized) > 4:
        variants.add(normalized[:-1])
    return variants


def _canonical_terms(text: str) -> set[str]:
    return {_canonical_token(token) for token in TOKEN_RE.findall(text)}


def _canonical_token(token: str) -> str:
    normalized = token.casefold()
    if normalized.endswith("ies") and len(normalized) > 4:
        return normalized[:-3] + "y"
    if normalized.endswith("ing") and len(normalized) > 5:
        root = normalized[:-3]
        if len(root) > 2 and root[-1] == root[-2]:
            root = root[:-1]
        return root + "e" if root.endswith("z") else root
    if normalized.endswith("ed") and len(normalized) > 4:
        root = normalized[:-2]
        return root + "e" if root.endswith("z") else root
    if normalized.endswith("es") and len(normalized) > 4:
        return normalized[:-2]
    if normalized.endswith("s") and len(normalized) > 4:
        return normalized[:-1]
    return normalized


def _rank_memory(
    row: sqlite3.Row,
    query: str,
    options: list[str],
    *,
    lexical_rank: int | None,
    vector_rank: int | None,
    vector_similarity: float,
    rrf_k: int,
    hybrid: bool,
    recency_score: float = 0.0,
) -> RankedMemory:
    query_terms = _canonical_terms(query)
    option_terms = _canonical_terms(" ".join(options))
    content_terms = _canonical_terms(str(row["content"]))
    coverage_score = len(query_terms & content_terms) / len(query_terms) if query_terms else 0.0
    option_coverage = len(option_terms & content_terms) / len(option_terms) if option_terms else 0.0
    bm25_score = float(row["bm25_score"])
    lexical_score = 1.0 / (1.0 + abs(bm25_score))
    if not hybrid:
        exact_boost = 0.2 if query.casefold() in str(row["content"]).casefold() else 0.0
        score = lexical_score + coverage_score + (0.2 * option_coverage) + exact_boost + recency_score
        return RankedMemory(
            id=str(row["id"]),
            content=str(row["content"]),
            score=score,
            created_at=str(row["created_at"]),
            content_hash=str(row["content_hash"]),
        )
    lexical_rrf = 0.6 / (rrf_k + lexical_rank) if lexical_rank is not None else 0.0
    vector_rrf = 0.4 / (rrf_k + vector_rank) if vector_rank is not None else 0.0
    exact_boost = 0.04 if query.casefold() in str(row["content"]).casefold() else 0.0
    return RankedMemory(
        id=str(row["id"]),
        content=str(row["content"]),
        score=(
            lexical_rrf
            + vector_rrf
            + (0.35 * vector_similarity)
            + (0.1 * coverage_score)
            + (0.02 * option_coverage)
            + exact_boost
            + recency_score
        ),
        created_at=str(row["created_at"]),
        content_hash=str(row["content_hash"]),
    )


def _candidate_ranks(
    rows: list[sqlite3.Row], query: str, options: list[str]
) -> dict[str, int]:
    """Order lexical candidates before reciprocal-rank fusion."""

    query_terms = _canonical_terms(query)
    option_terms = _canonical_terms(" ".join(options))

    def score(row: sqlite3.Row) -> tuple[float, str, str]:
        content = str(row["content"])
        content_terms = _canonical_terms(content)
        coverage = len(query_terms & content_terms) / len(query_terms) if query_terms else 0.0
        option_coverage = len(option_terms & content_terms) / len(option_terms) if option_terms else 0.0
        bm25_score = float(row["bm25_score"])
        lexical_score = 1.0 / (1.0 + abs(bm25_score))
        exact_boost = 0.2 if query.casefold() in content.casefold() else 0.0
        return (
            -(lexical_score + coverage + (0.2 * option_coverage) + exact_boost),
            str(row["created_at"]),
            str(row["id"]),
        )

    ranked = sorted(rows, key=score)
    return {str(row["id"]): rank for rank, row in enumerate(ranked, start=1)}


def _is_historical_query(query: str) -> bool:
    normalized = query.casefold()
    markers = ("used to", "formerly", "previously", "historically", "in the past")
    return any(marker in normalized for marker in markers)


def _is_current_query(query: str) -> bool:
    normalized = query.casefold()
    markers = (
        "currently",
        "current",
        "latest",
        "most recent",
        "now",
        "today",
        "目前",
        "现在",
        "最新",
    )
    return any(marker in normalized for marker in markers)


def _recency_scores(rows: list[sqlite3.Row]) -> dict[str, float]:
    timestamps = {str(row["id"]): _created_at_seconds(str(row["created_at"])) for row in rows}
    valid = [timestamp for timestamp in timestamps.values() if timestamp is not None]
    if len(valid) < 2:
        return {memory_id: 0.0 for memory_id in timestamps}
    oldest = min(valid)
    span = max(valid) - oldest
    return {
        memory_id: 0.15 * ((timestamp - oldest) / span) if timestamp is not None else 0.0
        for memory_id, timestamp in timestamps.items()
    }


def _created_at_seconds(value: str) -> float | None:
    try:
        return datetime.fromisoformat(value).timestamp()
    except ValueError:
        return None


def _elapsed_ms(started: float) -> float:
    return (time.perf_counter() - started) * 1_000
