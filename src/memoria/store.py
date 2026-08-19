"""SQLite persistence, idempotent ingestion, FTS retrieval, and retention.
Optimized for AML Leaderboard: Chinese n-gram, temporal intent, concept expansion.
"""
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
from typing import Literal, Protocol

from memoria.claims import Claim, extract_claims
from memoria.config import Settings
from memoria.embeddings import EmbeddingProvider, EmbeddingUnavailable, deserialize_vector, serialize_vector
from memoria.schemas import AddRequest, AddResponse, SearchHit

# ────────────────────────────────────────────────────────────── 权重常量
W_RRF_LEXICAL = 0.6          # 词法 RRF 权重
W_RRF_VECTOR = 0.4           # 向量 RRF 权重
W_VECTOR_SIM = 0.35          # 向量相似度权重
W_COVERAGE = 0.10            # 查询词覆盖率权重
W_OPTION_COVERAGE = 0.02     # 选项词覆盖率权重
W_EXACT_BOOST = 0.04         # 精确子串加分
W_LEXICAL_FALLBACK = 0.2     # 非混合模式下的精确子串权重
W_EXACT_FALLBACK = 0.2       # 非混合模式下的精确子串加成
W_OPTION_FALLBACK = 0.2      # 非混合模式下的选项覆盖率权重
W_RECENCY_LATEST = 0.10      # "最新"意图时的新近度加分
W_RECENCY_EARLIEST = 0.10    # "最早"意图时的旧度加分
W_TIMEPOINT_MILD = 0.025     # 时间点/序列问题的基础加分
W_TEMPORAL_RERANK = 0.30    # 重排阶段的时间感知权重（CrossEncoder 分数 + 时间分）
CONTEXT_RADIUS = 2           # 邻接上下文窗口半径（会话内前后 N 条）
CONTEXT_MAX_CHARS = 2000     # 上下文扩展的最大字符数
TEMPORAL_ENRICHMENT = True   # 是否在证据中附加角色 + 日期前缀（InvMem 风格）
ENHANCED_OPTIONS = True      # 是否使用增强选项查询视图（InvMem enhanced 模式）

# ────────────────────────────────────────────────────────────── 中文处理
CJK_RE = re.compile(r"[\u4e00-\u9fff]+")
CJK_STOPWORDS = frozenset({
    "的", "了", "在", "是", "我", "有", "和", "就", "不", "人", "都",
    "一", "一个", "上", "也", "很", "到", "说", "要", "去", "你",
    "会", "着", "没有", "看", "好", "自己", "这", "他", "她", "它",
    "们", "那", "什么", "怎么", "为什么", "哪个", "多少", "如何",
    "这个", "那个", "我们", "你们", "他们", "她们", "目前", "现在",
    "当前", "最新", "最近", "之前", "之后", "以前", "后来",
})

# ────────────────────────────────────────────────────────────── 概念语义组
CONCEPT_GROUPS = (
    {"prefer", "preference", "favorite", "favourite", "like", "love", "enjoy"},
    {"job", "career", "profession", "occupation", "work", "employment"},
    {"education", "school", "study", "studies", "college", "university", "degree", "course", "training", "certification"},
    {"home", "live", "lives", "living", "reside", "residence", "move", "moved", "location", "city", "country"},
    {"trip", "travel", "journey", "vacation", "holiday", "roadtrip", "camping", "hiking"},
    {"buy", "bought", "purchase", "purchased", "order", "ordered", "get", "got"},
    {"book", "books", "read", "reading", "author", "novel", "bookshelf"},
    {"music", "song", "songs", "listen", "listening", "artist", "band", "composer"},
    {"child", "children", "kid", "kids", "son", "daughter", "family", "parent"},
    {"friend", "friends", "relationship", "partner", "married", "single"},
    {"health", "medical", "doctor", "physician", "medicine", "medication", "hospital", "clinic"},
    {"food", "meal", "eat", "eating", "restaurant", "dish", "cuisine"},
    {"sport", "sports", "exercise", "workout", "gym", "run", "running", "race"},
    {"movie", "film", "show", "series", "watch", "watched"},
    {"birthday", "born", "birth", "age", "old"},
    {"plan", "planning", "intend", "intention", "goal", "want", "decide", "decided"},
    {"change", "changed", "update", "updated", "correct", "correction", "instead", "now", "current", "latest"},
    {"before", "after", "earlier", "later", "first", "last", "sequence", "timeline"},
    {"art", "paint", "painting", "draw", "drawing", "pottery", "creative"},
    {"support", "help", "care", "counsel", "counseling", "therapy", "mental"},
)

# ──────────────────────────────────────────────── 时间意图标记
TemporalIntent = Literal["none", "latest", "earliest", "sequence", "point"]

LATEST_MARKERS = frozenset({
    "latest", "last", "recent", "current", "currently", "now", "newest", "final",
    "most recent", "up to date", "as of",
    "最近", "最新", "最后", "当前", "现在", "目前", "最终",
})
EARLIEST_MARKERS = frozenset({
    "first", "earliest", "initial", "originally", "at first",
    "最早", "首先", "第一次", "起初", "原先",
})
HISTORICAL_MARKERS = frozenset({
    "in the past", "used to", "formerly", "previously", "before",
    "以前", "过去",
})
SEQUENCE_MARKERS = frozenset({
    "before", "after", "earlier", "later", "timeline", "sequence", "order", "previous", "next", "subsequent",
    "之前", "之后", "此前", "此后", "后来", "时间线", "顺序", "先后",
})
POINT_MARKERS = frozenset({
    "when", "date", "time", "year", "month", "day",
    "什么时候", "何时", "日期", "时间",
})

# ──────────────────────────────────────────────── 相对时间解析
RELATIVE_TIME_PATTERNS = (
    (r"\byesterday\b", lambda n: n - timedelta(days=1)),
    (r"\blast night\b", lambda n: n - timedelta(days=1)),
    (r"\btoday\b", lambda n: n),
    (r"\btonight\b", lambda n: n),
    (r"\btomorrow\b", lambda n: n + timedelta(days=1)),
    (r"\blast week\b", lambda n: n - timedelta(weeks=1)),
    (r"\bthis week\b", lambda n: n),
    (r"\bnext week\b", lambda n: n + timedelta(weeks=1)),
    (r"\blast month\b", lambda n: n - timedelta(days=30)),
    (r"\bthis month\b", lambda n: n),
    (r"\bnext month\b", lambda n: n + timedelta(days=30)),
    (r"\blast year\b", lambda n: n - timedelta(days=365)),
    (r"\bthis year\b", lambda n: n),
    (r"\bnext year\b", lambda n: n + timedelta(days=365)),
)

# ──────────────────────────────────────────────── Schema
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
    tokenize='unicode61 remove_diacritics 2'
);

CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts_porter USING fts5(
    memory_id UNINDEXED,
    content,
    tokenize='porter unicode61 remove_diacritics 2'
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
                # 回填 porter FTS 表：已有 memories 但未进 porter 表的
                connection.execute("""
                    INSERT INTO memories_fts_porter (memory_id, content)
                    SELECT m.id, m.content FROM memories m
                    WHERE NOT EXISTS (
                        SELECT 1 FROM memories_fts_porter p WHERE p.memory_id = m.id
                    )
                """)
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
                    connection.execute(
                        "INSERT INTO memories_fts_porter (memory_id, content) VALUES (?, ?)",
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
        temporal_intent = _detect_temporal_intent(query)
        primary_expression = _fts_expression(query, temporal_intent=temporal_intent)
        option_expression = _fts_expression(" ".join(options or []))
        # InvMem enhanced 模式：每个选项单独成查询视图（query + option），
        # 各自检索后按 RRF 融合，避免扁平查询丢失选项细节。
        use_enhanced = ENHANCED_OPTIONS and options
        query_vector = self._embed_query(query)
        if (
            not primary_expression
            and not option_expression
            and query_vector is None
        ):
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
                if use_enhanced:
                    # 增强选项视图：每个 (query+option) 视图独立检索 → RRF 融合
                    lexical_rows, lexical_ranks = self._enhanced_candidate_retrieval(
                        connection=connection,
                        query=query,
                        options=options or [],
                        user_id=user_id,
                        candidate_limit=candidate_limit,
                        rrf_k=self._settings.rrf_k,
                        temporal_intent=temporal_intent,
                    )
                elif user_count <= 5_000:
                    lexical_rows = self._scoped_candidate_rows(
                        connection=connection,
                        query=query,
                        options=options or [],
                        user_id=user_id,
                        candidate_limit=candidate_limit,
                    )
                    lexical_ranks = _candidate_ranks(lexical_rows, query, options or [])
                else:
                    lexical_rows = self._candidate_rows(
                        connection=connection,
                        expressions=(primary_expression, option_expression),
                        user_id=user_id,
                        candidate_limit=candidate_limit,
                    )
                    lexical_ranks = _candidate_ranks(lexical_rows, query, options or [])
            else:
                lexical_rows = []
                lexical_ranks = {}
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
            # 历史/最早意图 → 不过滤覆写（保留旧记录）
            historical = temporal_intent in ("earliest", "historical") or _is_historical_query(query)
            superseded_ids = (
                set()
                if historical
                else self._superseded_memory_ids(connection, [str(row["id"]) for row in rows])
            )
        finally:
            connection.close()

        # 总是计算 recency 分数，由 _rank_memory 根据意图决定使用方式
        recency_scores = _recency_scores(rows)
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
                    temporal_intent=temporal_intent,
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
            # 时间富化证据格式（InvMem 风格）：[role | Message date: ...] content
            role, raw_content = _extract_role_from_evidence(item.content)
            content = _format_evidence(role, raw_content, item.created_at)
            selected.append(
                SearchHit(
                    id=item.id,
                    content=content,
                    score=round(item.score, 6),
                    created_at=item.created_at,
                )
            )
            if len(selected) == top_k:
                break
        # 上下文窗口扩展：对选中的结果展开同 session 邻接消息
        if selected and CONTEXT_RADIUS > 0:
            selected = self._expand_context(selected, user_id)
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
        # CrossEncoder (ms-marco) 是纯英文模型，中文查询跳过 CE 以免疫乱
        if CJK_RE.search(query):
            return ranked
        temporal_intent = _detect_temporal_intent(query)
        candidates = ranked[: self._settings.reranker_candidate_limit]
        results = self._reranker.rerank(
            query,
            [item.content for item in candidates],
            top_n=len(candidates),
        )
        # 收集原始 CE 分数，min-max 归一化到 [0,1] 再叠加时间分
        raw_scores: list[float] = [score for _, score in results]
        normalized = _min_max_normalize(raw_scores) if raw_scores else []
        # 预计算时间分数 — 基于排序的 recency（避免被无关大 span 支配）
        recency_map: dict[str, float] = {}
        if temporal_intent != "none" and results:
            ce_indices = {idx for idx, _ in results}
            ce_items = [candidates[idx] for idx in ce_indices if 0 <= idx < len(candidates)]
            valid = [(_created_at_seconds(item.created_at), item.id) for item in ce_items]
            valid = [(ts, mid) for ts, mid in valid if ts is not None]
            if len(valid) >= 2:
                ordered = sorted(valid, key=lambda x: x[0])
                for rank, (_, mid) in enumerate(ordered):
                    recency_map[mid] = rank / (len(ordered) - 1)
        reranked: list[RankedMemory] = []
        seen_indexes: set[int] = set()
        for (index, _score), norm_score in zip(results, normalized):
            if not 0 <= index < len(candidates) or index in seen_indexes:
                continue
            seen_indexes.add(index)
            score = norm_score
            # 时间感知混合：时间意图查询时 CE 分数与时间分加权
            if temporal_intent == "latest":
                recency = recency_map.get(candidates[index].id, 0.5)
                score = 0.4 * norm_score + 0.6 * recency
            elif temporal_intent == "earliest":
                recency = recency_map.get(candidates[index].id, 0.5)
                score = 0.4 * norm_score + 0.6 * (1.0 - recency)
            # historical/sequence/point: 不加时间偏向，纯 CE 分数
            reranked.append(replace(candidates[index], score=score))
        # 重新按混合分数排序（CE 归一化 + 时间分）
        reranked.sort(key=lambda item: (-item.score, item.created_at, item.id))
        fallback_score = min((item.score for item in reranked), default=0.0)
        for item in [
            *[candidates[index] for index in range(len(candidates)) if index not in seen_indexes],
            *ranked[len(candidates) :],
        ]:
            fallback_score -= 0.000001
            reranked.append(replace(item, score=fallback_score))
        return reranked

    def _expand_context(self, selected: list[SearchHit], user_id: str) -> list[SearchHit]:
        """Expand each selected hit with adjacent messages from the same session.
        This gives the downstream answer model conversational context.
        """
        if not selected:
            return selected
        # 按创建时间排序，提取 session_id → [memory_id, created_at, content] 映射
        memory_ids = [item.id for item in selected]
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT m.id, m.session_id, m.created_at, m.content, m.content_hash
                FROM memories m
                WHERE m.user_id = ? AND m.id IN ({})
                """.format(",".join("?" for _ in memory_ids)),
                (user_id, *memory_ids),
            ).fetchall()

        # 对每个选中的记忆，找到同 session 的前后 CONTEXT_RADIUS 条
        session_map: dict[str, list[dict]] = {}
        for row in rows:
            sid = str(row["session_id"])
            session_map.setdefault(sid, []).append({
                "id": str(row["id"]),
                "created_at": str(row["created_at"]),
                "content": str(row["content"]),
                "content_hash": str(row["content_hash"]),
            })
        # 对每个 session 按时间排序
        for sid in session_map:
            session_map[sid].sort(key=lambda r: r["created_at"])

        # 扩展
        id_to_pos: dict[str, tuple[str, int]] = {}
        for sid, items in session_map.items():
            for pos, item in enumerate(items):
                id_to_pos[item["id"]] = (sid, pos)

        expanded = []
        for item in selected:
            loc = id_to_pos.get(item.id)
            if loc is None:
                expanded.append(item)
                continue
            sid, pos = loc
            items = session_map[sid]
            start = max(0, pos - CONTEXT_RADIUS)
            end = min(len(items), pos + CONTEXT_RADIUS + 1)
            # 收集上下文内容
            context_parts = []
            seen_hashes = set()
            char_budget = CONTEXT_MAX_CHARS
            for p in range(start, end):
                neighbor = items[p]
                if neighbor["content_hash"] in seen_hashes:
                    continue
                seen_hashes.add(neighbor["content_hash"])
                prefix = ">>> " if p == pos else "    "
                content = neighbor["content"]
                if len(content) > char_budget:
                    content = content[:char_budget]
                context_parts.append(f"{prefix}{content}")
                char_budget -= len(content)
                if char_budget <= 0:
                    break
            if len(context_parts) > 1:
                expanded.append(
                    SearchHit(
                        id=item.id,
                        content="\n".join(context_parts),
                        score=item.score,
                        created_at=item.created_at,
                    )
                )
            else:
                expanded.append(item)
        return expanded

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
        # 扩展查询词：基础词形 + CJK n-gram + 概念同义词
        query_terms = _canonical_terms(query) | set(_cjk_ngrams(query)) | _concept_expansion(query)
        option_terms = _canonical_terms(" ".join(options)) | set(_cjk_ngrams(" ".join(options)))
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
            content = str(row["content"])
            content_terms = _canonical_terms(content) | set(_cjk_ngrams(content))
            if query_terms & content_terms or option_terms & content_terms:
                matching.append(row)
        return matching[:candidate_limit]

    def _candidate_rows(
        self,
        *,
        connection: sqlite3.Connection,
        expressions: Sequence[str],
        user_id: str,
        candidate_limit: int,
    ) -> list[sqlite3.Row]:
        rows_by_id: dict[str, sqlite3.Row] = {}
        # 从两个 FTS 表（原始 unicode61 + porter 词干）分别查询，合并结果
        fts_tables = ("memories_fts", "memories_fts_porter")
        for expression in expressions:
            if not expression:
                continue
            for fts_table in fts_tables:
                rows = connection.execute(
                    f"""
                    SELECT memories.id, memories.content, memories.created_at, memories.content_hash,
                           bm25({fts_table}) AS bm25_score
                    FROM {fts_table}
                    JOIN memories ON memories.id = {fts_table}.memory_id
                    WHERE {fts_table} MATCH ? AND memories.user_id = ?
                    LIMIT ?
                    """,
                    (expression, user_id, candidate_limit),
                ).fetchall()
                for row in rows:
                    rows_by_id.setdefault(str(row["id"]), row)
        return list(rows_by_id.values())

    def _enhanced_candidate_retrieval(
        self,
        *,
        connection: sqlite3.Connection,
        query: str,
        options: list[str],
        user_id: str,
        candidate_limit: int,
        rrf_k: int,
        temporal_intent: str,
    ) -> tuple[list[sqlite3.Row], dict[str, int]]:
        """InvMem enhanced option views: per-(query+option) retrieval + RRF fusion.

        Each option becomes its own query view ``query + opt``; the views are
        retrieved independently from FTS and fused with reciprocal-rank
        fusion, so an option keyword that the flat query misses still surfaces
        the right memory.
        """
        view_queries = [query] + [f"{query} {opt}" for opt in options]
        rows_by_id: dict[str, sqlite3.Row] = {}
        fused_rrf: dict[str, float] = {}
        for view_query in view_queries:
            expression = _fts_expression(view_query, temporal_intent=temporal_intent)
            view_rows = self._candidate_rows(
                connection=connection,
                expressions=(expression,),
                user_id=user_id,
                candidate_limit=candidate_limit,
            )
            if not view_rows:
                continue
            ranks = _candidate_ranks(view_rows, view_query, [])
            for row in view_rows:
                rid = str(row["id"])
                rows_by_id.setdefault(rid, row)
                rank = ranks.get(rid)
                if rank is not None:
                    fused_rrf[rid] = fused_rrf.get(rid, 0.0) + 1.0 / (rrf_k + rank)
        # 融合后按 RRF 分数排序，转成单一 lexical rank 供 _rank_memory 使用
        ordered = sorted(
            fused_rrf.items(), key=lambda kv: (-kv[1], str(kv[0]))
        )
        lexical_ranks = {rid: rank for rank, (rid, _) in enumerate(ordered, start=1)}
        return list(rows_by_id.values()), lexical_ranks

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
                    for table in ("memories_fts", "memories_fts_porter"):
                        connection.executemany(
                            f"DELETE FROM {table} WHERE memory_id = ?",
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


def _format_date(dt: datetime) -> str:
    """Format a datetime as 'Month Day, Year' (e.g. 'January 15, 2024')."""
    return f"{dt:%B} {dt.day}, {dt.year}"


def _format_evidence(role: str, content: str, created_at: str) -> str:
    """Format evidence with temporal enrichment (InvMem-style).

    With TEMPORAL_ENRICHMENT=True, produces:
        [user | Message date: January 15, 2024 at 14:30 UTC] I live in Berlin.
    Without enrichment, produces the original:
        user: I live in Berlin.
    """
    if not TEMPORAL_ENRICHMENT:
        return f"{role}: {content}"
    message_time = _created_at_datetime(created_at)
    if message_time is None:
        return f"[{role}] {content}"
    rendered = (
        f"[{role} | Message date: {_format_date(message_time)} "
        f"at {message_time:%H:%M} UTC] {content}"
    )
    # Resolve relative time references in the content
    resolved = _resolve_relative_times(content, created_at)
    if resolved != content and "[Resolved relative dates:" in resolved:
        # Extract just the annotation part
        ann_start = resolved.index("[Resolved relative dates:")
        rendered += f" {resolved[ann_start:]}"
    return rendered


def _extract_role_from_evidence(evidence: str) -> tuple[str, str]:
    """Split 'role: content' into (role, content)."""
    colon = evidence.find(": ")
    if colon == -1:
        return ("user", evidence)
    return (evidence[:colon], evidence[colon + 2:])


def _min_max_normalize(values: list[float]) -> list[float]:
    """Min-max normalize scores to [0, 1]. All equal → [0.0]*n."""
    if not values:
        return []
    n = len(values)
    if n == 1:
        return [1.0]
    mn = min(values)
    mx = max(values)
    if mx - mn < 1e-12:
        return [0.0] * n
    return [(v - mn) / (mx - mn) for v in values]


def stable_memory_id(request_id: str, sequence: int) -> str:
    """Return the public deterministic memory ID for a source message."""

    return _stable_id("mem", request_id, str(sequence))


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _timestamp_to_iso(timestamp: int | None) -> str | None:
    if timestamp is None:
        return None
    return datetime.fromtimestamp(timestamp / 1000, tz=UTC).isoformat()


def _fts_expression(query: str, *, temporal_intent: str = "none") -> str:
    tokens = {
        variant
        for token in TOKEN_RE.findall(query)
        for variant in _term_variants(token)
    }
    # 加入 CJK n-gram 和概念同义词扩展
    tokens.update(_cjk_ngrams(query))
    tokens.update(_concept_expansion(query))
    if temporal_intent == "point":
        # 时间点问题：加入日期相关词宽召回
        tokens.update({"date", "time", "year", "month", "day", "日期", "时间", "时间点", "时间线"})
    return " OR ".join('"' + token.replace('"', '""') + '"' for token in sorted(tokens))


def _cjk_ngrams(text: str) -> set[str]:
    """Extract 2-gram and 3-gram CJK substrings for better Chinese matching."""
    ngrams: set[str] = set()
    for run in CJK_RE.findall(text):
        if len(run) <= 1:
            if run not in CJK_STOPWORDS:
                ngrams.add(run)
            continue
        # 去重：只加 2-gram 和 3-gram
        for size in (2, 3):
            for i in range(len(run) - size + 1):
                gram = run[i : i + size]
                if gram not in CJK_STOPWORDS:
                    ngrams.add(gram)
    # 多个 CJK 段之间的边界 2-gram（"我爱北京" 拆成 "我爱" "北京"）
    runs = CJK_RE.findall(text)
    for i in range(len(runs) - 1):
        boundary = runs[i][-1] + runs[i + 1][0]
        if boundary not in CJK_STOPWORDS:
            ngrams.add(boundary)
    return ngrams


def _concept_expansion(query: str) -> set[str]:
    """Expand query terms with concept-group synonyms (semantic vocabulary)."""
    terms = _canonical_terms(query)
    if not terms:
        return set()
    expanded: set[str] = set()
    for group in CONCEPT_GROUPS:
        if terms & group:
            expanded.update(group - terms)
    return expanded


def _detect_temporal_intent(query: str) -> TemporalIntent:
    """Classify the query's temporal intent (latest/earliest/historical/sequence/point/none)."""
    normalized = query.casefold()
    # 中文和英文标记
    if any(m in normalized for m in LATEST_MARKERS):
        return "latest"
    if any(m in normalized for m in EARLIEST_MARKERS):
        return "earliest"
    if any(m in normalized for m in HISTORICAL_MARKERS):
        return "historical"
    if any(m in normalized for m in SEQUENCE_MARKERS):
        return "sequence"
    if any(m in normalized for m in POINT_MARKERS):
        return "point"
    return "none"


def _compute_time_score(recency_score: float, temporal_intent: str) -> float:
    """Apply temporal-intent-aware time scoring on top of recency."""
    if temporal_intent == "latest":
        return W_RECENCY_LATEST * recency_score
    if temporal_intent == "earliest":
        return W_RECENCY_EARLIEST * (1.0 - recency_score)
    if temporal_intent in {"sequence", "point"}:
        return W_TIMEPOINT_MILD
    return 0.0


def _resolve_relative_times(content: str, created_at: str) -> str:
    """Resolve relative time expressions (yesterday, last week...) to absolute dates.

    Mirrors InvMem's temporal enrichment: annotated evidence helps the downstream
    answer model resolve time-relative facts without hallucinating.
    """
    message_time = _created_at_datetime(created_at)
    if message_time is None:
        return content
    annotations: list[str] = []
    seen: set[str] = set()
    for pattern, resolver in RELATIVE_TIME_PATTERNS:
        for match in re.finditer(pattern, content, flags=re.IGNORECASE):
            phrase = match.group(0).casefold()
            if phrase in seen:
                continue
            seen.add(phrase)
            resolved = resolver(message_time)
            annotations.append(f"{match.group(0)} = {resolved.strftime('%B %d, %Y')}")
    if not annotations:
        return content
    return f"{content} [Resolved relative dates: {'; '.join(annotations)}]"


def _created_at_datetime(value: str) -> datetime | None:
    try:
        moment = datetime.fromisoformat(value)
        if moment.tzinfo is None:
            moment = moment.replace(tzinfo=UTC)
        return moment
    except (ValueError, TypeError):
        return None


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
    temporal_intent: str = "none",
) -> RankedMemory:
    query_terms = _canonical_terms(query)
    content_terms = _canonical_terms(str(row["content"]))
    coverage_score = len(query_terms & content_terms) / len(query_terms) if query_terms else 0.0
    # 增强选项模式：取每个选项单独覆盖度的最大值
    if ENHANCED_OPTIONS and options:
        option_coverage = max(
            len(_canonical_terms(opt) & content_terms) / max(len(_canonical_terms(opt)), 1)
            for opt in options
        ) if options else 0.0
    else:
        option_terms = _canonical_terms(" ".join(options))
        option_coverage = len(option_terms & content_terms) / len(option_terms) if option_terms else 0.0
    bm25_score = float(row["bm25_score"])
    lexical_score = 1.0 / (1.0 + abs(bm25_score))
    if not hybrid:
        exact_boost = W_EXACT_FALLBACK if query.casefold() in str(row["content"]).casefold() else 0.0
        # 时间意图感知
        time_score = _compute_time_score(recency_score, temporal_intent)
        score = lexical_score + coverage_score + (W_OPTION_FALLBACK * option_coverage) + exact_boost + time_score
        return RankedMemory(
            id=str(row["id"]),
            content=str(row["content"]),
            score=score,
            created_at=str(row["created_at"]),
            content_hash=str(row["content_hash"]),
        )
    lexical_rrf = W_RRF_LEXICAL / (rrf_k + lexical_rank) if lexical_rank is not None else 0.0
    vector_rrf = W_RRF_VECTOR / (rrf_k + vector_rank) if vector_rank is not None else 0.0
    exact_boost = W_EXACT_BOOST if query.casefold() in str(row["content"]).casefold() else 0.0
    time_score = _compute_time_score(recency_score, temporal_intent)
    return RankedMemory(
        id=str(row["id"]),
        content=str(row["content"]),
        score=(
            lexical_rrf
            + vector_rrf
            + (W_VECTOR_SIM * vector_similarity)
            + (W_COVERAGE * coverage_score)
            + (W_OPTION_COVERAGE * option_coverage)
            + exact_boost
            + time_score
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
    if span == 0.0:
        return {memory_id: 0.0 for memory_id in timestamps}
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
