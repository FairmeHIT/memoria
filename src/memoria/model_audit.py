"""Body-free audit records for remote model calls."""
from __future__ import annotations

import sqlite3
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path


@dataclass(frozen=True, slots=True)
class ModelCallAudit:
    operation: str
    provider: str
    model: str
    input_count: int
    prompt_tokens: int | None
    total_tokens: int | None
    attempts: int
    elapsed_ms: float
    success: bool
    error_kind: str | None


class SqliteModelAuditRecorder:
    """Best-effort persistence of aggregate model call diagnostics."""

    def __init__(self, data_dir: Path) -> None:
        self._database_path = data_dir / "memoria.sqlite3"

    def record(self, audit: ModelCallAudit) -> None:
        connection = sqlite3.connect(self._database_path, timeout=0.25)
        try:
            connection.execute(
                """
                INSERT INTO model_audit (
                    audit_id, operation, provider, model, input_count,
                    prompt_tokens, total_tokens, attempts, elapsed_ms,
                    success, error_kind, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    uuid.uuid4().hex,
                    audit.operation,
                    audit.provider,
                    audit.model,
                    audit.input_count,
                    audit.prompt_tokens,
                    audit.total_tokens,
                    audit.attempts,
                    round(audit.elapsed_ms, 3),
                    int(audit.success),
                    audit.error_kind,
                    datetime.now(UTC).isoformat(),
                ),
            )
            connection.commit()
        except sqlite3.Error:
            # Model diagnostics must never turn a successful Add/Search into an error.
            return
        finally:
            connection.close()
