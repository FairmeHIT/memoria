from __future__ import annotations

from memoria.model_audit import ModelCallAudit, SqliteModelAuditRecorder
from memoria.config import Settings
from memoria.store import MemoryStore


def test_model_audit_persists_only_aggregate_qwen_call_fields(tmp_path) -> None:
    settings = Settings(
        data_dir=tmp_path,
        auth_scheme="none",
        api_key=None,
        retention_days=30,
        max_top_k=100,
    )
    store = MemoryStore(settings)
    store.initialize()
    recorder = SqliteModelAuditRecorder(settings.data_dir)

    recorder.record(
        ModelCallAudit(
            operation="embedding",
            provider="qwen",
            model="text-embedding-v4",
            input_count=2,
            prompt_tokens=17,
            total_tokens=17,
            attempts=1,
            elapsed_ms=12.5,
            success=True,
            error_kind=None,
        )
    )

    connection = store._connect()
    try:
        row = connection.execute(
            "SELECT operation, provider, model, input_count, prompt_tokens, total_tokens, attempts, success, error_kind FROM model_audit"
        ).fetchone()
    finally:
        connection.close()

    assert tuple(row) == ("embedding", "qwen", "text-embedding-v4", 2, 17, 17, 1, 1, None)
