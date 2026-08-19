"""Tests for memoria.seed_procedures, including applies_when/do_not_apply_when flow."""
from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from memoria.config import Settings
from memoria.runtime import create_runtime_store
from memoria.seed_procedures import (
    parse_procedure_markdown,
    seed_procedures_from_dir,
    seed_procedures_from_text,
)


@pytest.fixture
def short_circuit_reranker():
    """Override the reranker to avoid loading the real model."""
    with patch("memoria.qwen.create_reranker") as mock:
        mock.return_value = None
        yield mock


SAMPLE_MD = """---
schema: memoria_procedural.v1
owner: user
trust_state: user_stated
---

## Procedure build-and-deploy
- Type: `procedure`
- Description: Build and deploy the application
- Applies when: `working on deployment`
- Do not apply when: `running tests`
- Confidence: `0.9`

Steps:
1. Run `npm run build`
2. Run `docker-compose up`

## Preference editor-tools
- Type: `preference`
- Description: Prefers VS Code for TypeScript
- Applies when: `*.ts files`
- Do not apply when: `*.md files`
- Sentiment: `positive`

## Routine standup
- Type: `routine`
- Description: Daily standup at 9:30 AM
- Trigger: `every weekday morning`
- Applies when: `working day`
"""


class TestParseProcedureMarkdown:
    def test_parses_three_blocks(self):
        results = parse_procedure_markdown(SAMPLE_MD)
        assert len(results) == 3

    def test_procedure_fields(self):
        results = parse_procedure_markdown(SAMPLE_MD)
        proc = next(r for r in results if r.proc_type == "procedure")
        assert proc.description == "Build and deploy the application"
        assert proc.applies_when == "working on deployment"
        assert proc.do_not_apply_when == "running tests"
        assert proc.confidence == 0.9
        assert len(proc.steps) == 2

    def test_preference_fields(self):
        results = parse_procedure_markdown(SAMPLE_MD)
        pref = next(r for r in results if r.proc_type == "preference")
        assert pref.description == "Prefers VS Code for TypeScript"
        assert pref.applies_when == "*.ts files"
        assert pref.do_not_apply_when == "*.md files"
        assert pref.sentiment == "positive"

    def test_routine_fields(self):
        results = parse_procedure_markdown(SAMPLE_MD)
        routine = next(r for r in results if r.proc_type == "routine")
        assert routine.description == "Daily standup at 9:30 AM"
        assert routine.applies_when == "working day"
        assert routine.trigger == "every weekday morning"

    def test_empty_input(self):
        assert parse_procedure_markdown("") == []
        assert parse_procedure_markdown("---\nkey: val\n---") == []


class TestSeedProceduresFromText:
    def test_seed_count(self, short_circuit_reranker):
        with tempfile.TemporaryDirectory() as tmp:
            settings = Settings(
                data_dir=Path(tmp),
                auth_scheme="none", api_key=None,
                retention_days=365, max_top_k=1000,
            )
            store = create_runtime_store(settings)
            count = seed_procedures_from_text(
                store, SAMPLE_MD, user_id="test-user",
            )
            assert count == 3

    def test_applies_when_stored(self, short_circuit_reranker):
        with tempfile.TemporaryDirectory() as tmp:
            settings = Settings(
                data_dir=Path(tmp),
                auth_scheme="none", api_key=None,
                retention_days=365, max_top_k=1000,
            )
            store = create_runtime_store(settings)
            seed_procedures_from_text(store, SAMPLE_MD, user_id="test-user")
            conn = store._connect()
            row = conn.execute(
                "SELECT applies_when, do_not_apply_when FROM procedural_memories WHERE entity=?",
                ("Build and deploy the application",),
            ).fetchone()
            assert row is not None
            assert row["applies_when"] == "working on deployment"
            assert row["do_not_apply_when"] == "running tests"
            conn.close()

    def test_do_not_apply_excludes_boost(self, short_circuit_reranker):
        with tempfile.TemporaryDirectory() as tmp:
            settings = Settings(
                data_dir=Path(tmp),
                auth_scheme="none", api_key=None,
                retention_days=365, max_top_k=1000,
                reranker_backend="local",
                reranker_model="models/cross-encoder-ms-marco-MiniLM-L-6-v2",
                reranker_candidate_limit=100,
            )
            store = create_runtime_store(settings)
            seed_procedures_from_text(store, SAMPLE_MD, user_id="test-user")

            # 查询 "running tests" → do_not_apply_when 匹配，应排除
            hits = store.search(query="running tests", options=[], user_id="test-user", top_k=5)
            seeded = [h for h in hits if h.id.startswith("seed_")]
            assert len(seeded) == 0, (
                f"do_not_apply_when='running tests' should exclude, got {len(seeded)}"
            )

            # 查询 "how to deploy" → applies_when 匹配，应包含
            hits = store.search(
                query="how to deploy the application", options=[], user_id="test-user", top_k=5,
            )
            seeded = [h for h in hits if h.id.startswith("seed_")]
            assert len(seeded) > 0, (
                "applies_when='working on deployment' should include, got 0"
            )


class TestSeedProceduresFromDir:
    def test_from_directory(self, short_circuit_reranker):
        with tempfile.TemporaryDirectory() as tmp:
            md_dir = Path(tmp) / "procedures"
            md_dir.mkdir()
            (md_dir / "01-test.md").write_text(SAMPLE_MD, encoding="utf-8")

            settings = Settings(
                data_dir=Path(tmp) / "data",
                auth_scheme="none", api_key=None,
                retention_days=365, max_top_k=1000,
            )
            store = create_runtime_store(settings)
            count = seed_procedures_from_dir(store, md_dir, user_id="test-user")
            assert count == 3

    def test_missing_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            settings = Settings(
                data_dir=Path(tmp) / "data",
                auth_scheme="none", api_key=None,
                retention_days=365, max_top_k=1000,
            )
            store = create_runtime_store(settings)
            with pytest.raises(NotADirectoryError):
                seed_procedures_from_dir(store, Path("/nonexistent"), user_id="test-user")


class TestMigration:
    """Verify that the ALTER TABLE migration adds columns to old databases."""

    def test_migration_adds_columns(self, short_circuit_reranker):
        import sqlite3
        from datetime import datetime, timezone

        # 创建一个旧版本数据库（无 applies_when/do_not_apply_when 列）
        with tempfile.TemporaryDirectory() as tmp:
            db_path = Path(tmp) / "memoria.sqlite3"
            conn = sqlite3.connect(str(db_path))
            conn.execute("PRAGMA foreign_keys = ON")
            conn.execute("PRAGMA journal_mode = WAL")
            # 创建旧版 procedural_memories 表（无新列）
            conn.execute("""
                CREATE TABLE IF NOT EXISTS procedural_memories (
                    memory_id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    proc_type TEXT NOT NULL,
                    entity TEXT NOT NULL,
                    trigger_text TEXT DEFAULT '',
                    statement TEXT NOT NULL,
                    sentiment TEXT DEFAULT 'neutral',
                    confidence REAL DEFAULT 0.7,
                    created_at TEXT NOT NULL
                )
            """)
            conn.commit()
            conn.close()

            # 现在用 store 打开该数据库（应触发迁移）
            settings = Settings(
                data_dir=Path(tmp),
                auth_scheme="none", api_key=None,
                retention_days=365, max_top_k=1000,
            )
            store = create_runtime_store(settings)
            store.initialize()

            # 验证迁移已添加列
            conn2 = store._connect()
            cols = {
                row[1]
                for row in conn2.execute(
                    "SELECT * FROM pragma_table_info('procedural_memories')"
                ).fetchall()
            }
            assert "applies_when" in cols, "migration should add applies_when"
            assert "do_not_apply_when" in cols, "migration should add do_not_apply_when"
            conn2.close()