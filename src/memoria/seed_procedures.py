"""
seed_procedures.py — Seed procedural_memories from Markdown procedure files.

Inspired by Memorax's `.repo_memory/procedure-memory/*.md` format.

Format:
  Each .md file can contain one or more procedure blocks:

  ---
  schema: memoria_procedural.v1
  owner: user
  trust_state: user_stated
  ---

  ## Procedure <id>
  - Type: `procedure|preference|routine`
  - Description: <text>
  - Applies when: <condition>
  - Do not apply when: <condition>
  - Sentiment: `positive|negative|neutral` (default: neutral)
  - Confidence: `0.0-1.0` (default: 0.8)
  - Trigger: <optional trigger text>

  Optional body (for procedures):
  1. Step one
  2. Step two
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

from memoria.procedural import ProceduralMemory
from memoria.store import MemoryStore


@dataclass
class SeedProcedure:
    """一条从 Markdown 文件解析出的程序性记忆种子。"""

    proc_type: str  # procedure / preference / routine
    description: str
    applies_when: str = ""
    do_not_apply_when: str = ""
    sentiment: str = "neutral"
    confidence: float = 0.8
    trigger: str = ""
    steps: list[str] = field(default_factory=list)


def parse_procedure_markdown(text: str, source: str = "") -> list[SeedProcedure]:
    """解析 Memorax 风格的 procedure Markdown 文本。

    Returns:
        解析出的程序性记忆种子列表。
    """
    results: list[SeedProcedure] = []
    # 按 ## 标题分割块
    blocks = re.split(r"(?=^##\s+)", text, flags=re.MULTILINE)
    for block in blocks:
        block = block.strip()
        if not block or block.startswith("---"):
            continue
        proc = _parse_procedure_block(block, source)
        if proc:
            results.append(proc)
    return results


def _parse_procedure_block(block: str, source: str) -> SeedProcedure | None:
    """解析一个 ## Procedure 块。"""
    # 提取标题行
    title_match = re.match(r"^##\s+(Procedure|Preference|Routine)\s+(\S+)", block, re.IGNORECASE)
    if not title_match:
        return None
    type_label = title_match.group(1).lower()  # procedure / preference / routine
    proc_id = title_match.group(2)

    # 提取字段
    type_map = {"procedure": "procedure", "preference": "preference", "routine": "routine"}
    proc_type = type_map.get(type_label, "procedure")

    # 字段提取
    def _field(label: str) -> str:
        m = re.search(
            rf"^- {re.escape(label)}:\s*`([^`]*)`",
            block,
            re.IGNORECASE | re.MULTILINE,
        )
        if m:
            return m.group(1).strip()
        m = re.search(
            rf"^- {re.escape(label)}:\s*(.+?)$",
            block,
            re.IGNORECASE | re.MULTILINE,
        )
        return m.group(1).strip() if m else ""

    description = _field("Description") or _field("description")
    if not description:
        description = proc_id

    applies_when = _field("Applies when") or _field("applies when")
    do_not_apply_when = _field("Do not apply when") or _field("do not apply when")
    sentiment_raw = _field("Sentiment").lower()
    sentiment = sentiment_raw if sentiment_raw in ("positive", "negative", "neutral") else "neutral"
    confidence_raw = _field("Confidence")
    try:
        confidence = float(confidence_raw) if confidence_raw else 0.8
    except ValueError:
        confidence = 0.8
    trigger = _field("Trigger") or _field("trigger")

    # 提取步骤（编号列表）
    steps: list[str] = []
    for line in block.split("\n"):
        step_match = re.match(r"^\s*\d+[.．、]\s+(.+)$", line.strip())
        if step_match:
            steps.append(step_match.group(1).strip())

    return SeedProcedure(
        proc_type=proc_type,
        description=description,
        applies_when=applies_when,
        do_not_apply_when=do_not_apply_when,
        sentiment=sentiment,
        confidence=confidence,
        trigger=trigger,
        steps=steps,
    )


def seed_procedures_from_dir(
    store: MemoryStore,
    directory: Path,
    *,
    user_id: str = "seed-user",
    session_id: str = "seed-session",
) -> int:
    """从目录中的所有 .md 文件读取并注入程序性记忆种子。

    Returns:
        成功注入的种子数量。
    """
    if not directory.is_dir():
        raise NotADirectoryError(f"seed-procedures directory not found: {directory}")

    md_files = sorted(directory.glob("*.md"))

    seed_count = 0
    for md_file in md_files:
        text = md_file.read_text(encoding="utf-8")
        parsed = parse_procedure_markdown(text, source=str(md_file))
        for proc in parsed:
            _write_seed_procedure(store, proc, user_id=user_id, session_id=session_id)
            seed_count += 1
    return seed_count


def seed_procedures_from_text(
    store: MemoryStore,
    text: str,
    *,
    user_id: str = "seed-user",
    session_id: str = "seed-session",
    source: str = "inline",
) -> int:
    """从文本解析并注入程序性记忆种子。

    Returns:
        成功注入的种子数量。
    """
    parsed = parse_procedure_markdown(text, source=source)
    seed_count = 0
    for proc in parsed:
        _write_seed_procedure(store, proc, user_id=user_id, session_id=session_id)
        seed_count += 1
    return seed_count


def _write_seed_procedure(
    store: MemoryStore,
    seed: SeedProcedure,
    *,
    user_id: str,
    session_id: str,
) -> None:
    """将一条种子写入 store 的 procedural_memories 表 + 对应的回忆行。"""
    import sqlite3
    from datetime import datetime, timezone

    # 构建记忆 ID
    unique = f"{user_id}:{seed.proc_type}:{seed.description}"
    memory_id = f"seed_{hashlib.sha256(unique.encode()).hexdigest()[:20]}"
    now = datetime.now(timezone.utc).isoformat()
    content_hash = hashlib.sha256(seed.description.encode()).hexdigest()

    # 构建陈述句
    if seed.steps:
        statement = "; ".join(seed.steps)
    else:
        statement = seed.description

    conn = store._connect()
    try:
        # 临时禁用 FK 约束，绕过 add_requests → messages → memories 链
        conn.execute("PRAGMA foreign_keys = OFF")
        # 创建 add_request 行
        conn.execute(
            "INSERT OR IGNORE INTO add_requests (request_id, request_hash, user_id, session_id, committed_at) VALUES (?, ?, ?, ?, ?)",
            (memory_id, content_hash, user_id, session_id, now),
        )
        # 创建 message 行
        conn.execute(
            "INSERT OR IGNORE INTO messages (id, add_request_id, user_id, session_id, sequence, role, content, created_at, content_hash) VALUES (?, ?, ?, ?, 0, ?, ?, ?, ?)",
            (memory_id, memory_id, user_id, session_id, "user", seed.description, now, content_hash),
        )
        # 创建 memories 行
        conn.execute(
            "INSERT OR IGNORE INTO memories (id, message_id, user_id, session_id, content, created_at, content_hash) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (memory_id, memory_id, user_id, session_id, seed.description, now, content_hash),
        )
        # 写入 FTS 表
        conn.execute(
            "INSERT OR IGNORE INTO memories_fts (memory_id, content) VALUES (?, ?)",
            (memory_id, seed.description),
        )
        conn.execute(
            "INSERT OR IGNORE INTO memories_fts_porter (memory_id, content) VALUES (?, ?)",
            (memory_id, seed.description),
        )
        # 重新启用 FK
        conn.execute("PRAGMA foreign_keys = ON")
        # 写入程序性记忆
        conn.execute(
            "INSERT OR REPLACE INTO procedural_memories (memory_id, user_id, proc_type, entity, trigger_text, statement, sentiment, confidence, applies_when, do_not_apply_when, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (memory_id, user_id, seed.proc_type, seed.description[:200], seed.trigger[:200], statement[:500], seed.sentiment, seed.confidence, seed.applies_when[:500], seed.do_not_apply_when[:500], now),
        )
        conn.commit()
    finally:
        conn.close()


def main() -> None:
    """CLI entry point: memoria-seed-procedures"""
    import argparse

    from memoria.config import Settings
    from memoria.runtime import create_runtime_store

    parser = argparse.ArgumentParser(
        description="Seed procedural memories from Markdown procedure files"
    )
    parser.add_argument(
        "--input", type=Path, required=True,
        help="Directory containing .md procedure files, or a single .md file",
    )
    parser.add_argument("--data-dir", type=Path, default=Path("./data"))
    parser.add_argument("--user-id", default="seed-user", help="User ID for seeded memories")
    args = parser.parse_args()

    settings = Settings(
        data_dir=args.data_dir,
        auth_scheme="none",
        api_key=None,
        retention_days=365,
        max_top_k=1_000,
    )
    store = create_runtime_store(settings)

    input_path = args.input
    if input_path.is_file():
        text = input_path.read_text(encoding="utf-8")
        count = seed_procedures_from_text(
            store, text, user_id=args.user_id, source=str(input_path),
        )
    elif input_path.is_dir():
        count = seed_procedures_from_dir(
            store, input_path, user_id=args.user_id,
        )
    else:
        parser.error(f"input path not found: {input_path}")

    import json
    print(json.dumps({"seeded": count}, separators=(",", ":")))