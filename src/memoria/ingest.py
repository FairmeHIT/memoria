"""Benchmark-neutral JSONL importer for authorized local Add fixtures."""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, replace
from pathlib import Path

from pydantic import ValidationError

from memoria.config import Settings
from memoria.runtime import create_runtime_store
from memoria.schemas import AddRequest
from memoria.store import MemoryStore


@dataclass(frozen=True, slots=True)
class IngestResult:
    requests: int
    messages: int


def load_add_cases(path: Path) -> list[AddRequest]:
    cases: list[AddRequest] = []
    with path.open(encoding="utf-8") as source:
        lines = enumerate(source, start=1)
        for line_number, line in lines:
            if not line.strip():
                continue
            try:
                cases.append(AddRequest.model_validate(json.loads(line)))
            except (ValueError, TypeError, ValidationError) as error:
                raise ValueError(f"invalid Add case on line {line_number}") from error
    return cases


def ingest_file(store: MemoryStore, path: Path) -> IngestResult:
    cases = load_add_cases(path)
    for case in cases:
        store.add(case)
    return IngestResult(
        requests=len(cases),
        messages=sum(len(case.messages) for case in cases),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Import authorized Add JSONL fixtures into memoria")
    parser.add_argument("--data", type=Path, required=True, help="JSONL file containing Add requests")
    parser.add_argument("--data-dir", type=Path, default=Path("./data"))
    args = parser.parse_args()

    settings = replace(
        Settings.from_env(),
        data_dir=args.data_dir,
        auth_scheme="none",
        api_key=None,
        max_top_k=1_000,
    )
    store = create_runtime_store(settings)
    result = ingest_file(store, args.data)
    print(json.dumps({"requests": result.requests, "messages": result.messages}, separators=(",", ":")))
