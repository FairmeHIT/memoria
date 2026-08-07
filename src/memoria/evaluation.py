"""Offline retrieval metrics using the same Search implementation as the API."""
from __future__ import annotations

import argparse
import json
import math
from dataclasses import replace
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from statistics import fmean
from typing import Iterable, Sequence

from pydantic import BaseModel, ConfigDict, Field, field_validator

from memoria.config import Settings
from memoria.runtime import create_runtime_store
from memoria.store import MemoryStore


class EvaluationCase(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    query: str = Field(min_length=1, max_length=20_000)
    options: list[str] | None = Field(default=None, max_length=20)
    user_id: str = Field(min_length=1, max_length=512)
    relevant_ids: set[str] = Field(min_length=1, max_length=1_000)
    top_k: int = Field(default=100, ge=1, le=1_000)

    @field_validator("query", "user_id")
    @classmethod
    def text_must_not_be_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("value must not be blank")
        return value


def recall_at_k(retrieved: Sequence[str], relevant: set[str], k: int) -> float:
    if not relevant:
        return 0.0
    return len(set(retrieved[:k]) & relevant) / len(relevant)


def reciprocal_rank(retrieved: Sequence[str], relevant: set[str]) -> float:
    for index, memory_id in enumerate(retrieved, start=1):
        if memory_id in relevant:
            return 1.0 / index
    return 0.0


def ndcg_at_k(retrieved: Sequence[str], relevant: set[str], k: int) -> float:
    if not relevant:
        return 0.0
    dcg = sum(
        1.0 / math.log2(index + 2)
        for index, memory_id in enumerate(retrieved[:k])
        if memory_id in relevant
    )
    ideal_count = min(k, len(relevant))
    ideal_dcg = sum(1.0 / math.log2(index + 2) for index in range(ideal_count))
    return dcg / ideal_dcg if ideal_dcg else 0.0


def evaluate_cases(
    store: MemoryStore, cases: Iterable[EvaluationCase], *, workers: int = 1
) -> dict[str, float | int]:
    """Evaluate independently scoped queries with a bounded local worker pool."""

    if not 1 <= workers <= 16:
        raise ValueError("workers must be between 1 and 16")
    case_list = list(cases)
    if workers == 1:
        retrieved_ids = [_retrieve_ids(store, case) for case in case_list]
    else:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            retrieved_ids = list(executor.map(lambda case: _retrieve_ids(store, case), case_list))
    recalls: list[float] = []
    reciprocal_ranks: list[float] = []
    ndcgs: list[float] = []
    for case, ids in zip(case_list, retrieved_ids, strict=True):
        recalls.append(recall_at_k(ids, case.relevant_ids, case.top_k))
        reciprocal_ranks.append(reciprocal_rank(ids, case.relevant_ids))
        ndcgs.append(ndcg_at_k(ids, case.relevant_ids, case.top_k))
    return {
        "samples": len(case_list),
        "recall_at_k": round(fmean(recalls), 6) if recalls else 0.0,
        "mrr": round(fmean(reciprocal_ranks), 6) if reciprocal_ranks else 0.0,
        "ndcg_at_k": round(fmean(ndcgs), 6) if ndcgs else 0.0,
    }


def _retrieve_ids(store: MemoryStore, case: EvaluationCase) -> list[str]:
    hits = store.search(
        query=case.query,
        options=case.options,
        user_id=case.user_id,
        top_k=case.top_k,
    )
    return [hit.id for hit in hits]


def load_cases(path: Path) -> list[EvaluationCase]:
    cases: list[EvaluationCase] = []
    with path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            try:
                cases.append(EvaluationCase.model_validate(json.loads(line)))
            except (ValueError, TypeError) as error:
                raise ValueError(f"invalid evaluation case on line {line_number}") from error
    return cases


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate memoria retrieval against labeled JSONL cases")
    parser.add_argument("--data", type=Path, required=True, help="JSONL file with query and relevant_ids")
    parser.add_argument("--data-dir", type=Path, default=Path("./data"))
    parser.add_argument("--workers", type=int, default=1, help="Independent local search workers (1-16)")
    parser.add_argument("--report-out", type=Path, help="Optional path for aggregate metrics JSON")
    args = parser.parse_args()

    settings = replace(
        Settings.from_env(),
        data_dir=args.data_dir,
        auth_scheme="none",
        api_key=None,
        max_top_k=1_000,
    )
    store = create_runtime_store(settings)
    output = json.dumps(
        evaluate_cases(store, load_cases(args.data), workers=args.workers),
        separators=(",", ":"),
    )
    if args.report_out is not None:
        args.report_out.write_text(f"{output}\n", encoding="utf-8")
    print(output)
