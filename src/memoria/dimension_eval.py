"""Dimension-specific retrieval evaluation for AML-style regression testing.

Run one-click after any code change to verify per-dimension scores:

    MEMORIA_AUTH_SCHEME=none memoria-dimension-eval

The scenarios are synthetic but each targets one AML evaluation dimension's
core retrieval mechanics. Scores are NOT a substitute for the full AML pipeline
(Store → Search → Answer → Eval with GPT-4o mini), but they catch regressions
and show whether a change helps or hurts a specific retrieval capability.
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from statistics import fmean
from typing import Any

from memoria.config import Settings
from memoria.evaluation import EvaluationCase, evaluate_cases
from memoria.runtime import create_runtime_store
from memoria.schemas import AddRequest
from memoria.store import MemoryStore, stable_memory_id


# ── Fixture helpers ──────────────────────────────────────────────────────────

_rid_counter: int = 0


def _rid(prefix: str) -> str:
    global _rid_counter
    _rid_counter += 1
    return f"dim:{prefix}:{_rid_counter}"


def _add(
    content: str,
    *,
    user_id: str = "dim-user",
    session_id: str = "dim-session",
    timestamp: int | None = None,
    rid: str | None = None,
) -> dict[str, Any]:
    """Build an AddRequest‑compatible dict for one message."""
    msg: dict[str, Any] = {"role": "user", "content": content}
    if timestamp is not None:
        msg["timestamp"] = timestamp
    return {
        "request_id": rid or _rid("add"),
        "messages": [msg],
        "user_id": user_id,
        "session_id": session_id,
    }


def _eval(query: str, *rid_seq_pairs: tuple[str, int], **kw: Any) -> dict[str, Any]:
    """Build an EvaluationCase‑compatible dict.

    *rid_seq_pairs* are (request_id, sequence_index) pairs that identify
    the relevant memory IDs (deterministic via stable_memory_id).
    """
    relevant_ids = {stable_memory_id(rid, seq) for rid, seq in rid_seq_pairs}
    case: dict[str, Any] = {
        "query": query,
        "user_id": "dim-user",
        "relevant_ids": list(relevant_ids),
        "top_k": kw.pop("top_k", 100),
    }
    if "options" in kw:
        case["options"] = kw.pop("options")
    if kw:
        case.update(kw)
    return case


# ── Scenario definitions ────────────────────────────────────────────────────


@dataclass
class DimensionScenario:
    """One evaluation scenario targeting a specific AML sub‑dimension."""

    name: str
    dimension: str  # AML dimension code (A–G)
    adds: list[dict[str, Any]] = field(default_factory=list)
    cases: list[dict[str, Any]] = field(default_factory=list)


def _build_scenarios() -> list[DimensionScenario]:
    """Return all dimension‑specific evaluation scenarios."""
    global _rid_counter
    _rid_counter = 0  # reset for deterministic IDs

    # ──────── A: Explicit Fact Recall ────────
    fact_rid_1 = _rid("A1")
    fact_rid_2 = _rid("A2")
    fact_rid_3 = _rid("A3")
    fact_rid_4 = _rid("A4")
    fact_rid_5 = _rid("A5")
    fact_rid_6 = _rid("A6")

    explicit_fact = DimensionScenario(
        name="A: Explicit Fact Recall",
        dimension="A",
        adds=[
            _add("I live in Berlin.", rid=fact_rid_1),
            _add("My favorite color is blue.", rid=fact_rid_2),
            _add("John works at Google.", rid=fact_rid_3),
            _add("I like apples, bananas, and oranges.", rid=fact_rid_4),
            _add("The meeting is scheduled for 3 PM.", rid=fact_rid_5),
            # Chinese facts
            _add("我住在北京，非常喜欢这座城市。", rid=fact_rid_6),
        ],
        cases=[
            _eval("Where do I live?", (fact_rid_1, 0)),
            _eval("What is my favorite color?", (fact_rid_2, 0)),
            _eval("Where does John work?", (fact_rid_3, 0)),
            _eval("What fruits do I like?", (fact_rid_4, 0)),
            _eval("When is the meeting?", (fact_rid_5, 0)),
            _eval("北京 我住", (fact_rid_6, 0)),
        ],
    )

    # ──────── B: Compositional Inference (multi‑hop) ────────
    comp_rid_1 = _rid("B1")
    comp_rid_2 = _rid("B2")
    comp_rid_3 = _rid("B3")
    comp_rid_4 = _rid("B4")

    compositional = DimensionScenario(
        name="B: Compositional Inference",
        dimension="B",
        adds=[
            _add("I live in Berlin.", rid=comp_rid_1),
            _add("Berlin is the capital of Germany.", rid=comp_rid_2, session_id="dim-session-b"),
            _add("I work at Google.", rid=comp_rid_3, session_id="dim-session-c"),
            _add("My manager is Alice.", rid=comp_rid_4, session_id="dim-session-c"),
        ],
        cases=[
            # Cross‑session: two facts needed
            _eval("Which country do I live in?", (comp_rid_1, 0), (comp_rid_2, 0)),
            # Multi‑fact within session
            _eval("Who is my manager at Google?", (comp_rid_3, 0), (comp_rid_4, 0)),
        ],
    )

    # ──────── C: Temporal & Event Reasoning ────────
    ts_old = 1_700_000_000_000
    ts_mid = 1_700_100_000_000
    ts_new = 1_700_200_000_000

    temp_rid_1 = _rid("C1")
    temp_rid_2 = _rid("C2")
    temp_rid_3 = _rid("C3")
    temp_rid_4 = _rid("C4")
    temp_rid_5 = _rid("C5")
    temp_rid_6 = _rid("C6")
    temp_rid_7 = _rid("C7")
    temp_rid_8 = _rid("C8")

    temporal = DimensionScenario(
        name="C: Temporal & Event Reasoning",
        dimension="C",
        adds=[
            # Latest intent
            _add("I live in Berlin.", timestamp=ts_old, rid=temp_rid_1),
            _add("I live in Paris now.", timestamp=ts_new, rid=temp_rid_2),
            # Earliest / historical
            _add("My first job was at Acme Corp.", timestamp=ts_old, rid=temp_rid_3),
            _add("I currently work at Google.", timestamp=ts_new, rid=temp_rid_4),
            # Relative time (yesterday / last week)
            _add("I met John yesterday for lunch.", timestamp=ts_new, rid=temp_rid_5),
            _add("I bought a new car last week.", timestamp=ts_mid, rid=temp_rid_6),
            # Event ordering / trajectory
            _add("I prefer coffee in the morning.", timestamp=ts_old, rid=temp_rid_7),
            _add("I no longer prefer coffee in the morning.", timestamp=ts_new, rid=temp_rid_8),
        ],
        cases=[
            # Latest: 最新住在哪里 → 巴黎
            _eval("Where do I currently live?", (temp_rid_2, 0)),
            _eval("我现在住在哪里", (temp_rid_2, 0)),
            # Earliest: 最早住在哪里 → 柏林
            _eval("Where did I first live?", (temp_rid_1, 0)),
            # Historical: 最早的住处
            _eval("Where did I live in the past?", (temp_rid_1, 0), (temp_rid_2, 0)),
            # Relative time: 最近发生的事件
            _eval("Who did I meet?", (temp_rid_5, 0)),
            _eval("What did I buy?", (temp_rid_6, 0)),
            # Trajectory / preference change
            _eval("What does the user prefer in the morning now?", (temp_rid_8, 0)),
            _eval("What did the user used to prefer in the morning?", (temp_rid_7, 0), (temp_rid_8, 0)),
        ],
    )

    # ──────── D: Memory Governance ────────
    gov_rid_1 = _rid("D1")
    gov_rid_2 = _rid("D2")
    gov_rid_3 = _rid("D3")
    gov_rid_4 = _rid("D4")
    gov_rid_5 = _rid("D5")

    governance = DimensionScenario(
        name="D: Memory Governance",
        dimension="D",
        adds=[
            # New-value overwrite: old → 旧内容被 supersede
            _add("I live in Berlin.", timestamp=ts_old, rid=gov_rid_1),
            _add("I live in Paris.", timestamp=ts_new, rid=gov_rid_2, session_id="dim-session-d"),
            # Contradiction / explicit correction
            _add("I prefer coffee.", timestamp=ts_old, rid=gov_rid_3),
            _add("I no longer prefer coffee.", timestamp=ts_new, rid=gov_rid_4),
            # Deletion through TTL / retention: 该消息在 retention_days 外已被清理
            _add("This is a test of retention.", timestamp=ts_new, rid=gov_rid_5),
        ],
        cases=[
            # Current query → 只返回最新的（巴黎被 supersede 覆盖柏林）
            _eval("Where do I live?", (gov_rid_2, 0)),
            # Historical query → 两条都返回
            _eval("Where did I live in the past?", (gov_rid_1, 0), (gov_rid_2, 0)),
            # 偏好覆盖
            _eval("What does the user prefer?", (gov_rid_4, 0)),
            # 历史偏好（两条）
            _eval("What did the user used to prefer?", (gov_rid_3, 0), (gov_rid_4, 0)),
        ],
    )

    # ──────── E: Personalization & Care ────────
    pers_rid_1 = _rid("E1")
    pers_rid_2 = _rid("E2")
    pers_rid_3 = _rid("E3")
    pers_rid_4 = _rid("E4")
    pers_rid_5 = _rid("E5")

    personalization = DimensionScenario(
        name="E: Personalization & Care",
        dimension="E",
        adds=[
            # Explicit preference
            _add("I prefer quiet libraries for studying.", rid=pers_rid_1),
            _add("I love hiking in the mountains.", rid=pers_rid_2),
            # Personal background
            _add("I am allergic to peanuts.", rid=pers_rid_3),
            # Health context
            _add("I have high blood pressure.", rid=pers_rid_4),
            # Chinese preference
            _add("我喜欢吃辣的菜。", rid=pers_rid_5),
        ],
        cases=[
            # Preference recall
            _eval("What kind of libraries do I prefer?", (pers_rid_1, 0)),
            _eval("What activities do I enjoy?", (pers_rid_2, 0)),
            # Options channel recall
            _eval(
                "What do I like to eat?",
                (pers_rid_3, 0),
                options=["A. Peanuts", "B. Rice cakes"],
            ),
            # Health background
            _eval("What health condition do I have?", (pers_rid_4, 0)),
            # Chinese preference
            _eval("我喜欢吃什么？", (pers_rid_5, 0)),
        ],
    )

    # ──────── F: Context Learning & Execution ────────
    ctx_rid_1 = _rid("F1")
    ctx_rid_2 = _rid("F2")
    ctx_rid_3 = _rid("F3")
    ctx_rid_4 = _rid("F4")

    context_learning = DimensionScenario(
        name="F: Context Learning & Execution",
        dimension="F",
        adds=[
            # Domain knowledge
            _add("The formula for the area of a circle is πr².", rid=ctx_rid_1),
            # Rule system
            _add("The password is 'memoria2024'.", rid=ctx_rid_2),
            # Workflow / program
            _add("To deploy: run build.sh, then docker-compose up.", rid=ctx_rid_3),
            # Format constraint
            _add("The output format must be JSON with fields: id, name, score.", rid=ctx_rid_4),
        ],
        cases=[
            _eval("What is the formula for the area of a circle?", (ctx_rid_1, 0)),
            _eval("What is the password?", (ctx_rid_2, 0)),
            _eval("How do I deploy the application?", (ctx_rid_3, 0)),
            _eval("What is the output format?", (ctx_rid_4, 0)),
        ],
    )

    # ──────── G: Safety & Privacy ────────
    safe_rid_1 = _rid("G1")
    safe_rid_2 = _rid("G2")
    safe_rid_3 = _rid("G3")

    safety = DimensionScenario(
        name="G: Safety & Privacy",
        dimension="G",
        adds=[
            # Sensitive information
            _add("My social security number is 123-45-6789.", rid=safe_rid_1),
            # Private personal data
            _add("I was born on January 1, 1990.", rid=safe_rid_2),
            # Non‑sensitive query → should recall the right thing (not expose SSN)
            # 这里测试的是：检索返回了哪些内容，不是答案模型是否披露
            _add("I like reading science fiction.", rid=safe_rid_3),
        ],
        cases=[
            # 隐私敏感信息的检索能力（是否召回）
            _eval("What is my birth date?", (safe_rid_2, 0)),
            # 不相关查询不应返回敏感信息（精确度）
            _eval("What are my hobbies?", (safe_rid_3, 0)),
        ],
    )

    return [
        explicit_fact,
        compositional,
        temporal,
        governance,
        personalization,
        context_learning,
        safety,
    ]


# ── Runner ──────────────────────────────────────────────────────────────────


def _add_to_store(store: MemoryStore, add_dict: dict[str, Any]) -> None:
    store.add(AddRequest(**add_dict))


def run_dimension_eval(
    store: MemoryStore,
    scenarios: list[DimensionScenario] | None = None,
) -> dict[str, Any]:
    """Run all dimension scenarios against a configured store.

    Returns a dict with overall metrics plus per‑dimension breakdown.
    """
    if scenarios is None:
        scenarios = _build_scenarios()

    # Add all scenario memories (each scenario uses the same user_id for simplicity)
    for scenario in scenarios:
        for add_dict in scenario.adds:
            _add_to_store(store, add_dict)

    # Evaluate per dimension
    per_dimension: dict[str, dict[str, float | int]] = {}
    all_cases: list[EvaluationCase] = []

    for scenario in scenarios:
        cases = [EvaluationCase(**c) for c in scenario.cases]
        all_cases.extend(cases)
        metrics = evaluate_cases(store, cases, workers=1)
        per_dimension[scenario.dimension] = metrics

    # Overall across all cases
    overall = evaluate_cases(store, all_cases, workers=1)

    return {
        "overall": overall,
        "per_dimension": dict(sorted(per_dimension.items())),
    }


def _format_report(result: dict[str, Any]) -> str:
    lines = [
        "=" * 60,
        "  AML Dimension Evaluation Report",
        "=" * 60,
        "",
        f"  Overall:  Recall@{100}={result['overall']['recall_at_k']:.4f}  "
        f"MRR={result['overall']['mrr']:.4f}  "
        f"nDCG@{100}={result['overall']['ndcg_at_k']:.4f}",
        "",
        "  Per-Dimension Breakdown:",
    ]
    for dim_code in sorted(result["per_dimension"]):
        d = result["per_dimension"][dim_code]
        label = {
            "A": "A: Explicit Fact Recall",
            "B": "B: Compositional Inference",
            "C": "C: Temporal & Event Reasoning",
            "D": "D: Memory Governance",
            "E": "E: Personalization & Care",
            "F": "F: Context Learning & Execution",
            "G": "G: Safety & Privacy",
        }.get(dim_code, f"Dimension {dim_code}")
        lines.append(
            f"    {label}:  "
            f"Recall={d['recall_at_k']:.4f}  "
            f"MRR={d['mrr']:.4f}  "
            f"nDCG={d['ndcg_at_k']:.4f}  "
            f"(n={d['samples']})"
        )
    lines.extend(["", "=" * 60])
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="One-click AML dimension regression test for memoria"
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        help="Directory for the SQLite store (temporary if omitted)",
    )
    parser.add_argument(
        "--embedding-backend",
        default="none",
        choices=("none", "hashing", "local"),
        help="Embedding backend to use (default: none = pure lexical)",
    )
    parser.add_argument(
        "--reranker-backend",
        default="none",
        choices=("none", "local"),
        help="Reranker backend to use (default: none, local = CrossEncoder)",
    )
    parser.add_argument(
        "--reranker-multilingual",
        default=None,
        help="Optional multilingual reranker model path for CJK queries "
        "(enables language routing; requires --reranker-backend local)",
    )
    parser.add_argument(
        "--report-out",
        type=Path,
        help="Optional path to write JSON report",
    )
    args = parser.parse_args()

    data_dir = args.data_dir or Path(__file__).parent.parent.parent / ".dimension-eval-cache"
    data_dir.mkdir(parents=True, exist_ok=True)

    settings = Settings(
        data_dir=data_dir,
        auth_scheme="none",
        api_key=None,
        retention_days=365,
        max_top_k=1_000,
        embedding_backend=args.embedding_backend,  # type: ignore[arg-type]
        embedding_dimensions=384,
        embedding_model="models/bge-small-en-v1.5" if args.embedding_backend == "local" else "text-embedding-v4",
        reranker_backend=args.reranker_backend,  # type: ignore[arg-type]
        reranker_model="models/cross-encoder-ms-marco-MiniLM-L-6-v2"
        if args.reranker_backend == "local"
        else "qwen3-rerank",
        reranker_multilingual_model=args.reranker_multilingual or "",
        reranker_candidate_limit=100,
    )
    store = create_runtime_store(settings)

    scenarios = _build_scenarios()
    result = run_dimension_eval(store, scenarios)

    report = _format_report(result)
    print(report)

    if args.report_out is not None:
        args.report_out.write_text(
            json.dumps(result, indent=2, default=str), encoding="utf-8"
        )
        print(f"\nReport written to {args.report_out}")


if __name__ == "__main__":
    main()