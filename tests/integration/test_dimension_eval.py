"""One-click AML dimension regression test — integrates dimension_eval into pytest.

Run:
    pytest tests/ -k dimension -v

This loads the dimension scenarios, adds them to a fresh store, and asserts
that no dimension's Recall@100 drops below a configurable threshold.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from memoria.config import Settings
from memoria.dimension_eval import _build_scenarios, run_dimension_eval
from memoria.runtime import create_runtime_store


# ── Per-dimension recall floor (pure lexical, adjust as needed) ────────
# These are the baseline scores from the first run. Bump them up as you
# improve the system; they are regression guards, not aspirational targets.
BASELINE_RECALL: dict[str, float] = {
    "A": 0.8333,  # Explicit Fact Recall
    "B": 0.5000,  # Compositional Inference
    "C": 0.6250,  # Temporal & Event Reasoning
    "D": 0.8750,  # Memory Governance
    "E": 1.0000,  # Personalization & Care
    "F": 1.0000,  # Context Learning & Execution
    "G": 0.5000,  # Safety & Privacy
}

BASELINE_MRR: dict[str, float] = {
    "A": 0.8333,
    "B": 0.5000,
    "C": 0.5312,
    "D": 0.6333,
    "E": 0.7222,
    "F": 0.7656,
    "G": 0.0500,
}


@pytest.fixture
def dimension_store(tmp_path: Path):
    """Fresh store for dimension evaluation."""
    settings = Settings(
        data_dir=tmp_path,
        auth_scheme="none",
        api_key=None,
        retention_days=365,
        max_top_k=1_000,
        embedding_backend="none",
    )
    store = create_runtime_store(settings)
    return store


def test_dimension_eval_all_scenarios_run_without_error(dimension_store) -> None:
    """Smoke test: all dimension scenarios load and evaluate without crashing."""
    scenarios = _build_scenarios()
    assert len(scenarios) == 7, f"Expected 7 dimension scenarios, got {len(scenarios)}"
    result = run_dimension_eval(dimension_store, scenarios)
    assert "overall" in result
    assert "per_dimension" in result
    assert len(result["per_dimension"]) == 7


def test_dimension_eval_recall_above_baseline(dimension_store) -> None:
    """Regression guard: each dimension's Recall@100 must stay above baseline."""
    scenarios = _build_scenarios()
    result = run_dimension_eval(dimension_store, scenarios)
    dims = result["per_dimension"]

    failures = []
    for dim_code in sorted(dims):
        actual = dims[dim_code]["recall_at_k"]
        assert isinstance(actual, float), f"Recall for {dim_code} is not float: {actual}"
        baseline = BASELINE_RECALL.get(dim_code, 0.0)
        if actual < baseline - 0.01:  # allow 1% tolerance
            failures.append(f"{dim_code}: {actual:.4f} < baseline {baseline:.4f}")

    if failures:
        pytest.fail(f"Recall regression detected:\n" + "\n".join(failures))


def test_dimension_eval_individual_scenario(dimension_store) -> None:
    """Each dimension scenario can be run individually."""
    scenarios = _build_scenarios()
    for scenario in scenarios:
        result = run_dimension_eval(dimension_store, [scenario])
        dim = result["per_dimension"]
        assert scenario.dimension in dim, f"Missing dimension {scenario.dimension}"
        metrics = dim[scenario.dimension]
        assert metrics["samples"] == len(scenario.cases), (
            f"{scenario.dimension}: expected {len(scenario.cases)} samples, "
            f"got {metrics['samples']}"
        )
        assert 0.0 <= metrics["recall_at_k"] <= 1.0
        assert 0.0 <= metrics["mrr"] <= 1.0
        assert 0.0 <= metrics["ndcg_at_k"] <= 1.0