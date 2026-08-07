from __future__ import annotations

import pytest

from memoria.evaluation import ndcg_at_k, recall_at_k, reciprocal_rank


def test_retrieval_metrics_are_bounded_and_rank_sensitive() -> None:
    retrieved = ["m-wrong", "m-right", "m-other"]
    relevant = {"m-right"}

    assert recall_at_k(retrieved, relevant, 2) == 1.0
    assert reciprocal_rank(retrieved, relevant) == 0.5
    assert 0.0 < ndcg_at_k(retrieved, relevant, 2) < 1.0


@pytest.mark.parametrize("metric", [recall_at_k, ndcg_at_k])
def test_empty_relevance_has_zero_score(metric) -> None:
    assert metric(["m-1"], set(), 10) == 0.0
