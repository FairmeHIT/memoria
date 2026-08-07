from __future__ import annotations

from memoria.evaluation import EvaluationCase, evaluate_cases
from tests.conftest import add_payload


def test_evaluation_runner_reuses_store_search_and_reports_metrics(client) -> None:
    assert client.post("/v1/add", json=add_payload()).status_code == 200
    known = client.app.state.store.search(
        query="vegetarian", options=None, user_id="user-a", top_k=1
    )[0].id

    metrics = evaluate_cases(
        client.app.state.store,
        [
            EvaluationCase(
                query="What meals do I prefer?",
                user_id="user-a",
                relevant_ids={known},
                top_k=100,
            )
        ],
    )

    assert metrics == {"samples": 1, "recall_at_k": 1.0, "mrr": 1.0, "ndcg_at_k": 1.0}


def test_evaluation_runner_parallel_workers_preserve_metrics(client) -> None:
    assert client.post("/v1/add", json=add_payload()).status_code == 200
    known = client.app.state.store.search(
        query="vegetarian", options=None, user_id="user-a", top_k=1
    )[0].id
    cases = [
        EvaluationCase(
            query="What meals do I prefer?",
            user_id="user-a",
            relevant_ids={known},
            top_k=100,
        )
        for _ in range(4)
    ]

    serial = evaluate_cases(client.app.state.store, cases)
    parallel = evaluate_cases(client.app.state.store, cases, workers=2)

    assert parallel == serial
