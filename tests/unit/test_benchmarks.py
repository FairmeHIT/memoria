from __future__ import annotations

import json

from memoria.benchmarks import prepare_locomo, prepare_longmemeval


def test_prepare_locomo_uses_public_evidence_ids_without_gold_answers(tmp_path) -> None:
    conversations = tmp_path / "conversations.jsonl"
    conversations.write_text(
        json.dumps(
            {
                "sample_id": "conv-test",
                "sessions": [
                    {
                        "session_index": 1,
                        "messages": [
                            {"dia_id": "D1:1", "role": "user", "text": "I like tea."}
                        ],
                    }
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    questions = tmp_path / "questions.jsonl"
    questions.write_text(
        json.dumps(
            {
                "qa_id": "conv-test#q0000",
                "sample_id": "conv-test",
                "question": "What do I like?",
                "evidence": ["D1:1"],
                "answer": ["tea"],
            }
        )
        + "\n",
        encoding="utf-8",
    )

    adds, evaluations = prepare_locomo(conversations, questions)

    assert len(adds) == len(evaluations) == 1
    assert next(iter(evaluations[0].relevant_ids)).startswith("mem_")
    assert "answer" not in evaluations[0].model_dump()
    assert adds[0].user_id == "locomo:conv-test"


def test_prepare_longmemeval_maps_answer_sessions_to_source_memory_ids(tmp_path) -> None:
    source = tmp_path / "longmemeval.json"
    source.write_text(
        json.dumps(
            [
                {
                    "question_id": "q-test",
                    "question": "What did I buy?",
                    "answer": "a book",
                    "answer_session_ids": ["session-1"],
                    "haystack_session_ids": ["session-1"],
                    "haystack_sessions": [[{"role": "user", "content": "I bought a book."}]],
                }
            ]
        ),
        encoding="utf-8",
    )

    adds, evaluations = prepare_longmemeval(source)

    assert len(adds) == 1
    assert evaluations[0].relevant_ids
    assert evaluations[0].user_id == "longmemeval:q-test"
    assert "answer" not in evaluations[0].model_dump()
