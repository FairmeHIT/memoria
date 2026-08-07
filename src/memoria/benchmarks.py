"""Adapters for the downloaded, authorized benchmark layouts.

The adapters emit only Add-shaped source records and Search evaluation cases.
Gold answers and rubrics are intentionally never written to either output.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from memoria.evaluation import EvaluationCase
from memoria.schemas import AddRequest, Message
from memoria.store import stable_memory_id


def prepare_locomo(
    conversations_path: Path, questions_path: Path
) -> tuple[list[AddRequest], list[EvaluationCase]]:
    conversations = {
        str(row["sample_id"]): row
        for row in _read_jsonl(conversations_path)
    }
    adds: list[AddRequest] = []
    memory_by_dia: dict[str, str] = {}
    user_by_sample: dict[str, str] = {}
    for sample_id, conversation in conversations.items():
        user_id = f"locomo:{sample_id}"
        user_by_sample[sample_id] = user_id
        for session in conversation.get("sessions", []):
            session_index = int(session["session_index"])
            source_messages = [
                message for message in session.get("messages", []) if str(message.get("text", "")).strip()
            ]
            for chunk_index, start in enumerate(range(0, len(source_messages), 20)):
                source_chunk = source_messages[start : start + 20]
                request_id = f"locomo:{sample_id}:session:{session_index}:chunk:{chunk_index}"
                messages = [
                    Message(role=_role(message), content=str(message["text"]))
                    for message in source_chunk
                ]
                adds.append(
                    AddRequest(
                        request_id=request_id,
                        messages=messages,
                        user_id=user_id,
                        session_id=f"{sample_id}:session:{session_index}",
                    )
                )
                for sequence, source in enumerate(source_chunk):
                    memory_by_dia[str(source["dia_id"])] = stable_memory_id(request_id, sequence)

    evaluations: list[EvaluationCase] = []
    for question in _read_jsonl(questions_path):
        relevant_ids = {
            memory_by_dia[evidence_id]
            for evidence_id in question.get("evidence", [])
            if evidence_id in memory_by_dia
        }
        if relevant_ids:
            evaluations.append(
                EvaluationCase(
                    query=str(question["question"]),
                    user_id=user_by_sample[str(question["sample_id"])],
                    relevant_ids=relevant_ids,
                    top_k=100,
                )
            )
    return adds, evaluations


def prepare_longmemeval(path: Path) -> tuple[list[AddRequest], list[EvaluationCase]]:
    rows = json.loads(path.read_text(encoding="utf-8"))
    adds: list[AddRequest] = []
    evaluations: list[EvaluationCase] = []
    for row in rows:
        question_id = str(row["question_id"])
        user_id = f"longmemeval:{question_id}"
        memory_by_session: dict[str, list[str]] = {}
        for index, (session_id, session) in enumerate(
            zip(row.get("haystack_session_ids", []), row.get("haystack_sessions", []), strict=False)
        ):
            source_messages = [message for message in session if str(message.get("content", "")).strip()]
            session_memory_ids: list[str] = []
            for chunk_index, start in enumerate(range(0, len(source_messages), 20)):
                source_chunk = source_messages[start : start + 20]
                request_id = f"longmemeval:{question_id}:session:{index}:chunk:{chunk_index}"
                messages = [
                    Message(role=_role(message), content=str(message["content"]))
                    for message in source_chunk
                ]
                adds.append(
                    AddRequest(
                        request_id=request_id,
                        messages=messages,
                        user_id=user_id,
                        session_id=str(session_id),
                    )
                )
                session_memory_ids.extend(
                    stable_memory_id(request_id, sequence) for sequence in range(len(messages))
                )
            if session_memory_ids:
                memory_by_session[str(session_id)] = session_memory_ids
        relevant_ids = {
            memory_id
            for session_id in row.get("answer_session_ids", [])
            for memory_id in memory_by_session.get(str(session_id), [])
        }
        if relevant_ids:
            evaluations.append(
                EvaluationCase(
                    query=str(row["question"]),
                    user_id=user_id,
                    relevant_ids=relevant_ids,
                    top_k=100,
                )
            )
    return adds, evaluations


def write_prepared(
    adds: list[AddRequest], evaluations: list[EvaluationCase], adds_path: Path, eval_path: Path
) -> None:
    adds_path.parent.mkdir(parents=True, exist_ok=True)
    eval_path.parent.mkdir(parents=True, exist_ok=True)
    with adds_path.open("w", encoding="utf-8") as adds_file:
        for add in adds:
            adds_file.write(json.dumps(add.model_dump(mode="json"), ensure_ascii=False) + "\n")
    with eval_path.open("w", encoding="utf-8") as eval_file:
        for case in evaluations:
            eval_file.write(json.dumps(case.model_dump(mode="json"), ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare downloaded benchmark data for memoria")
    parser.add_argument("--benchmark", choices=("locomo_refined", "longmemeval"), required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--questions", type=Path)
    parser.add_argument("--adds-out", type=Path, required=True)
    parser.add_argument("--eval-out", type=Path, required=True)
    args = parser.parse_args()
    if args.benchmark == "locomo_refined":
        if args.questions is None:
            parser.error("--questions is required for locomo_refined")
        adds, evaluations = prepare_locomo(args.input, args.questions)
    else:
        adds, evaluations = prepare_longmemeval(args.input)
    write_prepared(adds, evaluations, args.adds_out, args.eval_out)
    print(json.dumps({"add_requests": len(adds), "evaluation_cases": len(evaluations)}))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as source:
        return [json.loads(line) for line in source if line.strip()]


def _role(message: dict[str, Any]) -> str:
    role = str(message.get("role", "")).lower()
    if role not in {"user", "assistant"}:
        raise ValueError("benchmark message role must be user or assistant")
    return role
