from __future__ import annotations

import json

import pytest

from memoria.ingest import load_add_cases


def test_load_add_cases_validates_leaderboard_add_shape(tmp_path) -> None:
    path = tmp_path / "adds.jsonl"
    path.write_text(
        json.dumps(
            {
                "request_id": "import-1",
                "messages": [{"role": "user", "content": "I like tea."}],
                "user_id": "import-user",
                "session_id": "import-session",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    cases = load_add_cases(path)

    assert len(cases) == 1
    assert cases[0].request_id == "import-1"


def test_load_add_cases_hides_malformed_line_details(tmp_path) -> None:
    path = tmp_path / "invalid.jsonl"
    path.write_text('{"request_id":"missing-fields"}\n', encoding="utf-8")

    with pytest.raises(ValueError, match="invalid Add case on line 1"):
        load_add_cases(path)


def test_load_add_cases_preserves_unicode_line_separators_inside_content(tmp_path) -> None:
    path = tmp_path / "unicode-separator.jsonl"
    path.write_text(
        json.dumps(
            {
                "request_id": "unicode-1",
                "messages": [{"role": "user", "content": "First\u2028second"}],
                "user_id": "import-user",
                "session_id": "import-session",
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    assert load_add_cases(path)[0].messages[0].content == "First\u2028second"
