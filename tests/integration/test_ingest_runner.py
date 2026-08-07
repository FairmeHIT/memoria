from __future__ import annotations

import json

from memoria.ingest import ingest_file


def test_ingest_file_makes_imported_add_cases_searchable(client, tmp_path) -> None:
    path = tmp_path / "adds.jsonl"
    path.write_text(
        json.dumps(
            {
                "request_id": "import-search-1",
                "messages": [{"role": "user", "content": "I collect blue notebooks."}],
                "user_id": "import-user",
                "session_id": "import-session",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    result = ingest_file(client.app.state.store, path)

    assert result.requests == 1
    assert result.messages == 1
    hits = client.app.state.store.search(
        query="What notebooks do I collect?",
        options=None,
        user_id="import-user",
        top_k=10,
    )
    assert hits[0].content == "user: I collect blue notebooks."
