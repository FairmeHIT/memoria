from __future__ import annotations

from memoria.retention import main


def test_retention_command_uses_environment_settings(monkeypatch, tmp_path, capsys) -> None:
    monkeypatch.setenv("MEMORIA_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("MEMORIA_AUTH_SCHEME", "none")

    main()

    assert capsys.readouterr().out == '{"deleted_requests": 0}\n'

