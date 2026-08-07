"""Command-line retention cleanup for expired evaluation data."""
from __future__ import annotations

import json

from memoria.config import Settings
from memoria.store import MemoryStore


def main() -> None:
    """Remove expired records and print only aggregate, body-free output."""

    settings = Settings.from_env()
    store = MemoryStore(settings)
    store.initialize()
    deleted_requests = store.cleanup_expired()
    print(json.dumps({"deleted_requests": deleted_requests}))

