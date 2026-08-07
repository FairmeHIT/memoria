"""Backfill vectors for a configured embedding provider without replaying Add."""
from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

from memoria.config import Settings
from memoria.runtime import create_runtime_store


def main() -> None:
    parser = argparse.ArgumentParser(description="Backfill vectors for existing memoria source records")
    parser.add_argument("--data-dir", type=Path, default=Path("./data"))
    parser.add_argument("--batch-size", type=int, default=100)
    args = parser.parse_args()

    settings = replace(
        Settings.from_env(),
        data_dir=args.data_dir,
        auth_scheme="none",
        api_key=None,
    )
    store = create_runtime_store(settings)
    print(json.dumps(store.reindex_embeddings(batch_size=args.batch_size), separators=(",", ":")))
