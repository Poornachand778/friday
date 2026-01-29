#!/usr/bin/env python3
"""Populate embeddings for LTM memories and content snippets."""

import argparse
import json
import math
import hashlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"


if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

try:
    from memory.store import MemoryStore  # type: ignore
except ImportError:
    # Fallback for when running from different directory
    from src.memory.store import MemoryStore  # type: ignore

VECTOR_DIM = 48


def embed_text(text: str) -> List[float]:
    vec = [0.0] * VECTOR_DIM
    if not text:
        return vec
    tokens = [tok for tok in text.lower().split() if tok]
    for tok in tokens:
        h = hashlib.sha256(tok.encode("utf-8")).hexdigest()
        bucket = int(h[:8], 16) % VECTOR_DIM
        sign = 1 if int(h[8:9], 16) % 2 else -1
        vec[bucket] += sign
    norm = math.sqrt(sum(v * v for v in vec))
    if norm == 0:
        return vec
    return [round(v / norm, 6) for v in vec]


def update_jsonl(path: Path, rows: List[Dict]) -> None:
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def process_file(path: Path, key_body: str) -> int:
    if not path.exists():
        return 0
    updated = 0
    rows: List[Dict] = []
    for row in MemoryStore()._iter_jsonl(str(path)):
        if not isinstance(row, dict):
            continue
        if not isinstance(row.get("embedding"), list):
            text = row.get(key_body, "")
            row["embedding"] = embed_text(text)
            row["updated_at"] = (
                datetime.now(timezone.utc).replace(microsecond=0).isoformat()
            )
            updated += 1
        rows.append(row)
    update_jsonl(path, rows)
    return updated


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compute embeddings for stored memories/snippets"
    )
    parser.add_argument("--ltm", default="memory/data/ltm_memories.jsonl")
    parser.add_argument("--snippets", default="memory/data/content_snippets.jsonl")
    args = parser.parse_args()

    ltm_updates = process_file(Path(args.ltm), "text")
    snippet_updates = process_file(Path(args.snippets), "body")
    print(f"Updated {ltm_updates} LTM rows and {snippet_updates} snippet rows.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
