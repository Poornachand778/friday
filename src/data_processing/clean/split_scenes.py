#!/usr/bin/env python
"""
Split the master screenplay markdown into individual scene files (json blobs).

Usage:
    poetry run python scripts/clean/split_scenes.py
"""

from pathlib import Path
import json
import re
from scripts.utils.md_parser import split_scenes

SOURCE = Path("data/film/scripts/aa_janta_naduma_draft.md")
OUTROOT = Path("data/clean_chunks/film/scenes")
OUTROOT.mkdir(parents=True, exist_ok=True)


def flag_profanity(text: str) -> bool:
    """Very crude swear detector."""
    return bool(re.search(r"\b(fuck|douche|shit)\b", text, re.I))


def main() -> None:
    md_text = SOURCE.read_text(encoding="utf-8")
    blocks = split_scenes(md_text)  # <- new helper
    for idx, block in enumerate(blocks, start=1):
        meta = {
            "scene_id": idx,
            "contains_profanity": flag_profanity(block),
            "lang": "en+te",
            "source": "aa_janta_naduma",
        }
        out_path = OUTROOT / f"scene_{idx:03d}.json"
        out_path.write_text(
            json.dumps({"meta": meta, "text": block}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    print(
        f"✅  Wrote {len(blocks)} scenes to {OUTROOT.resolve().relative_to(Path.cwd())}"
    )


if __name__ == "__main__":
    main()
