#!/usr/bin/env python
"""
Convert every #### SNIPPET block in
data/film/snippets/unplaced_dialogues.md
into a ChatML-style instruction/response pair.

Output → data/instructions/snippets_chatml.jsonl
"""

from pathlib import Path
import json
from scripts.utils.md_parser import parse_snippets

SRC = Path("data/film/snippets/unplaced_dialogues.md")
DEST = Path("data/instructions/snippets_chatml.jsonl")
DEST.parent.mkdir(parents=True, exist_ok=True)

records = []
for item in parse_snippets(SRC):
    # Header: "#### CONFESSION – Niha-Virgin-Reveal"
    title = item["header"].lstrip("# ").split("–", 1)[-1].strip()
    body = item["text"].strip()

    records.append(
        {"instruction": f"Use the dialogue snippet titled '{title}'.", "response": body}
    )

with DEST.open("w", encoding="utf-8") as fh:
    for rec in records:
        fh.write(json.dumps(rec, ensure_ascii=False) + "\n")

print(f"✅  wrote {len(records)} instruction pairs → {DEST}")
