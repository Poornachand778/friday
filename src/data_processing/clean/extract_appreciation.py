#!/usr/bin/env python
"""
Extract every '#### WORD: <term>' block from
data/storytelling/appreciation_vocab.md into JSON-Lines.

Output -> data/clean_chunks/vocab/appreciation.jsonl
"""

from pathlib import Path
import json
import re

SRC = Path("data/storytelling/appreciation_vocab.md")
DEST = Path("data/clean_chunks/vocab/appreciation.jsonl")
DEST.parent.mkdir(parents=True, exist_ok=True)

# matches: "#### WORD: archaic"
WORD_RE = re.compile(r"^####\s+WORD:\s*(.+)$", re.I)

records = []
current_word, current_lines = None, []

for ln in SRC.read_text(encoding="utf-8").splitlines():
    m = WORD_RE.match(ln)
    if m:
        # flush the previous word
        if current_word:
            records.append(
                {
                    "word": current_word.strip(),
                    "definition": " ".join(current_lines).strip(),
                }
            )
        current_word = m.group(1)
        current_lines = []
    else:
        # strip leading bullet / asterisks etc.
        current_lines.append(ln.strip(" -*"))

# final flush
if current_word:
    records.append(
        {"word": current_word.strip(), "definition": " ".join(current_lines).strip()}
    )

# write jsonl
with DEST.open("w", encoding="utf-8") as fh:
    for rec in records:
        fh.write(json.dumps(rec, ensure_ascii=False) + "\n")

print(f"✅  wrote {len(records)} vocab items -> {DEST}")
