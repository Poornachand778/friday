#!/usr/bin/env python3
"""Mine durable memories and reusable snippets from WhatsApp SFT data."""

import argparse
import json

from pathlib import Path
from typing import Dict, Iterable, List, Set, Tuple
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

TELUGU_RANGE = (0x0C00, 0x0C7F)
SCRIPT_TERMS = {
    "script",
    "screenplay",
    "scene",
    "beats",
    "beat",
    "character",
    "arc",
    "outline",
    "treatment",
    "logline",
    "pitch",
    "story",
    "film",
    "movie",
    "pilot",
}
PREFERENCE_TERMS = {
    "always",
    "never",
    "prefer",
    "keep",
    "remember",
    "rule",
    "guardrail",
    "checklist",
    "workflow",
    "process",
    "format",
    "template",
    "steps",
    "points",
    "reminder",
}


def detect_lang(text: str) -> str:
    for ch in text:
        code = ord(ch)
        if TELUGU_RANGE[0] <= code <= TELUGU_RANGE[1]:
            return "te"
    return "en"


def load_outputs(processed_dir: Path) -> Iterable[Tuple[str, Dict]]:
    for jsonl_path in sorted(processed_dir.glob("*.jsonl")):
        with jsonl_path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                yield jsonl_path.stem, row


def qualifies_for_memory(text: str) -> bool:
    normalized = text.lower()
    if len(text) < 40 or len(text) > 480:
        return False
    if normalized.count("?") > 1:
        return False
    return any(term in normalized for term in PREFERENCE_TERMS | SCRIPT_TERMS)


def qualifies_for_snippet(text: str) -> bool:
    if len(text) < 60:
        return False
    if text.count("\n") < 1:
        return False
    if not any(marker in text for marker in ("-", "•", "1.", "1)", "->")):
        return False
    return True


def title_from_text(text: str) -> str:
    first_line = text.strip().splitlines()[0].strip()
    return first_line[:80]


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Extract memories and snippets from processed WhatsApp data"
    )
    parser.add_argument("--processed-dir", default="data/processed/whatsapp")
    parser.add_argument("--max-memories", type=int, default=25)
    parser.add_argument("--max-snippets", type=int, default=10)
    args = parser.parse_args()

    processed_dir = Path(args.processed_dir)
    store = MemoryStore()

    existing_ltm: Set[str] = set()
    if Path(store.paths["ltm"]).exists():
        for row in store._iter_jsonl(store.paths["ltm"]):  # type: ignore[attr-defined]
            if isinstance(row, dict):
                existing_ltm.add(row.get("text", ""))

    existing_snippets: Set[str] = set()
    if Path(store.paths["snippets"]).exists():
        for row in store._iter_jsonl(store.paths["snippets"]):  # type: ignore[attr-defined]
            if isinstance(row, dict):
                existing_snippets.add(row.get("body", ""))

    memory_candidates: List[Tuple[str, Dict]] = []
    snippet_candidates: List[Tuple[str, Dict]] = []

    for chat, row in load_outputs(processed_dir):
        reply = row.get("output", "")
        if not reply:
            continue
        lang = detect_lang(reply)
        normalized = reply.strip()
        lowered = normalized.lower()
        is_script = any(term in lowered for term in SCRIPT_TERMS)
        if qualifies_for_memory(normalized) and normalized not in existing_ltm:
            tags = ["persona", "whatsapp", "tone:decisive"]
            if is_script:
                tags.append("film")
            memory_candidates.append(
                (
                    normalized,
                    {
                        "lang": lang,
                        "tags": tags,
                        "source": "whatsapp",
                    },
                )
            )
        if qualifies_for_snippet(normalized) and normalized not in existing_snippets:
            tags = ["persona", "whatsapp", "template"]
            domain = "film" if is_script else "general"
            snippet_candidates.append(
                (
                    normalized,
                    {
                        "lang": lang,
                        "tags": tags,
                        "domain": domain,
                    },
                )
            )

    stored_memories = 0
    for text, meta in memory_candidates[: args.max_memories]:
        store.add_ltm_memory(
            text=text,
            lang=meta["lang"],
            tags=meta["tags"],
            trust=4,
            source="whatsapp",
        )
        stored_memories += 1

    stored_snippets = 0
    for text, meta in snippet_candidates[: args.max_snippets]:
        store.add_snippet(
            title=title_from_text(text),
            body=text,
            lang=meta["lang"],
            tags=meta["tags"],
            version=1,
            domain=meta["domain"],
        )
        stored_snippets += 1

    print(
        f"Added {stored_memories} LTM memories and {stored_snippets} content snippets."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
