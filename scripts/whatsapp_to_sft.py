#!/usr/bin/env python3
"""Convert WhatsApp exports into SFT-ready JSONL pairs."""

import argparse
import json
import math
import os
import re
import sys
import uuid
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from functools import lru_cache
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

try:
    from memory.store import MemoryStore  # type: ignore
except ImportError:
    # Fallback for when running from different directory
    from src.memory.store import MemoryStore  # type: ignore

try:
    from wordfreq import zipf_frequency
except ImportError:  # pragma: no cover - dependency should be available in runtime env
    zipf_frequency = None

PERSONA_NAMES = {
    "poorna chand k",
    "poorna",
    "pc",
    "you",
}

TELUGU_RANGE = (0x0C00, 0x0C7F)
MESSAGE_RE = re.compile(r"^\s*\[([^\]]+)\]\s+([^:]+):\s?(.*)$")
PHONE_RE = re.compile(r"\+?\d[\d\s\-()]{5,}\d")
EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
EMAIL_LAX_RE = re.compile(r"\b[^\s@]+@[^\s@]+\b")
WORD_RE = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?")

DATETIME_FORMATS = [
    "%m/%d/%y, %H:%M:%S",
    "%m/%d/%y, %H:%M",
    "%m/%d/%Y, %H:%M:%S",
    "%m/%d/%Y, %H:%M",
    "%d/%m/%y, %H:%M:%S",
    "%d/%m/%y, %H:%M",
    "%d/%m/%Y, %H:%M:%S",
    "%d/%m/%Y, %H:%M",
    "%m/%d/%y, %I:%M:%S %p",
    "%m/%d/%y, %I:%M %p",
    "%m/%d/%Y, %I:%M:%S %p",
    "%m/%d/%Y, %I:%M %p",
    "%d/%m/%y, %I:%M:%S %p",
    "%d/%m/%y, %I:%M %p",
    "%d/%m/%Y, %I:%M:%S %p",
    "%d/%m/%Y, %I:%M %p",
]

SCRIPT_KEYWORDS = {
    "script",
    "screenplay",
    "scene",
    "beat",
    "logline",
    "character",
    "arc",
    "plot",
    "story",
    "treatment",
    "short film",
    "pilot",
    "genre",
    "dialogue",
    "dialog",
    "climax",
    "act",
    "montage",
    "villain",
    "protagonist",
    "antagonist",
}


@dataclass
class Message:
    timestamp: datetime
    speaker: str
    text: str


@dataclass
class Example:
    chat: str
    source_timestamp: datetime
    input_text: str
    output_text: str
    lang: str
    tags: List[str]
    meta: Dict[str, str]


class TeluguTransliterator:
    """Convert romanised Telugu lines into Telugu script via IndicXlit."""

    SKIP_WORDS = {"http", "https", "www"}
    ENGLISH_COMMON = {
        "a",
        "about",
        "after",
        "again",
        "all",
        "also",
        "am",
        "an",
        "and",
        "any",
        "are",
        "as",
        "at",
        "awesome",
        "back",
        "bad",
        "be",
        "because",
        "before",
        "better",
        "big",
        "boss",
        "but",
        "by",
        "call",
        "can",
        "cant",
        "check",
        "come",
        "cool",
        "correct",
        "day",
        "did",
        "didnt",
        "do",
        "dont",
        "even",
        "ever",
        "everyone",
        "example",
        "fade",
        "fine",
        "first",
        "for",
        "from",
        "get",
        "go",
        "good",
        "got",
        "great",
        "had",
        "has",
        "have",
        "hero",
        "house",
        "hello",
        "hey",
        "hi",
        "how",
        "if",
        "important",
        "in",
        "into",
        "is",
        "issue",
        "it",
        "just",
        "keep",
        "know",
        "later",
        "leave",
        "left",
        "let",
        "like",
        "little",
        "look",
        "location",
        "love",
        "life",
        "maybe",
        "meet",
        "meeting",
        "montage",
        "mind",
        "money",
        "more",
        "most",
        "move",
        "much",
        "need",
        "never",
        "new",
        "next",
        "night",
        "no",
        "not",
        "now",
        "ok",
        "okay",
        "opening",
        "on",
        "one",
        "only",
        "camera",
        "open",
        "plan",
        "please",
        "project",
        "ready",
        "really",
        "right",
        "same",
        "scene",
        "say",
        "script",
        "schedule",
        "second",
        "see",
        "share",
        "since",
        "slow",
        "soon",
        "sorry",
        "sure",
        "take",
        "talk",
        "team",
        "thank",
        "thanks",
        "that",
        "the",
        "then",
        "there",
        "these",
        "they",
        "thing",
        "think",
        "this",
        "those",
        "time",
        "today",
        "tomorrow",
        "tonight",
        "too",
        "try",
        "update",
        "world",
        "risk",
        "risks",
        "responsibility",
        "someone",
        "something",
        "everything",
        "nothing",
        "society",
        "situations",
        "free",
        "very",
        "wait",
        "want",
        "was",
        "we",
        "well",
        "were",
        "what",
        "when",
        "where",
        "which",
        "who",
        "why",
        "will",
        "voice",
        "work",
        "wow",
        "yeah",
        "yes",
        "yo",
        "you",
        "your",
    }

    def __init__(self) -> None:
        if zipf_frequency is None:
            raise RuntimeError(
                "wordfreq is required for transliteration. Run `pip install wordfreq`."
            )
        try:
            import argparse
            import torch
            from ai4bharat.transliteration import XlitEngine
        except ImportError as exc:
            raise RuntimeError(
                "ai4bharat-transliteration is required for --transliterate."
                " Install it (preferably in the friday_ft conda env) or rerun "
                "without the transliteration flag."
            ) from exc

        os.environ.setdefault("FAIRSEQ_DISABLE_HYDRA", "1")
        os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
        torch.serialization.add_safe_globals([argparse.Namespace])
        self._engine = XlitEngine(lang2use=["te"], beam_width=5, rescore=True)
        self._cache: Dict[str, str] = {}
        self._english_threshold = float(os.getenv("TRANSLITERATOR_EN_THRESHOLD", "3.0"))

    def _should_skip(self, word: str, start: int, end: int, text: str) -> bool:
        lower = word.lower()
        if lower in self.SKIP_WORDS or lower.startswith("http"):
            return True
        if word.isupper() and len(word) > 4:
            return True
        if self._is_probably_english(lower):
            return True
        if any(ch.isdigit() for ch in word):
            return True
        if start > 0:
            prev = text[start - 1]
            if prev in "{[_":
                return True
        if end < len(text):
            nxt = text[end]
            if nxt in "}_]":
                return True
        return False

    def _translit_word(self, word: str) -> str:
        key = word.lower()
        cached = self._cache.get(key)
        if cached is not None:
            return cached
        result = self._engine.translit_word(word)
        payload = result.get("te", word)
        if isinstance(payload, list):
            payload = payload[0]
        payload = payload.replace("\u200c", "")
        self._cache[key] = payload
        return payload

    def transliterate(self, text: str) -> str:
        if not text:
            return text
        if any(TELUGU_RANGE[0] <= ord(ch) <= TELUGU_RANGE[1] for ch in text):
            return text

        def _replace(match: "re.Match[str]") -> str:
            start, end = match.span()
            token = match.group(0)
            if self._should_skip(token, start, end, text):
                return token
            return self._translit_word(token)

        return WORD_RE.sub(_replace, text)

    @staticmethod
    def _normalize(word: str) -> str:
        return re.sub(r"[^a-z']", "", word.lower())

    @lru_cache(maxsize=5000)
    def _is_probably_english(self, word: str) -> bool:
        cleaned = self._normalize(word)
        if not cleaned:
            return True
        if cleaned in self.ENGLISH_COMMON:
            return True
        freq = zipf_frequency(cleaned, "en", wordlist="best")
        if freq == float("-inf"):
            return False
        return freq >= self._english_threshold


def parse_timestamp(raw: str) -> Optional[datetime]:
    cleaned = raw.replace("\u202f", " ").replace("\u200e", "").strip()
    for fmt in DATETIME_FORMATS:
        try:
            return datetime.strptime(cleaned, fmt)
        except ValueError:
            continue
    return None


def iter_messages(path: Path) -> List[Message]:
    messages: List[Message] = []
    current: Optional[Message] = None
    with path.open("r", encoding="utf-8", errors="ignore") as fh:
        for line in fh:
            stripped = line.rstrip("\n")
            if not stripped:
                continue
            normalized = stripped.replace("\ufeff", "").replace("\u200e", "")
            match = MESSAGE_RE.match(normalized)
            if match:
                ts_raw, speaker, text = match.groups()
                ts = parse_timestamp(ts_raw)
                if ts is None:
                    continue
                current = Message(ts, speaker.strip(), text.strip())
                messages.append(current)
            elif current is not None:
                current.text = f"{current.text}\n{normalized.strip()}"
    return messages


def persona_alias(speaker: str) -> bool:
    return speaker.lower().strip() in PERSONA_NAMES


def assign_name_placeholders(messages: Iterable[Message]) -> Dict[str, str]:
    mapping: Dict[str, str] = {}
    alphabet = [chr(code) for code in range(ord("A"), ord("Z") + 1)]
    idx = 0
    for msg in messages:
        speaker_key = msg.speaker.strip().lower()
        if persona_alias(speaker_key):
            continue
        if speaker_key not in mapping:
            placeholder = f"{{NAME_{alphabet[idx % len(alphabet)]}}}"
            idx += 1
            mapping[speaker_key] = placeholder
    return mapping


def _replace_names(text: str, mapping: Dict[str, str]) -> str:
    out = text
    for name_key, placeholder in mapping.items():
        pattern = re.compile(re.escape(name_key), re.IGNORECASE)
        out = pattern.sub(placeholder, out)
    return out


def redact_text(text: str, name_map: Dict[str, str]) -> str:
    result = text
    result = _replace_names(result, name_map)
    result = PHONE_RE.sub("{PHONE}", result)
    result = EMAIL_RE.sub("{EMAIL}", result)
    result = EMAIL_LAX_RE.sub("{EMAIL}", result)
    return result


def detect_lang(text: str) -> str:
    for ch in text:
        code = ord(ch)
        if TELUGU_RANGE[0] <= code <= TELUGU_RANGE[1]:
            return "te"
    return "en"


def ensure_prefix(text: str, lang: str) -> str:
    prefix = "బాస్," if lang == "te" else "Boss,"
    stripped = text.lstrip()
    if stripped.lower().startswith(prefix.lower()):
        return stripped
    if stripped:
        return f"{prefix} {stripped}"
    return prefix


def is_script_related(text: str) -> bool:
    lowered = text.lower()
    return any(keyword in lowered for keyword in SCRIPT_KEYWORDS)


def build_examples(
    messages: List[Message],
    chat: str,
    reply_window: int,
    transliterator: Optional[TeluguTransliterator] = None,
) -> List[Example]:
    name_map = assign_name_placeholders(messages)
    examples: List[Example] = []
    pending: List[Message] = []
    for msg in messages:
        if persona_alias(msg.speaker):
            if not pending:
                continue
            last_incoming = pending[-1]
            if msg.timestamp - last_incoming.timestamp > timedelta(
                minutes=reply_window
            ):
                pending = []
                continue
            incoming_parts = []
            for incoming in pending:
                redacted = redact_text(incoming.text, name_map)
                placeholder = name_map.get(
                    incoming.speaker.lower().strip(), incoming.speaker
                )
                incoming_parts.append(f"{placeholder}: {redacted}")
            incoming_text = "\n".join(incoming_parts).strip()
            if not incoming_text:
                pending = []
                continue
            redacted_reply = redact_text(msg.text, name_map)
            reply = ensure_prefix(redacted_reply, detect_lang(redacted_reply))
            body = (
                reply[len("బాస్,") :].strip()
                if reply.startswith("బాస్,")
                else reply[len("Boss,") :].strip()
            )
            telugu_body = any(
                TELUGU_RANGE[0] <= ord(ch) <= TELUGU_RANGE[1] for ch in body
            )
            latin_body = any(ch.isalpha() for ch in body)
            digit_body = any(ch.isdigit() for ch in body)
            if body.lower() in {"sticker omitted", "media omitted", "image omitted"}:
                pending = []
                continue
            if not (telugu_body or latin_body or digit_body):
                pending = []
                continue
            script_incoming = is_script_related(incoming_text)
            script_reply = is_script_related(reply)
            if transliterator:
                incoming_text = transliterator.transliterate(incoming_text)
                reply = transliterator.transliterate(reply)
            lang = detect_lang(reply)
            tags = ["persona", "whatsapp"]
            if script_incoming or script_reply:
                tags.append("script")
            if "?" in reply:
                tags.append("clarifying")
            example = Example(
                chat=chat,
                source_timestamp=msg.timestamp,
                input_text=incoming_text,
                output_text=reply,
                lang=lang,
                tags=tags,
                meta={"reply_ts": msg.timestamp.isoformat(), "chat": chat},
            )
            examples.append(example)
            pending = []
        else:
            if pending and msg.timestamp - pending[-1].timestamp > timedelta(
                minutes=reply_window
            ):
                pending = []
            text_lower = msg.text.lower().strip()
            if not text_lower:
                continue
            if text_lower in {"media omitted", "sticker omitted"}:
                continue
            if text_lower.endswith(" omitted") and "\nomitted" not in msg.text:
                continue
            if text_lower.startswith("messages and calls are end-to-end encrypted"):
                continue
            pending.append(msg)
    return examples


def load_inventory(inventory_path: Path) -> List[Tuple[str, Path]]:
    obj = json.loads(inventory_path.read_text(encoding="utf-8"))
    files: List[Tuple[str, Path]] = []
    for entry in obj.get("files", []):
        chat = entry.get("chat_name") or Path(entry["file_path"]).stem
        files.append((chat, Path(entry["file_path"])))
    return files


def write_jsonl(path: Path, rows: List[Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def update_sft_registry(store: MemoryStore, stats: Dict[str, Dict]) -> None:
    path = store.paths["sft"]
    rows = []
    if Path(path).exists():
        rows = list(store._iter_jsonl(path))  # type: ignore[attr-defined]
    by_path = {row.get("path"): row for row in rows if isinstance(row, dict)}
    updated_rows = []
    for dataset_path, info in stats.items():
        row = by_path.get(dataset_path, {})
        row.update(
            {
                "path": dataset_path,
                "size_examples": info["size"],
                "mix": info["mix"],
                "lang_mix": info["lang_mix"],
                "created_at": row.get("created_at", info["created_at"]),
            }
        )
        updated_rows.append(row)
    # Write rows sorted by path for determinism
    updated_rows.sort(key=lambda r: r["path"])
    write_jsonl(Path(path), updated_rows)


def compute_mix(examples: List[Example]) -> Tuple[Dict[str, float], Dict[str, float]]:
    total = len(examples)
    if total == 0:
        return {}, {}
    tag_counts: Counter = Counter()
    lang_counts: Counter = Counter()
    for ex in examples:
        unique_tags = set(ex.tags)
        for tag in unique_tags:
            tag_counts[tag] += 1
        lang_counts[ex.lang] += 1
    mix = {tag: round(cnt / total, 3) for tag, cnt in tag_counts.items()}
    lang_mix = {lang: round(cnt / total, 3) for lang, cnt in lang_counts.items()}
    return mix, lang_mix


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Convert WhatsApp chats to SFT dataset"
    )
    parser.add_argument("--inventory", default="data/inventory/whatsapp_index.json")
    parser.add_argument("--train-out", default="data/sft/iter2/whatsapp_train.jsonl")
    parser.add_argument("--val-out", default="data/sft/iter2/whatsapp_val.jsonl")
    parser.add_argument("--processed-dir", default="data/processed/whatsapp")
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--reply-window", type=int, default=30)
    parser.add_argument(
        "--transliterate",
        action="store_true",
        help="Transliterate romanised Telugu to Telugu script (requires ai4bharat-transliteration)",
    )
    args = parser.parse_args()

    inventory_path = Path(args.inventory)
    files = load_inventory(inventory_path)
    processed_dir = Path(args.processed_dir)
    processed_dir.mkdir(parents=True, exist_ok=True)

    store = MemoryStore()
    per_chat_examples: Dict[str, List[Example]] = {}
    created_at = datetime.now(timezone.utc).replace(microsecond=0).isoformat()

    transliterator: Optional[TeluguTransliterator] = None
    if args.transliterate:
        print("[setup] Initialising Telugu transliterator…", flush=True)
        try:
            transliterator = TeluguTransliterator()
        except RuntimeError as exc:
            parser.error(str(exc))
        print("[setup] Transliteration enabled.", flush=True)

    total_files = len(files)
    for idx_chat, (chat, file_path) in enumerate(files, 1):
        if not file_path.exists():
            continue
        print(f"[{idx_chat}/{total_files}] Processing {chat}…", flush=True)
        messages = iter_messages(file_path)
        examples = build_examples(messages, chat, args.reply_window, transliterator)
        per_chat_examples[chat] = examples
        chat_path = processed_dir / f"{chat}.jsonl"
        rows = []
        for idx, ex in enumerate(examples):
            rows.append(
                {
                    "id": f"{uuid.uuid4().hex}",
                    "input": ex.input_text,
                    "output": ex.output_text,
                    "lang": ex.lang,
                    "tags": ex.tags,
                    "meta": ex.meta,
                }
            )
        write_jsonl(chat_path, rows)
        print(
            f"[{idx_chat}/{total_files}] {chat}: {len(examples)} examples", flush=True
        )

    train_rows: List[Dict] = []
    val_rows: List[Dict] = []
    stats: Dict[str, Dict] = {}

    for chat, examples in per_chat_examples.items():
        if not examples:
            continue
        examples.sort(key=lambda ex: ex.source_timestamp)
        val_count = (
            max(1, math.ceil(len(examples) * args.val_ratio))
            if len(examples) >= 5
            else max(0, math.ceil(len(examples) * args.val_ratio))
        )
        if val_count >= len(examples):
            val_count = (
                max(1, len(examples) // 5) if len(examples) > 1 else len(examples)
            )
        train_split = examples[:-val_count] if val_count else examples
        val_split = examples[-val_count:] if val_count else []
        for collection, target in ((train_split, train_rows), (val_split, val_rows)):
            for ex in collection:
                target.append(
                    {
                        "id": f"{uuid.uuid4().hex}",
                        "input": ex.input_text,
                        "output": ex.output_text,
                        "lang": ex.lang,
                        "tags": ex.tags,
                        "meta": ex.meta,
                    }
                )
        mix, lang_mix = compute_mix(examples)
        dataset_path = f"data/processed/whatsapp/{chat}.jsonl"
        stats[dataset_path] = {
            "size": len(examples),
            "mix": mix,
            "lang_mix": lang_mix,
            "created_at": created_at,
        }

    write_jsonl(Path(args.train_out), train_rows)
    write_jsonl(Path(args.val_out), val_rows)

    now_ts = datetime.now(timezone.utc)
    train_mix, train_lang = compute_mix(
        [
            Example(
                chat="*",
                source_timestamp=now_ts,
                input_text=row["input"],
                output_text=row["output"],
                lang=row["lang"],
                tags=row["tags"],
                meta=row["meta"],
            )
            for row in train_rows
        ]
    )
    val_mix, val_lang = compute_mix(
        [
            Example(
                chat="*",
                source_timestamp=now_ts,
                input_text=row["input"],
                output_text=row["output"],
                lang=row["lang"],
                tags=row["tags"],
                meta=row["meta"],
            )
            for row in val_rows
        ]
    )

    stats[args.train_out] = {
        "size": len(train_rows),
        "mix": train_mix,
        "lang_mix": train_lang,
        "created_at": created_at,
    }
    stats[args.val_out] = {
        "size": len(val_rows),
        "mix": val_mix,
        "lang_mix": val_lang,
        "created_at": created_at,
    }

    update_sft_registry(store, stats)
    print(f"Wrote {len(train_rows)} train and {len(val_rows)} val examples.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
