#!/usr/bin/env python3
import argparse
import json
import os
import re
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Ensure local src/ is importable when called directly
REPO_ROOT = Path(__file__).resolve().parents[3]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

try:
    from memory.store import MemoryStore  # type: ignore
except ImportError:
    # Fallback for when running from different directory
    from src.memory.store import MemoryStore  # type: ignore


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def try_decode(data: bytes) -> Tuple[str, str]:
    """Decode bytes to str, returning (text, encoding_used)."""
    encodings = ["utf-8", "utf-8-sig", "cp1252", "latin-1"]
    for enc in encodings:
        try:
            return data.decode(enc), enc
        except UnicodeDecodeError:
            continue
    # Last resort
    return data.decode("latin-1", errors="replace"), "latin-1"


ANDROID_LINE = re.compile(
    r"^\s*\d{1,2}[\/\-]\d{1,2}[\/\-]\d{2,4},?\s+\d{1,2}:\d{2}(?:\s?[AP]M)?\s+-\s+.+?:"
)
IOS_LINE = re.compile(
    r"^\s*\[\d{1,2}[\/\-]\d{1,2}[\/\-]\d{2,4},?\s+\d{1,2}:\d{2}(?::\d{2})?\s?(?:AM|PM)?\]\s+.+?:"
)


def guess_format(lines: List[str]) -> str:
    sample = lines[:100]
    a = sum(1 for ln in sample if ANDROID_LINE.search(ln))
    i = sum(1 for ln in sample if IOS_LINE.search(ln))
    if a == 0 and i == 0:
        return "unknown"
    return "Android" if a >= i else "iOS"


def guess_chat_name(path: Path, first_line: str) -> str:
    # Prefer contact/group name from parent directory if file is named _chat.txt
    if path.stem == "_chat":
        parent = path.parent.name
        m = re.search(r"WhatsApp Chat with (.+)", parent, re.IGNORECASE)
        if m:
            return m.group(1).strip()
        m = re.search(r"WhatsApp Chat - (.+)", parent, re.IGNORECASE)
        if m:
            return m.group(1).strip()
        return parent
    # Else try from filename
    name = path.stem
    m = re.search(r"WhatsApp Chat with (.+)", path.name, re.IGNORECASE)
    if m:
        return m.group(1).strip()
    m = re.search(r"WhatsApp Chat - (.+)", path.name, re.IGNORECASE)
    if m:
        return m.group(1).strip()
    return name


def ensure_dirs(*paths: Path) -> None:
    for p in paths:
        p.mkdir(parents=True, exist_ok=True)


def unpack_zips(src_dir: Path, out_dir: Path) -> List[Path]:
    """Unpack all .zip files in src_dir into out_dir/<zip_basename>/ and normalize .txt to UTF-8."""
    ensure_dirs(out_dir)
    extracted_txt: List[Path] = []
    for z in sorted(src_dir.glob("*.zip")):
        target_root = out_dir / z.stem
        ensure_dirs(target_root)
        with zipfile.ZipFile(z, "r") as zf:
            for member in zf.infolist():
                # Skip directories
                if member.is_dir():
                    continue
                # Extract
                dest_path = target_root / member.filename
                ensure_dirs(dest_path.parent)
                with zf.open(member, "r") as src, open(dest_path, "wb") as dst:
                    dst.write(src.read())
                # Normalize .txt
                if dest_path.suffix.lower() == ".txt":
                    raw = dest_path.read_bytes()
                    text, enc = try_decode(raw)
                    # Normalize newlines
                    text = text.replace("\r\n", "\n").replace("\r", "\n")
                    dest_path.write_text(text, encoding="utf-8")
                    extracted_txt.append(dest_path)
    return extracted_txt


def build_inventory(raw_root: Path, inventory_path: Path) -> List[Dict]:
    """Create inventory JSON from all .txt files under raw_root."""
    items: List[Dict] = []
    for txt in sorted(raw_root.rglob("*.txt")):
        try:
            lines = txt.read_text(encoding="utf-8", errors="ignore").splitlines()
        except Exception:
            # Try fallback
            raw = txt.read_bytes()
            s, _ = try_decode(raw)
            lines = s.splitlines()
        fmt = guess_format(lines)
        chat_name = guess_chat_name(txt, lines[0] if lines else "")
        items.append(
            {
                "file_path": str(txt.as_posix()),
                "chat_name": chat_name,
                "line_count": len(lines),
                "guessed_format": fmt,
            }
        )
    ensure_dirs(inventory_path.parent)
    obj = {"created_at": utc_now(), "root": str(raw_root.as_posix()), "files": items}
    inventory_path.write_text(
        json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return items


def log_sft_datasets(items: List[Dict]) -> int:
    """Append one row per future dataset into memory/data/sft_datasets.jsonl.
    Uses processed placeholder path per chat.
    Skips duplicates by path.
    """
    store = MemoryStore()
    sft_path = store.paths["sft"]
    existing_paths = set()
    if os.path.exists(sft_path):
        for row in store._iter_jsonl(sft_path):  # type: ignore[attr-defined]
            p = row.get("path")
            if isinstance(p, str):
                existing_paths.add(p)
    new_rows = 0
    for it in items:
        chat = it.get("chat_name") or Path(it.get("file_path", "chat")).stem
        placeholder = f"data/processed/whatsapp/{chat}.jsonl"
        if placeholder in existing_paths:
            continue
        row = {
            "path": placeholder,
            "size_examples": 0,
            "mix": {},
            "lang_mix": {},
            "created_at": utc_now(),
        }
        store._append_jsonl(sft_path, row)  # type: ignore[attr-defined]
        new_rows += 1
    return new_rows


def run_all(src_dir: Path, raw_out: Path, inventory_out: Path) -> None:
    unpack_zips(src_dir, raw_out)
    items = build_inventory(raw_out, inventory_out)
    added = log_sft_datasets(items)
    print(
        f"Inventory written: {inventory_out} ({len(items)} files). SFT rows added: {added}"
    )


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="WhatsApp ingest: unpack, normalize, inventory"
    )
    p.add_argument(
        "--src-dir", default="data/raw/whatsapp/zips", help="Where the .zip files live"
    )
    p.add_argument(
        "--raw-out",
        default="data/raw/whatsapp",
        help="Where to extract normalized files",
    )
    p.add_argument(
        "--inventory-out",
        default="data/inventory/whatsapp_index.json",
        help="Output inventory JSON path",
    )
    sp = p.add_subparsers(dest="cmd", required=True)

    sp.add_parser("unpack", help="Unpack zips and normalize to UTF-8")
    sp.add_parser("inventory", help="Build inventory JSON from raw directory")
    sp.add_parser("all", help="Run unpack + inventory + SFT dataset logging")
    return p


def main(argv: Optional[List[str]] = None) -> int:
    ap = build_parser()
    args = ap.parse_args(argv)
    src_dir = Path(args.src_dir)
    raw_out = Path(args.raw_out)
    inventory_out = Path(args.inventory_out)

    if args.cmd == "unpack":
        files = unpack_zips(src_dir, raw_out)
        print(f"Unpacked and normalized {len(files)} text files into {raw_out}")
        return 0
    if args.cmd == "inventory":
        items = build_inventory(raw_out, inventory_out)
        added = log_sft_datasets(items)
        print(
            f"Inventory written: {inventory_out} ({len(items)} files). SFT rows added: {added}"
        )
        return 0
    if args.cmd == "all":
        run_all(src_dir, raw_out, inventory_out)
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
