#!/usr/bin/env python3
import json
import sys


def main(path: str) -> int:
    rows = []
    seen = set()
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            key = obj.get("path")
            if key in seen:
                continue
            seen.add(key)
            rows.append(obj)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"deduped {path} -> {len(rows)} rows")
    return 0


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: dedup_jsonl.py <path>")
        raise SystemExit(2)
    raise SystemExit(main(sys.argv[1]))
