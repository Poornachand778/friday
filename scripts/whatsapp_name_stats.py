#!/usr/bin/env python3
import glob
import re
from collections import Counter

FILES = glob.glob("data/raw/whatsapp/*/_chat.txt")
pat = re.compile(r"^\s*\[[^\]]+\]\s+([^:]+):")

overall = Counter()
per_file = {}
for f in FILES:
    c = Counter()
    with open(f, "r", encoding="utf-8", errors="ignore") as fh:
        for line in fh:
            m = pat.match(line)
            if not m:
                continue
            name = m.group(1).strip()
            c[name] += 1
            overall[name] += 1
    per_file[f] = c

print("Top names overall:")
for name, cnt in overall.most_common(10):
    print(f"{cnt}\t{name}")

print("\nPer-file top names:")
for f in sorted(per_file):
    tops = per_file[f].most_common(3)
    print(f"{f}:")
    for name, cnt in tops:
        print(f"  {cnt}\t{name}")
