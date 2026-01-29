import re
from pathlib import Path
from typing import List, Dict

SCENE_RE = re.compile(r"^###\s+SCENE\s+\d+", re.I)
SNIPPET_RE = re.compile(r"^####\s+([A-Z _-]+)\s+–\s+(.+)$")


def split_scenes(md_text: str) -> List[str]:
    """Return list of scene blocks (header included)."""
    blocks, cur = [], []
    for line in md_text.splitlines():
        if SCENE_RE.match(line) and cur:
            blocks.append("\n".join(cur).strip())
            cur = [line]
        else:
            cur.append(line)
    if cur:
        blocks.append("\n".join(cur).strip())
    return blocks


def parse_snippets(md_path: Path) -> List[Dict]:
    """Yield snippet dicts from unplaced_dialogues.md."""
    txt = md_path.read_text(encoding="utf-8")
    items, cur_head, cur_lines = [], None, []
    for ln in txt.splitlines():
        m = SNIPPET_RE.match(ln)
        if m:
            if cur_head:
                items.append({"header": cur_head, "text": "\n".join(cur_lines).strip()})
            cur_head, cur_lines = ln, []
        else:
            cur_lines.append(ln)
    if cur_head:
        items.append({"header": cur_head, "text": "\n".join(cur_lines).strip()})
    return items
