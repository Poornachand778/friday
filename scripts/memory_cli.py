#!/usr/bin/env python3
import argparse
import json
import os
import sys
from typing import List, Optional

# Ensure local src/ is importable
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC_DIR = os.path.join(REPO_ROOT, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

try:
    from memory.store import MemoryStore  # type: ignore
except ImportError:
    # Fallback for when running from different directory
    from src.memory.store import MemoryStore  # type: ignore


def _load_vector_from_file(path: str) -> List[float]:
    with open(path, "r", encoding="utf-8") as f:
        txt = f.read().strip()
    try:
        data = json.loads(txt)
        if isinstance(data, list):
            return [float(x) for x in data]
    except json.JSONDecodeError:
        pass
    # Fallback: comma/space separated floats
    parts = [p for p in txt.replace(",", " ").split() if p]
    return [float(p) for p in parts]


def cmd_add_ltm(args: argparse.Namespace) -> None:
    store = MemoryStore()
    vector = None
    if args.embedding_file:
        vector = _load_vector_from_file(args.embedding_file)
    mid = store.add_ltm_memory(
        text=args.text,
        lang=args.lang,
        tags=args.tags or [],
        trust=args.trust,
        embedding=vector,
        source=args.source,
    )
    print(mid)


def cmd_search_ltm(args: argparse.Namespace) -> None:
    store = MemoryStore()
    vector = None
    if args.vector_file:
        vector = _load_vector_from_file(args.vector_file)
    rows = store.search_ltm(
        query=args.query, query_vector=vector, tags=args.tags, top_k=args.top_k
    )
    print(json.dumps(rows, ensure_ascii=False, indent=2))


def cmd_add_snippet(args: argparse.Namespace) -> None:
    store = MemoryStore()
    vector = None
    if args.embedding_file:
        vector = _load_vector_from_file(args.embedding_file)
    sid = store.add_snippet(
        title=args.title,
        body=args.body,
        lang=args.lang,
        tags=args.tags or [],
        version=args.version,
        embedding=vector,
    )
    print(sid)


def cmd_search_snippet(args: argparse.Namespace) -> None:
    store = MemoryStore()
    vector = None
    if args.vector_file:
        vector = _load_vector_from_file(args.vector_file)
    rows = store.search_snippets(
        query=args.query, query_vector=vector, tags=args.tags, top_k=args.top_k
    )
    print(json.dumps(rows, ensure_ascii=False, indent=2))


def cmd_set_persona(args: argparse.Namespace) -> None:
    store = MemoryStore()
    with open(args.file, "r", encoding="utf-8") as f:
        obj = json.load(f)
    store.set_persona(obj)
    print("ok")


def cmd_get_persona(_: argparse.Namespace) -> None:
    store = MemoryStore()
    print(json.dumps(store.get_persona(), ensure_ascii=False, indent=2))


def cmd_log_interaction(args: argparse.Namespace) -> None:
    store = MemoryStore()
    iid = store.log_interaction(
        session_id=args.session_id,
        user_msg=args.user_msg,
        model_reply=args.model_reply,
        used_ltm_ids=args.used_ltm_ids or [],
        used_snippet_ids=args.used_snippet_ids or [],
        stm_summary=args.stm_summary,
        lang=args.lang,
    )
    print(iid)


def cmd_upsert_stm(args: argparse.Namespace) -> None:
    store = MemoryStore()
    sid = store.upsert_stm_session(
        session_id=args.session_id,
        rolling_summary=args.rolling_summary,
        target_lang=args.target_lang,
    )
    print(sid)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Friday memory CLI")
    sp = p.add_subparsers(dest="cmd", required=True)

    add_ltm = sp.add_parser("add-ltm", help="Add a long-term memory row")
    add_ltm.add_argument("--text", required=True)
    add_ltm.add_argument("--lang", default="te", choices=["te", "en", "mixed"])
    add_ltm.add_argument("--tags", nargs="*")
    add_ltm.add_argument("--trust", type=int, default=3)
    add_ltm.add_argument("--embedding-file")
    add_ltm.add_argument("--source")
    add_ltm.set_defaults(func=cmd_add_ltm)

    search_ltm = sp.add_parser("search-ltm", help="Search LTM by text or vector")
    search_ltm.add_argument("--query")
    search_ltm.add_argument("--vector-file")
    search_ltm.add_argument("--tags", nargs="*")
    search_ltm.add_argument("--top-k", type=int, default=5)
    search_ltm.set_defaults(func=cmd_search_ltm)

    add_snip = sp.add_parser("add-snippet", help="Add a content snippet")
    add_snip.add_argument("--title", required=True)
    add_snip.add_argument("--body", required=True)
    add_snip.add_argument("--lang", default="te", choices=["te", "en", "mixed"])
    add_snip.add_argument("--tags", nargs="*")
    add_snip.add_argument("--version", type=int, default=1)
    add_snip.add_argument("--embedding-file")
    add_snip.set_defaults(func=cmd_add_snippet)

    search_snip = sp.add_parser("search-snippet", help="Search content snippets")
    search_snip.add_argument("--query")
    search_snip.add_argument("--vector-file")
    search_snip.add_argument("--tags", nargs="*")
    search_snip.add_argument("--top-k", type=int, default=5)
    search_snip.set_defaults(func=cmd_search_snippet)

    set_persona = sp.add_parser("set-persona", help="Set persona from JSON file")
    set_persona.add_argument("--file", required=True)
    set_persona.set_defaults(func=cmd_set_persona)

    get_persona = sp.add_parser("get-persona", help="Get persona JSON")
    get_persona.set_defaults(func=cmd_get_persona)

    log_inter = sp.add_parser("log-interaction", help="Append an interaction row")
    log_inter.add_argument("--session-id", required=True)
    log_inter.add_argument("--user-msg", required=True)
    log_inter.add_argument("--model-reply", required=True)
    log_inter.add_argument("--used-ltm-ids", nargs="*")
    log_inter.add_argument("--used-snippet-ids", nargs="*")
    log_inter.add_argument("--stm-summary")
    log_inter.add_argument("--lang", default="te", choices=["te", "en", "mixed"])
    log_inter.set_defaults(func=cmd_log_interaction)

    upsert_stm = sp.add_parser(
        "upsert-stm", help="Create/update a short-term session row"
    )
    upsert_stm.add_argument("--session-id", required=True)
    upsert_stm.add_argument("--rolling-summary", required=True)
    upsert_stm.add_argument(
        "--target-lang", default="te", choices=["te", "en", "mixed"]
    )
    upsert_stm.set_defaults(func=cmd_upsert_stm)

    return p


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
