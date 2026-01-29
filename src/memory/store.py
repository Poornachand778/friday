import json
import math
import os
import uuid
from datetime import datetime, timezone
from typing import Dict, Iterable, List, Optional


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _cosine(a: List[float], b: List[float]) -> float:
    if not a or not b or len(a) != len(b):
        return -1.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0 or nb == 0:
        return -1.0
    return dot / (na * nb)


def _tokenize(text: str) -> List[str]:
    # Simple, language-agnostic tokenization
    return [
        t.lower()
        for t in "".join(ch if ch.isalnum() else " " for ch in text).split()
        if t
    ]


class MemoryStore:
    """File-based memory store for persona, principles, LTM, snippets, STM, and interactions."""

    def __init__(self, root_dir: str = "memory") -> None:
        self.root = root_dir
        self.paths = {
            "persona": os.path.join(self.root, "data/persona/profile.json"),
            "principles": os.path.join(self.root, "data/principles/rules.json"),
            "ltm": os.path.join(self.root, "data/ltm_memories.jsonl"),
            "snippets": os.path.join(self.root, "data/content_snippets.jsonl"),
            "stm": os.path.join(self.root, "data/stm_sessions.jsonl"),
            "interactions": os.path.join(self.root, "data/interactions.jsonl"),
            "sft": os.path.join(self.root, "data/sft_datasets.jsonl"),
            "dpo": os.path.join(self.root, "data/dpo_pairs.jsonl"),
            "eval_suites": os.path.join(self.root, "data/eval_suites.jsonl"),
            "eval_cases": os.path.join(self.root, "data/eval_cases.jsonl"),
            "eval_runs": os.path.join(self.root, "data/eval_runs.jsonl"),
            "eval_results": os.path.join(self.root, "data/eval_results.jsonl"),
            "adapters": os.path.join(self.root, "data/adapters.jsonl"),
        }

    # ---------- File IO helpers ----------
    def _read_json(self, path: str) -> Dict:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    def _write_json(self, path: str, obj: Dict) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(obj, f, ensure_ascii=False, indent=2)

    def _iter_jsonl(self, path: str) -> Iterable[Dict]:
        if not os.path.exists(path):
            return
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    continue

    def _append_jsonl(self, path: str, row: Dict) -> str:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
        return row.get("id", "")

    # ---------- Persona & Principles ----------
    def get_persona(self) -> Dict:
        return self._read_json(self.paths["persona"])

    def set_persona(self, persona: Dict) -> None:
        persona = dict(persona)
        persona["updated_at"] = _utc_now()
        self._write_json(self.paths["persona"], persona)

    def get_principles(self) -> Dict:
        return self._read_json(self.paths["principles"])

    def set_principles(self, rules: Dict) -> None:
        obj = dict(rules)
        obj["updated_at"] = _utc_now()
        self._write_json(self.paths["principles"], obj)

    # ---------- Long-Term Memories ----------
    def add_ltm_memory(
        self,
        text: str,
        *,
        lang: str = "te",
        tags: Optional[List[str]] = None,
        trust: int = 3,
        embedding: Optional[List[float]] = None,
        source: Optional[str] = None,
    ) -> str:
        mid = uuid.uuid4().hex
        now = _utc_now()
        row = {
            "id": mid,
            "text": text,
            "lang": lang,
            "tags": tags or [],
            "trust": int(trust),
            "embedding": embedding,
            "source": source,
            "created_at": now,
            "updated_at": now,
        }
        self._append_jsonl(self.paths["ltm"], row)
        return mid

    def search_ltm(
        self,
        *,
        query: Optional[str] = None,
        query_vector: Optional[List[float]] = None,
        tags: Optional[List[str]] = None,
        top_k: int = 5,
    ) -> List[Dict]:
        rows = list(self._iter_jsonl(self.paths["ltm"]))
        if not rows:
            return []

        # Filter by tags first (subset match)
        if tags:
            tagset = set(t.lower() for t in tags)
            rows = [
                r
                for r in rows
                if tagset.issubset(set(x.lower() for x in r.get("tags", [])))
            ]
            if not rows:
                return []

        scored: List[tuple] = []
        if query_vector is not None:
            for r in rows:
                v = r.get("embedding")
                if not isinstance(v, list):
                    continue
                scored.append((_cosine(query_vector, v), r))
        else:
            # Lexical scoring by token overlap + tag bonus
            q_tokens = set(_tokenize(query or ""))
            for r in rows:
                text_tokens = set(_tokenize(r.get("text", "")))
                overlap = len(q_tokens & text_tokens)
                tag_bonus = len(set(t.lower() for t in r.get("tags", [])) & q_tokens)
                score = overlap + 0.5 * tag_bonus
                scored.append((score, r))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [r for _, r in scored[: max(1, top_k)]]

    # ---------- Content Snippets ----------
    def add_snippet(
        self,
        *,
        title: str,
        body: str,
        lang: str = "te",
        tags: Optional[List[str]] = None,
        version: int = 1,
        embedding: Optional[List[float]] = None,
        domain: str = "general",
    ) -> str:
        sid = uuid.uuid4().hex
        now = _utc_now()
        row = {
            "id": sid,
            "title": title,
            "body": body,
            "lang": lang,
            "tags": tags or [],
            "version": int(version),
            "embedding": embedding,
            "domain": domain,
            "created_at": now,
            "updated_at": now,
        }
        self._append_jsonl(self.paths["snippets"], row)
        return sid

    def search_snippets(
        self,
        *,
        query: Optional[str] = None,
        query_vector: Optional[List[float]] = None,
        tags: Optional[List[str]] = None,
        top_k: int = 5,
    ) -> List[Dict]:
        rows = list(self._iter_jsonl(self.paths["snippets"]))
        if not rows:
            return []
        if tags:
            tagset = set(t.lower() for t in tags)
            rows = [
                r
                for r in rows
                if tagset.issubset(set(x.lower() for x in r.get("tags", [])))
            ]
            if not rows:
                return []
        scored: List[tuple] = []
        if query_vector is not None:
            for r in rows:
                v = r.get("embedding")
                if not isinstance(v, list):
                    continue
                scored.append((_cosine(query_vector, v), r))
        else:
            q_tokens = set(_tokenize(query or ""))
            for r in rows:
                text_tokens = set(
                    _tokenize((r.get("title", "") + " " + r.get("body", "")).strip())
                )
                overlap = len(q_tokens & text_tokens)
                tag_bonus = len(set(t.lower() for t in r.get("tags", [])) & q_tokens)
                score = overlap + 0.5 * tag_bonus
                scored.append((score, r))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [r for _, r in scored[: max(1, top_k)]]

    # ---------- Short-term sessions & interactions ----------
    def upsert_stm_session(
        self, *, session_id: str, rolling_summary: str, target_lang: str
    ) -> str:
        # Simple upsert: load all, replace or append. For small files this is fine.
        rows = list(self._iter_jsonl(self.paths["stm"]))
        out: List[Dict] = []
        updated = False
        now = _utc_now()
        for r in rows:
            if r.get("session_id") == session_id:
                r = {
                    "session_id": session_id,
                    "rolling_summary": rolling_summary,
                    "target_lang": target_lang,
                    "updated_at": now,
                }
                updated = True
            out.append(r)
        if not updated:
            out.append(
                {
                    "session_id": session_id,
                    "rolling_summary": rolling_summary,
                    "target_lang": target_lang,
                    "updated_at": now,
                }
            )
        # rewrite file
        os.makedirs(os.path.dirname(self.paths["stm"]), exist_ok=True)
        with open(self.paths["stm"], "w", encoding="utf-8") as f:
            for r in out:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        return session_id

    def log_interaction(
        self,
        *,
        session_id: str,
        user_msg: str,
        model_reply: str,
        used_ltm_ids: Optional[List[str]] = None,
        used_snippet_ids: Optional[List[str]] = None,
        stm_summary: Optional[str] = None,
        lang: str = "te",
    ) -> str:
        iid = uuid.uuid4().hex
        row = {
            "id": iid,
            "session_id": session_id,
            "user_msg": user_msg,
            "model_reply": model_reply,
            "used_ltm_ids": used_ltm_ids or [],
            "used_snippet_ids": used_snippet_ids or [],
            "stm_summary": stm_summary,
            "lang": lang,
            "created_at": _utc_now(),
        }
        self._append_jsonl(self.paths["interactions"], row)
        return iid
