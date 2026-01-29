#!/usr/bin/env python3
"""Quick smoke test for the MCP scene manager server.

Starts the local MCP server as a subprocess, runs initialize/list/call_tool,
prints formatted responses, and then shuts the server down. The script expects
DB credentials in the environment (for example via the project's .env file).
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict

ROOT = Path(__file__).resolve().parents[1]
SERVER = ROOT / "mcp" / "scene_manager" / "server.py"
DEFAULT_PROJECT = os.environ.get("MCP_TEST_PROJECT", "aa-janta-naduma")
SCENE_EMBED_MODEL = os.environ.get(
    "SCENE_EMBED_MODEL", "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
)


def load_env_file(path: Path) -> None:
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def pretty(obj: Dict[str, Any]) -> str:
    return json.dumps(obj, ensure_ascii=False, indent=2)


def main() -> int:
    load_env_file(ROOT / ".env")

    for key in ("DB_HOST", "DB_PORT", "DB_NAME", "DB_USER", "DB_PASSWORD"):
        if not os.environ.get(key):
            print(f"Missing required env var: {key}", file=sys.stderr)
            return 1

    env = os.environ.copy()
    env.setdefault("SCENE_EMBED_MODEL", SCENE_EMBED_MODEL)

    proc = subprocess.Popen(  # noqa: S603
        [sys.executable, str(SERVER), "--default-project", DEFAULT_PROJECT],
        cwd=str(ROOT),
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=env,
        text=True,
    )

    if proc.stdin is None or proc.stdout is None:
        print("Failed to open MCP server pipes", file=sys.stderr)
        return 1

    requests = [
        {"id": 1, "method": "initialize", "params": {}},
        {"id": 2, "method": "list_tools", "params": {}},
        {
            "id": 3,
            "method": "call_tool",
            "params": {
                "name": "scene_search",
                "arguments": {"query": "proposal scene", "top_k": 1},
            },
        },
        {"id": 4, "method": "shutdown", "params": {}},
    ]

    for req in requests:
        payload = json.dumps(req, ensure_ascii=False)
        proc.stdin.write(payload + "\n")
        proc.stdin.flush()
        line = proc.stdout.readline()
        if not line:
            print("Server terminated unexpectedly", file=sys.stderr)
            proc.terminate()
            return 1
        try:
            response = json.loads(line)
        except json.JSONDecodeError as exc:  # pragma: no cover - diagnostic aid
            print(f"Invalid JSON response: {line}\n{exc}", file=sys.stderr)
            proc.terminate()
            return 1
        print(f"Response {response.get('id')}:\n{pretty(response)}\n")

    stderr = proc.stderr.read().strip() if proc.stderr else ""
    if stderr:
        print("Server stderr:\n" + stderr, file=sys.stderr)

    return proc.wait(timeout=5)


if __name__ == "__main__":
    raise SystemExit(main())
