#!/usr/bin/env python3
"""Probe MCP tool registration and list available tools via config JSON.

Reads config/tools/scene_manager_mcp.json, spawns the configured command,
performs initialize + list_tools, prints tool names, and exits.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[1]
CFG = ROOT / "config" / "tools" / "scene_manager_mcp.json"


def load_env_file(path: Path) -> None:
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def main() -> int:
    cfg = json.loads(CFG.read_text())

    load_env_file(ROOT / ".env")

    cmd: List[str] = cfg["command"]
    args: List[str] = cfg.get("args", [])
    workdir = cfg.get("working_directory", ".")

    env = os.environ.copy()
    for k, v in (cfg.get("env") or {}).items():
        # Allow passthrough of ${VAR}
        if v and v.startswith("${") and v.endswith("}"):
            key = v[2:-1]
            env[k] = os.environ.get(key, "")
        else:
            env[k] = v

    proc = subprocess.Popen(  # noqa: S603
        cmd + args,
        cwd=str(ROOT / workdir),
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=env,
    )
    assert proc.stdin and proc.stdout

    init = json.dumps(
        {"id": 1, "method": "initialize", "params": {}}, ensure_ascii=False
    )
    lst = json.dumps(
        {"id": 2, "method": "list_tools", "params": {}}, ensure_ascii=False
    )
    bye = json.dumps({"id": 3, "method": "shutdown", "params": {}}, ensure_ascii=False)

    proc.stdin.write(init + "\n")
    proc.stdin.flush()
    print(proc.stdout.readline().strip())

    proc.stdin.write(lst + "\n")
    proc.stdin.flush()
    line = proc.stdout.readline().strip()
    obj = json.loads(line)
    names = [t["name"] for t in obj.get("result", {}).get("tools", [])]
    print("tools:", ", ".join(names))

    proc.stdin.write(bye + "\n")
    proc.stdin.flush()
    print(proc.stdout.readline().strip())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
