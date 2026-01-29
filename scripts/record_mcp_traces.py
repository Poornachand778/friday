#!/usr/bin/env python3
"""Record live MCP tool traces for fine-tune examples.

Creates a JSONL at data/traces/iteration2_live_traces.jsonl with two flows:
1) Reorder a "proposal" scene after a reference scene.
2) Move a confrontation scene to backlog and link as flashback.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

ROOT = Path(__file__).resolve().parents[1]
CFG = ROOT / "config" / "tools" / "scene_manager_mcp.json"
OUT = ROOT / "data" / "traces" / "iteration2_live_traces.jsonl"


def load_env_file(path: Path) -> None:
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def start_server() -> subprocess.Popen[str]:
    cfg = json.loads(CFG.read_text())
    cmd: List[str] = cfg["command"] + cfg.get("args", [])
    env = os.environ.copy()
    for k, v in (cfg.get("env") or {}).items():
        if v and v.startswith("${") and v.endswith("}"):
            env[k] = os.environ.get(v[2:-1], "")
        else:
            env[k] = v
    workdir = cfg.get("working_directory", ".")
    proc = subprocess.Popen(  # noqa: S603
        cmd,
        cwd=str(ROOT / workdir),
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=env,
    )
    assert proc.stdin and proc.stdout
    return proc


def rpc(proc: subprocess.Popen[str], msg: Dict[str, Any]) -> Dict[str, Any]:
    payload = json.dumps(msg, ensure_ascii=False)
    proc.stdin.write(payload + "\n")
    proc.stdin.flush()
    line = proc.stdout.readline()
    return json.loads(line)


def content_of(resp: Dict[str, Any]) -> Any:
    if resp.get("type") == "error":
        return {"error": resp.get("error")}
    return resp.get("result", {}).get("content")


def run() -> int:
    (ROOT / "data" / "traces").mkdir(parents=True, exist_ok=True)
    load_env_file(ROOT / ".env")
    for key in ("DB_HOST", "DB_PORT", "DB_NAME", "DB_USER", "DB_PASSWORD"):
        if not os.environ.get(key):
            print(f"Missing env: {key}", file=sys.stderr)
            return 1

    proc = start_server()
    # Initialize
    rpc(proc, {"id": 1, "method": "initialize", "params": {}})

    cases: List[Tuple[str, List[Dict[str, Any]], Dict[str, Any]]] = []

    # Flow 1: Proposal scene → reorder after SCN004
    msgs: List[Dict[str, Any]] = []
    msgs.append(
        {
            "role": "system",
            "content": "You are Friday; use MCP tools when editing screenplay scenes.",
        }
    )
    msgs.append(
        {
            "role": "user",
            "content": "Let's lock the proposal scene after the market encounter.",
        }
    )
    search = rpc(
        proc,
        {
            "id": 2,
            "method": "call_tool",
            "params": {
                "name": "scene_search",
                "arguments": {"query": "proposal scene", "top_k": 3},
            },
        },
    )
    msgs.append(
        {
            "role": "assistant",
            "content": "Fetching the proposal scene and placing it after the market beat.",
            "tool_calls": [
                {
                    "id": "call-1",
                    "type": "tool",
                    "name": "scene_search",
                    "arguments": {"query": "proposal scene", "top_k": 3},
                },
            ],
        }
    )
    msgs.append(
        {
            "role": "tool",
            "name": "scene_search",
            "content": json.dumps(content_of(search), ensure_ascii=False),
        }
    )
    top = (content_of(search) or [])[:1]
    if top:
        sc = top[0]["scene_code"]
        reorder = rpc(
            proc,
            {
                "id": 3,
                "method": "call_tool",
                "params": {
                    "name": "scene_reorder",
                    "arguments": {"scene_code": sc, "after_scene": "SCN004"},
                },
            },
        )
        msgs.append(
            {
                "role": "assistant",
                "content": f"Reordered {sc} after SCN004.",
                "tool_calls": [
                    {
                        "id": "call-2",
                        "type": "tool",
                        "name": "scene_reorder",
                        "arguments": {"scene_code": sc, "after_scene": "SCN004"},
                    },
                ],
            }
        )
        msgs.append(
            {
                "role": "tool",
                "name": "scene_reorder",
                "content": json.dumps(content_of(reorder), ensure_ascii=False),
            }
        )
    cases.append(
        ("scene_reorder_proposal", msgs, {"tools": ["scene_search", "scene_reorder"]})
    )

    # Flow 2: Confrontation → backlog + flashback link
    msgs2: List[Dict[str, Any]] = []
    msgs2.append(
        {
            "role": "system",
            "content": "You are Friday; confirm DB mutations explicitly.",
        }
    )
    msgs2.append(
        {
            "role": "user",
            "content": "Move the Niha confrontation scene to backlog and link as flashback to the hospital scene.",
        }
    )
    s2 = rpc(
        proc,
        {
            "id": 4,
            "method": "call_tool",
            "params": {
                "name": "scene_search",
                "arguments": {"query": "Niha confronts", "top_k": 2},
            },
        },
    )
    msgs2.append(
        {
            "role": "assistant",
            "content": "Locating the confrontation scene and updating status to backlog, then linking it as flashback.",
            "tool_calls": [
                {
                    "id": "call-3",
                    "type": "tool",
                    "name": "scene_search",
                    "arguments": {"query": "Niha confronts", "top_k": 2},
                },
            ],
        }
    )
    msgs2.append(
        {
            "role": "tool",
            "name": "scene_search",
            "content": json.dumps(content_of(s2), ensure_ascii=False),
        }
    )
    top2 = (content_of(s2) or [])[:1]
    if top2:
        sc2 = top2[0]["scene_code"]
        upd = rpc(
            proc,
            {
                "id": 5,
                "method": "call_tool",
                "params": {
                    "name": "scene_update",
                    "arguments": {
                        "scene_code": sc2,
                        "status": "backlog",
                        "notes": "Marked via live trace",
                    },
                },
            },
        )
        link = rpc(
            proc,
            {
                "id": 6,
                "method": "call_tool",
                "params": {
                    "name": "scene_link",
                    "arguments": {
                        "from_scene": sc2,
                        "to_scene": "SCN020",
                        "relation_type": "flashback",
                    },
                },
            },
        )
        msgs2.append(
            {
                "role": "assistant",
                "content": f"{sc2} moved to backlog and linked as flashback to SCN020.",
                "tool_calls": [
                    {
                        "id": "call-4",
                        "type": "tool",
                        "name": "scene_update",
                        "arguments": {"scene_code": sc2, "status": "backlog"},
                    },
                    {
                        "id": "call-5",
                        "type": "tool",
                        "name": "scene_link",
                        "arguments": {
                            "from_scene": sc2,
                            "to_scene": "SCN020",
                            "relation_type": "flashback",
                        },
                    },
                ],
            }
        )
        msgs2.append(
            {
                "role": "tool",
                "name": "scene_update",
                "content": json.dumps(content_of(upd), ensure_ascii=False),
            }
        )
        msgs2.append(
            {
                "role": "tool",
                "name": "scene_link",
                "content": json.dumps(content_of(link), ensure_ascii=False),
            }
        )
    cases.append(
        (
            "scene_backlog_and_flashback",
            msgs2,
            {"tools": ["scene_search", "scene_update", "scene_link"]},
        )
    )

    # Flow 3: Punch up proposal scene text and confirm revision
    msgs3: List[Dict[str, Any]] = []
    msgs3.append(
        {
            "role": "system",
            "content": "You are Friday; after editing, confirm the new revision snapshot.",
        }
    )
    msgs3.append(
        {
            "role": "user",
            "content": "Refresh the proposal scene with a sharper closing beat and confirm the revision.",
        }
    )
    s3 = rpc(
        proc,
        {
            "id": 7,
            "method": "call_tool",
            "params": {
                "name": "scene_search",
                "arguments": {"query": "coffee shop proposal", "top_k": 1},
            },
        },
    )
    msgs3.append(
        {
            "role": "assistant",
            "content": "Searching for the proposal beat before revising the closing line.",
            "tool_calls": [
                {
                    "id": "call-6",
                    "type": "tool",
                    "name": "scene_search",
                    "arguments": {"query": "coffee shop proposal", "top_k": 1},
                },
            ],
        }
    )
    msgs3.append(
        {
            "role": "tool",
            "name": "scene_search",
            "content": json.dumps(content_of(s3), ensure_ascii=False),
        }
    )
    top3 = (content_of(s3) or [])[:1]
    if top3:
        sc3 = top3[0]["scene_code"]
        detail_before = rpc(
            proc,
            {
                "id": 8,
                "method": "call_tool",
                "params": {"name": "scene_get", "arguments": {"scene_code": sc3}},
            },
        )
        detail_payload = content_of(detail_before) or {}
        orig_text = detail_payload.get("canonical_text", "")
        revised_text = (
            orig_text
            + '\n\n# Revised closing beat\nNiha: "Boss, coffee refill only; feelings later."'
        )
        update_resp = rpc(
            proc,
            {
                "id": 9,
                "method": "call_tool",
                "params": {
                    "name": "scene_update",
                    "arguments": {
                        "scene_code": sc3,
                        "canonical_text": revised_text,
                        "notes": "Sharpened closing beat via live trace",
                    },
                },
            },
        )
        detail_after = rpc(
            proc,
            {
                "id": 10,
                "method": "call_tool",
                "params": {"name": "scene_get", "arguments": {"scene_code": sc3}},
            },
        )
        msgs3.append(
            {
                "role": "assistant",
                "content": f"Revision applied to {sc3}; refreshed embedding and captured the latest snapshot.",
                "tool_calls": [
                    {
                        "id": "call-7",
                        "type": "tool",
                        "name": "scene_get",
                        "arguments": {"scene_code": sc3},
                    },
                    {
                        "id": "call-8",
                        "type": "tool",
                        "name": "scene_update",
                        "arguments": {"scene_code": sc3},
                    },
                    {
                        "id": "call-9",
                        "type": "tool",
                        "name": "scene_get",
                        "arguments": {"scene_code": sc3},
                    },
                ],
            }
        )
        msgs3.append(
            {
                "role": "tool",
                "name": "scene_get",
                "content": json.dumps(content_of(detail_before), ensure_ascii=False),
            }
        )
        msgs3.append(
            {
                "role": "tool",
                "name": "scene_update",
                "content": json.dumps(content_of(update_resp), ensure_ascii=False),
            }
        )
        msgs3.append(
            {
                "role": "tool",
                "name": "scene_get",
                "content": json.dumps(content_of(detail_after), ensure_ascii=False),
            }
        )
    cases.append(
        (
            "scene_revision_confirm",
            msgs3,
            {"tools": ["scene_search", "scene_get", "scene_update"]},
        )
    )

    # Flow 4: Revert revision to original text
    msgs4: List[Dict[str, Any]] = []
    msgs4.append(
        {
            "role": "system",
            "content": "You are Friday; undo the last textual tweak and confirm reversion.",
        }
    )
    msgs4.append(
        {
            "role": "user",
            "content": "Undo the proposal scene punch-up and restore the earlier wording.",
        }
    )
    s4 = rpc(
        proc,
        {
            "id": 11,
            "method": "call_tool",
            "params": {
                "name": "scene_search",
                "arguments": {"query": "coffee shop proposal", "top_k": 1},
            },
        },
    )
    msgs4.append(
        {
            "role": "assistant",
            "content": "Checking the proposal scene before reverting the closing beat.",
            "tool_calls": [
                {
                    "id": "call-10",
                    "type": "tool",
                    "name": "scene_search",
                    "arguments": {"query": "coffee shop proposal", "top_k": 1},
                },
            ],
        }
    )
    msgs4.append(
        {
            "role": "tool",
            "name": "scene_search",
            "content": json.dumps(content_of(s4), ensure_ascii=False),
        }
    )
    top4 = (content_of(s4) or [])[:1]
    if top4:
        sc4 = top4[0]["scene_code"]
        before_revert = rpc(
            proc,
            {
                "id": 12,
                "method": "call_tool",
                "params": {"name": "scene_get", "arguments": {"scene_code": sc4}},
            },
        )
        before_payload = content_of(before_revert) or {}
        current_text = before_payload.get("canonical_text", "")
        anchor = "\n\n# Revised closing beat"
        original_text = current_text.split(anchor)[0].rstrip()
        revert_text = original_text + "\n"
        revert_resp = rpc(
            proc,
            {
                "id": 13,
                "method": "call_tool",
                "params": {
                    "name": "scene_update",
                    "arguments": {
                        "scene_code": sc4,
                        "canonical_text": revert_text,
                        "notes": "Reverted to pre-punch-up wording",
                    },
                },
            },
        )
        after_revert = rpc(
            proc,
            {
                "id": 14,
                "method": "call_tool",
                "params": {"name": "scene_get", "arguments": {"scene_code": sc4}},
            },
        )
        msgs4.append(
            {
                "role": "assistant",
                "content": f"Reverted {sc4} to the earlier text and confirmed the snapshot.",
                "tool_calls": [
                    {
                        "id": "call-11",
                        "type": "tool",
                        "name": "scene_get",
                        "arguments": {"scene_code": sc4},
                    },
                    {
                        "id": "call-12",
                        "type": "tool",
                        "name": "scene_update",
                        "arguments": {"scene_code": sc4},
                    },
                    {
                        "id": "call-13",
                        "type": "tool",
                        "name": "scene_get",
                        "arguments": {"scene_code": sc4},
                    },
                ],
            }
        )
        msgs4.append(
            {
                "role": "tool",
                "name": "scene_get",
                "content": json.dumps(content_of(before_revert), ensure_ascii=False),
            }
        )
        msgs4.append(
            {
                "role": "tool",
                "name": "scene_update",
                "content": json.dumps(content_of(revert_resp), ensure_ascii=False),
            }
        )
        msgs4.append(
            {
                "role": "tool",
                "name": "scene_get",
                "content": json.dumps(content_of(after_revert), ensure_ascii=False),
            }
        )
    cases.append(
        (
            "scene_revision_revert",
            msgs4,
            {"tools": ["scene_search", "scene_get", "scene_update"]},
        )
    )

    # Flow 5: Toggle status from backlog to active and back
    msgs5: List[Dict[str, Any]] = []
    msgs5.append(
        {
            "role": "system",
            "content": "You are Friday; demonstrate status toggle with confirmations.",
        }
    )
    msgs5.append(
        {
            "role": "user",
            "content": "Toggle the confrontation scene back to active and then park it in backlog again.",
        }
    )
    s5 = rpc(
        proc,
        {
            "id": 15,
            "method": "call_tool",
            "params": {
                "name": "scene_search",
                "arguments": {"query": "Niha confronts", "top_k": 1},
            },
        },
    )
    msgs5.append(
        {
            "role": "assistant",
            "content": "Finding the confrontation scene before flipping its status.",
            "tool_calls": [
                {
                    "id": "call-14",
                    "type": "tool",
                    "name": "scene_search",
                    "arguments": {"query": "Niha confronts", "top_k": 1},
                },
            ],
        }
    )
    msgs5.append(
        {
            "role": "tool",
            "name": "scene_search",
            "content": json.dumps(content_of(s5), ensure_ascii=False),
        }
    )
    top5 = (content_of(s5) or [])[:1]
    if top5:
        sc5 = top5[0]["scene_code"]
        before_toggle = rpc(
            proc,
            {
                "id": 16,
                "method": "call_tool",
                "params": {"name": "scene_get", "arguments": {"scene_code": sc5}},
            },
        )
        before_status = (content_of(before_toggle) or {}).get("status", "backlog")
        first_status = "active" if before_status != "active" else "backlog"
        to_active = rpc(
            proc,
            {
                "id": 17,
                "method": "call_tool",
                "params": {
                    "name": "scene_update",
                    "arguments": {
                        "scene_code": sc5,
                        "status": first_status,
                        "notes": "Status toggle demonstration",
                    },
                },
            },
        )
        mid_detail = rpc(
            proc,
            {
                "id": 18,
                "method": "call_tool",
                "params": {"name": "scene_get", "arguments": {"scene_code": sc5}},
            },
        )
        reset_status = rpc(
            proc,
            {
                "id": 19,
                "method": "call_tool",
                "params": {
                    "name": "scene_update",
                    "arguments": {
                        "scene_code": sc5,
                        "status": before_status,
                        "notes": "Restored original status",
                    },
                },
            },
        )
        final_detail = rpc(
            proc,
            {
                "id": 20,
                "method": "call_tool",
                "params": {"name": "scene_get", "arguments": {"scene_code": sc5}},
            },
        )
        msgs5.append(
            {
                "role": "assistant",
                "content": f"Status flipped for {sc5} and then restored to {before_status}.",
                "tool_calls": [
                    {
                        "id": "call-15",
                        "type": "tool",
                        "name": "scene_get",
                        "arguments": {"scene_code": sc5},
                    },
                    {
                        "id": "call-16",
                        "type": "tool",
                        "name": "scene_update",
                        "arguments": {"scene_code": sc5},
                    },
                    {
                        "id": "call-17",
                        "type": "tool",
                        "name": "scene_get",
                        "arguments": {"scene_code": sc5},
                    },
                    {
                        "id": "call-18",
                        "type": "tool",
                        "name": "scene_update",
                        "arguments": {"scene_code": sc5},
                    },
                    {
                        "id": "call-19",
                        "type": "tool",
                        "name": "scene_get",
                        "arguments": {"scene_code": sc5},
                    },
                ],
            }
        )
        msgs5.append(
            {
                "role": "tool",
                "name": "scene_get",
                "content": json.dumps(content_of(before_toggle), ensure_ascii=False),
            }
        )
        msgs5.append(
            {
                "role": "tool",
                "name": "scene_update",
                "content": json.dumps(content_of(to_active), ensure_ascii=False),
            }
        )
        msgs5.append(
            {
                "role": "tool",
                "name": "scene_get",
                "content": json.dumps(content_of(mid_detail), ensure_ascii=False),
            }
        )
        msgs5.append(
            {
                "role": "tool",
                "name": "scene_update",
                "content": json.dumps(content_of(reset_status), ensure_ascii=False),
            }
        )
        msgs5.append(
            {
                "role": "tool",
                "name": "scene_get",
                "content": json.dumps(content_of(final_detail), ensure_ascii=False),
            }
        )
    cases.append(
        (
            "scene_status_toggle",
            msgs5,
            {"tools": ["scene_search", "scene_get", "scene_update"]},
        )
    )

    # Shutdown
    rpc(proc, {"id": 21, "method": "shutdown", "params": {}})

    with OUT.open("w", encoding="utf-8") as fh:
        for name, messages, meta in cases:
            fh.write(
                json.dumps(
                    {
                        "name": name,
                        "messages": messages,
                        "metadata": {"domain": "film", **meta},
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
    print(f"Wrote traces to {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(run())
