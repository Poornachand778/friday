import json
import pathlib
import requests
import time

IN = pathlib.Path("data/instructions/iteration1_train.jsonl")
OUT = pathlib.Path("data/instructions/iteration1_train.labeled.jsonl")
URL = "http://127.0.0.1:8000/v1/chat/completions"  # llama.cpp server


def gen(messages, max_tokens=512, temperature=0.7):
    r = requests.post(
        URL,
        json={
            "model": "friday-local",
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": False,
        },
        timeout=120,
    )
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"].strip()


with IN.open() as fin, OUT.open("w") as fout:
    for line in fin:
        if not line.strip():
            continue
        rec = json.loads(line)
        msgs = rec["messages"]
        # Check if assistant response needs generation/replacement
        needs_generation = True
        for m in msgs:
            if m["role"] == "assistant":
                content = m["content"].strip()
                # Skip if completely empty or just "..."
                if not content or content == "...":
                    continue
                # Check if it's a template with placeholders that need filling
                if "..." in content:
                    continue  # This is a template, needs generation
                # Check for other placeholder patterns
                if any(
                    pattern in content
                    for pattern in [
                        "...",
                        "First, guard the core beat; then...",
                        "Arjun: ...",
                        "Summary: ...",
                        "Take: ...",
                        "Definition: ...",
                        "Examples:",
                    ]
                ):
                    continue  # This is a template, needs generation
                # If we get here, it's a real response
                needs_generation = False
                break

        if needs_generation:
            # keep system+user, ask llama.cpp to draft assistant
            sys = [m for m in msgs if m["role"] == "system"]
            usr = [m for m in msgs if m["role"] == "user"]
            content = gen(sys + usr)
            rec["messages"] = sys + usr + [{"role": "assistant", "content": content}]
            time.sleep(0.2)  # be gentle
        fout.write(json.dumps(rec, ensure_ascii=False) + "\n")

print(f"✅ wrote labeled file → {OUT}")
