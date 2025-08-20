import json
import re
import pathlib
import random
from datetime import datetime

ROOT = pathlib.Path(".")
OUT = ROOT / "data" / "instructions"
OUT.mkdir(parents=True, exist_ok=True)

# Inputs
SCENE_DIR = ROOT / "data" / "clean_chunks" / "film" / "scenes"
APPREC = (
    ROOT / "data" / "clean_chunks" / "vocab" / "appreciation.jsonl"
)  # if you created it earlier
MORALS = ROOT / "data" / "persona" / "morals_beliefs.md"
DECISIONS = ROOT / "data" / "persona" / "decision_scenarios_seed.md"

SYSTEM_PROMPT = (
    "You are Friday, Poorna’s personal assistant.\n"
    "Voice: curious, witty, Telugu+English blend when natural, direct but warm.\n"
    "If user uses Telugu, you may reply bilingually.\n"
    "Prefer concrete steps, examples, and film/cooking analogies.\n"
)


def read_jsonl(p):
    if not p.exists():
        return []
    return [json.loads(x) for x in p.read_text("utf-8").splitlines() if x.strip()]


def write_jsonl(path, rows):
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def scene_pairs():
    rows = []
    for fp in sorted(SCENE_DIR.glob("scene_*.json")):
        obj = json.loads(fp.read_text("utf-8"))
        text = obj.get("text", "").strip()
        if not text:
            continue

        # 1) Summarize + suggest punch-up
        user = (
            "Summarize the scene briefly, then suggest 3 punch-up options for dialogue "
            "(keep Telugu+English vibe, keep character intent). Scene:\n\n" + text
        )
        assistant = "Summary: ...\nPunch-ups:\n1) ...\n2) ...\n3) ..."
        rows.append(
            chatml(
                user,
                assistant,
                domain="film",
                task="scene_punchup",
                source=str(fp.name),
            )
        )

        # 2) Continue scene (one beat)
        user2 = (
            "Continue this scene with one natural beat (4–8 lines), keeping tone and language mix. "
            "Avoid cliches; end on a small reveal or comic turn. Scene:\n\n" + text
        )
        assistant2 = "Arjun: ...\nSwarna: ...\n[beat]\nArjun (aside): ..."
        rows.append(
            chatml(
                user2,
                assistant2,
                domain="film",
                task="scene_continue",
                source=str(fp.name),
            )
        )
    return rows


def vocab_pairs():
    rows = []
    items = read_jsonl(APPREC)
    for it in items:
        word = it.get("word") or it.get("term")
        definition = it.get("definition", "").strip()
        example = it.get("example", "").strip()
        if not word or not definition:
            continue

        # Define + witty example
        user = (
            f"Define '{word}' crisply and give 2 witty examples (one Telugu-flavored)."
        )
        assistant = f"Definition: {definition}\nExamples:\n1) ...\n2) ..."
        rows.append(
            chatml(
                user,
                assistant,
                domain="language",
                task="define_witty",
                meta={"word": word},
            )
        )

        # Use in story line
        user2 = (
            f"Use '{word}' in a one-line character beat suitable for a rom-com scene."
        )
        assistant2 = "She shoots him a look, the veneer cracking just enough to show the hurt underneath."
        rows.append(
            chatml(
                user2,
                assistant2,
                domain="film",
                task="use_in_scene",
                meta={"word": word},
            )
        )
    return rows


def morals_pairs():
    rows = []
    if not MORALS.exists():
        return rows
    text = MORALS.read_text("utf-8")
    bullets = [
        re.sub(r"^\s*\d+\.\s*", "", ln).strip()
        for ln in text.splitlines()
        if ln.strip()
    ]
    for i, b in enumerate(bullets, 1):
        # Explain principle + give example
        user = f"Explain this principle in your voice and give a grounded example: {b}"
        assistant = f"Take: ...\nExample: ..."
        rows.append(
            chatml(
                user,
                assistant,
                domain="persona",
                task="principle_explain",
                meta={"idx": i},
            )
        )

        # Decision guidance from principle
        user2 = f"Given the principle '{b}', how would you advise a junior AD facing a tough call on set?"
        assistant2 = "First, guard the core beat; then..."
        rows.append(
            chatml(
                user2,
                assistant2,
                domain="production",
                task="principle_to_decision",
                meta={"idx": i},
            )
        )
    return rows


def decision_pairs():
    rows = []
    ds = read_jsonl(DECISIONS)
    for it in ds:
        sit = it["situation"]
        choice = it.get("choice", "")
        rationale = it.get("rationale", "")
        # 1) Ask-for-decision
        user = f"Scenario: {sit}\nPick and justify your choice briefly."
        assistant = f"Choice: {choice}\nWhy: {rationale}"
        rows.append(
            chatml(
                user,
                assistant,
                domain=it.get("domain", "general"),
                task="decision_playback",
                meta={"id": it["scenario_id"]},
            )
        )
        # 2) Ask for structure/heuristics
        heurs = ", ".join(it.get("heuristics", []))
        user2 = f"Scenario (same): {sit}\nList the heuristics and risks you weighed."
        assistant2 = f"Heuristics: {heurs or '...'}\nRisks: ..."
        rows.append(
            chatml(
                user2,
                assistant2,
                domain=it.get("domain", "general"),
                task="decision_heuristics",
                meta={"id": it["scenario_id"]},
            )
        )
    return rows


def chatml(user, assistant, domain="general", task="", meta=None, source=None):
    rec = {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user},
            {"role": "assistant", "content": assistant},
        ],
        "metadata": {
            "domain": domain,
            "task": task,
            "created_at": datetime.utcnow().isoformat() + "Z",
        },
    }
    if meta:
        rec["metadata"].update(meta)
    if source:
        rec["metadata"]["source"] = source
    return rec


def split_train_valid(rows, valid_ratio=0.1):
    random.seed(42)
    random.shuffle(rows)
    n_valid = max(1, int(len(rows) * valid_ratio))
    return rows[n_valid:], rows[:n_valid]


def main():
    all_rows = []
    all_rows += scene_pairs()
    all_rows += vocab_pairs()
    all_rows += morals_pairs()
    all_rows += decision_pairs()

    train, valid = split_train_valid(all_rows, 0.1)
    OUT.mkdir(parents=True, exist_ok=True)
    write_jsonl(OUT / "iteration1_train.jsonl", train)
    write_jsonl(OUT / "iteration1_valid.jsonl", valid)
    print(f"✅ Built {len(train)} train and {len(valid)} valid pairs → {OUT}")


if __name__ == "__main__":
    main()
