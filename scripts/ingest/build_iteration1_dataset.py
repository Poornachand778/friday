import json, random, pathlib

root = pathlib.Path("data")
out = root / "instructions" / "iteration1_train.jsonl"

datasets = [
    root / "instructions" / "snippets_chatml.jsonl",
    root / "clean_chunks" / "film" / "scenes",
    root / "clean_chunks" / "vocab" / "appreciation.jsonl",
]

pairs = []

# 1) dialogue snippets (already instruction/response)
with open(datasets[0]) as fh:
    for line in fh:
        pairs.append(json.loads(line))

# 2) scenes -> ask to summarise
for scene_file in sorted((datasets[1]).glob("scene_*.json")):
    blob = json.load(open(scene_file))
    pairs.append(
        {
            "instruction": f"Summarise scene {blob['meta']['scene_id']} briefly in English & Telugu.",
            "response": blob["text"][:2048],  # quick hack; we’ll learn summary token
        }
    )

# 3) vocab -> simple Q&A
with open(datasets[2]) as fh:
    for row in map(json.loads, fh):
        pairs.append(
            {
                "instruction": f"Define the word '{row['word']}'.",
                "response": row["definition"],
            }
        )

random.shuffle(pairs)
with open(out, "w") as fh:
    for p in pairs:
        fh.write(json.dumps(p, ensure_ascii=False) + "\n")

print("Pairs →", len(pairs), "saved to", out)
