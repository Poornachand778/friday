import re, json, pathlib, tqdm

RAW = pathlib.Path("data/storytelling/appreciation_to_writers_raw.md").read_text(
    "utf-8"
)
OUT_DIR = pathlib.Path("data/clean_chunks/storytelling")
OUT_DIR.mkdir(parents=True, exist_ok=True)

vocab, quotes, refl = [], [], []

# 1️⃣ Vocabulary lines:  "word - definition" or "# word"
vocab_re = re.compile(r"^\s*([A-Za-z\-']+)\s*[-–]\s*(.+)", re.M)
for term, definition in vocab_re.findall(RAW):
    example = ""
    # look ahead for "own sentence :" or a line starting with a capital letter
    eg_match = re.search(
        rf"{re.escape(term)}.*?(?:own sentence|example)\s*:\s*(.+)", RAW, re.I
    )
    if eg_match:
        example = eg_match.group(1).strip()
    vocab.append(
        {
            "term": term.lower(),
            "definition": definition.strip(),
            "example": example,
            "tags": ["vocab"],
        }
    )

# 2️⃣ Quotes: anything between ** stars or preceded by a dash & not a definition
for line in RAW.splitlines():
    line = line.strip()
    if line.startswith("*") or (line.startswith("-") and len(line.split()) > 2):
        if line.lower().startswith(("*friend", "*euphemism")):
            continue  # headings
        quotes.append({"quote": line.lstrip("-* ").strip(), "source": "unknown"})

# 3️⃣ Reflections: lines with first-person pronouns and >8 words, not already captured
for line in RAW.splitlines():
    if (
        (" I " in f" {line} ")
        and len(line.split()) > 8
        and not line.strip().startswith("#")
    ):
        refl.append({"text": line.strip(), "mood": "raw_note"})

# 💾 dump
json.dump(
    vocab, open(OUT_DIR / "vocab.jsonl", "w", encoding="utf-8"), ensure_ascii=False
)
json.dump(
    quotes, open(OUT_DIR / "quotes.jsonl", "w", encoding="utf-8"), ensure_ascii=False
)
json.dump(
    refl, open(OUT_DIR / "reflections.jsonl", "w", encoding="utf-8"), ensure_ascii=False
)

print(f"Saved {len(vocab)} vocab, {len(quotes)} quotes, {len(refl)} reflections")
