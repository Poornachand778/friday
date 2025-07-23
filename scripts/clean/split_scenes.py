from utils.md_parser import split_scenes

import re, pathlib, json, tqdm

SOURCE = pathlib.Path("data/film/scripts/aa_janta_naduma_draft.md")
OUTDIR = pathlib.Path("data/clean_chunks/film")
OUTDIR.mkdir(parents=True, exist_ok=True)

text = SOURCE.read_text("utf-8")

# 1) simple split on 'Scene', 'Scene :' or angle-bracket scene markers
scene_regex = re.compile(r"(?:^|\n)(?:<\s*Scene.*?>|Scene\s*\d+)", re.IGNORECASE)
scenes = scene_regex.split(text)
scenes = [s.strip() for s in scenes if s.strip()]

for idx, scene in enumerate(scenes):
    # tag profanity: crude check
    has_swear = bool(re.search(r"\b(fuck|douche|shit)\b", scene, re.I))
    meta = {
        "scene_id": idx,
        "contains_profanity": has_swear,
        "lang": "en+te",
        "source": "aa_janta_naduma",
    }
    (OUTDIR / f"scene_{idx:03d}.json").write_text(
        json.dumps({"meta": meta, "text": scene}, ensure_ascii=False)
    )
