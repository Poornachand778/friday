import json, pathlib, tqdm, random

INDIR = pathlib.Path("data/clean_chunks/film")
OUT = pathlib.Path("data/instructions/v0.1/film.jsonl")
OUT.parent.mkdir(parents=True, exist_ok=True)

system_prompt = "You are Friday, a witty Telugu–English film-production assistant."


def make_pair(scene_txt: str):
    # simple synthetic question
    q_templates = [
        "Summarise this scene in 3 crisp bullet points.",
        "Rewrite the dialogue adding Chandler-style sarcasm where appropriate.",
        "Identify production tasks needed to shoot this scene.",
    ]
    q = random.choice(q_templates)
    return {
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": q + "\n\n[SCENE]\n" + scene_txt},
            {"role": "assistant", "content": ""},  # blank for now
        ]
    }


with OUT.open("w", encoding="utf-8") as fout:
    for fp in tqdm.tqdm(INDIR.glob("scene_*.json")):
        scene = json.loads(fp.read_text())["text"]
        fout.write(json.dumps(make_pair(scene), ensure_ascii=False) + "\n")
