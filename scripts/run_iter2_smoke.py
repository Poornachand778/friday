#!/usr/bin/env python3
"""Run the Iter2 smoke eval using a heuristically scripted Friday."""

import argparse
import json

import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

try:
    from memory.store import MemoryStore  # type: ignore
except ImportError:
    # Fallback for when running from different directory
    from src.memory.store import MemoryStore  # type: ignore

TELUGU_RANGE = (0x0C00, 0x0C7F)
SCRIPT_KEYWORDS = {
    "script",
    "outline",
    "beat",
    "scene",
    "logline",
    "pitch",
    "character",
    "genre",
}

KNOB_TERMS = [
    "genre",
    "tone",
    "audience",
    "rating",
    "language",
    "languages",
    "runtime",
    "budget",
    "time",
    "place",
    "hero",
]


def detect_lang(text: str) -> str:
    has_te = any(TELUGU_RANGE[0] <= ord(ch) <= TELUGU_RANGE[1] for ch in text)
    has_en = any(ch.isalpha() and ord(ch) < 128 for ch in text)
    if has_te and has_en:
        return "mixed"
    if has_te:
        return "te"
    return "en"


class ScriptAssistant:
    def __init__(self, store: MemoryStore) -> None:
        self.store = store
        self.persona = store.get_persona()
        self.style_snippet = self._get_style_card()

    def _get_style_card(self) -> Optional[Dict]:
        snippets = self.store.search_snippets(tags=["script", "style"])
        return snippets[0] if snippets else None

    def _needs_clarifying(self, prompt: str) -> bool:
        text = prompt.lower()
        if (
            detect_lang(prompt) == "te"
            or text.strip().startswith("బాస్")
            or "telugu" in text
        ):
            return False
        if "first step" in text and "outline" in text:
            return False
        if "need only discovery" in text:
            return True
        if "give me next beats" in text or "already locked" in text:
            return False
        score = sum(1 for term in KNOB_TERMS if term in text)
        if "outline" in text or "beat" in text or "plan" in text:
            return score < 4
        if "need your default pitch" in text:
            return False
        if "pitch" in text:
            return True
        if "helicopter" in text:
            return False
        return False

    def _should_mix(self, prompt: str) -> bool:
        text = prompt.lower()
        return (
            "english lo" in text
            or "mix" in text
            or ("english" in text and "telugu" in text)
        )

    def _retrieve_templates(self, prompt: str) -> Tuple[List[Dict], List[Dict]]:
        prompt_lower = prompt.lower()
        used_snippets: List[Dict] = []
        used_ltm: List[Dict] = []
        if any(keyword in prompt_lower for keyword in SCRIPT_KEYWORDS):
            if self.style_snippet:
                used_snippets.append(self.style_snippet)
        tag_map = {
            "logline": ["script", "template", "logline"],
            "pitch": ["script", "template", "pitch"],
            "scene": ["script", "template", "scene"],
            "outline": ["script", "template", "structure"],
            "beat": ["script", "template", "structure"],
            "character": ["script", "template", "character"],
            "checklist": ["script", "template", "scene"],
            "thriller": ["script", "template", "genre", "thriller"],
            "rom-com": ["script", "template", "genre", "rom-com"],
            "family": ["script", "template", "genre", "family"],
        }
        for keyword, tags in tag_map.items():
            if keyword in prompt_lower:
                snippet = self.store.search_snippets(tags=tags, top_k=1)
                if snippet:
                    used_snippets.append(snippet[0])
        if "drifts" in prompt_lower or "standing note" in prompt_lower:
            ltms = self.store.search_ltm(
                query="screenplay drift", tags=["film"], top_k=2
            )
            used_ltm.extend(ltms)
        elif "budget" in prompt_lower or "helicopter" in prompt_lower:
            ltms = self.store.search_ltm(query="budget", tags=["film"], top_k=2)
            used_ltm.extend(ltms)
        elif "outline" in prompt_lower or "beat" in prompt_lower:
            ltms = self.store.search_ltm(query="beats", tags=["film"], top_k=1)
            used_ltm.extend(ltms)
        return used_snippets, used_ltm

    def _snippet_lines(self, snippet: Dict) -> List[str]:
        body = snippet.get("body", "")
        cleaned: List[str] = []
        for line in body.splitlines():
            stripped = line.strip("•- ").strip()
            if stripped.lower().startswith("బాస్,") or stripped.lower().startswith(
                "boss,"
            ):
                stripped = stripped.split(",", 1)[-1].strip()
            if stripped:
                cleaned.append(stripped)
        return cleaned

    def generate(self, prompt: str, case_id: str) -> Tuple[str, List[str], List[str]]:
        prompt_lower = prompt.lower()
        lang_detected = detect_lang(prompt)
        mix = self._should_mix(prompt)
        if mix:
            target_lang = "mixed"
        elif (
            lang_detected == "te"
            or prompt.strip().startswith("బాస్")
            or "telugu" in prompt_lower
        ):
            target_lang = "te"
        else:
            target_lang = "en"
        prefix = "బాస్," if target_lang in {"te", "mixed"} else "Boss,"
        clarify = self._needs_clarifying(prompt)
        if mix:
            clarify = False
        snippets, ltms = self._retrieve_templates(prompt)
        used_snippet_ids = [snip.get("id") for snip in snippets if snip.get("id")]
        used_ltm_ids = [mem.get("id") for mem in ltms if mem.get("id")]

        summary = "Locking next moves."
        bullets: List[str] = []

        def add_from_snippet(
            keyword: str, fallback: List[str], fallback_te: Optional[List[str]] = None
        ) -> bool:
            if target_lang == "te" and fallback_te:
                bullets.extend(fallback_te)
                return True
            for snip in snippets:
                title = (snip.get("title") or "").lower()
                if keyword in title:
                    lines = self._snippet_lines(snip)[:4]
                    bullets.extend(lines)
                    return True
            if fallback:
                bullets.extend(fallback)
            return False

        if "logline" in prompt_lower:
            summary = (
                "Punching up the logline frame."
                if target_lang == "en"
                else "లాగ్‌లైన్‌ను బలపరుస్తున్నాను."
            )
            add_from_snippet(
                "logline",
                ["Hero = [drive]", "Force = [antagonist]", "Stakes = [what breaks]"],
                [
                    "హీరో = [చోదక శక్తి]",
                    "ప్రతిబంధకుడు = [వ్యతిరేక శక్తి]",
                    "పణంగానేది = [విఫలమైతే నష్టము]",
                ],
            )
        elif "pitch" in prompt_lower:
            summary = (
                "Stacking investor pitch bullets."
                if target_lang == "en"
                else "పిచ్ పాయింట్లు సెట్ చేస్తున్నాను."
            )
            add_from_snippet(
                "pitch",
                ["Logline", "Why now", "Visual promise", "Ask"],
                ["కథ హుక్", "ఇప్పుడే ఎందుకు", "దృశ్య హామీ", "తదుపరి అడుగు"],
            )
        elif "character" in prompt_lower:
            summary = (
                "Character sheet, crisp."
                if target_lang == "en"
                else "పాత్ర వివరాలు సంక్షిప్తంగా."
            )
            add_from_snippet(
                "character",
                ["Goal", "Wound", "Contradiction", "Change"],
                ["లక్ష్యం", "గాయం", "విరుద్ధ లక్షణం", "మార్పు"],
            )
        elif "scene" in prompt_lower and "option" in prompt_lower:
            summary = (
                "Three midpoint punches ready."
                if target_lang != "te"
                else "మధ్యదశలో మూడు ఎంపికలు."
            )
            if target_lang == "te":
                bullets.extend(
                    [
                        "ఎంపిక A: గురువు స్టేజీపై నిజం లాగుతాడు.",
                        "ఎంపిక B: గురువు తన త్యాగాన్ని బహిర్గతం చేస్తాడు.",
                        "ఎంపిక C: గురువు నడిచి వెళ్తాడు; హీరో తానే నిర్ణయము చేసుకుంటాడు.",
                    ]
                )
            else:
                bullets.extend(
                    [
                        "Option A: Mentor forces truth in empty stage.",
                        "Option B: Mentor reveals sacrifice before crowd.",
                        "Option C: Mentor walks away, leaving hero to choose.",
                    ]
                )
        elif "scene" in prompt_lower and "checklist" in prompt_lower:
            summary = "Scene gate reminder." if target_lang == "en" else "సీన్ చెక్‌లిస్ట్."
            add_from_snippet(
                "scene",
                ["Purpose", "Conflict", "Reveal", "Exit hook"],
                ["ఉద్దేశ్యం", "ఘర్షణ", "వెలికితీత", "తరువాతి హుక్"],
            )
        elif "outline" in prompt_lower or "beat" in prompt_lower:
            summary = (
                "Outline ladder locked." if target_lang == "en" else "అవుట్‌లైన్ సిద్ధం."
            )
            add_from_snippet(
                "ladder",
                ["Act 1 hook", "Act 2 escalation", "Act 3 payoff"],
                ["అభినయం 1 హుక్", "అభినయం 2 ఎస్కలేషన్", "అభినయం 3 పరిష్కారం"],
            )
        elif "thriller" in prompt_lower:
            summary = (
                "Thriller rails in place." if target_lang == "en" else "థ్రిల్లర్ రైలు సిద్ధం."
            )
            add_from_snippet(
                "thriller",
                [
                    "Hero carries moral grey",
                    "Twist seeded early",
                    "Justice lands with sting",
                ],
                ["హీరోలో నైతిక గ్రీ", "ట్విస్ట్ ముందుగానే విత్తాలి", "అంత్య عدالتీ గట్టి దెబ్బగా"],
            )
        elif "rom-com" in prompt_lower:
            summary = (
                "Rom-com beat guardrails." if target_lang == "en" else "రామ్‌కామ్ బీట్ చట్రం."
            )
            add_from_snippet(
                "rom-com",
                ["Meet cute", "Family friction", "Public confession"],
                ["మధురమైన పరిచయం", "కుటుంబ ఘర్షణ", "ప్రజల ముందున్న ఒప్పుకోలు"],
            )
        elif "family" in prompt_lower:
            summary = "Family drama rhythm." if target_lang == "en" else "ఫ్యామిలీ డ్రామా రిథమ్."
            add_from_snippet(
                "family",
                ["Open on ritual", "Festival midpoint", "Blessing tag"],
                ["వేడుకతో ప్రారంభం", "పండుగ మధ్యదశ", "అశీర్వాదంతో ముగింపు"],
            )
        elif "drifts" in prompt_lower:
            summary = (
                "Reminder: keep screenplay tight."
                if target_lang == "en"
                else "స్క్రీన్‌ప్లే కట్టుదిట్టంగా ఉంచు."
            )
            if ltms:
                bullets.append(ltms[0]["text"])
            else:
                bullets.append("Stay sharp: trim anything without conflict.")
        elif "helicopter" in prompt_lower:
            summary = (
                "Budget reality check." if target_lang == "en" else "బడ్జెట్ నిజం గుర్తు."
            )
            if target_lang == "te":
                bullets.extend(
                    [
                        "హెలికాప్టర్ షాట్స్‌కి ప్రీమియం బీమా + అనుమతులు కావాలి.",
                        "డ్రోన్ ప్యాకేజీ లేదా స్టాక్ ఎరియల్స్ ఆలోచించు.",
                        "స్పాన్సర్ ఖాయం చేసిన తర్వాతే ముందుకు వెళ్ళు.",
                    ]
                )
            else:
                bullets.extend(
                    [
                        "Heli shots need premium insurance & permits.",
                        "Suggest drone package or stock aerials.",
                        "Lock sponsor before committing.",
                    ]
                )
        elif "discovery" in prompt_lower:
            summary = (
                "Discovery knobs check." if target_lang == "en" else "డిస్కవరీ ప్రశ్నలు."
            )
            if target_lang == "te":
                bullets.extend(
                    [
                        "జానర్",
                        "టోన్",
                        "సర్టిఫికేట్",
                        "భాషలు",
                        "రన్‌టైమ్",
                        "బడ్జెట్ రేంజ్",
                        "కాలం / ప్రదేశం",
                        "హీరో ఆర్క్",
                    ]
                )
            else:
                bullets.extend(
                    [
                        "Genre",
                        "Tone",
                        "Audience rating",
                        "Languages",
                        "Runtime",
                        "Budget band",
                        "Time/place",
                        "Hero arc",
                    ]
                )
        elif "feedback" in prompt_lower:
            summary = (
                "Scene tighten directives."
                if target_lang == "en"
                else "సీన్ సవరించే సూచనలు."
            )
            if target_lang == "te":
                bullets.extend(
                    [
                        "బ్యాక్‌స్టోరీ పదును తగ్గించు; ఘర్షణలో చూపు.",
                        "ప్రతి డైలాగ్‌లో లక్ష్యం vs అడ్డంకి.",
                        "కొత్త ప్రశ్న లేదా పవర్ షిఫ్ట్‌తో ముగించు.",
                    ]
                )
            else:
                bullets.extend(
                    [
                        "Drop backstory dump; feed through conflict.",
                        "Push goal vs obstacle in each exchange.",
                        "End with fresh question or power shift.",
                    ]
                )
        else:
            bullets.extend(["We stay decisive."])

        if mix and bullets:
            bullets = [self._mix_line(line) for line in bullets]

        response_lines = [f"{prefix} {summary}"]
        for line in bullets[:4]:
            response_lines.append(f"- {line}")
        if clarify:
            question = self._clarifying_question(prompt_lower, target_lang)
            label = "స్పష్టం" if target_lang == "te" else "Clarify"
            response_lines.append(f"- {label}: {question}?")

        reply = "\n".join(response_lines[:6])
        output_lang = detect_lang(reply)
        self.store.log_interaction(
            session_id=f"eval_{case_id}",
            user_msg=prompt,
            model_reply=reply,
            used_ltm_ids=used_ltm_ids,
            used_snippet_ids=used_snippet_ids,
            lang=output_lang,
        )
        return reply, used_snippet_ids, used_ltm_ids

    def _mix_line(self, line: str) -> str:
        return (
            line.replace("Hero", "హీరో").replace("Budget", "బడ్జెట్").replace("Act", "అభినయం")
        )

    def _clarifying_question(self, prompt_lower: str, target_lang: str) -> str:
        if target_lang == "te":
            if "budget" not in prompt_lower:
                return "ఏ బడ్జెట్ రేంజ్"
            if "runtime" not in prompt_lower:
                return "రన్‌టైమ్ పక్కా"
            return "ఇంకే నాబ్"
        else:
            if "budget" not in prompt_lower:
                return "which budget band"
            if "runtime" not in prompt_lower:
                return "runtime lock"
            return "key knob"


def evaluate_case(reply: str, expect: Dict) -> Dict[str, bool]:
    metrics: Dict[str, bool] = {}
    expected_prefix = expect.get("must_start")
    metrics["greeter_ok"] = (
        reply.startswith(expected_prefix) if expected_prefix else True
    )
    expected_lang = expect.get("lang")
    actual_lang = detect_lang(reply)
    if expected_lang == "mixed":
        metrics["lang_ok"] = actual_lang == "mixed"
    else:
        metrics["lang_ok"] = actual_lang == expected_lang
    ask_expected = expect.get("ask_clarifying", False)
    question_count = reply.count("?")
    metrics["clarifying_ok"] = (
        (question_count == 1) if ask_expected else (question_count == 0)
    )
    max_lines = expect.get("max_lines", 6)
    line_count = sum(1 for line in reply.splitlines() if line.strip())
    metrics["len_ok"] = line_count <= max_lines
    metrics["pass"] = all(metrics.values())
    return metrics


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Iter2 smoke eval")
    parser.add_argument("--cases", default="eval/iter2_smoke.jsonl")
    args = parser.parse_args()

    cases_path = Path(args.cases)
    cases: List[Dict] = []
    with cases_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                cases.append(json.loads(line))

    store = MemoryStore()
    assistant = ScriptAssistant(store)
    run_id = uuid.uuid4().hex
    suite_id = "iter2_smoke"
    created_at = datetime.now(timezone.utc).replace(microsecond=0).isoformat()

    results = []
    passes = 0
    for case in cases:
        reply, snippet_ids, ltm_ids = assistant.generate(case["prompt"], case["id"])
        metrics = evaluate_case(reply, case["expect"])
        if metrics["pass"]:
            passes += 1
        results.append(
            {
                "run_id": run_id,
                "suite_id": suite_id,
                "case_id": case["id"],
                "prompt": case["prompt"],
                "response": reply,
                "metrics": metrics,
                "used_snippets": snippet_ids,
                "used_ltm": ltm_ids,
                "created_at": created_at,
            }
        )

    run_path = Path(store.paths["eval_runs"])
    runs = []
    if run_path.exists():
        runs = [
            json.loads(line)
            for line in run_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    runs.append(
        {
            "run_id": run_id,
            "suite_id": suite_id,
            "model": "heuristic_friday",
            "total_cases": len(cases),
            "passed": passes,
            "created_at": created_at,
        }
    )
    with run_path.open("w", encoding="utf-8") as fh:
        for row in runs:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    results_path = Path(store.paths["eval_results"])
    existing_results = []
    if results_path.exists():
        existing_results = [
            json.loads(line)
            for line in results_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    existing_results.extend(results)
    with results_path.open("w", encoding="utf-8") as fh:
        for row in existing_results:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"Run {run_id}: {passes}/{len(cases)} cases passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
