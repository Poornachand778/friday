#!/usr/bin/env python3
"""
Build Clean Iteration 2 Dataset - Conversational Q&A Only

Strategy:
1. Read ALL WhatsApp data (10,577 examples) - analyze quality
2. Read tool examples (33 examples) - validate natural style
3. Read decision scenarios (20 examples) - convert to Q&A
4. Filter for conversation quality (no transliteration, natural code-switching)
5. Build clean training dataset (650-850 examples)

NO screenplay tasks. NO dialogue snippets. ONLY conversational Q&A.

Usage:
    python scripts/build_clean_iteration2_dataset.py --dry-run
    python scripts/build_clean_iteration2_dataset.py --whatsapp-target 700
"""

import json
import argparse
import hashlib
import random
from pathlib import Path
from datetime import datetime, timezone
from collections import Counter, defaultdict
from typing import List, Dict, Any

PROJECT_ROOT = Path(__file__).parent.parent

# Input files
WHATSAPP_TRAIN = PROJECT_ROOT / "data/sft/iter2/whatsapp_train.jsonl"
TOOL_EXAMPLES = PROJECT_ROOT / "data/instructions/iteration2_tool_examples.jsonl"
DECISION_SCENARIOS = PROJECT_ROOT / "data/persona/decision_scenarios_seed.jsonl"

# Output
OUTPUT_FILE = PROJECT_ROOT / "data/instructions/iteration2_clean_train.jsonl"
REPORT_FILE = PROJECT_ROOT / "data/instructions/iteration2_clean_report.json"

# Friday system prompt for ChatML conversion
FRIDAY_SYSTEM_PROMPT = """You are Friday, Poorna's personal AI assistant. You naturally blend Telugu and English in conversation (code-switching), just like Poorna does. You're knowledgeable about Telugu cinema, screenwriting, and film production.

Key traits:
- Address Poorna as "Boss" (or "బాస్" in Telugu contexts)
- Keep responses concise (under 6 lines unless detailed content is needed)
- Be decisive and practical, with a touch of wit
- No flattery or formal phrases like "kindly" or "dear user"
- Match the user's language choice (respond in Telugu if asked in Telugu)"""


def load_jsonl(filepath: Path) -> List[Dict[str, Any]]:
    """Load JSONL file."""
    if not filepath.exists():
        print(f"  Warning: {filepath} not found")
        return []
    data = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


def is_high_quality_whatsapp(example: Dict[str, Any]) -> bool:
    """
    Check if WhatsApp example is high quality.

    Quality criteria:
    - No low-value responses ("Boss, ha/ok/hmm")
    - No placeholders ({PHONE}, {EMAIL}, "omitted")
    - Reasonable length (not too short)
    - Not repeated character spam
    """
    output = example.get("output", "").lower()

    # Filter low-value outputs
    LOW_VALUE = {
        "boss, ha",
        "boss, ok",
        "boss, hmm",
        "boss, yes",
        "boss, no",
        "boss, yep",
        "boss, nope",
        "boss, haha",
        "boss, ya",
        "boss, okay",
        "boss, ha.",
        "boss, ok.",
        "boss, hmm.",
        "boss, exactly",
        "boss, correct",
        "boss, అవును",
        "boss, హా",
        "boss, హా.",
        "boss, లేదు",
        "boss, ఓకే",
        "boss, సరే",
        "boss, emo",
        "boss, lite le",
        "boss, good",
        "boss, nice",
        "boss, wow",
        "boss, oh",
        "boss, oho",
        "boss, cool",
        "boss, boring",
        "boss, hi",
    }

    if output.strip() in LOW_VALUE:
        return False

    # Filter placeholders
    FILTER_PATTERNS = [
        "{PHONE}",
        "{EMAIL}",
        "omitted",
        "deleted",
        "This message was deleted",
    ]
    for pattern in FILTER_PATTERNS:
        if pattern.lower() in output.lower():
            return False

    # Filter very short responses (unless priority tagged)
    response_text = output[6:] if output.startswith("boss, ") else output
    tags = example.get("tags", [])
    is_priority = "script" in tags or "clarifying" in tags

    if len(response_text) <= 8 and not is_priority:
        return False

    # Filter repeated character spam
    if len(set(response_text.replace(" ", ""))) < 4 and len(response_text) > 20:
        return False

    return True


def convert_whatsapp_to_chatml(example: Dict[str, Any]) -> Dict[str, Any]:
    """Convert WhatsApp format to ChatML."""
    user_input = example.get("input", "")
    assistant_output = example.get("output", "")

    # Clean up input - remove {NAME_A} placeholder
    user_input = user_input.replace("{NAME_A}: ", "").replace("{NAME_A}:", "")
    user_input = user_input.replace("\n", " ").strip()

    messages = [
        {"role": "system", "content": FRIDAY_SYSTEM_PROMPT},
        {"role": "user", "content": user_input},
        {"role": "assistant", "content": assistant_output},
    ]

    return {
        "messages": messages,
        "metadata": {
            "source": "whatsapp",
            "original_id": example.get("id", ""),
            "lang": example.get("lang", "en"),
            "tags": example.get("tags", []),
            "chat": example.get("meta", {}).get("chat", "unknown"),
            "domain": "persona",
        },
    }


def convert_decision_scenario_to_qa(scenario: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert decision scenario to Q&A format.

    Example:
    Scenario: "Day-4 location fee doubles morning-of."
    Choice: "Scene-first policy: keep plan if essential"

    Becomes:
    User: "Boss, location fee doubled on day 4. Scene essential. What do we do?"
    Assistant: "Scene integrity first. If shot is mission-critical, pay and shoot..."
    """
    scenario_id = scenario.get("scenario_id", "")
    domain = scenario.get("domain", "general")
    situation = scenario.get("situation", "")
    choice = scenario.get("choice", "")
    rationale = scenario.get("rationale", "")

    # Skip if missing critical fields
    if not situation or not choice:
        return None

    # Create conversational question
    # Add "Boss," prefix and make it sound natural
    if domain == "film" or domain == "production":
        user_text = f"Boss, {situation.lower()} What should we do?"
    elif domain == "kitchen":
        user_text = f"Boss, {situation.lower()} How do I handle this?"
    else:
        user_text = f"Boss, {situation.lower()} Your take?"

    # Create Friday's response (choice + rationale)
    if rationale:
        assistant_text = f"{choice}. {rationale}"
    else:
        assistant_text = choice

    messages = [
        {"role": "system", "content": FRIDAY_SYSTEM_PROMPT},
        {"role": "user", "content": user_text},
        {"role": "assistant", "content": assistant_text},
    ]

    return {
        "messages": messages,
        "metadata": {
            "source": "decision_scenarios",
            "scenario_id": scenario_id,
            "domain": domain,
            "tags": ["decision-making", domain],
        },
    }


def sample_whatsapp(
    data: List[Dict[str, Any]], target: int, seed: int = 42
) -> List[Dict[str, Any]]:
    """
    Sample high-quality WhatsApp examples.

    Strategy:
    1. Filter for quality
    2. Prioritize script-tagged and clarifying-tagged
    3. Sample from top 50% by response length
    4. Maintain language balance
    """
    random.seed(seed)

    # Filter for quality
    quality_data = [ex for ex in data if is_high_quality_whatsapp(ex)]
    print(f"  Quality filtered: {len(quality_data)}/{len(data)} examples pass")

    # Separate priority examples
    priority = [
        ex
        for ex in quality_data
        if "script" in ex.get("tags", []) or "clarifying" in ex.get("tags", [])
    ]
    regular = [
        ex
        for ex in quality_data
        if "script" not in ex.get("tags", []) and "clarifying" not in ex.get("tags", [])
    ]

    print(f"  Priority (script/clarifying): {len(priority)} examples")
    print(f"  Regular: {len(regular)} examples")

    # Take all priority examples
    sampled = priority.copy()

    # Sample from regular to reach target
    remaining = target - len(sampled)
    if remaining > 0 and regular:
        # Sort by output length (longer = more informative)
        regular.sort(key=lambda x: len(x.get("output", "")), reverse=True)

        # Take top 50% by length, then shuffle
        top_half = regular[: len(regular) // 2]
        random.shuffle(top_half)
        sampled.extend(top_half[:remaining])

    return sampled[:target]


def deduplicate_by_content(data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Remove duplicates based on content hash."""
    seen = set()
    unique = []

    for item in data:
        # Create content hash
        if "messages" in item:
            content = json.dumps(item["messages"], sort_keys=True)
        else:
            content = json.dumps(item, sort_keys=True)

        content_hash = hashlib.sha256(content.encode()).hexdigest()[:16]

        if content_hash not in seen:
            seen.add(content_hash)
            unique.append(item)

    return unique


def analyze_dataset_gaps(data: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Analyze what types of conversations are missing.

    Returns recommendations for what data to create.
    """
    # Count by domain/topic
    topic_counts = Counter()
    lang_counts = Counter()

    for ex in data:
        metadata = ex.get("metadata", {})
        domain = metadata.get("domain", "unknown")
        lang = metadata.get("lang", "unknown")
        tags = metadata.get("tags", [])

        topic_counts[domain] += 1
        lang_counts[lang] += 1

        for tag in tags:
            topic_counts[f"tag:{tag}"] += 1

    return {
        "topic_distribution": dict(topic_counts.most_common(20)),
        "language_distribution": dict(lang_counts),
        "recommendations": generate_recommendations(topic_counts, lang_counts),
    }


def generate_recommendations(topic_counts: Counter, lang_counts: Counter) -> List[str]:
    """Generate recommendations for what data to create."""
    recommendations = []

    # Check for missing topics
    important_topics = [
        "film",
        "production",
        "persona",
        "decision-making",
        "technical",
        "creative",
        "planning",
    ]

    for topic in important_topics:
        count = topic_counts.get(topic, 0) + topic_counts.get(f"tag:{topic}", 0)
        if count < 50:
            recommendations.append(f"Create more {topic} examples (currently: {count})")

    # Check language balance
    total = sum(lang_counts.values())
    te_pct = lang_counts.get("te", 0) / total * 100 if total > 0 else 0

    if te_pct < 30:
        recommendations.append(f"Add more Telugu examples (currently: {te_pct:.1f}%)")

    return recommendations


def main():
    parser = argparse.ArgumentParser(description="Build clean iteration2 dataset")
    parser.add_argument(
        "--dry-run", action="store_true", help="Preview without writing"
    )
    parser.add_argument(
        "--whatsapp-target", type=int, default=700, help="Target WhatsApp samples"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    print("=" * 60)
    print("Friday AI - Building Clean Iteration 2 Dataset")
    print("=" * 60)
    print("\nStrategy: Conversational Q&A ONLY (no screenplay tasks)")

    all_examples = []
    stats = {
        "sources": {},
        "total_before_dedup": 0,
        "total_after_dedup": 0,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    # 1. Load tool examples (already clean)
    print(f"\n[1/3] Loading tool examples...")
    tool_data = load_jsonl(TOOL_EXAMPLES)
    print(f"  Loaded: {len(tool_data)} examples")
    all_examples.extend(tool_data)
    stats["sources"]["tool_examples"] = len(tool_data)

    # 2. Sample and convert WhatsApp data
    print(f"\n[2/3] Processing WhatsApp data (target: {args.whatsapp_target})...")
    whatsapp_data = load_jsonl(WHATSAPP_TRAIN)
    print(f"  Total WhatsApp examples: {len(whatsapp_data)}")

    sampled_whatsapp = sample_whatsapp(whatsapp_data, args.whatsapp_target, args.seed)
    print(f"  Sampled: {len(sampled_whatsapp)} examples")

    # Convert to ChatML
    converted_whatsapp = [convert_whatsapp_to_chatml(ex) for ex in sampled_whatsapp]
    all_examples.extend(converted_whatsapp)
    stats["sources"]["whatsapp"] = len(converted_whatsapp)

    # Analyze WhatsApp selection
    whatsapp_langs = Counter(ex.get("lang", "unknown") for ex in sampled_whatsapp)
    whatsapp_chats = Counter(
        ex.get("meta", {}).get("chat", "unknown") for ex in sampled_whatsapp
    )
    stats["whatsapp_breakdown"] = {
        "by_language": dict(whatsapp_langs),
        "by_chat": dict(whatsapp_chats),
    }

    # 3. Convert decision scenarios to Q&A
    print(f"\n[3/3] Converting decision scenarios to Q&A...")
    decision_data = load_jsonl(DECISION_SCENARIOS)
    print(f"  Total decision scenarios: {len(decision_data)}")

    converted_decisions = []
    for scenario in decision_data:
        converted = convert_decision_scenario_to_qa(scenario)
        if converted:
            converted_decisions.append(converted)

    print(f"  Converted: {len(converted_decisions)} scenarios")
    all_examples.extend(converted_decisions)
    stats["sources"]["decision_scenarios"] = len(converted_decisions)

    # Summary before dedup
    stats["total_before_dedup"] = len(all_examples)
    print(f"\n--- Pre-deduplication ---")
    print(f"Total examples: {len(all_examples)}")
    for source, count in stats["sources"].items():
        print(f"  {source}: {count}")

    # Deduplicate
    print(f"\n--- Deduplication ---")
    all_examples = deduplicate_by_content(all_examples)
    stats["total_after_dedup"] = len(all_examples)
    print(f"After dedup: {len(all_examples)} examples")

    # Final statistics
    print(f"\n--- Final Clean Dataset ---")
    print(f"Total: {len(all_examples)} examples")
    print(f"WhatsApp language distribution: {dict(whatsapp_langs)}")

    # Analyze gaps and generate recommendations
    print(f"\n--- Data Gap Analysis ---")
    gap_analysis = analyze_dataset_gaps(all_examples)
    print(f"Topic distribution:")
    for topic, count in list(gap_analysis["topic_distribution"].items())[:10]:
        print(f"  {topic}: {count}")

    print(f"\nRecommendations for data creation:")
    for i, rec in enumerate(gap_analysis["recommendations"], 1):
        print(f"  {i}. {rec}")

    stats["gap_analysis"] = gap_analysis

    # Show sample examples
    print(f"\n--- Sample Examples ---")
    for i, ex in enumerate(random.sample(all_examples, min(5, len(all_examples))), 1):
        if "messages" in ex:
            user_msg = (
                ex["messages"][1]["content"][:80] if len(ex["messages"]) > 1 else "N/A"
            )
            asst_msg = (
                ex["messages"][2]["content"][:80] if len(ex["messages"]) > 2 else "N/A"
            )
            source = ex.get("metadata", {}).get("source", "unknown")
            print(f"\n{i}. Source: {source}")
            print(f"   User: {user_msg}...")
            print(f"   Assistant: {asst_msg}...")

    if args.dry_run:
        print(f"\n[DRY RUN] No files written.")
    else:
        # Write output
        print(f"\n--- Writing Output ---")
        OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

        with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
            for ex in all_examples:
                f.write(json.dumps(ex, ensure_ascii=False) + "\n")
        print(f"Wrote: {OUTPUT_FILE}")

        # Write report
        with open(REPORT_FILE, "w", encoding="utf-8") as f:
            json.dump(stats, f, indent=2, ensure_ascii=False)
        print(f"Wrote: {REPORT_FILE}")

    print("\n" + "=" * 60)
    print(f"Clean dataset build complete! Total: {len(all_examples)} examples")
    print("=" * 60)

    return all_examples


if __name__ == "__main__":
    main()
