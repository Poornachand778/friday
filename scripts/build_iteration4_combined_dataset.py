#!/usr/bin/env python3
"""
Build Iteration 4 Combined Training Dataset
============================================

Combines all training data sources:
1. Phase 1 Q&A pairs (117) - from JSON
2. Tool examples (50) - from JSONL
3. Interview exchanges (120) - from JSONL
4. WhatsApp curated (350) - from JSONL
5. Contrastive pairs (25) - from JSONL

Total target: ~660 high-quality examples
"""

import json
import hashlib
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any

# Paths
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
INSTRUCTIONS_DIR = DATA_DIR / "instructions"
TRAINING_DIR = DATA_DIR / "training_collection"

# Input files
PHASE1_JSON = TRAINING_DIR / "phase1_base_questions.json"
TOOL_EXAMPLES = INSTRUCTIONS_DIR / "iteration2_tool_examples.jsonl"
INTERVIEW_DATA = INSTRUCTIONS_DIR / "interview_iter2.jsonl"
WHATSAPP_DATA = INSTRUCTIONS_DIR / "whatsapp_curated_iter2.jsonl"
CONTRASTIVE_DATA = INSTRUCTIONS_DIR / "contrastive_pairs_iter2.jsonl"

# Output
OUTPUT_FILE = INSTRUCTIONS_DIR / "iteration4_combined_train.jsonl"
STATS_FILE = INSTRUCTIONS_DIR / "iteration4_combined_stats.json"

# System prompt
SYSTEM_PROMPT = """You are Friday, Poorna's AI assistant. You blend Telugu and English naturally, addressing him as 'Boss'. Be concise, helpful, and direct. No flattery or excessive formality."""


def content_hash(messages: List[Dict]) -> str:
    """Generate hash for deduplication"""
    content = "".join(m.get("content", "") for m in messages)
    return hashlib.sha256(content.encode()).hexdigest()[:16]


def convert_phase1_to_chatml(phase1_path: Path) -> List[Dict]:
    """Convert Phase 1 JSON to ChatML format"""
    with open(phase1_path) as f:
        data = json.load(f)

    examples = []
    for category, cat_data in data.get("categories", {}).items():
        for pair in cat_data.get("pairs", []):
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": pair["user"]},
                {"role": "assistant", "content": pair["assistant"]},
            ]

            examples.append(
                {
                    "messages": messages,
                    "metadata": {
                        "source": "phase1_qa",
                        "category": category,
                        "id": pair.get("id", "unknown"),
                        "language": detect_language(pair["assistant"]),
                    },
                }
            )

    return examples


def detect_language(text: str) -> str:
    """Simple language detection based on Telugu characters"""
    telugu_chars = sum(1 for c in text if "\u0C00" <= c <= "\u0C7F")
    telugu_words = sum(
        1 for word in text.split() if any("\u0C00" <= c <= "\u0C7F" for c in word)
    )

    # Check for romanized Telugu markers
    romanized_markers = [
        "nenu",
        "naku",
        "Boss",
        "chestha",
        "kavali",
        "antey",
        "undhi",
        "cheppu",
        "enti",
        "kosam",
        "aythe",
        "inka",
        "ledu",
        "avutundi",
    ]
    has_romanized = any(marker.lower() in text.lower() for marker in romanized_markers)

    if telugu_chars > 5 or telugu_words > 2:
        return "te"
    elif has_romanized:
        return "te-en"  # Code-switched
    else:
        return "en"


def load_jsonl(path: Path) -> List[Dict]:
    """Load JSONL file"""
    examples = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                examples.append(json.loads(line))
    return examples


def merge_datasets() -> tuple:
    """Merge all data sources and return examples + stats"""
    all_examples = []
    seen_hashes = set()
    stats = {"sources": {}, "languages": {}, "duplicates_removed": 0}

    # 1. Phase 1 Q&A
    if PHASE1_JSON.exists():
        phase1 = convert_phase1_to_chatml(PHASE1_JSON)
        for ex in phase1:
            h = content_hash(ex["messages"])
            if h not in seen_hashes:
                seen_hashes.add(h)
                all_examples.append(ex)
                stats["languages"][ex["metadata"]["language"]] = (
                    stats["languages"].get(ex["metadata"]["language"], 0) + 1
                )
            else:
                stats["duplicates_removed"] += 1
        stats["sources"]["phase1_qa"] = len(phase1)
        print(f"Phase 1 Q&A: {len(phase1)} examples")

    # 2. Tool examples
    if TOOL_EXAMPLES.exists():
        tools = load_jsonl(TOOL_EXAMPLES)
        count = 0
        for ex in tools:
            h = content_hash(ex.get("messages", []))
            if h not in seen_hashes:
                seen_hashes.add(h)
                all_examples.append(ex)
                lang = ex.get("metadata", {}).get("language", "te-en")
                stats["languages"][lang] = stats["languages"].get(lang, 0) + 1
                count += 1
            else:
                stats["duplicates_removed"] += 1
        stats["sources"]["tool_examples"] = count
        print(f"Tool examples: {count} examples")

    # 3. Interview data
    if INTERVIEW_DATA.exists():
        interviews = load_jsonl(INTERVIEW_DATA)
        count = 0
        for ex in interviews:
            h = content_hash(ex.get("messages", []))
            if h not in seen_hashes:
                seen_hashes.add(h)
                all_examples.append(ex)
                lang = ex.get("metadata", {}).get("language", "te-en")
                stats["languages"][lang] = stats["languages"].get(lang, 0) + 1
                count += 1
            else:
                stats["duplicates_removed"] += 1
        stats["sources"]["interview"] = count
        print(f"Interview data: {count} examples")

    # 4. WhatsApp curated
    if WHATSAPP_DATA.exists():
        whatsapp = load_jsonl(WHATSAPP_DATA)
        count = 0
        for ex in whatsapp:
            h = content_hash(ex.get("messages", []))
            if h not in seen_hashes:
                seen_hashes.add(h)
                all_examples.append(ex)
                lang = ex.get("metadata", {}).get("language", "te-en")
                stats["languages"][lang] = stats["languages"].get(lang, 0) + 1
                count += 1
            else:
                stats["duplicates_removed"] += 1
        stats["sources"]["whatsapp"] = count
        print(f"WhatsApp curated: {count} examples")

    # 5. Contrastive pairs (convert to SFT format - use chosen response)
    if CONTRASTIVE_DATA.exists():
        contrastive = load_jsonl(CONTRASTIVE_DATA)
        count = 0
        for ex in contrastive:
            # Convert contrastive to SFT (use chosen response)
            if "prompt" in ex and "chosen" in ex:
                messages = [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": ex["prompt"]},
                    {"role": "assistant", "content": ex["chosen"]},
                ]
                h = content_hash(messages)
                if h not in seen_hashes:
                    seen_hashes.add(h)
                    all_examples.append(
                        {
                            "messages": messages,
                            "metadata": {
                                "source": "contrastive",
                                "language": detect_language(ex["chosen"]),
                            },
                        }
                    )
                    stats["languages"]["te-en"] = stats["languages"].get("te-en", 0) + 1
                    count += 1
                else:
                    stats["duplicates_removed"] += 1
        stats["sources"]["contrastive"] = count
        print(f"Contrastive pairs: {count} examples")

    stats["total"] = len(all_examples)
    return all_examples, stats


def analyze_tokens(examples: List[Dict]) -> Dict:
    """Analyze token distribution (rough estimate: 1 token ~ 4 chars)"""
    lengths = []
    for ex in examples:
        total_chars = sum(len(m.get("content", "")) for m in ex.get("messages", []))
        lengths.append(total_chars // 4)  # Rough token estimate

    if not lengths:
        return {}

    lengths.sort()
    return {
        "min_tokens": min(lengths),
        "max_tokens": max(lengths),
        "mean_tokens": sum(lengths) // len(lengths),
        "p50_tokens": lengths[len(lengths) // 2],
        "p95_tokens": lengths[int(len(lengths) * 0.95)],
        "p99_tokens": (
            lengths[int(len(lengths) * 0.99)] if len(lengths) > 100 else lengths[-1]
        ),
    }


def main():
    print("=" * 60)
    print("Building Iteration 4 Combined Training Dataset")
    print("=" * 60)

    # Merge all sources
    examples, stats = merge_datasets()

    # Analyze tokens
    token_stats = analyze_tokens(examples)
    stats["token_distribution"] = token_stats

    # Add metadata
    stats["created"] = datetime.now().isoformat()
    stats["output_file"] = str(OUTPUT_FILE)

    # Write output JSONL
    with open(OUTPUT_FILE, "w") as f:
        for ex in examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    # Write stats
    with open(STATS_FILE, "w") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Total examples: {stats['total']}")
    print(f"Duplicates removed: {stats['duplicates_removed']}")
    print(f"\nSources:")
    for source, count in stats["sources"].items():
        print(f"  - {source}: {count}")
    print(f"\nLanguages:")
    for lang, count in stats["languages"].items():
        print(f"  - {lang}: {count}")
    print(f"\nToken distribution:")
    for key, val in token_stats.items():
        print(f"  - {key}: {val}")
    print(f"\nOutput: {OUTPUT_FILE}")
    print(f"Stats: {STATS_FILE}")


if __name__ == "__main__":
    main()
