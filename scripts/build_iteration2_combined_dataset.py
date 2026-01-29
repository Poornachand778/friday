#!/usr/bin/env python3
"""
Build Combined Dataset for Friday AI Iteration 2

Merges all curated data sources into a single training file:
- Interview exchanges (120 examples - ChatML format)
- Curated WhatsApp examples (350 examples - ChatML format with transliteration)
- Contrastive pairs (25 pairs - DPO format)
- Tool examples (33 examples - ChatML format)

Total: ~528 examples

Usage:
    python scripts/build_iteration2_combined_dataset.py
    python scripts/build_iteration2_combined_dataset.py --dry-run
"""

import json
import argparse
import hashlib
import random
from pathlib import Path
from datetime import datetime, timezone
from collections import Counter
from typing import List, Dict, Any

PROJECT_ROOT = Path(__file__).parent.parent

# Input files (all curated)
SOURCES = {
    "interview": PROJECT_ROOT / "data/instructions/interview_iter2.jsonl",
    "whatsapp": PROJECT_ROOT / "data/instructions/whatsapp_curated_iter2.jsonl",
    "contrastive": PROJECT_ROOT / "data/instructions/contrastive_pairs_iter2.jsonl",
    "tools": PROJECT_ROOT / "data/instructions/iteration2_tool_examples.jsonl",
}

# Output
OUTPUT_FILE = PROJECT_ROOT / "data/instructions/iteration2_combined_train.jsonl"
STATS_FILE = PROJECT_ROOT / "data/instructions/iteration2_combined_train.stats.json"

# System prompt for contrastive pairs conversion
FRIDAY_SYSTEM_PROMPT = """You are Friday, Poorna's personal AI assistant. Communicate using natural Telugu-English code-switching.

Traits:
- Address Poorna as "Boss"
- Be direct and concise (no flattery)
- Use Telugu for emotional/cultural topics
- Use English for technical concepts
- Never use corporate AI phrases
- Keep responses brief (1-3 sentences typical)"""


def convert_contrastive_to_chatml(example: Dict[str, Any]) -> Dict[str, Any]:
    """Convert DPO contrastive pair format to ChatML format for SFT training."""
    return {
        "messages": [
            {"role": "system", "content": FRIDAY_SYSTEM_PROMPT},
            {"role": "user", "content": example["prompt"]},
            {"role": "assistant", "content": example["chosen"]},
        ],
        "metadata": {
            "source": "contrastive",
            "category": example.get("category", ""),
            "rejected_response": example.get("rejected", ""),
            "format": "converted_from_dpo",
        },
    }


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


def content_hash(example: Dict[str, Any]) -> str:
    """Generate hash for deduplication."""
    # For ChatML format
    if "messages" in example:
        content = json.dumps(example["messages"], sort_keys=True, ensure_ascii=False)
    # For contrastive/DPO format
    elif "prompt" in example and "chosen" in example:
        content = (
            f"{example['prompt']}|{example['chosen']}|{example.get('rejected', '')}"
        )
    else:
        content = json.dumps(example, sort_keys=True, ensure_ascii=False)

    return hashlib.sha256(content.encode()).hexdigest()[:16]


def get_domain(example: Dict[str, Any], source: str) -> str:
    """Determine domain category."""
    if source == "tools":
        return "tools"
    if source == "contrastive":
        return "contrastive"

    # Check metadata
    metadata = example.get("metadata", {})
    tags = metadata.get("tags", [])
    topic = metadata.get("topic", "")

    # Film-related
    film_keywords = [
        "film",
        "screenplay",
        "scene",
        "dialogue",
        "story",
        "character",
        "production",
        "movie",
        "cinema",
    ]
    if any(kw in str(tags).lower() or kw in topic.lower() for kw in film_keywords):
        return "film"

    # Persona-related
    persona_keywords = [
        "persona",
        "belief",
        "value",
        "emotion",
        "relationship",
        "dream",
        "fear",
        "tech",
        "decision",
    ]
    if any(kw in str(tags).lower() or kw in topic.lower() for kw in persona_keywords):
        return "persona"

    # WhatsApp is general conversation
    if source == "whatsapp":
        return "conversation"

    return "general"


def get_language(example: Dict[str, Any]) -> str:
    """Determine language category."""
    metadata = example.get("metadata", {})

    # Direct lang field
    lang = metadata.get("lang", metadata.get("language", ""))
    if lang in ["te", "telugu"]:
        return "te"
    if lang in ["te-en", "en-te"]:
        return "te-en"
    if lang in ["en", "english"]:
        return "en"

    # Check had_telugu flag
    if metadata.get("had_telugu", False):
        return "te-en"

    return "en"


def normalize_example(example: Dict[str, Any], source: str) -> Dict[str, Any]:
    """Ensure consistent format and add source metadata."""
    # Add source if not present
    if "metadata" not in example:
        example["metadata"] = {}

    example["metadata"]["dataset_source"] = source
    example["metadata"]["iter2_added_at"] = datetime.now(timezone.utc).isoformat()

    return example


def main():
    parser = argparse.ArgumentParser(description="Build iteration2 combined dataset")
    parser.add_argument(
        "--dry-run", action="store_true", help="Preview without writing"
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for shuffling"
    )
    args = parser.parse_args()

    print("=" * 60)
    print("Friday AI - Building Iteration 2 Combined Dataset")
    print("=" * 60)

    all_examples = []
    seen_hashes = set()
    stats = {
        "sources": {},
        "domains": Counter(),
        "languages": Counter(),
        "duplicates_removed": 0,
        "total_examples": 0,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    # Load and process each source
    for source_name, source_path in SOURCES.items():
        print(f"\n[{source_name.upper()}] Loading from {source_path.name}...")
        examples = load_jsonl(source_path)

        source_stats = {"loaded": len(examples), "added": 0, "duplicates": 0}

        for ex in examples:
            # Convert contrastive pairs from DPO to ChatML format
            if source_name == "contrastive" and "prompt" in ex and "chosen" in ex:
                ex = convert_contrastive_to_chatml(ex)

            # Check for duplicates
            h = content_hash(ex)
            if h in seen_hashes:
                source_stats["duplicates"] += 1
                stats["duplicates_removed"] += 1
                continue

            seen_hashes.add(h)

            # Normalize and add metadata
            ex = normalize_example(ex, source_name)

            # Track domain and language
            domain = get_domain(ex, source_name)
            lang = get_language(ex)
            stats["domains"][domain] += 1
            stats["languages"][lang] += 1

            all_examples.append(ex)
            source_stats["added"] += 1

        stats["sources"][source_name] = source_stats
        print(
            f"  Loaded: {source_stats['loaded']}, Added: {source_stats['added']}, Duplicates: {source_stats['duplicates']}"
        )

    stats["total_examples"] = len(all_examples)

    # Shuffle for training (deterministic)
    random.seed(args.seed)
    random.shuffle(all_examples)

    # Print summary
    print("\n" + "=" * 60)
    print("COMBINED DATASET SUMMARY")
    print("=" * 60)
    print(f"\nTotal examples: {stats['total_examples']}")
    print(f"Duplicates removed: {stats['duplicates_removed']}")

    print("\nBy source:")
    for source, s in stats["sources"].items():
        print(f"  {source}: {s['added']} examples")

    print("\nBy domain:")
    for domain, count in sorted(stats["domains"].items(), key=lambda x: -x[1]):
        pct = count / stats["total_examples"] * 100
        print(f"  {domain}: {count} ({pct:.1f}%)")

    print("\nBy language:")
    for lang, count in sorted(stats["languages"].items(), key=lambda x: -x[1]):
        pct = count / stats["total_examples"] * 100
        print(f"  {lang}: {count} ({pct:.1f}%)")

    # Show sample examples
    print("\n--- Sample Examples ---")
    for i, ex in enumerate(all_examples[:3]):
        if "messages" in ex:
            user_msg = (
                ex["messages"][1]["content"][:50] if len(ex["messages"]) > 1 else "N/A"
            )
            asst_msg = (
                ex["messages"][2]["content"][:50] if len(ex["messages"]) > 2 else "N/A"
            )
            source = ex.get("metadata", {}).get("dataset_source", "unknown")
            print(f"\n{i+1}. Source: {source}")
            print(f"   User: {user_msg}...")
            print(f"   Asst: {asst_msg}...")
        elif "prompt" in ex and "chosen" in ex:
            print(f"\n{i+1}. Source: contrastive")
            print(f"   Prompt: {ex['prompt'][:50]}...")
            print(f"   Chosen: {ex['chosen'][:50]}...")

    # Convert counters to dict for JSON
    stats["domains"] = dict(stats["domains"])
    stats["languages"] = dict(stats["languages"])

    if args.dry_run:
        print(f"\n[DRY RUN] No files written.")
    else:
        # Write combined dataset
        print(f"\n--- Writing Output ---")
        OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

        with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
            for ex in all_examples:
                f.write(json.dumps(ex, ensure_ascii=False) + "\n")
        print(f"Wrote: {OUTPUT_FILE}")
        print(f"  Size: {OUTPUT_FILE.stat().st_size / 1024:.1f} KB")

        # Write stats
        with open(STATS_FILE, "w", encoding="utf-8") as f:
            json.dump(stats, f, indent=2, ensure_ascii=False)
        print(f"Wrote: {STATS_FILE}")

    print("\n" + "=" * 60)
    print(f"Dataset build complete! Total: {stats['total_examples']} examples")
    print("=" * 60)

    return stats


if __name__ == "__main__":
    main()
