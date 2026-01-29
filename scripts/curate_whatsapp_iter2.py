#!/usr/bin/env python3
"""
WhatsApp Data Curation Script for Friday AI Iteration 2

Curates high-quality WhatsApp examples for training:
1. Loads raw WhatsApp data
2. Transliterates native Telugu to romanized (for TTS compatibility)
3. Filters by quality criteria
4. Samples target number of examples
5. Converts to ChatML format

Usage:
  python scripts/curate_whatsapp_iter2.py --target 350 --output data/instructions/whatsapp_curated_iter2.jsonl
"""

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path
from typing import Optional

# Add scripts to path for imports
sys.path.insert(0, str(Path(__file__).parent))
from transliterate_telugu import transliterate_text, has_telugu


# System prompt for Friday
SYSTEM_PROMPT = """You are Friday, Poorna's personal AI assistant. Communicate using natural Telugu-English code-switching.

Traits:
- Address Poorna as "Boss"
- Be direct and concise (no flattery)
- Use Telugu for emotional/cultural topics
- Use English for technical concepts
- Never use corporate AI phrases
- Keep responses brief (1-3 sentences typical)"""


def load_whatsapp_data(path: Path) -> list[dict]:
    """Load JSONL file."""
    examples = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                examples.append(json.loads(line))
    return examples


def quality_filter(example: dict, min_response_len: int = 15) -> tuple[bool, str]:
    """
    Filter example by quality criteria.

    Returns:
        (passes_filter, reason)
    """
    output = example.get("output", "")

    # Check minimum length
    if len(output) < min_response_len:
        return False, "too_short"

    # Check for omitted content
    omitted_patterns = [
        "audio omitted",
        "image omitted",
        "video omitted",
        "sticker omitted",
    ]
    for pattern in omitted_patterns:
        if pattern.lower() in output.lower():
            return False, "omitted_content"

    # Check for placeholder-only responses
    if output.strip() in ["Boss, {PHONE}", "Boss, {NAME_A}", "{PHONE}", "{NAME_A}"]:
        return False, "placeholder_only"

    # Check for very generic responses
    generic = ["Boss, Hi", "Boss, Ok", "Boss, Ha", "Boss, Hmm", "Boss, Yes", "Boss, No"]
    if output.strip() in generic and len(output) < 12:
        return False, "too_generic"

    return True, "passed"


def categorize_language(text: str) -> str:
    """Categorize text by language content."""
    has_te = has_telugu(text)

    # Check for English words (simple heuristic)
    english_words = len([w for w in text.split() if w.isascii() and w.isalpha()])
    total_words = len(text.split())

    if has_te:
        english_ratio = english_words / max(total_words, 1)
        if english_ratio > 0.7:
            return "english_dominant"
        elif english_ratio > 0.3:
            return "code_switched"
        else:
            return "telugu_dominant"
    else:
        return "english_only"


def convert_to_chatml(example: dict, transliterated_output: str) -> dict:
    """Convert WhatsApp example to ChatML format."""
    # Clean up input (remove {NAME_A} prefixes)
    user_input = example.get("input", "")
    # Remove name prefixes like "{NAME_A}: " at start of lines
    import re

    user_input = re.sub(r"\{NAME_[A-Z]\}:\s*", "", user_input)
    user_input = user_input.strip()

    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_input},
            {"role": "assistant", "content": transliterated_output},
        ],
        "metadata": {
            "source": "whatsapp",
            "original_id": example.get("id", ""),
            "lang": example.get("lang", ""),
            "chat": example.get("meta", {}).get("chat", ""),
            "had_telugu": has_telugu(example.get("output", "")),
            "tags": example.get("tags", []),
        },
    }


def curate_whatsapp(
    input_path: Path,
    output_path: Path,
    target_count: int = 350,
    min_response_len: int = 15,
    seed: int = 42,
) -> dict:
    """
    Main curation function.

    Args:
        input_path: Path to raw WhatsApp JSONL
        output_path: Path for curated output
        target_count: Number of examples to sample
        min_response_len: Minimum response length
        seed: Random seed for reproducibility

    Returns:
        Statistics dictionary
    """
    random.seed(seed)

    print(f"Loading data from {input_path}...")
    raw_examples = load_whatsapp_data(input_path)
    print(f"Loaded {len(raw_examples)} raw examples")

    stats = {
        "total_raw": len(raw_examples),
        "filter_reasons": Counter(),
        "language_dist": Counter(),
        "chat_sources": Counter(),
        "passed_filter": 0,
        "final_count": 0,
        "telugu_transliterated": 0,
    }

    # Filter and process
    filtered_examples = []

    for example in raw_examples:
        passes, reason = quality_filter(example, min_response_len)

        if not passes:
            stats["filter_reasons"][reason] += 1
            continue

        stats["passed_filter"] += 1

        # Categorize language before transliteration
        output = example.get("output", "")
        lang_cat = categorize_language(output)
        stats["language_dist"][lang_cat] += 1

        # Track chat source
        chat = example.get("meta", {}).get("chat", "unknown")
        stats["chat_sources"][chat] += 1

        # Transliterate if has Telugu
        if has_telugu(output):
            transliterated = transliterate_text(output)
            stats["telugu_transliterated"] += 1
        else:
            transliterated = output

        # Convert to ChatML
        chatml_example = convert_to_chatml(example, transliterated)
        filtered_examples.append(chatml_example)

    print(f"After filtering: {len(filtered_examples)} examples")

    # Sample if we have more than target
    if len(filtered_examples) > target_count:
        # Stratified sampling by language category
        # Aim for: 57% Telugu-dominant, 29% code-switched, 14% English-only
        by_lang = {
            "telugu_dominant": [],
            "code_switched": [],
            "english_only": [],
            "english_dominant": [],
        }

        for ex in filtered_examples:
            # Re-categorize based on original (metadata has had_telugu flag)
            if ex["metadata"]["had_telugu"]:
                # Was Telugu, now check original lang tag
                if ex["metadata"]["lang"] == "te":
                    by_lang["telugu_dominant"].append(ex)
                else:
                    by_lang["code_switched"].append(ex)
            else:
                by_lang["english_only"].append(ex)

        # Calculate target per category
        targets = {
            "telugu_dominant": int(target_count * 0.57),  # 200
            "code_switched": int(target_count * 0.29),  # 100
            "english_only": int(target_count * 0.14),  # 50
        }

        sampled = []
        for cat, target in targets.items():
            available = by_lang.get(cat, [])
            actual = min(len(available), target)
            if available:
                sampled.extend(random.sample(available, actual))
            print(f"  {cat}: sampled {actual} from {len(available)}")

        # If we didn't reach target, fill from any category
        remaining = target_count - len(sampled)
        if remaining > 0:
            already_sampled_ids = {ex["metadata"]["original_id"] for ex in sampled}
            candidates = [
                ex
                for ex in filtered_examples
                if ex["metadata"]["original_id"] not in already_sampled_ids
            ]
            if candidates:
                additional = random.sample(candidates, min(remaining, len(candidates)))
                sampled.extend(additional)
                print(f"  Added {len(additional)} more to reach target")

        final_examples = sampled
    else:
        final_examples = filtered_examples

    stats["final_count"] = len(final_examples)

    # Shuffle final examples
    random.shuffle(final_examples)

    # Write output
    print(f"Writing {len(final_examples)} examples to {output_path}...")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        for ex in final_examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    # Write stats
    stats_path = output_path.with_suffix(".stats.json")
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "total_raw": stats["total_raw"],
                "passed_filter": stats["passed_filter"],
                "final_count": stats["final_count"],
                "telugu_transliterated": stats["telugu_transliterated"],
                "filter_reasons": dict(stats["filter_reasons"]),
                "language_dist": dict(stats["language_dist"]),
                "chat_sources": dict(stats["chat_sources"]),
                "target": target_count,
                "min_response_len": min_response_len,
            },
            f,
            indent=2,
        )

    print(f"Stats written to {stats_path}")

    return stats


def main():
    parser = argparse.ArgumentParser(
        description="Curate WhatsApp data for Friday AI training"
    )
    parser.add_argument(
        "--input",
        "-i",
        default="data/sft/iter2/whatsapp_train.jsonl",
        help="Input WhatsApp JSONL file",
    )
    parser.add_argument(
        "--output",
        "-o",
        default="data/instructions/whatsapp_curated_iter2.jsonl",
        help="Output curated JSONL file",
    )
    parser.add_argument(
        "--target",
        "-t",
        type=int,
        default=350,
        help="Target number of examples (default: 350)",
    )
    parser.add_argument(
        "--min-len", type=int, default=15, help="Minimum response length (default: 15)"
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed (default: 42)"
    )

    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    if not input_path.exists():
        print(f"Error: Input file not found: {input_path}", file=sys.stderr)
        sys.exit(1)

    stats = curate_whatsapp(
        input_path=input_path,
        output_path=output_path,
        target_count=args.target,
        min_response_len=args.min_len,
        seed=args.seed,
    )

    print("\n=== Curation Summary ===")
    print(f"Raw examples: {stats['total_raw']}")
    print(f"Passed filter: {stats['passed_filter']}")
    print(f"Final count: {stats['final_count']}")
    print(f"Telugu transliterated: {stats['telugu_transliterated']}")
    print(f"\nFilter rejection reasons:")
    for reason, count in stats["filter_reasons"].most_common():
        print(f"  {reason}: {count}")


if __name__ == "__main__":
    main()
