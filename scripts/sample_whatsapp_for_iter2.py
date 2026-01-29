#!/usr/bin/env python3
"""
WhatsApp Data Sampling Script for Friday AI Iteration 2

This script intelligently samples high-quality examples from WhatsApp training data
based on quality criteria, language balance, and tag priorities.

Quality Criteria:
- Prioritize script-tagged and clarifying-tagged examples
- Filter out very short/low-value responses
- Maintain language balance (Telugu/English/Mixed)
- Balance across chat sources
- Preserve personality-rich responses

Usage:
    python scripts/sample_whatsapp_for_iter2.py --dry-run  # Preview without writing
    python scripts/sample_whatsapp_for_iter2.py            # Run sampling
    python scripts/sample_whatsapp_for_iter2.py --target 700  # Custom target
"""

import json
import argparse
import hashlib
import random
from pathlib import Path
from datetime import datetime
from collections import Counter, defaultdict
from typing import List, Dict, Any, Tuple

# Paths
PROJECT_ROOT = Path(__file__).parent.parent
INPUT_FILE = PROJECT_ROOT / "data/sft/iter2/whatsapp_train.jsonl"
OUTPUT_DIR = PROJECT_ROOT / "data/sft/iter2"
SAMPLED_OUTPUT = OUTPUT_DIR / "whatsapp_sampled_train.jsonl"
REJECTED_OUTPUT = OUTPUT_DIR / "rejected_samples.jsonl"
REPORT_OUTPUT = OUTPUT_DIR / "sampling_report.json"

# Low-value single-word responses to filter (after "Boss, " prefix)
LOW_VALUE_RESPONSES = {
    "ha",
    "ok",
    "hmm",
    "yes",
    "no",
    "yep",
    "nope",
    "haha",
    "ya",
    "ha.",
    "ok.",
    "hmm.",
    "exactly",
    "correct",
    "right",
    "true",
    "అవును",
    "హా",
    "హా.",
    "లేదు",
    "ఓకే",
    "సరే",
}

# Patterns indicating placeholder/omitted content
FILTER_PATTERNS = [
    "{PHONE}",
    "{EMAIL}",
    "omitted",
    "deleted",
    "This message was deleted",
]


def load_data(filepath: Path) -> List[Dict[str, Any]]:
    """Load JSONL data from file."""
    data = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line.strip()))
    return data


def get_response_text(output: str) -> str:
    """Extract the actual response text after 'Boss, ' prefix."""
    if output.startswith("Boss, "):
        return output[6:]
    return output


def has_telugu(text: str) -> bool:
    """Check if text contains Telugu characters."""
    for char in text:
        if "\u0C00" <= char <= "\u0C7F":
            return True
    return False


def calculate_quality_score(example: Dict[str, Any]) -> Tuple[float, List[str]]:
    """
    Calculate quality score for an example.
    Returns (score, reasons) where score is 0-100.
    """
    output = example.get("output", "")
    tags = example.get("tags", [])
    lang = example.get("lang", "en")
    response_text = get_response_text(output)

    score = 50.0  # Base score
    reasons = []

    # Tag bonuses (highest priority)
    if "script" in tags:
        score += 25
        reasons.append("+25: script tag")
    if "clarifying" in tags:
        score += 20
        reasons.append("+20: clarifying tag")

    # Length scoring
    response_len = len(response_text)
    if response_len > 100:
        score += 15
        reasons.append(f"+15: long response ({response_len} chars)")
    elif response_len > 50:
        score += 10
        reasons.append(f"+10: medium response ({response_len} chars)")
    elif response_len > 25:
        score += 5
        reasons.append(f"+5: decent length ({response_len} chars)")
    elif response_len <= 12:
        score -= 20
        reasons.append(f"-20: very short ({response_len} chars)")

    # Telugu content bonus (indicates code-switching)
    if has_telugu(response_text):
        score += 10
        reasons.append("+10: contains Telugu")

    # Low-value response penalty
    response_lower = response_text.lower().strip()
    if response_lower in LOW_VALUE_RESPONSES:
        score -= 40
        reasons.append("-40: low-value response")

    # Filter pattern penalty
    for pattern in FILTER_PATTERNS:
        if pattern.lower() in output.lower():
            score -= 50
            reasons.append(f"-50: contains '{pattern}'")
            break

    # Emoji presence (shows personality)
    if any(char in output for char in ["😂", "🥲", "😅", "🤣", "😊"]):
        score += 5
        reasons.append("+5: has expressive emoji")

    # Question mark in response (engaging)
    if "?" in response_text:
        score += 5
        reasons.append("+5: asks question back")

    return max(0, min(100, score)), reasons


def analyze_data(data: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Analyze the full dataset and return statistics."""
    stats = {
        "total": len(data),
        "by_language": Counter(d.get("lang", "unknown") for d in data),
        "by_chat": Counter(d.get("meta", {}).get("chat", "unknown") for d in data),
        "by_tags": Counter(),
        "response_lengths": [],
        "quality_distribution": defaultdict(int),
    }

    for d in data:
        tags = d.get("tags", [])
        for tag in tags:
            stats["by_tags"][tag] += 1

        response_text = get_response_text(d.get("output", ""))
        stats["response_lengths"].append(len(response_text))

        score, _ = calculate_quality_score(d)
        if score >= 80:
            stats["quality_distribution"]["excellent (80+)"] += 1
        elif score >= 60:
            stats["quality_distribution"]["good (60-79)"] += 1
        elif score >= 40:
            stats["quality_distribution"]["medium (40-59)"] += 1
        else:
            stats["quality_distribution"]["low (<40)"] += 1

    return stats


def sample_data(
    data: List[Dict[str, Any]],
    target_count: int,
    min_quality: float = 45.0,
    seed: int = 42,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
    """
    Sample high-quality examples from the data.

    Returns:
        (sampled_examples, rejected_examples, sampling_stats)
    """
    random.seed(seed)

    # Score all examples
    scored_data = []
    for example in data:
        score, reasons = calculate_quality_score(example)
        scored_data.append(
            {
                "example": example,
                "score": score,
                "reasons": reasons,
            }
        )

    # Separate by quality threshold
    high_quality = [d for d in scored_data if d["score"] >= min_quality]
    low_quality = [d for d in scored_data if d["score"] < min_quality]

    # Sort high quality by score (descending)
    high_quality.sort(key=lambda x: x["score"], reverse=True)

    # First pass: always include script-tagged and clarifying-tagged
    must_include = []
    optional_pool = []

    for item in high_quality:
        tags = item["example"].get("tags", [])
        if "script" in tags or "clarifying" in tags:
            must_include.append(item)
        else:
            optional_pool.append(item)

    sampled = must_include.copy()

    # Second pass: sample from optional pool to reach target
    remaining_needed = target_count - len(sampled)

    if remaining_needed > 0 and optional_pool:
        # Balance by chat source
        by_chat = defaultdict(list)
        for item in optional_pool:
            chat = item["example"].get("meta", {}).get("chat", "unknown")
            by_chat[chat].append(item)

        # Sample proportionally from each chat
        chat_counts = Counter()
        for chat, items in by_chat.items():
            # Take top examples by score, proportional to chat size
            proportion = len(items) / len(optional_pool)
            take_count = max(
                1, int(remaining_needed * proportion * 1.2)
            )  # Oversample slightly

            # Sort by score and take top
            items.sort(key=lambda x: x["score"], reverse=True)
            for item in items[:take_count]:
                if len(sampled) < target_count:
                    sampled.append(item)
                    chat_counts[chat] += 1

    # Deduplicate by content hash
    seen_hashes = set()
    unique_sampled = []
    for item in sampled:
        content = item["example"].get("output", "") + item["example"].get("input", "")
        content_hash = hashlib.sha256(content.encode()).hexdigest()[:16]
        if content_hash not in seen_hashes:
            seen_hashes.add(content_hash)
            unique_sampled.append(item)

    # Prepare outputs
    sampled_examples = [item["example"] for item in unique_sampled]
    rejected_examples = [item["example"] for item in low_quality]

    # Calculate sampling statistics
    sampled_langs = Counter(ex.get("lang", "unknown") for ex in sampled_examples)
    sampled_chats = Counter(
        ex.get("meta", {}).get("chat", "unknown") for ex in sampled_examples
    )
    sampled_tags = Counter()
    for ex in sampled_examples:
        for tag in ex.get("tags", []):
            sampled_tags[tag] += 1

    sampling_stats = {
        "input_total": len(data),
        "sampled_count": len(sampled_examples),
        "rejected_count": len(rejected_examples),
        "target_count": target_count,
        "min_quality_threshold": min_quality,
        "must_include_count": len(must_include),
        "sampled_by_language": dict(sampled_langs),
        "sampled_by_chat": dict(sampled_chats),
        "sampled_by_tags": dict(sampled_tags),
        "quality_scores": {
            "min": (
                min(item["score"] for item in unique_sampled) if unique_sampled else 0
            ),
            "max": (
                max(item["score"] for item in unique_sampled) if unique_sampled else 0
            ),
            "avg": (
                sum(item["score"] for item in unique_sampled) / len(unique_sampled)
                if unique_sampled
                else 0
            ),
        },
        "language_balance": {
            "telugu_pct": (
                100 * sampled_langs.get("te", 0) / len(sampled_examples)
                if sampled_examples
                else 0
            ),
            "english_pct": (
                100 * sampled_langs.get("en", 0) / len(sampled_examples)
                if sampled_examples
                else 0
            ),
        },
    }

    return sampled_examples, rejected_examples, sampling_stats


def write_jsonl(data: List[Dict[str, Any]], filepath: Path):
    """Write data to JSONL file."""
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def main():
    parser = argparse.ArgumentParser(description="Sample WhatsApp data for iteration 2")
    parser.add_argument(
        "--dry-run", action="store_true", help="Preview without writing files"
    )
    parser.add_argument(
        "--target", type=int, default=700, help="Target number of samples"
    )
    parser.add_argument(
        "--min-quality", type=float, default=45.0, help="Minimum quality score"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    print("=" * 60)
    print("Friday AI - WhatsApp Data Sampling for Iteration 2")
    print("=" * 60)

    # Load data
    print(f"\nLoading data from: {INPUT_FILE}")
    data = load_data(INPUT_FILE)
    print(f"Loaded {len(data)} examples")

    # Analyze full dataset
    print("\n--- Full Dataset Analysis ---")
    full_stats = analyze_data(data)
    print(f"Total examples: {full_stats['total']}")
    print(f"By language: {dict(full_stats['by_language'])}")
    print(f"By chat: {dict(full_stats['by_chat'])}")
    print(f"Quality distribution: {dict(full_stats['quality_distribution'])}")
    print(f"Script-tagged: {full_stats['by_tags'].get('script', 0)}")
    print(f"Clarifying-tagged: {full_stats['by_tags'].get('clarifying', 0)}")

    # Sample data
    print(
        f"\n--- Sampling (target: {args.target}, min_quality: {args.min_quality}) ---"
    )
    sampled, rejected, stats = sample_data(
        data, target_count=args.target, min_quality=args.min_quality, seed=args.seed
    )

    print(f"Sampled: {stats['sampled_count']} examples")
    print(f"Rejected: {stats['rejected_count']} examples")
    print(f"Must-include (script/clarifying): {stats['must_include_count']}")
    print(
        f"Quality scores: min={stats['quality_scores']['min']:.1f}, max={stats['quality_scores']['max']:.1f}, avg={stats['quality_scores']['avg']:.1f}"
    )
    print(
        f"Language balance: Telugu {stats['language_balance']['telugu_pct']:.1f}%, English {stats['language_balance']['english_pct']:.1f}%"
    )
    print(f"By chat: {stats['sampled_by_chat']}")

    # Show sample examples
    print("\n--- Sample High-Quality Examples ---")
    for i, ex in enumerate(sampled[:5]):
        score, reasons = calculate_quality_score(ex)
        print(f"\n{i+1}. Score: {score:.1f} | Lang: {ex.get('lang')}")
        print(f"   Input: {ex.get('input', '')[:80]}...")
        print(f"   Output: {ex.get('output', '')[:80]}...")
        print(f"   Reasons: {'; '.join(reasons[:3])}")

    if args.dry_run:
        print("\n[DRY RUN] No files written.")
    else:
        # Write outputs
        print(f"\n--- Writing Outputs ---")

        write_jsonl(sampled, SAMPLED_OUTPUT)
        print(f"Wrote {len(sampled)} sampled examples to: {SAMPLED_OUTPUT}")

        write_jsonl(rejected, REJECTED_OUTPUT)
        print(f"Wrote {len(rejected)} rejected examples to: {REJECTED_OUTPUT}")

        # Write report
        report = {
            "timestamp": datetime.utcnow().isoformat(),
            "input_file": str(INPUT_FILE),
            "output_file": str(SAMPLED_OUTPUT),
            "parameters": {
                "target": args.target,
                "min_quality": args.min_quality,
                "seed": args.seed,
            },
            "full_dataset_stats": {
                "total": full_stats["total"],
                "by_language": dict(full_stats["by_language"]),
                "by_chat": dict(full_stats["by_chat"]),
                "quality_distribution": dict(full_stats["quality_distribution"]),
            },
            "sampling_stats": stats,
        }

        with open(REPORT_OUTPUT, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        print(f"Wrote sampling report to: {REPORT_OUTPUT}")

    print("\n" + "=" * 60)
    print("Sampling complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
