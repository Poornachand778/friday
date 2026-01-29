#!/usr/bin/env python3
"""
Interview Export Script
=======================

Exports interview sessions to ChatML training format with quality filtering.

Usage:
    python scripts/interview_export.py                    # Export all sessions
    python scripts/interview_export.py --reviewed-only    # Only reviewed sessions
    python scripts/interview_export.py --stats            # Show statistics
    python scripts/interview_export.py --validate         # Validate output format
"""

import argparse
import hashlib
import json
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

# Paths
DATA_DIR = REPO_ROOT / "data" / "interviews"
RAW_DIR = DATA_DIR / "raw"
REVIEWED_DIR = DATA_DIR / "reviewed"
EXPORTED_DIR = DATA_DIR / "exported"

# System prompt for Friday
FRIDAY_SYSTEM_PROMPT = """You are Friday, Poorna's AI assistant. You blend Telugu and English naturally, addressing him as 'Boss'. Be concise, helpful, and direct. No flattery or excessive formality."""


def load_sessions(
    include_raw: bool = True, include_reviewed: bool = True
) -> List[Dict]:
    """Load all interview sessions"""
    sessions = []

    directories = []
    if include_reviewed:
        directories.append(REVIEWED_DIR)
    if include_raw:
        directories.append(RAW_DIR)

    for directory in directories:
        if not directory.exists():
            continue
        for filepath in directory.glob("*.json"):
            try:
                with open(filepath, "r", encoding="utf-8") as f:
                    session = json.load(f)
                    session["_source_file"] = str(filepath)
                    sessions.append(session)
            except json.JSONDecodeError as e:
                print(f"Warning: Could not parse {filepath}: {e}")

    return sessions


def compute_hash(text: str) -> str:
    """Compute hash for deduplication"""
    return hashlib.sha256(text.encode()).hexdigest()[:16]


def filter_quality(exchange: Dict) -> bool:
    """Filter low-quality exchanges"""
    # Support both old format (user) and new format (response)
    user_response = exchange.get("response", exchange.get("user", ""))

    # Skip empty responses
    if not user_response.strip():
        return False

    # Skip very short responses (less than 10 chars)
    if len(user_response.strip()) < 10:
        return False

    # Skip single-word responses
    words = user_response.split()
    if len(words) < 3:
        return False

    return True


def convert_to_chatml(
    sessions: List[Dict],
    deduplicate: bool = True,
    min_quality: bool = True,
) -> List[Dict]:
    """Convert sessions to ChatML format"""
    examples = []
    seen_hashes: Set[str] = set()

    for session in sessions:
        topic = session.get("topic", "general")
        session_id = session.get("session_id", "unknown")

        for exchange in session.get("exchanges", []):
            # Quality filter
            if min_quality and not filter_quality(exchange):
                continue

            # The interviewer's question becomes what the user asks Friday
            # The user's response is how Friday should respond (capturing their style)
            # Support both old format (assistant/user) and new format (question/response)
            user_input = exchange.get("question", exchange.get("assistant", ""))
            friday_response = exchange.get("response", exchange.get("user", ""))

            if not user_input or not friday_response:
                continue

            # Deduplication
            if deduplicate:
                content_hash = compute_hash(friday_response)
                if content_hash in seen_hashes:
                    continue
                seen_hashes.add(content_hash)

            example = {
                "messages": [
                    {"role": "system", "content": FRIDAY_SYSTEM_PROMPT},
                    {"role": "user", "content": user_input},
                    {"role": "assistant", "content": friday_response},
                ],
                "metadata": {
                    "source": "interview",
                    "session_id": session_id,
                    "topic": topic,
                    "language": exchange.get("language", "en"),
                    "turn": exchange.get("turn", 0),
                },
            }
            examples.append(example)

    return examples


def compute_statistics(examples: List[Dict]) -> Dict[str, Any]:
    """Compute dataset statistics"""
    stats = {
        "total_examples": len(examples),
        "topics": Counter(),
        "languages": Counter(),
        "sources": Counter(),
        "token_estimates": {
            "min": float("inf"),
            "max": 0,
            "total": 0,
        },
        "response_lengths": [],
    }

    for example in examples:
        metadata = example.get("metadata", {})
        stats["topics"][metadata.get("topic", "unknown")] += 1
        stats["languages"][metadata.get("language", "unknown")] += 1
        stats["sources"][metadata.get("source", "unknown")] += 1

        # Estimate tokens (rough: 4 chars per token)
        messages = example.get("messages", [])
        total_chars = sum(len(m.get("content", "")) for m in messages)
        token_estimate = total_chars // 4

        stats["token_estimates"]["min"] = min(
            stats["token_estimates"]["min"], token_estimate
        )
        stats["token_estimates"]["max"] = max(
            stats["token_estimates"]["max"], token_estimate
        )
        stats["token_estimates"]["total"] += token_estimate

        # Response length
        response = messages[-1].get("content", "") if messages else ""
        stats["response_lengths"].append(len(response))

    if examples:
        stats["token_estimates"]["avg"] = stats["token_estimates"]["total"] // len(
            examples
        )
        stats["response_lengths_avg"] = sum(stats["response_lengths"]) // len(
            stats["response_lengths"]
        )
    else:
        stats["token_estimates"]["min"] = 0
        stats["token_estimates"]["avg"] = 0
        stats["response_lengths_avg"] = 0

    # Convert Counters to dicts for JSON serialization
    stats["topics"] = dict(stats["topics"])
    stats["languages"] = dict(stats["languages"])
    stats["sources"] = dict(stats["sources"])
    del stats["response_lengths"]  # Too verbose

    return stats


def validate_chatml(examples: List[Dict]) -> List[str]:
    """Validate ChatML format"""
    errors = []

    for i, example in enumerate(examples):
        if "messages" not in example:
            errors.append(f"Example {i}: Missing 'messages' field")
            continue

        messages = example["messages"]
        if not isinstance(messages, list):
            errors.append(f"Example {i}: 'messages' should be a list")
            continue

        if len(messages) < 2:
            errors.append(f"Example {i}: Need at least 2 messages")
            continue

        # Check roles
        roles = [m.get("role") for m in messages]
        valid_roles = {"system", "user", "assistant"}
        for role in roles:
            if role not in valid_roles:
                errors.append(f"Example {i}: Invalid role '{role}'")

        # Check content
        for j, msg in enumerate(messages):
            if "content" not in msg:
                errors.append(f"Example {i}, message {j}: Missing 'content'")
            elif not msg["content"].strip():
                errors.append(f"Example {i}, message {j}: Empty content")

    return errors


def export_jsonl(examples: List[Dict], output_path: Path) -> None:
    """Export to JSONL format"""
    with open(output_path, "w", encoding="utf-8") as f:
        for example in examples:
            f.write(json.dumps(example, ensure_ascii=False) + "\n")


def main():
    parser = argparse.ArgumentParser(description="Export interviews to training format")
    parser.add_argument(
        "--reviewed-only", action="store_true", help="Only include reviewed sessions"
    )
    parser.add_argument(
        "--no-filter", action="store_true", help="Disable quality filtering"
    )
    parser.add_argument(
        "--no-dedupe", action="store_true", help="Disable deduplication"
    )
    parser.add_argument("--stats", action="store_true", help="Show statistics only")
    parser.add_argument(
        "--validate", action="store_true", help="Validate output format"
    )
    parser.add_argument("--output", "-o", help="Output file path")
    args = parser.parse_args()

    # Ensure directories exist
    EXPORTED_DIR.mkdir(parents=True, exist_ok=True)

    # Load sessions
    include_raw = not args.reviewed_only
    sessions = load_sessions(include_raw=include_raw, include_reviewed=True)

    if not sessions:
        print("No sessions found.")
        return

    print(f"Loaded {len(sessions)} sessions")

    # Convert to ChatML
    examples = convert_to_chatml(
        sessions,
        deduplicate=not args.no_dedupe,
        min_quality=not args.no_filter,
    )

    print(f"Generated {len(examples)} training examples")

    # Show stats
    if args.stats or not args.output:
        stats = compute_statistics(examples)
        print("\n" + "=" * 50)
        print("DATASET STATISTICS")
        print("=" * 50)
        print(f"Total examples: {stats['total_examples']}")
        print(f"\nTopics:")
        for topic, count in sorted(stats["topics"].items(), key=lambda x: -x[1]):
            print(f"  {topic}: {count}")
        print(f"\nLanguages:")
        for lang, count in sorted(stats["languages"].items(), key=lambda x: -x[1]):
            print(f"  {lang}: {count}")
        print(f"\nToken estimates:")
        print(f"  Min: {stats['token_estimates']['min']}")
        print(f"  Max: {stats['token_estimates']['max']}")
        print(f"  Avg: {stats['token_estimates']['avg']}")
        print(f"\nAvg response length: {stats['response_lengths_avg']} chars")

    # Validate
    if args.validate:
        errors = validate_chatml(examples)
        if errors:
            print(f"\n{len(errors)} validation errors:")
            for err in errors[:10]:
                print(f"  - {err}")
            if len(errors) > 10:
                print(f"  ... and {len(errors) - 10} more")
        else:
            print("\nValidation passed!")

    # Export
    if args.output or (not args.stats and not args.validate):
        output_path = (
            Path(args.output)
            if args.output
            else EXPORTED_DIR
            / f"interview_train_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl"
        )
        export_jsonl(examples, output_path)
        print(f"\nExported to: {output_path}")

        # Save stats alongside
        stats_path = output_path.with_suffix(".stats.json")
        stats = compute_statistics(examples)
        with open(stats_path, "w") as f:
            json.dump(stats, f, indent=2)
        print(f"Statistics saved to: {stats_path}")


if __name__ == "__main__":
    main()
