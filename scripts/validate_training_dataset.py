#!/usr/bin/env python3
"""
Training Dataset Validation Script for Friday AI

Validates iteration2_combined_train.jsonl against success criteria:
- JSON schema validation (ChatML format)
- Token count analysis (min/max/mean/p95/p99)
- Language detection verification
- Domain distribution check
- Duplicate detection (SHA256 hash)
- System prompt consistency
- Metadata completeness
- Tool-call format validation

Usage:
    python scripts/validate_training_dataset.py
    python scripts/validate_training_dataset.py --dataset data/instructions/custom.jsonl
"""

import json
import hashlib
import argparse
from pathlib import Path
from collections import Counter, defaultdict
from typing import List, Dict, Any, Tuple
import statistics

PROJECT_ROOT = Path(__file__).parent.parent
DEFAULT_DATASET = PROJECT_ROOT / "data/instructions/iteration2_combined_train.jsonl"
REPORT_OUTPUT = PROJECT_ROOT / "data/instructions/iteration2_dataset_report.json"

# Success criteria thresholds
TARGET_TOTAL = (750, 1050)  # Min, Max
DOMAIN_TARGETS = {
    "film": (0.35, 0.45),  # 35-45%
    "persona": (0.25, 0.35),  # 25-35%
    "tools": (0.15, 0.25),  # 15-25% (adjusted for 33 tool examples)
    "general": (0.05, 0.15),  # 5-15%
}
LANG_TARGETS = {
    "te": (0.35, 0.50),  # 35-50% Telugu
    "en": (0.30, 0.40),  # 30-40% English
    "mixed": (0.15, 0.25),  # 15-25% Mixed
}

# Expected ChatML schema
REQUIRED_KEYS = ["messages"]
REQUIRED_MESSAGE_KEYS = ["role", "content"]
VALID_ROLES = ["system", "user", "assistant", "tool"]


def load_dataset(filepath: Path) -> List[Dict[str, Any]]:
    """Load JSONL dataset."""
    data = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if line:
                try:
                    data.append(json.loads(line))
                except json.JSONDecodeError as e:
                    print(f"  ❌ Line {line_num}: Invalid JSON - {e}")
                    raise
    return data


def has_telugu(text: str) -> bool:
    """Check if text contains Telugu characters."""
    for char in text:
        if "\u0C00" <= char <= "\u0C7F":
            return True
    return False


def detect_language(text: str) -> str:
    """Detect language: te, en, or mixed."""
    has_te = has_telugu(text)
    # Simple heuristic: if has Telugu chars, check if also has substantial English
    if has_te:
        # Count Latin chars (rough English proxy)
        latin_count = sum(1 for c in text if c.isalpha() and ord(c) < 128)
        telugu_count = sum(1 for c in text if "\u0C00" <= c <= "\u0C7F")

        if latin_count > 0 and telugu_count > 0:
            return "mixed"
        return "te"
    return "en"


def estimate_tokens(text: str) -> int:
    """Rough token count estimate (actual tokenizer not available)."""
    # Simple heuristic: ~1.3 tokens per word for English, ~1.5 for Telugu
    words = len(text.split())
    telugu_chars = sum(1 for c in text if "\u0C00" <= c <= "\u0C7F")
    # If has Telugu, increase estimate
    multiplier = 1.5 if telugu_chars > 0 else 1.3
    return int(words * multiplier)


def validate_chatml_schema(
    example: Dict[str, Any], index: int
) -> Tuple[bool, List[str]]:
    """Validate example against ChatML schema."""
    errors = []

    # Check required keys
    if "messages" not in example:
        errors.append(f"Example {index}: Missing 'messages' key")
        return False, errors

    messages = example["messages"]
    if not isinstance(messages, list):
        errors.append(f"Example {index}: 'messages' must be a list")
        return False, errors

    if len(messages) < 2:
        errors.append(
            f"Example {index}: Must have at least 2 messages (system + user or user + assistant)"
        )
        return False, errors

    # Validate each message
    for msg_idx, msg in enumerate(messages):
        if not isinstance(msg, dict):
            errors.append(f"Example {index}, Message {msg_idx}: Must be a dict")
            continue

        if "role" not in msg:
            errors.append(f"Example {index}, Message {msg_idx}: Missing 'role'")
        elif msg["role"] not in VALID_ROLES:
            errors.append(
                f"Example {index}, Message {msg_idx}: Invalid role '{msg['role']}'"
            )

        if "content" not in msg and "tool_calls" not in msg:
            errors.append(
                f"Example {index}, Message {msg_idx}: Missing 'content' or 'tool_calls'"
            )

    return len(errors) == 0, errors


def analyze_dataset(data: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Comprehensive dataset analysis."""
    print("\n--- Dataset Analysis ---")

    stats = {
        "total_examples": len(data),
        "valid_examples": 0,
        "schema_errors": [],
        "duplicates": 0,
        "domain_distribution": Counter(),
        "language_distribution": Counter(),
        "source_distribution": Counter(),
        "token_stats": {
            "counts": [],
            "min": 0,
            "max": 0,
            "mean": 0,
            "median": 0,
            "p95": 0,
            "p99": 0,
        },
        "system_prompt_consistency": True,
        "system_prompts": Counter(),
        "tool_examples_count": 0,
        "tool_operations": Counter(),
        "has_metadata": 0,
    }

    seen_hashes = set()

    for idx, example in enumerate(data, 1):
        # Schema validation
        is_valid, errors = validate_chatml_schema(example, idx)
        if is_valid:
            stats["valid_examples"] += 1
        else:
            stats["schema_errors"].extend(errors)

        # Duplicate detection
        content = json.dumps(example.get("messages", []), sort_keys=True)
        content_hash = hashlib.sha256(content.encode()).hexdigest()[:16]
        if content_hash in seen_hashes:
            stats["duplicates"] += 1
        seen_hashes.add(content_hash)

        # Extract metadata
        metadata = example.get("metadata", {})
        if metadata:
            stats["has_metadata"] += 1

            domain = metadata.get("domain", "general")
            stats["domain_distribution"][domain] += 1

            source = metadata.get("source", "unknown")
            stats["source_distribution"][source] += 1

        # Language detection (from user message)
        messages = example.get("messages", [])
        user_msgs = [m for m in messages if m.get("role") == "user"]
        if user_msgs:
            user_text = user_msgs[0].get("content", "")
            lang = detect_language(user_text)
            stats["language_distribution"][lang] += 1

        # System prompt tracking
        system_msgs = [m for m in messages if m.get("role") == "system"]
        if system_msgs:
            system_content = system_msgs[0].get("content", "")[:100]  # First 100 chars
            stats["system_prompts"][system_content] += 1

        # Token count estimation
        full_text = " ".join(m.get("content", "") for m in messages if "content" in m)
        token_count = estimate_tokens(full_text)
        stats["token_stats"]["counts"].append(token_count)

        # Tool operation tracking
        for msg in messages:
            if "tool_calls" in msg:
                stats["tool_examples_count"] += 1
                for tool_call in msg.get("tool_calls", []):
                    tool_name = tool_call.get("name", "unknown")
                    stats["tool_operations"][tool_name] += 1
                break  # Count example once

    # Calculate token statistics
    if stats["token_stats"]["counts"]:
        counts = sorted(stats["token_stats"]["counts"])
        stats["token_stats"]["min"] = counts[0]
        stats["token_stats"]["max"] = counts[-1]
        stats["token_stats"]["mean"] = int(statistics.mean(counts))
        stats["token_stats"]["median"] = int(statistics.median(counts))
        stats["token_stats"]["p95"] = int(counts[int(len(counts) * 0.95)])
        stats["token_stats"]["p99"] = int(counts[int(len(counts) * 0.99)])

    # System prompt consistency check
    stats["system_prompt_consistency"] = (
        len(stats["system_prompts"]) <= 3
    )  # Allow up to 3 variations

    return stats


def check_success_criteria(stats: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """Check dataset against success criteria."""
    print("\n--- Success Criteria Validation ---")

    passed = True
    warnings = []

    total = stats["total_examples"]

    # 1. Total examples in range
    min_total, max_total = TARGET_TOTAL
    if min_total <= total <= max_total:
        print(f"✅ Total examples: {total} (target: {min_total}-{max_total})")
    else:
        print(f"❌ Total examples: {total} (target: {min_total}-{max_total})")
        passed = False

    # 2. No duplicates
    if stats["duplicates"] == 0:
        print(f"✅ No duplicates found")
    else:
        print(f"⚠️  Duplicates found: {stats['duplicates']}")
        warnings.append(f"{stats['duplicates']} duplicate examples")

    # 3. Domain balance
    domain_dist = stats["domain_distribution"]
    print(f"\n📊 Domain Distribution:")
    for domain, (min_pct, max_pct) in DOMAIN_TARGETS.items():
        count = domain_dist.get(domain, 0)
        pct = count / total if total > 0 else 0
        target_str = f"{min_pct*100:.0f}-{max_pct*100:.0f}%"

        if min_pct <= pct <= max_pct:
            print(f"  ✅ {domain}: {count} ({pct*100:.1f}%) - target: {target_str}")
        else:
            print(f"  ⚠️  {domain}: {count} ({pct*100:.1f}%) - target: {target_str}")
            warnings.append(f"{domain} domain outside target range")

    # 4. Language balance (only for WhatsApp subset, not overall)
    lang_dist = stats["language_distribution"]
    print(f"\n🌐 Language Distribution:")
    for lang, (min_pct, max_pct) in LANG_TARGETS.items():
        count = lang_dist.get(lang, 0)
        pct = count / total if total > 0 else 0
        target_str = f"{min_pct*100:.0f}-{max_pct*100:.0f}%"

        if min_pct <= pct <= max_pct:
            print(f"  ✅ {lang}: {count} ({pct*100:.1f}%) - target: {target_str}")
        else:
            print(
                f"  ℹ️  {lang}: {count} ({pct*100:.1f}%) - target: {target_str} (informational)"
            )

    # 5. Schema validation
    if stats["valid_examples"] == total:
        print(f"\n✅ All {total} examples passed schema validation")
    else:
        print(
            f"\n❌ Schema validation failed: {total - stats['valid_examples']} examples have errors"
        )
        passed = False

    # 6. Token statistics
    token_stats = stats["token_stats"]
    print(f"\n📏 Token Statistics:")
    print(f"  Min: {token_stats['min']} tokens")
    print(f"  Max: {token_stats['max']} tokens")
    print(f"  Mean: {token_stats['mean']} tokens")
    print(f"  Median: {token_stats['median']} tokens")
    print(f"  P95: {token_stats['p95']} tokens")
    print(f"  P99: {token_stats['p99']} tokens")

    if token_stats["p95"] > 2048:
        warnings.append(
            f"P95 token count ({token_stats['p95']}) exceeds 2048 - consider increasing max_seq_len"
        )
    else:
        print(f"  ✅ P95 within 2048 token limit")

    # 7. Tool examples coverage
    tool_count = stats["tool_examples_count"]
    print(f"\n🛠️  Tool Examples:")
    print(f"  Total tool-using examples: {tool_count}")

    if tool_count >= 30:
        print(f"  ✅ Tool examples count: {tool_count} (target: 30-40)")
    else:
        print(f"  ⚠️  Tool examples count: {tool_count} (target: 30-40)")
        warnings.append(f"Only {tool_count} tool examples (target: 30-40)")

    print(f"\n  Tool operations coverage:")
    for tool_name, count in stats["tool_operations"].most_common():
        print(f"    {tool_name}: {count}")

    # 8. System prompt consistency
    if stats["system_prompt_consistency"]:
        print(
            f"\n✅ System prompt consistency maintained ({len(stats['system_prompts'])} variations)"
        )
    else:
        print(f"\n⚠️  Many system prompt variations ({len(stats['system_prompts'])})")
        warnings.append(
            f"Too many system prompt variations: {len(stats['system_prompts'])}"
        )

    # 9. Metadata completeness
    metadata_pct = stats["has_metadata"] / total if total > 0 else 0
    if metadata_pct >= 0.95:
        print(
            f"✅ Metadata present: {stats['has_metadata']}/{total} ({metadata_pct*100:.1f}%)"
        )
    else:
        print(
            f"⚠️  Metadata present: {stats['has_metadata']}/{total} ({metadata_pct*100:.1f}%)"
        )
        warnings.append(f"Only {metadata_pct*100:.1f}% of examples have metadata")

    return passed, warnings


def main():
    parser = argparse.ArgumentParser(description="Validate training dataset")
    parser.add_argument(
        "--dataset", type=Path, default=DEFAULT_DATASET, help="Path to dataset file"
    )
    args = parser.parse_args()

    print("=" * 60)
    print("Friday AI - Training Dataset Validation")
    print("=" * 60)
    print(f"\nDataset: {args.dataset}")

    # Load dataset
    print("\n[1/3] Loading dataset...")
    data = load_dataset(args.dataset)
    print(f"  Loaded: {len(data)} examples")

    # Analyze dataset
    print("\n[2/3] Analyzing dataset...")
    stats = analyze_dataset(data)

    # Check success criteria
    print("\n[3/3] Validating against success criteria...")
    passed, warnings = check_success_criteria(stats)

    # Summary
    print("\n" + "=" * 60)
    if passed and len(warnings) == 0:
        print("✅ VALIDATION PASSED - Dataset ready for training!")
    elif passed and len(warnings) > 0:
        print(f"⚠️  VALIDATION PASSED with {len(warnings)} warnings:")
        for w in warnings:
            print(f"  - {w}")
    else:
        print("❌ VALIDATION FAILED - Please fix errors before training")
        if stats["schema_errors"]:
            print(f"\nSchema errors ({len(stats['schema_errors'])}):")
            for err in stats["schema_errors"][:10]:  # Show first 10
                print(f"  - {err}")
    print("=" * 60)

    # Write report
    print(f"\nWriting report to: {REPORT_OUTPUT}")
    report = {
        "dataset_path": str(args.dataset),
        "validation_passed": passed,
        "warnings": warnings,
        "statistics": {
            "total_examples": stats["total_examples"],
            "valid_examples": stats["valid_examples"],
            "duplicates": stats["duplicates"],
            "domain_distribution": dict(stats["domain_distribution"]),
            "language_distribution": dict(stats["language_distribution"]),
            "source_distribution": dict(stats["source_distribution"]),
            "token_stats": stats["token_stats"],
            "tool_examples_count": stats["tool_examples_count"],
            "tool_operations": dict(stats["tool_operations"]),
            "has_metadata": stats["has_metadata"],
            "system_prompt_variations": len(stats["system_prompts"]),
        },
    }

    with open(REPORT_OUTPUT, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"Report written successfully!")

    return 0 if passed else 1


if __name__ == "__main__":
    exit(main())
