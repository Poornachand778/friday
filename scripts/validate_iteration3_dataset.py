#!/usr/bin/env python3
"""
Validate Iteration 3 Training Dataset
=====================================

Confirms the interview-only dataset meets quality standards:
- All examples have valid ChatML structure
- Response lengths are appropriate (>20 words target)
- No problematic patterns (Ye?, short responses)
- Language distribution is healthy
"""

import json
import re
from pathlib import Path
from collections import Counter

DATASET_PATH = (
    Path(__file__).parent.parent
    / "data/instructions/iteration3_interview_only_train.jsonl"
)


def count_words(text: str) -> int:
    """Count words in text."""
    return len(text.split())


def detect_language(text: str) -> str:
    """Simple language detection based on script."""
    telugu_pattern = re.compile(r"[\u0C00-\u0C7F]")
    has_telugu = bool(telugu_pattern.search(text))
    has_english = bool(re.search(r"[a-zA-Z]{3,}", text))

    if has_telugu and has_english:
        return "te-en"
    elif has_telugu:
        return "te"
    else:
        return "en"


def check_problematic_patterns(text: str) -> list:
    """Check for patterns that caused iteration 2 failure."""
    issues = []

    # Check for "Ye?" pattern
    if re.search(r"\bYe\?\s*$", text, re.IGNORECASE):
        issues.append("ends_with_ye")

    # Check for very short response
    if count_words(text) <= 5:
        issues.append("very_short")

    # Check for repeated words (degeneration)
    words = text.lower().split()
    if len(words) > 3:
        for i in range(len(words) - 2):
            if words[i] == words[i + 1] == words[i + 2]:
                issues.append("word_repetition")
                break

    # Check for meta-content pollution
    meta_patterns = [
        "voice call",
        "deleted this message",
        "audio omitted",
        "image omitted",
    ]
    for pattern in meta_patterns:
        if pattern.lower() in text.lower():
            issues.append("meta_content")
            break

    return issues


def validate_dataset():
    """Main validation function."""
    print("=" * 60)
    print("Iteration 3 Dataset Validation")
    print("=" * 60)
    print(f"File: {DATASET_PATH}")
    print()

    if not DATASET_PATH.exists():
        print("ERROR: Dataset file not found!")
        return False

    examples = []
    parse_errors = 0

    with open(DATASET_PATH, "r", encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            try:
                data = json.loads(line.strip())
                examples.append(data)
            except json.JSONDecodeError:
                print(f"  Parse error on line {i}")
                parse_errors += 1

    print(f"Total examples: {len(examples)}")
    print(f"Parse errors: {parse_errors}")
    print()

    # Analyze responses
    word_counts = []
    char_counts = []
    languages = []
    all_issues = []
    topics = Counter()

    for ex in examples:
        messages = ex.get("messages", [])
        metadata = ex.get("metadata", {})

        # Get assistant response (last message with role=assistant)
        assistant_msg = None
        for msg in reversed(messages):
            if msg.get("role") == "assistant":
                assistant_msg = msg.get("content", "")
                break

        if assistant_msg:
            word_counts.append(count_words(assistant_msg))
            char_counts.append(len(assistant_msg))
            languages.append(detect_language(assistant_msg))
            issues = check_problematic_patterns(assistant_msg)
            if issues:
                all_issues.extend(issues)

        # Track topics
        if "topic" in metadata:
            topics[metadata["topic"]] += 1

    # Statistics
    print("Response Length Analysis:")
    print("-" * 40)
    if word_counts:
        avg_words = sum(word_counts) / len(word_counts)
        min_words = min(word_counts)
        max_words = max(word_counts)

        # Distribution buckets
        very_short = sum(1 for w in word_counts if w <= 5)
        short = sum(1 for w in word_counts if 6 <= w <= 20)
        medium = sum(1 for w in word_counts if 21 <= w <= 50)
        long = sum(1 for w in word_counts if w > 50)

        print(f"  Average words/response: {avg_words:.1f}")
        print(f"  Min words: {min_words}")
        print(f"  Max words: {max_words}")
        print()
        print("  Distribution:")
        print(
            f"    ≤5 words (very short):  {very_short:3d} ({100*very_short/len(word_counts):5.1f}%)"
        )
        print(
            f"    6-20 words (short):     {short:3d} ({100*short/len(word_counts):5.1f}%)"
        )
        print(
            f"    21-50 words (medium):   {medium:3d} ({100*medium/len(word_counts):5.1f}%)"
        )
        print(
            f"    >50 words (long):       {long:3d} ({100*long/len(word_counts):5.1f}%)"
        )

    print()
    print("Language Distribution:")
    print("-" * 40)
    lang_counts = Counter(languages)
    for lang, count in lang_counts.most_common():
        pct = 100 * count / len(languages)
        label = {
            "te-en": "Telugu-English mix",
            "te": "Telugu only",
            "en": "English only",
        }.get(lang, lang)
        print(f"  {label}: {count:3d} ({pct:5.1f}%)")

    print()
    print("Topic Distribution:")
    print("-" * 40)
    for topic, count in topics.most_common():
        pct = 100 * count / len(examples)
        print(f"  {topic}: {count:3d} ({pct:5.1f}%)")

    print()
    print("Quality Check:")
    print("-" * 40)
    issue_counts = Counter(all_issues)
    if issue_counts:
        print("  ISSUES FOUND:")
        for issue, count in issue_counts.most_common():
            print(f"    {issue}: {count}")
    else:
        print("  No problematic patterns detected!")

    # Final verdict
    print()
    print("=" * 60)

    # Quality criteria
    passed = True
    checks = []

    # Check 1: Average response length > 20 words
    if avg_words >= 20:
        checks.append(("Avg response length ≥20 words", True, f"{avg_words:.1f}"))
    else:
        checks.append(("Avg response length ≥20 words", False, f"{avg_words:.1f}"))
        passed = False

    # Check 2: Less than 10% very short responses
    very_short_pct = 100 * very_short / len(word_counts)
    if very_short_pct < 10:
        checks.append(("Very short responses <10%", True, f"{very_short_pct:.1f}%"))
    else:
        checks.append(("Very short responses <10%", False, f"{very_short_pct:.1f}%"))
        passed = False

    # Check 3: No "Ye?" patterns
    ye_count = issue_counts.get("ends_with_ye", 0)
    if ye_count == 0:
        checks.append(("No 'Ye?' patterns", True, "0"))
    else:
        checks.append(("No 'Ye?' patterns", False, str(ye_count)))
        passed = False

    # Check 4: No meta-content
    meta_count = issue_counts.get("meta_content", 0)
    if meta_count == 0:
        checks.append(("No meta-content pollution", True, "0"))
    else:
        checks.append(("No meta-content pollution", False, str(meta_count)))
        passed = False

    # Check 5: Good language mix (at least 30% code-switched)
    code_switch_pct = 100 * lang_counts.get("te-en", 0) / len(languages)
    if code_switch_pct >= 30:
        checks.append(("Code-switching ≥30%", True, f"{code_switch_pct:.1f}%"))
    else:
        checks.append(("Code-switching ≥30%", False, f"{code_switch_pct:.1f}%"))
        # Not a hard failure, just a warning

    print("Quality Checks:")
    for check_name, check_passed, value in checks:
        status = "PASS" if check_passed else "FAIL"
        print(f"  [{status}] {check_name}: {value}")

    print()
    if passed:
        print("RESULT: Dataset PASSED all critical quality checks")
        print("Ready for Iteration 3 training!")
    else:
        print("RESULT: Dataset FAILED some quality checks")
        print("Review issues before training.")

    print("=" * 60)

    return passed


if __name__ == "__main__":
    validate_dataset()
