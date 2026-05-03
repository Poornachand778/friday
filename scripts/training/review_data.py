#!/usr/bin/env python3
"""
Friday Training Data Review Tool
================================

Interactive tool to review and fix training data quality.
Goes through each example, showing issues and allowing fixes.

Usage:
    python scripts/training/review_data.py data/interviews/raw/
    python scripts/training/review_data.py data/interviews/raw/ --output data/interviews/reviewed/
"""

import argparse
import json
import re
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Quality patterns
HEDGING_PATTERNS = [
    (r"\bi think\b", "I think"),
    (r"\bmaybe\b", "maybe"),
    (r"\bperhaps\b", "perhaps"),
    (r"\bmight be\b", "might be"),
    (r"\bcould be\b", "could be"),
    (r"\bi believe\b", "I believe"),
    (r"\bi guess\b", "I guess"),
    (r"\bpossibly\b", "possibly"),
]

FLATTERY_PATTERNS = [
    (r"\bgreat question\b", "great question"),
    (r"\bhappy to help\b", "happy to help"),
    (r"\bcertainly\b", "certainly"),
    (r"\babsolutely\b", "absolutely"),
    (r"\bof course!\b", "of course!"),
]


def highlight_issues(text: str) -> Tuple[str, List[str]]:
    """Highlight issues in text and return list of issues"""
    issues = []
    highlighted = text

    for pattern, name in HEDGING_PATTERNS:
        if re.search(pattern, text, re.IGNORECASE):
            issues.append(f"HEDGING: '{name}'")

    for pattern, name in FLATTERY_PATTERNS:
        if re.search(pattern, text, re.IGNORECASE):
            issues.append(f"FLATTERY: '{name}'")

    # Check for Boss usage
    if not re.search(r"\bboss\b", text, re.IGNORECASE):
        issues.append("NO_BOSS: Missing 'Boss' usage")

    # Check length
    word_count = len(text.split())
    if word_count > 80:
        issues.append(f"LONG: {word_count} words (target: <60)")

    return highlighted, issues


def print_exchange(
    idx: int, total: int, question: str, response: str, issues: List[str]
):
    """Print an exchange for review"""
    print("\n" + "=" * 70)
    print(f"EXAMPLE {idx + 1}/{total}")
    print("=" * 70)

    print(f"\n📝 QUESTION:")
    print(f"   {question}")

    print(f"\n💬 RESPONSE ({len(response.split())} words):")
    # Wrap response for readability
    words = response.split()
    line = "   "
    for word in words:
        if len(line) + len(word) > 70:
            print(line)
            line = "   " + word
        else:
            line += " " + word if line.strip() else word
    if line.strip():
        print(line)

    if issues:
        print(f"\n⚠️  ISSUES:")
        for issue in issues:
            print(f"   • {issue}")
    else:
        print(f"\n✅ NO ISSUES DETECTED")


def get_user_action() -> str:
    """Get user action for current example"""
    print("\n" + "-" * 40)
    print("Actions:")
    print("  [a] Accept as-is")
    print("  [e] Edit response")
    print("  [d] Delete example")
    print("  [s] Skip (decide later)")
    print("  [q] Quit and save progress")
    print("-" * 40)

    while True:
        action = input("Action: ").strip().lower()
        if action in ["a", "e", "d", "s", "q"]:
            return action
        print("Invalid action. Use a/e/d/s/q")


def edit_response(current: str) -> str:
    """Let user edit response"""
    print("\nCurrent response:")
    print(f"  {current}")
    print("\nEnter new response (or press Enter to keep current):")
    print("(Type 'MULTILINE' to enter multiple lines, end with 'END')")

    new = input("> ").strip()

    if new.upper() == "MULTILINE":
        lines = []
        print("Enter lines (type 'END' when done):")
        while True:
            line = input()
            if line.strip().upper() == "END":
                break
            lines.append(line)
        new = " ".join(lines)

    return new if new else current


def load_interview_file(path: Path) -> Dict:
    """Load a single interview file"""
    with open(path) as f:
        return json.load(f)


def save_interview_file(path: Path, data: Dict):
    """Save interview file"""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def review_session(session: Dict, session_name: str) -> Tuple[Dict, Dict]:
    """Review a single interview session"""
    exchanges = session.get("exchanges", [])
    reviewed = []
    stats = {"accepted": 0, "edited": 0, "deleted": 0, "skipped": 0}

    print(f"\n{'#' * 70}")
    print(f"SESSION: {session_name}")
    print(f"Topic: {session.get('topic', 'unknown')}")
    print(f"Exchanges: {len(exchanges)}")
    print(f"{'#' * 70}")

    for i, exchange in enumerate(exchanges):
        question = exchange.get("question", "")
        response = exchange.get("response", "")

        _, issues = highlight_issues(response)
        print_exchange(i, len(exchanges), question, response, issues)

        action = get_user_action()

        if action == "q":
            # Save remaining as skipped
            for remaining in exchanges[i:]:
                reviewed.append(remaining)
                stats["skipped"] += 1
            break

        elif action == "a":
            reviewed.append(exchange)
            stats["accepted"] += 1
            print("✓ Accepted")

        elif action == "e":
            new_response = edit_response(response)
            exchange["response"] = new_response
            exchange["reviewed"] = True
            exchange["reviewed_at"] = datetime.now().isoformat()
            reviewed.append(exchange)
            stats["edited"] += 1
            print("✓ Edited and saved")

        elif action == "d":
            stats["deleted"] += 1
            print("✗ Deleted")

        elif action == "s":
            exchange["needs_review"] = True
            reviewed.append(exchange)
            stats["skipped"] += 1
            print("→ Skipped for later")

    # Update session
    session["exchanges"] = reviewed
    session["reviewed"] = True
    session["reviewed_at"] = datetime.now().isoformat()
    session["review_stats"] = stats

    return session, stats


def print_summary(all_stats: Dict):
    """Print review summary"""
    print("\n" + "=" * 70)
    print("REVIEW SUMMARY")
    print("=" * 70)
    print(f"  Accepted: {all_stats['accepted']}")
    print(f"  Edited:   {all_stats['edited']}")
    print(f"  Deleted:  {all_stats['deleted']}")
    print(f"  Skipped:  {all_stats['skipped']}")
    total = sum(all_stats.values())
    print(f"  Total:    {total}")

    if all_stats["edited"] > 0 or all_stats["deleted"] > 0:
        print("\n✓ Changes saved")


def quick_fix_boss(response: str) -> str:
    """Suggest quick fix to add Boss"""
    # Common patterns to add Boss
    if response.lower().startswith("yes"):
        return "Yes Boss" + response[3:]
    elif response.lower().startswith("no"):
        return "No Boss" + response[2:]
    elif response.lower().startswith("okay"):
        return "Okay Boss" + response[4:]
    elif response.lower().startswith("sure"):
        return "Sure Boss" + response[4:]
    else:
        return f"Boss, {response[0].lower()}{response[1:]}"


def batch_fix_mode(input_dir: Path, output_dir: Path):
    """Non-interactive batch fix mode"""
    print("\n" + "=" * 70)
    print("BATCH FIX MODE")
    print("=" * 70)
    print("Applying automatic fixes:")
    print("  • Adding 'Boss' where missing")
    print("  • Flagging hedging for manual review")
    print()

    files = list(input_dir.glob("*.json"))
    total_fixed = 0
    total_flagged = 0

    for file_path in files:
        session = load_interview_file(file_path)
        exchanges = session.get("exchanges", [])

        for exchange in exchanges:
            response = exchange.get("response", "")
            _, issues = highlight_issues(response)

            # Auto-fix: Add Boss if missing
            if "NO_BOSS" in str(issues):
                exchange["original_response"] = response
                exchange["response"] = quick_fix_boss(response)
                exchange["auto_fixed"] = True
                total_fixed += 1

            # Flag hedging for manual review
            if any("HEDGING" in issue for issue in issues):
                exchange["needs_review"] = True
                exchange["review_reason"] = "hedging"
                total_flagged += 1

        # Save to output
        output_path = output_dir / file_path.name
        save_interview_file(output_path, session)
        print(f"  Processed: {file_path.name}")

    print()
    print(f"Auto-fixed (Boss added): {total_fixed}")
    print(f"Flagged for review: {total_flagged}")
    print(f"Output: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Review Friday training data")
    parser.add_argument(
        "input", type=str, help="Input directory with interview JSON files"
    )
    parser.add_argument(
        "--output", "-o", type=str, help="Output directory for reviewed files"
    )
    parser.add_argument(
        "--batch", action="store_true", help="Batch fix mode (non-interactive)"
    )
    parser.add_argument("--file", "-f", type=str, help="Review single file only")

    args = parser.parse_args()

    input_dir = Path(args.input)
    output_dir = Path(args.output) if args.output else input_dir.parent / "reviewed"

    if not input_dir.exists():
        print(f"Error: Input directory not found: {input_dir}")
        sys.exit(1)

    if args.batch:
        batch_fix_mode(input_dir, output_dir)
        return

    # Interactive mode
    if args.file:
        files = [input_dir / args.file]
    else:
        files = sorted(input_dir.glob("*.json"))

    if not files:
        print("No JSON files found")
        sys.exit(1)

    print("\n" + "=" * 70)
    print("FRIDAY TRAINING DATA REVIEW")
    print("=" * 70)
    print(f"Input:  {input_dir}")
    print(f"Output: {output_dir}")
    print(f"Files:  {len(files)}")
    print("\nReview each example and fix issues.")
    print("Goal: Every response should have 'Boss', no hedging, <60 words")
    print("=" * 70)

    input("\nPress Enter to start review...")

    all_stats = {"accepted": 0, "edited": 0, "deleted": 0, "skipped": 0}

    for file_path in files:
        session = load_interview_file(file_path)
        reviewed_session, stats = review_session(session, file_path.stem)

        # Update totals
        for key in all_stats:
            all_stats[key] += stats[key]

        # Save reviewed session
        output_path = output_dir / file_path.name
        save_interview_file(output_path, reviewed_session)

        # Check if user wants to continue
        if stats.get("quit"):
            break

    print_summary(all_stats)


if __name__ == "__main__":
    main()
