#!/usr/bin/env python3
"""
Friday Interview Data Auto-Fix Script
======================================

Automatically fixes common quality issues in interview data:
1. Adds "Boss" naturally to responses
2. Removes/rewrites hedging phrases
3. Flags long responses for manual review

Usage:
    python scripts/training/fix_interview_data.py
    python scripts/training/fix_interview_data.py --dry-run
    python scripts/training/fix_interview_data.py --output data/interviews/fixed/
"""

import argparse
import json
import re
import shutil
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple


# Hedging patterns to remove/rewrite
HEDGING_REPLACEMENTS = [
    # (pattern, replacement, description)
    (r"^I think\s+", "", "Remove 'I think' at start"),
    (r"^I believe\s+", "", "Remove 'I believe' at start"),
    (r"^I guess\s+", "", "Remove 'I guess' at start"),
    (r"^Maybe\s+", "", "Remove 'Maybe' at start"),
    (r"^Perhaps\s+", "", "Remove 'Perhaps' at start"),
    (r"\bI think\b", "", "Remove 'I think' mid-sentence"),
    (r"\bi believe\b", "", "Remove 'i believe' mid-sentence"),
    (r"\bmaybe\b", "", "Remove 'maybe'"),
    (r"\bperhaps\b", "", "Remove 'perhaps'"),
    (r"\bmight be\b", "is", "Replace 'might be' with 'is'"),
    (r"\bcould be\b", "is", "Replace 'could be' with 'is'"),
    (r"\bpossibly\b", "", "Remove 'possibly'"),
]

# Flattery patterns to remove
FLATTERY_PATTERNS = [
    (r"\bgreat question[!.]?\s*", "", "Remove 'great question'"),
    (r"\bhappy to help[!.]?\s*", "", "Remove 'happy to help'"),
    (r"\bcertainly[!,]?\s*", "", "Remove 'certainly'"),
    (r"\babsolutely[!,]?\s*", "", "Remove 'absolutely'"),
]


@dataclass
class FixResult:
    """Result of fixing a single response"""

    original: str
    fixed: str
    changes: List[str] = field(default_factory=list)
    needs_manual_review: bool = False
    review_reason: str = ""


def add_boss_naturally(response: str) -> Tuple[str, str]:
    """
    Add 'Boss' naturally to the beginning of a response.
    Returns (fixed_response, change_description)
    """
    response = response.strip()

    # Already has Boss
    if re.search(r"^\s*boss[,\s]", response, re.IGNORECASE):
        return response, ""

    # Patterns for natural Boss placement
    # If starts with "Yes", "No", "Okay", "Sure", "Right", "Correct"
    affirmatives = [
        (r"^(Yes)[,\s]", r"Yes Boss, "),
        (r"^(No)[,\s]", r"No Boss, "),
        (r"^(Okay)[,\s]", r"Okay Boss, "),
        (r"^(Sure)[,\s]", r"Sure Boss, "),
        (r"^(Right)[,\s]", r"Right Boss, "),
        (r"^(Correct)[,\s]", r"Correct Boss, "),
        (r"^(Definitely)[,\s]", r"Definitely Boss, "),
        (r"^(Actually)[,\s]", r"Actually Boss, "),
        (r"^(Honestly)[,\s]", r"Honestly Boss, "),
        (r"^(Look)[,\s]", r"Look Boss, "),
    ]

    for pattern, replacement in affirmatives:
        if re.match(pattern, response, re.IGNORECASE):
            # Preserve original case
            match = re.match(pattern, response, re.IGNORECASE)
            word = match.group(1)
            fixed = re.sub(
                pattern, f"{word} Boss, ", response, count=1, flags=re.IGNORECASE
            )
            return fixed, f"Added 'Boss' after '{word}'"

    # If starts with Telugu word/phrase, add "Boss, " at start
    telugu_starters = [
        r"^(Naku|Nenu|Mana|Ee|Ala|Ante|Inka|Adhi|Enti|Chala|Haha|Hmm|Interesting)",
    ]
    for pattern in telugu_starters:
        if re.match(pattern, response, re.IGNORECASE):
            return (
                f"Boss, {response[0].lower()}{response[1:]}",
                "Added 'Boss, ' at start (Telugu)",
            )

    # If starts with "I" (common), restructure
    if response.startswith("I ") or response.startswith("I'"):
        # "I want..." -> "Boss, I want..."
        return f"Boss, {response[0].lower()}{response[1:]}", "Added 'Boss, ' at start"

    # If starts with "It" or "There" or "The"
    if re.match(r"^(It|There|The|This|That)\s", response):
        return f"Boss, {response[0].lower()}{response[1:]}", "Added 'Boss, ' at start"

    # Default: Just prepend "Boss, " and lowercase first letter
    if response[0].isupper():
        return (
            f"Boss, {response[0].lower()}{response[1:]}",
            "Added 'Boss, ' at start (default)",
        )
    else:
        return f"Boss, {response}", "Added 'Boss, ' at start (default)"


def remove_hedging(response: str) -> Tuple[str, List[str]]:
    """
    Remove hedging phrases from response.
    Returns (fixed_response, list_of_changes)
    """
    changes = []
    fixed = response

    for pattern, replacement, description in HEDGING_REPLACEMENTS:
        if re.search(pattern, fixed, re.IGNORECASE):
            fixed = re.sub(pattern, replacement, fixed, flags=re.IGNORECASE)
            changes.append(description)

    # Clean up double spaces and punctuation issues
    fixed = re.sub(r"\s+", " ", fixed)
    fixed = re.sub(r"\s+([,.])", r"\1", fixed)
    fixed = re.sub(r"^[,.\s]+", "", fixed)  # Remove leading punctuation

    return fixed.strip(), changes


def remove_flattery(response: str) -> Tuple[str, List[str]]:
    """Remove flattery phrases"""
    changes = []
    fixed = response

    for pattern, replacement, description in FLATTERY_PATTERNS:
        if re.search(pattern, fixed, re.IGNORECASE):
            fixed = re.sub(pattern, replacement, fixed, flags=re.IGNORECASE)
            changes.append(description)

    return fixed.strip(), changes


def check_length(response: str) -> Tuple[bool, str]:
    """Check if response needs manual trimming"""
    words = len(response.split())

    if words > 150:
        return True, f"VERY LONG: {words} words (target <60)"
    elif words > 80:
        return True, f"LONG: {words} words (target <60)"

    return False, ""


def fix_response(response: str) -> FixResult:
    """Apply all fixes to a response"""
    result = FixResult(original=response, fixed=response)

    # Step 1: Remove flattery
    result.fixed, flattery_changes = remove_flattery(result.fixed)
    result.changes.extend(flattery_changes)

    # Step 2: Remove hedging
    result.fixed, hedging_changes = remove_hedging(result.fixed)
    result.changes.extend(hedging_changes)

    # Step 3: Add Boss
    result.fixed, boss_change = add_boss_naturally(result.fixed)
    if boss_change:
        result.changes.append(boss_change)

    # Step 4: Check length
    needs_review, review_reason = check_length(result.fixed)
    if needs_review:
        result.needs_manual_review = True
        result.review_reason = review_reason

    # Clean up
    result.fixed = result.fixed.strip()

    # Ensure first letter after "Boss, " is properly cased
    result.fixed = re.sub(
        r"^(Boss,\s+)([a-z])", lambda m: m.group(1) + m.group(2).lower(), result.fixed
    )

    return result


def process_interview_file(
    input_path: Path, output_path: Path, dry_run: bool = False
) -> Dict:
    """Process a single interview file"""
    with open(input_path) as f:
        data = json.load(f)

    stats = {
        "file": input_path.stem,
        "exchanges": 0,
        "boss_added": 0,
        "hedging_removed": 0,
        "needs_review": 0,
        "changes": [],
    }

    for exchange in data.get("exchanges", []):
        stats["exchanges"] += 1
        original = exchange.get("response", "")

        result = fix_response(original)

        if result.changes:
            stats["changes"].append(
                {
                    "turn": exchange.get("turn"),
                    "changes": result.changes,
                    "needs_review": result.needs_manual_review,
                    "review_reason": result.review_reason,
                }
            )

            if any("Boss" in c for c in result.changes):
                stats["boss_added"] += 1
            if any("hedg" in c.lower() or "Remove" in c for c in result.changes):
                stats["hedging_removed"] += 1
            if result.needs_manual_review:
                stats["needs_review"] += 1

        # Update the exchange
        if not dry_run:
            exchange["original_response"] = original
            exchange["response"] = result.fixed
            exchange["auto_fixed"] = True
            exchange["fix_changes"] = result.changes
            if result.needs_manual_review:
                exchange["needs_review"] = True
                exchange["review_reason"] = result.review_reason

    # Save if not dry run
    if not dry_run:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    return stats


def main():
    parser = argparse.ArgumentParser(
        description="Auto-fix interview data quality issues"
    )
    parser.add_argument(
        "--input",
        "-i",
        type=str,
        default="data/interviews/raw",
        help="Input directory with interview JSON files",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default="data/interviews/fixed",
        help="Output directory for fixed files",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be changed without modifying files",
    )
    parser.add_argument(
        "--backup", action="store_true", help="Create backup of original files"
    )

    args = parser.parse_args()

    input_dir = Path(args.input)
    output_dir = Path(args.output)

    if not input_dir.exists():
        print(f"Error: Input directory not found: {input_dir}")
        return

    files = sorted(input_dir.glob("*.json"))
    if not files:
        print("No JSON files found")
        return

    # Header
    print("\n" + "=" * 70)
    print("FRIDAY INTERVIEW DATA AUTO-FIX")
    print("=" * 70)
    print(f"Mode: {'DRY RUN' if args.dry_run else 'LIVE'}")
    print(f"Input: {input_dir}")
    print(f"Output: {output_dir}")
    print(f"Files: {len(files)}")
    print("=" * 70)

    # Create backup if requested
    if args.backup and not args.dry_run:
        backup_dir = (
            input_dir.parent / f"raw_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )
        shutil.copytree(input_dir, backup_dir)
        print(f"\nBackup created: {backup_dir}")

    # Process all files
    total_stats = {
        "files": 0,
        "exchanges": 0,
        "boss_added": 0,
        "hedging_removed": 0,
        "needs_review": 0,
    }

    all_changes = []

    for file_path in files:
        output_path = output_dir / file_path.name
        stats = process_interview_file(file_path, output_path, args.dry_run)

        total_stats["files"] += 1
        total_stats["exchanges"] += stats["exchanges"]
        total_stats["boss_added"] += stats["boss_added"]
        total_stats["hedging_removed"] += stats["hedging_removed"]
        total_stats["needs_review"] += stats["needs_review"]

        if stats["changes"]:
            all_changes.append(stats)
            print(f"\n  {file_path.stem}")
            print(f"    Boss added: {stats['boss_added']}/{stats['exchanges']}")
            print(f"    Hedging removed: {stats['hedging_removed']}")
            print(f"    Needs review: {stats['needs_review']}")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Files processed: {total_stats['files']}")
    print(f"Total exchanges: {total_stats['exchanges']}")
    print(
        f"Boss added: {total_stats['boss_added']} ({100*total_stats['boss_added']/total_stats['exchanges']:.1f}%)"
    )
    print(f"Hedging removed: {total_stats['hedging_removed']}")
    print(
        f"Needs manual review: {total_stats['needs_review']} ({100*total_stats['needs_review']/total_stats['exchanges']:.1f}%)"
    )

    if args.dry_run:
        print("\n[DRY RUN - No files modified]")
    else:
        print(f"\nFixed files saved to: {output_dir}")
        print("\nNext step: Run review_data.py to manually review flagged responses")
        print(f"  python scripts/training/review_data.py {output_dir}")

    # Show sample changes
    print("\n" + "=" * 70)
    print("SAMPLE CHANGES")
    print("=" * 70)

    samples_shown = 0
    for file_stats in all_changes:
        for change in file_stats["changes"]:
            if samples_shown >= 5:
                break
            print(f"\n  {file_stats['file']} turn {change['turn']}:")
            for c in change["changes"][:3]:
                print(f"    - {c}")
            if change["needs_review"]:
                print(f"    ⚠ {change['review_reason']}")
            samples_shown += 1
        if samples_shown >= 5:
            break


if __name__ == "__main__":
    main()
