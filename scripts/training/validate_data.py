#!/usr/bin/env python3
"""
Friday Training Data Validator
==============================

Validates training data quality before training.
Scores each example and flags issues.

Usage:
    python scripts/training/validate_data.py data/training/train.jsonl
    python scripts/training/validate_data.py data/training/train.jsonl --strict
"""

import argparse
import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Quality patterns
HEDGING_PATTERNS = [
    r"\bi think\b",
    r"\bmaybe\b",
    r"\bperhaps\b",
    r"\bmight be\b",
    r"\bcould be\b",
    r"\bpossibly\b",
    r"\bin my opinion\b",
    r"\bi believe\b",
]

FLATTERY_PATTERNS = [
    r"\bgreat question\b",
    r"\bhappy to help\b",
    r"\bcertainly\b",
    r"\babsolutely\b",
    r"\bof course!\b",
    r"\bmy pleasure\b",
    r"\bexcellent\b",
    r"\bwonderful\b",
]

WRONG_IDENTITY_PATTERNS = [
    r"\bi am (an |a )?ai\b",
    r"\bi am (a )?language model\b",
    r"\bi am chatgpt\b",
    r"\bi am claude\b",
    r"\bi am assistant\b",
    r"\bi don't have (personal )?(feelings|emotions|opinions)\b",
    r"\bas an ai\b",
]


@dataclass
class ValidationResult:
    """Result of validating a single example"""

    index: int
    valid: bool
    score: float  # 0-5
    issues: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    example: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ValidationReport:
    """Complete validation report"""

    file_path: str
    total_examples: int
    valid_count: int
    invalid_count: int
    avg_score: float
    issues_by_type: Dict[str, int] = field(default_factory=dict)
    results: List[ValidationResult] = field(default_factory=list)


class DataValidator:
    """Validates Friday training data quality"""

    def __init__(self, strict: bool = False):
        self.strict = strict
        self.min_score = 3.5 if strict else 3.0

    def load_jsonl(self, path: str) -> List[Dict]:
        """Load JSONL file"""
        examples = []
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    examples.append(json.loads(line))
        return examples

    def validate_example(self, index: int, example: Dict) -> ValidationResult:
        """Validate a single training example"""
        issues = []
        warnings = []
        score = 5.0  # Start at perfect

        # Check structure
        if "messages" not in example:
            issues.append("Missing 'messages' field")
            return ValidationResult(
                index=index, valid=False, score=0, issues=issues, example=example
            )

        messages = example["messages"]

        # Check message structure
        for i, msg in enumerate(messages):
            if "role" not in msg or "content" not in msg:
                issues.append(f"Message {i}: Missing role or content")
                score -= 2.0

        # Get assistant responses for analysis
        assistant_msgs = [
            msg["content"] for msg in messages if msg.get("role") == "assistant"
        ]

        if not assistant_msgs:
            issues.append("No assistant response found")
            score -= 3.0

        # Analyze each assistant response
        for response in assistant_msgs:
            response_lower = response.lower()

            # Check for hedging
            for pattern in HEDGING_PATTERNS:
                if re.search(pattern, response_lower):
                    issues.append(f"Hedging phrase detected: {pattern}")
                    score -= 1.0
                    break  # Only penalize once

            # Check for flattery
            for pattern in FLATTERY_PATTERNS:
                if re.search(pattern, response_lower):
                    issues.append(f"Flattery phrase detected: {pattern}")
                    score -= 1.0
                    break

            # Check for wrong identity
            for pattern in WRONG_IDENTITY_PATTERNS:
                if re.search(pattern, response_lower):
                    issues.append(f"Wrong identity detected: {pattern}")
                    score -= 2.0
                    break

            # Check response length
            word_count = len(response.split())
            if word_count > 100:
                warnings.append(f"Long response: {word_count} words")
                score -= 0.5
            elif word_count > 150:
                issues.append(f"Very long response: {word_count} words")
                score -= 1.0

            # Check for empty or too short
            if word_count < 2:
                warnings.append(f"Very short response: {word_count} words")

        # Check system prompt if present
        system_msgs = [
            msg["content"] for msg in messages if msg.get("role") == "system"
        ]
        if system_msgs:
            system_prompt = system_msgs[0].lower()
            if "friday" not in system_prompt and "boss" not in system_prompt:
                warnings.append("System prompt doesn't mention Friday or Boss")

        # Ensure score is within bounds
        score = max(0, min(5, score))

        valid = score >= self.min_score and len(issues) == 0

        return ValidationResult(
            index=index,
            valid=valid,
            score=score,
            issues=issues,
            warnings=warnings,
            example=example,
        )

    def validate_file(self, path: str) -> ValidationReport:
        """Validate entire training file"""
        print(f"\nValidating: {path}")
        print("=" * 60)

        examples = self.load_jsonl(path)
        results = []
        issues_by_type = {}

        for i, example in enumerate(examples):
            result = self.validate_example(i, example)
            results.append(result)

            # Track issue types
            for issue in result.issues:
                issue_type = issue.split(":")[0] if ":" in issue else issue
                issues_by_type[issue_type] = issues_by_type.get(issue_type, 0) + 1

        valid_count = sum(1 for r in results if r.valid)
        invalid_count = len(results) - valid_count
        avg_score = sum(r.score for r in results) / len(results) if results else 0

        return ValidationReport(
            file_path=path,
            total_examples=len(examples),
            valid_count=valid_count,
            invalid_count=invalid_count,
            avg_score=avg_score,
            issues_by_type=issues_by_type,
            results=results,
        )

    def print_report(self, report: ValidationReport):
        """Print validation report"""
        print(f"\nTotal Examples: {report.total_examples}")
        print(
            f"Valid: {report.valid_count} ({report.valid_count/report.total_examples*100:.1f}%)"
        )
        print(
            f"Invalid: {report.invalid_count} ({report.invalid_count/report.total_examples*100:.1f}%)"
        )
        print(f"Average Score: {report.avg_score:.2f}/5.0")

        if report.issues_by_type:
            print("\nIssues by Type:")
            for issue_type, count in sorted(
                report.issues_by_type.items(), key=lambda x: -x[1]
            ):
                print(f"  - {issue_type}: {count}")

        # Show worst examples
        worst = sorted(report.results, key=lambda r: r.score)[:5]
        if worst and worst[0].score < 4.0:
            print("\nWorst Examples:")
            for result in worst:
                if result.score < 4.0:
                    print(f"\n  Example {result.index} (Score: {result.score:.1f})")
                    if result.issues:
                        for issue in result.issues[:3]:
                            print(f"    ✗ {issue}")
                    if result.warnings:
                        for warning in result.warnings[:2]:
                            print(f"    ⚠ {warning}")

        # Summary
        print("\n" + "=" * 60)
        if report.invalid_count == 0:
            print("✓ ALL EXAMPLES VALID")
        else:
            print(f"✗ {report.invalid_count} EXAMPLES NEED ATTENTION")

    def filter_valid(self, input_path: str, output_path: str) -> Tuple[int, int]:
        """Filter to only valid examples"""
        examples = self.load_jsonl(input_path)
        valid_examples = []

        for i, example in enumerate(examples):
            result = self.validate_example(i, example)
            if result.valid:
                valid_examples.append(example)

        with open(output_path, "w") as f:
            for example in valid_examples:
                f.write(json.dumps(example) + "\n")

        return len(valid_examples), len(examples) - len(valid_examples)

    def analyze_distribution(self, path: str):
        """Analyze data distribution"""
        examples = self.load_jsonl(path)

        # Collect stats
        response_lengths = []
        has_telugu = 0
        has_boss = 0
        topics = {}

        telugu_pattern = re.compile(
            r"[\u0C00-\u0C7F]|baag|nenu|enti|chala|inka", re.IGNORECASE
        )
        boss_pattern = re.compile(r"\bboss\b", re.IGNORECASE)

        for example in examples:
            messages = example.get("messages", [])

            for msg in messages:
                if msg.get("role") == "assistant":
                    content = msg.get("content", "")
                    response_lengths.append(len(content.split()))

                    if telugu_pattern.search(content):
                        has_telugu += 1
                    if boss_pattern.search(content):
                        has_boss += 1

            # Try to categorize by tags
            tags = example.get("tags", [])
            for tag in tags:
                topics[tag] = topics.get(tag, 0) + 1

        print("\n" + "=" * 60)
        print("DATA DISTRIBUTION ANALYSIS")
        print("=" * 60)

        print(f"\nTotal Examples: {len(examples)}")

        if response_lengths:
            print(f"\nResponse Length (words):")
            print(f"  Min: {min(response_lengths)}")
            print(f"  Max: {max(response_lengths)}")
            print(f"  Avg: {sum(response_lengths)/len(response_lengths):.1f}")
            print(f"  Median: {sorted(response_lengths)[len(response_lengths)//2]}")

        print(f"\nLanguage Mix:")
        print(f"  Telugu content: {has_telugu} ({has_telugu/len(examples)*100:.1f}%)")
        print(f"  Boss usage: {has_boss} ({has_boss/len(examples)*100:.1f}%)")

        if topics:
            print(f"\nTopics:")
            for topic, count in sorted(topics.items(), key=lambda x: -x[1])[:10]:
                print(f"  - {topic}: {count}")


def main():
    parser = argparse.ArgumentParser(description="Validate Friday training data")
    parser.add_argument("file", type=str, help="Path to JSONL training file")
    parser.add_argument("--strict", action="store_true", help="Use strict validation")
    parser.add_argument(
        "--filter", type=str, help="Output path for filtered valid data"
    )
    parser.add_argument(
        "--analyze", action="store_true", help="Show distribution analysis"
    )

    args = parser.parse_args()

    if not Path(args.file).exists():
        print(f"Error: File not found: {args.file}")
        sys.exit(1)

    validator = DataValidator(strict=args.strict)

    if args.analyze:
        validator.analyze_distribution(args.file)

    report = validator.validate_file(args.file)
    validator.print_report(report)

    if args.filter:
        valid, removed = validator.filter_valid(args.file, args.filter)
        print(f"\nFiltered: {valid} valid examples saved to {args.filter}")
        print(f"Removed: {removed} invalid examples")


if __name__ == "__main__":
    main()
