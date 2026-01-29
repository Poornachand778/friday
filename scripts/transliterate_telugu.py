#!/usr/bin/env python3
"""
Telugu Transliteration Script for Friday AI

Converts native Telugu script to romanized text for TTS compatibility.
XTTS v2 doesn't support Telugu natively, so we transliterate to romanized
text that can be pronounced phonetically.

Example:
  "నిన్ను ఎలా మర్చిపోతానే" → "ninnu elaa marchipOtaanE"
  "Boss, నిన్ను ఎలా" → "Boss, ninnu elaa"

Usage:
  python scripts/transliterate_telugu.py --text "నిన్ను ఎలా"
  python scripts/transliterate_telugu.py --file input.jsonl --output output.jsonl
"""

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Optional

from indic_transliteration import sanscript
from indic_transliteration.sanscript import transliterate


def is_telugu_char(char: str) -> bool:
    """Check if a character is in the Telugu Unicode range."""
    return "\u0C00" <= char <= "\u0C7F"


def has_telugu(text: str) -> bool:
    """Check if text contains any Telugu characters."""
    return any(is_telugu_char(c) for c in text)


def telugu_to_roman(text: str, scheme: str = sanscript.ITRANS) -> str:
    """
    Convert Telugu Unicode to romanized text.

    Args:
        text: Input text (may contain Telugu, English, or mixed)
        scheme: Transliteration scheme (ITRANS, IAST, HK, etc.)

    Returns:
        Text with Telugu converted to romanized form, English unchanged
    """
    if not text or not has_telugu(text):
        return text

    result = []
    telugu_buffer = []

    for char in text:
        if is_telugu_char(char):
            telugu_buffer.append(char)
        else:
            # Flush Telugu buffer if non-empty
            if telugu_buffer:
                telugu_text = "".join(telugu_buffer)
                roman_text = transliterate(telugu_text, sanscript.TELUGU, scheme)
                result.append(roman_text)
                telugu_buffer = []
            result.append(char)

    # Flush any remaining Telugu
    if telugu_buffer:
        telugu_text = "".join(telugu_buffer)
        roman_text = transliterate(telugu_text, sanscript.TELUGU, scheme)
        result.append(roman_text)

    return "".join(result)


def normalize_romanized(text: str) -> str:
    """
    Normalize romanized text for TTS readability.

    - Lowercase for consistency
    - Clean up unusual characters from ITRANS
    - Improve pronunciation hints
    """
    # Remove ITRANS-specific markers that might confuse TTS
    text = text.replace(".h", "")  # Visarga marker
    text = text.replace(".m", "m")  # Anusvara simplification
    text = text.replace("~n", "n")  # Simplify nasal
    text = text.replace("~N", "n")

    # Make more TTS-friendly
    # Capital vowels in ITRANS indicate long vowels - keep them for now
    # but can optionally lowercase

    return text


def transliterate_text(text: str, normalize: bool = True) -> str:
    """
    Main transliteration function.

    Converts Telugu to romanized and optionally normalizes.
    """
    roman = telugu_to_roman(text)
    if normalize:
        roman = normalize_romanized(roman)
    return roman


def transliterate_jsonl_field(
    input_path: Path,
    output_path: Path,
    fields: list[str] = ["output"],
    preserve_original: bool = True,
) -> dict:
    """
    Transliterate specific fields in a JSONL file.

    Args:
        input_path: Input JSONL file
        output_path: Output JSONL file
        fields: List of fields to transliterate
        preserve_original: If True, keeps original in field_original

    Returns:
        Statistics about the transliteration
    """
    stats = {
        "total": 0,
        "telugu_found": 0,
        "transliterated": 0,
        "unchanged": 0,
        "errors": 0,
    }

    with open(input_path, "r", encoding="utf-8") as fin, open(
        output_path, "w", encoding="utf-8"
    ) as fout:

        for line in fin:
            stats["total"] += 1
            try:
                obj = json.loads(line.strip())

                for field in fields:
                    if field in obj and obj[field]:
                        original = obj[field]

                        if has_telugu(original):
                            stats["telugu_found"] += 1
                            transliterated = transliterate_text(original)

                            if preserve_original:
                                obj[f"{field}_original"] = original

                            obj[field] = transliterated
                            stats["transliterated"] += 1
                        else:
                            stats["unchanged"] += 1

                fout.write(json.dumps(obj, ensure_ascii=False) + "\n")

            except Exception as e:
                stats["errors"] += 1
                print(f"Error on line {stats['total']}: {e}", file=sys.stderr)
                # Write original line on error
                fout.write(line)

    return stats


def main():
    parser = argparse.ArgumentParser(
        description="Transliterate Telugu text to romanized form for TTS"
    )

    subparsers = parser.add_subparsers(dest="command", help="Commands")

    # Text command
    text_parser = subparsers.add_parser("text", help="Transliterate text string")
    text_parser.add_argument("--input", "-i", required=True, help="Input text")
    text_parser.add_argument(
        "--no-normalize", action="store_true", help="Don't normalize output"
    )

    # File command
    file_parser = subparsers.add_parser("file", help="Transliterate JSONL file")
    file_parser.add_argument("--input", "-i", required=True, help="Input JSONL file")
    file_parser.add_argument("--output", "-o", required=True, help="Output JSONL file")
    file_parser.add_argument(
        "--fields",
        "-f",
        nargs="+",
        default=["output"],
        help="Fields to transliterate (default: output)",
    )
    file_parser.add_argument(
        "--no-preserve", action="store_true", help="Don't preserve original text"
    )

    # Test command
    test_parser = subparsers.add_parser("test", help="Run transliteration tests")

    args = parser.parse_args()

    if args.command == "text":
        result = transliterate_text(args.input, not args.no_normalize)
        print(f"Input:  {args.input}")
        print(f"Output: {result}")

    elif args.command == "file":
        input_path = Path(args.input)
        output_path = Path(args.output)

        if not input_path.exists():
            print(f"Error: Input file not found: {input_path}", file=sys.stderr)
            sys.exit(1)

        print(f"Transliterating {input_path} -> {output_path}")
        print(f"Fields: {args.fields}")

        stats = transliterate_jsonl_field(
            input_path,
            output_path,
            fields=args.fields,
            preserve_original=not args.no_preserve,
        )

        print("\nStatistics:")
        for key, value in stats.items():
            print(f"  {key}: {value}")

    elif args.command == "test":
        run_tests()

    else:
        parser.print_help()


def run_tests():
    """Run transliteration tests."""
    test_cases = [
        # Pure Telugu
        ("నిన్ను ఎలా మర్చిపోతానే", True),
        ("బాగున్నాను", True),
        # Code-switched
        ("Boss, నిన్ను ఎలా", True),
        ("Shop ki వెళ్ళాం, sir అక్కడ test చేశారు", True),
        # Pure English
        ("Hello, how are you?", False),
        ("Boss, doing good", False),
        # Numbers and symbols
        ("100 రూపాయలు", True),
        ("😂 హహా", True),
    ]

    print("Running transliteration tests...\n")
    passed = 0
    failed = 0

    for text, should_have_telugu in test_cases:
        has_te = has_telugu(text)
        result = transliterate_text(text)
        result_has_te = has_telugu(result)

        # After transliteration, result should not have Telugu
        if should_have_telugu:
            success = has_te and not result_has_te
        else:
            success = not has_te and result == text

        status = "PASS" if success else "FAIL"
        if success:
            passed += 1
        else:
            failed += 1

        print(f"[{status}] '{text}'")
        print(f"       -> '{result}'")
        print()

    print(f"\nResults: {passed} passed, {failed} failed")
    return failed == 0


if __name__ == "__main__":
    main()
