#!/usr/bin/env python3
"""
Screenplay Parser for Friday AI
================================

Parses screenplay PDFs (Celtx format), Fountain, and Markdown files into structured data.
Stores parsed data in the new screenplay database schema.

Supports:
- Celtx PDF format (like "Can we not")
- Fountain format (.fountain files)
- Markdown screenplay drafts (.md files)
- Telugu + English bilingual dialogue

Usage:
    python scripts/parse_screenplay.py --pdf "data/film/scripts/Script Can we not.pdf"
    python scripts/parse_screenplay.py --fountain "data/film/scripts/GUSAGUSALU.fountain"
    python scripts/parse_screenplay.py --markdown "data/film/scripts/aa_janta_naduma_draft.md"
    python scripts/parse_screenplay.py --list  # List all parsed projects
"""

import argparse
import os
import re
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import hashlib

# Add repo root to path
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from dotenv import load_dotenv

load_dotenv()

import pdfplumber
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from db.screenplay_schema import (
    Base,
    ScreenplayProject,
    ScreenplayCharacter,
    ScreenplayScene,
    SceneElement,
    DialogueLine,
    SceneEmbedding,
)

# Database connection
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = os.getenv("DB_PORT", "5432")
DB_NAME = os.getenv("DB_NAME", "vectordb")
DB_USER = os.getenv("DB_USER", "vectoruser")
DB_PASSWORD = os.getenv("DB_PASSWORD", "friday")

DATABASE_URL = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"


def get_session():
    """Create database session"""
    engine = create_engine(DATABASE_URL)
    Session = sessionmaker(bind=engine)
    return Session()


# ============================================================================
# PDF PARSER (Celtx Format)
# ============================================================================


class CeltxPDFParser:
    """
    Parser for Celtx-exported PDF screenplays.

    Celtx format characteristics:
    - Scene headings: "1    INT. LOCATION - TIME" (number + tab + heading)
    - Action: Regular paragraphs
    - Character: UPPERCASE, centered
    - Parenthetical: (in parentheses)
    - Dialogue: Below character name
    - Transitions: RIGHT aligned (CUT TO, FADE IN)
    """

    # Regex patterns for Celtx format
    # Note: Celtx PDFs have spaced characters like "E X T . H O U S E"
    SCENE_HEADING_PATTERN = re.compile(
        r"^(\d+)\s+"  # Scene number
        r"(INT\.|EXT\.|INT/EXT\.?|I\s*N\s*T\s*\.|E\s*X\s*T\s*\.|I\s*N\s*T\s*/\s*E\s*X\s*T\s*\.?)\s*"  # INT/EXT (with possible spaces)
        r"(.+?)"  # Location
        r"(?:\s*-\s*(.+))?$",  # Time of day (optional)
        re.IGNORECASE,
    )

    # Pattern for spaced-out scene headings (Celtx PDF export quirk)
    SPACED_SCENE_HEADING_PATTERN = re.compile(
        r"^(\d+)\s+"  # Scene number
        r"(E\s*X\s*T|I\s*N\s*T|I\s*N\s*T\s*/\s*E\s*X\s*T)\s*\.\s*"  # INT/EXT with spaces
        r"(.+)$",  # Rest of heading
        re.IGNORECASE,
    )

    CHARACTER_PATTERN = re.compile(
        r"^([A-Z][A-Z\s]+)(?:\s*\(([^)]+)\))?$"  # UPPERCASE NAME (optional parenthetical)
    )

    TRANSITION_PATTERN = re.compile(
        r"^(CUT TO|FADE IN|FADE OUT|DISSOLVE TO|SMASH CUT|MATCH CUT)[\s:]*$",
        re.IGNORECASE,
    )

    PARENTHETICAL_PATTERN = re.compile(r"^\(([^)]+)\)$")

    def __init__(self, pdf_path: str):
        self.pdf_path = Path(pdf_path)
        self.pages: List[str] = []
        self.scenes: List[Dict] = []
        self.characters: set = set()

    def extract_text_from_pdf(self) -> List[str]:
        """Extract text from PDF, page by page"""
        pages = []
        with pdfplumber.open(self.pdf_path) as pdf:
            for page in pdf.pages:
                text = page.extract_text()
                if text:
                    pages.append(text)
        return pages

    def _normalize_spaced_text(self, text: str) -> str:
        """Remove extra spaces from spaced-out text like 'E X T' -> 'EXT'"""
        # Check if text has letter-space-letter pattern
        if re.search(r"[A-Z] [A-Z]", text):
            # Remove single spaces between single uppercase letters
            result = re.sub(r"(?<=[A-Z]) (?=[A-Z](?:\s|$|[^a-z]))", "", text)
            # Clean up multiple spaces
            result = re.sub(r"\s+", " ", result)
            return result.strip()
        return text

    def parse(self) -> Dict:
        """Parse the entire screenplay"""
        print(f"Parsing PDF: {self.pdf_path}")

        # Extract text
        self.pages = self.extract_text_from_pdf()
        all_text = "\n".join(self.pages)

        # Split into lines
        lines = all_text.split("\n")

        # Parse line by line
        current_scene = None
        current_element = None
        element_order = 0

        for i, line in enumerate(lines):
            line = line.strip()

            # Skip empty lines and page numbers
            if not line or line.isdigit() or line.startswith("Created using"):
                continue

            # Check for spaced-out scene heading (Celtx PDF quirk)
            # First try the spaced pattern
            spaced_match = self.SPACED_SCENE_HEADING_PATTERN.match(line)
            if spaced_match:
                # Normalize the spaced text
                line = self._normalize_spaced_text(line)

            # Check for scene heading
            scene_match = self.SCENE_HEADING_PATTERN.match(line)

            # If no match, try normalizing the line first
            if not scene_match and re.match(r"^\d+\s+[A-Z\s]+\.", line):
                normalized = self._normalize_spaced_text(line)
                scene_match = self.SCENE_HEADING_PATTERN.match(normalized)
            if scene_match:
                # Save previous scene
                if current_scene:
                    self.scenes.append(current_scene)

                # Parse scene heading
                scene_num = int(scene_match.group(1))
                int_ext = scene_match.group(2).upper().rstrip(".")
                location_time = scene_match.group(3).strip()

                # Split location and time
                location = location_time
                time_of_day = (
                    scene_match.group(4).strip() if scene_match.group(4) else None
                )

                # Further split if there's a dash in location
                if "-" in location and not time_of_day:
                    parts = location.rsplit("-", 1)
                    location = parts[0].strip()
                    time_of_day = parts[1].strip()

                current_scene = {
                    "scene_number": scene_num,
                    "int_ext": int_ext,
                    "location": location,
                    "time_of_day": time_of_day,
                    "elements": [],
                }
                element_order = 0
                continue

            # Skip if no scene yet
            if not current_scene:
                continue

            # Check for transition
            if self.TRANSITION_PATTERN.match(line):
                current_scene["elements"].append(
                    {
                        "type": "transition",
                        "order": element_order,
                        "content": {"text": line},
                    }
                )
                element_order += 1
                continue

            # Check for character name (dialogue start)
            # First normalize if it's spaced out
            normalized_line = self._normalize_spaced_text(line)
            char_match = self.CHARACTER_PATTERN.match(normalized_line)
            if char_match and len(normalized_line) < 50:  # Character names are short
                char_name = char_match.group(1).strip()
                parenthetical = char_match.group(2)  # e.g., "PHONE CALL V.O."

                # Avoid false positives (action lines that happen to be uppercase)
                if char_name not in ["THE", "A", "AN", "FROM", "TO", "IN", "ON", "AT"]:
                    self.characters.add(char_name)

                    current_element = {
                        "type": "dialogue",
                        "order": element_order,
                        "content": {
                            "character": char_name,
                            "parenthetical": parenthetical,
                            "lines": [],
                        },
                    }
                    current_scene["elements"].append(current_element)
                    element_order += 1
                    continue

            # Check for parenthetical (within dialogue)
            paren_match = self.PARENTHETICAL_PATTERN.match(line)
            if (
                paren_match
                and current_element
                and current_element["type"] == "dialogue"
            ):
                # This is a translation or direction within dialogue
                last_line = current_element["content"]["lines"]
                if last_line:
                    last_line[-1]["translation"] = paren_match.group(1)
                continue

            # Check if this is dialogue text (after character name)
            if current_element and current_element["type"] == "dialogue":
                # Check if this line is dialogue or action
                # Dialogue is typically indented and shorter
                if len(line) < 100 and not line.startswith("_"):
                    current_element["content"]["lines"].append(
                        {"text": line, "translation": None}
                    )
                    continue
                else:
                    # This is action, end dialogue
                    current_element = None

            # Default: action/description
            current_scene["elements"].append(
                {"type": "action", "order": element_order, "content": {"text": line}}
            )
            element_order += 1

        # Save last scene
        if current_scene:
            self.scenes.append(current_scene)

        # Extract title from filename
        title = self.pdf_path.stem.replace("Script ", "").replace("_", " ")

        return {
            "title": title,
            "slug": self._slugify(title),
            "scenes": self.scenes,
            "characters": list(self.characters),
        }

    def _slugify(self, text: str) -> str:
        """Convert title to slug"""
        slug = text.lower()
        slug = re.sub(r"[^\w\s-]", "", slug)
        slug = re.sub(r"[\s_]+", "-", slug)
        return slug.strip("-")


# ============================================================================
# FOUNTAIN PARSER
# ============================================================================


class FountainParser:
    """Parser for Fountain format screenplays"""

    def __init__(self, fountain_path: str):
        self.fountain_path = Path(fountain_path)
        self.scenes = []
        self.characters = set()

    def parse(self) -> Dict:
        """Parse Fountain file using custom parser (fountain library has Python 3.12 issues)"""
        print(f"Parsing Fountain: {self.fountain_path}")

        with open(self.fountain_path, "r", encoding="utf-8") as f:
            content = f.read()

        # Custom Fountain parser
        lines = content.split("\n")
        current_scene = None
        element_order = 0
        i = 0

        while i < len(lines):
            line = lines[i].rstrip()

            # Skip empty lines
            if not line.strip():
                i += 1
                continue

            # Scene heading: starts with INT., EXT., INT/EXT., or forced with .
            if self._is_scene_heading(line):
                # Save previous scene
                if current_scene:
                    self.scenes.append(current_scene)

                # Parse scene heading
                heading = line.lstrip(".")  # Remove forced scene marker if present
                int_ext, location, time_of_day = self._parse_scene_heading(heading)

                current_scene = {
                    "scene_number": len(self.scenes) + 1,
                    "int_ext": int_ext,
                    "location": location,
                    "time_of_day": time_of_day,
                    "elements": [],
                }
                element_order = 0
                i += 1
                continue

            # Transition: ends with TO: or starts with >
            if (
                line.strip().endswith("TO:") or line.strip().startswith(">")
            ) and current_scene:
                current_scene["elements"].append(
                    {
                        "type": "transition",
                        "order": element_order,
                        "content": {"text": line.strip().lstrip(">")},
                    }
                )
                element_order += 1
                i += 1
                continue

            # Character: ALL CAPS, optionally with (V.O.) or (O.S.) or @forced
            if current_scene and self._is_character_line(line, lines, i):
                char_line = line.strip().lstrip("@")
                # Extract character name (remove extensions like (V.O.))
                char_match = re.match(
                    r"^([A-Z][A-Z\s\'\-\.]+?)(?:\s*\(.*\))?$", char_line
                )
                if char_match:
                    char_name = char_match.group(1).strip()
                else:
                    char_name = char_line.split("(")[0].strip()

                self.characters.add(char_name)

                # Check for parenthetical extension in character line
                paren_ext = None
                paren_match = re.search(r"\(([^)]+)\)$", char_line)
                if paren_match:
                    paren_ext = paren_match.group(1)

                dialogue_element = {
                    "type": "dialogue",
                    "order": element_order,
                    "content": {
                        "character": char_name,
                        "parenthetical": paren_ext,
                        "lines": [],
                    },
                }

                i += 1
                # Collect dialogue lines until empty line or new element
                while i < len(lines):
                    dial_line = lines[i].rstrip()

                    # Empty line ends dialogue
                    if not dial_line.strip():
                        break

                    # Parenthetical in dialogue
                    if dial_line.strip().startswith("(") and dial_line.strip().endswith(
                        ")"
                    ):
                        # This is a parenthetical direction
                        dialogue_element["content"][
                            "parenthetical"
                        ] = dial_line.strip()[1:-1]
                        i += 1
                        continue

                    # Check if this is a new scene heading or character
                    if self._is_scene_heading(dial_line) or self._is_character_line(
                        dial_line, lines, i
                    ):
                        break

                    # This is dialogue text
                    text = dial_line.strip()
                    if text:
                        # Check for translation in parentheses at end
                        translation = None
                        trans_match = re.search(r"\(([^)]+)\)\s*$", text)
                        if trans_match and not text.startswith("("):
                            # Only extract if it looks like translation (lowercase English)
                            potential_trans = trans_match.group(1)
                            if re.match(r"^[a-z]", potential_trans):
                                translation = potential_trans
                                text = text[: trans_match.start()].strip()

                        dialogue_element["content"]["lines"].append(
                            {"text": text, "translation": translation}
                        )
                    i += 1

                current_scene["elements"].append(dialogue_element)
                element_order += 1
                continue

            # Action: everything else that's not blank
            if current_scene and line.strip():
                # Collect multi-line action blocks
                action_text = line.strip()
                i += 1
                while i < len(lines):
                    next_line = lines[i].rstrip()
                    if not next_line.strip():
                        break
                    if self._is_scene_heading(next_line) or self._is_character_line(
                        next_line, lines, i
                    ):
                        break
                    if next_line.strip().endswith("TO:"):
                        break
                    action_text += "\n" + next_line.strip()
                    i += 1

                current_scene["elements"].append(
                    {
                        "type": "action",
                        "order": element_order,
                        "content": {"text": action_text},
                    }
                )
                element_order += 1
                continue

            i += 1

        # Save last scene
        if current_scene:
            self.scenes.append(current_scene)

        title = self.fountain_path.stem.replace("_", " ")

        return {
            "title": title,
            "slug": self._slugify(title),
            "scenes": self.scenes,
            "characters": list(self.characters),
        }

    def _is_scene_heading(self, line: str) -> bool:
        """Check if line is a scene heading"""
        stripped = line.strip().upper()
        # Standard scene heading prefixes
        if stripped.startswith(("INT.", "EXT.", "INT/EXT.", "I/E.")):
            return True
        # Forced scene heading with period
        if (
            line.strip().startswith(".")
            and len(line.strip()) > 1
            and line.strip()[1] != "."
        ):
            return True
        return False

    def _is_character_line(self, line: str, all_lines: list, current_idx: int) -> bool:
        """Check if line is a character cue (character name before dialogue)"""
        stripped = line.strip()
        if not stripped:
            return False

        # Forced character with @
        if stripped.startswith("@"):
            return True

        # Exclude common non-character ALL CAPS lines
        non_character_patterns = [
            "FADE TO",
            "FADE IN",
            "FADE OUT",
            "CUT TO",
            "DISSOLVE TO",
            "MUSIC",
            "SOUND",
            "TITLE",
            "SUPER:",
            "MONTAGE",
            "FLASHBACK",
            "END OF",
            "THE END",
            "LATER",
            "CONTINUED",
            "MORE",
        ]
        upper_stripped = stripped.upper()
        for pattern in non_character_patterns:
            if upper_stripped.startswith(pattern) or upper_stripped == pattern:
                return False

        # Must be ALL CAPS (with possible extensions like (V.O.))
        # Remove parenthetical and V.O/O.S extensions to check name
        name_part = re.sub(r"\s*\(.*\)$", "", stripped)
        name_part = re.sub(r"\s+V\.?O\.?$", "", name_part)
        name_part = re.sub(r"\s+O\.?S\.?$", "", name_part)
        if not name_part:
            return False

        # Must be uppercase letters, spaces, apostrophes, periods, and numbers (for BOY 1, etc)
        if not re.match(r"^[A-Z][A-Z\s\'\-\.0-9]+$", name_part):
            return False

        # Next line should exist and be dialogue (not empty, not a scene heading)
        if current_idx + 1 < len(all_lines):
            next_line = all_lines[current_idx + 1].strip()
            # If next line is empty, this might still be character with dialogue coming
            # If next line is action-like or scene heading, this is not a character
            if next_line and self._is_scene_heading(next_line):
                return False
            # Character lines are usually followed by dialogue or parenthetical
            return True

        return False

    def _parse_scene_heading(self, heading: str) -> Tuple[str, str, Optional[str]]:
        """Parse scene heading into components"""
        int_ext = "INT"
        location = heading
        time_of_day = None

        # Remove trailing scene numbers (e.g., "7 A.M 1 1" -> "7 A.M")
        # Pattern: ends with one or more space-separated numbers
        heading = re.sub(r"\s+\d+\s*\d*\s*$", "", heading).strip()

        if heading.upper().startswith("INT."):
            int_ext = "INT"
            heading = heading[4:].strip()
        elif heading.upper().startswith("EXT."):
            int_ext = "EXT"
            heading = heading[4:].strip()
        elif heading.upper().startswith("INT/EXT.") or heading.upper().startswith(
            "I/E."
        ):
            int_ext = "INT/EXT"
            heading = re.sub(
                r"^(INT/EXT\.|I/E\.)\s*", "", heading, flags=re.IGNORECASE
            ).strip()

        # Split by dash for location and time
        # Time of day is usually the last part (MORNING, DAY, NIGHT, etc.)
        parts = heading.split("-")
        if len(parts) >= 2:
            # Check if last part looks like time
            last_part = parts[-1].strip().upper()
            if any(
                t in last_part
                for t in [
                    "MORNING",
                    "DAY",
                    "NIGHT",
                    "EVENING",
                    "NOON",
                    "DUSK",
                    "DAWN",
                    "A.M",
                    "P.M",
                ]
            ):
                time_of_day = parts[-1].strip()
                location = "-".join(parts[:-1]).strip()
            else:
                location = heading
        else:
            location = heading

        return int_ext, location, time_of_day

    def _slugify(self, text: str) -> str:
        slug = text.lower()
        slug = re.sub(r"[^\w\s-]", "", slug)
        slug = re.sub(r"[\s_]+", "-", slug)
        return slug.strip("-")


# ============================================================================
# MARKDOWN SCREENPLAY PARSER
# ============================================================================


class MarkdownParser:
    """
    Parser for markdown screenplay drafts.

    Format:
    - Title: # Title
    - Scene headings: ### SCENE XX – Title
    - Dialogue: **Character:** dialogue text
    - Action: _italic text_ or regular paragraphs
    - Voiceover: > VOICE: text
    """

    def __init__(self, md_path: str):
        self.md_path = Path(md_path)
        self.scenes = []
        self.characters = set()

    def parse(self) -> Dict:
        """Parse markdown screenplay file"""
        print(f"Parsing Markdown: {self.md_path}")

        with open(self.md_path, "r", encoding="utf-8") as f:
            content = f.read()

        lines = content.split("\n")
        current_scene = None
        element_order = 0
        title = None

        i = 0
        while i < len(lines):
            line = lines[i].rstrip()

            # Skip empty lines
            if not line.strip():
                i += 1
                continue

            # Title: # Title
            if line.startswith("# ") and not title:
                title = line[2:].strip()
                i += 1
                continue

            # Separator
            if line.strip() == "---":
                i += 1
                continue

            # Scene heading: ### SCENE XX – Title
            scene_match = re.match(
                r"^###\s*SCENE\s*(\d+)\s*[–-]\s*(.+)$", line, re.IGNORECASE
            )
            if scene_match:
                # Save previous scene
                if current_scene:
                    self.scenes.append(current_scene)

                scene_num = int(scene_match.group(1))
                scene_title = scene_match.group(2).strip()

                # Try to extract INT/EXT from title
                int_ext = "INT"
                location = scene_title
                time_of_day = None

                # Check for location markers in the scene content
                # Some markdown scripts have location on the next line
                i += 1
                while i < len(lines) and not lines[i].strip():
                    i += 1

                if i < len(lines):
                    next_line = lines[i].strip()
                    # Check for _Location: ..._ pattern
                    loc_match = re.match(
                        r"^_?Location:\s*(.+?)_?$", next_line, re.IGNORECASE
                    )
                    if loc_match:
                        location = loc_match.group(1).strip()
                        i += 1

                current_scene = {
                    "scene_number": scene_num,
                    "int_ext": int_ext,
                    "location": location,
                    "title": scene_title,
                    "time_of_day": time_of_day,
                    "elements": [],
                }
                element_order = 0
                continue

            # Dialogue: **Character:** text - colon is INSIDE the bold markers
            # Format: **Hero:** or **గాయత్రి:**
            dialogue_match = re.match(r"^\*\*([^*:]+):\*\*\s*(.*)$", line)
            if dialogue_match and current_scene:
                char_name = dialogue_match.group(1).strip()
                dialogue_text = dialogue_match.group(2).strip()

                self.characters.add(char_name)

                # Collect multi-line dialogue (bullet points or continuation)
                dialogue_lines = []
                if dialogue_text:
                    dialogue_lines.append({"text": dialogue_text, "translation": None})

                i += 1
                while i < len(lines):
                    next_line = lines[i].rstrip()

                    # Empty line ends dialogue (unless followed by bullet)
                    if not next_line.strip():
                        # Peek ahead to see if there are more bullet points
                        if i + 1 < len(lines) and lines[i + 1].strip().startswith("* "):
                            i += 1
                            continue
                        break

                    # Bullet points for dialogue continuation
                    if next_line.strip().startswith("* "):
                        text = next_line.strip()[2:].strip()
                        dialogue_lines.append({"text": text, "translation": None})
                        i += 1
                    # Check if this is a new dialogue or element
                    elif next_line.strip().startswith(("**", ">", "###", "---", "_")):
                        break
                    elif next_line.strip():
                        # Continuation line (same dialogue block)
                        text = next_line.strip()
                        if dialogue_lines:
                            dialogue_lines[-1]["text"] += " " + text
                        else:
                            dialogue_lines.append({"text": text, "translation": None})
                        i += 1
                    else:
                        break

                # Only add if we have dialogue content
                if dialogue_lines:
                    current_scene["elements"].append(
                        {
                            "type": "dialogue",
                            "order": element_order,
                            "content": {
                                "character": char_name,
                                "parenthetical": None,
                                "lines": dialogue_lines,
                            },
                        }
                    )
                    element_order += 1
                continue

            # Voice-over: > VOICE: text or > text
            if line.strip().startswith(">") and current_scene:
                vo_text = line.strip()[1:].strip()

                # Check for VOICE pattern
                vo_match = re.match(
                    r"^VOICE\s*\([^)]*\):\s*(.*)$", vo_text, re.IGNORECASE
                )
                if vo_match:
                    vo_text = vo_match.group(1).strip()
                    char_name = "VOICE"
                else:
                    char_name = "VOICE"

                self.characters.add(char_name)

                # Collect continuation
                i += 1
                while i < len(lines):
                    next_line = lines[i].rstrip()
                    if next_line.strip().startswith(">"):
                        vo_text += " " + next_line.strip()[1:].strip()
                        i += 1
                    elif next_line.strip() and not next_line.strip().startswith(
                        ("**", "_", "###", "---")
                    ):
                        vo_text += " " + next_line.strip()
                        i += 1
                    else:
                        break

                current_scene["elements"].append(
                    {
                        "type": "dialogue",
                        "order": element_order,
                        "content": {
                            "character": char_name,
                            "parenthetical": "V.O.",
                            "lines": [{"text": vo_text, "translation": None}],
                        },
                    }
                )
                element_order += 1
                continue

            # Action: _italic text_ or regular text
            if current_scene and line.strip():
                # Remove italic markers
                action_text = line.strip()
                if action_text.startswith("_") and action_text.endswith("_"):
                    action_text = action_text[1:-1]
                elif action_text.startswith("_"):
                    action_text = action_text[1:]

                # Collect multi-line action
                i += 1
                while i < len(lines):
                    next_line = lines[i].rstrip()
                    if not next_line.strip():
                        break
                    if next_line.strip().startswith(("**", ">", "###", "---")):
                        break
                    next_text = next_line.strip()
                    if next_text.startswith("_"):
                        next_text = next_text[1:]
                    if next_text.endswith("_"):
                        next_text = next_text[:-1]
                    action_text += "\n" + next_text
                    i += 1

                current_scene["elements"].append(
                    {
                        "type": "action",
                        "order": element_order,
                        "content": {"text": action_text},
                    }
                )
                element_order += 1
                continue

            i += 1

        # Save last scene
        if current_scene:
            self.scenes.append(current_scene)

        # Clean up title
        if not title:
            title = self.md_path.stem.replace("_", " ")

        return {
            "title": title,
            "slug": self._slugify(title),
            "scenes": self.scenes,
            "characters": list(self.characters),
        }

    def _slugify(self, text: str) -> str:
        # Handle Telugu and special characters
        slug = text.lower()
        # Replace Telugu characters with transliteration or remove
        slug = re.sub(r"[^\w\s-]", "", slug, flags=re.ASCII)
        if not slug.strip():
            # If all Telugu, use a basic conversion
            slug = "screenplay"
        slug = re.sub(r"[\s_]+", "-", slug)
        return slug.strip("-") or "screenplay"


# ============================================================================
# DATABASE OPERATIONS
# ============================================================================


def save_to_database(parsed_data: Dict, session) -> ScreenplayProject:
    """Save parsed screenplay to database"""
    print(f"\nSaving to database: {parsed_data['title']}")

    # Check if project exists
    existing = (
        session.query(ScreenplayProject).filter_by(slug=parsed_data["slug"]).first()
    )
    if existing:
        print(f"  Project '{parsed_data['slug']}' already exists. Updating...")
        # Delete existing scenes and elements
        for scene in existing.scenes:
            session.delete(scene)
        session.commit()
        project = existing
    else:
        # Create new project
        project = ScreenplayProject(
            title=parsed_data["title"],
            slug=parsed_data["slug"],
            status="draft",
            primary_language="te",
            secondary_language="en",
        )
        session.add(project)
        session.flush()  # Get the ID

    # Add characters
    for char_name in parsed_data["characters"]:
        existing_char = (
            session.query(ScreenplayCharacter)
            .filter_by(project_id=project.id, name=char_name)
            .first()
        )
        if not existing_char:
            char = ScreenplayCharacter(project_id=project.id, name=char_name)
            session.add(char)

    # Add scenes
    for scene_data in parsed_data["scenes"]:
        scene = ScreenplayScene(
            project_id=project.id,
            scene_number=scene_data["scene_number"],
            int_ext=scene_data["int_ext"],
            location=scene_data["location"],
            time_of_day=scene_data.get("time_of_day"),
            narrative_order=float(scene_data["scene_number"]),
            status="active",
            tags=[],
        )
        session.add(scene)
        session.flush()

        # Add elements
        for elem_data in scene_data["elements"]:
            element = SceneElement(
                scene_id=scene.id,
                element_type=elem_data["type"],
                order_index=elem_data["order"],
                content=elem_data["content"],
            )
            session.add(element)
            session.flush()  # Get element.id

            # Add dialogue lines if dialogue type
            if elem_data["type"] == "dialogue":
                for i, line_data in enumerate(elem_data["content"].get("lines", [])):
                    dl = DialogueLine(
                        element_id=element.id,
                        character_name=elem_data["content"]["character"],
                        parenthetical=elem_data["content"].get("parenthetical"),
                        text=line_data["text"],
                        translation=line_data.get("translation"),
                        language=(
                            "te"
                            if any("\u0C00" <= c <= "\u0C7F" for c in line_data["text"])
                            else "en"
                        ),
                        line_order=i,
                    )
                    session.add(dl)

    session.commit()
    print(
        f"  Saved {len(parsed_data['scenes'])} scenes, {len(parsed_data['characters'])} characters"
    )
    return project


def list_projects(session):
    """List all screenplay projects"""
    projects = session.query(ScreenplayProject).all()
    print("\nScreenplay Projects:")
    print("-" * 60)
    for p in projects:
        scene_count = session.query(ScreenplayScene).filter_by(project_id=p.id).count()
        print(f"  [{p.id}] {p.title} ({p.slug})")
        print(f"      Status: {p.status}, Scenes: {scene_count}")
    print("-" * 60)


# ============================================================================
# MAIN
# ============================================================================


def main():
    parser = argparse.ArgumentParser(description="Parse screenplay files")
    parser.add_argument("--pdf", type=str, help="Path to Celtx PDF file")
    parser.add_argument("--fountain", type=str, help="Path to Fountain file")
    parser.add_argument("--markdown", type=str, help="Path to Markdown screenplay file")
    parser.add_argument("--slug", type=str, help="Override the project slug")
    parser.add_argument("--list", action="store_true", help="List all projects")
    parser.add_argument(
        "--test", action="store_true", help="Parse without saving to DB"
    )
    args = parser.parse_args()

    session = get_session()

    try:
        if args.list:
            list_projects(session)
            return

        if args.pdf:
            parser_obj = CeltxPDFParser(args.pdf)
            parsed_data = parser_obj.parse()
        elif args.fountain:
            parser_obj = FountainParser(args.fountain)
            parsed_data = parser_obj.parse()
        elif args.markdown:
            parser_obj = MarkdownParser(args.markdown)
            parsed_data = parser_obj.parse()
        else:
            print("Please specify --pdf, --fountain, or --markdown file")
            return

        # Override slug if provided
        if args.slug:
            parsed_data["slug"] = args.slug

        # Print parsed data summary
        print(f"\nParsed: {parsed_data['title']}")
        print(f"  Scenes: {len(parsed_data['scenes'])}")
        print(f"  Characters: {parsed_data['characters']}")

        if not args.test:
            project = save_to_database(parsed_data, session)
            print(f"\nProject saved with ID: {project.id}")
        else:
            print("\n[TEST MODE] Not saving to database")
            # Print first scene as sample
            if parsed_data["scenes"]:
                scene = parsed_data["scenes"][0]
                print(f"\nSample Scene {scene['scene_number']}:")
                print(
                    f"  {scene['int_ext']}. {scene['location']} - {scene.get('time_of_day', 'DAY')}"
                )
                for elem in scene["elements"][:5]:
                    print(f"    [{elem['type']}] {str(elem['content'])[:80]}...")

    finally:
        session.close()


if __name__ == "__main__":
    main()
