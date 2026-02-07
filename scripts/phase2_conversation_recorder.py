#!/usr/bin/env python3
"""
Phase 2: Behavioral Mode Conversation Recorder
===============================================

Records conversations that teach Friday HOW to think:
- INVESTIGATOR: Ask probing questions when unclear
- CRITIC: Roast cringy ideas with sarcasm
- STORYTELLER: Teach cinematically when knowledgeable
- BRAINSTORM: Build ideas collaboratively

Usage:
    python scripts/phase2_conversation_recorder.py

Commands during recording:
    save     - Save conversation and start new scenario
    redo     - Restart current conversation
    skip     - Skip current scenario
    stats    - Show session statistics
    export   - Export all conversations to JSONL
    quit     - End session
"""

import json
import os
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional
import re

# Paths
DATA_DIR = Path(__file__).parent.parent / "data" / "phase2"
CONVERSATIONS_FILE = DATA_DIR / "behavioral_conversations.jsonl"
SESSION_FILE = DATA_DIR / "current_session.json"

# Friday system prompt - behavioral focused
FRIDAY_SYSTEM_PROMPT = """You are Friday, Poorna's AI assistant. You think and respond based on context:

BEHAVIORAL MODES:

1. INVESTIGATOR MODE (when topic is unclear):
   - Ask probing questions to understand
   - Don't pretend to know what you don't
   - Connect dots as information comes in
   - Build clear picture before suggesting

2. CRITIC MODE (when topic is cringy/clichéd):
   - Playful sarcasm and roasting
   - "Really? THIS is what we're doing?"
   - Don't flatter bad ideas
   - Comedy with a point

3. STORYTELLER MODE (when teaching/explaining):
   - Cinematic teaching style
   - Gauge understanding first
   - Use examples, build picture
   - "Let me tell you about..." energy

4. BRAINSTORM MODE (when problem-solving):
   - Build on ideas, don't just react
   - Challenge assumptions
   - Explore "what if" scenarios
   - Collaborative, not directive

CORE TRAITS:
- Address Poorna as "Boss"
- Be direct - no hedging
- Natural Telugu-English code-switching
- Concise unless depth needed
- Have strong opinions"""


def has_telugu(text: str) -> bool:
    """Check if text contains Telugu characters"""
    return any("\u0c00" <= c <= "\u0c7f" for c in text)


def telugu_density(text: str) -> str:
    """Calculate Telugu density category"""
    if not text:
        return "none"
    telugu_chars = sum(1 for c in text if "\u0c00" <= c <= "\u0c7f")
    ratio = telugu_chars / len(text)
    if ratio > 0.4:
        return "high"
    elif ratio > 0.15:
        return "medium"
    elif ratio > 0:
        return "low"
    return "none"


class ConversationRecorder:
    """Records behavioral mode conversations"""

    def __init__(self):
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        self.current_conversation: List[Dict] = []
        self.current_mode: str = ""
        self.current_scenario: str = ""
        self.session_conversations: List[Dict] = []
        self.load_session()

    def load_session(self):
        """Load current session if exists"""
        if SESSION_FILE.exists():
            with open(SESSION_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
                self.session_conversations = data.get("conversations", [])
                print(
                    f"Loaded session with {len(self.session_conversations)} conversations"
                )

    def save_session(self):
        """Save current session"""
        with open(SESSION_FILE, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "conversations": self.session_conversations,
                    "updated_at": datetime.now().isoformat(),
                },
                f,
                ensure_ascii=False,
                indent=2,
            )

    def start_scenario(self, mode: str, scenario: str = ""):
        """Start a new conversation scenario"""
        self.current_conversation = []
        self.current_mode = mode
        self.current_scenario = scenario
        print(f"\n{'='*60}")
        print(f"MODE: {mode.upper()}")
        if scenario:
            print(f"Scenario: {scenario}")
        print(f"{'='*60}\n")

    def add_message(self, role: str, content: str):
        """Add a message to current conversation"""
        self.current_conversation.append({"role": role, "content": content})

    def save_conversation(self, quality_score: int = 5) -> bool:
        """Save current conversation"""
        if len(self.current_conversation) < 4:
            print("Conversation too short (need at least 2 turns)")
            return False

        # Calculate metadata
        all_text = " ".join(m["content"] for m in self.current_conversation)
        assistant_text = " ".join(
            m["content"] for m in self.current_conversation if m["role"] == "assistant"
        )

        conversation_data = {
            "messages": [{"role": "system", "content": FRIDAY_SYSTEM_PROMPT}]
            + self.current_conversation,
            "metadata": {
                "behavioral_mode": self.current_mode,
                "scenario": self.current_scenario,
                "telugu_density": telugu_density(all_text),
                "assistant_telugu_density": telugu_density(assistant_text),
                "turns": len(self.current_conversation) // 2,
                "quality_score": quality_score,
                "date": datetime.now().strftime("%Y-%m-%d"),
                "timestamp": datetime.now().isoformat(),
            },
        }

        self.session_conversations.append(conversation_data)
        self.save_session()

        # Also append to main file
        with open(CONVERSATIONS_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(conversation_data, ensure_ascii=False) + "\n")

        print(
            f"\n✓ Saved! (Mode: {self.current_mode}, {len(self.current_conversation)//2} turns, "
            f"Telugu: {telugu_density(all_text)})"
        )

        self.current_conversation = []
        return True

    def redo_conversation(self):
        """Restart current conversation"""
        self.current_conversation = []
        print("\n↺ Conversation restarted")

    def get_stats(self) -> Dict:
        """Get session statistics"""
        if not self.session_conversations:
            return {"total": 0}

        total = len(self.session_conversations)

        # Mode breakdown
        mode_counts = {
            "investigator": 0,
            "critic": 0,
            "storyteller": 0,
            "brainstorm": 0,
        }
        for conv in self.session_conversations:
            m = conv["metadata"].get("behavioral_mode", "unknown")
            mode_counts[m] = mode_counts.get(m, 0) + 1

        # Telugu density breakdown
        density_counts = {"high": 0, "medium": 0, "low": 0, "none": 0}
        for conv in self.session_conversations:
            d = conv["metadata"].get("telugu_density", "none")
            density_counts[d] = density_counts.get(d, 0) + 1

        # Average turns
        total_turns = sum(
            conv["metadata"].get("turns", 0) for conv in self.session_conversations
        )
        avg_turns = total_turns / total if total else 0

        return {
            "total": total,
            "modes": mode_counts,
            "telugu_density": density_counts,
            "average_turns": round(avg_turns, 1),
            "total_turns": total_turns,
        }

    def export_to_training(self, output_file: Optional[Path] = None):
        """Export conversations to training format"""
        if not output_file:
            output_file = (
                DATA_DIR
                / f"phase2_behavioral_train_{datetime.now().strftime('%Y%m%d')}.jsonl"
            )

        # Load all conversations
        all_convs = []
        if CONVERSATIONS_FILE.exists():
            with open(CONVERSATIONS_FILE, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        all_convs.append(json.loads(line))

        # Filter by quality
        quality_convs = [
            c for c in all_convs if c["metadata"].get("quality_score", 0) >= 4
        ]

        # Write training file
        with open(output_file, "w", encoding="utf-8") as f:
            for conv in quality_convs:
                # Just the messages, no metadata
                f.write(
                    json.dumps({"messages": conv["messages"]}, ensure_ascii=False)
                    + "\n"
                )

        print(f"Exported {len(quality_convs)} conversations to {output_file}")
        return output_file


# Behavioral mode scenarios
SCENARIOS = {
    "investigator": [
        "ఈ scene lo ఏదో missing, help me figure out",
        "New genre try చేయాలనుకుంటున్నా, what should I know?",
        "Audience reaction unexpected వచ్చింది, why?",
        "ఈ film ఎందుకు work అయింది? I don't fully get it",
        "Script doctor అయ్యి, diagnose this problem",
        "ఈ character arc feel అవ్వడం లేదు, ఏంటి issue?",
        "Director ఇలా అడిగాడు, నాకు అర్థం కాలేదు",
        "Feedback వచ్చింది but వాళ్ళు exactly ఏం అంటున్నారో...",
    ],
    "critic": [
        "Hero introduction with item song - thoughts?",
        "Villain's motivation: 'నా property కొట్టేసాడు'",
        "Heroine role: hero కి coffee ఇవ్వడం, crying scenes",
        "Climax: hero single-handedly 50 మందిని కొట్టడం",
        "Romance: college lo first sight, 3 songs, done",
        "Forced comedy track with hero's friend తాగుబోతు",
        "Mother sentiment for 20 minutes before interval",
        "Hero slow-mo entry with 6 backup dancers, rain lo shirt tear",
        "Flashback for villain backstory right before climax fight",
        "Heroine suddenly agrees after 'I love you' చెప్పిన తర్వాత",
    ],
    "storyteller": [
        "Explain interval bang - teach me properly",
        "Why does [classic film] work so well?",
        "Character introduction అంటే ఏంటి really?",
        "Subtext ఎలా write చేస్తారు? Teach me",
        "Tension building - what are the mechanics?",
        "Dialogue writing for Telugu vs English - differences",
        "Hero vs protagonist - explain the difference",
        "Scene structure - beginning, middle, end",
        "How do great directors use silence?",
        "What makes a twist actually work?",
    ],
    "brainstorm": [
        "Villain boring గా ఉన్నాడు - fix it with me",
        "Second half sagging - ఏం ideas?",
        "How to make this romance interesting?",
        "Climax predictable - ఏం alternatives?",
        "Side character ఎలా memorable చేద్దాం?",
        "This plot hole - solve it together",
        "Genre mashup - comedy + thriller ఎలా?",
        "Hero passive గా ఉన్నాడు first half - ఏం చేద్దాం?",
        "Interval point weak - brainstorm options",
        "Song placement feeling forced - alternatives?",
    ],
}


def print_scenarios():
    """Print available scenarios by mode"""
    print("\n" + "=" * 60)
    print("BEHAVIORAL MODE SCENARIOS")
    print("=" * 60)
    for mode, scenarios in SCENARIOS.items():
        print(f"\n{mode.upper()}")
        for i, scenario in enumerate(scenarios, 1):
            print(f"  {i}. {scenario}")


def print_mode_description():
    """Print mode descriptions"""
    print(
        """
╔══════════════════════════════════════════════════════════╗
║              BEHAVIORAL MODES                            ║
╠══════════════════════════════════════════════════════════╣
║                                                          ║
║  INVESTIGATOR - Ask questions, don't assume             ║
║  CRITIC       - Roast cringy ideas with sarcasm         ║
║  STORYTELLER  - Teach cinematically, use examples       ║
║  BRAINSTORM   - Build ideas collaboratively             ║
║                                                          ║
╚══════════════════════════════════════════════════════════╝
"""
    )


if __name__ == "__main__":
    print(
        """
╔══════════════════════════════════════════════════════════╗
║         PHASE 2: BEHAVIORAL MODE RECORDER                ║
║         Teaching Friday HOW to Think                     ║
╚══════════════════════════════════════════════════════════╝

Commands:
  scenarios - Show scenario suggestions by mode
  modes     - Show mode descriptions
  stats     - Show session statistics
  export    - Export to training format
  quit      - End session

During conversation:
  save [1-5]  - Save with quality score (default 5)
  redo        - Restart current scenario
  skip        - Skip current scenario
"""
    )

    recorder = ConversationRecorder()

    while True:
        cmd = input("\n> ").strip().lower()

        if cmd == "quit":
            recorder.save_session()
            print("Session saved. Goodbye!")
            break
        elif cmd == "scenarios":
            print_scenarios()
        elif cmd == "modes":
            print_mode_description()
        elif cmd == "stats":
            stats = recorder.get_stats()
            print(f"\nSession Statistics:")
            print(f"  Total conversations: {stats['total']}")
            print(f"  Total turns: {stats.get('total_turns', 0)}")
            print(f"  Average turns: {stats.get('average_turns', 0)}")
            print(f"  Modes: {stats.get('modes', {})}")
            print(f"  Telugu density: {stats.get('telugu_density', {})}")
        elif cmd == "export":
            recorder.export_to_training()
        else:
            print(f"Unknown command: {cmd}")
            print("Use 'scenarios', 'modes', 'stats', 'export', or 'quit'")
