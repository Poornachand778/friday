#!/usr/bin/env python3
"""
Interview Collector for Friday AI Training Data
================================================

Uses Claude to interview you about various topics to generate high-quality
training data that captures your personality, opinions, and communication style.

Usage:
    python scripts/interview_collector.py                    # Interactive mode
    python scripts/interview_collector.py --topic persona    # Specific topic
    python scripts/interview_collector.py --resume           # Resume last session
    python scripts/interview_collector.py --list             # List all sessions
    python scripts/interview_collector.py --export           # Export to ChatML

Cost: ~$0.01-0.02 per interview session (using Claude Haiku)
"""

import argparse
import json
import os
import sys
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

# Add repo root to path
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

try:
    import anthropic
except ImportError:
    print("ERROR: anthropic package not installed")
    print("Run: pip install anthropic")
    sys.exit(1)

try:
    import yaml
except ImportError:
    print("ERROR: pyyaml package not installed")
    print("Run: pip install pyyaml")
    sys.exit(1)


# Paths
DATA_DIR = REPO_ROOT / "data" / "interviews"
RAW_DIR = DATA_DIR / "raw"
REVIEWED_DIR = DATA_DIR / "reviewed"
EXPORTED_DIR = DATA_DIR / "exported"
TOPICS_FILE = REPO_ROOT / "scripts" / "interview_topics.yaml"


# Interview system prompt
INTERVIEWER_PROMPT = """You are an expert interviewer helping to collect training data for an AI assistant called Friday.

Your goal is to ask thoughtful, probing questions to understand the user's:
- Personality and communication style
- Opinions and beliefs
- How they think about problems
- Their preferences and values
- Their knowledge in specific domains

Guidelines:
1. Ask ONE question at a time
2. Follow up on interesting answers - dig deeper
3. If the user responds in Telugu or Telugu-English mix, respond naturally in the same style
4. Keep questions conversational, not interrogative
5. Show genuine curiosity about their perspective
6. After 8-10 exchanges, you can wrap up or transition to a new sub-topic
7. Avoid yes/no questions - ask open-ended ones
8. If they give short answers, probe for more detail

Current topic: {topic}
Sub-topic focus: {subtopic}

Remember: The goal is to capture HOW they think and express themselves, not just WHAT they think."""


class InterviewSession:
    """Manages a single interview session"""

    def __init__(
        self,
        session_id: Optional[str] = None,
        topic: str = "general",
        subtopic: str = "open",
    ):
        self.session_id = (
            session_id
            or f"{topic}_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"
        )
        self.topic = topic
        self.subtopic = subtopic
        self.exchanges: List[Dict[str, str]] = []
        self.created_at = datetime.now().isoformat()
        self.updated_at = self.created_at
        self.status = "in_progress"  # in_progress, completed, reviewed
        self.language_detected: List[str] = []
        self.notes = ""

    def add_exchange(
        self, user_message: str, assistant_response: str, language: str = "en"
    ):
        """Add a Q&A exchange"""
        self.exchanges.append(
            {
                "turn": len(self.exchanges) + 1,
                "user": user_message,
                "assistant": assistant_response,
                "language": language,
                "timestamp": datetime.now().isoformat(),
            }
        )
        self.updated_at = datetime.now().isoformat()
        if language not in self.language_detected:
            self.language_detected.append(language)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for saving"""
        return {
            "session_id": self.session_id,
            "topic": self.topic,
            "subtopic": self.subtopic,
            "exchanges": self.exchanges,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "status": self.status,
            "language_detected": self.language_detected,
            "notes": self.notes,
            "exchange_count": len(self.exchanges),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "InterviewSession":
        """Load from dictionary"""
        session = cls(
            session_id=data["session_id"],
            topic=data["topic"],
            subtopic=data.get("subtopic", "open"),
        )
        session.exchanges = data.get("exchanges", [])
        session.created_at = data.get("created_at", session.created_at)
        session.updated_at = data.get("updated_at", session.updated_at)
        session.status = data.get("status", "in_progress")
        session.language_detected = data.get("language_detected", [])
        session.notes = data.get("notes", "")
        return session

    def save(self):
        """Save session to file"""
        filepath = RAW_DIR / f"{self.session_id}.json"
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=2)
        return filepath

    @classmethod
    def load(cls, session_id: str) -> Optional["InterviewSession"]:
        """Load session from file"""
        filepath = RAW_DIR / f"{session_id}.json"
        if not filepath.exists():
            return None
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls.from_dict(data)


class InterviewCollector:
    """Main interview collector"""

    def __init__(self):
        self.client = anthropic.Anthropic()
        self.topics = self._load_topics()
        self.current_session: Optional[InterviewSession] = None

    def _load_topics(self) -> Dict[str, Any]:
        """Load topic definitions from YAML"""
        if not TOPICS_FILE.exists():
            # Return default topics if file doesn't exist
            return {
                "persona": {
                    "name": "Personality & Beliefs",
                    "subtopics": [
                        "life_philosophy",
                        "relationships",
                        "values",
                        "humor",
                    ],
                    "starter": "Let's explore your personality and what makes you tick.",
                },
                "film": {
                    "name": "Film & Storytelling",
                    "subtopics": [
                        "narrative",
                        "characters",
                        "telugu_cinema",
                        "direction",
                    ],
                    "starter": "Let's talk about your approach to filmmaking and storytelling.",
                },
                "technical": {
                    "name": "Technical & Work",
                    "subtopics": [
                        "workflow",
                        "tools",
                        "problem_solving",
                        "preferences",
                    ],
                    "starter": "Let's discuss how you approach technical work and problem-solving.",
                },
                "casual": {
                    "name": "Casual & Daily Life",
                    "subtopics": ["food", "travel", "hobbies", "random_opinions"],
                    "starter": "Let's have a casual chat about everyday things.",
                },
                "telugu": {
                    "name": "Telugu Conversations",
                    "subtopics": ["culture", "language", "traditions", "opinions"],
                    "starter": "మీతో తెలుగులో మాట్లాడదాం. మీ అభిప్రాయాలు తెలుసుకుందాం.",
                },
                "emotional": {
                    "name": "Emotional Range",
                    "subtopics": ["excited", "thoughtful", "frustrated", "playful"],
                    "starter": "Let's explore different emotional perspectives on things.",
                },
            }

        with open(TOPICS_FILE, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)

    def _detect_language(self, text: str) -> str:
        """Simple language detection"""
        # Telugu Unicode range
        telugu_chars = sum(1 for c in text if "\u0C00" <= c <= "\u0C7F")
        total_chars = len(text.replace(" ", ""))

        if total_chars == 0:
            return "en"

        telugu_ratio = telugu_chars / total_chars

        if telugu_ratio > 0.5:
            return "te"
        elif telugu_ratio > 0.1:
            return "te-en"  # Code-switched
        else:
            return "en"

    def _get_claude_response(
        self, messages: List[Dict], topic: str, subtopic: str
    ) -> str:
        """Get response from Claude"""
        system_prompt = INTERVIEWER_PROMPT.format(topic=topic, subtopic=subtopic)

        response = self.client.messages.create(
            model="claude-3-5-haiku-20241022",  # Cost-effective for interviews
            max_tokens=500,
            system=system_prompt,
            messages=messages,
        )

        return response.content[0].text

    def start_session(
        self, topic: str = "general", subtopic: str = "open"
    ) -> InterviewSession:
        """Start a new interview session"""
        self.current_session = InterviewSession(topic=topic, subtopic=subtopic)
        return self.current_session

    def resume_session(self, session_id: str) -> Optional[InterviewSession]:
        """Resume an existing session"""
        session = InterviewSession.load(session_id)
        if session:
            self.current_session = session
        return session

    def get_interviewer_question(self) -> str:
        """Get the next question from the interviewer"""
        if not self.current_session:
            raise ValueError("No active session")

        # Build message history
        messages = []

        # Add topic context as first user message
        topic_info = self.topics.get(self.current_session.topic, {})
        starter = topic_info.get(
            "starter", f"Let's discuss {self.current_session.topic}."
        )

        if not self.current_session.exchanges:
            # First question - provide context
            messages.append(
                {
                    "role": "user",
                    "content": f"Start the interview. Topic: {self.current_session.topic}, Focus: {self.current_session.subtopic}. Begin with an engaging opening question.",
                }
            )
        else:
            # Continue conversation
            for exchange in self.current_session.exchanges:
                messages.append({"role": "assistant", "content": exchange["assistant"]})
                messages.append({"role": "user", "content": exchange["user"]})

            # Ask for next question
            messages.append(
                {
                    "role": "user",
                    "content": "[Continue the interview with a follow-up question based on my previous answer]",
                }
            )

        return self._get_claude_response(
            messages, self.current_session.topic, self.current_session.subtopic
        )

    def process_response(self, user_response: str, interviewer_question: str) -> None:
        """Process user's response and save"""
        if not self.current_session:
            raise ValueError("No active session")

        language = self._detect_language(user_response)
        self.current_session.add_exchange(user_response, interviewer_question, language)
        self.current_session.save()

    def list_sessions(self) -> List[Dict]:
        """List all sessions"""
        sessions = []
        for filepath in RAW_DIR.glob("*.json"):
            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)
                sessions.append(
                    {
                        "session_id": data["session_id"],
                        "topic": data["topic"],
                        "exchanges": data.get(
                            "exchange_count", len(data.get("exchanges", []))
                        ),
                        "status": data.get("status", "unknown"),
                        "updated_at": data.get("updated_at", "unknown"),
                    }
                )
        return sorted(sessions, key=lambda x: x["updated_at"], reverse=True)

    def export_to_chatml(self, output_file: Optional[Path] = None) -> Path:
        """Export all reviewed sessions to ChatML format"""
        output_file = (
            output_file
            or EXPORTED_DIR
            / f"interview_train_{datetime.now().strftime('%Y%m%d')}.jsonl"
        )

        examples = []

        # Process raw and reviewed directories
        for directory in [RAW_DIR, REVIEWED_DIR]:
            for filepath in directory.glob("*.json"):
                with open(filepath, "r", encoding="utf-8") as f:
                    session = json.load(f)

                # Convert each exchange to ChatML format
                for exchange in session.get("exchanges", []):
                    example = {
                        "messages": [
                            {
                                "role": "system",
                                "content": "You are Friday, Poorna's AI assistant. Address him as 'Boss'. Blend Telugu and English naturally. Be concise, helpful, and direct.",
                            },
                            {
                                "role": "user",
                                "content": exchange[
                                    "assistant"
                                ],  # Interviewer's question becomes user input
                            },
                            {
                                "role": "assistant",
                                "content": exchange[
                                    "user"
                                ],  # User's response becomes Friday's response pattern
                            },
                        ],
                        "metadata": {
                            "source": "interview",
                            "session_id": session["session_id"],
                            "topic": session["topic"],
                            "language": exchange.get("language", "en"),
                            "turn": exchange.get("turn", 0),
                        },
                    }
                    examples.append(example)

        # Write JSONL
        with open(output_file, "w", encoding="utf-8") as f:
            for example in examples:
                f.write(json.dumps(example, ensure_ascii=False) + "\n")

        return output_file


def print_header():
    """Print welcome header"""
    print("\n" + "=" * 60)
    print("  Friday AI - Interview Collector")
    print("  Generating training data through conversation")
    print("=" * 60 + "\n")


def print_topics(collector: InterviewCollector):
    """Print available topics"""
    print("\nAvailable topics:")
    for key, topic in collector.topics.items():
        name = topic.get("name", key)
        subtopics = topic.get("subtopics", [])
        print(f"  {key}: {name}")
        if subtopics:
            print(f"       Subtopics: {', '.join(subtopics)}")
    print()


def interactive_session(collector: InterviewCollector, topic: str, subtopic: str):
    """Run an interactive interview session"""
    session = collector.start_session(topic, subtopic)

    print(f"\nStarting interview session: {session.session_id}")
    print(f"Topic: {topic} | Subtopic: {subtopic}")
    print("\nCommands: 'quit' to end, 'skip' to skip question, 'save' to save & exit")
    print("-" * 60 + "\n")

    while True:
        try:
            # Get interviewer's question
            print("Thinking...", end="\r")
            question = collector.get_interviewer_question()
            print(" " * 20, end="\r")  # Clear "Thinking..."

            print(f"\nInterviewer: {question}\n")

            # Get user response
            print("You: ", end="")
            user_input = input().strip()

            if user_input.lower() == "quit":
                session.status = "completed"
                session.save()
                print(f"\nSession saved: {session.session_id}")
                print(f"Total exchanges: {len(session.exchanges)}")
                break

            if user_input.lower() == "skip":
                print("(Skipped)\n")
                continue

            if user_input.lower() == "save":
                session.save()
                print(f"\nSession saved: {session.session_id}")
                print(f"Total exchanges: {len(session.exchanges)}")
                break

            if not user_input:
                print("(Empty response, try again)\n")
                continue

            # Save the exchange
            collector.process_response(user_input, question)
            print(
                f"  [Saved: exchange #{len(session.exchanges)}, lang={session.language_detected[-1] if session.language_detected else 'en'}]"
            )

        except KeyboardInterrupt:
            print("\n\nInterrupted. Saving session...")
            session.save()
            print(f"Session saved: {session.session_id}")
            break
        except Exception as e:
            print(f"\nError: {e}")
            print("Saving session and exiting...")
            session.save()
            break


def main():
    parser = argparse.ArgumentParser(description="Interview Collector for Friday AI")
    parser.add_argument(
        "--topic",
        "-t",
        help="Interview topic (persona, film, technical, casual, telugu, emotional)",
    )
    parser.add_argument("--subtopic", "-s", default="open", help="Subtopic focus")
    parser.add_argument("--resume", "-r", help="Resume session by ID")
    parser.add_argument("--list", "-l", action="store_true", help="List all sessions")
    parser.add_argument(
        "--export", "-e", action="store_true", help="Export to ChatML format"
    )
    parser.add_argument("--topics", action="store_true", help="Show available topics")
    args = parser.parse_args()

    # Check API key
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("ERROR: ANTHROPIC_API_KEY environment variable not set")
        sys.exit(1)

    collector = InterviewCollector()

    print_header()

    if args.list:
        sessions = collector.list_sessions()
        if not sessions:
            print("No sessions found.")
        else:
            print(f"Found {len(sessions)} sessions:\n")
            for s in sessions[:20]:  # Show last 20
                status_icon = "✓" if s["status"] == "completed" else "○"
                print(f"  {status_icon} {s['session_id']}")
                print(
                    f"      Topic: {s['topic']} | Exchanges: {s['exchanges']} | {s['status']}"
                )
        return

    if args.export:
        output = collector.export_to_chatml()
        print(f"Exported to: {output}")
        # Count examples
        with open(output, "r") as f:
            count = sum(1 for _ in f)
        print(f"Total examples: {count}")
        return

    if args.topics:
        print_topics(collector)
        return

    if args.resume:
        session = collector.resume_session(args.resume)
        if not session:
            print(f"Session not found: {args.resume}")
            sys.exit(1)
        print(f"Resuming session: {session.session_id}")
        print(f"Existing exchanges: {len(session.exchanges)}")
        interactive_session(collector, session.topic, session.subtopic)
        return

    # Interactive topic selection if not specified
    topic = args.topic
    if not topic:
        print_topics(collector)
        print("Enter topic (or press Enter for 'persona'): ", end="")
        topic = input().strip() or "persona"

    if topic not in collector.topics:
        print(f"Unknown topic: {topic}")
        print_topics(collector)
        sys.exit(1)

    subtopic = args.subtopic
    if subtopic == "open" and collector.topics[topic].get("subtopics"):
        subtopics = collector.topics[topic]["subtopics"]
        print(f"\nSubtopics for {topic}: {', '.join(subtopics)}")
        print("Enter subtopic (or press Enter for 'open'): ", end="")
        subtopic = input().strip() or "open"

    interactive_session(collector, topic, subtopic)


if __name__ == "__main__":
    main()
