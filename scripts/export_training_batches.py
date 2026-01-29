#!/usr/bin/env python3
"""
Export Training Batches for Iterative Model Evolution
======================================================

Creates incremental training batches to test model evolution:
- Batch 1: Persona + Relationships (16 examples) - Core identity
- Batch 2: + Humor + Food (32 examples) - Personality
- Batch 3: + Film + Work (48 examples) - Domain knowledge
- Batch 4: + Tech + Telugu Culture (64 examples) - Full persona

Usage:
    python scripts/export_training_batches.py
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data" / "interviews"
RAW_DIR = DATA_DIR / "raw"
OUTPUT_DIR = REPO_ROOT / "data" / "instructions" / "interview_batches"

# System prompt for Friday
FRIDAY_SYSTEM_PROMPT = """You are Friday, Poorna's AI assistant. You blend Telugu and English naturally, addressing him as 'Boss'. Be concise, helpful, and direct. No flattery or excessive formality."""

# Batch definitions - cumulative
BATCHES = {
    "batch1_identity": ["persona", "relationships"],
    "batch2_personality": ["persona", "relationships", "humor", "food"],
    "batch3_domains": ["persona", "relationships", "humor", "food", "film", "work"],
    "batch4_full": [
        "persona",
        "relationships",
        "humor",
        "food",
        "film",
        "work",
        "tech",
        "telugu_culture",
    ],
}


def load_all_sessions() -> Dict[str, Dict]:
    """Load all interview sessions by topic"""
    sessions_by_topic = {}

    for filepath in RAW_DIR.glob("*.json"):
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                session = json.load(f)
                topic = session.get("topic", "unknown")
                sessions_by_topic[topic] = session
        except json.JSONDecodeError as e:
            print(f"Warning: Could not parse {filepath}: {e}")

    return sessions_by_topic


def convert_session_to_examples(session: Dict) -> List[Dict]:
    """Convert a session to ChatML training examples"""
    examples = []
    topic = session.get("topic", "general")
    session_id = session.get("session_id", "unknown")

    for exchange in session.get("exchanges", []):
        question = exchange.get("question", "")
        response = exchange.get("response", "")

        if not question or not response:
            continue

        # Skip very short responses
        if len(response.strip()) < 10 or len(response.split()) < 3:
            continue

        example = {
            "messages": [
                {"role": "system", "content": FRIDAY_SYSTEM_PROMPT},
                {"role": "user", "content": question},
                {"role": "assistant", "content": response},
            ],
            "metadata": {
                "source": "interview",
                "session_id": session_id,
                "topic": topic,
                "language": exchange.get("language", "en"),
                "turn": exchange.get("turn", 0),
                "themes": exchange.get("themes", []),
            },
        }
        examples.append(example)

    return examples


def export_batch(
    batch_name: str, topics: List[str], sessions_by_topic: Dict[str, Dict]
) -> int:
    """Export a single batch"""
    examples = []

    for topic in topics:
        if topic in sessions_by_topic:
            topic_examples = convert_session_to_examples(sessions_by_topic[topic])
            examples.extend(topic_examples)
            print(f"  {topic}: {len(topic_examples)} examples")

    # Export
    output_path = OUTPUT_DIR / f"{batch_name}.jsonl"
    with open(output_path, "w", encoding="utf-8") as f:
        for example in examples:
            f.write(json.dumps(example, ensure_ascii=False) + "\n")

    # Stats
    stats = {
        "batch_name": batch_name,
        "topics": topics,
        "total_examples": len(examples),
        "language_distribution": {},
        "created_at": datetime.now().isoformat(),
    }

    for example in examples:
        lang = example["metadata"].get("language", "unknown")
        stats["language_distribution"][lang] = (
            stats["language_distribution"].get(lang, 0) + 1
        )

    stats_path = OUTPUT_DIR / f"{batch_name}.stats.json"
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)

    return len(examples)


def main():
    """Main function"""
    print("=" * 60)
    print("INTERVIEW TRAINING BATCH EXPORT")
    print("=" * 60)

    # Ensure output directory exists
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load all sessions
    sessions_by_topic = load_all_sessions()
    print(f"\nLoaded topics: {list(sessions_by_topic.keys())}")

    # Export each batch
    print("\n" + "-" * 60)
    for batch_name, topics in BATCHES.items():
        print(f"\n{batch_name.upper()} ({len(topics)} topics):")
        count = export_batch(batch_name, topics, sessions_by_topic)
        print(f"  TOTAL: {count} examples → {batch_name}.jsonl")

    print("\n" + "=" * 60)
    print("EXPORT COMPLETE")
    print("=" * 60)
    print(f"\nFiles saved to: {OUTPUT_DIR}")
    print("\nNext steps:")
    print("1. Train on batch1_identity.jsonl → Test core persona")
    print("2. Train on batch2_personality.jsonl → Test humor/food knowledge")
    print("3. Train on batch3_domains.jsonl → Test film/work domains")
    print("4. Train on batch4_full.jsonl → Full Friday persona")


if __name__ == "__main__":
    main()
