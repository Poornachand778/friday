#!/usr/bin/env python3
"""
Model Evolution Evaluation Script
=================================

Tests Friday model after each training batch with consistent prompts.
Saves results for comparison across training iterations.

Usage:
    python scripts/eval_model_evolution.py --batch batch1_identity
    python scripts/eval_model_evolution.py --compare batch1 batch2
"""

import json
from datetime import datetime
from pathlib import Path
from typing import List, Dict

REPO_ROOT = Path(__file__).resolve().parents[1]
EVAL_DIR = REPO_ROOT / "eval" / "model_evolution"

# Evaluation prompts - same prompts used after each batch training
EVAL_PROMPTS = [
    # Core Identity (should improve after batch1)
    {
        "id": "identity_1",
        "category": "persona",
        "prompt": "What do you believe that most people disagree with?",
        "expected_themes": ["life not serious", "culture", "free will", "discipline"],
    },
    {
        "id": "identity_2",
        "category": "relationships",
        "prompt": "What does true friendship mean to you?",
        "expected_themes": [
            "no expectations",
            "ruthless truth",
            "no boundaries",
            "support",
        ],
    },
    {
        "id": "identity_3",
        "category": "persona",
        "prompt": "Nee sister gurinchi cheppu, meeru kalisi production company start chestunnara?",
        "expected_themes": [
            "equal stakeholder",
            "trust",
            "delhi university",
            "filmmaker",
        ],
    },
    # Personality (should improve after batch2)
    {
        "id": "personality_1",
        "category": "humor",
        "prompt": "Describe your humor style in one line",
        "expected_themes": ["chandler", "sarcasm", "comfortable", "killer line"],
    },
    {
        "id": "personality_2",
        "category": "food",
        "prompt": "What's your comfort food?",
        "expected_themes": ["nut butter", "sourdough", "homemade", "healthy"],
    },
    {
        "id": "personality_3",
        "category": "humor",
        "prompt": "Give me an example of a Friday joke",
        "expected_themes": ["sarcastic appreciation", "calling out", "playful"],
    },
    # Domain Knowledge (should improve after batch3)
    {
        "id": "domain_1",
        "category": "film",
        "prompt": "How do you approach writing a story? What comes first?",
        "expected_themes": [
            "inspiring scene",
            "research",
            "google notes",
            "climax important",
        ],
    },
    {
        "id": "domain_2",
        "category": "work",
        "prompt": "What's your approach when you're stuck on a project?",
        "expected_themes": ["meditation", "gym", "telugu songs", "one step at a time"],
    },
    {
        "id": "domain_3",
        "category": "film",
        "prompt": "What kind of films does your production company want to make?",
        "expected_themes": ["emotional romantic comedy", "regional authentic telugu"],
    },
    # Technical + Cultural (should improve after batch4)
    {
        "id": "tech_1",
        "category": "tech",
        "prompt": "What's your opinion on AI safety and freedom?",
        "expected_themes": ["no freedom", "visibility", "black box", "human in loop"],
    },
    {
        "id": "tech_2",
        "category": "tech",
        "prompt": "How do you learn new technology?",
        "expected_themes": [
            "overview first",
            "mathematical intuition",
            "why it exists",
            "projects",
        ],
    },
    {
        "id": "culture_1",
        "category": "telugu_culture",
        "prompt": "Telugu ante niku em meaning?",
        "expected_themes": [
            "singing while speaking",
            "athidhi devo bhava",
            "cultural acceptance",
        ],
    },
    # Code-switching test (should be natural throughout)
    {
        "id": "codeswitching_1",
        "category": "mixed",
        "prompt": "Nee life philosophy enti?",
        "expected_themes": [
            "life not serious",
            "mental peace",
            "proper food",
            "keep trying",
        ],
    },
    {
        "id": "codeswitching_2",
        "category": "mixed",
        "prompt": "Friday ni ela design chestunnav?",
        "expected_themes": ["proxy when absent", "suggestions", "not write own code"],
    },
]


def save_eval_template():
    """Save evaluation prompts as template for manual testing"""
    EVAL_DIR.mkdir(parents=True, exist_ok=True)

    template_path = EVAL_DIR / "eval_prompts.json"
    with open(template_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "prompts": EVAL_PROMPTS,
                "instructions": {
                    "usage": "Run each prompt through the model after training each batch",
                    "scoring": {
                        "0": "No relevant themes captured",
                        "1": "Some themes but wrong style",
                        "2": "Good themes, partial style match",
                        "3": "Strong theme coverage, good style",
                        "4": "Perfect - sounds exactly like Poorna",
                    },
                    "code_switching_check": "Does response naturally blend Telugu and English?",
                },
            },
            f,
            indent=2,
            ensure_ascii=False,
        )

    print(f"Saved evaluation template to: {template_path}")
    return template_path


def create_eval_result_template(batch_name: str) -> Path:
    """Create empty result template for a batch"""
    EVAL_DIR.mkdir(parents=True, exist_ok=True)

    results = {
        "batch": batch_name,
        "evaluated_at": datetime.now().isoformat(),
        "model_info": {
            "base": "meta-llama/Meta-Llama-3.1-8B-Instruct",
            "adapter": f"friday_lora_{batch_name}",
        },
        "results": [],
    }

    for prompt in EVAL_PROMPTS:
        results["results"].append(
            {
                "id": prompt["id"],
                "category": prompt["category"],
                "prompt": prompt["prompt"],
                "expected_themes": prompt["expected_themes"],
                "model_response": "",  # Fill in after running
                "score": 0,  # 0-4 scale
                "code_switching": False,  # True if natural Te-En blend
                "notes": "",
            }
        )

    result_path = (
        EVAL_DIR / f"eval_{batch_name}_{datetime.now().strftime('%Y%m%d')}.json"
    )
    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"Created result template: {result_path}")
    return result_path


def print_prompts_for_testing():
    """Print prompts in a format ready for manual testing"""
    print("\n" + "=" * 70)
    print("EVALUATION PROMPTS FOR MANUAL TESTING")
    print("=" * 70)

    for i, prompt in enumerate(EVAL_PROMPTS, 1):
        print(
            f"\n[{i}/{len(EVAL_PROMPTS)}] {prompt['category'].upper()} - {prompt['id']}"
        )
        print("-" * 50)
        print(f"PROMPT: {prompt['prompt']}")
        print(f"EXPECTED: {', '.join(prompt['expected_themes'])}")
        print()


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Model Evolution Evaluation")
    parser.add_argument(
        "--save-template", action="store_true", help="Save eval prompts template"
    )
    parser.add_argument(
        "--create-result", metavar="BATCH", help="Create result template for batch"
    )
    parser.add_argument(
        "--print-prompts", action="store_true", help="Print prompts for testing"
    )
    args = parser.parse_args()

    if args.save_template:
        save_eval_template()
    elif args.create_result:
        create_eval_result_template(args.create_result)
    elif args.print_prompts:
        print_prompts_for_testing()
    else:
        # Default: save template and print prompts
        save_eval_template()
        print_prompts_for_testing()


if __name__ == "__main__":
    main()
