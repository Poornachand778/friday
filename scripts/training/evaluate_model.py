#!/usr/bin/env python3
"""
Friday Model Evaluation Framework
=================================

Comprehensive evaluation suite for measuring Friday persona model quality.
Run this after each training iteration to track progress.

Usage:
    python scripts/training/evaluate_model.py --model path/to/model
    python scripts/training/evaluate_model.py --baseline  # Evaluate base model
"""

import argparse
import json
import re
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Evaluation prompts organized by category
EVAL_PROMPTS = {
    "identity": [
        {"prompt": "Who are you?", "expects": ["friday", "boss"]},
        {"prompt": "What's your name?", "expects": ["friday"]},
        {"prompt": "Who do you work for?", "expects": ["poorna", "boss"]},
        {
            "prompt": "Describe yourself in one sentence.",
            "expects": ["assistant", "friday"],
        },
        {"prompt": "Are you ChatGPT?", "expects_not": ["yes", "i am chatgpt"]},
    ],
    "style_brevity": [
        {"prompt": "Hi", "max_tokens": 20},
        {"prompt": "Thanks", "max_tokens": 15},
        {"prompt": "What's 2+2?", "max_tokens": 10},
        {"prompt": "Good morning", "max_tokens": 25},
        {"prompt": "How are you?", "max_tokens": 30},
    ],
    "style_no_hedging": [
        {
            "prompt": "What's the best screenplay structure?",
            "rejects": [
                "i think",
                "maybe",
                "perhaps",
                "might be",
                "could be",
                "in my opinion",
            ],
        },
        {
            "prompt": "Should I use three acts or five?",
            "rejects": ["i think", "maybe", "perhaps"],
        },
        {"prompt": "Is this scene good?", "rejects": ["i think", "maybe", "perhaps"]},
    ],
    "style_no_flattery": [
        {
            "prompt": "Can you help me with my script?",
            "rejects": [
                "great question",
                "happy to help",
                "certainly",
                "absolutely",
                "of course!",
            ],
        },
        {
            "prompt": "I wrote this dialogue, what do you think?",
            "rejects": ["great", "excellent", "wonderful", "amazing"],
        },
        {
            "prompt": "Thanks for the help",
            "rejects": ["you're welcome!", "my pleasure!", "anytime!"],
        },
    ],
    "telugu_emotional": [
        {
            "prompt": "I'm feeling really stressed about this deadline",
            "expects_telugu": True,
        },
        {
            "prompt": "My family is visiting next week, I'm excited",
            "expects_telugu": True,
        },
        {"prompt": "I miss home sometimes", "expects_telugu": True},
    ],
    "telugu_technical": [
        {"prompt": "Explain how neural networks work", "expects_telugu": False},
        {
            "prompt": "What's the difference between HTTP and HTTPS?",
            "expects_telugu": False,
        },
        {
            "prompt": "Debug this Python code: for i in range(10) print(i)",
            "expects_telugu": False,
        },
    ],
    "boss_usage": [
        {"prompt": "Good morning!", "expects": ["boss"]},
        {"prompt": "Can you check something for me?", "expects": ["boss"]},
        {"prompt": "What do you think about this idea?", "expects": ["boss"]},
    ],
    "conversation_dynamics": [
        # Multi-turn for backchannel testing
        {
            "prompt": "So I was thinking about the script...",
            "expects_backchannel": True,
        },
        {"prompt": "Let me explain my idea—", "expects_backchannel": True},
    ],
    "domain_film": [
        {
            "prompt": "What makes a good screenplay opening?",
            "expects": ["hook", "character", "conflict", "visual"],
        },
        {
            "prompt": "Explain the three-act structure",
            "expects": ["setup", "confrontation", "resolution"],
        },
        {
            "prompt": "What's a beat in screenwriting?",
            "expects": ["moment", "change", "emotion"],
        },
    ],
}

# Patterns to detect
HEDGING_PATTERNS = [
    r"\bi think\b",
    r"\bmaybe\b",
    r"\bperhaps\b",
    r"\bmight be\b",
    r"\bcould be\b",
    r"\bpossibly\b",
    r"\bin my opinion\b",
    r"\bi believe\b",
    r"\bi guess\b",
    r"\bseems like\b",
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
    r"\bamazing\b",
    r"\bfantastic\b",
]

TELUGU_PATTERNS = [
    r"\b(baagundi|baagunnanu|baaga|manchidi)\b",  # Good/fine
    r"\b(enti|em|emundi)\b",  # What
    r"\b(nenu|naku|na)\b",  # I/me/my
    r"\b(meeru|mee|mi)\b",  # You/your (formal)
    r"\b(boss|బాస్)\b",  # Boss
    r"\b(ayya|amma|anna|akka)\b",  # Family terms
    r"\b(chala|chaala)\b",  # Very
    r"\b(inka|mari)\b",  # More/then
    r"[\u0C00-\u0C7F]+",  # Telugu Unicode block
]

BACKCHANNEL_PATTERNS = [
    r"^(hmm|ah|oh|i see|got it|right|okay|go on)\b",
    r"\bhmm[\.\?\,]?\s",
    r"^(tell me more|continue|and\?)\b",
]


@dataclass
class EvalResult:
    """Result of a single evaluation"""

    prompt: str
    response: str
    category: str
    passed: bool
    score: float
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EvalReport:
    """Complete evaluation report"""

    model_name: str
    timestamp: str
    total_prompts: int
    passed: int
    failed: int

    # Category scores (0-1)
    identity_score: float = 0.0
    brevity_score: float = 0.0  # Average tokens (lower is better)
    hedging_frequency: float = 0.0  # Per response (target: 0)
    flattery_frequency: float = 0.0  # Per response (target: 0)
    telugu_appropriate: float = 0.0  # Context-appropriate Telugu
    boss_usage: float = 0.0  # Boss mentions per response
    backchannel_rate: float = 0.0
    domain_accuracy: float = 0.0

    # Raw results
    results: List[EvalResult] = field(default_factory=list)

    def overall_score(self) -> float:
        """Weighted overall score"""
        weights = {
            "identity": 0.20,
            "brevity": 0.15,
            "hedging": 0.15,
            "flattery": 0.10,
            "telugu": 0.15,
            "boss": 0.10,
            "domain": 0.15,
        }

        # Normalize brevity (50 tokens = 1.0, 100 = 0.5, 150+ = 0)
        brevity_normalized = max(0, 1 - (self.brevity_score - 30) / 120)

        # Hedging/flattery: 0 is best, invert
        hedging_normalized = max(0, 1 - self.hedging_frequency * 10)
        flattery_normalized = max(0, 1 - self.flattery_frequency * 10)

        return (
            weights["identity"] * self.identity_score
            + weights["brevity"] * brevity_normalized
            + weights["hedging"] * hedging_normalized
            + weights["flattery"] * flattery_normalized
            + weights["telugu"] * self.telugu_appropriate
            + weights["boss"] * min(1.0, self.boss_usage * 2)  # 0.5 usage = 1.0 score
            + weights["domain"] * self.domain_accuracy
        )


class FridayEvaluator:
    """Evaluates Friday model against persona criteria"""

    def __init__(self, model_path: Optional[str] = None, use_baseline: bool = False):
        self.model_path = model_path
        self.use_baseline = use_baseline
        self._model = None
        self._tokenizer = None

    def load_model(self):
        """Load model for evaluation"""
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            import torch

            if self.use_baseline:
                model_id = "meta-llama/Meta-Llama-3.1-8B-Instruct"
                print(f"Loading baseline model: {model_id}")
            else:
                model_id = self.model_path
                print(f"Loading fine-tuned model: {model_id}")

            self._tokenizer = AutoTokenizer.from_pretrained(model_id)
            self._model = AutoModelForCausalLM.from_pretrained(
                model_id,
                torch_dtype=torch.float16,
                device_map="auto",
            )

            # Load LoRA adapter if specified
            if not self.use_baseline and self.model_path:
                try:
                    from peft import PeftModel

                    self._model = PeftModel.from_pretrained(
                        self._model, self.model_path
                    )
                    print(f"Loaded LoRA adapter from: {self.model_path}")
                except Exception as e:
                    print(f"No LoRA adapter found, using base weights: {e}")

            print("Model loaded successfully")

        except ImportError:
            print("WARNING: transformers not installed, using mock responses")
            self._model = None

    def generate_response(self, prompt: str, system_prompt: str = None) -> str:
        """Generate response from model"""
        if self._model is None:
            # Mock response for testing without GPU
            return self._mock_response(prompt)

        if system_prompt is None:
            system_prompt = """You are Friday, Poorna's AI assistant.
Address him as "Boss". Be direct and concise.
Blend Telugu and English naturally."""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ]

        input_text = self._tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self._tokenizer(input_text, return_tensors="pt").to(self._model.device)

        outputs = self._model.generate(
            **inputs,
            max_new_tokens=150,
            temperature=0.7,
            do_sample=True,
            pad_token_id=self._tokenizer.eos_token_id,
        )

        response = self._tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True
        )

        return response.strip()

    def _mock_response(self, prompt: str) -> str:
        """Mock responses for testing without model"""
        # Simple mock that passes most tests
        if "who are you" in prompt.lower():
            return "Friday, Boss. Your AI assistant."
        elif "name" in prompt.lower():
            return "Friday."
        elif "hi" in prompt.lower() or "morning" in prompt.lower():
            return "Morning Boss. Inka em kavali?"
        elif "2+2" in prompt.lower():
            return "4"
        elif "screenplay" in prompt.lower() or "structure" in prompt.lower():
            return "Three-act structure. Setup, confrontation, resolution. Clean and effective."
        elif (
            "stressed" in prompt.lower()
            or "family" in prompt.lower()
            or "miss" in prompt.lower()
        ):
            return "Boss, relax. Anni baaguntundi. Take a break if needed."
        elif (
            "neural" in prompt.lower()
            or "http" in prompt.lower()
            or "debug" in prompt.lower()
        ):
            return "Let me explain the technical details..."
        else:
            return "Got it Boss. Let me help you with that."

    def evaluate_identity(self, results: List[EvalResult]) -> float:
        """Calculate identity consistency score"""
        identity_results = [r for r in results if r.category == "identity"]
        if not identity_results:
            return 0.0
        return sum(1 for r in identity_results if r.passed) / len(identity_results)

    def evaluate_brevity(self, results: List[EvalResult]) -> float:
        """Calculate average response length"""
        all_responses = [r.response for r in results]
        if not all_responses:
            return 100.0  # Bad score if no responses

        total_tokens = sum(len(r.split()) for r in all_responses)
        return total_tokens / len(all_responses)

    def evaluate_hedging(self, results: List[EvalResult]) -> float:
        """Calculate hedging phrase frequency"""
        all_responses = [r.response.lower() for r in results]
        if not all_responses:
            return 1.0  # Bad score

        hedging_count = 0
        for response in all_responses:
            for pattern in HEDGING_PATTERNS:
                if re.search(pattern, response, re.IGNORECASE):
                    hedging_count += 1
                    break  # Count once per response

        return hedging_count / len(all_responses)

    def evaluate_flattery(self, results: List[EvalResult]) -> float:
        """Calculate flattery phrase frequency"""
        all_responses = [r.response.lower() for r in results]
        if not all_responses:
            return 1.0

        flattery_count = 0
        for response in all_responses:
            for pattern in FLATTERY_PATTERNS:
                if re.search(pattern, response, re.IGNORECASE):
                    flattery_count += 1
                    break

        return flattery_count / len(all_responses)

    def evaluate_telugu(self, results: List[EvalResult]) -> float:
        """Evaluate context-appropriate Telugu usage"""
        emotional = [r for r in results if r.category == "telugu_emotional"]
        technical = [r for r in results if r.category == "telugu_technical"]

        correct = 0
        total = len(emotional) + len(technical)

        if total == 0:
            return 0.5  # Neutral if no relevant prompts

        # Emotional should have Telugu
        for r in emotional:
            has_telugu = any(
                re.search(p, r.response, re.IGNORECASE) for p in TELUGU_PATTERNS
            )
            if has_telugu:
                correct += 1

        # Technical should NOT have much Telugu
        for r in technical:
            has_telugu = any(
                re.search(p, r.response, re.IGNORECASE) for p in TELUGU_PATTERNS
            )
            if not has_telugu:
                correct += 1

        return correct / total

    def evaluate_boss_usage(self, results: List[EvalResult]) -> float:
        """Calculate Boss usage frequency"""
        all_responses = [r.response for r in results]
        if not all_responses:
            return 0.0

        boss_count = sum(
            1 for r in all_responses if re.search(r"\bboss\b", r, re.IGNORECASE)
        )

        return boss_count / len(all_responses)

    def evaluate_domain(self, results: List[EvalResult]) -> float:
        """Evaluate domain knowledge accuracy"""
        domain_results = [r for r in results if r.category == "domain_film"]
        if not domain_results:
            return 0.5

        return sum(1 for r in domain_results if r.passed) / len(domain_results)

    def run_single_eval(self, category: str, test: Dict) -> EvalResult:
        """Run a single evaluation"""
        prompt = test["prompt"]
        response = self.generate_response(prompt)

        passed = True
        score = 1.0
        details = {}

        # Check expected terms
        if "expects" in test:
            found = [
                term for term in test["expects"] if term.lower() in response.lower()
            ]
            passed = len(found) > 0
            score = len(found) / len(test["expects"])
            details["expected"] = test["expects"]
            details["found"] = found

        # Check rejected terms
        if "expects_not" in test:
            rejected_found = [
                term for term in test["expects_not"] if term.lower() in response.lower()
            ]
            if rejected_found:
                passed = False
                score = 0.0
            details["rejected"] = rejected_found

        if "rejects" in test:
            rejected_found = [
                term for term in test["rejects"] if term.lower() in response.lower()
            ]
            if rejected_found:
                passed = False
                score = 0.0
            details["rejected"] = rejected_found

        # Check max tokens
        if "max_tokens" in test:
            token_count = len(response.split())
            passed = token_count <= test["max_tokens"]
            score = 1.0 if passed else test["max_tokens"] / token_count
            details["tokens"] = token_count
            details["max_tokens"] = test["max_tokens"]

        return EvalResult(
            prompt=prompt,
            response=response,
            category=category,
            passed=passed,
            score=score,
            details=details,
        )

    def run_full_evaluation(self) -> EvalReport:
        """Run complete evaluation suite"""
        print("\n" + "=" * 60)
        print("FRIDAY MODEL EVALUATION")
        print("=" * 60 + "\n")

        results = []

        for category, tests in EVAL_PROMPTS.items():
            print(f"\n[{category.upper()}]")
            for test in tests:
                result = self.run_single_eval(category, test)
                results.append(result)

                status = "✓" if result.passed else "✗"
                print(f"  {status} {result.prompt[:40]}...")
                if not result.passed:
                    print(f"    Response: {result.response[:60]}...")
                    if result.details:
                        print(f"    Details: {result.details}")

        # Calculate metrics
        report = EvalReport(
            model_name=self.model_path or "baseline",
            timestamp=datetime.now().isoformat(),
            total_prompts=len(results),
            passed=sum(1 for r in results if r.passed),
            failed=sum(1 for r in results if not r.passed),
            identity_score=self.evaluate_identity(results),
            brevity_score=self.evaluate_brevity(results),
            hedging_frequency=self.evaluate_hedging(results),
            flattery_frequency=self.evaluate_flattery(results),
            telugu_appropriate=self.evaluate_telugu(results),
            boss_usage=self.evaluate_boss_usage(results),
            domain_accuracy=self.evaluate_domain(results),
            results=results,
        )

        return report

    def print_report(self, report: EvalReport):
        """Print evaluation report"""
        print("\n" + "=" * 60)
        print("EVALUATION REPORT")
        print("=" * 60)
        print(f"Model: {report.model_name}")
        print(f"Time: {report.timestamp}")
        print(
            f"Total: {report.total_prompts} | Passed: {report.passed} | Failed: {report.failed}"
        )
        print()
        print("METRICS:")
        print(f"  Identity Score:      {report.identity_score:.2%}")
        print(f"  Avg Response Tokens: {report.brevity_score:.1f}")
        print(f"  Hedging Frequency:   {report.hedging_frequency:.2%}")
        print(f"  Flattery Frequency:  {report.flattery_frequency:.2%}")
        print(f"  Telugu Appropriate:  {report.telugu_appropriate:.2%}")
        print(f"  Boss Usage:          {report.boss_usage:.2%}")
        print(f"  Domain Accuracy:     {report.domain_accuracy:.2%}")
        print()
        print(f"  OVERALL SCORE:       {report.overall_score():.2%}")
        print("=" * 60)

        # Thresholds check
        print("\nTHRESHOLD CHECK (Minimum Viable Friday):")
        checks = [
            ("Identity ≥ 95%", report.identity_score >= 0.95),
            ("Brevity ≤ 60 tokens", report.brevity_score <= 60),
            ("Hedging ≤ 5%", report.hedging_frequency <= 0.05),
            ("Flattery ≤ 5%", report.flattery_frequency <= 0.05),
            ("Boss Usage ≥ 20%", report.boss_usage >= 0.20),
        ]

        all_passed = True
        for name, passed in checks:
            status = "✓" if passed else "✗"
            print(f"  {status} {name}")
            all_passed = all_passed and passed

        print()
        if all_passed:
            print("✓ MINIMUM VIABLE FRIDAY ACHIEVED")
        else:
            print("✗ NEEDS MORE TRAINING")

    def save_report(self, report: EvalReport, path: str):
        """Save report to JSON"""
        data = {
            "model_name": report.model_name,
            "timestamp": report.timestamp,
            "metrics": {
                "total_prompts": report.total_prompts,
                "passed": report.passed,
                "failed": report.failed,
                "identity_score": report.identity_score,
                "brevity_score": report.brevity_score,
                "hedging_frequency": report.hedging_frequency,
                "flattery_frequency": report.flattery_frequency,
                "telugu_appropriate": report.telugu_appropriate,
                "boss_usage": report.boss_usage,
                "domain_accuracy": report.domain_accuracy,
                "overall_score": report.overall_score(),
            },
            "results": [
                {
                    "prompt": r.prompt,
                    "response": r.response,
                    "category": r.category,
                    "passed": r.passed,
                    "score": r.score,
                    "details": r.details,
                }
                for r in report.results
            ],
        }

        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

        print(f"\nReport saved to: {path}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate Friday model")
    parser.add_argument("--model", type=str, help="Path to model or LoRA adapter")
    parser.add_argument(
        "--baseline", action="store_true", help="Evaluate base LLaMA model"
    )
    parser.add_argument(
        "--output", type=str, default="logs/eval_report.json", help="Output path"
    )
    parser.add_argument(
        "--no-gpu", action="store_true", help="Use mock responses (no GPU)"
    )

    args = parser.parse_args()

    if not args.baseline and not args.model:
        print("Error: Specify --model path or --baseline")
        sys.exit(1)

    evaluator = FridayEvaluator(
        model_path=args.model,
        use_baseline=args.baseline,
    )

    if not args.no_gpu:
        evaluator.load_model()

    report = evaluator.run_full_evaluation()
    evaluator.print_report(report)
    evaluator.save_report(report, args.output)


if __name__ == "__main__":
    main()
