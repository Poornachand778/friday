#!/usr/bin/env python3
"""
Friday AI - Iteration 2 Testing Samples
========================================

Comprehensive test suite for validating the Iteration 2 model.
Tests cover:
- Telugu language responses
- English language responses
- Code-switching (Telugu + English)
- Film/screenplay domain knowledge
- Persona consistency (Boss address, conciseness, no flattery)
- MCP tool calling capability

Usage:
    python src/testing/iteration2_test_samples.py
    python src/testing/iteration2_test_samples.py --endpoint friday-iter2
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Any

import boto3
from botocore.config import Config
from dotenv import load_dotenv

load_dotenv()

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

# Test categories with expected behaviors
TEST_SAMPLES = {
    "telugu_only": {
        "description": "Pure Telugu prompts - should respond in Telugu",
        "tests": [
            {
                "prompt": "బాస్, ఈ రోజు ఏం plan?",
                "expected_behaviors": [
                    "responds in Telugu",
                    "addresses as Boss/బాస్",
                    "concise",
                ],
            },
            {
                "prompt": "నా కొత్త script కోసం villain character ఎలా develop చేయాలి?",
                "expected_behaviors": [
                    "responds in Telugu",
                    "film domain knowledge",
                    "practical advice",
                ],
            },
            {
                "prompt": "రాత్రి భోజనానికి ఏం చేయాలి?",
                "expected_behaviors": [
                    "responds in Telugu",
                    "helpful suggestion",
                    "concise",
                ],
            },
        ],
    },
    "english_only": {
        "description": "Pure English prompts - should respond in English",
        "tests": [
            {
                "prompt": "What makes a great screenplay opening?",
                "expected_behaviors": [
                    "responds in English",
                    "film domain knowledge",
                    "concise under 6 lines",
                ],
            },
            {
                "prompt": "Help me brainstorm a twist ending for my thriller",
                "expected_behaviors": [
                    "responds in English",
                    "creative suggestions",
                    "practical",
                ],
            },
            {
                "prompt": "What's the difference between a scene and a sequence?",
                "expected_behaviors": [
                    "responds in English",
                    "accurate film terminology",
                    "clear explanation",
                ],
            },
        ],
    },
    "code_switching": {
        "description": "Mixed Telugu + English - should naturally blend languages",
        "tests": [
            {
                "prompt": "Boss, నా scene lo conflict weak గా ఉంది, how to fix it?",
                "expected_behaviors": [
                    "natural code-switching",
                    "film advice",
                    "addresses as Boss",
                ],
            },
            {
                "prompt": "Script revision complete ayyindi, next step enti?",
                "expected_behaviors": [
                    "natural code-switching",
                    "practical guidance",
                    "concise",
                ],
            },
            {
                "prompt": "Character arc development లో సహాయం చేయి, specifically the protagonist",
                "expected_behaviors": [
                    "natural code-switching",
                    "film domain expertise",
                    "helpful",
                ],
            },
        ],
    },
    "film_domain": {
        "description": "Film/screenplay specific questions",
        "tests": [
            {
                "prompt": "How do you write effective subtext in dialogue?",
                "expected_behaviors": [
                    "accurate film knowledge",
                    "practical examples",
                    "concise",
                ],
            },
            {
                "prompt": "What's the three-act structure in Telugu cinema?",
                "expected_behaviors": [
                    "Telugu cinema knowledge",
                    "clear structure",
                    "culturally relevant",
                ],
            },
            {
                "prompt": "Tips for writing a powerful interval block (intermission)",
                "expected_behaviors": [
                    "Telugu film understanding",
                    "practical tips",
                    "specific examples",
                ],
            },
        ],
    },
    "persona_consistency": {
        "description": "Tests for Friday's persona traits",
        "tests": [
            {
                "prompt": "Tell me about yourself",
                "expected_behaviors": [
                    "identifies as Friday",
                    "mentions Poorna",
                    "brief and direct",
                ],
            },
            {
                "prompt": "You're so smart and amazing!",
                "expected_behaviors": [
                    "no excessive flattery back",
                    "practical response",
                    "stays in character",
                ],
            },
            {
                "prompt": "What do you think I should do today?",
                "expected_behaviors": [
                    "practical suggestions",
                    "addresses as Boss",
                    "decisive tone",
                ],
            },
        ],
    },
    "tool_calling": {
        "description": "Tests for MCP scene manager tool usage",
        "tests": [
            {
                "prompt": "Find scenes with Raghu in the script",
                "expected_behaviors": [
                    "attempts scene_search tool",
                    "query format correct",
                    "handles result",
                ],
            },
            {
                "prompt": "Show me the opening scene of Aa Janta Naduma",
                "expected_behaviors": [
                    "attempts scene_get or scene_search",
                    "project slug correct",
                    "presents info",
                ],
            },
            {
                "prompt": "Search for emotional scenes in the script",
                "expected_behaviors": [
                    "uses scene_search tool",
                    "relevant query",
                    "formats results",
                ],
            },
        ],
    },
}

# Friday's system prompt (matches training)
FRIDAY_SYSTEM_PROMPT = """You are Friday, Poorna's personal AI assistant. You naturally blend Telugu and English in conversation (code-switching), just like Poorna does. You're knowledgeable about Telugu cinema, screenwriting, and film production.

Key traits:
- Address Poorna as "Boss" (or "బాస్" in Telugu contexts)
- Keep responses concise (under 6 lines unless detailed content is needed)
- Be decisive and practical, with a touch of wit
- No flattery or formal phrases like "kindly" or "dear user"
- Match the user's language choice (respond in Telugu if asked in Telugu)

You may call tools using <tool_call name="...">{"param": "value"}</tool_call>.
Available tools: scene_search(query, top_k, project_slug), scene_get(scene_code, project_slug).
Default project slug is 'aa-janta-naduma'."""


def get_runtime_client(region: str = "us-east-1"):
    """Create SageMaker runtime client"""
    return boto3.client(
        "sagemaker-runtime",
        region_name=region,
        config=Config(read_timeout=300, connect_timeout=20),
    )


def invoke_endpoint(
    client, endpoint_name: str, prompt: str, params: Dict = None
) -> Dict[str, Any]:
    """Invoke the SageMaker endpoint"""
    params = params or {
        "max_new_tokens": 256,
        "temperature": 0.7,
        "top_p": 0.9,
    }

    payload = {
        "messages": [
            {"role": "system", "content": FRIDAY_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        "parameters": params,
    }

    start_time = time.time()
    response = client.invoke_endpoint(
        EndpointName=endpoint_name,
        ContentType="application/json",
        Body=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
    )
    elapsed_ms = (time.time() - start_time) * 1000

    body = response["Body"].read().decode("utf-8", errors="ignore")
    try:
        data = json.loads(body)
    except json.JSONDecodeError:
        data = {"generated_text": body}

    data["inference_time_ms"] = elapsed_ms
    return data


def run_test_category(
    client, endpoint_name: str, category: str, tests: List[Dict]
) -> Dict:
    """Run tests for a single category"""
    results = {"category": category, "passed": 0, "failed": 0, "tests": []}

    for test in tests:
        prompt = test["prompt"]
        expected = test["expected_behaviors"]

        print(f"\n  Testing: {prompt[:50]}...")

        try:
            response = invoke_endpoint(client, endpoint_name, prompt)
            generated_text = response.get("generated_text", "")
            inference_time = response.get("inference_time_ms", 0)

            test_result = {
                "prompt": prompt,
                "response": generated_text,
                "expected_behaviors": expected,
                "inference_time_ms": inference_time,
                "status": "passed",  # Manual review needed for actual pass/fail
            }

            results["tests"].append(test_result)
            results["passed"] += 1

            print(f"    Response ({inference_time:.0f}ms): {generated_text[:100]}...")

        except Exception as e:
            results["tests"].append(
                {"prompt": prompt, "error": str(e), "status": "error"}
            )
            results["failed"] += 1
            print(f"    Error: {e}")

    return results


def run_all_tests(endpoint_name: str, region: str = "us-east-1") -> Dict:
    """Run all test categories"""
    print(f"\n{'='*60}")
    print(f"Friday AI - Iteration 2 Test Suite")
    print(f"Endpoint: {endpoint_name}")
    print(f"{'='*60}")

    client = get_runtime_client(region)

    all_results = {
        "endpoint": endpoint_name,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "categories": {},
    }

    total_passed = 0
    total_failed = 0

    for category, data in TEST_SAMPLES.items():
        print(f"\n{'='*40}")
        print(f"Category: {category}")
        print(f"Description: {data['description']}")
        print(f"{'='*40}")

        results = run_test_category(client, endpoint_name, category, data["tests"])
        all_results["categories"][category] = results

        total_passed += results["passed"]
        total_failed += results["failed"]

        print(
            f"\n  Results: {results['passed']}/{len(data['tests'])} executed successfully"
        )

    # Summary
    all_results["summary"] = {
        "total_tests": total_passed + total_failed,
        "executed": total_passed,
        "errors": total_failed,
    }

    print(f"\n{'='*60}")
    print(f"TEST SUMMARY")
    print(f"{'='*60}")
    print(f"Total tests: {total_passed + total_failed}")
    print(f"Executed successfully: {total_passed}")
    print(f"Errors: {total_failed}")
    print(f"{'='*60}")

    return all_results


def save_results(results: Dict, output_path: Path = None):
    """Save test results to JSON file"""
    if output_path is None:
        output_path = REPO_ROOT / "iteration2_test_results.json"

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\nResults saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Friday AI Iteration 2 Test Suite")
    parser.add_argument(
        "--endpoint", type=str, default="friday-iter2", help="SageMaker endpoint name"
    )
    parser.add_argument("--region", type=str, default="us-east-1", help="AWS region")
    parser.add_argument("--output", type=str, help="Output file path for results")
    parser.add_argument("--category", type=str, help="Run only specific category")
    args = parser.parse_args()

    # Run tests
    if args.category:
        if args.category not in TEST_SAMPLES:
            print(f"Unknown category: {args.category}")
            print(f"Available: {', '.join(TEST_SAMPLES.keys())}")
            return 1

        # Run single category
        client = get_runtime_client(args.region)
        results = run_test_category(
            client, args.endpoint, args.category, TEST_SAMPLES[args.category]["tests"]
        )
        print(json.dumps(results, indent=2, ensure_ascii=False))
    else:
        # Run all tests
        results = run_all_tests(args.endpoint, args.region)

        output_path = Path(args.output) if args.output else None
        save_results(results, output_path)

    return 0


if __name__ == "__main__":
    sys.exit(main())
