#!/usr/bin/env python3
"""
Test Friday model with questions from training data
Saves expected vs actual responses for review
"""

import json
import boto3
import os
from datetime import datetime

# AWS credentials
os.environ["AWS_ACCESS_KEY_ID"] = "YOUR_AWS_KEY_ID"
os.environ["AWS_SECRET_ACCESS_KEY"] = "YOUR_AWS_SECRET"
os.environ["AWS_DEFAULT_REGION"] = "us-east-1"

ENDPOINT_NAME = "friday-iter3"
SYSTEM_PROMPT = "You are Friday, Poorna's AI assistant. Blend Telugu and English naturally. Address him as 'Boss'. Be concise and direct. No flattery or excessive formality."


def invoke_endpoint(prompt: str) -> str:
    """Invoke SageMaker endpoint"""
    runtime = boto3.client("sagemaker-runtime")

    payload = {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        "parameters": {"max_new_tokens": 200, "temperature": 0.7},
    }

    response = runtime.invoke_endpoint(
        EndpointName=ENDPOINT_NAME,
        ContentType="application/json",
        Body=json.dumps(payload),
    )

    result = json.loads(response["Body"].read().decode())
    return result.get("generated_text", "")


def main():
    # Load training questions
    with open("data/training_collection/phase1_base_questions.json") as f:
        data = json.load(f)

    results = {
        "timestamp": datetime.now().isoformat(),
        "endpoint": ENDPOINT_NAME,
        "system_prompt": SYSTEM_PROMPT,
        "tests": [],
    }

    # Test questions from each category
    for category, cat_data in data.get("categories", {}).items():
        print(f"\n=== Testing {category} ===")

        for pair in cat_data.get("pairs", [])[:3]:  # Test first 3 from each category
            question = pair["user"]
            expected = pair["assistant"]

            print(f"\nQ: {question}")

            try:
                actual = invoke_endpoint(question)
                print(f"Expected: {expected[:80]}...")
                print(f"Actual:   {actual[:80]}...")
            except Exception as e:
                actual = f"ERROR: {e}"
                print(f"Error: {e}")

            results["tests"].append(
                {
                    "category": category,
                    "id": pair.get("id", "unknown"),
                    "question": question,
                    "expected": expected,
                    "actual": actual,
                }
            )

    # Save results
    output_file = (
        f"eval/training_questions_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    )
    os.makedirs("eval", exist_ok=True)

    with open(output_file, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\n\nResults saved to: {output_file}")

    # Also create markdown summary
    md_file = output_file.replace(".json", ".md")
    with open(md_file, "w") as f:
        f.write(f"# Friday Training Questions Test\n\n")
        f.write(f"**Date**: {results['timestamp']}\n")
        f.write(f"**Endpoint**: {ENDPOINT_NAME}\n\n")

        current_cat = None
        for test in results["tests"]:
            if test["category"] != current_cat:
                current_cat = test["category"]
                f.write(f"\n## {current_cat.upper()}\n\n")

            f.write(f"### {test['id']}: {test['question']}\n\n")
            f.write(f"**Expected:**\n> {test['expected']}\n\n")
            f.write(f"**Actual:**\n> {test['actual']}\n\n")
            f.write("---\n\n")

    print(f"Markdown saved to: {md_file}")


if __name__ == "__main__":
    main()
