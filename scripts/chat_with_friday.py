#!/usr/bin/env python3
"""Simple CLI chat with Friday SageMaker endpoint."""

import boto3
import json
import os
from dotenv import load_dotenv

load_dotenv()

runtime = boto3.client(
    "sagemaker-runtime", region_name=os.getenv("AWS_DEFAULT_REGION", "us-east-1")
)
endpoint = os.getenv("SAGEMAKER_ENDPOINT_NAME", "friday-iter2")

SYSTEM_PROMPT = """You are Friday, Poorna's personal AI assistant.
- Address Poorna as 'Boss'
- Use natural Telugu-English code-switching
- Be direct, concise, and helpful
- For screenplay work, you can suggest scene_search, scene_get, scene_update tools"""

conversation = [{"role": "system", "content": SYSTEM_PROMPT}]

print("=" * 60)
print("🎭 Chat with Friday (Iteration 2)")
print(f"   Endpoint: {endpoint}")
print("   Type 'quit' to exit, 'clear' to reset conversation")
print("=" * 60)
print()

while True:
    try:
        user_input = input("You: ").strip()
    except (KeyboardInterrupt, EOFError):
        print("\nBye, Boss!")
        break

    if not user_input:
        continue
    if user_input.lower() == "quit":
        print("Bye, Boss!")
        break
    if user_input.lower() == "clear":
        conversation = [{"role": "system", "content": SYSTEM_PROMPT}]
        print("[Conversation cleared]")
        continue

    conversation.append({"role": "user", "content": user_input})

    payload = {
        "messages": conversation,
        "parameters": {"max_new_tokens": 200, "temperature": 0.7},
    }

    try:
        response = runtime.invoke_endpoint(
            EndpointName=endpoint,
            ContentType="application/json",
            Body=json.dumps(payload),
        )
        result = json.loads(response["Body"].read().decode())

        if isinstance(result, dict) and "generated_text" in result:
            reply = result["generated_text"]
        elif isinstance(result, list) and len(result) > 0:
            reply = result[0].get("generated_text", str(result[0]))
        else:
            reply = str(result)

        conversation.append({"role": "assistant", "content": reply})
        print(f"\nFriday: {reply}\n")

    except Exception as e:
        print(f"\n[Error: {e}]\n")
