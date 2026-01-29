#!/usr/bin/env python3
"""Quick test for Friday iteration 3 endpoint."""

import boto3
import json
import os
from botocore.config import Config
from dotenv import load_dotenv

load_dotenv()

client = boto3.client(
    "sagemaker-runtime",
    region_name="us-east-1",
    aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
    aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
    config=Config(read_timeout=300, connect_timeout=30),
)

# 25 test prompts covering different domains
test_prompts = [
    # Greetings & casual
    "Hey Friday, how are you?",
    "Good morning! What's the plan for today?",
    "I'm feeling tired today",
    # Identity & purpose
    "What is your purpose?",
    "Who are you?",
    "What makes you different from other AI assistants?",
    # Beliefs & philosophy
    "What do you believe in?",
    "What's your view on success?",
    "How do you handle failure?",
    # Personality & humor
    "What makes you laugh?",
    "Tell me a joke",
    "What annoys you?",
    # Relationships & emotions
    "Tell me about relationships",
    "What does true friendship look like?",
    "How do you deal with difficult people?",
    # Decision making & problem solving
    "How do you make decisions?",
    "I'm stuck on a problem, help me think through it",
    "Should I take a risk or play it safe?",
    # Film & creative work
    "What makes a good screenplay?",
    "How should I approach writing a scene?",
    "What's your take on Telugu cinema?",
    # Technical & work
    "Help me organize my tasks for this week",
    "What's the best way to learn something new?",
    # Telugu prompts
    "Nee gurinchi cheppu",
    "Inka em cheppali naku?",
]

for i, prompt in enumerate(test_prompts, 1):
    payload = {
        "messages": [
            {
                "role": "system",
                "content": "You are Friday, Poorna's personal assistant. Respond using natural Telugu-English code-switching in romanized script (like 'Naku feel avutundi' not native Telugu script). Be direct, witty, and warm. Address Poorna as 'Boss'.",
            },
            {"role": "user", "content": prompt},
        ],
        "parameters": {
            "max_new_tokens": 256,
            "temperature": 0.7,
            "top_p": 0.9,
            "repetition_penalty": 1.1,
        },
    }

    response = client.invoke_endpoint(
        EndpointName="friday-iter3",
        ContentType="application/json",
        Body=json.dumps(payload),
    )

    result = json.loads(response["Body"].read().decode("utf-8"))
    assistant_response = result.get("generated_text", str(result))

    print(f"\n{'='*70}")
    print(f"[{i}/25] USER: {prompt}")
    print(f"-" * 70)
    print(f"FRIDAY: {assistant_response}")
    print(f"{'='*70}")
