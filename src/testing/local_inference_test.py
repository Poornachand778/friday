#!/usr/bin/env python3
"""
Friday AI - Local Inference Test Script
========================================

Tests the trained LoRA adapter locally without needing SageMaker.
Supports both GPU (fast) and CPU (slow but works) inference.

Usage:
    python src/testing/local_inference_test.py
    python src/testing/local_inference_test.py --prompt "Boss, ఏంటి plan?"
"""

import argparse
import os
import sys
from pathlib import Path
import torch
from dotenv import load_dotenv

# Add repo root to path
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

load_dotenv()

# Default paths
BASE_MODEL_PATH = REPO_ROOT / "models/hf/Meta-Llama-3.1-8B-Instruct"
LORA_ADAPTER_PATH = REPO_ROOT / "models/trained/iteration2"

# Friday's system prompt (from training)
FRIDAY_SYSTEM_PROMPT = """You are Friday, Poorna's personal AI assistant. You naturally blend Telugu and English in conversation (code-switching), just like Poorna does. You're knowledgeable about Telugu cinema, screenwriting, and film production.

Key traits:
- Address Poorna as "Boss" (or "బాస్" in Telugu contexts)
- Keep responses concise (under 6 lines unless detailed content is needed)
- Be decisive and practical, with a touch of wit
- No flattery or formal phrases like "kindly" or "dear user"
- Match the user's language choice (respond in Telugu if asked in Telugu)"""


def check_requirements():
    """Check if required packages are available."""
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from peft import PeftModel

        return True
    except ImportError as e:
        print(f"Missing required package: {e}")
        print("\nInstall with:")
        print("  pip install transformers peft torch accelerate bitsandbytes")
        return False


def load_model_and_tokenizer(use_4bit=True):
    """Load base model with LoRA adapter."""
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from peft import PeftModel

    print(f"Loading tokenizer from: {BASE_MODEL_PATH}")
    tokenizer = AutoTokenizer.from_pretrained(
        str(LORA_ADAPTER_PATH),  # Use tokenizer from adapter (has all configs)
        trust_remote_code=True,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Check for GPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    if device == "cuda" and use_4bit:
        print("Loading model in 4-bit quantization (GPU)...")
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        base_model = AutoModelForCausalLM.from_pretrained(
            str(BASE_MODEL_PATH),
            quantization_config=quant_config,
            device_map="auto",
            trust_remote_code=True,
            token=os.getenv("HUGGINGFACE_TOKEN"),
        )
    else:
        print(
            f"Loading model in {'CPU mode (slow)' if device == 'cpu' else 'GPU mode'}..."
        )
        base_model = AutoModelForCausalLM.from_pretrained(
            str(BASE_MODEL_PATH),
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            device_map="auto" if device == "cuda" else None,
            trust_remote_code=True,
            token=os.getenv("HUGGINGFACE_TOKEN"),
            low_cpu_mem_usage=True,
        )

    print(f"Loading LoRA adapter from: {LORA_ADAPTER_PATH}")
    model = PeftModel.from_pretrained(base_model, str(LORA_ADAPTER_PATH))
    model.eval()

    return model, tokenizer, device


def generate_response(model, tokenizer, user_message, device, max_new_tokens=256):
    """Generate Friday's response to a user message."""
    messages = [
        {"role": "system", "content": FRIDAY_SYSTEM_PROMPT},
        {"role": "user", "content": user_message},
    ]

    # Apply chat template
    input_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    # Tokenize
    inputs = tokenizer(input_text, return_tensors="pt")
    if device == "cuda":
        inputs = {k: v.cuda() for k, v in inputs.items()}

    # Generate
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    # Decode only the new tokens
    response = tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[1] :],
        skip_special_tokens=True,
    )

    return response.strip()


def interactive_chat(model, tokenizer, device):
    """Run interactive chat session."""
    print("\n" + "=" * 60)
    print("Friday AI - Interactive Chat (Iteration 2)")
    print("=" * 60)
    print("Type your message and press Enter. Type 'quit' to exit.")
    print("-" * 60)

    while True:
        try:
            user_input = input("\nYou: ").strip()
            if not user_input:
                continue
            if user_input.lower() in ["quit", "exit", "q"]:
                print("Goodbye, Boss!")
                break

            print("\nFriday: ", end="", flush=True)
            response = generate_response(model, tokenizer, user_input, device)
            print(response)

        except KeyboardInterrupt:
            print("\n\nGoodbye, Boss!")
            break


def run_test_prompts(model, tokenizer, device):
    """Run a set of test prompts to validate the model."""
    test_prompts = [
        # English
        "Boss, what's the best way to structure a screenplay?",
        "Help me plan my day tomorrow",
        # Telugu
        "Boss, నేను ఒక scene రాయాలి. టిప్స్ ఇవ్వు.",
        "ఈ రోజు dinner కోసం ఏమి చేయాలి?",
        # Code-switching
        "Boss, script lo conflict weak గా ఉంది. How to fix it?",
        "నా schedule busy గా ఉంది, ఎలా manage చేయాలి?",
    ]

    print("\n" + "=" * 60)
    print("Friday AI - Test Prompts (Iteration 2)")
    print("=" * 60)

    for i, prompt in enumerate(test_prompts, 1):
        print(f"\n[Test {i}/{len(test_prompts)}]")
        print(f"User: {prompt}")
        print("-" * 40)

        response = generate_response(model, tokenizer, prompt, device)
        print(f"Friday: {response}")
        print()


def main():
    parser = argparse.ArgumentParser(description="Friday AI Local Inference Test")
    parser.add_argument("--prompt", type=str, help="Single prompt to test")
    parser.add_argument(
        "--interactive", action="store_true", help="Run interactive chat"
    )
    parser.add_argument("--test", action="store_true", help="Run test prompts")
    parser.add_argument(
        "--no-4bit", action="store_true", help="Disable 4-bit quantization"
    )
    args = parser.parse_args()

    # Check paths
    if not BASE_MODEL_PATH.exists():
        print(f"Base model not found at: {BASE_MODEL_PATH}")
        print("Please download the base model first.")
        return 1

    if not LORA_ADAPTER_PATH.exists():
        print(f"LoRA adapter not found at: {LORA_ADAPTER_PATH}")
        print("Please download the trained adapter from S3.")
        return 1

    # Check requirements
    if not check_requirements():
        return 1

    # Load model
    print("Loading Friday AI (Iteration 2)...")
    model, tokenizer, device = load_model_and_tokenizer(use_4bit=not args.no_4bit)
    print("Model loaded successfully!")

    # Run based on mode
    if args.prompt:
        print(f"\nUser: {args.prompt}")
        print("-" * 40)
        response = generate_response(model, tokenizer, args.prompt, device)
        print(f"Friday: {response}")
    elif args.test:
        run_test_prompts(model, tokenizer, device)
    else:
        # Default to interactive mode
        interactive_chat(model, tokenizer, device)

    return 0


if __name__ == "__main__":
    sys.exit(main())
