#!/usr/bin/env python3
"""
Production inference server for Friday AI
Handles real-time inference with PEFT LoRA adapters on SageMaker
Optimized lazy loading for fast /ping and first inference
"""

import os
import json
import threading
from pathlib import Path
from typing import Any, Dict
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

# Global model state with thread safety
_MODEL = None
_TOKENIZER = None
_LOCK = threading.Lock()


def _cache_dir():
    """Ensure cache directory exists and return path"""
    d = os.environ.get("TRANSFORMERS_CACHE", "/tmp/hf")
    Path(d).mkdir(parents=True, exist_ok=True)
    return d


def _load_model_once(model_dir: str):
    """Load model and tokenizer once with thread safety"""
    global _MODEL, _TOKENIZER
    if _MODEL is not None:
        return

    with _LOCK:
        if _MODEL is not None:
            return

        base_id = os.environ["BASE_MODEL_ID"]
        hf_token = os.environ.get("HF_TOKEN")
        cache_dir = _cache_dir()

        print(f"🤖 Loading model {base_id} to cache {cache_dir}")

        # Enable TF32 for better performance on Ampere+ GPUs
        if torch.cuda.is_available():
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

        # Try 4-bit quantization with graceful fallback to fp16
        bnb_config = None
        try:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.float16,
            )
            print("⚡ Using 4-bit quantization")
        except Exception as e:
            print(f"⚠️ BitsAndBytes failed, falling back to fp16: {e}")
            bnb_config = None

        # Load tokenizer
        _TOKENIZER = AutoTokenizer.from_pretrained(
            base_id, token=hf_token, use_fast=True, cache_dir=cache_dir
        )

        # Set pad token for LLaMA (required for proper generation)
        if _TOKENIZER.pad_token is None:
            _TOKENIZER.pad_token = _TOKENIZER.eos_token
            _TOKENIZER.pad_token_id = _TOKENIZER.eos_token_id

        # Load base model
        model_kwargs = {
            "token": hf_token,
            "device_map": "auto",
            "cache_dir": cache_dir,
        }

        if bnb_config:
            model_kwargs["quantization_config"] = bnb_config
        else:
            model_kwargs["torch_dtype"] = torch.float16

        base = AutoModelForCausalLM.from_pretrained(base_id, **model_kwargs)

        # Load LoRA adapters from model directory
        adapters = os.path.join(model_dir, "adapters")
        _MODEL = PeftModel.from_pretrained(base, adapters).eval()

        print("✅ Model loaded successfully")


def model_fn(model_dir: str) -> Dict[str, Any]:
    """SageMaker model loading - starts background loading for fast /ping"""
    print(f"📦 Model function called with {model_dir}")

    # Pre-warm in the background so /ping stays fast and first Invoke is quicker
    threading.Thread(target=_load_model_once, args=(model_dir,), daemon=True).start()

    return {"model_dir": model_dir}


def input_fn(body: str, content_type="application/json"):
    """Parse inference request"""
    data = json.loads(body)
    if isinstance(data, dict):
        return data
    if isinstance(data, str):
        return {"inputs": data, "parameters": {}}
    raise ValueError("Invalid payload")


def predict_fn(data, context):
    """Run inference with token limits and proper output handling"""
    # Ensure model is loaded
    _load_model_once(context["model_dir"])

    text = data.get("inputs") or ""
    params = data.get("parameters") or {}

    # Get token limits from environment
    max_input_length = int(os.environ.get("MAX_INPUT_LENGTH", "4096"))
    max_total_tokens = int(os.environ.get("MAX_TOTAL_TOKENS", "8192"))

    # Tokenize and validate input length
    inputs = _TOKENIZER(text, return_tensors="pt").to(_MODEL.device)
    input_length = inputs.input_ids.shape[1]

    # Reject inputs that are too long
    if input_length > max_input_length:
        raise ValueError(
            f"Input too long: {input_length} tokens (max {max_input_length})"
        )

    # Calculate available tokens for generation
    max_new_tokens = int(params.get("max_new_tokens", 128))
    available_tokens = max_total_tokens - input_length
    max_new_tokens = min(max_new_tokens, available_tokens)

    if max_new_tokens <= 0:
        raise ValueError(
            f"No tokens available for generation (input: {input_length}, max_total: {max_total_tokens})"
        )

    temperature = float(params.get("temperature", 0.7))
    top_p = float(params.get("top_p", 0.9))
    stops = params.get("stop") or []

    # Generate with proper inference mode and pad token
    with torch.inference_mode():
        output_ids = _MODEL.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=temperature > 0,
            temperature=temperature if temperature > 0 else None,
            top_p=top_p if temperature > 0 else None,
            eos_token_id=_TOKENIZER.eos_token_id,
            pad_token_id=_TOKENIZER.pad_token_id,
        )

    # Slice off the input tokens to avoid echoing the prompt
    new_tokens = output_ids[0][input_length:]
    generated_text = _TOKENIZER.decode(new_tokens, skip_special_tokens=True)

    # Apply stop sequences
    for seq in stops:
        if seq and seq in generated_text:
            generated_text = generated_text.split(seq)[0]
            break

    return {
        "generated_text": generated_text,
        "tokens": {
            "input_tokens": input_length,
            "generated_tokens": len(new_tokens),
            "max_new_tokens": max_new_tokens,
        },
    }


def output_fn(pred, accept="application/json"):
    """Format response"""
    return json.dumps(pred), accept
