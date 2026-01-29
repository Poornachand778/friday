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
from typing import Any, Dict, List, Union
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
    if isinstance(data, list):
        return {"inputs": data, "parameters": {}}
    raise ValueError("Invalid payload")


def _ensure_inputs(data: Dict[str, Any]) -> List[str]:
    raw_inputs: Union[str, List[str]] = data.get("inputs") or ""
    if isinstance(raw_inputs, str):
        return [raw_inputs]
    if isinstance(raw_inputs, list):
        if not raw_inputs:
            raise ValueError("Input list cannot be empty")
        if not all(isinstance(x, str) for x in raw_inputs):
            raise ValueError("All inputs must be strings")
        return raw_inputs
    raise ValueError("inputs must be a string or list of strings")


def _prepare_chat_prompts(data: Dict[str, Any]) -> List[str]:
    raw_messages = data.get("messages")
    if not raw_messages:
        return []

    # Normalize to list of conversations
    if (
        isinstance(raw_messages, list)
        and raw_messages
        and isinstance(raw_messages[0], list)
    ):
        conversations = raw_messages
    else:
        conversations = [raw_messages]

    prompts: List[str] = []
    for conversation in conversations:
        prepared: List[Dict[str, str]] = []
        for message in conversation:
            role = message.get("role", "user")
            content = message.get("content") or ""
            if role == "tool":
                name = message.get("name", "tool")
                content = f"[tool:{name}] {content}".strip()
                role = "user"
            prepared.append({"role": role, "content": content})
        prompt_text = _TOKENIZER.apply_chat_template(
            prepared,
            add_generation_prompt=True,
            tokenize=False,
        )
        prompts.append(prompt_text)
    return prompts


def predict_fn(data, context):
    """Run inference with token limits and proper output handling"""
    # Ensure model is loaded
    _load_model_once(context["model_dir"])

    prompts = _prepare_chat_prompts(data)
    if not prompts:
        prompts = _ensure_inputs(data)
    params = data.get("parameters") or {}

    # Get token limits from environment
    max_input_length = int(os.environ.get("MAX_INPUT_LENGTH", "4096"))
    max_total_tokens = int(os.environ.get("MAX_TOTAL_TOKENS", "8192"))

    # Tokenize with padding/truncation for batch support
    tokenized = _TOKENIZER(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_input_length,
    )
    input_lengths = (tokenized["input_ids"] != _TOKENIZER.pad_token_id).sum(dim=1)
    tokenized = {k: v.to(_MODEL.device) for k, v in tokenized.items()}

    # Reject inputs that were truncated
    if torch.any(input_lengths >= max_input_length):
        raise ValueError(
            f"One or more inputs exceed max length ({max_input_length} tokens); please shorten the prompt."
        )

    # Calculate available tokens for generation
    max_new_tokens = int(params.get("max_new_tokens", 128))
    available_tokens = max_total_tokens - input_lengths.max().item()
    max_new_tokens = min(max_new_tokens, available_tokens)

    if max_new_tokens <= 0:
        raise ValueError(
            f"No tokens available for generation (longest input: {input_lengths.max().item()}, max_total: {max_total_tokens})"
        )

    temperature = float(params.get("temperature", 0.7))
    top_p = float(params.get("top_p", 0.9))
    top_k = params.get("top_k")
    do_sample = params.get("do_sample")
    if do_sample is None:
        do_sample = temperature > 0
    stops = params.get("stop") or []
    if isinstance(stops, str):
        stops = [stops]

    seed = params.get("seed")
    if seed is not None:
        seed = int(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    # Generate with proper inference mode and pad token
    with torch.inference_mode():
        generate_kwargs = {
            **tokenized,
            "max_new_tokens": max_new_tokens,
            "do_sample": bool(do_sample),
            "eos_token_id": _TOKENIZER.eos_token_id,
            "pad_token_id": _TOKENIZER.pad_token_id,
        }
        if temperature is not None and do_sample:
            generate_kwargs["temperature"] = max(temperature, 1e-5)
            generate_kwargs["top_p"] = top_p
            if top_k is not None:
                generate_kwargs["top_k"] = int(top_k)

        output_ids = _MODEL.generate(**generate_kwargs)

    # Slice off the input tokens to avoid echoing the prompt
    generations: List[str] = []
    completion_lengths: List[int] = []

    for idx, prompt_len in enumerate(input_lengths.tolist()):
        gen_ids = output_ids[idx][prompt_len:]
        text = _TOKENIZER.decode(gen_ids, skip_special_tokens=True)

        for seq in stops:
            if seq and seq in text:
                text = text.split(seq)[0]
                truncated_ids = _TOKENIZER(text, add_special_tokens=False)["input_ids"]
                gen_ids = torch.tensor(truncated_ids, device=_MODEL.device)
                break

        generations.append(text.strip())
        completion_lengths.append(int(len(gen_ids)))

    prompt_tokens_total = int(input_lengths.sum().item())
    completion_tokens_total = int(sum(completion_lengths))

    response: Dict[str, Any] = {
        "generated_text": generations[0] if len(generations) == 1 else generations,
        "usage": {
            "prompt_tokens": prompt_tokens_total,
            "completion_tokens": completion_tokens_total,
            "total_tokens": prompt_tokens_total + completion_tokens_total,
        },
        "details": {
            "prompt_tokens_per_sample": input_lengths.tolist(),
            "completion_tokens_per_sample": completion_lengths,
            "max_new_tokens": max_new_tokens,
        },
    }

    return response


def output_fn(pred, accept="application/json"):
    """Format response"""
    return json.dumps(pred)
