import os
from transformers import AutoTokenizer

MODEL = os.getenv("MODEL_NAME", "meta-llama/Meta-Llama-3.1-8B-Instruct")
HF_TOKEN = os.getenv("HUGGINGFACE_TOKEN")

if not HF_TOKEN:
    raise SystemExit(
        "HUGGINGFACE_TOKEN not set in environment. Export it or load from .env before running."
    )

print(f"Loading tokenizer for {MODEL} ...")
tok = AutoTokenizer.from_pretrained(
    MODEL,
    use_fast=True,
    trust_remote_code=True,
    use_auth_token=HF_TOKEN,
)

messages = [
    {"role": "system", "content": "You are Friday, my assistant."},
    {"role": "user", "content": "Say hi."},
    {"role": "assistant", "content": "Hi!"},
]

ids = tok.apply_chat_template(messages, tokenize=True, return_tensors=None)
print("Encoded length:", len(ids))
snippet = tok.decode(ids[: min(120, len(ids))])
print("Snippet:\n", snippet)
