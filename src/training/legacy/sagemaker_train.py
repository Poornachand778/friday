#!/usr/bin/env python3
"""SageMaker training entrypoint for Friday Iteration 2."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List

import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    BitsAndBytesConfig,
)


def load_dataset(path: Path) -> List[Dict]:
    data: List[Dict] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                data.append(json.loads(line))
    return data


def build_chat(example: Dict, tokenizer, max_length: int | None = None) -> Dict | None:
    msgs = []
    for msg in example["messages"]:
        role = msg.get("role")
        content = msg.get("content") or ""
        if role == "assistant" and msg.get("tool_calls"):
            extras = []
            for call in msg["tool_calls"]:
                name = call.get("name", "tool")
                args = call.get("arguments", {})
                extras.append(
                    f'<tool_call name="{name}">{json.dumps(args, ensure_ascii=False)}</tool_call>'
                )
            if extras:
                content = (content + "\n" + "\n".join(extras)).strip()
        elif role == "tool":
            name = msg.get("name", "tool")
            content = f"[tool:{name}] {content}".strip()
            role = "user"
        msgs.append({"role": role, "content": content})

    input_ids = tokenizer.apply_chat_template(
        msgs, add_generation_prompt=False, tokenize=True, return_tensors=None
    )
    labels = [-100] * len(input_ids)
    partial: List[Dict] = []
    prev = 0
    assistant_tokens = 0
    for message in msgs:
        partial.append(message)
        ids = tokenizer.apply_chat_template(
            partial, add_generation_prompt=False, tokenize=True, return_tensors=None
        )
        if message["role"] == "assistant":
            for idx in range(prev, len(ids)):
                labels[idx] = input_ids[idx]
                assistant_tokens += 1
        prev = len(ids)
    if max_length is not None and len(input_ids) > max_length:
        input_ids = input_ids[:max_length]
        labels = labels[:max_length]
        assistant_tokens = sum(1 for x in labels if x != -100)

    if assistant_tokens == 0:
        return None
    return {"input_ids": input_ids, "labels": labels}


def collate(tokenizer, pad_to=8):
    pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id

    def _fn(features: List[Dict]):
        features = [feat for feat in features if any(x != -100 for x in feat["labels"])]
        if not features:
            raise ValueError("Batch without assistant tokens detected")

        max_len = max(len(f["input_ids"]) for f in features)
        if pad_to and max_len % pad_to:
            max_len = ((max_len // pad_to) + 1) * pad_to

        batch_input, batch_labels, batch_attn = [], [], []
        tokens = total = 0
        for feat in features:
            ids = feat["input_ids"]
            labs = feat["labels"]
            pad = max_len - len(ids)
            batch_input.append(ids + [pad_id] * pad)
            batch_labels.append(labs + [-100] * pad)
            batch_attn.append([1] * len(ids) + [0] * pad)
            tokens += sum(1 for x in labs if x != -100)
            total += len(labs)

        return {
            "input_ids": torch.tensor(batch_input, dtype=torch.long),
            "labels": torch.tensor(batch_labels, dtype=torch.long),
            "attention_mask": torch.tensor(batch_attn, dtype=torch.long),
        }

    return _fn


def enable_input_grads(model):
    """Ensure embedding outputs require grad for LoRA + checkpointing."""
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    else:
        emb = model.get_input_embeddings()

        def _out_require_grad(module, inputs, output):
            if isinstance(output, torch.Tensor):
                output.requires_grad_(True)

        emb.register_forward_hook(_out_require_grad)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", default="meta-llama/Meta-Llama-3.1-8B-Instruct")
    parser.add_argument(
        "--train-data", default="/opt/ml/input/data/training/train.jsonl"
    )
    parser.add_argument(
        "--valid-data", default="/opt/ml/input/data/training/valid.jsonl"
    )
    parser.add_argument("--output-dir", default="/opt/ml/model")
    parser.add_argument("--lora-rank", type=int, default=32)
    parser.add_argument("--lora-alpha", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--warmup-steps", type=int, default=100)
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--eval-steps", type=int, default=200)
    parser.add_argument("--save-steps", type=int, default=200)
    parser.add_argument("--max-length", type=int, default=4096)
    args = parser.parse_args()

    print("🚀 Friday Iteration 2 fine-tuning")
    print(f"   Model: {args.model_name}")
    print(f"   Train data: {args.train_data}")
    print(f"   Valid data: {args.valid_data}")

    token = os.getenv("HUGGINGFACE_TOKEN")
    if not token:
        raise ValueError("HUGGINGFACE_TOKEN missing")

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name,
        token=token,
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # QLoRA: 4-bit quantization + LoRA adapters
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    if torch.cuda.is_available():
        try:
            torch.backends.cuda.matmul.allow_tf32 = True
        except Exception:
            pass

    quant_cfg = None
    num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
    print(f"🔍 Detected {num_gpus} GPU(s)")

    if torch.cuda.is_available():
        quant_cfg = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,  # Use bfloat16 for better stability
        )

    # Use device_map="auto" for smart distribution across GPUs (per chronicles)
    device_map = "auto" if torch.cuda.is_available() else None

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch_dtype,
        token=token,
        trust_remote_code=True,
        use_cache=False,
        low_cpu_mem_usage=True,
        device_map=device_map,
        quantization_config=quant_cfg,
    )
    if torch.cuda.is_available():
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
        enable_input_grads(model)
        model.config.use_cache = False
        if hasattr(model.config, "pretraining_tp"):
            model.config.pretraining_tp = 1

    lora_cfg = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        lora_dropout=0.05,
        bias="none",
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    train_records = load_dataset(Path(args.train_data))
    valid_records = load_dataset(Path(args.valid_data))

    train_samples: List[Dict] = []
    skipped = 0
    for ex in train_records:
        chat = build_chat(ex, tokenizer, max_length=args.max_length)
        if chat is None:
            skipped += 1
            continue
        train_samples.append(chat)
    if skipped:
        print(
            f"⚠️ Skipped {skipped} training examples with zero assistant tokens after truncation"
        )

    valid_samples: List[Dict] = []
    val_skipped = 0
    for ex in valid_records:
        chat = build_chat(ex, tokenizer, max_length=args.max_length)
        if chat is None:
            val_skipped += 1
            continue
        valid_samples.append(chat)
    if val_skipped:
        print(
            f"⚠️ Skipped {val_skipped} validation examples with zero assistant tokens after truncation"
        )

    train_ds = Dataset.from_list(train_samples)
    valid_ds = Dataset.from_list(valid_samples)

    collator = collate(tokenizer, pad_to=8)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_train_epochs=args.epochs,
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
        logging_steps=args.logging_steps,
        evaluation_strategy="steps",
        eval_steps=args.eval_steps,
        save_steps=args.save_steps,
        bf16=False,
        fp16=torch_dtype == torch.float16,
        gradient_checkpointing=True if torch.cuda.is_available() else False,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        save_total_limit=2,
        ddp_find_unused_parameters=False,
        dataloader_num_workers=0,  # Memory optimization per chronicles
        group_by_length=True,
        optim="paged_adamw_8bit" if torch.cuda.is_available() else "adamw_torch",
        report_to=[],
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=valid_ds,
        data_collator=collator,
        tokenizer=tokenizer,
    )
    trainer.train()
    trainer.save_state()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
