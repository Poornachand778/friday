#!/usr/bin/env python
"""
One-file QLoRA trainer (uses HF PEFT) for Llama-3.1-8B-Instruct.
Trains on data/instructions/iteration1_train.jsonl
"""

import torch
import datasets
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model, TaskType


base_model = "models/hf/Meta-Llama-3.1-8B-Instruct"
train_path = "data/instructions/iteration1_train.jsonl"
output_dir = "models/adapters/friday-v0.1"

tokenizer = AutoTokenizer.from_pretrained(base_model, use_fast=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
PAD_ID = tokenizer.pad_token_id


def to_chatml(ex):
    # Build full chat with user + assistant for Llama 3.1 template
    messages = [
        {"role": "user", "content": ex["instruction"]},
        {"role": "assistant", "content": ex["response"]},
    ]
    full_ids = tokenizer.apply_chat_template(
        messages, add_generation_prompt=False, tokenize=True, return_tensors=None
    )
    # Prompt portion (mask out of loss): user only plus assistant header
    prompt_ids = tokenizer.apply_chat_template(
        messages[:-1], add_generation_prompt=True, tokenize=True, return_tensors=None
    )
    labels = full_ids.copy()
    for i in range(len(prompt_ids)):
        labels[i] = -100
    return {"input_ids": full_ids, "labels": labels}


ds = datasets.load_dataset("json", data_files=train_path, split="train")
ds = ds.map(to_chatml, remove_columns=ds.column_names)

# Device / dtype selection
use_cuda = torch.cuda.is_available()
dtype = (
    torch.bfloat16
    if (use_cuda and torch.cuda.is_bf16_supported())
    else (torch.float16 if use_cuda else torch.float32)
)
device_map = (
    "auto" if use_cuda else ("mps" if torch.backends.mps.is_available() else "cpu")
)

model = AutoModelForCausalLM.from_pretrained(
    base_model, torch_dtype=dtype, device_map=device_map, low_cpu_mem_usage=True
)

peft_cfg = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ],
    bias="none",
    task_type=TaskType.CAUSAL_LM,
    lora_dropout=0.05,
)
model = get_peft_model(model, peft_cfg)

# Sanity: ensure adapters attached
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
total = sum(p.numel() for p in model.parameters())
print(f"Trainable params: {trainable:,}/{total:,}")
assert trainable > 0, "No trainable params — check target_modules."


class CausalCollator:
    def __init__(self, tokenizer, pad_id):
        self.tok = tokenizer
        self.pad_id = pad_id

    def __call__(self, batch):
        max_len = max(len(x["input_ids"]) for x in batch)
        input_ids, labels, attn = [], [], []
        for ex in batch:
            ids, labs = ex["input_ids"], ex["labels"]
            pad = max_len - len(ids)
            input_ids.append(ids + [self.pad_id] * pad)
            labels.append(labs + [-100] * pad)
            attn.append([1] * len(ids) + [0] * pad)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "attention_mask": torch.tensor(attn, dtype=torch.long),
        }


collator = CausalCollator(tokenizer, PAD_ID)

args = TrainingArguments(
    output_dir=output_dir,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8 if use_cuda else 1,
    num_train_epochs=1,
    learning_rate=1e-4,
    logging_steps=10,
    save_strategy="epoch",
    bf16=(dtype == torch.bfloat16),
    fp16=(dtype == torch.float16),
    gradient_checkpointing=True if use_cuda else False,
)

trainer = Trainer(model=model, args=args, train_dataset=ds, data_collator=collator)
trainer.train()
trainer.model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)
print("LoRA adapter saved →", output_dir)
