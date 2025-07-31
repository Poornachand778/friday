#!/usr/bin/env python
"""
One-file QLoRA trainer (uses HF PEFT) for Llama-3.1-8B-Instruct.
Trains on data/instructions/iteration1_train.jsonl
"""

import os, json, torch, datasets
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from peft import LoraConfig, get_peft_model
from pathlib import Path

base_model = "models/hf/Meta-Llama-3.1-8B-Instruct"
train_path = "data/instructions/iteration1_train.jsonl"
output_dir = "models/adapters/friday-v0.1"

tokenizer = AutoTokenizer.from_pretrained(base_model, use_fast=True)
tokenizer.pad_token = tokenizer.eos_token


def to_chatml(ex):
    return {
        "input_ids": tokenizer(
            f"<|user|>\n{ex['instruction']}\n<|assistant|>", return_tensors="pt"
        ).input_ids.squeeze(),
        "labels": tokenizer(
            ex["response"] + tokenizer.eos_token, return_tensors="pt"
        ).input_ids.squeeze(),
    }


ds = datasets.load_dataset("json", data_files=train_path, split="train")
ds = ds.map(to_chatml, remove_columns=ds.column_names)

model = AutoModelForCausalLM.from_pretrained(
    base_model, torch_dtype=torch.float16, device_map="mps", low_cpu_mem_usage=True
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
    task_type="CAUSAL_LM",
    lora_dropout=0.05,
)
model = get_peft_model(model, peft_cfg)

args = TrainingArguments(
    output_dir=output_dir,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    num_train_epochs=1,
    # bf16=False, fp16=False,
    learning_rate=1e-4,
    logging_steps=10,
    save_strategy="epoch",
)

from transformers import Trainer

trainer = Trainer(model=model, args=args, train_dataset=ds)
trainer.train()
trainer.model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)
print("LoRA adapter saved →", output_dir)
