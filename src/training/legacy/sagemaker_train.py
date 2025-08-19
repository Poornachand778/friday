#!/usr/bin/env python3
"""
SageMaker Training Script for Friday AI LoRA Fine-tuning
"""

import os
import json
import torch
import argparse
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
)

# Enable TF32 for A10G GPU performance boost
if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True

from peft import LoraConfig, get_peft_model, TaskType
from datasets import Dataset


def enable_input_grads(model):
    """
    Ensure the embedding *outputs* require grad so gradient-checkpointed blocks
    have a graph to backprop through when the base weights are frozen (LoRA).
    Uses Transformers' built-in hook when available; otherwise registers a forward hook.
    """
    if hasattr(model, "enable_input_require_grads"):
        # Newer Transformers exposes this for exactly this case
        model.enable_input_require_grads()
    else:
        # Fallback: hook the input embeddings to mark outputs as requiring grad
        emb = model.get_input_embeddings()

        def _out_require_grad(module, inputs, output):
            if isinstance(output, torch.Tensor):
                output.requires_grad_(True)

        emb.register_forward_hook(_out_require_grad)


def print_trainable_summary(model):
    """Print clear summary of trainable vs total parameters."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(
        f"🧮 Trainable params: {trainable:,} / {total:,} "
        f"({100 * trainable / total:.4f}%)"
    )


class CausalLMCollator:
    """Pad chat examples and preserve label masking (-100 on non-assistant tokens)."""

    def __init__(self, tokenizer, pad_to_multiple_of=8, max_length=None):
        self.tokenizer = tokenizer
        self.pad_to_multiple_of = pad_to_multiple_of
        self.max_length = max_length

    def __call__(self, features):
        pad_id = self.tokenizer.pad_token_id or self.tokenizer.eos_token_id

        # Truncate sequences if max_length is specified
        if self.max_length:
            for f in features:
                if len(f["input_ids"]) > self.max_length:
                    f["input_ids"] = f["input_ids"][: self.max_length]
                    f["labels"] = f["labels"][: self.max_length]

        max_len = max(len(f["input_ids"]) for f in features)
        if self.pad_to_multiple_of and max_len % self.pad_to_multiple_of:
            max_len = (
                (max_len // self.pad_to_multiple_of) + 1
            ) * self.pad_to_multiple_of
        input_ids, labels, attn = [], [], []

        # FAIL-FAST CHECK: Ensure we have actual assistant tokens to learn from
        total_non_masked = 0
        total_tokens = 0

        for f in features:
            ids = f["input_ids"]
            labs = f["labels"]
            pad = max_len - len(ids)
            input_ids.append(ids + [pad_id] * pad)
            labels.append(labs + [-100] * pad)
            attn.append([1] * len(ids) + [0] * pad)

            # Count non-masked tokens for validation
            non_masked = sum(1 for x in labs if x != -100)
            total_non_masked += non_masked
            total_tokens += len(labs)

        # CRITICAL CHECK: If all labels are -100, training will fail
        if total_non_masked == 0:
            print("🚨 CRITICAL ERROR: ALL LABELS ARE MASKED (-100)!")
            print(f"   Total tokens in batch: {total_tokens}")
            print(f"   Non-masked tokens: {total_non_masked}")
            print("   This means no assistant tokens to learn from!")
            print(f"   Sample labels: {labels[0][:20] if labels else 'None'}")
            raise ValueError("All labels are -100. No tokens to learn from!")

        print(
            f"✅ Batch validation: {total_non_masked}/{total_tokens} tokens for training ({total_non_masked / total_tokens:.1%})"
        )

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "attention_mask": torch.tensor(attn, dtype=torch.long),
        }


def load_dataset(file_path):
    """Load JSONL dataset"""
    data = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data


def build_chat_example(example, tokenizer):
    """Produce input_ids & labels using tokenizer chat template. Mask non-assistant tokens with -100."""
    msgs = example["messages"]
    full_ids = tokenizer.apply_chat_template(
        msgs, add_generation_prompt=False, tokenize=True, return_tensors=None
    )
    labels = [-100] * len(full_ids)
    prev = 0
    partial = []
    for m in msgs:
        partial.append(m)
        turn_ids = tokenizer.apply_chat_template(
            partial, add_generation_prompt=False, tokenize=True, return_tensors=None
        )
        if m.get("role") == "assistant":
            for i in range(prev, len(turn_ids)):
                labels[i] = full_ids[i]
        prev = len(turn_ids)
    return {"input_ids": full_ids, "labels": labels}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-name", type=str, default="meta-llama/Meta-Llama-3.1-8B-Instruct"
    )
    parser.add_argument(
        "--train-data",
        type=str,
        default="/opt/ml/input/data/training/iteration1_train.labeled.jsonl",
    )
    parser.add_argument(
        "--valid-data",
        type=str,
        default="/opt/ml/input/data/training/iteration1_valid.jsonl",
    )
    parser.add_argument("--output-dir", type=str, default="/opt/ml/model")
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Alias for per-device-train-batch-size",
    )
    parser.add_argument("--per-device-train-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--warmup-steps", type=int, default=50)
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--eval-steps", type=int, default=100)
    parser.add_argument("--save-steps", type=int, default=500)
    parser.add_argument(
        "--max-length",
        type=int,
        default=None,
        help="Maximum sequence length (for memory control)",
    )

    args = parser.parse_args()

    # Use batch-size if provided, otherwise use per-device-train-batch-size
    if hasattr(args, "batch_size") and args.batch_size != 1:
        args.per_device_train_batch_size = args.batch_size

    print("🚀 Starting Friday AI LoRA fine-tuning")
    print(f"   Model: {args.model_name}")
    print(f"   Training data: {args.train_data}")
    print(f"   Validation data: {args.valid_data}")
    print(f"   LoRA rank: {args.lora_rank}, alpha: {args.lora_alpha}")

    # Load tokenizer and model with robust error handling
    print("📥 Loading tokenizer and model...")

    # Get HuggingFace token from environment
    hf_token = os.getenv("HUGGINGFACE_TOKEN")
    if not hf_token:
        print("❌ HUGGINGFACE_TOKEN not found in environment")
        raise ValueError("HuggingFace token required for Meta-Llama models")

    # Check if we should use a different model for compatibility
    model_name = args.model_name
    if "Meta-Llama-3.1-8B-Instruct" in model_name:
        print("🔄 Using Llama 3.1 with specific compatibility settings...")

    # Load tokenizer first
    tokenizer = AutoTokenizer.from_pretrained(
        model_name, trust_remote_code=True, token=hf_token
    )

    # Ensure pad token is set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"✅ Tokenizer loaded: {len(tokenizer)} tokens")

    # Load model with explicit device placement instead of auto mapping
    print("🔄 Loading model...")
    # Multi-GPU memory optimization - use 4-bit quantization for base model
    num_gpus = torch.cuda.device_count()
    print(f"🔍 Detected {num_gpus} GPU(s)")

    if num_gpus > 1:
        print("🚀 Multi-GPU setup detected - using memory optimization")
        # For multi-GPU, use 4-bit quantization to fit model across GPUs
        from transformers import BitsAndBytesConfig

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )

        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            device_map="auto",  # Automatic device placement across GPUs
            trust_remote_code=True,
            token=hf_token,
            use_cache=False,
            low_cpu_mem_usage=True,
        )
        print("✅ Model loaded with 4-bit quantization across GPUs")
    else:
        print("📱 Single GPU setup - using standard loading")
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=(
                torch.bfloat16
                if (torch.cuda.is_available() and torch.cuda.is_bf16_supported())
                else torch.float16
            ),
            trust_remote_code=True,
            token=hf_token,
            use_cache=False,
            low_cpu_mem_usage=True,
        )

        # Move model to GPU if available (explicit placement)
        if torch.cuda.is_available():
            print(f"🔄 Moving model to GPU: {torch.cuda.get_device_name()}")
            model = model.cuda()
            print("✅ Model on GPU")
        else:
            print("⚠️ No GPU available, using CPU")

    # Ensure cache disabled during training (saves memory)
    if hasattr(model, "config"):
        model.config.use_cache = False

    print(f"✅ Model loaded: {model.config.model_type}")

    # Configure LoRA with comprehensive target modules
    print("🔧 Configuring LoRA...")

    # Get all available linear layer names for debugging
    def get_linear_module_names(model):
        """Find all linear layer names in the model"""
        linear_names = set()
        for name, module in model.named_modules():
            if isinstance(module, torch.nn.Linear):
                parts = name.split(".")
                linear_names.add(parts[-1])
        return linear_names

    available_modules = get_linear_module_names(model)
    print(f"📋 Available linear modules: {sorted(available_modules)}")

    # Comprehensive target modules for Llama 3.1
    target_modules = [
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ]

    # Verify target modules exist
    missing_modules = [m for m in target_modules if m not in available_modules]
    if missing_modules:
        print(f"⚠️ Missing target modules: {missing_modules}")
        # Use only available modules
        target_modules = [m for m in target_modules if m in available_modules]

    print(f"📋 Using LoRA target modules: {target_modules}")

    if not target_modules:
        raise RuntimeError("No valid target modules found for LoRA!")

    # Check if model is quantized (4-bit)
    is_quantized = hasattr(model, "hf_quantizer") or any(
        hasattr(module, "weight") and hasattr(module.weight, "quant_state")
        for module in model.modules()
    )

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.1,
        target_modules=target_modules,
        bias="none",
        modules_to_save=None,
        init_lora_weights=True,
    )

    if is_quantized:
        print("🔧 QLoRA setup detected - using optimized configuration")
        # QLoRA-specific optimizations
        lora_config.task_type = TaskType.CAUSAL_LM
    else:
        print("🔧 Standard LoRA setup")

    try:
        print("🔄 Applying LoRA to model...")
        model = get_peft_model(model, lora_config)
        enable_input_grads(model)  # Critical fix for LoRA + gradient checkpointing
        print("✅ LoRA applied successfully!")
        model.print_trainable_parameters()
        print_trainable_summary(model)  # Clear parameter summary

        # FAIL-FAST CHECK 1: Ensure we have trainable parameters
        trainable_params = [p for p in model.parameters() if p.requires_grad]
        if len(trainable_params) == 0:
            print("🚨 CRITICAL ERROR: NO TRAINABLE PARAMETERS!")
            print("   This means LoRA wasn't applied properly.")
            print("   All model weights are frozen - no learning possible!")
            raise RuntimeError(
                "LoRA configuration resulted in zero trainable parameters!"
            )

        total = sum(p.numel() for p in model.parameters())
        trainable_ct = sum(p.numel() for p in trainable_params)
        print(
            f"🧮 Trainable parameter ratio: {trainable_ct / total:.4%} ({trainable_ct:,}/{total:,})"
        )

        # FAIL-FAST CHECK 2: Verify LoRA modules are actually attached
        lora_modules_found = []
        for name, module in model.named_modules():
            if hasattr(module, "lora_A") or hasattr(module, "lora_B"):
                lora_modules_found.append(name)

        if not lora_modules_found:
            print("🚨 CRITICAL ERROR: NO LORA MODULES ATTACHED!")
            print("   LoRA config was created but no modules have lora_A/lora_B!")
            print("   This indicates a module name mismatch.")
            raise RuntimeError("No LoRA modules found in model!")

        print(f"✅ LoRA modules attached: {len(lora_modules_found)}")
        print("🔍 Sample LoRA modules:")
        for name in lora_modules_found[:3]:
            print(f"   {name}")

        # FAIL-FAST CHECK 3: Verify trainable parameters are LoRA parameters
        trainable_lora_params = []
        for name, param in model.named_parameters():
            if param.requires_grad and ("lora_A" in name or "lora_B" in name):
                trainable_lora_params.append(name)

        if not trainable_lora_params:
            print("🚨 CRITICAL ERROR: TRAINABLE PARAMS ARE NOT LORA PARAMS!")
            print("   Model has trainable params but they're not LoRA adapters!")
            print("   This means base model is unfrozen (wrong!).")
            print("   Trainable parameter sample:")
            for name, param in model.named_parameters():
                if param.requires_grad:
                    print(f"     {name}: {param.shape}")
                    break
            raise RuntimeError("Trainable parameters are not LoRA parameters!")

        print(
            f"✅ LoRA parameter validation: {len(trainable_lora_params)} trainable LoRA params"
        )
        print("🔍 Sample trainable LoRA parameters:")
        for name in trainable_lora_params[:3]:
            print(f"   {name}")

    except Exception as e:
        print(f"❌ Error applying LoRA: {e}")
        print("📋 Available modules in model:")
        for name, module in model.named_modules():
            if hasattr(module, "weight"):
                print(f"   {name}: {module.weight.shape}")
        raise

    # Load and process datasets
    print("📊 Loading training datasets...")
    train_data = load_dataset(args.train_data)
    valid_data = load_dataset(args.valid_data)

    print(f"   Training samples: {len(train_data)}")
    print(f"   Validation samples: {len(valid_data)}")

    # Build chat-format datasets directly to input_ids & labels
    train_dataset = Dataset.from_list(train_data).map(
        lambda ex: build_chat_example(ex, tokenizer)
    )
    valid_dataset = Dataset.from_list(valid_data).map(
        lambda ex: build_chat_example(ex, tokenizer)
    )

    lens = [len(r) for r in train_dataset["input_ids"][:50]]
    if lens:
        print(
            f"📏 Chat length stats (first 50): min={min(lens)}, max={max(lens)}, avg={sum(lens) / len(lens):.1f}"
        )
    assist_tokens = (
        len([x for x in train_dataset[0]["labels"] if x != -100])
        if len(train_dataset)
        else 0
    )
    print("🧪 Sample assistant token count (sample 0):", assist_tokens)

    # FAIL-FAST CHECK 4: Comprehensive label validation
    def _validate_labels_comprehensive(ds, name):
        print(f"🔍 Validating {name} labels...")
        total_examples = len(ds)
        total_tokens = 0
        total_assistant_tokens = 0
        examples_with_no_assistant = []

        for i in range(min(10, total_examples)):  # Check first 10 examples
            lbl = ds[i]["labels"]
            example_tokens = len(lbl)
            example_assistant = sum(1 for x in lbl if x != -100)

            total_tokens += example_tokens
            total_assistant_tokens += example_assistant

            if example_assistant == 0:
                examples_with_no_assistant.append(i)
                print(f"🚨 Example {i}: ALL {example_tokens} tokens are masked!")
                print(f"   Input IDs: {ds[i]['input_ids'][:20]}...")
                print(f"   Labels: {lbl[:20]}...")

            else:
                print(
                    f"✅ Example {i}: {example_assistant}/{example_tokens} assistant tokens ({example_assistant / example_tokens:.1%})"
                )

        if examples_with_no_assistant:
            print(
                f"� CRITICAL ERROR: {len(examples_with_no_assistant)} examples have NO ASSISTANT TOKENS!"
            )
            print(f"   Examples with all-masked labels: {examples_with_no_assistant}")
            print("   This will cause 'no grad' error during training!")
            raise ValueError(f"{name} dataset has examples with all labels = -100")

        print(
            f"✅ {name} validation: {total_assistant_tokens}/{total_tokens} total assistant tokens ({total_assistant_tokens / total_tokens:.1%})"
        )

        if total_assistant_tokens == 0:
            print(f"🚨 CRITICAL ERROR: ENTIRE {name} DATASET HAS NO ASSISTANT TOKENS!")
            raise ValueError(f"All labels in {name} dataset are -100!")

        return total_assistant_tokens, total_tokens

    train_assist, train_total = _validate_labels_comprehensive(
        train_dataset, "TRAINING"
    )
    valid_assist, valid_total = _validate_labels_comprehensive(
        valid_dataset, "VALIDATION"
    )

    print("📊 Dataset Summary:")
    print(
        f"   Training: {train_assist:,} assistant tokens / {train_total:,} total ({train_assist / train_total:.2%})"
    )
    print(
        f"   Validation: {valid_assist:,} assistant tokens / {valid_total:,} total ({valid_assist / valid_total:.2%})"
    )

    if train_assist / train_total < 0.01:  # Less than 1% assistant tokens
        print(
            f"⚠️ WARNING: Very low assistant token ratio ({train_assist / train_total:.2%})"
        )
        print("   This might indicate chat template issues!")

    print("✅ Comprehensive label validation passed")

    # Detect multi-GPU setup for memory optimization
    num_gpus = torch.cuda.device_count()

    # Memory-optimized training arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps
        * (4 if num_gpus > 1 else 1),  # Increase for multi-GPU
        num_train_epochs=args.epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_steps=args.warmup_steps,
        logging_steps=args.logging_steps,
        eval_steps=args.eval_steps,
        save_steps=args.save_steps,
        evaluation_strategy="steps" if args.eval_steps > 0 else "no",
        save_strategy="steps",
        tf32=True,
        bf16=torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
        fp16=not (torch.cuda.is_available() and torch.cuda.is_bf16_supported()),
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={
            "use_reentrant": False
        },  # Non-reentrant for LoRA compatibility
        report_to=None,
        remove_unused_columns=False,
        dataloader_pin_memory=False,
        save_total_limit=2,
        # Additional memory optimizations for multi-GPU
        max_grad_norm=1.0,
        ddp_find_unused_parameters=False if num_gpus > 1 else None,
        dataloader_num_workers=0,  # Reduce memory overhead
    )

    print(
        f"🔧 Training config: {num_gpus} GPU(s), batch_size={args.per_device_train_batch_size}, grad_accum={training_args.gradient_accumulation_steps}"
    )

    # Disable cache for gradient checkpointing compatibility
    model.config.use_cache = False

    # Data collator with optional max length for memory control
    data_collator = CausalLMCollator(tokenizer, max_length=args.max_length)

    # FAIL-FAST CHECK 5: Test gradient flow with dummy batch
    print("🧪 Testing gradient flow with dummy batch...")
    try:
        # Get a small sample to test
        test_sample = [train_dataset[0]]  # Single example
        test_batch = data_collator(test_sample)

        # Move to same device as model
        device = next(model.parameters()).device
        test_batch = {
            k: v.to(device) if isinstance(v, torch.Tensor) else v
            for k, v in test_batch.items()
        }

        # Ensure input tensors can propagate gradients
        for key in ["input_ids", "attention_mask", "labels"]:
            if key in test_batch:
                test_batch[key] = (
                    test_batch[key].detach().requires_grad_(False)
                )  # Input data should NOT require grad

        print(f"🔍 Input device: {test_batch['input_ids'].device}")
        print(f"🔍 Model device: {device}")
        print(f"🔍 Model parameters device: {next(model.parameters()).device}")

        # Verify model parameters are on correct device and require grad
        lora_param_count = 0
        for name, param in model.named_parameters():
            if "lora_" in name and param.requires_grad:
                lora_param_count += 1
                if lora_param_count == 1:  # Show first one
                    print(f"🔍 Sample LoRA param {name}:")
                    print(f"   Device: {param.device}")
                    print(f"   Requires grad: {param.requires_grad}")
                    print(f"   Dtype: {param.dtype}")

        print(f"🔍 Total LoRA params requiring grad: {lora_param_count}")

        # Test forward pass first (no grad for speed)
        model.eval()
        with torch.no_grad():
            outputs = model(**test_batch)
            loss = outputs.loss
            print(f"✅ Forward pass works, loss: {loss.item():.4f}")

        # Test gradient computation - CRITICAL for LoRA + checkpointing
        print("🧪 Testing gradient computation with LoRA + checkpointing...")
        model.train()  # Must be in training mode

        # Explicitly enable gradients and test
        with torch.enable_grad():
            outputs = model(**test_batch)
            loss = outputs.loss

            print(f"✅ Training mode loss: {loss.item():.4f}")
            print(f"✅ Loss requires grad: {loss.requires_grad}")
            print(f"✅ Loss grad_fn: {loss.grad_fn}")

            if not loss.requires_grad:
                print("🚨 CRITICAL ERROR: LOSS DOESN'T REQUIRE GRADIENTS!")
                print(f"   Loss: {loss}")
                print(f"   Loss.requires_grad: {loss.requires_grad}")
                print(f"   Loss.grad_fn: {loss.grad_fn}")

                # Debug model state
                print("🔍 Model training mode:", model.training)
                print("🔍 Checking embedding layer...")
                if hasattr(model, "model") and hasattr(model.model, "embed_tokens"):
                    # Check embedding parameters
                    embed_param_grad = model.model.embed_tokens.weight.requires_grad
                    print(f"   Embedding params require_grad: {embed_param_grad}")

                    # Test embedding output
                    embed_out = model.model.embed_tokens(test_batch["input_ids"])
                    print(
                        f"   Embedding output requires_grad: {embed_out.requires_grad}"
                    )
                else:
                    print("   Could not find embedding layer")

                print("🔍 Checking LoRA parameter states:")
                lora_count = 0
                for name, param in model.named_parameters():
                    if "lora_" in name and param.requires_grad:
                        lora_count += 1
                        if lora_count <= 3:
                            print(f"   {name}: requires_grad={param.requires_grad}")
                print(f"   Total LoRA params requiring grad: {lora_count}")

                raise RuntimeError("Loss tensor doesn't require gradients!")

            # Test backward pass
            loss.backward()

            # Check if any LoRA parameters got gradients
            grad_params = []
            for name, param in model.named_parameters():
                if param.requires_grad and param.grad is not None:
                    grad_params.append(name)

            if grad_params:
                print(
                    f"✅ Gradient flow test PASSED: {len(grad_params)} params got gradients"
                )
                print(f"   Sample params with gradients: {grad_params[:3]}")
                print("✅ All gradient flow checks passed!")
            else:
                print("🚨 CRITICAL ERROR: NO PARAMETERS RECEIVED GRADIENTS!")
                print("   This means gradient computation failed!")
                print("   Checking trainable parameters...")
                for name, param in model.named_parameters():
                    if param.requires_grad:
                        print(
                            f"     {name}: requires_grad={param.requires_grad}, grad={param.grad}"
                        )
                        break
                raise RuntimeError(
                    "Gradient flow test failed - no parameters received gradients!"
                )

    except Exception as e:
        print(f"🚨 GRADIENT FLOW TEST FAILED: {e}")
        print("   This indicates the model/data setup has issues!")
        raise

    # Sanity check: test gradient flow (should NOT raise after the fix)
    print("🔎 Running sanity check for gradient flow...")
    from torch.utils.data import DataLoader

    sanity_loader = DataLoader(
        train_dataset, batch_size=1, shuffle=False, collate_fn=data_collator
    )
    batch = next(iter(sanity_loader))
    batch = {k: v.to(model.device) for k, v in batch.items()}

    model.train()
    out = model(**batch)
    print("🔎 Sanity forward loss:", float(out.loss))
    print(f"🔎 Loss requires_grad: {out.loss.requires_grad}")
    print(f"🔎 Loss grad_fn: {out.loss.grad_fn}")

    if not out.loss.requires_grad:
        print(
            "🚨 CRITICAL: Loss doesn't require gradients! Check LoRA + checkpointing setup."
        )
        raise RuntimeError("Sanity check failed: loss doesn't require gradients")

    out.loss.backward()  # Should NOT raise now
    model.zero_grad(set_to_none=True)
    print("✅ Sanity check passed: gradient flow works correctly!")

    print("✅ All checks passed! Model ready for training.")

    # Trainer
    print("🏋️ Initializing trainer...")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        data_collator=data_collator,
    )

    # Train
    print("🎯 Starting training...")
    trainer.train()

    # Save model
    print("💾 Saving fine-tuned model...")
    trainer.save_model()
    tokenizer.save_pretrained(args.output_dir)

    # Save training info
    training_info = {
        "model_name": args.model_name,
        "lora_rank": args.lora_rank,
        "lora_alpha": args.lora_alpha,
        "epochs": args.epochs,
        "learning_rate": args.learning_rate,
        "training_samples": len(train_data),
        "validation_samples": len(valid_data),
    }

    with open(os.path.join(args.output_dir, "training_info.json"), "w") as f:
        json.dump(training_info, f, indent=2)

    print("✅ Training completed successfully!")
    print(f"📦 Model saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
