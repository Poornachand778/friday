# 🤖 Friday AI - Personal JARVIS Fine-tuning System

> **Complete end-to-end LoRA fine-tuning pipeline for Meta-Llama-3.1-8B-Instruct on AWS SageMaker**

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![AWS SageMaker](https://img.shields.io/badge/AWS-SageMaker-orange.svg)](https://aws.amazon.com/sagemaker/)
[![Meta Llama 3.1](https://img.shields.io/badge/Model-Llama--3.1--8B-green.svg)](https://huggingface.co/meta-llama/Meta-Llama-3.1-8B-Instruct)

## 🚀 Quick Start (5 Minutes to Training)

### Prerequisites

- **AWS Account** with SageMaker access
- **Python 3.8+**
- **HuggingFace Account** (for Llama access)
- **16GB+ RAM** (for local development)

### 1. Clone & Setup

```bash
git clone <repository-url>
cd Friday
python -m venv friday_ft
source friday_ft/bin/activate  # On Windows: friday_ft\Scripts\activate
pip install -r requirements.txt
```

### 2. Configure AWS & Secrets

```bash
# Configure AWS CLI
aws configure

# Copy environment template and add your secrets
cp .env.template .env
# Edit .env with your actual credentials (NEVER commit this file!)

# Required secrets to add to .env:
# - AWS_ACCESS_KEY_ID: Your AWS access key
# - AWS_SECRET_ACCESS_KEY: Your AWS secret key
# - SAGEMAKER_ROLE: Your SageMaker execution role ARN
# - HUGGINGFACE_TOKEN: Your HuggingFace access token
# - S3_BUCKET: Your S3 bucket name
```

**🔒 Security Note:** Never commit `.env` files with real secrets to git!

### 3. Start Training

```bash
# Multi-GPU training (4x A10G GPUs)
python scripts/train_multigpu.py

# OR single GPU (memory optimized)
python scripts/train_memory_diet.py
```

**That's it!** Training starts immediately and takes ~4-5 minutes for 1 epoch.

---

## 📁 Project Structure

```
Friday/
├── scripts/
│   ├── train_multigpu.py          # 🚀 Multi-GPU training entry point
│   ├── train_memory_diet.py       # 🥗 Single GPU memory-optimized
│   ├── vscode_sagemaker_trainer.py # 🎯 Core training orchestrator
│   └── train/
│       └── sagemaker_train.py     # 🔧 SageMaker training script (the engine)
├── data/
│   └── instructions/
│       ├── iteration1_train.labeled.jsonl  # 📚 Proven training data
│       └── iteration1_valid.jsonl          # ✅ Validation set
├── models/
│   └── hf/Meta-Llama-3.1-8B-Instruct/     # 🤖 Base model
├── requirements.txt               # 📦 Python dependencies
├── .env.template                 # 🔐 Configuration template
└── FINETUNING_CHRONICLES.md      # 📖 Technical journey documentation
```

---

## ⚙️ Configuration

### Required Environment Variables (`.env`)

```bash
# AWS Credentials
AWS_ACCESS_KEY_ID=your_access_key
AWS_SECRET_ACCESS_KEY=your_secret_key
AWS_DEFAULT_REGION=us-east-1

# SageMaker IAM Role (create one with SageMaker permissions)
SAGEMAKER_ROLE=arn:aws:iam::YOUR_ACCOUNT:role/service-role/AmazonSageMaker-ExecutionRole

# HuggingFace Token (get from: https://huggingface.co/settings/tokens)
HUGGINGFACE_TOKEN=hf_your_token_here

# S3 Storage
S3_BUCKET=your-training-bucket
S3_PREFIX=friday-finetuning
```

### AWS Setup Checklist

1. **SageMaker Role**: Create IAM role with these policies:

   - `AmazonSageMakerFullAccess`
   - `AmazonS3FullAccess` (or specific bucket access)

2. **Instance Quotas**: Request limits for:

   - `ml.g5.2xlarge` (single GPU): 1 instance
   - `ml.g5.12xlarge` (4x GPU): 1 instance

3. **HuggingFace Access**: Accept Meta Llama license at [meta-llama/Meta-Llama-3.1-8B-Instruct](https://huggingface.co/meta-llama/Meta-Llama-3.1-8B-Instruct)

---

## 🎯 Training Options

### Option 1: Multi-GPU (Recommended)

```bash
python scripts/train_multigpu.py
```

- **Instance**: ml.g5.12xlarge (4x A10G GPUs, 96GB total)
- **Batch Size**: 8 effective (2 per device)
- **Training Time**: ~4 minutes for 1 epoch
- **Cost**: ~$2-3 for full training

### Option 2: Single GPU (Budget)

```bash
python scripts/train_memory_diet.py
```

- **Instance**: ml.g5.2xlarge (1x A10G GPU, 24GB)
- **Batch Size**: 1
- **Training Time**: ~8-10 minutes for 1 epoch
- **Cost**: ~$1 for full training

### Option 3: Custom Training

```python
from scripts.vscode_sagemaker_trainer import VSCodeSageMakerTrainer

trainer = VSCodeSageMakerTrainer()
s3_inputs = trainer.upload_data()

estimator, job_name = trainer.create_training_job(
    s3_inputs,
    epochs=2,
    batch_size=4,
    learning_rate=1e-4,
    instance_type="ml.g5.2xlarge"
)

trainer.monitor_training(estimator, job_name)
```

---

## 🧠 Model & Architecture

- **Base Model**: Meta-Llama-3.1-8B-Instruct
- **Fine-tuning**: QLoRA (4-bit quantization + LoRA adapters)
- **LoRA Config**: r=16, α=32, dropout=0.05
- **Target Modules**: All attention & MLP layers
- **Gradient Fix**: `enable_input_grads()` for LoRA+checkpointing compatibility

### Key Technical Innovations

- **Fixed gradient checkpointing** with LoRA (major breakthrough!)
- **Smart memory management** with device_map="auto"
- **Automatic padding** and sequence truncation
- **Comprehensive validation** with fail-fast checks

---

## 📊 Training Data

### Current Dataset (`iteration1_train.labeled.jsonl`)

- **Size**: 240+ examples
- **Format**: ChatML (system/user/assistant)
- **Domains**:
  - Telugu film production knowledge
  - Dialogue snippets with humor/sarcasm
  - Scene summarization
  - Personal assistant responses

### Data Sources

- `data/film/snippets/unplaced_dialogues.md` → Dialogue training
- `data/clean_chunks/film/scenes/` → Scene understanding
- `data/persona/` → Personal assistant behavior

---

## 🔧 Troubleshooting

### Common Issues

**1. "Service quota exceeded"**

```bash
# Request quota increase in AWS Console
# Go to: Service Quotas → Amazon SageMaker → Instance quotas
```

**2. "Module not found" errors**

```bash
pip install -r requirements.txt
```

**3. "Access denied" to model**

```bash
# 1. Accept license at HuggingFace
# 2. Check HUGGINGFACE_TOKEN in .env
```

**4. Training fails with OOM**

```bash
# Use memory diet version
python scripts/train_memory_diet.py
```

**5. "All labels are -100" error**

```bash
# Check data format - this is handled automatically now
# Our latest code includes comprehensive validation
```

---

## 📈 Monitoring Training

### Real-time Monitoring

The training scripts automatically:

- Upload data to S3
- Start SageMaker job
- Stream logs in real-time
- Download trained model when complete

### Key Metrics to Watch

- **Loss**: Should decrease from ~3.0 → ~1.6
- **Memory**: Should stay under GPU limits
- **Training Speed**: ~30-40 seconds per 10 steps

### Successful Training Example

```
Episode 1/1: 100%|████████| 30/30 [03:37<00:00]
Train Loss: 1.634
✅ Training completed successfully!
📥 Downloading model artifacts...
```

---

## 🎪 Production Deployment

### Download Trained Model

```bash
# Models are auto-downloaded to: models/trained/
# Use with transformers + PEFT for inference
```

### Local Inference Example

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# Load base model
model = AutoModelForCausalLM.from_pretrained("meta-llama/Meta-Llama-3.1-8B-Instruct")
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3.1-8B-Instruct")

# Load LoRA adapters
model = PeftModel.from_pretrained(model, "models/trained/friday-lora-XXXXXX")

# Chat with Friday
messages = [
    {"role": "system", "content": "You are Friday, a witty Telugu-English film assistant."},
    {"role": "user", "content": "What's your favorite dialogue from Telugu cinema?"}
]

inputs = tokenizer.apply_chat_template(messages, return_tensors="pt")
outputs = model.generate(inputs, max_new_tokens=100)
response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(response)
```

---

## 🎯 Success Metrics

Our fine-tuning system has achieved:

- ✅ **100% Success Rate** with gradient checkpointing fix
- ✅ **Sub-5 minute training** on multi-GPU
- ✅ **Stable loss reduction** (3.0 → 1.6)
- ✅ **Zero OOM errors** with proper memory management
- ✅ **Production-ready pipeline** with full automation

---

## 📚 Documentation

- **[FINETUNING_CHRONICLES.md](FINETUNING_CHRONICLES.md)**: Complete technical journey with 16 episodes of failures, breakthroughs, and final success
- **[docs/SAGEMAKER_GUIDE.md](docs/SAGEMAKER_GUIDE.md)**: Detailed SageMaker configuration guide
- **[docs/domain_matrix.md](docs/domain_matrix.md)**: Training domains and data structure

---

## 🤝 Contributing

This is a personal JARVIS project, but the training pipeline is production-ready and can be adapted for other use cases.

### Key Scripts to Understand

1. `scripts/train_multigpu.py` - Entry point for training
2. `scripts/vscode_sagemaker_trainer.py` - Core orchestration
3. `scripts/train/sagemaker_train.py` - The actual training engine

---

## 📄 License

Personal use project. Training pipeline architecture can be adapted with attribution.

---

## 🎬 Credits

Built with determination, Telugu cinema inspiration, and way too much coffee. Special thanks to the Meta AI team for Llama 3.1 and the HuggingFace team for making fine-tuning accessible.

**"Daddy's home!"** - Friday AI's wake phrase 🚀
