# 🎭 Friday AI Fine-tuning Project - Knowledge Transfer Document

## 📋 **Project Overview**

**Project**: Friday AI Fine-tuning  
**Objective**: Create a personalized AI assistant with Telugu film knowledge, sarcastic humor, and production skills  
**Base Model**: Meta-Llama-3.1-8B-Instruct  
**Fine-tuning Method**: QLoRA (Quantized Low-Rank Adaptation)  
**Infrastructure**: AWS SageMaker  
**Status**: ✅ Successfully trained and deployed

---

## 🏗️ **Technical Architecture**

### **Core Technologies**

- **Base Model**: Meta-Llama-3.1-8B-Instruct (15GB)
- **Fine-tuning**: QLoRA with LoRA rank=16, alpha=32
- **Quantization**: 4-bit BitsAndBytesConfig for memory efficiency
- **Training Framework**: Transformers + PEFT + Accelerate
- **Infrastructure**: AWS SageMaker (ml.g5.12xlarge, ml.g5.2xlarge)
- **Data Format**: ChatML conversation format

### **Memory Optimization Strategy**

```python
# QLoRA Configuration
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

# LoRA Configuration
lora_config = LoraConfig(
    r=16,                    # Low rank
    lora_alpha=32,          # Scaling parameter
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
    lora_dropout=0.1
)
```

---

## 📊 **Training Data & Results**

### **Dataset Composition**

- **Training Samples**: 242 conversations
- **Validation Samples**: 26 conversations
- **Data Sources**:
  - Film production scenarios
  - Telugu cinema references
  - Sarcastic personality examples
  - Personal assistant tasks

### **Final Model Performance**

- **Training Job**: `friday-lora-20250813-233531`
- **Duration**: 652 seconds (11 minutes)
- **Cost**: ~$2-3 per training run
- **Output**: 168MB LoRA adapters
- **Instance**: ml.g5.12xlarge (4x A10G GPUs, 96GB RAM)

---

## 🚧 **Major Challenges & Solutions**

### **Challenge 1: AWS Service Quotas** ⚠️

**Problem**:

```
ERROR: Service quota exceeded for ml.g5.2xlarge instances
Current limit: 0, Requested: 1
```

**Solution**:

- Requested quota increase through AWS Console
- Escalated via support ticket for faster approval
- Alternative: Started with ml.m5.xlarge for debugging

---

### **Challenge 2: Memory Out-of-Memory (OOM)** 💥

**Problem**:

```
RuntimeError: CUDA out of memory. Tried to allocate 2.00 GiB
```

**Root Cause**: Multi-GPU setup was replicating entire model across GPUs instead of sharding

**Solution**: Implemented gradient checkpointing fix

```python
# Critical fix for multi-GPU memory issues
if training_args.gradient_checkpointing:
    model.enable_input_require_grads()  # This was the key!

# Memory optimization
model.config.use_cache = False
```

---

### **Challenge 3: Tokenization Data Type Error** 🐛

**Problem**:

```
RuntimeError: Could not infer dtype of dict
```

**Root Cause**: Batched tokenization created nested lists, `.copy()` method failed

**Solution**: Smart copy handling

```python
# Before (broken)
tokenized["labels"] = tokenized["input_ids"].copy()

# After (fixed)
if isinstance(tokenized["input_ids"][0], list):
    tokenized["labels"] = [ids.copy() for ids in tokenized["input_ids"]]
else:
    tokenized["labels"] = tokenized["input_ids"].copy()
```

---

### **Challenge 4: HuggingFace Authentication** 🔐

**Problem**:

```
GatedRepoError: Cannot access gated repo meta-llama/Meta-Llama-3.1-8B-Instruct
```

**Solution**:

- Obtained HuggingFace access token
- Added to environment variables
- Configured SageMaker environment properly

---

### **Challenge 5: Model Loading Performance** ⚡

**Problem**: 15GB base model too large for local testing

**Solution**:

- QLoRA quantization (15GB → ~4GB in memory)
- LoRA adapters only 168MB
- Base model cached locally, adapters portable

---

## 🛠️ **Key Scripts & Components**

### **Training Pipeline**

1. **`train_multigpu.py`** - Multi-GPU training entry point
2. **`train_memory_diet.py`** - Single GPU memory-optimized version
3. **`vscode_sagemaker_trainer.py`** - Core SageMaker integration

### **Data Processing**

1. **`convert_scene_to_chatml.py`** - Convert scenes to ChatML format
2. **`build_iteration1_dataset.py`** - Create training datasets
3. **`snippets_to_chatml.py`** - Process dialogue snippets

### **Infrastructure**

1. **`setup_aws_interactive.py`** - AWS credential setup
2. **`.env.template`** - Environment configuration template
3. **`requirements.txt`** - Python dependencies

---

## 📈 **Performance Metrics**

### **Hardware Utilization**

- **ml.g5.12xlarge**: 4x A10G (24GB each) = 96GB total
- **GPU Memory**: ~85% utilization during training
- **Training Speed**: ~11 minutes for 1 epoch
- **Cost Efficiency**: $2-3 per training iteration

### **Model Quality Indicators**

- ✅ Telugu film references preserved
- ✅ Sarcastic personality maintained
- ✅ Technical film knowledge intact
- ✅ Assistant capabilities functional
- ✅ Creative writing abilities enhanced

---

## 💰 **Cost Analysis**

| Component               | Instance Type  | Duration | Cost per Run |
| ----------------------- | -------------- | -------- | ------------ |
| Multi-GPU Training      | ml.g5.12xlarge | 11 min   | $2-3         |
| Single GPU Training     | ml.g5.2xlarge  | 25 min   | $0.8-1.2     |
| Data Storage (S3)       | Standard       | Monthly  | $0.10        |
| **Total per iteration** | -              | -        | **$2-3**     |

---

## 🎯 **Friday AI Personality Achieved**

### **Telugu Film Knowledge** 🎬

```
Input: "What's your favorite Telugu movie dialogue?"
Output: "Ah, asking for favorites! That's like asking Rajinikanth to pick his best punchline - impossible! But if I had to choose, I'd go with 'Nenu saitham' from Baahubali. Simple, powerful, and gets the point across without a 10-minute monologue."
```

### **Sarcastic Humor** 😏

```
Input: "My code has 50 bugs. Help me debug."
Output: "Oh, another aspiring developer! Let me guess - you wrote it at 3 AM fueled by coffee and optimism? No worries, I've seen worse. First rule: comments are your friends, not optional decorations."
```

### **Film Production Expertise** 🎥

```
Input: "Help me plan a short film shoot"
Output: "Pre-production is your best friend - plan every shot, every prop, every sandwich for the crew. Trust me, hangry technicians are scarier than any villain you can write!"
```

---

## 🚀 **Deployment Architecture**

### **Current Setup**

- **Base Model**: 15GB Meta-Llama-3.1-8B-Instruct (local cache)
- **LoRA Adapters**: 168MB (models/trained/)
- **Tokenizer**: Complete tokenization components
- **Configuration**: adapter_config.json with r=16, α=32

### **Testing Options**

1. **Local Testing**: Requires 16GB+ GPU or 32GB+ RAM
2. **SageMaker Endpoint**: Real-time inference (~$0.75/hour)
3. **Batch Testing**: SageMaker training job (~$0.50 per test)

---

## 📚 **Key Learning Points**

### **Technical Insights**

1. **QLoRA is memory-efficient** but requires proper gradient checkpointing
2. **Multi-GPU training needs careful memory management** - model replication vs sharding
3. **Tokenization edge cases** can break training pipelines silently
4. **AWS quotas** are the first bottleneck, not technical challenges

### **Process Insights**

1. **Start small** (ml.g5.2xlarge) then scale up (ml.g5.12xlarge)
2. **Memory optimization first**, performance second
3. **Comprehensive error handling** saves debugging time
4. **Environment isolation** prevents configuration conflicts

### **Business Insights**

1. **Cost-effective**: $2-3 per training iteration is very reasonable
2. **Fast iteration**: 11-minute training cycles enable rapid experimentation
3. **Portable results**: 168MB adapters are easy to deploy/share
4. **Quality preservation**: Fine-tuning maintains base model capabilities

---

## 🔧 **Quick Start for Interns**

### **Environment Setup**

```bash
# 1. Clone repository
git clone https://github.com/Poornachand778/friday.git
cd friday

# 2. Create environment
conda create -n friday_ft python=3.11
conda activate friday_ft
pip install -r requirements.txt

# 3. Configure AWS
cp .env.template .env
# Edit .env with your credentials

# 4. Test setup
python scripts/setup_aws_interactive.py
```

### **Run Training**

```bash
# Multi-GPU (recommended)
python scripts/train_multigpu.py

# Single GPU (budget option)
python scripts/train_memory_diet.py
```

### **Monitor Progress**

- AWS Console → SageMaker → Training Jobs
- CloudWatch logs for detailed debugging
- S3 bucket for model artifacts

---

## ⚠️ **Common Pitfalls & Solutions**

| Issue          | Symptom                        | Solution                      |
| -------------- | ------------------------------ | ----------------------------- |
| Quota Exceeded | Training job fails immediately | Request AWS quota increase    |
| OOM Error      | CUDA memory error              | Enable gradient checkpointing |
| Auth Error     | HuggingFace 401                | Set HUGGINGFACE_TOKEN         |
| Slow Training  | >30 min per epoch              | Use ml.g5.12xlarge            |
| Poor Quality   | Generic responses              | Check LoRA adapters loaded    |

---

## 📞 **Support & Resources**

### **Documentation**

- `README.md` - Complete setup guide
- `FINETUNING_CHRONICLES.md` - Detailed problem-solving journey
- AWS SageMaker docs - Instance types and pricing

### **Key Contacts**

- **Project Lead**: [Your name]
- **AWS Support**: For quota/billing issues
- **HuggingFace**: For model access issues

### **Emergency Procedures**

1. **Training fails**: Check CloudWatch logs first
2. **Cost overrun**: Stop all SageMaker jobs immediately
3. **Access issues**: Verify .env file configuration

---

**🎭 Final Note**: "Fine-tuning AI is like directing a Telugu movie - you need patience, creativity, and the ability to fix things when they inevitably break. But when it works, it's pure magic!" ✨

---

_Document prepared by: Friday AI Development Team_  
_Last updated: August 15, 2025_
