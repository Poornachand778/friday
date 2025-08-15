# Friday AI - SageMaker LoRA Fine-tuning Guide

## Overview

This guide walks you through fine-tuning your Friday AI assistant using LoRA (Low-Rank Adaptation) on AWS SageMaker. The total cost should be under $5 for training.

## Prerequisites

- AWS Account with SageMaker access
- AWS CLI configured or IAM role with proper permissions
- Your training data (iteration1_train.labeled.jsonl)

## Step 1: AWS Setup

### 1.1 Create IAM Role

Create an IAM role with these policies:

- `AmazonSageMakerFullAccess`
- `AmazonS3FullAccess`

### 1.2 Create S3 Bucket

```bash
aws s3 mb s3://friday-ai-training-<your-unique-suffix> --region us-east-1
```

## Step 2: Upload Training Data

### 2.1 Install AWS CLI (if not done)

```bash
pip install awscli boto3
aws configure  # Enter your credentials
```

### 2.2 Upload Your Data

```bash
# Upload training data
aws s3 cp data/instructions/iteration1_train.labeled.jsonl s3://your-bucket/friday-finetuning/data/train.jsonl
aws s3 cp data/instructions/iteration1_valid.jsonl s3://your-bucket/friday-finetuning/data/valid.jsonl

# Verify upload
aws s3 ls s3://your-bucket/friday-finetuning/data/
```

## Step 3: Launch SageMaker Studio

### 3.1 Open SageMaker Console

1. Go to AWS Console → SageMaker
2. Create a Domain (if first time)
3. Launch Studio

### 3.2 Create Notebook Instance (Alternative)

If you prefer notebook instances:

1. SageMaker → Notebook instances → Create
2. Instance type: `ml.t3.medium` (cheap for setup)
3. IAM role: Use the role created in Step 1.1

## Step 4: Training Configuration

### 4.1 Expected Costs

| Resource                 | Cost/Hour      | Expected Duration | Total Cost   |
| ------------------------ | -------------- | ----------------- | ------------ |
| ml.g5.2xlarge (training) | $1.50          | 1-2 hours         | $2-3         |
| ml.t3.medium (notebook)  | $0.05          | Setup time        | ~$0.50       |
| S3 Storage               | $0.02/GB/month | Model storage     | ~$0.50/month |
| **Total Training Cost**  |                |                   | **~$3-4**    |

### 4.2 Instance Types Comparison

- **ml.g5.2xlarge**: Recommended (1 GPU, faster training)
- **ml.g5.xlarge**: Budget option (1 GPU, slower)
- **ml.p3.2xlarge**: Alternative (1 V100 GPU)

## Step 5: Training Script

The notebook contains a complete training script with:

- **LoRA Configuration**: rank=16, alpha=32 (efficient training)
- **Batch Size**: 2 (memory optimized)
- **Learning Rate**: 2e-4 (stable for fine-tuning)
- **Epochs**: 3 (prevents overfitting)

## Step 6: Run Training

### 6.1 Open the Notebook

Upload `friday_sagemaker_training.ipynb` to SageMaker Studio

### 6.2 Update Configuration

```python
BUCKET_NAME = "your-bucket-name-here"  # Change this!
REGION = "us-east-1"  # Your preferred region
```

### 6.3 Execute Cells

Run each cell in sequence. The training will:

1. Verify data upload
2. Create training script
3. Start SageMaker job
4. Monitor progress

### 6.4 Monitor Training

- SageMaker Console → Training jobs
- Watch logs and metrics
- Training should complete in 1-2 hours

## Step 7: Download Trained Model

### 7.1 After Training Completes

```bash
# Download model artifacts
aws s3 sync s3://your-bucket/friday-finetuning/output/friday-lora-*/output/model.tar.gz ./models/
```

### 7.2 Extract Model

```bash
cd models
tar -xzf model.tar.gz
```

## Step 8: Test Locally (Optional)

### 8.1 Load Trained Model

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# Load base model
base_model = AutoModelForCausalLM.from_pretrained("meta-llama/Meta-Llama-3.1-8B-Instruct")
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3.1-8B-Instruct")

# Load LoRA weights
model = PeftModel.from_pretrained(base_model, "./models/")

# Test
prompt = "Define 'serendipity' crisply and give 2 witty examples (one Telugu-flavored)."
inputs = tokenizer(prompt, return_tensors="pt")
outputs = model.generate(**inputs, max_length=200)
print(tokenizer.decode(outputs[0]))
```

## Step 9: Deploy (Optional)

### 9.1 Real-time Endpoint

- Cost: ~$0.75/hour (ml.g5.xlarge)
- Use for production applications

### 9.2 Serverless Inference

- Cost: Pay per request
- Better for low-traffic scenarios

## Troubleshooting

### Common Issues

1. **Permission Errors**: Check IAM role has SageMaker + S3 permissions
2. **Data Not Found**: Verify S3 upload paths match training script
3. **Memory Errors**: Reduce batch size or use larger instance
4. **Long Training**: Normal for 240+ examples, should complete in 1-2 hours

### Cost Optimization

- Use Spot instances (50% discount, might be interrupted)
- Stop training early if loss converges
- Delete endpoints when not in use

## Expected Results

After training, your Friday model should:

- ✅ Generate definitions with Telugu-English mix
- ✅ Create witty examples in your voice
- ✅ Continue film scenes naturally
- ✅ Provide decision advice in your style

## Next Steps

1. **Test thoroughly** with various prompts
2. **Compare** with base model performance
3. **Deploy** to production endpoint
4. **Iterate** with more training data if needed

## Support

If you encounter issues:

1. Check CloudWatch logs in SageMaker console
2. Verify S3 permissions and data paths
3. Monitor GPU memory usage during training
4. Adjust hyperparameters if needed

**Total estimated cost: $3-5 for complete training pipeline** 🎯
