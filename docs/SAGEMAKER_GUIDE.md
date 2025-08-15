# Friday AI + VS Code + SageMaker: Complete Guide

Based on your setup, here are **three excellent ways** to run SageMaker from VS Code:

## 🎯 **Your Best Options**

### **Option 1: VS Code Local → SageMaker Cloud (RECOMMENDED)**

✅ **What you get:**

- Write and debug code locally in VS Code
- Submit training jobs to AWS SageMaker cloud
- Monitor progress in VS Code terminal
- Keep costs low ($3-5 per training)

### **Option 2: SageMaker Studio Code**

✅ **What you get:**

- VS Code running in AWS cloud
- Direct access to all AWS services
- No local resource limitations

### **Option 3: Remote SSH to SageMaker Instance**

✅ **What you get:**

- Connect VS Code to dedicated SageMaker instance
- Full control over environment

---

## 🚀 **QUICK START: Option 1 (Recommended)**

### **Step 1: AWS Credentials Setup**

You have several options for AWS credentials:

```bash
# Option A: AWS CLI (Simplest)
aws configure
# Enter: Access Key, Secret Key, Region (us-east-1), Format (json)

# Option B: Environment Variables
export AWS_ACCESS_KEY_ID="your-access-key"
export AWS_SECRET_ACCESS_KEY="your-secret-key"
export AWS_DEFAULT_REGION="us-east-1"

# Option C: Use .env file (Copy .env.template)
cp .env.template .env
# Edit .env with your AWS credentials
```

### **Step 2: Test AWS Connection**

```bash
# Test AWS connection
aws sts get-caller-identity

# Test SageMaker
python -c "import boto3; print('Region:', boto3.Session().region_name)"
```

### **Step 3: Run Training from VS Code**

```bash
# Submit training job to SageMaker
python scripts/vscode_sagemaker_trainer.py

# Monitor in VS Code terminal
# Training runs in AWS cloud (not local Mac)
```

---

## 🔧 **AWS Credentials - Multiple Methods**

### **Method 1: AWS CLI (Recommended)**

```bash
# Install AWS CLI (already done)
aws configure

# You'll be prompted for:
AWS Access Key ID: AKIA...
AWS Secret Access Key: your-secret-key
Default region name: us-east-1
Default output format: json
```

### **Method 2: VS Code Settings**

Your `.vscode/settings.json` already has AWS configuration:

```json
{
  "aws.profile": "default",
  "aws.region": "us-east-1",
  "python.envFile": "${workspaceFolder}/.env"
}
```

### **Method 3: Environment File**

Create `.env` from template:

```bash
cp .env.template .env
# Edit .env with your AWS credentials
```

### **Method 4: AWS Toolkit Extension**

- Use the AWS Toolkit extension (already installed)
- Sign in through VS Code interface
- Manage profiles visually

---

## 💰 **Cost Structure**

### **Training Costs (AWS SageMaker)**

- **Instance**: ml.g5.2xlarge = $1.50/hour
- **Training time**: 1-2 hours
- **Storage**: ~$0.10/month
- **Total per training**: $2-4

### **Development Costs (FREE)**

- VS Code: Free
- Local development: Free
- AWS Toolkit: Free
- Small data transfers: Nearly free

---

## 🔍 **How It Works**

### **Your Development Workflow:**

1. **Write code** in VS Code locally
2. **Debug** with local Python environment
3. **Submit jobs** to SageMaker cloud
4. **Monitor** from VS Code terminal
5. **Download results** when complete

### **What Runs Where:**

- **VS Code**: Your Mac (local)
- **Training**: AWS SageMaker (cloud)
- **Model storage**: AWS S3 (cloud)
- **Monitoring**: VS Code + AWS Console

### **Data Flow:**

```
Local VS Code → Upload data to S3 → SageMaker training → Model in S3 → Download to local
```

---

## 🚀 **Next Steps**

### **Immediate Actions:**

1. **Configure AWS credentials** (choose one method above)
2. **Test connection**: `aws sts get-caller-identity`
3. **Run training**: `python scripts/vscode_sagemaker_trainer.py`

### **Alternative: SageMaker Studio Code**

If you prefer cloud-based development:

1. Go to AWS SageMaker Console
2. Create SageMaker Studio Domain
3. Launch Studio with Code Editor
4. Upload your project files

---

## 🔧 **Troubleshooting**

### **Common Issues:**

**"Credentials not found"**

```bash
# Check credentials
aws configure list
aws sts get-caller-identity
```

**"SageMaker role not found"**

```bash
# Create SageMaker role
aws iam create-role --role-name SageMakerExecutionRole --assume-role-policy-document file://trust-policy.json
```

**"S3 bucket doesn't exist"**

```bash
# Create bucket
aws s3 mb s3://friday-ai-training --region us-east-1
```

---

## 📊 **Monitoring Options**

### **In VS Code:**

- Terminal output from training script
- AWS Toolkit extension views
- Integrated terminal commands

### **AWS Console:**

- SageMaker training jobs
- CloudWatch logs
- S3 bucket contents

### **Command Line:**

```bash
# Check training job
aws sagemaker describe-training-job --training-job-name friday-lora-20240101-120000

# View logs
aws logs describe-log-groups --log-group-name-prefix /aws/sagemaker
```

---

## 🎉 **Summary**

**You can absolutely use SageMaker from VS Code!**

- ✅ Keep VS Code as your development environment
- ✅ Use AWS cloud for heavy training (no Mac memory issues)
- ✅ Low cost ($3-5 per training)
- ✅ Professional ML workflow
- ✅ All tools already installed and configured

**Ready to start?** Just configure AWS credentials and run:

```bash
python scripts/vscode_sagemaker_trainer.py
```
