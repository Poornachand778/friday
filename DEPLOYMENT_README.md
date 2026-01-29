# 🎭 Friday AI - Production SageMaker Endpoint Deployment

Production-ready real-time inference deployment for your fine-tuned Friday AI model.

## 🚀 Quick Deployment

### Prerequisites

```bash
# 1. Ensure you have Friday AI model artifacts
ls models/trained/  # Should contain adapter_*.* files

# 2. Configure AWS credentials
aws configure

# 3. Setup environment
pip install -r requirements.txt
cp .env.template .env  # Edit with your values
```

### One-Command Deployment

```bash
# Complete deployment pipeline
python setup_secrets.py --token YOUR_HF_TOKEN
python package_model_artifacts.py
python deploy_friday_endpoint.py

# Test the deployment
python smoke_test.py
```

## 📋 **Deployment Components**

### **Core Scripts**

- `deploy_friday_endpoint.py` - Main deployment orchestrator
- `deployment/code/inference.py` - Production inference server
- `package_model_artifacts.py` - Model artifact packager
- `smoke_test.py` - Endpoint testing suite
- `delete_friday_endpoint.py` - Safe resource cleanup

### **Support Scripts**

- `setup_secrets.py` - AWS Secrets Manager setup
- `deployment/code/requirements.txt` - Inference dependencies

---

## 🔧 **Detailed Setup**

### **1. Setup Secrets Manager**

```bash
# Store HuggingFace token securely
python setup_secrets.py --token hf_xxxxxxxxxxxx

# Verify secret exists
aws secretsmanager get-secret-value --secret-id friday-ai/hf-token
```

### **2. Package Model Artifacts**

```bash
# Create model.tar.gz and upload to S3
python package_model_artifacts.py

# Manual packaging (if needed)
python package_model_artifacts.py --no-upload --archive custom_model.tar.gz
```

### **3. Deploy Endpoint**

```bash
# Full deployment with defaults
python deploy_friday_endpoint.py

# Custom configuration
python deploy_friday_endpoint.py \
  --endpoint-name friday-production \
  --instance-type ml.g5.4xlarge \
  --instance-count 2 \
  --volume-size 512
```

### **4. Test Deployment**

```bash
# Full test suite
python smoke_test.py

# Specific tests
python smoke_test.py --test single
python smoke_test.py --test batch
python smoke_test.py --endpoint friday-production
```

---

## 💻 **Inference API**

### **Request Format**

```json
{
  "inputs": "What's your favorite Telugu movie dialogue?",
  "parameters": {
    "max_new_tokens": 200,
    "temperature": 0.7,
    "top_p": 0.9,
    "top_k": 50,
    "repetition_penalty": 1.1,
    "stop": ["\\n\\n", "Human:"],
    "seed": 42,
    "do_sample": true
  }
}
```

### **Response Format**

```json
{
  "generated_text": "Ah, asking for favorites! That's like asking Rajinikanth...",
  "usage": {
    "prompt_tokens": 45,
    "completion_tokens": 127,
    "total_tokens": 172
  },
  "model_id": "meta-llama/Meta-Llama-3.1-8B-Instruct+friday-lora",
  "created": 1692123456,
  "finish_reason": "stop"
}
```

### **Batch Requests**

```json
{
  "inputs": ["Tell me a film joke", "Help with screenplay", "Cooking advice"],
  "parameters": {
    "max_new_tokens": 100,
    "temperature": 0.8
  }
}
```

---

## 🎯 **Configuration Options**

### **Instance Types**

| Instance       | vCPU | GPU Memory | RAM   | Cost/Hour | Use Case                 |
| -------------- | ---- | ---------- | ----- | --------- | ------------------------ |
| ml.g5.xlarge   | 4    | 24GB       | 16GB  | $1.01     | Development              |
| ml.g5.2xlarge  | 8    | 24GB       | 32GB  | $1.34     | Production (Recommended) |
| ml.g5.4xlarge  | 16   | 24GB       | 64GB  | $2.03     | High throughput          |
| ml.g5.12xlarge | 48   | 96GB       | 192GB | $5.67     | Large batch processing   |

### **Environment Variables**

```bash
# Model configuration
BASE_MODEL_ID=meta-llama/Meta-Llama-3.1-8B-Instruct
USE_4BIT=true
MAX_INPUT_LENGTH=4096
MAX_TOTAL_TOKENS=8192

# Caching (critical for disk space)
TRANSFORMERS_CACHE=/opt/ml/model
HF_HOME=/opt/ml/model

# Security
HF_TOKEN=<from-secrets-manager>
```

### **Autoscaling Configuration**

- **Min instances**: 1
- **Max instances**: 3
- **Metric**: InvocationsPerInstance
- **Target**: 70 invocations/instance
- **Scale-out cooldown**: 300s
- **Scale-in cooldown**: 300s

---

## 📊 **Monitoring & Logs**

### **CloudWatch Metrics**

- `Invocations` - Total requests
- `InvocationsPerInstance` - Load per instance
- `Invocation4XXErrors` - Client errors
- `Invocation5XXErrors` - Server errors
- `ModelLatency` - Response time

### **Custom Metrics (Logged)**

```json
{
  "event": "inference_complete",
  "inference_time_ms": 1250.5,
  "prompt_tokens": 45,
  "completion_tokens": 127,
  "total_tokens": 172,
  "batch_size": 1,
  "timestamp": "2025-08-15T10:30:00Z"
}
```

### **Log Groups**

- `/aws/sagemaker/Endpoints/{endpoint-name}` - Inference logs
- Container startup and health check logs
- Error stack traces (redacted for security)

---

## 🛡️ **Security & Best Practices**

### **IAM Permissions Required**

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": [
        "sagemaker:CreateModel",
        "sagemaker:CreateEndpointConfig",
        "sagemaker:CreateEndpoint",
        "sagemaker:UpdateEndpoint",
        "sagemaker:DeleteModel",
        "sagemaker:DeleteEndpointConfig",
        "sagemaker:DeleteEndpoint",
        "sagemaker:DescribeModel",
        "sagemaker:DescribeEndpointConfig",
        "sagemaker:DescribeEndpoint"
      ],
      "Resource": "*"
    },
    {
      "Effect": "Allow",
      "Action": ["secretsmanager:GetSecretValue"],
      "Resource": "arn:aws:secretsmanager:*:*:secret:friday-ai/*"
    },
    {
      "Effect": "Allow",
      "Action": ["s3:GetObject", "s3:PutObject"],
      "Resource": "arn:aws:s3:::sagemaker-*/*"
    }
  ]
}
```

### **Network Security**

- Endpoints are publicly accessible by default
- Use VPC endpoints for private access
- Implement API Gateway + authentication for production

### **Data Privacy**

- No training data is stored in inference containers
- HuggingFace token stored in Secrets Manager
- Request/response data logged to CloudWatch (be mindful of PII)

---

## 🔧 **Troubleshooting**

### **Common Issues**

#### **1. Model Loading Timeout**

```
Container startup health check timeout (900s)
```

**Solution**: Increase `volume_size` or `container_startup_health_check_timeout`

#### **2. CUDA Out of Memory**

```
RuntimeError: CUDA out of memory
```

**Solutions**:

- Reduce `max_new_tokens` in requests
- Use larger instance type (ml.g5.4xlarge)
- Enable request batching limits

#### **3. HuggingFace Authentication**

```
401 Client Error: Unauthorized for gated model
```

**Solution**:

```bash
python setup_secrets.py --token YOUR_HF_TOKEN
```

#### **4. Disk Space Issues**

```
No space left on device
```

**Solutions**:

- Increase `volume_size` (recommended: 256GB+)
- Verify `TRANSFORMERS_CACHE=/opt/ml/model`

### **Debug Commands**

```bash
# Check endpoint status
aws sagemaker describe-endpoint --endpoint-name friday-rt

# View logs
aws logs describe-log-groups --log-group-name-prefix /aws/sagemaker/Endpoints/friday-rt

# Test connectivity
python smoke_test.py --test health

# List resources
python delete_friday_endpoint.py --list
```

---

## 💰 **Cost Optimization**

### **Cost Analysis**

- **ml.g5.2xlarge**: ~$32/day ($0.75-1.34/hour depending on region)
- **Autoscaling**: Automatically scales to 0 during inactivity
- **Storage**: ~$0.10/month per GB (EBS volumes)
- **Data transfer**: $0.09/GB (outbound)

### **Optimization Strategies**

1. **Use autoscaling** - Reduces cost during low usage
2. **Right-size instances** - Start with ml.g5.2xlarge
3. **Monitor utilization** - CloudWatch metrics
4. **Delete when not needed** - Use `delete_friday_endpoint.py`

### **Budget Alerts**

```bash
# Set up billing alert
aws budgets create-budget --account-id YOUR_ACCOUNT_ID --budget '{
  "BudgetName": "Friday-AI-Endpoint",
  "BudgetLimit": {"Amount": "100", "Unit": "USD"},
  "TimeUnit": "MONTHLY",
  "BudgetType": "COST"
}'
```

---

## 🚨 **Emergency Procedures**

### **Stop All Billing Immediately**

```bash
# Delete endpoint (stops billing)
python delete_friday_endpoint.py --delete --force --endpoint friday-rt

# Or via AWS CLI
aws sagemaker delete-endpoint --endpoint-name friday-rt
```

### **Backup/Recovery**

- **Model artifacts**: Stored in S3 (`s3://bucket/friday/model.tar.gz`)
- **Configuration**: All settings in deployment scripts
- **Secrets**: Stored in Secrets Manager
- **Recovery time**: 10-15 minutes for full redeployment

### **Rollback Procedure**

1. Deploy previous model version: Update `model_data_url`
2. Use blue/green deployment: Create new endpoint, switch traffic
3. Zero-downtime updates supported via `update_endpoint`

---

## 📚 **Advanced Configuration**

### **Custom Inference Logic**

Edit `deployment/code/inference.py` to:

- Add custom preprocessing
- Implement request routing
- Add response caching
- Integrate with external APIs

### **Alternative Serving Backends**

```python
# TGI (Text Generation Inference) - commented example
image_uri = "763104351884.dkr.ecr.us-east-1.amazonaws.com/huggingface-text-generation-inference:1.0.3-tgi0.8.2-gpu-py39-cu118-ubuntu20.04"
```

### **Multi-Model Endpoints**

- Load multiple LoRA adapters
- Route requests based on parameters
- A/B testing different model versions

---

## 📞 **Support**

### **Documentation**

- [SageMaker Real-time Inference](https://docs.aws.amazon.com/sagemaker/latest/dg/realtime-endpoints.html)
- [HuggingFace on SageMaker](https://huggingface.co/docs/sagemaker/inference)
- [PEFT Documentation](https://huggingface.co/docs/peft)

### **Monitoring Dashboards**

- AWS Console → SageMaker → Endpoints
- CloudWatch → Dashboards → SageMaker
- Cost Explorer → Service: SageMaker

### **Getting Help**

1. Check CloudWatch logs first
2. Run smoke tests: `python smoke_test.py`
3. Verify secrets: `aws secretsmanager get-secret-value --secret-id friday-ai/hf-token`
4. AWS Support for infrastructure issues

---

**🎭 Ready to deploy Friday AI to production!** 🚀
