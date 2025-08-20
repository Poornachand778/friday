# 🎭 Friday AI SageMaker Deployment Chronicles

_The complete journey from failed deployments to production-ready real-time inference_

## � FINAL STATUS: PRODUCTION READY ✅

**Date**: August 16, 2025  
**Validation**: 6/6 checks passing  
**Endpoint**: `friday-rt` - InService with autoscaling (1-3 instances)  
**Cost**: ~$1.006/hour per instance (~$24/day minimum)

---

## 🎯 Current Status (August 16, 2025 - COMPLETE) ✅

### ✅ **FULLY OPERATIONAL**

All deployment issues resolved. Endpoint is production-ready with:

1. **✅ Infrastructure Deployment** - All AWS components created and configured
2. **✅ Container Image** - Correct HuggingFace PyTorch DLC (2.6.0-transformers4.49.0-gpu-py312-cu124-ubuntu22.04)
3. **✅ Cache Directory Fix** - HF caches moved from read-only `/opt/ml/model` to writable `/tmp/hf`
4. **✅ Token Management** - HF_TOKEN properly configured in environment
5. **✅ Autoscaling Setup** - Fixed parameter issues, 1-3 instances with service-linked role
6. **✅ Inference Quality** - No prompt echoing, proper token limits, pad token configuration
7. **✅ Performance Optimizations** - TF32 enabled, inference_mode(), BitsAndBytes with fp16 fallback
8. **✅ Robust Error Handling** - Graceful degradation and comprehensive validation

### 🚀 Production Features

- **Real-time Endpoint**: `friday-rt` with Meta-Llama-3.1-8B-Instruct + Friday LoRA
- **Auto-scaling**: 1-3 ml.g5.2xlarge instances based on invocation rate
- **Performance**: Optimized inference with TF32, proper caching, and token management
- **Reliability**: 300s timeout for first inference, graceful BitsAndBytes fallback
- **Monitoring**: CloudWatch logs, autoscaling metrics, comprehensive health checks

---

## 📚 The Complete Journey: Issues & Solutions

### 🔥 **Critical Issue #1: Wrong Container Image**

**Problem**: Using PyTorch DLC instead of HuggingFace DLC

- PyTorch container expected TorchServe `.mar` files
- Our script-mode `inference.py` was ignored → HTTP 500 on `/ping`
- All inference requests timed out

**Solution**:

- Switched to HuggingFace PyTorch Inference DLC
- Container: `763104351884.dkr.ecr.us-east-1.amazonaws.com/huggingface-pytorch-inference:2.6.0-transformers4.49.0-gpu-py312-cu124-ubuntu22.04`
- Supports SageMaker script mode natively

### 🔥 **Critical Issue #2: Read-Only File System [Errno 30]**

**Problem**: HuggingFace caches pointed to `/opt/ml/model` (read-only)

- First inference tried to download base model (~15GB) to read-only path
- Error: `[Errno 30] Read-only file system: '/opt/ml/model/models--meta-llama--Meta-Llama-3.1-8B-Instruct'`

**Solution**:

```python
# In deployer environment:
"TRANSFORMERS_CACHE": "/tmp/hf",
"HF_HOME": "/tmp/hf",
"HF_DATASETS_CACHE": "/tmp/hf",

# In inference.py:
cache_dir = os.environ.get("TRANSFORMERS_CACHE", "/tmp/hf")
Path(cache_dir).mkdir(parents=True, exist_ok=True)
# Pass cache_dir to all from_pretrained() calls
```

### 🔧 **Issue #3: Autoscaling Parameter Error**

**Problem**: `register_scalable_target()` failed with incorrect `RoleArn` parameter
**Solution**: Removed `RoleArn` entirely - SageMaker uses service-linked role automatically

### 🔧 **Issue #4: S3 Bucket Creation**

**Problem**: Manual bucket naming `sagemaker-{region}-{account}` might not exist
**Solution**: Use `self.sagemaker_session.default_bucket()` - auto-creates if needed

### 🔧 **Issue #5: Tar Packaging Structure**

**Problem**: `arcname=f"code/{file_path.name}"` flattened directory structure
**Solution**: Preserve relative paths: `arcname=f"code/{file_path.relative_to(inference_dir)}"`

### 🔧 **Issue #6: Token Management**

**Problem**: Environment had `HUGGINGFACE_TOKEN` but code expected `HF_TOKEN`
**Solution**: Added `HF_TOKEN=hf_your_token_here` to `.env`

### 🔧 **Issue #7: Prompt Echoing**

**Problem**: `generate()` returns full sequence (prompt + new tokens)
**Solution**: Slice off input tokens: `new_tokens = output_ids[0][input_length:]`

### 🔧 **Issue #8: No Token Limits**

**Problem**: Long inputs could cause timeouts/VRAM issues
**Solution**: Added input validation and max_new_tokens clamping:

```python
if input_length > max_input_length:
    raise ValueError(f"Input too long: {input_length} tokens")
max_new_tokens = min(max_new_tokens, max_total_tokens - input_length)
```

### 🔧 **Issue #9: LLaMA Pad Token**

**Problem**: LLaMA tokenizer has `pad_token_id=None`
**Solution**: `_TOKENIZER.pad_token = _TOKENIZER.eos_token`

### ⚡ **Performance Optimizations Added**

1. **TF32 on Ampere GPUs**: `torch.backends.cuda.matmul.allow_tf32 = True`
2. **Inference Mode**: `torch.inference_mode()` vs `torch.no_grad()`
3. **BitsAndBytes Fallback**: Graceful fallback to fp16 if 4-bit fails
4. **Lazy Loading**: Background model loading for fast `/ping` responses

---

## �️ **Deployment Architecture**

### **Container Stack**

- **Base**: HuggingFace PyTorch Inference DLC 2.6.0
- **Model**: Meta-Llama-3.1-8B-Instruct (15GB) + Friday LoRA adapters (10MB)
- **Quantization**: 4-bit BitsAndBytes (fallback to fp16)
- **Cache**: `/tmp/hf` (writable, ~20GB for base model)

### **Infrastructure**

- **Endpoint**: `friday-rt` on ml.g5.2xlarge instances
- **Autoscaling**: 1-3 instances, target 70 invocations/instance
- **S3**: Model artifacts in SageMaker default bucket
- **Monitoring**: CloudWatch logs + metrics

### **Key Files**

- **Deployer**: `src/deployment/deploy_friday_endpoint.py`
- **Inference**: `src/inference/sagemaker_code/inference.py`
- **Requirements**: `src/inference/sagemaker_code/requirements.txt`
- **Validation**: `validate_deployment.py`
- **Environment**: `.env` (with HF_TOKEN)

---

## 🚀 **Quick Deployment Guide**

### **Prerequisites**

```bash
# 1. Environment setup
conda activate friday_ft
source .env  # Must contain HF_TOKEN

# 2. Required files
src/inference/sagemaker_code/inference.py    # ✅ Fixed
src/inference/sagemaker_code/requirements.txt # ✅ Fixed
models/trained/*.safetensors                  # LoRA weights
models/trained/*.json                         # LoRA config
```

### **Deploy Command**

```bash
python src/deployment/deploy_friday_endpoint.py
```

### **Validate Deployment**

```bash
python validate_deployment.py
# Should show: 6/6 checks passed ✅
```

### **Test Inference**

```python
import boto3, json
from botocore.config import Config

rt = boto3.client("sagemaker-runtime", "us-east-1",
                  config=Config(read_timeout=300))  # Long timeout for first call

response = rt.invoke_endpoint(
    EndpointName="friday-rt",
    ContentType="application/json",
    Body=json.dumps({
        "inputs": "Hello, how are you today?",
        "parameters": {"max_new_tokens": 50, "temperature": 0.7}
    })
)
print(json.loads(response["Body"].read()))
```

---

## ⚠️ **Lessons Learned & Best Practices**

### **Container Selection**

- ✅ **DO**: Use HuggingFace DLC for script-mode inference with transformers
- ❌ **DON'T**: Use PyTorch DLC unless you're using TorchServe with .mar files

### **File System Management**

- ✅ **DO**: Point all HF caches to `/tmp/hf` or `/opt/ml/input/data`
- ❌ **DON'T**: Write anything to `/opt/ml/model` (read-only on real-time endpoints)

### **Token Management**

- ✅ **DO**: Use consistent env var names (`HF_TOKEN`)
- ✅ **DO**: Test token loading: `source .env && echo $HF_TOKEN`
- ❌ **DON'T**: Assume different token variable names will work

### **Performance & Reliability**

- ✅ **DO**: Set long timeouts for first inference (300s)
- ✅ **DO**: Implement graceful fallbacks (4-bit → fp16)
- ✅ **DO**: Slice off prompt tokens to avoid echoing
- ✅ **DO**: Enforce token limits to prevent timeouts
- ❌ **DON'T**: Skip input validation or assume unlimited context

### **Deployment Process**

- ✅ **DO**: Always run `validate_deployment.py` after deployment
- ✅ **DO**: Check CloudWatch logs if inference fails
- ✅ **DO**: Test with simple prompts first, then complex ones
- ❌ **DON'T**: Deploy without verifying all artifacts exist

---

## 🎯 **Troubleshooting Checklist**

If deployment fails, check in order:

### **1. Container & Image Issues**

- [ ] Using HuggingFace DLC (not PyTorch DLC)?
- [ ] Image tag exists in us-east-1?
- [ ] Check: `aws ecr describe-images --registry-id 763104351884 --repository-name huggingface-pytorch-inference`

### **2. Cache & File System**

- [ ] All HF env vars point to `/tmp/hf`?
- [ ] `inference.py` creates cache_dir and passes it to `from_pretrained()`?
- [ ] CloudWatch logs show `[Errno 30]` read-only errors?

### **3. Token & Authentication**

- [ ] `HF_TOKEN` in .env file?
- [ ] `source .env && echo $HF_TOKEN` shows token?
- [ ] Token has access to Meta-Llama models?

### **4. Model Artifacts**

- [ ] LoRA files exist in `models/trained/`?
- [ ] Tar packaging preserves directory structure?
- [ ] S3 upload successful?

### **5. Inference Issues**

- [ ] First call uses 300s timeout?
- [ ] Model loading completes (check logs)?
- [ ] Input within token limits?
- [ ] Response doesn't echo prompt?

---

## 💾 **Final Configuration Files**

### **Environment Variables (.env)**

```env
HF_TOKEN=hf_your_token_here
AWS_DEFAULT_REGION=us-east-1
# ... other AWS configs
```

### **Container Environment (in deployer)**

```python
environment = {
    "BASE_MODEL_ID": "meta-llama/Meta-Llama-3.1-8B-Instruct",
    "HF_TOKEN": hf_token,
    "TRANSFORMERS_CACHE": "/tmp/hf",
    "HF_HOME": "/tmp/hf",
    "HF_DATASETS_CACHE": "/tmp/hf",
    "HF_HUB_ENABLE_HF_TRANSFER": "1",
    "PYTHONUNBUFFERED": "1",
    "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
    "USE_4BIT": "true",
    "MAX_INPUT_LENGTH": "4096",
    "MAX_TOTAL_TOKENS": "8192",
    "SAGEMAKER_PROGRAM": "inference.py",
    "SAGEMAKER_SUBMIT_DIRECTORY": "/opt/ml/code",
    "SAGEMAKER_CONTAINER_LOG_LEVEL": "20",
}
```

---

## 🎊 **Success Metrics**

**Final Validation Results (August 16-17, 2025)**:

- ✅ Model: PASS - `friday-rt-model` operational since 2025-08-16 22:55:12
- ✅ Endpoint Config: PASS - `friday-rt-config` with ml.g5.2xlarge instances
- ✅ Endpoint: PASS - `friday-rt` InService since 2025-08-16 23:01:17
- ✅ Autoscaling: PASS - 1-3 instances, 70 invocations/instance target
- ✅ Logs: PASS - No error messages in CloudWatch logs
- ✅ Inference: PASS - 381-character response in test

**Performance**:

- First inference: ~30-60s (model download + generation)
- Subsequent inferences: ~2-5s
- Auto-scaling: Responsive to load (1-3 instances)
- Cost: $1.006/hour per instance (~$24.14/day minimum)

**Reliability**:

- Health checks: Passing consistently
- Error handling: Graceful degradation implemented
- Monitoring: Full CloudWatch integration active
- Uptime: 99.9%+ expected based on SageMaker SLA

---

_🎭 The Friday AI deployment saga: From read-only file systems to production excellence. May future deployments be swift and error-free!_
