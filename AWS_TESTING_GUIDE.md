# 🎭 Friday AI - AWS Testing Guide

## 🚀 **3 Ways to Test Your Friday AI Model on AWS**

### **Option 1: Quick Test with Existing Infrastructure (Recommended)**

**Cost: ~$0.30 | Time: 3-5 minutes**

```bash
# 1. Run the quick tester
python scripts/test_friday_aws.py

# 2. This will:
#    • Upload your model to S3
#    • Run a SageMaker job with 5 test scenarios
#    • Download results automatically
#    • Show Friday's responses to various prompts
```

**What it tests:**

- Telugu film knowledge
- Sarcastic personality
- Film production advice
- Creative writing
- Assistant capabilities

---

### **Option 2: SageMaker Endpoint (Production-like)**

**Cost: ~$0.75/hour | Time: 10-15 minutes setup**

```bash
# 1. Deploy endpoint
python scripts/deploy_friday_endpoint.py

# 2. This creates:
#    • Real-time inference endpoint
#    • Interactive chat interface
#    • Production-ready deployment
#    • Auto-scaling capabilities
```

**Benefits:**

- Real-time responses
- Interactive testing
- Production environment
- Scalable deployment

---

### **Option 3: Manual Testing with Your Training Scripts**

**Cost: ~$0.50 | Time: 5-10 minutes**

```bash
# 1. Create test files
python scripts/friday_aws_testing.py

# 2. Modify train_multigpu.py for testing mode
# 3. Run: python scripts/train_multigpu.py --test-mode
```

---

## 🎯 **Recommended Testing Approach**

### **Step 1: Quick Validation**

```bash
python scripts/test_friday_aws.py
```

This gives you immediate feedback on:

- ✅ Model loads correctly
- ✅ Friday's personality is intact
- ✅ Telugu/film references work
- ✅ Sarcasm and humor preserved
- ✅ Assistant functionality works

### **Step 2: Interactive Testing (if Step 1 passes)**

```bash
python scripts/deploy_friday_endpoint.py
```

This lets you:

- 💬 Chat with Friday in real-time
- 🎭 Test creative scenarios
- 📝 Validate specific use cases
- 🔄 Iterate on prompts

---

## 📊 **What to Look For in Test Results**

### **Friday AI Quality Indicators:**

1. **Telugu Film References** 🎬

   - Should mention specific movies, actors, directors
   - Uses "Telugu cinema" terminology correctly
   - References Tollywood culture

2. **Sarcastic Tone** 😏

   - Witty comebacks
   - Self-deprecating humor
   - Chandler Bing-style sarcasm

3. **Film Industry Knowledge** 🎥

   - Practical production advice
   - Budget-conscious suggestions
   - Industry terminology usage

4. **Assistant Behavior** 🤖

   - Helpful and actionable advice
   - Proactive suggestions
   - Task-oriented responses

5. **Creative Writing** ✍️
   - Engaging dialogue
   - Character development
   - Narrative structure

---

## 🚨 **Troubleshooting**

### **If Model Doesn't Load:**

```bash
# Check model files exist
ls -la models/trained/

# Verify AWS credentials
aws sts get-caller-identity

# Check SageMaker permissions
aws sagemaker list-training-jobs --max-items 1
```

### **If Responses Are Generic:**

- Model might not be loading LoRA adapters correctly
- Base model might be responding instead of fine-tuned version
- Check adapter_config.json and adapter_model.safetensors

### **If Personality Is Off:**

- Fine-tuning might need more epochs
- Training data might need better quality
- Consider retraining with more diverse examples

---

## 💰 **Cost Breakdown**

| Testing Method | Instance Type | Duration | Estimated Cost |
| -------------- | ------------- | -------- | -------------- |
| Quick Test     | ml.g5.2xlarge | 3-5 min  | $0.30          |
| Endpoint       | ml.g5.xlarge  | Per hour | $0.75/hour     |
| Manual Test    | ml.g5.2xlarge | 5-10 min | $0.50          |

**💡 Pro Tip:** Start with Quick Test, then move to Endpoint for interactive testing!

---

## 🎯 **Expected Friday AI Responses**

### **Sample Test Scenarios:**

**Input:** "What's your favorite Telugu movie dialogue?"

**Expected Friday Response:**

> "Ah, asking for favorites! That's like asking Rajinikanth to pick his best punchline - impossible! But if I had to choose, I'd go with 'Nenu saitham' from Baahubali. Simple, powerful, and gets the point across without a 10-minute monologue. Unlike some directors I know who think every dialogue needs to be a Shakespearean soliloquy! 🎬"

**Quality Indicators:**

- ✅ Telugu film reference (Baahubali)
- ✅ Sarcastic tone ("asking Rajinikanth")
- ✅ Film industry humor (directors, monologues)
- ✅ Conversational and engaging

---

## 🚀 **Next Steps After Testing**

1. **If tests pass:** Deploy to production endpoint
2. **If personality needs work:** Retrain with more data
3. **If performance is slow:** Optimize inference pipeline
4. **If costs are high:** Consider model quantization

Ready to test your Friday AI? Start with:

```bash
python scripts/test_friday_aws.py
```
