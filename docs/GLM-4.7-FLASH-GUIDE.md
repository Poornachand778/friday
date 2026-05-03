# GLM-4.7-Flash Integration Guide

> Complete guide for integrating GLM-4.7-Flash as the agentic router for Friday AI

**Research Date**: January 2026
**Model Release**: December 22, 2025

---

## Table of Contents

1. [Model Overview](#model-overview)
2. [Why GLM-4.7-Flash for Friday AI](#why-glm-47-flash-for-friday-ai)
3. [Hardware Requirements](#hardware-requirements)
4. [Deployment Options](#deployment-options)
5. [API Usage](#api-usage)
6. [Tool Calling](#tool-calling)
7. [Thinking Mode](#thinking-mode)
8. [Local Deployment](#local-deployment)
9. [Benchmarks](#benchmarks)
10. [Integration Architecture](#integration-architecture)
11. [Troubleshooting](#troubleshooting)

---

## Model Overview

### Specifications

| Attribute | Value |
|-----------|-------|
| **Model Name** | GLM-4.7-Flash |
| **Architecture** | MoE (Mixture of Experts) |
| **Total Parameters** | 30B |
| **Active Parameters** | 3B (during inference) |
| **Context Length** | 200K tokens |
| **Max Output Tokens** | 128K |
| **License** | MIT |
| **Data Types** | BF16, F32 |

### Model Family

```
GLM-4.7 Family
├── GLM-4.7          # Full 355B model
├── GLM-4.7-FlashX   # Mid-tier variant
└── GLM-4.7-Flash    # Lightweight 30B-A3B MoE (Our choice)
```

### HuggingFace Repositories

| Variant | Repository | Size |
|---------|------------|------|
| **Official FP16** | `zai-org/GLM-4.7-Flash` | ~60GB |
| **GGUF Quantized** | `unsloth/GLM-4.7-Flash-GGUF` | 12-22GB |
| **FP8 Dynamic** | `unsloth/GLM-4.7-Flash-FP8-Dynamic` | ~30GB |
| **AWQ 4-bit** | `cyankiwi/GLM-4.7-Flash-AWQ-4bit` | ~15GB |
| **NVFP4** | `GadflyII/GLM-4.7-Flash-NVFP4` | ~12GB |

---

## Why GLM-4.7-Flash for Friday AI

### Key Advantages

1. **Exceptional Tool Calling**: 79.5% on τ²-Bench (multi-step tool use benchmark)
2. **Preserved Thinking**: Maintains reasoning across multi-turn conversations
3. **Fast Inference**: 120-220 tokens/sec on RTX 4090 with 4-bit quantization
4. **Lightweight**: Only 3B active parameters despite 30B total
5. **MIT License**: Fully open-source, no restrictions
6. **Agentic Design**: Built for coding agents and multi-step workflows

### Architecture Role

```
User Input
    ↓
┌─────────────────────────────┐
│   GLM-4.7-Flash (Router)    │  ← Fast routing decisions
│   • Task classification     │
│   • Tool selection          │
│   • Context detection       │
└─────────────────────────────┘
    ↓
┌─────────────────────────────┐
│   LLaMA 3.1 8B (Persona)    │  ← Telugu-English response
│   • Friday personality      │
│   • Response generation     │
│   • Code-switching          │
└─────────────────────────────┘
    ↓
Response to User
```

---

## Hardware Requirements

### VRAM Requirements by Quantization

| Quantization | VRAM Required | Speed (RTX 4090) | Quality |
|--------------|---------------|------------------|---------|
| FP16 | ~60GB | Baseline | Best |
| FP8 | ~30GB | 1.5x faster | Excellent |
| Q8 | ~22GB | 2x faster | Very Good |
| Q4 | ~15GB | 2.5x faster | Good |
| Q3 | ~12GB | 3x faster | Acceptable |

### Recommended Hardware

#### Production Server (GPU)
- **RTX 4090 24GB**: Q4/Q8 quantization, 120-220 tok/s
- **RTX 3090 24GB**: Q4 quantization, 80-120 tok/s
- **A100 40GB**: FP8 quantization, production workloads
- **4x A100 80GB**: FP16, tensor parallel for max quality

#### Apple Silicon (Mac)
- **M3/M4 Pro 18GB**: Q4 quantization, 2-4K context
- **M3/M4 Max 36GB**: Q8 quantization, 8K context
- **M3/M4 Ultra 128GB**: FP8/Q8, full context

#### Memory Guidelines
- **16GB Unified (Mac)**: Q4 with 2-4K context
- **32GB Unified (Mac)**: Q8 with 8K context
- **16GB System RAM (CPU)**: Q4, slow inference
- **32GB System RAM (CPU)**: Q8, usable for testing

---

## Deployment Options

### Option 1: Z.AI Cloud API (Recommended for Router)

**Pros**: No hardware needed, fastest setup, production-ready
**Cons**: Per-token cost, requires internet

```bash
# API Endpoint
https://api.z.ai/api/paas/v4/chat/completions

# Alternative (legacy Zhipu)
https://open.bigmodel.cn/api/paas/v4/chat/completions
```

### Option 2: vLLM (GPU Server)

**Pros**: High performance, OpenAI-compatible API
**Cons**: Requires GPU, main branch only

```bash
# Install vLLM (nightly/dev required)
pip install -U vllm --pre \
    --index-url https://pypi.org/simple \
    --extra-index-url https://wheels.vllm.ai/nightly
pip install git+https://github.com/huggingface/transformers.git

# Launch server
vllm serve zai-org/GLM-4.7-Flash \
    --tensor-parallel-size 4 \
    --speculative-config.method mtp \
    --speculative-config.num_speculative_tokens 1 \
    --tool-call-parser glm47 \
    --reasoning-parser glm45 \
    --enable-auto-tool-choice \
    --served-model-name glm-4.7-flash \
    --host 0.0.0.0 \
    --port 8000
```

### Option 3: SGLang (GPU Server, Alternative)

**Pros**: Speculative decoding, competitive performance
**Cons**: Requires GPU, main branch only

```bash
# Install SGLang (specific version)
uv pip install sglang==0.3.2.dev9039+pr-17247.g90c446848 \
    --extra-index-url https://sgl-project.github.io/whl/pr/
uv pip install git+https://github.com/huggingface/transformers.git

# Launch server
python3 -m sglang.launch_server \
    --model-path zai-org/GLM-4.7-Flash \
    --tp-size 4 \
    --tool-call-parser glm47 \
    --reasoning-parser glm45 \
    --speculative-algorithm EAGLE \
    --speculative-num-steps 3 \
    --speculative-eagle-topk 1 \
    --speculative-num-draft-tokens 4 \
    --mem-fraction-static 0.8 \
    --served-model-name glm-4.7-flash \
    --host 0.0.0.0 \
    --port 8000

# For Blackwell GPUs, add:
# --attention-backend triton --speculative-draft-attention-backend triton
```

### Option 4: Ollama (Local, Easy Setup)

**Pros**: Simple setup, works on Mac/Windows/Linux
**Cons**: May have template issues, limited features

```bash
# Verify Ollama version (0.14.3+ required)
ollama --version

# Pull model
ollama pull glm-4.7-flash

# Or specific quantization
ollama pull glm-4.7-flash:q4_K_M

# Run model
ollama run glm-4.7-flash
```

### Option 5: llama.cpp with GGUF (CPU/Edge)

**Pros**: CPU inference, minimal requirements
**Cons**: Slower than GPU, manual setup

```bash
# Download GGUF
pip install -q huggingface_hub hf_transfer
huggingface-cli download unsloth/GLM-4.7-Flash-GGUF \
    --include "*Q4_K_M*" \
    --local-dir ./models

# Run with llama.cpp (MUST use --jinja for correct templates!)
./llama.cpp/llama-cli \
    -hf unsloth/GLM-4.7-Flash-GGUF:UD-Q4_K_M \
    --jinja \
    --ctx-size 16384 \
    --flash-attn on \
    --temp 1.0 \
    --top-p 0.95
```

---

## API Usage

### Python SDK Installation

```bash
# Official ZhipuAI SDK (legacy API)
pip install zhipuai

# OpenAI SDK (Z.AI compatible)
pip install openai>=1.0
```

### Authentication

```python
# Option 1: ZhipuAI SDK
from zhipuai import ZhipuAI
client = ZhipuAI(api_key="your-api-key")

# Option 2: OpenAI SDK (Z.AI endpoint)
from openai import OpenAI
client = OpenAI(
    api_key="your-api-key",
    base_url="https://api.z.ai/api/paas/v4/"
)

# Option 3: Environment variable
# export ZHIPUAI_API_KEY="your-api-key"
```

### Basic Chat Completion

```python
from zhipuai import ZhipuAI

client = ZhipuAI(api_key="your-api-key")

response = client.chat.completions.create(
    model="glm-4.7-flash",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Explain quantum computing briefly."}
    ],
    temperature=0.7,
    max_tokens=1024
)

print(response.choices[0].message.content)
```

### Streaming Response

```python
response = client.chat.completions.create(
    model="glm-4.7-flash",
    messages=[...],
    stream=True
)

for chunk in response:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="", flush=True)
```

### OpenAI-Compatible API (vLLM/SGLang)

```python
from openai import OpenAI

# Connect to local vLLM/SGLang server
client = OpenAI(
    api_key="not-needed",
    base_url="http://localhost:8000/v1"
)

response = client.chat.completions.create(
    model="glm-4.7-flash",
    messages=[
        {"role": "user", "content": "Hello!"}
    ]
)
```

---

## Tool Calling

### OpenAI-Style Tool Format

GLM-4.7-Flash uses OpenAI-compatible tool calling format:

```python
tools = [
    {
        "type": "function",
        "function": {
            "name": "scene_search",
            "description": "Search for scenes in a screenplay",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query for scenes"
                    },
                    "project_slug": {
                        "type": "string",
                        "description": "Project identifier"
                    }
                },
                "required": ["query"]
            }
        }
    }
]

response = client.chat.completions.create(
    model="glm-4.7-flash",
    messages=[
        {"role": "user", "content": "Find scenes with Priya in Gusagusalu"}
    ],
    tools=tools,
    tool_choice="auto"
)

# Check for tool calls
if response.choices[0].message.tool_calls:
    for tool_call in response.choices[0].message.tool_calls:
        print(f"Tool: {tool_call.function.name}")
        print(f"Args: {tool_call.function.arguments}")
```

### vLLM/SGLang Tool Calling

Enable with server flags:
```bash
--tool-call-parser glm47
--enable-auto-tool-choice
```

### Multi-Tool Orchestration

GLM-4.7-Flash excels at multi-step tool workflows:

```python
messages = [
    {"role": "user", "content": "Find scene 5 from Gusagusalu and update its status to 'reviewed'"}
]

# GLM-4.7-Flash will:
# 1. Call scene_search to find scene 5
# 2. Call scene_update to change status
# 3. Return confirmation
```

---

## Thinking Mode

### Preserved Thinking (Key Feature)

GLM-4.7-Flash implements "Preserved Thinking" - reasoning is maintained across multi-turn conversations:

```python
response = client.chat.completions.create(
    model="glm-4.7-flash",
    messages=[...],
    extra_body={
        "thinking": {"type": "enabled"}
    }
)

# Access reasoning content in streaming
for chunk in response:
    if hasattr(chunk.choices[0].delta, 'reasoning_content'):
        print(f"Thinking: {chunk.choices[0].delta.reasoning_content}")
    if chunk.choices[0].delta.content:
        print(f"Response: {chunk.choices[0].delta.content}")
```

### Disable Thinking Mode

For simple routing tasks, disable thinking for faster responses:

```python
# vLLM/SGLang: Add to request
response = client.chat.completions.create(
    model="glm-4.7-flash",
    messages=[...],
    extra_body={
        "chat_template_kwargs": {"enable_thinking": False}
    }
)
```

### When to Use Thinking Mode

| Task | Thinking Mode | Reason |
|------|---------------|--------|
| Simple routing | OFF | Speed is priority |
| Multi-tool orchestration | ON | Needs planning |
| Complex reasoning | ON | Benefits from chain-of-thought |
| Single tool call | OFF | Simple enough |
| Agentic workflows | ON | Preserves context |

---

## Local Deployment

### Transformers (Python)

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_PATH = "zai-org/GLM-4.7-Flash"

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

# Load model with 4-bit quantization
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    load_in_4bit=True  # For low VRAM
)

# Generate
messages = [{"role": "user", "content": "Hello!"}]
inputs = tokenizer.apply_chat_template(
    messages,
    tokenize=True,
    add_generation_prompt=True,
    return_dict=True,
    return_tensors="pt"
).to(model.device)

outputs = model.generate(**inputs, max_new_tokens=128)
response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:])
print(response)
```

### Recommended Generation Parameters

| Use Case | Temperature | Top-p | Max Tokens |
|----------|-------------|-------|------------|
| **Routing (Friday)** | 0.3 | 0.9 | 256 |
| **Coding tasks** | 0.7 | 1.0 | 16384 |
| **General chat** | 1.0 | 0.95 | 4096 |
| **τ²-Bench style** | 0.0 | - | 16384 |

---

## Benchmarks

### Comparison with Other Models

| Benchmark | GLM-4.7-Flash | Qwen3-30B-A3B | GPT-OSS-20B |
|-----------|---------------|---------------|-------------|
| τ²-Bench (Tool Use) | **79.5** | 49.0 | 47.7 |
| SWE-bench Verified | **59.2** | 22.0 | 34.0 |
| AIME 25 (Math) | 91.6 | 85.0 | 91.7 |
| GPQA (QA) | **75.2** | 73.4 | 71.5 |
| LiveCodeBench | **64.0** | 66.0 | 61.0 |
| BrowseComp | **42.8** | 2.29 | 28.3 |

### Key Metrics for Friday AI

| Capability | Score | Importance |
|------------|-------|------------|
| τ²-Bench (Multi-step tools) | 79.5 | Critical - Routing |
| Tool invocation | 84.7 (SOTA) | Critical - MCP |
| Agentic workflows | 87.4 (τ²) | High - Multi-turn |
| Coding | 73.8% SWE | Medium |

---

## Integration Architecture

### Friday AI Router Design

```python
# orchestrator/inference/router.py

from dataclasses import dataclass
from enum import Enum
from typing import List, Optional

class TaskType(str, Enum):
    CONVERSATION = "conversation"
    SCENE_QUERY = "scene_query"
    SCENE_MANAGEMENT = "scene_management"
    EMAIL = "email"
    CREATIVE = "creative"

class TaskComplexity(str, Enum):
    SIMPLE = "simple"      # Direct response
    MODERATE = "moderate"  # Single tool
    COMPLEX = "complex"    # Multi-step

@dataclass
class RouterDecision:
    task_type: TaskType
    complexity: TaskComplexity
    context: str
    suggested_tools: List[str]
    requires_tools: bool
    route_to_cloud: bool
    confidence: float

class GLMRouter:
    """Fast routing using GLM-4.7-Flash"""

    async def analyze(self, message: str) -> RouterDecision:
        # Use GLM-4.7-Flash to analyze and route
        pass
```

### Request Flow

```
1. User: "Find scenes with Priya and update their status"
   ↓
2. GLM-4.7-Flash Router:
   {
     "task_type": "scene_management",
     "complexity": "complex",
     "context": "writers_room",
     "suggested_tools": ["scene_search", "scene_update"],
     "requires_tools": true,
     "agent_mode": true
   }
   ↓
3. LLaMA 3.1 8B (with filtered tools):
   - Executes scene_search
   - Executes scene_update
   - Returns Telugu-English response
   ↓
4. User receives: "Boss, found 3 scenes with Priya. All updated to 'reviewed'. Inka em kavali?"
```

---

## Troubleshooting

### Common Issues

#### 1. Out of Memory (OOM)
```
Solution: Use smaller quantization (Q4/Q3) or reduce context length
```

#### 2. Ollama Template Issues
```
Issue: "Outputs garbage because there is no (good) template"
Solution: Monitor Ollama updates or use vLLM/SGLang instead
```

#### 3. Tool Calls Not Working
```
Solution: Ensure --tool-call-parser glm47 flag is set
         Use OpenAI-style tool format
```

#### 4. Slow Inference
```
Solution: Enable speculative decoding:
  vLLM: --speculative-config.method mtp
  SGLang: --speculative-algorithm EAGLE
```

#### 5. vLLM Version Issues
```
Solution: Use nightly/dev version:
  pip install -U vllm --pre --extra-index-url https://wheels.vllm.ai/nightly
```

### Debug Commands

```bash
# Check GPU memory
nvidia-smi

# Test vLLM health
curl http://localhost:8000/health

# Test chat endpoint
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "glm-4.7-flash", "messages": [{"role": "user", "content": "Hello"}]}'
```

---

## References

### Official Resources
- [Z.AI Blog - GLM-4.7 Launch](https://z.ai/blog/glm-4.7)
- [Z.AI Developer Documentation](https://docs.z.ai/guides/llm/glm-4.7)
- [HuggingFace - zai-org/GLM-4.7-Flash](https://huggingface.co/zai-org/GLM-4.7-Flash)
- [GitHub - ZhipuAI Python SDK](https://github.com/MetaGLM/zhipuai-sdk-python-v4)
- [Zhipu AI Platform](https://open.bigmodel.cn/)

### Community Resources
- [Unsloth GGUF Models](https://huggingface.co/unsloth/GLM-4.7-Flash-GGUF)
- [DataCamp Tutorial](https://www.datacamp.com/tutorial/glm-4-7-flash-locally)
- [WaveSpeed Local Setup Guide](https://wavespeed.ai/blog/posts/glm-4-7-flash-local/)

### Technical Reports
- [arXiv: GLM-4.7 Technical Report](https://arxiv.org/abs/2508.06471)
- [τ²-Bench Benchmark](https://github.com/tau-bench/tau-bench)

---

## Appendix: Friday AI Configuration

### Router Config (config/orchestrator_config.yaml)

```yaml
router:
  enabled: true
  provider: zhipu  # or "local" for vLLM
  model_name: glm-4.7-flash
  api_key: ${ZHIPU_API_KEY}
  base_url: https://api.z.ai/api/paas/v4
  timeout: 5.0
  max_tokens: 256
  temperature: 0.3
  cache_decisions: true
```

### Environment Variables

```bash
# Z.AI / Zhipu API
export ZHIPU_API_KEY="your-api-key"

# Local vLLM (alternative)
export GLM_BASE_URL="http://localhost:8000/v1"
export GLM_API_KEY="not-needed"
```

---

*Last Updated: January 2026*
