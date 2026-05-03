# Research: NVIDIA DGX Spark for Friday Pre-Production Workflow

**Date**: 2025-02-05
**Status**: Completed
**Verdict**: SUITABLE ✅ with caveats

---

## Quick Answer

| Use Case | DGX Spark Compatible? | Notes |
|----------|----------------------|-------|
| **Script Writing (LLaMA 8B)** | ✅ Excellent | 20+ tok/s decode, runs easily |
| **Storyboard (FLUX 12B)** | ✅ Excellent | 1K image every 2.6s at FP4 |
| **Kitchen Cooking Assistant** | ✅ Excellent | Qwen2.5-VL 7B + YOLOv8 fits easily |
| **Voice Pipeline (STT+LLM+TTS)** | ✅ Good | 600ms-1s end-to-end demonstrated |
| **Large Models (70B+)** | ✅ Unique advantage | 128GB unified memory |
| **ALL models simultaneously** | ✅ Yes | ~58GB used, ~70GB remaining |

**Main Trade-off**: Slower token generation than RTX 4090/5090, but can load MUCH larger models.

---

## DGX Spark Specifications

| Spec | Value |
|------|-------|
| **Chip** | GB10 Grace Blackwell Superchip |
| **CPU** | 20-core ARM (10× Cortex-X925 + 10× Cortex-A725) |
| **GPU** | Blackwell (6,144 CUDA cores) |
| **Memory** | 128GB unified LPDDR5x |
| **Memory Bandwidth** | 273 GB/s |
| **AI Performance** | 1 PFLOP (FP4) |
| **Storage** | 4TB NVMe SSD |
| **Power** | 240W (typical ~170W under load) |
| **Size** | 150 × 150 × 50.5 mm (~1.2 kg) |
| **Price** | $3,999 - $4,300 |

---

## Benchmark Results

### Script Writing Model (LLaMA 3.1 8B)

| Metric | DGX Spark | RTX 4090 |
|--------|-----------|----------|
| **Prefill (prompt processing)** | 7,991 tok/s | Similar |
| **Decode (generation)** | 20-38 tok/s | ~60-80 tok/s |
| **Can load model?** | ✅ Easily | ✅ Easily |

**Verdict**: ✅ Works great for 8B models. Token generation is memory-bandwidth limited but still responsive.

### Storyboard Model (FLUX/SDXL)

| Model | Performance |
|-------|-------------|
| **FLUX 12B (FP4)** | 1K image every 2.6 seconds |
| **SDXL 1.0 (BF16)** | 7 images/minute at 1K resolution |

**Verdict**: ✅ Excellent. ComfyUI runs natively. FLUX Dreambooth LoRA fine-tuning supported.

### Voice Pipeline (STT + LLM + TTS)

Demonstrated real-time voice agent on DGX Spark:

| Component | Latency |
|-----------|---------|
| **Whisper STT** | 100-200ms per utterance |
| **LLM inference** | 300-600ms per response |
| **TTS synthesis** | 100-150ms per utterance |
| **End-to-end** | 600ms - 1 second |

**Reference**: [spark-voice-pipeline](https://github.com/Logos-Flux/spark-voice-pipeline) achieves 766ms to first audio.

**Verdict**: ✅ Meets our sub-800ms target. All three models run simultaneously.

---

## DGX Spark's Unique Advantage: 128GB Unified Memory

This is where DGX Spark shines vs consumer GPUs:

| Model Size | RTX 4090 (24GB) | RTX 5090 (32GB) | DGX Spark (128GB) |
|------------|-----------------|-----------------|-------------------|
| LLaMA 8B | ✅ | ✅ | ✅ |
| LLaMA 70B | ❌ OOM | ❌ OOM | ✅ Loads fine |
| DeepSeek R1 70B | ❌ Crash | ❌ Crash | ✅ Runs |
| GPT-OSS 120B | ❌ | ❌ | ✅ |
| 200B models | ❌ | ❌ | ✅ |

**Key Insight**: You can load a 70B or 120B model that would crash on any consumer GPU.

---

## Multi-Model Simultaneous Operation

DGX Spark can run multiple models in parallel:

```
┌─────────────────────────────────────────────────────────────┐
│                 DGX Spark (128GB Unified)                    │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐       │
│  │ LLaMA 8B     │  │ FLUX 12B     │  │ Whisper      │       │
│  │ (Friday)     │  │ (Storyboard) │  │ + TTS        │       │
│  │ ~16GB FP16   │  │ ~24GB FP8    │  │ ~8GB         │       │
│  └──────────────┘  └──────────────┘  └──────────────┘       │
│                                                              │
│  Remaining: ~80GB for context, caching, additional models   │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

**NVIDIA's Multi-Agent Playbook** confirms: DGX Spark can run multiple LLMs and VLMs in parallel.

---

## Comparison: DGX Spark vs Alternatives

### Option 1: DGX Spark ($4,000)

| Pros | Cons |
|------|------|
| 128GB unified memory | Slower token generation (273 GB/s bandwidth) |
| Runs 70B+ models | More expensive than DIY |
| Low power (240W) | ARM-based (some software compatibility) |
| Compact, quiet | LPDDR5x is slower than HBM/GDDR6X |
| NVIDIA software stack | |
| Multi-model friendly | |

### Option 2: RTX 4090 Build (~$3,500-4,000)

| Pros | Cons |
|------|------|
| Faster token generation (~80 tok/s) | Only 24GB VRAM |
| 1 TB/s memory bandwidth | Cannot load 70B+ models |
| Mature ecosystem | Higher power (~450W GPU alone) |
| Can upgrade to 5090 later | Louder |

### Option 3: 2× RTX 3090 (~$2,000-2,500)

| Pros | Cons |
|------|------|
| 48GB total VRAM | Requires model sharding |
| Cheaper | Very high power (~700W) |
| Fast generation | Complex setup |
| | Loud, hot |

### Option 4: DGX Station (~$50,000+?)

| Pros | Cons |
|------|------|
| 784GB memory | Overkill for your needs |
| 20 PFLOPS | Enterprise pricing |
| Runs 1T parameter models | |

---

## Friday Pre-Production Workflow Fit

### Use Case 1: Script Writing (Friday LLM)

**Model**: LLaMA 3.1 8B + Telugu LoRA + Persona LoRA

| Aspect | DGX Spark Capability |
|--------|---------------------|
| Model loading | ✅ Trivial (8B fits easily) |
| Inference speed | ✅ 20-38 tok/s (acceptable) |
| Context window | ✅ Large context supported |
| LoRA hot-swap | ✅ Memory allows multiple adapters |

### Use Case 2: Storyboard Generation

**Model**: FLUX 12B or SDXL with LoRAs

| Aspect | DGX Spark Capability |
|--------|---------------------|
| Image generation | ✅ 1K image in 2.6s (FLUX FP4) |
| ComfyUI | ✅ Native support |
| LoRA fine-tuning | ✅ Dreambooth LoRA playbook available |
| Batch generation | ✅ 7 images/min (SDXL) |

### Use Case 3: Kitchen Cooking Assistant (Kitch Model)

**Models**: Qwen2.5-VL 7B (Vision-Language) + YOLOv8 (Food Detection)

| Aspect | DGX Spark Capability |
|--------|---------------------|
| Ingredient recognition | ✅ Camera → YOLOv8 → identify foods |
| Recipe generation | ✅ Qwen2.5-VL understands images + generates text |
| Voice interaction | ✅ Share voice pipeline with Friday |
| Real-time guidance | ✅ Step-by-step cooking instructions |
| Memory footprint | ✅ ~8-10GB total (fits easily) |

#### Recommended Kitchen Model Stack

```
┌─────────────────────────────────────────────────────────────┐
│                Kitchen Cooking Assistant                     │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  Camera Input ──▶ YOLOv8 (Food Detection) ──▶ Ingredient List│
│       │                  (~1-2GB)                            │
│       │                                                      │
│       └──────▶ Qwen2.5-VL 7B (Vision-Language)              │
│                     (~6-8GB)                                 │
│                        │                                     │
│                        ▼                                     │
│              Recipe Generation / Cooking Guidance            │
│                        │                                     │
│                        ▼                                     │
│              Voice Output (shared TTS pipeline)              │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

#### Model Options for Kitchen

| Model | Size | Purpose | Notes |
|-------|------|---------|-------|
| **Qwen2.5-VL 7B** | ~6-8GB | Vision + language | Best balance, can see fridge contents |
| **YOLOv8-food** | ~1-2GB | Real-time detection | 97% accuracy on food items |
| **FoodSky** | ~7B | Food-specific LLM | Passes chef exams (Chinese) |
| **TinyLLaMA** | ~1GB | Lightweight fallback | For simple queries |

#### Kitchen Hardware Requirements

| Item | Purpose | Est. Cost |
|------|---------|-----------|
| **Wide-angle Camera** | See countertop/fridge | $50-100 |
| **Waterproof Speaker** | Kitchen audio output | $50-100 |
| **Waterproof Mic** | Voice commands while cooking | $30-50 |
| **Display (optional)** | Show recipe steps visually | $100-200 |

### Use Case 4: Multi-Room Voice (Writers Room, Kitchen, etc.)

**Models**: Whisper STT + Friday LLM + TTS (Chatterbox/IndicF5)

| Aspect | DGX Spark Capability |
|--------|---------------------|
| Real-time voice | ✅ 600ms-1s end-to-end |
| All models simultaneous | ✅ 128GB allows all |
| Wake word | ✅ CPU handles this |
| Room-aware routing | ✅ Kitchen queries → Kitch model |

---

## Complete Memory Budget (All Models Loaded)

```
┌─────────────────────────────────────────────────────────────┐
│              DGX Spark 128GB - Full Allocation               │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌──────────────────┐  ┌──────────────────┐                 │
│  │ Friday LLaMA 8B  │  │ FLUX 12B         │                 │
│  │ + Telugu LoRA    │  │ (Storyboard)     │                 │
│  │ ~16GB FP16       │  │ ~24GB FP8        │                 │
│  └──────────────────┘  └──────────────────┘                 │
│                                                              │
│  ┌──────────────────┐  ┌──────────────────┐                 │
│  │ Qwen2.5-VL 7B    │  │ Voice Pipeline   │                 │
│  │ + YOLOv8         │  │ Whisper + TTS    │                 │
│  │ (Kitchen)        │  │                  │                 │
│  │ ~10GB            │  │ ~8GB             │                 │
│  └──────────────────┘  └──────────────────┘                 │
│                                                              │
│  TOTAL USED: ~58GB                                          │
│  REMAINING:  ~70GB (for context, caching, future models)    │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

**Verdict**: ✅ ALL FOUR USE CASES FIT COMFORTABLY

---

## Additional Hardware Requirements

If you get DGX Spark, you'll also need:

### For Voice Pipeline (All Rooms)

| Item | Purpose | Est. Cost |
|------|---------|-----------|
| **USB Microphone** (per room) | Voice input | $50-150 |
| **Speakers** (per room) | TTS output | $50-150 |
| **Network switch** | Connect all rooms | $50-100 |

### For Storyboard Workflow

| Item | Purpose | Est. Cost |
|------|---------|-----------|
| **4K Monitor** | Review generated images | $300-500 |
| **Drawing Tablet** (optional) | Sketch input | $100-300 |

### For Development

| Item | Purpose | Est. Cost |
|------|---------|-----------|
| **External SSD** | Backup/model storage | $100-200 |
| **UPS** | Power protection | $100-200 |

### Network (For Linking 2× DGX Spark)

If you want to run 405B models by linking two DGX Sparks:

| Item | Purpose | Est. Cost |
|------|---------|-----------|
| **Second DGX Spark** | Double memory (256GB) | $4,000 |
| **QSFP cable** | 200Gbps link | $50-100 |

---

## My Recommendation

### For Your Current Needs: DGX Spark is SUITABLE ✅

**Why it works**:
1. **Script writing**: 8B model runs great
2. **Storyboard**: FLUX/SDXL work well, ComfyUI supported
3. **Voice**: Demonstrated sub-800ms pipeline
4. **Multi-model**: 128GB handles all simultaneously
5. **Quiet & compact**: Fits in Writers Room without noise
6. **Power efficient**: 240W vs 1000W+ for multi-GPU

**Why it might NOT be ideal**:
1. **Token speed**: If you need blazing fast generation, RTX 4090/5090 is faster
2. **Price/perf ratio**: DIY multi-GPU can be cheaper for raw speed
3. **ARM ecosystem**: Some tools may have compatibility issues

### Alternative Consideration

If budget is flexible and you want maximum flexibility:

**Option A: DGX Spark + RTX 4090 workstation**
- DGX Spark: Development, multi-model, large models
- RTX 4090: Fast inference when you need speed
- Total: ~$8,000

**Option B: Single RTX 5090 build**
- 32GB VRAM (still can't do 70B)
- Much faster for 8B models
- Cheaper (~$3,000-3,500 full build)
- But loses multi-model advantage

---

## Software Stack on DGX Spark

Pre-configured for AI development:

| Component | Included |
|-----------|----------|
| **OS** | DGX OS (Ubuntu-based) |
| **Containers** | Docker, NGC containers |
| **LLM Serving** | vLLM, TensorRT-LLM |
| **Image Gen** | ComfyUI, Diffusers |
| **Dev Tools** | JupyterLab, VS Code |
| **Playbooks** | Pre-built AI workflows |

---

## Sources

### DGX Spark Specifications
- [NVIDIA DGX Spark Official](https://marketplace.nvidia.com/en-us/enterprise/personal-ai-supercomputers/dgx-spark/)
- [DGX Spark User Guide](https://docs.nvidia.com/dgx/dgx-spark/dgx-spark.pdf)
- [Micro Center Product Page](https://www.microcenter.com/product/699008/nvidia-dgx-spark)

### Benchmarks
- [LMSYS In-Depth Review](https://lmsys.org/blog/2025-10-13-nvidia-dgx-spark/)
- [Hardware Corner Benchmarks](https://www.hardware-corner.net/first-dgx-spark-llm-benchmarks/)
- [IntuitionLabs Review](https://intuitionlabs.ai/articles/nvidia-dgx-spark-review)
- [llama.cpp Discussion](https://github.com/ggml-org/llama.cpp/discussions/16578)

### Voice Pipeline
- [spark-voice-pipeline](https://github.com/Logos-Flux/spark-voice-pipeline)
- [Multilingual Voice Agent](https://www.genaiprotos.com/project/multilingual-voice-agent/)
- [NVIDIA Open Voice Agents](https://www.daily.co/blog/building-voice-agents-with-nvidia-open-models/)

### Image Generation
- [ComfyUI on DGX Spark](https://build.nvidia.com/spark/comfy-ui)
- [FLUX Fine-tuning Playbook](https://build.nvidia.com/spark/flux-finetuning)
- [NVIDIA Performance Blog](https://developer.nvidia.com/blog/how-nvidia-dgx-sparks-performance-enables-intensive-ai-tasks/)

### Comparisons
- [DGX Spark vs Alternatives](https://research.aimultiple.com/dgx-spark-alternatives/)
- [ProxPC Performance Test](https://www.proxpc.com/blogs/nvidia-dgx-spark-gb10-performance-test-vs-5090-llm-image-and-video-generation)

### Kitchen/Cooking AI Models
- [FoodSky LLM](https://pmc.ncbi.nlm.nih.gov/articles/PMC12142648/) - Food-specific LLM passing chef exams
- [ChefAssistAI (RAG-based)](https://github.com/sam23121/ChefAssistAI/) - Open-source cooking assistant
- [Chef Dalle Multimodal](https://www.mdpi.com/2073-431X/13/7/156) - Voice + Vision + Text cooking system
- [ARChef with Gemini](https://www.promptlayer.com/research-papers/your-ai-sous-chef-cooking-with-augmented-reality) - AR cooking assistant
- [SmartChef AI](https://smartchefmobile.com/) - Camera-based ingredient recognition
- [Qwen2.5-VL Local Deployment](https://www.labellerr.com/blog/run-qwen2-5-vl-locally/) - Vision-language model guide
- [YOLOv8 Hardware Requirements](https://www.proxpc.com/blogs/system-hardware-requirements-for-yolo-in-2025)

---

*Research conducted: 2025-02-05*
