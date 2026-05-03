# Research: Telugu QLoRA Adapters + Adapter Stacking for Friday

**Date**: 2025-02-04
**Status**: Completed
**Verdict**: DOABLE ✅

---

## The Proposal

```
Base Model: LLaMA 3.1 8B Instruct
      ↓
+ Telugu Language LoRA (open-source, if available)
      ↓
+ Friday Persona LoRA (trained on behavioral data)
      ↓
= Friday: Telugu-fluent, thinks like Boss
      ↓
+ Voice (TTS/STT)
= Real-time voice conversation
```

---

## 1. Telugu LoRA Adapters: What's Available?

| Resource | Type | Telugu Support | Notes |
|----------|------|----------------|-------|
| [Telugu-LLM-Labs](https://huggingface.co/Telugu-LLM-Labs) | HF Organization | ✅ Dedicated | Ravi Theja & Ramsri Goutham's Telugu-focused models/datasets |
| [AI4Bharat IndicBERT](https://huggingface.co/ai4bharat/indic-bert) | Encoder | ✅ Yes | 12 Indic languages including Telugu |
| [AI4Bharat IndicBART](https://huggingface.co/ai4bharat/IndicBART) | Seq2Seq | ✅ Yes | YANMTT toolkit for fine-tuning |
| [IndicTrans2](https://github.com/AI4Bharat/IndicTrans2) | Translation | ✅ Yes | **LoRA fine-tuning scripts available** |
| [Sarvam-M](https://www.sarvam.ai/blogs/sarvam-m) | Hybrid LLM | ✅ 8% Telugu | Production-ready Indic reasoning model |
| [IndicLLMSuite](https://github.com/AI4Bharat/IndicLLMSuite) | Dataset | ✅ Yes | 74.7M prompt-response pairs for SFT |

**Finding**: No direct "Telugu LoRA for LLaMA 3.1 8B" exists as a plug-and-play adapter. However, **IndicLLMSuite** provides Telugu SFT data (IndicAlign) that can be used to train a Telugu adapter.

### Key Datasets for Telugu LoRA Training

- **Sangraha**: 251B tokens across 22 Indic languages (pretraining data)
- **IndicAlign**: 74.7M prompt-response pairs for instruction fine-tuning
- Telugu-LLM-Labs datasets on Hugging Face

---

## 2. Adapter Stacking: Methods

| Method | How It Works | Latency Impact | Best For |
|--------|--------------|----------------|----------|
| **Merge at export** | Permanently fuse adapters into base model | None | Single persona, production |
| **Dynamic loading** | Switch adapters per request | ~2x slower | Multi-persona serving |
| **LoRA Composition** | Weighted blend of multiple adapters | Minimal | Combining skills |

### Recommended Tools

- [LoRAX Merging](https://loraexchange.ai/guides/merging_adapters/) - Multiple merge strategies (linear, TIES)
- [LLaMA Factory merge](https://llamafactory.readthedocs.io/en/latest/getting_started/merge_lora.html) - `llamafactory-cli export merge_config.yaml`
- [vLLM LoRA](https://docs.vllm.ai/en/latest/features/lora/) - Production serving with `--lora-modules`

### Advanced Techniques (2025)

- **LoRA-Switch**: Token-wise routing, 2.4x latency reduction
- **IBM aLoRA**: Activated LoRAs, 20-30x faster per task by reusing KV cache
- **LoRAServe**: Dynamic placement across GPUs for heterogeneous adapters

---

## 3. Voice Conversation Latency

### Target: < 800ms end-to-end (natural conversation threshold)

| Component | Target Latency | Achievable? | Solution |
|-----------|---------------|-------------|----------|
| Microphone → ASR | 40ms | ✅ | NVIDIA Nemotron: sub-25ms |
| ASR → LLM | 300ms | ✅ | Quantized 8B local: ~200-400ms |
| LLM → TTS | 150ms | ✅ | Microsoft VibeVoice: ~300ms TTFT |
| **Total** | **< 800ms** | ✅ | Achievable with proper setup |

### 2025 Voice AI Benchmarks

- Sub-200ms speech-to-speech now possible
- 4-bit quantization: **40% latency reduction** with minimal quality loss
- NVIDIA voice agents achieve real-time with open models

### Recommended Stack for Friday

```
Hardware: RTX 4090 (24GB)
Quantization: 4-bit (bitsandbytes)
Inference: vLLM with merged LoRA
ASR: Faster-Whisper (local)
TTS: XTTS v2 or Indic Parler-TTS (Telugu support)
Target: < 600ms end-to-end
```

---

## 4. Implementation Path

### Step 1: Train Telugu LoRA
```bash
# Using IndicAlign Telugu subset
# Framework: LLaMA-Factory or Unsloth
# Base: meta-llama/Meta-Llama-3.1-8B-Instruct
# Output: telugu_lora_adapter/
```

### Step 2: Train Persona LoRA
```bash
# Using Phase 2 behavioral data (INVESTIGATOR, CRITIC, etc.)
# Framework: Same as Telugu LoRA
# Output: friday_persona_lora_adapter/
```

### Step 3: Merge Adapters
```yaml
# merge_config.yaml for LLaMA-Factory
model_name_or_path: meta-llama/Meta-Llama-3.1-8B-Instruct
adapter_name_or_path:
  - telugu_lora_adapter
  - friday_persona_lora_adapter
export_dir: friday_merged_model
```

### Step 4: Deploy with Voice
```python
# vLLM serving
vllm serve friday_merged_model --quantization bitsandbytes

# Voice pipeline
ASR: Faster-Whisper large-v3
TTS: XTTS v2 with Boss voice clone
Wake word: OpenWakeWord "Hey Friday"
```

---

## 5. Verdict

| Question | Answer |
|----------|--------|
| Can we find Telugu LoRA? | Not plug-and-play, but **data available to train one** |
| Can we stack Telugu + Persona LoRAs? | **YES** - merge or compose at inference |
| Will it talk/think/decide like Boss? | **YES** - if Phase 2 behavioral data is quality |
| Latency-free voice? | **YES** - sub-800ms achievable with 4-bit + local inference |

---

## Sources

### Telugu/Indic Resources
- https://huggingface.co/Telugu-LLM-Labs
- https://huggingface.co/ai4bharat/indic-bert
- https://github.com/AI4Bharat/IndicLLMSuite
- https://www.sarvam.ai/blogs/sarvam-m
- https://github.com/AI4Bharat/IndicTrans2

### LoRA Stacking
- https://loraexchange.ai/guides/merging_adapters/
- https://llamafactory.readthedocs.io/en/latest/getting_started/merge_lora.html
- https://docs.vllm.ai/en/latest/features/lora/
- https://research.ibm.com/blog/inference-friendly-aloras-lora
- https://medium.com/@abheshith7/multi-lora-and-lora-composition-the-ultimate-guide-with-diagrams

### Voice Latency
- https://arxiv.org/html/2508.04721v1
- https://www.daily.co/blog/building-voice-agents-with-nvidia-open-models/
- https://huggingface.co/microsoft/VibeVoice-Realtime-0.5B
- https://www.videosdk.live/developer-hub/llm/llm-api-for-real-time-voice

---

*Research conducted: 2025-02-04*
