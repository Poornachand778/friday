# Research: Chatterbox TTS for Friday Voice Pipeline

**Date**: 2025-02-05
**Status**: Completed
**Verdict**: STRONG CANDIDATE ⭐ (with Telugu fine-tuning required)

---

## Clarification: Not DeepSeek

**Chatterbox is from Resemble AI, not DeepSeek.** DeepSeek has not released a TTS model. Resemble AI released Chatterbox as their first production-grade open-source TTS.

---

## Chatterbox Overview

| Aspect | Details |
|--------|---------|
| **Developer** | Resemble AI |
| **License** | MIT (commercial use OK) ✅ |
| **Architecture** | 0.5B Llama backbone |
| **Training Data** | 500,000 hours cleaned speech |
| **Languages** | 23 (no Telugu) |
| **Voice Cloning** | 5-10 seconds reference audio |
| **Latency** | Sub-200ms ✅ |
| **Quality Score** | 95/100 (vs XTTS v2: 75/100) |

---

## Model Variants

| Model | Parameters | Speed | Use Case |
|-------|------------|-------|----------|
| **Chatterbox** | 0.5B | Standard | High quality |
| **Chatterbox-Turbo** | 350M | 10x faster | Production agents |
| **Chatterbox-Multilingual** | 0.5B | Standard | 23 languages |

### Turbo Advantages
- Distilled decoder: 10 steps → 1 step
- Lower VRAM requirements
- Paralinguistic tags: `[laugh]`, `[cough]`, `[chuckle]`
- Sub-200ms inference

---

## Language Support

### Supported (23 languages)
Arabic, Danish, German, Greek, **English**, Spanish, Finnish, French, Hebrew, **Hindi**, Italian, Japanese, Korean, Malay, Dutch, Norwegian, Polish, Portuguese, Russian, Swedish, Swahili, Turkish, Chinese

### NOT Supported
❌ **Telugu** - Critical gap for Friday

---

## Quality Benchmarks

| Comparison | Result |
|------------|--------|
| **Chatterbox vs ElevenLabs** | 63.75% preferred Chatterbox |
| **Chatterbox vs XTTS v2** | 95/100 vs 75/100 |
| **Chatterbox vs ElevenLabs (latency)** | Both sub-200ms |

### Unique Features
1. **Emotion Exaggeration Control** - First open-source TTS with this
2. **Paralinguistic Tags** - Natural `[laugh]`, `[sigh]` insertions
3. **PerTh Watermarking** - Imperceptible neural watermarks (safety)

---

## VRAM Requirements

| Use Case | VRAM | Hardware Example |
|----------|------|------------------|
| **Inference (light)** | 6-8GB | RTX 3060Ti |
| **Inference (production)** | 16-24GB | RTX 4090 |
| **LoRA Fine-tuning** | 18GB+ | RTX 4090 |
| **Full Fine-tuning** | 24GB+ | A100 / H100 |

---

## Telugu Support Options

### Option 1: Fine-tune Chatterbox for Telugu

**Tool**: [chatterbox-finetuning](https://github.com/gokhaneraslan/chatterbox-finetuning)

**What's needed**:
- Telugu speech dataset (audio + transcriptions)
- 18GB+ VRAM for LoRA fine-tuning
- Smart vocabulary extension (toolkit handles this)

**Training Config**:
```python
# For 1 hour of target speaker audio
epochs = 150  # or 1000 steps
batch_size = 4  # for 12GB VRAM
batch_size = 2  # for lower VRAM, with grad_accum=32
```

**Pros**: Best quality, full control, MIT license
**Cons**: Requires Telugu dataset + training effort

### Option 2: Use IndicF5 for Telugu

**Model**: [AI4Bharat/IndicF5](https://huggingface.co/ai4bharat/IndicF5)

| Aspect | Details |
|--------|---------|
| **Telugu Support** | ✅ Native |
| **Voice Cloning** | ✅ Via reference audio |
| **Languages** | 11 Indian languages |
| **Architecture** | F5-TTS optimized for Indic |
| **Output** | 24kHz audio |
| **Latency** | Unknown (not published) |

**Pros**: Works out of the box for Telugu
**Cons**: No latency benchmarks, research-grade

### Option 3: Indic Parler-TTS (NO Voice Cloning)

**Model**: [AI4Bharat/indic-parler-tts](https://huggingface.co/ai4bharat/indic-parler-tts)

| Aspect | Details |
|--------|---------|
| **Telugu Support** | ✅ Native (21 languages) |
| **Voice Cloning** | ❌ Explicitly not supported |
| **Quality** | MOS 4.5, WER 5.2% (best in class) |
| **Voices** | 69 preset voices |

**Pros**: Highest quality for Indic languages
**Cons**: Cannot clone Boss's voice

---

## Comparison: Chatterbox vs XTTS v2 vs IndicF5

| Feature | Chatterbox | XTTS v2 | IndicF5 |
|---------|------------|---------|---------|
| **License** | MIT ✅ | Non-commercial ❌ | Apache 2.0 ✅ |
| **Telugu** | ❌ (fine-tune needed) | ❌ | ✅ Native |
| **Voice Cloning** | ✅ 5-10s audio | ✅ 6s audio | ✅ Reference audio |
| **Latency** | Sub-200ms ✅ | ~300ms | Unknown |
| **Quality** | 95/100 | 75/100 | Good (no benchmark) |
| **Emotion Control** | ✅ | Partial | ❌ |
| **Maintenance** | Active ✅ | Community only | Active ✅ |
| **Hindi** | ✅ | ✅ | ✅ |

---

## Recommendation for Friday

### Best Path: Hybrid Approach

```
┌─────────────────────────────────────────────────────────────┐
│                Friday Voice Pipeline (Revised)               │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  Language Detection (from LLM response)                      │
│         │                                                    │
│         ├── English/Hindi ──▶ Chatterbox-Turbo              │
│         │                     (sub-200ms, emotion control)   │
│         │                                                    │
│         └── Telugu ──────────▶ IndicF5                       │
│                                (native support, voice clone) │
│                                                              │
│  Both use Boss's voice clone (different reference formats)   │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### Why This Works

| Need | Solution |
|------|----------|
| **Telugu voice** | IndicF5 (native support) |
| **English quality** | Chatterbox-Turbo (best-in-class) |
| **Voice cloning** | Both support reference-based cloning |
| **Low latency** | Chatterbox: sub-200ms, IndicF5: TBD |
| **Commercial use** | Both MIT/Apache licensed |

### Alternative: Single Model (More Effort)

Fine-tune Chatterbox for Telugu:
1. Collect Telugu speech data (Boss's voice recordings work here too)
2. Use chatterbox-finetuning toolkit
3. Extend vocabulary for Telugu characters
4. Train LoRA (18GB VRAM, ~150 epochs)

**Result**: Single model for all languages, best quality, but requires training.

---

## Quick Start

### Chatterbox Installation
```bash
pip install chatterbox-tts
```

### Basic Usage
```python
from chatterbox import ChatterboxTTS

model = ChatterboxTTS()
audio = model.generate(
    text="Boss, baagunnanu. Inka em kavali?",
    voice="path/to/boss_reference.wav",
    emotion_scale=1.2  # Slight emotion boost
)
```

### IndicF5 Installation
```bash
pip install indicf5
# or clone from https://github.com/AI4Bharat/IndicF5
```

### IndicF5 Usage
```python
from indicf5 import IndicF5

model = IndicF5()
audio = model.generate(
    text="బాస్, బాగున్నాను. ఇంకేం కావాలి?",
    reference_audio="path/to/boss_telugu.wav",
    reference_text="నేను ఫ్రైడే, మీ అసిస్టెంట్"
)
```

---

## Action Items

1. **Benchmark IndicF5 latency** - Critical unknown
2. **Collect Boss voice samples**:
   - 5-10 minutes Telugu speech
   - 5-10 minutes English speech
   - Various emotions (calm, excited, thoughtful)
3. **Test Chatterbox-Turbo** for English/Hindi portions
4. **Build language router** in voice daemon
5. **Optional**: Fine-tune Chatterbox for Telugu if IndicF5 latency is too high

---

## XTTS v2 Deprecation Recommendation

Given:
- Coqui (company) shut down in 2024
- Non-commercial license limits production use
- Lower quality (75/100 vs 95/100)
- No emotion control

**Recommendation**: Replace XTTS v2 with Chatterbox + IndicF5 hybrid.

---

## Sources

### Chatterbox
- [GitHub - resemble-ai/chatterbox](https://github.com/resemble-ai/chatterbox)
- [Resemble AI Official](https://www.resemble.ai/chatterbox/)
- [Chatterbox-Turbo](https://www.resemble.ai/chatterbox-turbo/)
- [Chatterbox Fine-tuning Toolkit](https://github.com/gokhaneraslan/chatterbox-finetuning)
- [DigitalOcean Tutorial](https://www.digitalocean.com/community/tutorials/resemble-chatterbox-tts-text-to-speech)
- [BentoML TTS Comparison 2026](https://www.bentoml.com/blog/exploring-the-world-of-open-source-text-to-speech-models)

### Telugu/Indic TTS
- [AI4Bharat IndicF5](https://huggingface.co/ai4bharat/IndicF5)
- [AI4Bharat Indic Parler-TTS](https://huggingface.co/ai4bharat/indic-parler-tts)
- [AI4Bharat TTS Hub](https://ai4bharat.iitm.ac.in/areas/tts)

### Comparisons
- [XTTS v2 vs Chatterbox Turbo](https://kugu.ai/compare/xtts-v2-vs-chatterbox-turbo)
- [Inferless TTS Comparison](https://www.inferless.com/learn/comparing-different-text-to-speech---tts--models-part-2)

---

*Research conducted: 2025-02-05*
