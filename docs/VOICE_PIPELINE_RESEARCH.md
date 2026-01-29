# Friday AI - Voice Pipeline Research Summary

**Date:** January 2026
**Status:** Research Complete - Ready for Testing

---

## Executive Summary

The voice pipeline infrastructure is already well-implemented. This document summarizes research findings and provides actionable steps for MacBook testing while waiting for AWS access for iteration 2 training.

---

## 1. Current Implementation Status

### Files Already Implemented

| Component | File | Status |
|-----------|------|--------|
| **STT Service** | `voice/stt/faster_whisper_service.py` | Complete |
| **TTS Service** | `voice/tts/xtts_service.py` | Complete |
| **Wake Word** | `voice/wakeword/openwakeword_service.py` | Complete |
| **Voice Daemon** | `voice/daemon.py` | Complete |
| **Config System** | `voice/config.py` | Complete |
| **Audio Capture** | `voice/audio/capture.py` | Complete |
| **Audio Playback** | `voice/audio/playback.py` | Complete |
| **VAD** | `voice/audio/vad.py` | Complete |
| **Audio Storage** | `voice/storage/audio_storage.py` | Complete |

### Dependencies (Already in requirements.txt)

```
# Voice Pipeline - STT
faster-whisper>=0.10.0
ctranslate2>=4.0.0

# Voice Pipeline - TTS
TTS>=0.22.0                        # Coqui XTTS v2

# Voice Pipeline - Wake Word
openwakeword>=0.5.0
onnxruntime>=1.16.0

# Voice Pipeline - Audio I/O
sounddevice>=0.4.6
soundfile>=0.12.1
webrtcvad>=2.0.10
```

---

## 2. XTTS v2 Research Findings

### Voice Cloning Capability

- **Minimum Audio Required:** 6 seconds of reference audio
- **Supported Languages:** 17 languages including Hindi (closest to Telugu)
- **Telugu Support:** NOT native - uses Hindi fallback (same script family)
- **Quality:** High quality with proper reference audio

### MacBook Installation (Apple Silicon)

The `coqui-tts` package now supports macOS with prebuilt wheels (v0.24.2+).

**Recommended Installation:**
```bash
# Create conda environment
conda create --name friday_voice python=3.9
conda activate friday_voice

# Install TTS
pip install TTS>=0.22.0
```

### Current Implementation Notes

From `voice/tts/xtts_service.py`:
- Uses Hindi (`hi`) as fallback for Telugu (`te`)
- Supports voice cloning via `speaker_wav` parameter
- Model: `tts_models/multilingual/multi-dataset/xtts_v2`
- Sample rate: 24000 Hz

### Voice Profile System

The TTS service supports loading voice profiles:
```python
tts = XTTSService()
tts.load_voice_profile("friday_airy", ["path/to/airy_sample.wav"])
result = tts.synthesize("Boss, em kavali?", language="te", profile="friday_airy")
```

### "Airy" Voice Preparation

User will provide reference recording. Requirements:
- **Duration:** 6+ seconds (more is better for quality)
- **Format:** WAV, 16-bit, mono
- **Content:** Clear speech in desired voice style
- **Environment:** Quiet recording, minimal background noise
- **Language:** Can include Telugu phrases for better pronunciation

---

## 3. Faster-Whisper (STT) Research Findings

### Telugu-English Code-Switching Challenges

Research shows Whisper has limitations with code-switching:
- Standard Whisper shows poor performance on mixed-language sentences
- Extended versions with language detectors improve this

### Recommended Approach for Telugu-English

1. **Use large-v3 model** for best multilingual performance
2. **Enable VAD filter** to improve accuracy
3. **Word timestamps** help identify language boundaries
4. **Consider Pingala V1** as alternative for Indic code-switching

### Current Implementation

From `voice/stt/faster_whisper_service.py`:
- Supports all Whisper model sizes (tiny to large-v3)
- Auto-detects Telugu/English
- Word-level timestamps available
- VAD filtering enabled by default

### MacBook Device Settings

For MacBook (no CUDA), update config:
```yaml
stt:
  device: cpu
  compute_type: float32  # or int8 for faster but less accurate
  model: medium  # Start with medium, upgrade to large-v3 if CPU handles it
```

---

## 4. OpenWakeWord Research Findings

### Custom Wake Word Training

OpenWakeWord can train custom models with 100% synthetic speech:
- Uses neural TTS (Piper) to generate training clips
- Training takes ~45 minutes in Colab notebook
- No manual data collection needed

### Wake Words for Friday

| Wake Phrase | Status | Notes |
|-------------|--------|-------|
| "hey_friday" | Need to train | Primary wake word |
| "wake_up_daddys_home" | Need to train | Custom phrase |
| "hey_jarvis" | Built-in | Can use for initial testing |

### Training Process

1. Use OpenWakeWord Colab notebook
2. Generate synthetic speech for target phrase
3. Train small model on frozen feature extractor
4. Export as `.onnx` or `.tflite`
5. Place in `voice/models/wakeword/`

### Implementation Notes

- Audio must be 16kHz, 1820 sample chunks (or multiple)
- Built-in models: alexa, hey_jarvis, hey_mycroft, timer, weather

---

## 5. MacBook Testing Plan

### Phase 1: Install Dependencies

```bash
# Activate environment
conda activate friday_ft  # or create new voice env

# Install voice pipeline dependencies
pip install faster-whisper TTS openwakeword sounddevice soundfile webrtcvad PyYAML
```

### Phase 2: Create MacBook Config

Create `config/voice_config_macbook.yaml`:

```yaml
audio:
  sample_rate: 16000
  channels: 1
  dtype: int16
  chunk_size: 512
  vad_aggressiveness: 2
  silence_threshold_ms: 500
  max_recording_seconds: 30

stt:
  engine: faster_whisper
  model: medium          # Use medium for CPU
  device: cpu            # MacBook has no CUDA
  compute_type: float32  # CPU-compatible
  language: null         # Auto-detect
  word_timestamps: true
  vad_filter: true
  beam_size: 5

tts:
  engine: xtts_v2
  device: cpu            # CPU for MacBook
  default_language: te
  default_profile: friday_airy
  speed: 1.0

wakeword:
  engine: openwakeword
  models:
    - name: hey_jarvis   # Use built-in for initial test
      threshold: 0.5
  inference_framework: onnx

storage:
  enabled: true
  base_path: voice/data/recordings
  organize_by_date: true
  save_user_audio: true
  save_response_audio: true

daemon:
  mode: standalone
  auto_start: false
  idle_timeout_seconds: 300
  location: macbook_dev
```

### Phase 3: Test Individual Components

**Test 1: STT Only**
```python
from voice.stt import FasterWhisperSTT
from voice.config import STTConfig

config = STTConfig(model="medium", device="cpu", compute_type="float32")
stt = FasterWhisperSTT(config)

# Test with audio file
result = stt.transcribe("test_audio.wav")
print(f"Text: {result.text}")
print(f"Language: {result.language}")
```

**Test 2: TTS Only**
```python
from voice.tts import XTTSService
from voice.config import TTSConfig

config = TTSConfig(device="cpu")
tts = XTTSService(config)

# Test English
result = tts.synthesize("Boss, what's next?", language="en")

# Test romanized Telugu (using Hindi model)
result = tts.synthesize("Boss, emi kavali?", language="hi")

# Save to file
from voice.tts.xtts_service import save_audio
save_audio(result.audio, "test_output.wav", result.sample_rate)
```

**Test 3: Wake Word**
```python
from voice.wakeword import OpenWakeWordService
from voice.config import WakeWordConfig

config = WakeWordConfig(models=[{"name": "hey_jarvis", "threshold": 0.5}])
wakeword = OpenWakeWordService(config)

# Process audio file for wake word
import soundfile as sf
import numpy as np

audio, sr = sf.read("test_wakeword.wav", dtype="int16")
detections = wakeword.process_batch(audio)
for d in detections:
    print(f"Detected: {d.wake_word} (confidence={d.confidence:.2f})")
```

### Phase 4: Full Daemon Test

```bash
# Start voice daemon
python -m voice.daemon --config config/voice_config_macbook.yaml --log-level DEBUG
```

---

## 6. Key Research Sources

### XTTS v2
- [HuggingFace Model Card](https://huggingface.co/coqui/XTTS-v2)
- [Coqui TTS Documentation](https://docs.coqui.ai/en/latest/models/xtts.html)
- [PyPI Package](https://pypi.org/project/coqui-tts/)
- [macOS Installation Guide](https://github.com/coqui-ai/TTS/discussions/2177)

### Faster-Whisper / STT
- [Whisper Telugu Research](https://arxiv.org/html/2412.19785v1)
- [Pingala V1 for Indic Languages](https://www.shunyalabs.ai/blog/benchmarking-top-open-source-speech-recognition-models)
- [Code-Switching Extension](https://ieeexplore.ieee.org/document/10929894/)

### OpenWakeWord
- [GitHub Repository](https://github.com/dscripka/openWakeWord)
- [Custom Training Discussion](https://github.com/dscripka/openWakeWord/discussions/45)
- [Home Assistant Integration](https://www.home-assistant.io/voice_control/create_wake_word/)

---

## 7. Next Steps

### Immediate (This Week)
1. [ ] Install voice dependencies on MacBook
2. [ ] Create `config/voice_config_macbook.yaml`
3. [ ] Test STT with Telugu-English audio sample
4. [ ] Test TTS with romanized Telugu text
5. [ ] Test wake word with "hey_jarvis"

### When "Airy" Voice Sample Received
1. [ ] Process audio (convert to WAV, normalize)
2. [ ] Load as voice profile in XTTS
3. [ ] Test synthesis quality
4. [ ] Fine-tune if needed

### Custom Wake Word Training
1. [ ] Use OpenWakeWord Colab to train "hey_friday"
2. [ ] Train "wake_up_daddys_home"
3. [ ] Deploy custom models to `voice/models/wakeword/`

### After AWS Access Restored
1. [ ] Complete iteration 2 training
2. [ ] Deploy Friday LLM to endpoint
3. [ ] Connect voice daemon to orchestrator
4. [ ] Full end-to-end voice test

---

## 8. Known Limitations

### Telugu TTS
- XTTS doesn't natively support Telugu script
- Using Hindi model as fallback (similar phonetics)
- Romanized input works better for pronunciation
- Quality depends on reference audio

### Code-Switching STT
- Standard Whisper struggles with mid-sentence language switches
- Extended models (Pingala) offer better support
- May need post-processing for optimal results

### MacBook Performance
- CPU-only inference will be slower
- Medium Whisper model recommended (vs large-v3)
- First TTS synthesis takes time (model loading)
- Consider using GPU server for production

---

*Document generated: January 2026*
