#!/usr/bin/env python3
"""
Voice Pipeline Test Script for Friday AI
=========================================

Tests individual voice components on MacBook (CPU mode).

Usage:
    python scripts/test_voice_pipeline.py --test all
    python scripts/test_voice_pipeline.py --test stt
    python scripts/test_voice_pipeline.py --test tts
    python scripts/test_voice_pipeline.py --test wakeword
"""

import argparse
import sys
import time
from pathlib import Path

# Add repo root to path
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))


def test_stt():
    """Test Speech-to-Text (Faster-Whisper)"""
    print("\n" + "=" * 60)
    print("Testing STT (Faster-Whisper)")
    print("=" * 60)

    try:
        from voice.stt import FasterWhisperSTT
        from voice.config import STTConfig

        print("\n[1] Initializing STT with CPU config...")
        config = STTConfig(
            model="medium",  # Use medium for CPU
            device="cpu",
            compute_type="float32",
            word_timestamps=True,
            vad_filter=True,
        )
        stt = FasterWhisperSTT(config)
        print("    OK - STT service created")

        print("\n[2] Loading model (this may take a minute first time)...")
        start = time.time()
        stt._ensure_model_loaded()
        print(f"    OK - Model loaded in {time.time() - start:.1f}s")

        # Test with sample text (synthesize if no audio file)
        print("\n[3] Testing transcription...")

        # Check if test audio exists
        test_audio = REPO_ROOT / "voice/data/test_audio.wav"
        if test_audio.exists():
            result = stt.transcribe(str(test_audio))
            print(f"    Transcript: {result.text}")
            print(
                f"    Language: {result.language} (confidence: {result.language_probability:.2f})"
            )
            print(f"    Duration: {result.duration:.2f}s")
            print(
                f"    Processing: {result.processing_time:.2f}s (RTF: {result.rtf:.2f})"
            )
        else:
            print(f"    No test audio at {test_audio}")
            print(
                "    To test: Create a WAV file with speech and place at voice/data/test_audio.wav"
            )

        print("\n[4] STT Test: PASSED")
        return True

    except ImportError as e:
        print(f"\n    FAILED - Import error: {e}")
        print("    Install: pip install faster-whisper ctranslate2")
        return False
    except Exception as e:
        print(f"\n    FAILED - {e}")
        return False


def test_tts():
    """Test Text-to-Speech (XTTS v2)"""
    print("\n" + "=" * 60)
    print("Testing TTS (XTTS v2)")
    print("=" * 60)

    try:
        from voice.tts import XTTSService
        from voice.config import TTSConfig

        print("\n[1] Initializing TTS with CPU config...")
        config = TTSConfig(
            device="cpu",
            default_language="en",
        )
        tts = XTTSService(config)
        print("    OK - TTS service created")

        print("\n[2] Loading model (this may take several minutes first time)...")
        start = time.time()
        tts._ensure_model_loaded()
        print(f"    OK - Model loaded in {time.time() - start:.1f}s")

        print("\n[3] Testing English synthesis...")
        test_text = "Boss, what do you need?"
        start = time.time()
        result = tts.synthesize(test_text, language="en")
        print(f"    Text: '{test_text}'")
        print(f"    Duration: {result.duration:.2f}s")
        print(f"    Processing: {result.processing_time:.2f}s (RTF: {result.rtf:.2f})")

        # Save output
        output_path = REPO_ROOT / "voice/data/test_tts_output.wav"
        output_path.parent.mkdir(parents=True, exist_ok=True)

        from voice.tts.xtts_service import save_audio

        save_audio(result.audio, output_path, result.sample_rate)
        print(f"    Saved to: {output_path}")

        print("\n[4] Testing romanized Telugu (via Hindi model)...")
        test_text_te = "Boss, emi kavali?"  # Romanized Telugu
        result_te = tts.synthesize(test_text_te, language="hi")  # Use Hindi for Telugu
        print(f"    Text: '{test_text_te}'")
        print(f"    Duration: {result_te.duration:.2f}s")

        output_path_te = REPO_ROOT / "voice/data/test_tts_telugu.wav"
        save_audio(result_te.audio, output_path_te, result_te.sample_rate)
        print(f"    Saved to: {output_path_te}")

        print("\n[5] TTS Test: PASSED")
        return True

    except ImportError as e:
        print(f"\n    FAILED - Import error: {e}")
        print("    Install: pip install TTS")
        return False
    except Exception as e:
        print(f"\n    FAILED - {e}")
        import traceback

        traceback.print_exc()
        return False


def test_wakeword():
    """Test Wake Word Detection (OpenWakeWord)"""
    print("\n" + "=" * 60)
    print("Testing Wake Word (OpenWakeWord)")
    print("=" * 60)

    try:
        from voice.wakeword import OpenWakeWordService
        from voice.config import WakeWordConfig

        print("\n[1] Initializing wake word service...")
        config = WakeWordConfig(
            models=[{"name": "hey_jarvis", "threshold": 0.5}],
            inference_framework="onnx",
        )
        wakeword = OpenWakeWordService(config)
        print("    OK - Wake word service created")

        print("\n[2] Loading models...")
        start = time.time()
        wakeword._ensure_model_loaded()
        print(f"    OK - Models loaded in {time.time() - start:.1f}s")
        print(f"    Active models: {wakeword.active_models}")

        print("\n[3] Testing with audio file...")
        test_audio = REPO_ROOT / "voice/data/test_wakeword.wav"
        if test_audio.exists():
            import soundfile as sf
            import numpy as np

            audio, sr = sf.read(str(test_audio), dtype="int16")
            # Resample if needed
            if sr != 16000:
                print(f"    Warning: Audio is {sr}Hz, expected 16000Hz")

            detections = wakeword.process_batch(audio)
            if detections:
                for d in detections:
                    print(
                        f"    Detected: {d.wake_word} (confidence: {d.confidence:.3f})"
                    )
            else:
                print("    No wake word detected in audio")
        else:
            print(f"    No test audio at {test_audio}")
            print("    To test: Say 'Hey Jarvis' into a WAV file")

        print("\n[4] Wake Word Test: PASSED")
        return True

    except ImportError as e:
        print(f"\n    FAILED - Import error: {e}")
        print("    Install: pip install openwakeword onnxruntime")
        return False
    except Exception as e:
        print(f"\n    FAILED - {e}")
        import traceback

        traceback.print_exc()
        return False


def test_audio_io():
    """Test Audio I/O"""
    print("\n" + "=" * 60)
    print("Testing Audio I/O")
    print("=" * 60)

    try:
        import sounddevice as sd

        print("\n[1] Checking audio devices...")
        devices = sd.query_devices()

        input_device = sd.query_devices(kind="input")
        output_device = sd.query_devices(kind="output")

        print(f"    Default Input: {input_device['name']}")
        print(f"    Default Output: {output_device['name']}")

        print("\n[2] Testing microphone (3 second recording)...")
        print("    Speak into your microphone now...")

        duration = 3  # seconds
        sample_rate = 16000
        recording = sd.rec(
            int(duration * sample_rate),
            samplerate=sample_rate,
            channels=1,
            dtype="int16",
        )
        sd.wait()

        # Check if we got audio
        import numpy as np

        max_amplitude = np.max(np.abs(recording))
        print(f"    Recording complete. Max amplitude: {max_amplitude}")

        if max_amplitude > 100:
            print("    Microphone is working!")
        else:
            print("    Warning: Low amplitude - check microphone")

        # Save recording
        import soundfile as sf

        output_path = REPO_ROOT / "voice/data/test_mic_recording.wav"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        sf.write(str(output_path), recording, sample_rate)
        print(f"    Saved to: {output_path}")

        print("\n[3] Testing speaker (playing beep)...")

        # Generate a simple beep
        frequency = 440  # Hz
        beep_duration = 0.5  # seconds
        t = np.linspace(0, beep_duration, int(sample_rate * beep_duration), False)
        beep = (np.sin(2 * np.pi * frequency * t) * 0.3).astype(np.float32)

        sd.play(beep, sample_rate)
        sd.wait()
        print("    Beep played. Did you hear it?")

        print("\n[4] Audio I/O Test: PASSED")
        return True

    except ImportError as e:
        print(f"\n    FAILED - Import error: {e}")
        print("    Install: pip install sounddevice soundfile")
        return False
    except Exception as e:
        print(f"\n    FAILED - {e}")
        import traceback

        traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser(description="Test Friday Voice Pipeline")
    parser.add_argument(
        "--test",
        choices=["all", "stt", "tts", "wakeword", "audio"],
        default="all",
        help="Which component to test",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("Friday AI - Voice Pipeline Test Suite")
    print("=" * 60)
    print(f"Testing: {args.test}")

    results = {}

    if args.test in ["all", "audio"]:
        results["audio"] = test_audio_io()

    if args.test in ["all", "stt"]:
        results["stt"] = test_stt()

    if args.test in ["all", "tts"]:
        results["tts"] = test_tts()

    if args.test in ["all", "wakeword"]:
        results["wakeword"] = test_wakeword()

    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)

    all_passed = True
    for component, passed in results.items():
        status = "PASSED" if passed else "FAILED"
        print(f"  {component}: {status}")
        if not passed:
            all_passed = False

    if all_passed:
        print("\nAll tests passed!")
        print("\nNext steps:")
        print("  1. Record an 'Airy' voice sample (6+ seconds)")
        print("  2. Train custom wake words ('hey_friday', 'wake_up_daddys_home')")
        print(
            "  3. Run full daemon: python -m voice.daemon --config config/voice_config_macbook.yaml"
        )
    else:
        print("\nSome tests failed. Check the errors above.")
        print("Make sure all dependencies are installed:")
        print(
            "  pip install faster-whisper TTS openwakeword sounddevice soundfile webrtcvad"
        )

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
