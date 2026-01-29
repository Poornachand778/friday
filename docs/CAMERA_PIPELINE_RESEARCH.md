# Friday AI - Camera/Vision Pipeline Research

**Date:** January 2026
**Status:** Research Complete

---

## Executive Summary

After researching face recognition and emotion detection options, **DeepFace** emerges as the optimal choice for Friday AI. It provides both user identification AND emotion detection in a single lightweight library.

---

## 1. Recommended Solution: DeepFace

### Why DeepFace?

| Feature | DeepFace | InsightFace | MediaPipe + FER |
|---------|----------|-------------|-----------------|
| Face Recognition | Yes (99.40% accuracy) | Yes | No (detection only) |
| Emotion Detection | Built-in | No | Requires FER |
| Lightweight | Yes | Heavy | Medium |
| Real-time Support | Yes | Yes | Yes |
| MacBook Compatible | Yes (CPU) | Needs GPU ideally | Yes |
| Single Library | Yes | No | No (2 libraries) |
| Active Maintenance | Yes | Yes | Mixed |

### DeepFace Capabilities

From [DeepFace GitHub](https://github.com/serengil/deepface):

1. **Face Recognition**
   - Wraps 10+ models: VGG-Face, FaceNet, ArcFace, DeepID, etc.
   - 99.40% accuracy on LFW benchmark
   - Face verification and identification

2. **Facial Attribute Analysis**
   - Age estimation
   - Gender classification
   - Emotion recognition (angry, disgust, fear, happy, sad, surprise, neutral)
   - Race/ethnicity (optional)

3. **Face Detection Backends**
   - OpenCV
   - SSD
   - Dlib
   - MTCNN
   - RetinaFace
   - MediaPipe

---

## 2. Implementation Plan

### Installation

```bash
pip install deepface
# Also installs: tensorflow, opencv-python, mtcnn, etc.
```

### Basic Usage

```python
from deepface import DeepFace

# Face verification (is this Boss?)
result = DeepFace.verify(
    img1_path="boss_reference.jpg",
    img2_path="current_frame.jpg",
    model_name="ArcFace"  # Best accuracy
)
print(f"Same person: {result['verified']}")
print(f"Distance: {result['distance']}")

# Face identification (who is this?)
dfs = DeepFace.find(
    img_path="current_frame.jpg",
    db_path="face_database/",
    model_name="ArcFace"
)

# Facial attributes (emotion, age, gender)
analysis = DeepFace.analyze(
    img_path="current_frame.jpg",
    actions=['emotion', 'age', 'gender']
)
print(f"Emotion: {analysis[0]['dominant_emotion']}")
print(f"Age: {analysis[0]['age']}")
```

### Real-Time Video Analysis

```python
import cv2
from deepface import DeepFace

# Real-time streaming analysis
DeepFace.stream(
    db_path="face_database/",
    model_name="ArcFace",
    detector_backend="opencv",
    source=0  # Webcam
)
```

---

## 3. Face Database Structure

### Directory Layout

```
data/faces/
├── boss/
│   ├── poorna_1.jpg
│   ├── poorna_2.jpg
│   └── poorna_3.jpg
├── team/
│   ├── member1/
│   │   └── member1_1.jpg
│   └── member2/
│       └── member2_1.jpg
└── unknown/
    └── (auto-captured faces)
```

### Database Schema Addition

```sql
-- Add to existing schema
CREATE TABLE face_profiles (
    id SERIAL PRIMARY KEY,
    name VARCHAR(128) NOT NULL,
    access_level VARCHAR(32) DEFAULT 'team',  -- boss, team, unknown
    embedding BYTEA,  -- Face embedding vector
    reference_images TEXT[],  -- Paths to reference images
    created_at TIMESTAMP DEFAULT NOW(),
    last_seen TIMESTAMP,
    is_active BOOLEAN DEFAULT TRUE
);

CREATE TABLE face_access_log (
    id SERIAL PRIMARY KEY,
    face_profile_id INTEGER REFERENCES face_profiles(id),
    timestamp TIMESTAMP DEFAULT NOW(),
    location VARCHAR(64),  -- writers_room, kitchen, etc.
    emotion VARCHAR(32),
    confidence FLOAT,
    action_taken VARCHAR(64)  -- granted_access, denied, captured_unknown
);
```

---

## 4. Access Control Integration

### Access Levels

| Level | Description | Capabilities |
|-------|-------------|--------------|
| `boss` | Poorna only | Full access, confidential discussions, all tools |
| `team` | Saved team members | Basic commands, read-only scripts, no confidential |
| `unknown` | Unrecognized faces | Minimal interaction, face captured for review |

### Implementation Flow

```python
from deepface import DeepFace
from dataclasses import dataclass
from enum import Enum

class AccessLevel(Enum):
    BOSS = "boss"
    TEAM = "team"
    UNKNOWN = "unknown"

@dataclass
class UserContext:
    name: str
    access_level: AccessLevel
    emotion: str
    confidence: float
    is_thinking: bool  # Detected from expression
    is_frustrated: bool

def identify_user(frame) -> UserContext:
    """Identify user and detect emotional state"""

    # Find face in database
    try:
        matches = DeepFace.find(
            img_path=frame,
            db_path="data/faces/",
            model_name="ArcFace",
            enforce_detection=False
        )

        if len(matches) > 0 and len(matches[0]) > 0:
            # Known person
            match_path = matches[0].iloc[0]['identity']
            if 'boss' in match_path:
                access_level = AccessLevel.BOSS
                name = "Boss"
            else:
                access_level = AccessLevel.TEAM
                name = extract_name_from_path(match_path)
        else:
            access_level = AccessLevel.UNKNOWN
            name = "Unknown"

    except Exception:
        access_level = AccessLevel.UNKNOWN
        name = "Unknown"

    # Analyze emotion
    try:
        analysis = DeepFace.analyze(
            img_path=frame,
            actions=['emotion'],
            enforce_detection=False
        )
        emotion = analysis[0]['dominant_emotion']
        emotion_scores = analysis[0]['emotion']
    except Exception:
        emotion = "neutral"
        emotion_scores = {}

    # Detect thinking/frustration states
    is_thinking = emotion in ['neutral', 'sad'] and emotion_scores.get('neutral', 0) > 50
    is_frustrated = emotion in ['angry', 'disgust'] or emotion_scores.get('angry', 0) > 30

    return UserContext(
        name=name,
        access_level=access_level,
        emotion=emotion,
        confidence=emotion_scores.get(emotion, 0) / 100,
        is_thinking=is_thinking,
        is_frustrated=is_frustrated
    )
```

---

## 5. Emotion-Aware Context Building

### Emotional States to Detect

| State | Indicators | Friday's Response |
|-------|------------|-------------------|
| **Thinking/Struggling** | Neutral + long pause, furrowed brow | Wait patiently, offer help subtly |
| **Excited** | Happy, fast speech | Match energy, engage enthusiastically |
| **Frustrated** | Angry, disgust | Be extra helpful, simplify responses |
| **Focused** | Neutral, steady gaze | Minimal interruption, brief responses |
| **Tired** | Sad, low energy | Suggest breaks, be gentle |

### Context Integration

```python
def build_multimodal_context(
    transcript: str,
    user_context: UserContext,
    session_history: list
) -> dict:
    """Build context for LLM including emotional state"""

    context = {
        "user": {
            "name": user_context.name,
            "access_level": user_context.access_level.value,
        },
        "emotional_context": {
            "current_emotion": user_context.emotion,
            "is_thinking": user_context.is_thinking,
            "is_frustrated": user_context.is_frustrated,
        },
        "transcript": transcript,
        "session_turns": len(session_history),
    }

    # Add behavioral hints for LLM
    if user_context.is_frustrated:
        context["system_hint"] = "User seems frustrated. Be extra helpful and concise."
    elif user_context.is_thinking:
        context["system_hint"] = "User is thinking. Wait for them to finish before responding."

    return context
```

---

## 6. MacBook Camera Integration

### Camera Options

1. **MacBook Built-in Camera**
   - Access via OpenCV: `cv2.VideoCapture(0)`
   - Good for development testing

2. **OnePlus 8 Pro (via ADB or IP webcam)**
   - Higher quality camera
   - Can run as IP webcam app
   - Connect via `cv2.VideoCapture("http://phone-ip:port/video")`

### Camera Service Implementation

```python
import cv2
import threading
import time
from typing import Optional, Callable

class CameraService:
    """Camera capture with face analysis"""

    def __init__(
        self,
        camera_id: int = 0,
        analysis_interval: float = 1.0,  # Analyze every N seconds
    ):
        self.camera_id = camera_id
        self.analysis_interval = analysis_interval
        self._cap: Optional[cv2.VideoCapture] = None
        self._running = False
        self._current_context: Optional[UserContext] = None
        self._last_analysis = 0

    def start(self):
        """Start camera capture"""
        self._cap = cv2.VideoCapture(self.camera_id)
        self._running = True

    def get_frame(self):
        """Get current frame"""
        if self._cap is None:
            return None
        ret, frame = self._cap.read()
        return frame if ret else None

    def analyze_current_user(self) -> Optional[UserContext]:
        """Analyze current frame for user identity and emotion"""
        now = time.time()

        # Rate limit analysis
        if now - self._last_analysis < self.analysis_interval:
            return self._current_context

        frame = self.get_frame()
        if frame is None:
            return None

        self._current_context = identify_user(frame)
        self._last_analysis = now

        return self._current_context

    def stop(self):
        """Stop camera"""
        self._running = False
        if self._cap:
            self._cap.release()
```

---

## 7. Privacy Considerations

### Data Handling

1. **Unknown Faces**
   - Capture temporarily for potential "remember this person" command
   - Delete after session if not saved
   - Never store without explicit consent

2. **Face Embeddings**
   - Store embeddings, not raw images (privacy-friendly)
   - Encrypt embeddings at rest
   - Clear audit log periodically

3. **Commands**
   - "Friday, remember this person as [name]" - Save to team
   - "Friday, forget [name]" - Remove from database
   - "Friday, who do you see?" - Report identified users

---

## 8. Dependencies

```
# Add to requirements.txt
deepface>=0.0.93
opencv-python>=4.8.0
tensorflow>=2.13.0  # Backend for DeepFace models
```

### Model Sizes (Downloaded on First Use)

| Model | Size | Accuracy | Speed |
|-------|------|----------|-------|
| VGG-Face | ~500MB | 98.78% | Slow |
| FaceNet | ~90MB | 99.20% | Medium |
| ArcFace | ~120MB | 99.40% | Medium |
| SFace | ~10MB | 99.60% | Fast |

**Recommendation:** Start with ArcFace (best accuracy), consider SFace for production (fast + accurate).

---

## 9. File Structure

```
vision/
├── __init__.py
├── camera_service.py      # Camera capture
├── face_recognition.py    # DeepFace wrapper
├── emotion_detector.py    # Emotion analysis
├── access_control.py      # Access level management
├── context_builder.py     # Multi-modal context
└── config.py              # Vision configuration

data/faces/
├── boss/                  # Poorna's reference images
├── team/                  # Team member folders
└── unknown/               # Temporary captures

config/
└── vision_config.yaml     # Camera settings
```

---

## 10. Next Steps

### Immediate
1. [ ] Install DeepFace: `pip install deepface`
2. [ ] Test face detection on MacBook camera
3. [ ] Create Boss reference images (3-5 photos)
4. [ ] Test emotion detection accuracy

### Integration
1. [ ] Create vision/ module structure
2. [ ] Implement CameraService
3. [ ] Add face_profiles table to database
4. [ ] Integrate with orchestrator context builder

### Later
1. [ ] Train on Indian expressions (if accuracy is low)
2. [ ] Add OnePlus 8 camera support
3. [ ] Implement "remember/forget" commands

---

## 11. Research Sources

- [DeepFace GitHub](https://github.com/serengil/deepface) - Lightweight face recognition library
- [InsightFace](https://github.com/deepinsight/insightface) - State-of-the-art face analysis
- [MediaPipe Face Mesh](https://mediapipe.readthedocs.io/en/latest/solutions/face_detection.html) - Real-time face landmarks
- [FER Tutorial](https://learnopencv.com/facial-emotion-recognition/) - Emotion recognition guide
- [FFEM Tool](https://medium.com/@jorgefmp.mle/fast-facial-emotion-monitoring-ffem-an-open-source-tool-for-simplified-facial-emotion-2f127a874721) - MediaPipe + DeepFace combo

---

*Document generated: January 2026*
