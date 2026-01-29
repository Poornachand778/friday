# Friday AI - Complete System Architecture v2.0

**Document Version:** 2.0
**Created:** January 2026
**Status:** Design Phase
**Focus:** Script Writing Assistant with Multi-Modal Input

---

## Executive Summary

Friday is an autonomous, voice-controlled AI creative collaborator designed for screenwriting. It combines:
- **Voice Interface**: Always-listening, Telugu-English code-switching
- **Visual Context**: Camera-based user identification, emotion detection, expression analysis
- **Autonomous Agents**: Background script analysis and improvement suggestions
- **Tool Integration**: Scene management, email, calendar, research capabilities
- **Access Control**: Face-based authentication for confidential discussions

---

## System Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           FRIDAY AI SYSTEM                                  │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                        INPUT LAYER                                   │   │
│  │                                                                      │   │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────────┐  │   │
│  │  │    VOICE     │  │    CAMERA    │  │      WAKE WORD           │  │   │
│  │  │              │  │              │  │                          │  │   │
│  │  │ • Microphone │  │ • MacBook    │  │ • "Friday"               │  │   │
│  │  │ • VAD filter │  │ • OnePlus 8  │  │ • "Wake up daddy's home" │  │   │
│  │  │ • STT Whisper│  │ • Edge cams  │  │ • Custom phrases         │  │   │
│  │  │              │  │              │  │                          │  │   │
│  │  └──────────────┘  └──────────────┘  └──────────────────────────┘  │   │
│  │         │                 │                      │                  │   │
│  │         ▼                 ▼                      ▼                  │   │
│  │  ┌──────────────────────────────────────────────────────────────┐  │   │
│  │  │              MULTI-MODAL CONTEXT BUILDER                      │  │   │
│  │  │                                                               │  │   │
│  │  │  • Voice transcript (Telugu-English)                         │  │   │
│  │  │  • User identity (face recognition)                          │  │   │
│  │  │  • Emotion state (expression analysis)                       │  │   │
│  │  │  • Conversation pauses, hesitations, struggle indicators     │  │   │
│  │  │  • Access level (Boss vs Team member)                        │  │   │
│  │  │                                                               │  │   │
│  │  └──────────────────────────────────────────────────────────────┘  │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                      │                                      │
│                                      ▼                                      │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                       ORCHESTRATOR (FastAPI)                         │   │
│  │                                                                      │   │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────────┐  │   │
│  │  │   CONTEXT    │  │   MEMORY     │  │    ACCESS CONTROL        │  │   │
│  │  │   DETECTOR   │  │   MANAGER    │  │                          │  │   │
│  │  │              │  │              │  │ • Face DB lookup         │  │   │
│  │  │ • Script mode│  │ • STM (Redis)│  │ • Boss = full access     │  │   │
│  │  │ • Research   │  │ • LTM (PG)   │  │ • Team = restricted      │  │   │
│  │  │ • Casual     │  │ • Persona    │  │ • Unknown = basic only   │  │   │
│  │  │              │  │ • History    │  │                          │  │   │
│  │  └──────────────┘  └──────────────┘  └──────────────────────────┘  │   │
│  │                                                                      │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                      │                                      │
│                                      ▼                                      │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                         FRIDAY LLM                                   │   │
│  │                                                                      │   │
│  │  Base: Meta-Llama-3.1-8B-Instruct + LoRA adapters                   │   │
│  │  Inference: AWS SageMaker (dev) → Local vLLM (prod)                 │   │
│  │                                                                      │   │
│  │  Personality:                                                        │   │
│  │  • Telugu-English code-switching                                    │   │
│  │  • "Boss" address, no flattery                                      │   │
│  │  • Direct, concise, witty                                           │   │
│  │  • Film production expertise                                        │   │
│  │                                                                      │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                      │                                      │
│                                      ▼                                      │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                         TOOL LAYER (MCP)                             │   │
│  │                                                                      │   │
│  │  ┌────────────┐ ┌────────────┐ ┌────────────┐ ┌────────────────┐   │   │
│  │  │   SCENE    │ │   EMAIL    │ │  CALENDAR  │ │    RESEARCH    │   │   │
│  │  │  MANAGER   │ │   (Gmail)  │ │            │ │   (Web + RAG)  │   │   │
│  │  │            │ │            │ │            │ │                │   │   │
│  │  │ • search   │ │ • send     │ │ • read     │ │ • web search   │   │   │
│  │  │ • get      │ │ • draft    │ │ • create   │ │ • film refs    │   │   │
│  │  │ • update   │ │ • reply    │ │ • remind   │ │ • legal docs   │   │   │
│  │  │ • reorder  │ │            │ │            │ │                │   │   │
│  │  │ • link     │ │            │ │            │ │                │   │   │
│  │  └────────────┘ └────────────┘ └────────────┘ └────────────────┘   │   │
│  │                                                                      │   │
│  │  ┌────────────────────────────────────────────────────────────────┐ │   │
│  │  │              AUTONOMOUS AGENT (Background)                      │ │   │
│  │  │                                                                 │ │   │
│  │  │  • Script analysis while idle                                   │ │   │
│  │  │  • Pattern detection, inconsistency flagging                    │ │   │
│  │  │  • Draft improvements → "Friday Suggestions" backlog            │ │   │
│  │  │  • Research compilation                                         │ │   │
│  │  │  • Waits for discussion prompt before presenting                │ │   │
│  │  └────────────────────────────────────────────────────────────────┘ │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                      │                                      │
│                                      ▼                                      │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                        OUTPUT LAYER                                  │   │
│  │                                                                      │   │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────────┐  │   │
│  │  │    VOICE     │  │   DISPLAY    │  │      STORAGE             │  │   │
│  │  │    (TTS)     │  │  (Optional)  │  │                          │  │   │
│  │  │              │  │              │  │ • Audio logs → training  │  │   │
│  │  │ • XTTS v2    │  │ • Terminal   │  │ • Transcripts            │  │   │
│  │  │ • "Airy"     │  │ • Web UI     │  │ • Context snapshots      │  │   │
│  │  │   voice style│  │ • Screen     │  │ • Face DB (authorized)   │  │   │
│  │  │              │  │   share      │  │                          │  │   │
│  │  └──────────────┘  └──────────────┘  └──────────────────────────┘  │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Component Details

### 1. Voice Pipeline

#### Speech-to-Text (STT)
- **Engine**: Faster-Whisper (large-v3)
- **Languages**: Telugu + English (auto-detect)
- **Features**: Word timestamps, speaker diarization potential
- **Hardware**: CPU-based on MacBook, GPU-accelerated on server

#### Text-to-Speech (TTS)
- **Engine**: XTTS v2 (Coqui)
- **Default Voice**: "Airy" style (reference recording to be provided)
- **Output Format**: Romanized Telugu for pronunciation
- **Voice Profiles**: Multiple voices configurable (Boss's voice separate)

#### Wake Word Detection
- **Engine**: OpenWakeWord
- **Primary**: "Friday"
- **Secondary**: "Wake up daddy's home"
- **Custom**: Additional catchy phrases (TBD)

#### Voice Activity Detection (VAD)
- **Engine**: WebRTC VAD
- **Purpose**: Filter silence, detect speech boundaries
- **Sensitivity**: Adjustable (2-3 level)

---

### 2. Camera/Vision Pipeline

#### User Identification
- **Purpose**: Authenticate who's speaking
- **Technology**: Face recognition (research needed: deepface, face_recognition, or InsightFace)
- **Access Levels**:
  - **Boss (Poorna)**: Full access to all features, confidential data
  - **Team Members (Saved)**: Restricted access, no confidential discussions
  - **Unknown**: Basic functionality only, face captured for potential save

#### Emotion/Expression Detection
- **Purpose**: Provide conversational context to LLM
- **Detected States**:
  - Thinking/struggling (long pauses, furrowed brow)
  - Excited (animated expressions, faster speech)
  - Frustrated (tension, sighs)
  - Focused (minimal movement, steady gaze)
- **Technology**: Research needed (MediaPipe, FER, or custom model)

#### Privacy Controls
- **Default**: Unknown faces captured but deleted after session
- **Save Command**: "Friday, remember this person as [name]"
- **Delete Command**: "Friday, forget [name]"
- **Face DB**: PostgreSQL with encrypted face embeddings

---

### 3. Autonomous Agent System

#### Background Script Analyzer
```python
class ScriptAnalyzerAgent:
    """
    Runs when Friday is idle, analyzes screenplay for:
    - Plot inconsistencies
    - Character arc gaps
    - Dialogue improvements
    - Pacing issues
    - Missing scene connections
    """

    def analyze(self, project_slug: str):
        # Load all scenes
        # Run analysis prompts through LLM
        # Generate suggestions
        # Save to "friday_suggestions" table
        pass

    def get_pending_suggestions(self, project_slug: str) -> List[Suggestion]:
        # Return suggestions not yet discussed
        pass
```

#### Suggestion Backlog Schema
```sql
CREATE TABLE friday_suggestions (
    id SERIAL PRIMARY KEY,
    project_id INTEGER REFERENCES script_projects(id),
    scene_id INTEGER REFERENCES script_scenes(id),  -- NULL if project-level
    suggestion_type VARCHAR(64),  -- 'dialogue', 'plot', 'pacing', 'character'
    title VARCHAR(256),
    description TEXT,
    proposed_change TEXT,  -- Draft content if applicable
    priority INTEGER DEFAULT 3,  -- 1=high, 5=low
    status VARCHAR(32) DEFAULT 'pending',  -- pending, discussed, accepted, rejected
    created_at TIMESTAMP DEFAULT NOW(),
    discussed_at TIMESTAMP,
    decision_notes TEXT
);
```

---

### 4. Access Control Matrix

| Feature | Boss (Poorna) | Saved Team Member | Unknown Person |
|---------|---------------|-------------------|----------------|
| Voice commands | Full | Full | Basic |
| Scene editing | Full | Read-only | None |
| Confidential discussions | Yes | No | No |
| Email sending | Yes | No | No |
| Calendar access | Yes | No | No |
| Friday suggestions | Yes | View only | No |
| Research requests | Yes | Yes | Limited |
| Remember faces | Can command | No | No |
| View stored faces | Yes | No | No |

---

### 5. Development Phases

#### Phase 1: Core Voice + Camera (MacBook Development)

**Voice Components:**
- [ ] STT integration (Faster-Whisper)
- [ ] TTS integration (XTTS v2 with "Airy" reference)
- [ ] Wake word training (OpenWakeWord)
- [ ] VAD setup (WebRTC)
- [ ] Voice daemon → Orchestrator connection

**Camera Components:**
- [ ] Face detection (MacBook camera / OnePlus)
- [ ] Face recognition for Boss identification
- [ ] Basic emotion detection
- [ ] Face save/delete commands
- [ ] Access control integration

**Integration:**
- [ ] Multi-modal context builder
- [ ] Orchestrator updates for visual context
- [ ] End-to-end voice loop

#### Phase 2: Autonomous Agents

- [ ] Script analyzer agent
- [ ] Friday suggestions backlog
- [ ] Background task scheduler
- [ ] Proactive suggestion system
- [ ] Research agent (web + document)

#### Phase 3: Hardware Migration

- [ ] Server build (RTX 4090)
- [ ] vLLM local inference setup
- [ ] Edge microphone configuration
- [ ] Multi-room routing
- [ ] Internet access for research/tools

#### Phase 4: Polish & Scale

- [ ] Voice quality refinement
- [ ] Latency optimization (<2s response)
- [ ] Additional wake phrases
- [ ] Calendar integration
- [ ] Social media posting (future)

---

## Technology Stack

### Confirmed
| Component | Technology | Status |
|-----------|------------|--------|
| LLM Base | Meta-Llama-3.1-8B-Instruct | ✅ |
| Fine-tuning | QLoRA (r=32, α=64) | ✅ |
| Training Infra | AWS SageMaker | ✅ |
| Database | PostgreSQL | ✅ |
| Vector Search | sentence-transformers + pgvector | ✅ |
| API Framework | FastAPI | ✅ |
| MCP Tools | scene_manager, gmail | ✅ |

### To Be Implemented
| Component | Technology | Research Status |
|-----------|------------|-----------------|
| STT | Faster-Whisper large-v3 | Ready to implement |
| TTS | XTTS v2 | Needs voice reference |
| Wake Word | OpenWakeWord | Needs training |
| Face Recognition | deepface / InsightFace | Needs research |
| Emotion Detection | MediaPipe / FER | Needs research |
| Local Inference | vLLM | After hardware |

---

## Data Flow Examples

### Example 1: Boss Asks for Script Update

```
1. [Wake Word] "Friday" detected
2. [Camera] Face recognized → Boss (full access)
3. [STT] "Show me the climax scene suggestions"
4. [Access Control] Verified: Boss can access friday_suggestions
5. [Orchestrator] Context: Script mode, project: Gusagusalu
6. [Tool Call] scene_search("climax") + get_suggestions(project)
7. [LLM] Generate response with suggestions
8. [TTS] Speak response in "Airy" voice
```

### Example 2: Unknown Person Asks Confidential Question

```
1. [Wake Word] "Friday" detected
2. [Camera] Face NOT recognized → Unknown person
3. [STT] "What's the budget for Poorna's film?"
4. [Access Control] DENIED: Confidential topic, unknown user
5. [LLM] Generate polite deflection
6. [TTS] "Boss, someone's asking about the budget. I'll wait for you to authorize."
7. [Camera] Capture face for potential save later
```

### Example 3: Autonomous Night Analysis

```
1. [Scheduler] 2:00 AM, Friday idle
2. [Agent] Load Gusagusalu scenes
3. [LLM] Analyze each scene for issues
4. [Agent] Found: Scene 18-19 transition gap
5. [Agent] Generate suggestion: "Add audio bridge"
6. [DB] Save to friday_suggestions (status: pending)
7. [Next Morning] Boss enters room
8. [Friday] "Boss, morning. Nenu overnight analysis chesanu -
    one transition issue found in Scene 18-19. Discuss cheddama?"
```

---

## Open Research Items

1. **Face Recognition on MacBook**: Performance of real-time face recognition without GPU
2. **XTTS Telugu Quality**: Romanization scheme that sounds most natural
3. **Emotion Detection Accuracy**: Which model works best for Indian expressions
4. **Wake Word Training**: Custom phrase accuracy with limited samples
5. **Multi-camera Sync**: How to handle multiple edge cameras feeding one server

---

## Next Actions

1. **Immediate**: Complete iteration 2 training when AWS access restored
2. **This Week**: Test voice pipeline (STT + TTS) on MacBook
3. **Next Week**: Research and prototype face recognition
4. **Ongoing**: Collect "Airy" voice reference for TTS cloning

---

*Document created by Friday AI Development Session - January 2026*
