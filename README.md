# Friday AI - Personal JARVIS Assistant

> A complete JARVIS-style AI assistant for screenwriting with Telugu-English bilingual personality, voice interaction, document understanding, and memory.

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Tests](https://img.shields.io/badge/tests-5939_pass-brightgreen.svg)](#testing)
[![LLaMA 3.1 8B](https://img.shields.io/badge/Model-LLaMA--3.1--8B-green.svg)](https://huggingface.co/meta-llama/Meta-Llama-3.1-8B-Instruct)

---

## Overview

Friday is a personal AI assistant built for screenwriting workflows. It combines a fine-tuned LLaMA 3.1 8B model with a multi-layered memory system, document processing pipeline, voice interaction, and 30 MCP tools for scene management, book mentoring, email, and more.

**Key capabilities:**
- Scene management (search, create, update, reorder, link scenes)
- Book understanding (ingest PDFs, extract concepts/principles, mentor sessions)
- Multi-layered memory (working, short-term, long-term, knowledge graph)
- Voice pipeline (wake word, STT, TTS with voice cloning)
- Telugu-English code-switching personality
- Gmail integration for screenplay delivery

---

## Architecture

```
User (Voice/Text)
        |
  [Wake Word: "Hey Friday"]
        |
  [Faster-Whisper STT]
        |
  [FridayOrchestrator]
    |         |          |
    |    [GLM Router]    |
    |    (30 tools)      |
    |         |          |
[Context   [MCP Tools]  [Memory
 Builder]   |  |  |      Manager]
    |       |  |  |        |
    |    Scene Doc Voice   |
    |    Mgr   Proc Pipe  Working
    |          |          Short-term
    |       [Book         Long-term
    |     Understanding]  Knowledge Graph
    |          |          Profile Store
    |       Mentor
    |       Engine
    |
  [LLaMA 3.1 8B + LoRA]
        |
  [XTTS v2 TTS]
        |
    Speaker Output
```

---

## Quick Start

### Prerequisites

- Python 3.10+
- PostgreSQL (for screenplay database)
- Qdrant (optional, for vector search)

### Setup

```bash
git clone <repository-url>
cd Friday
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

cp .env.template .env
# Edit .env with your configuration
```

### Run the Server

```bash
uvicorn orchestrator.main:app --reload --port 8000
```

### Run Tests

```bash
python -m pytest tests/ -x -q
# 5939 passed, 5 skipped in ~13s
```

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/chat` | Chat with Friday |
| `POST` | `/chat/voice` | Voice-specific chat (for daemon) |
| `GET` | `/tools` | List available tools |
| `POST` | `/tools/execute` | Execute a tool directly |
| `GET` | `/sessions` | List active sessions |
| `POST` | `/sessions` | Create new session |
| `GET` | `/context` | Get current context room |
| `POST` | `/context/{type}` | Switch context room |
| `GET` | `/health` | Health check |

---

## Project Structure

```
Friday/
├── orchestrator/               # Central orchestrator
│   ├── core.py                 # FridayOrchestrator (session, context, routing)
│   ├── main.py                 # FastAPI server
│   ├── routes/                 # API route handlers (chat, tools, sessions)
│   ├── config.py               # Orchestrator configuration
│   ├── context/                # Context rooms & detection
│   │   ├── contexts.py         # 4 rooms: writers_room, kitchen, storyboard, general
│   │   ├── detector.py         # Keyword + location context detection
│   │   └── builder.py          # Context builder with memory/doc injection
│   ├── inference/              # LLM inference
│   │   ├── router.py           # GLM Router (30-tool system prompt + keyword fallback)
│   │   └── local_llm.py        # Local LLM client (vLLM/llama.cpp)
│   ├── memory/                 # Memory adapters
│   │   └── working_memory_adapter.py
│   └── tools/                  # Tool registry
│       └── registry.py         # 30 MCP tools registered
│
├── memory/                     # Multi-layered memory system
│   ├── manager.py              # MemoryManager (coordinates all layers)
│   ├── config.py               # Memory configuration
│   ├── layers/
│   │   ├── working.py          # Working memory (context window, token counting)
│   │   ├── short_term.py       # Short-term (SQLite, 7-day retention)
│   │   ├── long_term.py        # Long-term (vector embeddings, decay scoring)
│   │   ├── knowledge_graph.py  # Knowledge graph (NetworkX, triplet extraction)
│   │   └── profile.py          # Profile store (persistent identity)
│   └── operations/
│       ├── triplet_extractor.py # Entity-relation extraction
│       ├── decay.py            # Memory decay daemon
│       └── conversation.py     # Conversation memory (turn management)
│
├── documents/                  # Document processing pipeline
│   ├── manager.py              # DocumentManager (coordinator)
│   ├── config.py               # Document configuration
│   ├── models.py               # Document, Page, Chunk dataclasses
│   ├── ocr/
│   │   └── deepseek_engine.py  # DeepSeek-OCR 2 (4-bit quantized)
│   ├── pipeline/
│   │   ├── pdf_processor.py    # PDF to images conversion
│   │   └── chunker.py          # Semantic chunker (chapter-aware)
│   ├── storage/
│   │   ├── document_store.py   # SQLite storage with FTS5
│   │   └── understanding_store.py # Book understanding persistence
│   ├── retrieval/
│   │   ├── searcher.py         # Hybrid search (BM25 + vector)
│   │   └── citation.py         # Citation tracking & formatting
│   └── understanding/          # Book understanding layer
│       ├── comprehension.py    # BookComprehensionEngine
│       ├── mentor.py           # MentorEngine (analyze, brainstorm, check_rules)
│       ├── job_tracker.py      # StudyJobTracker (progress, ETA)
│       ├── graph_integration.py # Book-to-knowledge-graph linking
│       └── models.py           # Concept, Principle, Technique, Example
│
├── mcp/                        # MCP tool servers
│   ├── scene_manager/          # Scene management (8 tools)
│   │   ├── service.py          # Scene CRUD operations
│   │   └── server.py           # MCP JSON-RPC server
│   ├── documents/              # Document processing (20+ tools)
│   │   ├── service.py          # Document + book study + mentor tools
│   │   └── server.py           # MCP JSON-RPC server
│   ├── gmail/                  # Email tools (2 tools)
│   │   ├── service.py          # Gmail API integration
│   │   └── server.py           # MCP JSON-RPC server
│   └── voice/                  # Voice control tools (6 tools)
│       ├── service.py          # Voice daemon control
│       └── server.py           # MCP JSON-RPC server
│
├── voice/                      # Voice interaction pipeline
│   ├── daemon.py               # Voice daemon state machine
│   ├── config.py               # Voice configuration
│   ├── audio/
│   │   ├── capture.py          # Microphone input (sounddevice)
│   │   ├── playback.py         # Speaker output
│   │   └── vad.py              # WebRTC Voice Activity Detection
│   ├── stt/
│   │   ├── faster_whisper_service.py # Faster-Whisper STT
│   │   └── language_detector.py      # Telugu/English detection
│   ├── tts/
│   │   ├── xtts_service.py     # XTTS v2 TTS with voice cloning
│   │   └── voice_profiles.py   # Voice profile management
│   ├── wakeword/
│   │   ├── openwakeword_service.py # "Hey Friday" detection
│   │   └── trainer.py          # Custom wake word trainer
│   └── storage/
│       ├── audio_storage.py    # WAV + transcript persistence
│       └── training_generator.py # Training data from conversations
│
├── db/                         # Database layer
│   ├── config.py               # DatabaseSettings, get_engine()
│   ├── utils.py                # create_all(), get_schema_snapshot()
│   ├── schema.py               # Screenplay schema (projects, scenes, characters)
│   ├── agent_schema.py         # Agent schema (suggestions, analysis)
│   ├── voice_schema.py         # Voice schema (sessions, turns, profiles)
│   ├── training_schema.py      # Training schema (datasets, runs, artifacts)
│   ├── screenplay_schema.py    # Extended screenplay models
│   ├── vector_store.py         # Qdrant abstraction (VectorStore ABC)
│   ├── init.sql                # PostgreSQL extensions (pgvector, FTS)
│   └── migrations/             # Schema migration scripts
│
├── src/
│   ├── training/               # SageMaker LoRA fine-tuning
│   │   └── vscode_sagemaker_trainer.py
│   ├── inference/              # SageMaker inference
│   └── memory/
│       └── store.py            # Legacy file-based memory store
│
├── tests/                      # 59 test files, 5939 tests
│   ├── test_orchestrator_core.py
│   ├── test_router.py
│   ├── test_working_memory.py
│   ├── test_knowledge_graph.py
│   ├── test_comprehension_engine.py
│   ├── test_mentor_engine.py
│   ├── test_voice_daemon.py
│   └── ... (56 more test files)
│
├── data/                       # Training & persona data
│   ├── instructions/           # Training datasets (ChatML format)
│   ├── interviews/             # Interview sessions for persona capture
│   ├── persona/                # Personality definitions
│   └── phase2/                 # Phase 2 behavioral conversations
│
├── docs/                       # Documentation
│   ├── ARCHITECTURE_VISUAL.md  # System architecture diagrams
│   ├── CONTEXT_WINDOW_MANAGEMENT.md
│   ├── TRAINING_METHODOLOGY.md
│   └── research/               # Research notes
│
├── Dockerfile                  # Docker deployment
├── docker-compose.dgx.yaml    # DGX Spark compose
├── PROJECT_STATUS.md           # Detailed component status
└── requirements.txt            # Python dependencies
```

---

## MCP Tools (30 Registered)

### Scene Manager (8 tools)
| Tool | Description |
|------|-------------|
| `scene_search` | Search scenes by text or vector similarity |
| `scene_get` | Get scene details by number |
| `scene_update` | Update scene text, status, metadata |
| `scene_create` | Create a new scene |
| `scene_delete` | Delete a scene |
| `scene_reorder` | Reorder scenes (before/after positioning) |
| `scene_link` | Link scenes (flashback, sequence, parallel) |
| `scene_list` | List all scenes in a project |

### Document & Book Tools (14 tools)
| Tool | Description |
|------|-------------|
| `document_ingest` | Upload and process a PDF |
| `document_search` | Search across documents with citations |
| `document_get_chapter` | Get full chapter text |
| `document_get_page` | Get specific page content |
| `document_compare` | Compare two documents on a topic |
| `document_list` | List all ingested documents |
| `book_study` | Study a book, extract all knowledge |
| `book_study_status` | Live progress with ETA |
| `book_study_jobs` | List all study jobs |
| `book_list_studied` | List studied books |
| `book_get_understanding` | Get full understanding |
| `mentor_load_books` | Load books for mentor session |
| `mentor_analyze` | Analyze scene against book knowledge |
| `mentor_brainstorm` | Brainstorm using book principles |

### Mentor Tools (4 tools)
| Tool | Description |
|------|-------------|
| `mentor_check_rules` | Check if scene follows/violates rules |
| `mentor_find_inspiration` | Find relevant examples from books |
| `mentor_ask` | Ask what books say about a topic |
| `mentor_compare` | Compare views across books |

### Other Tools (4 tools)
| Tool | Description |
|------|-------------|
| `knowledge_search` | Search knowledge graph |
| `gmail_send_screenplay` | Email a screenplay |
| `gmail_send_email` | Send a general email |
| `voice_speak` | Synthesize and speak text |

---

## Memory System

Friday uses a multi-layered memory architecture:

| Layer | Purpose | Storage |
|-------|---------|---------|
| **Working Memory** | Current conversation context window | In-memory (token-counted) |
| **Short-term Memory** | Recent facts, 7-day retention | SQLite |
| **Long-term Memory** | Persistent memories with decay scoring | SQLite + vector embeddings |
| **Knowledge Graph** | Entity relationships (NetworkX) | In-memory + serialized |
| **Profile Store** | Persistent identity, version history | JSON files |
| **Conversation Memory** | Turn management across sessions | SQLite |

---

## Docker Deployment (DGX Spark)

```bash
# Copy environment config
cp .env.dgx.example .env

# Edit configuration
# FRIDAY_LLM_BASE_URL, FRIDAY_LLM_MODEL, FRIDAY_LLM_BACKEND

# Launch services
docker compose -f docker-compose.dgx.yaml up -d
```

Services: vLLM (port 8001), Orchestrator (port 8000), PostgreSQL, Redis, Qdrant.

---

## Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `FRIDAY_LLM_BASE_URL` | `http://localhost:8001` | LLM inference endpoint |
| `FRIDAY_LLM_MODEL` | `meta-llama/Meta-Llama-3.1-8B-Instruct` | Model name |
| `FRIDAY_LLM_BACKEND` | `vllm` | Backend type (vllm/llamacpp) |
| `FRIDAY_DEFAULT_PROJECT` | `aa-janta-naduma` | Default screenplay project |
| `FRIDAY_PORT` | `8000` | Server port |
| `FRIDAY_HOST` | `0.0.0.0` | Server host |
| `DB_HOST` | `localhost` | PostgreSQL host |
| `DB_PORT` | `5432` | PostgreSQL port |
| `DB_NAME` | `vectordb` | Database name |
| `QDRANT_URL` | - | Qdrant connection URL |

---

## Testing

```bash
# Run all tests
python -m pytest tests/ -x -q

# Run specific test file
python -m pytest tests/test_orchestrator_core.py -v

# Run with coverage
python -m pytest tests/ --cov=orchestrator --cov=memory --cov=documents
```

**59 test files | 5,939 tests | 5 skipped (need LLM backend) | ~13 seconds**

---

## Training Pipeline

LoRA fine-tuning on AWS SageMaker for personality capture:

```bash
# Set environment
export MODEL_NAME=meta-llama/Meta-Llama-3.1-8B-Instruct

# Launch training (~$3 on ml.g5.2xlarge)
python src/training/vscode_sagemaker_trainer.py
```

**LoRA Config:** r=32, alpha=64, 3 epochs, lr=1e-4

---

## Documentation

| Document | Description |
|----------|-------------|
| [PROJECT_STATUS.md](PROJECT_STATUS.md) | Detailed component status |
| [docs/ARCHITECTURE_VISUAL.md](docs/ARCHITECTURE_VISUAL.md) | System architecture diagrams |
| [docs/CONTEXT_WINDOW_MANAGEMENT.md](docs/CONTEXT_WINDOW_MANAGEMENT.md) | Working memory design |
| [docs/TRAINING_METHODOLOGY.md](docs/TRAINING_METHODOLOGY.md) | Scientific training approach |
| [docs/PHASE2_CONVERSATIONAL_DATA.md](docs/PHASE2_CONVERSATIONAL_DATA.md) | Phase 2 data collection guide |
| [docs/research/](docs/research/) | Research notes (TTS, Telugu LoRA, vector DBs) |

---

## License

Personal use project. Training pipeline architecture can be adapted with attribution.
