# Friday AI - Project Status

> Last Updated: February 7, 2026 | Branch: `iteration_5` | Tests: 5,939 pass, 5 skip

---

## Current State: Quality Foundation Complete

The full Friday AI system is built and comprehensively tested. Every module has
test coverage (59 test files, 5,939 tests). The next phase is training data
collection and model deployment.

---

## Component Status

### Orchestrator
<!-- DONE: FridayOrchestrator with session management -->
<!-- DONE: FastAPI server (chat, tools, sessions, health endpoints) -->
<!-- DONE: GLM-4.7-Flash router with 30-tool system prompt + keyword fallback -->
<!-- DONE: 4 context rooms (writers_room, kitchen, storyboard, general) -->
<!-- DONE: Context builder with document/memory injection -->
<!-- DONE: Full MCP tool routing via async_execute() -->
<!-- DONE: Memory context injection via ContextBuilder -->

### Memory System
<!-- DONE: Working Memory with context window management -->
<!-- DONE: Token counting with tiktoken/fallback -->
<!-- DONE: Context poisoning detection -->
<!-- DONE: Hybrid buffer (compressed + verbatim) -->
<!-- DONE: Short-term memory (SQLite, 7-day retention) -->
<!-- DONE: Long-term memory (vector embeddings, decay scoring) -->
<!-- DONE: Knowledge graph (triplet extraction, NetworkX) -->
<!-- DONE: Profile store (persistent identity, version history) -->
<!-- DONE: Decay daemon (scheduled consolidation) -->
<!-- DONE: Conversation memory (turn management) -->
<!-- DONE: Memory manager (coordinates all layers) -->

### Document Processing
<!-- DONE: DeepSeek-OCR 2 integration (4-bit quantized) -->
<!-- DONE: Semantic chunker (chapter-aware, overlap handling) -->
<!-- DONE: SQLite document store with FTS5 search -->
<!-- DONE: Hybrid retrieval (BM25 + semantic vector search) -->
<!-- DONE: Citation tracker with page-range formatting -->
<!-- DONE: Cloud sync for server deployment -->

### Book Understanding
<!-- DONE: Book Comprehension Engine (concepts, principles, techniques, examples) -->
<!-- DONE: Mentor Engine (analyze, brainstorm, check_rules, find_inspiration) -->
<!-- DONE: Study Job Tracker (live progress, ETA, voice-friendly status) -->
<!-- DONE: Book-to-Knowledge Graph integration -->
<!-- DONE: Understanding Store with FTS5 -->

### MCP Tools (30 tools registered)
<!-- DONE: Scene Manager (search, get, update, reorder, link, create, delete, list) -->
<!-- DONE: Document Processor (ingest, search, get_chapter, get_page, compare, list) -->
<!-- DONE: Book Study (study, status, jobs, list_studied, get_understanding) -->
<!-- DONE: Mentor (load_books, analyze, brainstorm, check_rules, find_inspiration, ask, compare) -->
<!-- DONE: Knowledge search -->
<!-- DONE: Gmail MCP (send_screenplay, send_email) -->
<!-- DONE: Voice MCP (start/stop, speak, sessions, export, profiles) -->

### Voice Pipeline (code done, needs hardware testing)
<!-- DONE: Voice daemon state machine -->
<!-- DONE: Faster-Whisper STT with language detection -->
<!-- DONE: XTTS v2 TTS with voice cloning profiles -->
<!-- DONE: OpenWakeWord for "Hey Friday" -->
<!-- DONE: WebRTC VAD -->
<!-- DONE: Audio capture/playback -->
<!-- DONE: Audio storage for training data generation -->
<!-- DONE: Wake word trainer for custom phrases -->
<!-- TODO: Hardware integration testing with microphone/speakers -->
<!-- TODO: Voice sample collection for XTTS cloning -->

### Database Layer
<!-- DONE: Screenplay schema (projects, scenes, characters, elements, dialogue) -->
<!-- DONE: Agent schema (suggestions, analysis runs, face profiles) -->
<!-- DONE: Voice schema (sessions, turns, profiles, training examples) -->
<!-- DONE: Training schema (datasets, model versions, runs, artifacts) -->
<!-- DONE: Qdrant VectorStore abstraction with singleton -->
<!-- DONE: Migration scripts with rollback -->

### Training Pipeline
<!-- DONE: SageMaker LoRA fine-tuning (single/multi-GPU) -->
<!-- DONE: Scientific training methodology with experiment tracking -->
<!-- DONE: Interview data collection (120 exchanges, 15 topics) -->
<!-- DONE: Phase 2 behavioral conversation recorder -->
<!-- DONE: Data quality tools (auto-fix, validation) -->
<!-- REVIEW: Transformed interview data needs Boss review -->
<!-- TODO: Curate 350 WhatsApp examples from 6,631 pool -->
<!-- TODO: Create 25 contrastive pairs (chosen vs rejected) -->
<!-- TODO: Build combined iteration2 dataset (~525 examples) -->

### Infrastructure
<!-- DONE: Docker DGX deployment (Dockerfile, compose, init.sql) -->
<!-- DONE: FastAPI server with CORS, lifespan management -->
<!-- DONE: Environment-based configuration -->
<!-- TODO: DGX Spark model downloads and Docker testing -->

---

## Test Coverage

59 test files | 5,939 tests | 5 skipped (need LLM backend) | ~13 seconds

```
python -m pytest tests/ -x -q
```

---

## Pending Tasks

### No Boss Needed
<!-- TODO: DeepSeek OCR 2 - Test with sample PDFs on GPU -->
<!-- TODO: Telugu LoRA - Download IndicAlign + train adapter -->
<!-- TODO: Voice Pipeline - Benchmark IndicF5 latency -->
<!-- TODO: DGX Spark - Model downloads, test Docker compose -->

### Needs Boss
<!-- TODO: Phase 2 behavioral conversations (3/150 done) -->
<!-- TODO: Voice sample collection (Telugu+English) -->
<!-- TODO: WhatsApp data curation (350 from 6,631 pool) -->
<!-- TODO: Contrastive pair creation (25 examples) -->

---

## Quick Links

| Resource | Path |
|----------|------|
| Architecture Diagram | [docs/ARCHITECTURE_VISUAL.md](docs/ARCHITECTURE_VISUAL.md) |
| Context Window Docs | [docs/CONTEXT_WINDOW_MANAGEMENT.md](docs/CONTEXT_WINDOW_MANAGEMENT.md) |
| Memory Architecture | [docs/architecture/FRIDAY_MEMORY_ARCHITECTURE.md](docs/architecture/FRIDAY_MEMORY_ARCHITECTURE.md) |
| Training Methodology | [docs/TRAINING_METHODOLOGY.md](docs/TRAINING_METHODOLOGY.md) |
| Phase 2 Data Guide | [docs/PHASE2_CONVERSATIONAL_DATA.md](docs/PHASE2_CONVERSATIONAL_DATA.md) |
| Research Notes | [docs/research/](docs/research/) |

---

## Tag Legend

| Tag | Meaning | Color |
|-----|---------|-------|
| `TODO` | Work to be done | Yellow |
| `NEXT` | Immediate next task | Blue |
| `REVIEW` | Needs human review | Purple |
| `DONE` | Completed | Green |
| `FIXME` | Bug or issue | Red |
| `BLOCKED` | Waiting on something | Dark Red |
