# Friday AI - Project Status

> Open Todo Tree sidebar (click the tree icon) to see all tasks across the codebase.

---

## Current Sprint

<!-- NEXT: Training Data Pipeline - Iteration 2 -->
<!-- REVIEW: Interview transformations in data/interviews/transformed/ -->

---

## Component Status

### Memory System
<!-- DONE: Working Memory with context window management -->
<!-- DONE: Token counting with tiktoken/fallback -->
<!-- DONE: Context poisoning detection -->
<!-- DONE: Hybrid buffer (compressed + verbatim) -->
<!-- DONE: Short-term memory (SQLite) -->
<!-- DONE: Long-term memory (vector search) -->
<!-- DONE: Knowledge graph (triplet extraction) -->

### Training Pipeline
<!-- DONE: 120 interview exchanges collected -->
<!-- DONE: Interview transformation to behavioral data -->
<!-- REVIEW: Transformed data needs Boss review -->
<!-- TODO: Curate 350 WhatsApp examples from 6,631 pool -->
<!-- TODO: Create 25 contrastive pairs (chosen vs rejected) -->
<!-- TODO: Expand tool examples from 12 to 30 -->
<!-- TODO: Build combined iteration2 dataset (~525 examples) -->
<!-- TODO: Set MODEL_NAME=meta-llama/Meta-Llama-3.1-8B-Instruct in .env -->

### Voice Pipeline
<!-- TODO: Voice daemon with state machine -->
<!-- TODO: Faster-Whisper STT integration -->
<!-- TODO: XTTS v2 TTS with voice cloning -->
<!-- TODO: OpenWakeWord for "Hey Friday" -->
<!-- TODO: Audio storage for training data generation -->

### MCP Tools
<!-- DONE: Scene Manager (search, update, reorder, link) -->
<!-- DONE: Gmail MCP for email -->
<!-- TODO: Document Processor (DeepSeek-OCR integration) -->
<!-- TODO: Calendar integration -->

### Orchestrator
<!-- DONE: Basic FridayOrchestrator -->
<!-- DONE: GLM-4 router integration -->
<!-- TODO: Full MCP tool routing -->
<!-- TODO: Memory context injection -->

---

## Quick Links

| Resource | Path |
|----------|------|
| Architecture Diagram | [docs/diagrams/friday_architecture.drawio](docs/diagrams/friday_architecture.drawio) |
| Visual Architecture | [docs/ARCHITECTURE_VISUAL.md](docs/ARCHITECTURE_VISUAL.md) |
| Context Window Docs | [docs/CONTEXT_WINDOW_MANAGEMENT.md](docs/CONTEXT_WINDOW_MANAGEMENT.md) |
| Training Review Prompt | [prompts/training_data_review_prompt.md](prompts/training_data_review_prompt.md) |
| Master Plan | Check `.claude/plans/` directory |

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

---

## How to Use

1. **Todo Tree Sidebar**: Click tree icon in left panel
2. **Filter by tag**: Click tag name in sidebar
3. **Jump to code**: Click any TODO item
4. **Architecture**: Open `.drawio` file to edit diagrams
5. **Mermaid Preview**: Open `.md` file, press `Cmd+Shift+V`
