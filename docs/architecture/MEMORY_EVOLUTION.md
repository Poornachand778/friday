# Memory Architecture Evolution

How Friday's memory system evolved from research to implementation.

---

## Phase 1: Initial Research (2025-01-29)

### Supermemory Analysis

**Source**: https://github.com/supermemoryai/supermemory

**Key Insights Adopted**:
1. **Dual Timestamps** - `document_time` (when stored) vs `event_time` (when referenced event occurs)
   - Enables temporal queries: "What did we discuss about the March deadline?"
   - Implemented in: `memory/layers/short_term.py` → `STMEntry.event_dates`

2. **Memory Atomicity** - Break conversations into atomic facts
   - Better retrieval precision
   - Implemented in: `STMEntry.key_facts` field

3. **Intelligent Decay** - Not all memories decay equally
   - Higher importance = slower decay
   - Implemented in: `memory/operations/decay.py`

**What We Skipped**:
- Their Cloudflare Workers architecture (we use local SQLite)
- Browser extension features (not needed for Friday)

**Research Doc**: `docs/research/SUPERMEMORY_ANALYSIS.md`

---

### Cognee Analysis

**Source**: https://github.com/topoteretes/cognee

**Key Insights Adopted**:
1. **Knowledge Graph for Relationships**
   - Screenplay work is inherently relational (characters, scenes, projects)
   - "What scenes involve Ravi?" requires graph traversal
   - Implemented in: `memory/layers/knowledge_graph.py`

2. **Triplet Extraction** (Subject-Relation-Object)
   - "Boss discussed climax" → (Boss, discusses, climax)
   - Powers relationship queries
   - Implemented in: `memory/operations/triplet_extractor.py`

3. **Hybrid Search** (Vector + Graph)
   - Vector similarity for content matching
   - Graph traversal for relationship queries
   - Combined approach: LTM (vector) + KnowledgeGraph (graph)

**What We Skipped**:
- Neo4j (too heavy, used NetworkX instead)
- Their full ECL pipeline (simplified for our needs)
- LanceDB (kept our SQLite + sentence-transformers)

**Research Doc**: `docs/research/COGNEE_ANALYSIS.md`

---

## Phase 2: Architecture Design

### Brain-Inspired Layers

```
┌─────────────────────────────────────────────────────────────┐
│                     MEMORY MANAGER                           │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌──────────────┐  Attention stack (7±2 items)              │
│  │   WORKING    │  Current conversation context             │
│  │   MEMORY     │  Token-limited, auto-summarizes           │
│  └──────┬───────┘                                           │
│         │                                                    │
│         ▼ (session end)                                      │
│  ┌──────────────┐  SQLite + FTS5                            │
│  │    SHORT     │  7-day retention                          │
│  │    TERM      │  Dual timestamps                          │
│  │   MEMORY     │  Access tracking for decay                │
│  └──────┬───────┘                                           │
│         │                                                    │
│         ▼ (consolidation)                                    │
│  ┌──────────────┐  Vector embeddings                        │
│  │    LONG      │  paraphrase-multilingual-mpnet-base-v2    │
│  │    TERM      │  Permanent storage                        │
│  │   MEMORY     │  Telugu keyword search                    │
│  └──────┬───────┘                                           │
│         │                                                    │
│         ▼ (triplet extraction)                               │
│  ┌──────────────┐  NetworkX graph                           │
│  │  KNOWLEDGE   │  Entity relationships                     │
│  │    GRAPH     │  GLM-powered extraction                   │
│  └──────────────┘                                           │
│                                                              │
│  ┌──────────────┐  JSON persistence                         │
│  │   PROFILE    │  Identity facts                           │
│  │    STORE     │  NEVER decays                             │
│  └──────────────┘  Voice confirmation for changes           │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### Key Design Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| STM Storage | SQLite + FTS5 | Fast, embedded, full-text search |
| LTM Embeddings | sentence-transformers | Multilingual (Telugu + English) |
| Graph DB | NetworkX | Lightweight, no external service |
| Triplet Extraction | GLM-4.7-Flash | Fast, already integrated for routing |
| Decay Algorithm | Ebbinghaus curve | Brain-inspired, proven model |

---

## Phase 3: Implementation

### Files Created

```
memory/
├── __init__.py                    # Package exports
├── config.py                      # Configuration dataclasses
├── manager.py                     # Central coordinator
├── layers/
│   ├── working.py                 # Attention stack, 7±2 items
│   ├── short_term.py              # SQLite + FTS5
│   ├── long_term.py               # Vector embeddings
│   ├── profile.py                 # Identity store
│   └── knowledge_graph.py         # NetworkX graph (Cognee-inspired)
├── operations/
│   ├── triplet_extractor.py       # GLM-powered extraction
│   └── decay.py                   # Ebbinghaus decay daemon
└── telugu/
    └── processor.py               # Telugu-English code-switching
```

### Integration Points

1. **MemoryManager** is the central coordinator
2. **store_fact()** auto-extracts triplets to graph
3. **DecayDaemon** runs background consolidation
4. **graph_query()** enables relationship queries

---

## Future Evolution Ideas

- [ ] **Sensory Buffer** - Raw audio buffer for voice (100ms-2s)
- [ ] **Timeline Layer** - Explicit temporal events
- [ ] **Pattern Layer** - Learned behaviors and preferences
- [ ] **Multi-modal Memory** - Images, screenshots, storyboards

---

## How to Continue Development

1. **Adding a new memory feature**:
   - Research existing solutions → `docs/research/{NAME}_ANALYSIS.md`
   - Design integration → update this doc
   - Implement → add to appropriate layer
   - Update `CHANGELOG.md`
   - Git commit with descriptive message

2. **Debugging memory issues**:
   - Check `manager.health_check()` for stats
   - Review decay daemon logs
   - Inspect knowledge graph with `graph_query()`

3. **Testing memory**:
   ```python
   from memory import MemoryManager
   manager = MemoryManager()
   await manager.initialize()

   # Store a fact
   await manager.store_fact("Ravi confronts his father in Scene 5")

   # Query relationships
   results = await manager.graph_query("Ravi")
   ```
