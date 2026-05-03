# Supermemory Architecture Analysis for Friday AI

> Deep research on memory architecture for 24/7 autonomous AI operation

**Research Date**: January 29, 2026

---

## Executive Summary

Supermemory is a **state-of-the-art memory engine** that achieved SOTA on LongMemEval benchmark. Their architecture is well-suited for inspiration but **not a direct fit** for Friday AI's use case. Key takeaways:

| Aspect | Supermemory | Friday AI Needs |
|--------|-------------|-----------------|
| Deployment | Cloud API service | Local 24/7 server |
| Focus | Multi-user SaaS | Single user (Boss) |
| Memory | Document/URL ingestion | Conversational memory |
| Scale | Billions of docs | Personal context |

**Recommendation**: Adopt their **core concepts** (dual timestamps, memory atomicity, intelligent decay) but build a **custom lightweight implementation** for Friday.

---

## Supermemory Architecture Deep Dive

### Core Innovation: Brain-Inspired Memory Layers

```
┌─────────────────────────────────────────────────────────┐
│                 SUPERMEMORY ARCHITECTURE                 │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  HOT MEMORY (Working)                                   │
│  ├─ Cloudflare KV                                       │
│  ├─ Sub-millisecond access                              │
│  └─ Recent, frequently accessed                         │
│                                                         │
│  WARM MEMORY (Short-term)                               │
│  ├─ PostgreSQL + Vector extensions                      │
│  ├─ Semantic search enabled                             │
│  └─ Session-level context                               │
│                                                         │
│  COLD MEMORY (Long-term)                                │
│  ├─ Vector database (custom on R2)                      │
│  ├─ Intelligent decay applied                           │
│  └─ Archival, less frequently accessed                  │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

### Key Technical Innovations

#### 1. Dual-Layer Timestamping

This is **critical for temporal reasoning** - the #1 weakness in most RAG systems.

```python
# Supermemory's approach
{
    "memory": "Boss mentioned he wants to finish Gusagusalu by March",
    "documentDate": "2026-01-29T10:30:00Z",  # When conversation happened
    "eventDate": "2026-03-01T00:00:00Z"       # When event will occur
}
```

**Why it matters**: Without this, the model can't answer "What did Boss say about deadlines last week?" vs "What's coming up next month?"

#### 2. Memory Atomicity (High-Signal Extraction)

```
RAW CONVERSATION (noisy):
"Hey Friday, so I was thinking about the climax scene,
you know the one where Ravi confronts his father?
I think it needs more emotional punch.
Maybe add a flashback? What do you think?"

         ↓ ATOMICITY EXTRACTION ↓

MEMORY (high-signal):
"Boss wants climax scene (Ravi-father confrontation)
to have more emotional punch. Considering flashback."

CHUNK (preserved for detail):
[Full original conversation stored separately]
```

**Why it matters**: Searching high-signal memories is faster and more accurate than searching noisy chunks.

#### 3. Intelligent Decay (Active Forgetting)

```python
# Decay formula (conceptual)
memory_score = (
    relevance_weight * semantic_similarity +
    recency_weight * time_decay_function(age) +
    frequency_weight * access_count +
    importance_weight * user_reinforcement
)

# If score < threshold → memory decays
# If score > threshold → memory strengthened
```

**Key insight**: "Forgetting is not a bug, it's a feature." Without decay:
- Noise accumulates
- Retrieval quality degrades
- Context becomes bloated
- Old, outdated info pollutes responses

---

## LongMemEval Benchmark Insights

Supermemory achieved **SOTA on LongMemEval** which tests 5 core memory abilities:

| Ability | What it tests | Supermemory Score |
|---------|---------------|-------------------|
| **Information Extraction** | Capturing key details | High |
| **Multi-Session Reasoning** | Connecting across conversations | 71.43% |
| **Temporal Reasoning** | Time-based relationships | 76.69% |
| **Knowledge Updates** | Handling corrections/changes | High |
| **Abstention** | Knowing when NOT to answer | High |

### Why This Matters for Friday

Friday running 24/7 will accumulate **massive conversation history**. Without proper memory architecture:
- "What did I say about that scene last week?" → Wrong answer
- Outdated preferences override current ones
- Context window fills with irrelevant old data
- Performance degrades over time

---

## Comparison: Memory Solutions

| Feature | Supermemory | Mem0 | Custom (Recommended) |
|---------|-------------|------|---------------------|
| Self-hosted | Enterprise only | Yes | Yes |
| Latency | ~400ms | Higher | Can be <100ms |
| GitHub Stars | 13.6K | 43.5K | N/A |
| Temporal reasoning | Excellent | Good | Implement ourselves |
| Single-user focus | No (multi-tenant) | No | Yes (perfect fit) |
| Telugu support | No special handling | No | Can optimize |
| Cost | API fees | Self-host free | Free |
| Control | Limited | Full | Full |

---

## Recommended Architecture for Friday AI

### Design Principles (Borrowed from Supermemory)

1. **Dual timestamps** for every memory
2. **Memory atomicity** - extract high-signal facts
3. **Intelligent decay** - forget what's not useful
4. **Hierarchical storage** - hot/warm/cold tiers
5. **Semantic + temporal search** combined

### Proposed Friday Memory Architecture

```
┌─────────────────────────────────────────────────────────┐
│              FRIDAY MEMORY ARCHITECTURE                  │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  WORKING MEMORY (Redis/In-Memory)                       │
│  ├─ Current conversation context                        │
│  ├─ Last 20 turns                                       │
│  ├─ Active project context (Gusagusalu, etc.)          │
│  └─ TTL: Session duration                               │
│                                                         │
│  SHORT-TERM MEMORY (SQLite + Embeddings)                │
│  ├─ Recent conversations (7 days)                       │
│  ├─ Extracted facts/preferences                         │
│  ├─ Project-specific memories                           │
│  └─ Decay: Weekly consolidation                         │
│                                                         │
│  LONG-TERM MEMORY (Vector DB - Local)                   │
│  ├─ Consolidated knowledge                              │
│  ├─ Boss's preferences & patterns                       │
│  ├─ Project archives                                    │
│  ├─ Important events with timestamps                    │
│  └─ Decay: Monthly pruning                              │
│                                                         │
│  PROFILE (Persistent Facts)                             │
│  ├─ Static: "Boss prefers direct communication"         │
│  ├─ Dynamic: "Currently working on Gusagusalu"          │
│  └─ Never decays, only updates                          │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

### Memory Flow

```
User Message
    ↓
┌─────────────────┐
│ GLM-4.7 Router  │ ← Analyzes intent, suggests tools
└────────┬────────┘
         ↓
┌─────────────────┐
│ Memory Retrieval │
│ ├─ Profile       │ ← Always included
│ ├─ Working mem   │ ← Recent context
│ ├─ Semantic search│ ← If topic-related
│ └─ Temporal query │ ← If time-related
└────────┬────────┘
         ↓
┌─────────────────┐
│ LLaMA 3.1 8B    │ ← Response generation
└────────┬────────┘
         ↓
┌─────────────────┐
│ Memory Storage   │
│ ├─ Add to working│
│ ├─ Extract facts │ ← Atomicity
│ └─ Update profile│ ← If preference detected
└─────────────────┘
```

### Implementation Components

#### 1. Memory Schema

```python
@dataclass
class Memory:
    id: str
    content: str                    # High-signal extracted fact
    raw_chunk: str                  # Original conversation
    embedding: List[float]          # Vector for semantic search

    # Dual timestamps
    created_at: datetime            # When conversation happened
    event_date: Optional[datetime]  # When referenced event occurs

    # Metadata
    memory_type: str                # fact, preference, event, project
    project: Optional[str]          # Gusagusalu, Kitchen, etc.
    confidence: float               # Extraction confidence

    # Decay tracking
    access_count: int               # How often retrieved
    last_accessed: datetime         # For recency scoring
    importance: float               # User-reinforced importance

    # Telugu support
    language: str                   # en, te, mixed
    telugu_keywords: List[str]      # For Telugu-aware search
```

#### 2. Decay Algorithm

```python
def calculate_memory_score(memory: Memory) -> float:
    """
    Score determines if memory should be kept or decayed.
    Inspired by Supermemory's intelligent decay.
    """
    now = datetime.now()

    # Recency factor (exponential decay)
    days_old = (now - memory.last_accessed).days
    recency = math.exp(-0.1 * days_old)  # Decay rate

    # Frequency factor (log scale)
    frequency = math.log(memory.access_count + 1) / 10

    # Importance (user-reinforced)
    importance = memory.importance

    # Event relevance (upcoming events score higher)
    event_relevance = 0
    if memory.event_date:
        days_until = (memory.event_date - now).days
        if 0 <= days_until <= 30:  # Upcoming month
            event_relevance = 0.5

    # Weighted combination
    score = (
        0.3 * recency +
        0.2 * frequency +
        0.3 * importance +
        0.2 * event_relevance
    )

    return score

def decay_memories(threshold: float = 0.2):
    """Remove memories below threshold, consolidate similar ones."""
    for memory in get_all_memories():
        score = calculate_memory_score(memory)
        if score < threshold:
            archive_or_delete(memory)
        elif should_consolidate(memory):
            consolidate_with_similar(memory)
```

#### 3. Fact Extraction (Atomicity)

```python
EXTRACTION_PROMPT = """
Extract key facts from this conversation turn.
Return JSON array of atomic facts.

Rules:
- Each fact should be self-contained
- Include temporal references if mentioned
- Capture preferences, decisions, and plans
- Ignore small talk and filler

Conversation:
{conversation}

Output format:
[
  {{"fact": "...", "type": "preference|event|decision|info", "event_date": null|"YYYY-MM-DD"}}
]
"""

async def extract_facts(conversation: str) -> List[Dict]:
    """Use GLM-4.7-Flash for fast fact extraction."""
    response = await glm_router.chat(
        messages=[{"role": "user", "content": EXTRACTION_PROMPT.format(
            conversation=conversation
        )}],
        temperature=0.1,  # Deterministic
        max_tokens=500
    )
    return json.loads(response.content)
```

#### 4. Profile Management

```python
class UserProfile:
    """
    Persistent facts about Boss that never decay.
    Updated only when new information confirms changes.
    """

    static_facts: Dict[str, str] = {
        "name": "Poorna",
        "profession": "Telugu screenwriter",
        "communication_style": "Direct, no flattery",
        "languages": ["Telugu", "English"],
        "address_as": "Boss",
    }

    dynamic_facts: Dict[str, Any] = {
        "current_project": "Gusagusalu",
        "current_focus": "Climax scene refinement",
        "recent_mood": "focused",
        "last_updated": "2026-01-29"
    }

    preferences: Dict[str, str] = {
        "response_length": "concise",
        "telugu_usage": "natural, for emotions",
        "feedback_style": "direct with opinions",
    }

    def update_dynamic(self, key: str, value: Any):
        """Update dynamic fact with timestamp."""
        self.dynamic_facts[key] = value
        self.dynamic_facts["last_updated"] = datetime.now().isoformat()
```

---

## Implementation Roadmap

### Phase 1: Core Memory (Week 1-2)
- [ ] Memory schema and SQLite storage
- [ ] Basic working memory (conversation buffer)
- [ ] Simple embedding-based retrieval
- [ ] Profile system

### Phase 2: Intelligence (Week 3-4)
- [ ] Fact extraction with GLM-4.7-Flash
- [ ] Dual timestamp implementation
- [ ] Decay algorithm
- [ ] Temporal query support

### Phase 3: Optimization (Week 5-6)
- [ ] Telugu-aware embeddings
- [ ] Memory consolidation
- [ ] Profile auto-updating
- [ ] Performance tuning

### Phase 4: Integration (Week 7-8)
- [ ] Integrate with FridayOrchestrator
- [ ] Connect to GLM router
- [ ] End-to-end testing
- [ ] 24/7 operation validation

---

## Key Takeaways

### What to Adopt from Supermemory

1. **Dual-layer timestamps** - Critical for temporal reasoning
2. **Memory atomicity** - Extract high-signal facts from noisy conversations
3. **Intelligent decay** - Active forgetting prevents noise accumulation
4. **Hierarchical storage** - Hot/warm/cold tiers for efficient retrieval
5. **Profile separation** - Static vs dynamic facts

### What NOT to Adopt

1. **Cloud-first architecture** - We need local 24/7 operation
2. **Multi-tenant design** - Single user (Boss) only
3. **Document ingestion focus** - We focus on conversations
4. **External API dependency** - Self-contained is better

### Friday-Specific Additions

1. **Telugu-aware search** - Embeddings that understand code-switching
2. **Project context** - Gusagusalu, Kitchen as memory scopes
3. **Creative memory** - Screenplay ideas, character notes
4. **Preference learning** - Understand Boss's style over time

---

## References

- [Supermemory GitHub](https://github.com/supermemoryai/supermemory)
- [Supermemory Research - SOTA on LongMemEval](https://supermemory.ai/research)
- [Memory Engine Architecture Blog](https://blog.supermemory.ai/memory-engine/)
- [LongMemEval Paper](https://arxiv.org/abs/2410.10813)
- [DeepWiki - System Architecture](https://deepwiki.com/supermemoryai/supermemory/1.1-system-architecture)
- [BetterStack - Long-term Memory Guide](https://betterstack.com/community/guides/ai/memory-with-supermemory/)
- [Mem0 Research](https://mem0.ai/research)
- [Agent Memory Survey](https://github.com/Shichun-Liu/Agent-Memory-Paper-List)

---

## Verdict

**Should we use Supermemory directly?** No.

**Should we learn from their architecture?** Absolutely.

Their core innovations - dual timestamps, memory atomicity, intelligent decay - are exactly what Friday needs for 24/7 autonomous operation. But their cloud SaaS model and multi-tenant design don't fit our single-user, local deployment needs.

**Recommendation**: Build a custom memory layer inspired by Supermemory's principles, optimized for:
- Single user (Boss)
- Telugu-English conversations
- Creative/screenplay domain
- Local 24/7 operation
- Privacy (no data leaves the server)

---

*"Your brain doesn't store everything perfectly as you see it—and that's actually a feature, not a bug."* - Supermemory Blog
