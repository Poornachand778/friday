# Context Window Management Architecture

## Overview

Friday operates 24/7, meaning conversation context can grow indefinitely. This document describes the architecture that prevents context overflow, hallucination loops, and the "lost-in-the-middle" effect.

## The Problem

LLMs have finite context windows (typically 4K-128K tokens). Without management:

1. **Token Overflow**: Hard failure when exceeding model limits
2. **Lost-in-the-Middle**: LLMs weigh beginning/end heavily, ignoring middle content
3. **Context Poisoning**: One hallucination enters context, gets repeatedly referenced, reinforcing the error

## Solution: Hybrid Sliding Window

```
┌─────────────────────────────────────────────────────────────────┐
│                    CONTEXT WINDOW (max_tokens)                  │
├─────────────────────────────────────────────────────────────────┤
│  [COMPRESSED HISTORY]  │  [RECENT VERBATIM]  │  [ATTENTION]     │
│  ~20% of capacity      │  ~60% of capacity   │  ~20% reserved   │
│  Summarized turns      │  Last N full turns  │  Topics + LTM    │
└─────────────────────────────────────────────────────────────────┘
```

### Buffer Zones

| Zone | Allocation | Contents |
|------|------------|----------|
| Compressed History | 20% | Summarized older turns |
| Recent Verbatim | 60% | Last N complete turns |
| Attention Reserve | 20% | Topics, prefetched LTM, tool results |

## Capacity-Based Triggers

Instead of reacting at 100% capacity (panic mode), we proactively manage context:

| Zone | Threshold | Action |
|------|-----------|--------|
| Normal | 0-70% | No action needed |
| Proactive | 70-85% | Gentle summarization of oldest turn |
| Aggressive | 85-95% | Compress multiple turns |
| Emergency | 95%+ | Drop oldest content, log warning |

### Why 70%?

Starting at 70% gives headroom for:
- Large user messages
- Tool call results
- LTM prefetch
- Unexpected content bursts

## Token Counting

### Accurate Counting

When `tiktoken` is available:
```python
import tiktoken
encoding = tiktoken.get_encoding("cl100k_base")
tokens = len(encoding.encode(text))
```

### Fallback Estimation

For Telugu-English code-switching:
- English: ~4 characters per token
- Telugu: ~2.5 characters per token (complex script)

```python
def estimate_tokens(text: str) -> int:
    telugu_chars = count_telugu(text)
    english_chars = len(text) - telugu_chars
    return int(telugu_chars / 2.5 + english_chars / 4.0)
```

## Context Poisoning Detection

### The Loop Problem

1. LLM makes incorrect claim
2. Claim enters context
3. LLM references its own claim as "fact"
4. Error reinforces itself

### Detection Strategies

**1. Repetition Detection**
```python
# Track content hashes
if content_hash appears >= 3 times:
    flag_as_potential_hallucination()
```

**2. Confidence Scoring**

Uncertainty markers reduce confidence:
- "I think", "I believe", "might be", "possibly"
- Self-references: "as I mentioned", "like I said"

**3. Quarantine System**

Low-confidence turns (`< 0.5`) are:
- Marked as `is_quarantined = True`
- Excluded from summaries
- Can be dropped first during emergency pruning

## Implementation

### Key Classes

```python
# Token counting
class TokenCounter:
    @classmethod
    def count(cls, text: str) -> int: ...

# Poisoning detection
class ContextPoisoningDetector:
    def analyze_turn(turn) -> (confidence, warnings): ...
    def should_quarantine(turn) -> bool: ...

# Compressed history block
@dataclass
class CompressedHistory:
    summary: str
    turn_count: int
    timestamp_start: datetime
    timestamp_end: datetime
    tokens: int
    topics_covered: List[str]

# Main working memory
class WorkingMemory:
    # Hybrid buffer
    _compressed_history: List[CompressedHistory]
    _turns: List[ConversationTurn]  # Recent verbatim

    # Capacity management
    @property
    def capacity_percentage(self) -> float: ...
    @property
    def capacity_zone(self) -> str: ...

    # Proactive compression
    def _manage_capacity(self) -> None: ...
    def _proactive_summarize(self, turns_to_compress: int) -> None: ...
```

### Configuration

```python
@dataclass
class WorkingMemoryConfig:
    max_turns: int = 10
    max_tokens: int = 4000

    # Capacity thresholds
    proactive_threshold: float = 0.70
    aggressive_threshold: float = 0.85
    emergency_threshold: float = 0.95

    # Minimum verbatim turns (never compress)
    min_verbatim_turns: int = 3

    # Poisoning detection
    repetition_threshold: int = 3
    low_confidence_threshold: float = 0.5
```

## LLM Summarizer Injection

For production, inject an LLM-based summarizer:

```python
async def llm_summarizer(turns: List[ConversationTurn]) -> str:
    # Filter out quarantined turns
    valid_turns = [t for t in turns if not t.is_quarantined]

    text = "\n".join(
        f"User: {t.user_message}\nAssistant: {t.assistant_response}"
        for t in valid_turns
    )

    return await llm.complete(
        f"Summarize this conversation briefly, preserving key facts:\n{text}"
    )

wm = WorkingMemory()
wm.set_summarizer(llm_summarizer)
```

## Monitoring

### Health Check

```python
status = wm.get_health_status()
# {
#     "healthy": True,
#     "capacity_zone": "proactive",
#     "capacity_percentage": 0.72,
#     "quarantined_count": 1,
#     "warnings": ["Context at 70%+ capacity - proactive compression active"]
# }
```

### Alerts

Set up monitoring for:
- `capacity_zone == "emergency"` → Critical alert
- `quarantined_count > 3` → Potential hallucination loop
- `capacity_percentage > 0.85` for extended periods → Session health issue

## Integration with Memory Layers

```
┌─────────────────────────────────────────────────────────┐
│                    Friday Memory Stack                   │
├─────────────────────────────────────────────────────────┤
│  Working Memory (this document)                          │
│  ├── Compressed History ──┐                              │
│  ├── Recent Verbatim      │                              │
│  └── Attention Stack      │                              │
│                           ▼                              │
│  Short-Term Memory ◄──────┤  (Session summaries go here)│
│  ├── Session summaries    │                              │
│  └── Key facts            │                              │
│                           ▼                              │
│  Long-Term Memory ◄───────┤  (Important facts promoted)  │
│  ├── Semantic search      │                              │
│  └── Embeddings           │                              │
│                           ▼                              │
│  Knowledge Graph ◄────────┘  (Relationships extracted)   │
│  ├── Entities                                            │
│  └── Triplets                                            │
└─────────────────────────────────────────────────────────┘
```

When a session ends:
1. Working Memory compressed history → STM session summary
2. High-importance facts → LTM
3. Entities and relationships → Knowledge Graph

## Why Not External Solutions?

We evaluated `ultracontext-node` and similar tools:

| Factor | External Solution | Our Implementation |
|--------|-------------------|-------------------|
| Privacy | Data leaves system | Fully local |
| Cost | API calls per message | Zero marginal cost |
| Control | Limited customization | Full control |
| Telugu Support | Generic | Optimized for code-switching |
| Integration | Separate service | Native to Friday |

## Future Improvements

1. **Smarter Summarization**: Use Friday's own LoRA for summaries that match Boss's style
2. **Semantic Compression**: Keep semantically important content, not just recent
3. **Cross-Session Context**: Load relevant history from previous sessions automatically
4. **Contradiction Detection**: Flag when current response contradicts earlier statements
