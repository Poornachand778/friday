# Friday AI Memory Architecture
## Brain-Inspired, Voice-Controlled, Fully Autonomous

> "The measure of intelligence is the ability to change." — Albert Einstein

**Design Version**: 1.0
**Author**: Claude (for Poorna/Boss)
**Date**: January 29, 2026

---

## Table of Contents

1. [Design Philosophy](#design-philosophy)
2. [Architecture Overview](#architecture-overview)
3. [Memory Layers Deep Dive](#memory-layers-deep-dive)
4. [Voice Control System](#voice-control-system)
5. [Automated Operations](#automated-operations)
6. [Failure Modes & Self-Healing](#failure-modes--self-healing)
7. [Telugu-English Processing](#telugu-english-processing)
8. [Implementation Specifications](#implementation-specifications)
9. [Monitoring & Observability](#monitoring--observability)
10. [Edge Cases & Solutions](#edge-cases--solutions)

---

## Design Philosophy

### Core Principles

```
┌─────────────────────────────────────────────────────────────────────────┐
│                     FRIDAY MEMORY DESIGN PRINCIPLES                      │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  1. THINK WHILE TALKING (Stream Processing)                             │
│     └─ Memory operations happen IN PARALLEL with response generation    │
│     └─ No blocking - speaking never waits for memory                    │
│     └─ Background threads handle storage, retrieval runs ahead          │
│                                                                         │
│  2. FORGET TO REMEMBER (Intelligent Decay)                              │
│     └─ Active forgetting prevents noise accumulation                    │
│     └─ Important memories strengthen, irrelevant ones fade              │
│     └─ Consolidation compresses redundant memories                      │
│                                                                         │
│  3. VOICE IS FIRST CLASS (Speech-Native Design)                         │
│     └─ Every memory operation has voice trigger                         │
│     └─ Natural language memory queries                                  │
│     └─ Audio context preserved alongside text                           │
│                                                                         │
│  4. ZERO HUMAN INTERVENTION (Full Automation)                           │
│     └─ Self-monitoring, self-healing                                    │
│     └─ Automated backup and recovery                                    │
│     └─ Team only reviews logs, never touches system                     │
│                                                                         │
│  5. CONTEXT IS KING (Situational Awareness)                             │
│     └─ Room-aware memory (Writers Room vs Kitchen)                      │
│     └─ Project-scoped retrieval (Gusagusalu memories for Gusagusalu)    │
│     └─ Temporal awareness (recent vs historical)                        │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### Human Brain Mapping

| Human Brain | Friday Equivalent | Function |
|-------------|-------------------|----------|
| **Sensory Memory** | Audio Buffer | Raw input (100ms-2s) |
| **Working Memory** | Active Context | Current conversation (7±2 items) |
| **Short-Term Memory** | Session Memory | Recent conversations (hours-days) |
| **Long-Term Explicit** | LTM Facts | Declarative knowledge |
| **Long-Term Implicit** | LTM Patterns | Behavioral patterns, preferences |
| **Semantic Memory** | Knowledge Graph | Conceptual relationships |
| **Episodic Memory** | Event Timeline | Timestamped experiences |
| **Procedural Memory** | Tool Schemas | How to do things |
| **Prefrontal Cortex** | Router/Orchestrator | Decision making |
| **Hippocampus** | Memory Consolidator | Memory formation/retrieval |

---

## Architecture Overview

### High-Level System Design

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        FRIDAY MEMORY SYSTEM                                  │
│                     (Brain-Inspired Architecture)                            │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐  │
│  │   SENSORY   │───▶│   WORKING   │───▶│ SHORT-TERM  │───▶│  LONG-TERM  │  │
│  │   BUFFER    │    │   MEMORY    │    │   MEMORY    │    │   MEMORY    │  │
│  │  (100ms)    │    │  (Active)   │    │  (7 days)   │    │ (Permanent) │  │
│  └─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘  │
│        │                  │                  │                  │          │
│        │                  │                  │                  │          │
│        ▼                  ▼                  ▼                  ▼          │
│  ┌─────────────────────────────────────────────────────────────────────┐  │
│  │                     MEMORY BUS (Event-Driven)                        │  │
│  │  ├─ memory.sensory.received   ├─ memory.working.updated             │  │
│  │  ├─ memory.stm.stored         ├─ memory.ltm.consolidated            │  │
│  │  ├─ memory.decay.triggered    ├─ memory.search.completed            │  │
│  │  └─ memory.voice.command      └─ memory.profile.updated             │  │
│  └─────────────────────────────────────────────────────────────────────┘  │
│        │                  │                  │                  │          │
│        ▼                  ▼                  ▼                  ▼          │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐  │
│  │   PROFILE   │    │  KNOWLEDGE  │    │   EVENT     │    │   PATTERN   │  │
│  │   STORE     │    │   GRAPH     │    │  TIMELINE   │    │   STORE     │  │
│  │ (Identity)  │    │ (Concepts)  │    │ (Episodes)  │    │ (Behaviors) │  │
│  └─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘  │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐  │
│  │                    BACKGROUND SERVICES                               │  │
│  │  ├─ DecayDaemon (hourly)      ├─ ConsolidatorDaemon (daily)         │  │
│  │  ├─ EmbeddingWorker (async)   ├─ BackupDaemon (6-hourly)            │  │
│  │  ├─ HealthMonitor (1-min)     ├─ MetricsCollector (5-min)           │  │
│  │  └─ VoiceCommandListener      └─ SelfHealingService                 │  │
│  └─────────────────────────────────────────────────────────────────────┘  │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Data Flow: Think While Talk

```
User speaks: "Boss, remember that climax idea I mentioned last week?"

┌──────────────────────────────────────────────────────────────────────────────┐
│ T=0ms: Audio captured                                                        │
├──────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌─────────────┐                                                             │
│  │ Audio Buffer│──┐                                                          │
│  └─────────────┘  │                                                          │
│                   │                                                          │
│ T=200ms: Parallel operations start                                           │
│                   │                                                          │
│         ┌────────┴────────┬─────────────────┬─────────────────┐             │
│         ▼                 ▼                 ▼                 ▼             │
│  ┌─────────────┐   ┌─────────────┐   ┌─────────────┐   ┌─────────────┐      │
│  │ STT Process │   │ Intent Det. │   │ Memory Pre- │   │ Context     │      │
│  │ (Whisper)   │   │ (Router)    │   │ fetch Start │   │ Loading     │      │
│  └──────┬──────┘   └──────┬──────┘   └──────┬──────┘   └──────┬──────┘      │
│         │                 │                 │                 │             │
│ T=500ms: Transcript ready, memory search in progress                         │
│         │                 │                 │                 │             │
│         └────────┬────────┴─────────────────┴─────────────────┘             │
│                  ▼                                                           │
│           ┌─────────────┐                                                    │
│           │ Orchestrator│                                                    │
│           │  (Router)   │                                                    │
│           └──────┬──────┘                                                    │
│                  │                                                           │
│ T=600ms: Memory results ready, LLM starts generating                         │
│                  │                                                           │
│         ┌───────┴───────┐                                                    │
│         ▼               ▼                                                    │
│  ┌─────────────┐ ┌─────────────┐                                            │
│  │ LLM Generate│ │ Mem Storage │  ←── Parallel: Store new memory while      │
│  │ (Stream)    │ │ (Background)│      generating response                    │
│  └──────┬──────┘ └─────────────┘                                            │
│         │                                                                    │
│ T=800ms: First token spoken, memory stored                                   │
│         ▼                                                                    │
│  ┌─────────────┐                                                             │
│  │ TTS Stream  │  ←── Response streams out while background tasks continue   │
│  └─────────────┘                                                             │
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘

Total latency: 800ms to first word (target: <1 second)
Memory operations: Zero blocking
```

---

## Memory Layers Deep Dive

### Layer 1: Sensory Buffer (100ms - 2s)

**Purpose**: Raw audio capture before processing

```python
@dataclass
class SensoryFrame:
    audio_chunk: bytes              # Raw PCM audio
    timestamp: datetime             # Capture time
    sample_rate: int                # 16000 Hz
    duration_ms: int                # Chunk duration
    vad_speech_prob: float          # Voice activity score

    # Metadata for later use
    room_id: str                    # Where captured
    ambient_noise_db: float         # Background noise level
```

**Automation**:
- Auto-discards non-speech frames (VAD < 0.5)
- Circular buffer, auto-overwrites after 2s
- No human intervention ever needed

**Failure Mode**: Buffer overflow
**Self-Healing**: Drop oldest frames, log warning, continue

---

### Layer 2: Working Memory (Active Context)

**Purpose**: Current conversation state (like human's "7±2 items")

```python
@dataclass
class WorkingMemory:
    # Current conversation
    turns: List[ConversationTurn]   # Last 10 turns max
    token_count: int                # Current token usage

    # Active context
    current_room: str               # writers_room, kitchen, etc.
    current_project: Optional[str]  # gusagusalu, etc.
    active_task: Optional[str]      # What we're doing

    # Attention stack (what's being focused on)
    attention_stack: List[Dict]     # [{topic, relevance, timestamp}]

    # Ephemeral state
    pending_tool_calls: List[str]   # Tools waiting for response
    emotional_context: str          # detected mood
    language_mode: str              # en, te, mixed

    # Prefetched memories (ready for use)
    prefetched_ltm: List[Memory]    # Relevant long-term memories
    prefetched_snippets: List[str]  # Content templates

class WorkingMemoryConfig:
    max_turns: int = 10             # Conversation history limit
    max_tokens: int = 4000          # Token budget
    attention_decay_rate: float = 0.1  # How fast attention fades
    prefetch_top_k: int = 5         # LTM results to pre-load
```

**Key Feature: Attention Stack**

```python
def update_attention(self, topic: str, relevance: float):
    """
    Human-like attention management.
    Topics decay over time, strong signals boost to top.
    """
    # Decay existing items
    for item in self.attention_stack:
        item['relevance'] *= (1 - self.config.attention_decay_rate)

    # Add/update topic
    existing = next((i for i in self.attention_stack if i['topic'] == topic), None)
    if existing:
        existing['relevance'] = max(existing['relevance'], relevance)
        existing['timestamp'] = datetime.now()
    else:
        self.attention_stack.append({
            'topic': topic,
            'relevance': relevance,
            'timestamp': datetime.now()
        })

    # Prune weak attention (below threshold)
    self.attention_stack = [i for i in self.attention_stack if i['relevance'] > 0.2]

    # Sort by relevance
    self.attention_stack.sort(key=lambda x: x['relevance'], reverse=True)

    # Keep top 7 (human working memory limit)
    self.attention_stack = self.attention_stack[:7]
```

**Automation**:
- Auto-summarizes when exceeding token limit
- Auto-prefetches LTM based on current topic
- Auto-detects language mode from input

**Failure Mode**: Token overflow
**Self-Healing**: Auto-summarize oldest turns, preserve most recent 3

---

### Layer 3: Short-Term Memory (Hours to Days)

**Purpose**: Recent conversations, session continuity

```python
@dataclass
class ShortTermMemory:
    id: str                         # UUID
    session_id: str                 # Conversation session

    # Content
    summary: str                    # Compressed representation
    key_facts: List[str]            # Extracted facts
    raw_turns: List[Dict]           # Original conversation

    # Dual timestamps (Supermemory innovation)
    created_at: datetime            # When conversation happened
    event_dates: List[datetime]     # Referenced future/past events

    # Metadata
    room: str                       # Where conversation happened
    project: Optional[str]          # Related project
    topics: List[str]               # Discussed topics
    language: str                   # Dominant language

    # Scoring for decay
    access_count: int = 0           # How often retrieved
    last_accessed: datetime         # For recency scoring
    importance: float = 0.5         # User-reinforced importance

class STMConfig:
    retention_days: int = 7         # Default retention
    max_entries: int = 500          # Max STM entries
    consolidation_threshold: float = 0.3  # Below this, consolidate to LTM
```

**Storage**: SQLite with FTS5 for text search + vector embeddings

```sql
CREATE TABLE short_term_memories (
    id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL,
    summary TEXT NOT NULL,
    key_facts JSON,
    raw_turns JSON,

    -- Dual timestamps
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    event_dates JSON,

    -- Metadata
    room TEXT,
    project TEXT,
    topics JSON,
    language TEXT,

    -- Scoring
    access_count INTEGER DEFAULT 0,
    last_accessed TIMESTAMP,
    importance REAL DEFAULT 0.5,

    -- Embedding (stored separately for efficiency)
    embedding_id TEXT REFERENCES embeddings(id)
);

-- FTS5 for fast text search
CREATE VIRTUAL TABLE stm_fts USING fts5(
    id, summary, key_facts,
    content=short_term_memories,
    tokenize='porter unicode61'
);
```

**Automation**:
- Auto-extracts facts using GLM-4.7-Flash
- Auto-generates summary when session ends
- Auto-calculates importance from user signals

**Failure Mode**: Storage full
**Self-Healing**: Trigger emergency consolidation, archive oldest to LTM

---

### Layer 4: Long-Term Memory (Permanent)

**Purpose**: Consolidated knowledge, persistent facts

```python
@dataclass
class LongTermMemory:
    id: str

    # Content
    content: str                    # High-signal extracted fact
    source_summary: str             # Where this came from

    # Categorization
    memory_type: MemoryType         # fact, preference, event, pattern
    domain: str                     # film, personal, technical

    # Dual timestamps
    created_at: datetime            # When learned
    event_date: Optional[datetime]  # When event occurs (if applicable)
    valid_until: Optional[datetime] # Expiry (for time-bound facts)

    # Relationships
    related_memories: List[str]     # Connected memory IDs
    project: Optional[str]          # Project scope
    entities: List[str]             # People, places, things mentioned

    # Embedding
    embedding: List[float]          # 768-dim vector

    # Scoring
    confidence: float               # How certain (0-1)
    trust_level: int                # Source reliability (1-5)
    access_count: int               # Retrieval frequency
    last_accessed: datetime         # For decay
    importance: float               # Decayable importance score

    # Telugu support
    language: str                   # en, te, mixed
    telugu_keywords: List[str]      # For Telugu-aware search

class MemoryType(Enum):
    FACT = "fact"                   # "Boss's birthday is October 15"
    PREFERENCE = "preference"       # "Boss prefers direct communication"
    EVENT = "event"                 # "Gusagusalu deadline is March"
    PATTERN = "pattern"             # "Boss usually writes in mornings"
    DECISION = "decision"           # "We decided to use flashback structure"
    RELATIONSHIP = "relationship"   # "Ravi is the protagonist's father"
```

**Storage**: PostgreSQL with pgvector

```sql
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE long_term_memories (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    content TEXT NOT NULL,
    source_summary TEXT,

    memory_type TEXT NOT NULL,
    domain TEXT,

    -- Dual timestamps
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    event_date TIMESTAMP,
    valid_until TIMESTAMP,

    -- Relationships
    related_memories UUID[],
    project TEXT,
    entities TEXT[],

    -- Vector embedding
    embedding vector(768),

    -- Scoring
    confidence REAL DEFAULT 0.8,
    trust_level INTEGER DEFAULT 3,
    access_count INTEGER DEFAULT 0,
    last_accessed TIMESTAMP,
    importance REAL DEFAULT 0.5,

    -- Telugu
    language TEXT DEFAULT 'en',
    telugu_keywords TEXT[]
);

-- Vector similarity search index
CREATE INDEX ltm_embedding_idx ON long_term_memories
USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);

-- Time-based queries
CREATE INDEX ltm_event_date_idx ON long_term_memories(event_date);
CREATE INDEX ltm_created_at_idx ON long_term_memories(created_at);

-- Entity search
CREATE INDEX ltm_entities_idx ON long_term_memories USING gin(entities);
```

**Automation**:
- Auto-consolidated from STM when threshold reached
- Auto-decayed by DecayDaemon
- Auto-linked to related memories

**Failure Mode**: Embedding generation fails
**Self-Healing**: Store without embedding, queue for later processing

---

### Layer 5: Profile Store (Identity)

**Purpose**: Persistent facts about Boss that NEVER decay

```python
@dataclass
class UserProfile:
    """
    The unchanging core identity.
    Updated only through explicit voice commands or clear contradictions.
    """

    # Static identity (almost never changes)
    static: Dict[str, Any] = field(default_factory=lambda: {
        "name": "Poorna",
        "role": "Telugu screenwriter",
        "languages": ["Telugu", "English"],
        "address_as": "Boss",
        "communication_style": "Direct, no flattery, concise",
    })

    # Dynamic state (changes with context)
    dynamic: Dict[str, Any] = field(default_factory=lambda: {
        "current_project": None,
        "current_room": "writers_room",
        "recent_mood": "neutral",
        "active_deadlines": [],
        "last_interaction": None,
    })

    # Learned preferences (updated from patterns)
    preferences: Dict[str, Any] = field(default_factory=lambda: {
        "response_length": "concise",
        "telugu_usage": "natural, for emotions",
        "feedback_style": "direct with opinions",
        "working_hours": "flexible",
    })

    # Relationships (people Friday knows about)
    relationships: Dict[str, Dict] = field(default_factory=dict)
    # Example: {"Ravi": {"relation": "character", "project": "gusagusalu"}}

    # Projects (all known projects)
    projects: Dict[str, Dict] = field(default_factory=dict)
    # Example: {"gusagusalu": {"status": "active", "deadline": "2026-03"}}

class ProfileManager:
    def update_static(self, key: str, value: Any, voice_confirmed: bool = False):
        """
        Static facts require voice confirmation for changes.
        Prevents accidental identity drift.
        """
        if not voice_confirmed:
            raise VoiceConfirmationRequired(
                f"Changing static profile '{key}' requires voice confirmation. "
                f"Say: 'Friday, update my profile: {key} is {value}'"
            )

        self.profile.static[key] = value
        self._audit_log("static_update", key, value)

    def update_dynamic(self, key: str, value: Any):
        """Dynamic state auto-updates without confirmation."""
        self.profile.dynamic[key] = value
        self.profile.dynamic["last_updated"] = datetime.now()

    def learn_preference(self, key: str, value: Any, confidence: float):
        """
        Preferences learned from patterns.
        High-confidence patterns update preferences.
        """
        if confidence > 0.8:  # High confidence threshold
            self.profile.preferences[key] = value
            self._audit_log("preference_learned", key, value, confidence)
```

**Storage**: JSON file + version control (git-like history)

```
memory/
├── profile/
│   ├── current.json              # Active profile
│   ├── history/                  # Version history
│   │   ├── 2026-01-29_001.json
│   │   ├── 2026-01-28_001.json
│   │   └── ...
│   └── audit.log                 # All changes logged
```

**Automation**:
- Auto-backs up before any change
- Auto-learns preferences from patterns (if confidence > 0.8)
- Auto-updates dynamic state

**Failure Mode**: Profile corruption
**Self-Healing**: Restore from last known good version in history/

---

### Layer 6: Knowledge Graph (Conceptual Relationships)

**Purpose**: Understand relationships between concepts

```python
@dataclass
class KnowledgeNode:
    id: str
    name: str                       # "Ravi", "climax scene", "father"
    node_type: str                  # character, scene, concept, project
    attributes: Dict[str, Any]      # Flexible attributes

@dataclass
class KnowledgeEdge:
    source_id: str
    target_id: str
    relation: str                   # "father_of", "part_of", "related_to"
    weight: float                   # Relationship strength
    context: Optional[str]          # When/why this relationship exists

class KnowledgeGraph:
    """
    Graph database for conceptual relationships.
    Enables queries like: "Who is Ravi's father?" or "All scenes in Act 2"
    """

    def query(self, cypher: str) -> List[Dict]:
        """Query using Cypher-like syntax."""
        pass

    def find_path(self, from_node: str, to_node: str) -> List[KnowledgeEdge]:
        """Find relationship path between concepts."""
        pass

    def get_neighborhood(self, node_id: str, depth: int = 2) -> Dict:
        """Get related concepts within N hops."""
        pass
```

**Storage**: Neo4j or NetworkX (for smaller scale)

**Example Queries**:
```cypher
// Find all characters in Gusagusalu
MATCH (c:Character)-[:APPEARS_IN]->(p:Project {name: 'gusagusalu'})
RETURN c.name, c.role

// Find relationship between Ravi and his father
MATCH path = (a:Character {name: 'Ravi'})-[*1..3]-(b:Character {name: 'father'})
RETURN path

// All scenes involving emotional confrontation
MATCH (s:Scene)-[:HAS_THEME]->(t:Theme {name: 'confrontation'})
WHERE s.emotion_level > 0.7
RETURN s
```

**Automation**:
- Auto-extracts entities from conversations
- Auto-creates relationships from context
- Auto-prunes orphan nodes

---

### Layer 7: Event Timeline (Episodic Memory)

**Purpose**: Time-ordered events for temporal queries

```python
@dataclass
class TimelineEvent:
    id: str

    # Core event data
    description: str                # What happened
    event_type: str                 # conversation, deadline, milestone, reminder

    # Temporal data (Supermemory-inspired dual timestamps)
    document_time: datetime         # When we learned about it
    event_time: datetime            # When event actually occurs/occurred
    duration: Optional[timedelta]   # How long (for spans)
    recurrence: Optional[str]       # "daily", "weekly", etc.

    # Context
    project: Optional[str]
    room: Optional[str]
    related_memories: List[str]

    # Status
    status: str                     # upcoming, past, cancelled
    importance: float

class Timeline:
    def query_upcoming(self, days: int = 7) -> List[TimelineEvent]:
        """What's happening in the next N days?"""
        now = datetime.now()
        cutoff = now + timedelta(days=days)
        return self.events.filter(
            event_time__gte=now,
            event_time__lte=cutoff,
            status='upcoming'
        ).order_by('event_time')

    def query_range(self, start: datetime, end: datetime) -> List[TimelineEvent]:
        """Events within a time range."""
        pass

    def query_relative(self, query: str) -> List[TimelineEvent]:
        """
        Natural language temporal queries.

        Examples:
        - "last week" → events with document_time in past 7 days
        - "next month" → events with event_time in next 30 days
        - "when we discussed the climax" → semantic + temporal search
        """
        pass
```

**Temporal Query Examples**:

| Query | Interpretation |
|-------|----------------|
| "What did I say last week?" | document_time in past 7 days |
| "What's coming up in March?" | event_time in March 2026 |
| "When did we decide on the flashback?" | Search decisions, return document_time |
| "Deadlines this month" | event_type=deadline, event_time in current month |

**Automation**:
- Auto-extracts event dates from conversations
- Auto-updates status (upcoming → past)
- Auto-sends reminders for important events

---

### Layer 8: Pattern Store (Implicit Learning)

**Purpose**: Behavioral patterns and habits

```python
@dataclass
class Pattern:
    id: str

    # Pattern description
    pattern_type: str               # time_preference, topic_interest, language_switch
    description: str                # Human-readable description

    # Statistical data
    occurrences: int                # How many times observed
    confidence: float               # Statistical confidence
    last_observed: datetime

    # Context
    conditions: Dict[str, Any]      # When this pattern applies

    # Examples
    examples: List[str]             # Specific instances

class PatternType(Enum):
    TIME_PREFERENCE = "time_preference"      # "Boss writes scripts in the morning"
    TOPIC_INTEREST = "topic_interest"        # "Boss often discusses character arcs"
    LANGUAGE_SWITCH = "language_switch"      # "Boss uses Telugu for emotional topics"
    COMMUNICATION = "communication"          # "Boss prefers short responses"
    WORKFLOW = "workflow"                    # "Boss reviews scenes in order"
    EMOTIONAL = "emotional"                  # "Boss gets frustrated with repetition"

class PatternDetector:
    """
    Background service that analyzes interactions to detect patterns.
    """

    def analyze_session(self, session: List[ConversationTurn]) -> List[Pattern]:
        """Extract patterns from a conversation session."""
        patterns = []

        # Time analysis
        hour = session[0].timestamp.hour
        if 5 <= hour <= 9:
            patterns.append(self._update_pattern(
                "morning_work",
                "Boss often works in the morning"
            ))

        # Language analysis
        telugu_ratio = self._calculate_telugu_ratio(session)
        if telugu_ratio > 0.5:
            patterns.append(self._update_pattern(
                "telugu_preference",
                f"High Telugu usage in this session ({telugu_ratio:.0%})"
            ))

        # Topic analysis
        topics = self._extract_topics(session)
        for topic, count in topics.items():
            if count >= 3:
                patterns.append(self._update_pattern(
                    f"interest_{topic}",
                    f"Repeated interest in {topic}"
                ))

        return patterns
```

**Automation**:
- Runs after every session
- Auto-updates preferences when pattern confidence > 0.8
- Auto-prunes weak patterns (confidence < 0.3)

---

## Voice Control System

### Voice Command Registry

```python
class VoiceCommandRegistry:
    """
    All memory operations accessible via voice.
    Natural language → structured command.
    """

    commands = {
        # Memory Search
        "remember": {
            "patterns": [
                "do you remember {query}",
                "what did I say about {query}",
                "{query} గురించి ఏమి చెప్పాను",
                "recall {query}",
            ],
            "handler": "memory_search",
            "params": ["query"]
        },

        # Memory Store
        "store": {
            "patterns": [
                "remember this: {content}",
                "save this: {content}",
                "ఇది గుర్తుంచుకో: {content}",
                "note that {content}",
            ],
            "handler": "memory_store",
            "params": ["content"]
        },

        # Profile Update
        "profile_update": {
            "patterns": [
                "update my profile: {key} is {value}",
                "I prefer {preference}",
                "my {attribute} is {value}",
            ],
            "handler": "profile_update",
            "params": ["key", "value"]
        },

        # Memory Delete
        "forget": {
            "patterns": [
                "forget about {topic}",
                "delete memories about {topic}",
                "{topic} గురించి మర్చిపో",
            ],
            "handler": "memory_delete",
            "params": ["topic"],
            "requires_confirmation": True
        },

        # Temporal Query
        "timeline": {
            "patterns": [
                "what happened {time_ref}",
                "what's coming up {time_ref}",
                "{time_ref} ఏం జరిగింది",
                "upcoming deadlines",
            ],
            "handler": "timeline_query",
            "params": ["time_ref"]
        },

        # Project Context
        "project_switch": {
            "patterns": [
                "switch to {project}",
                "let's work on {project}",
                "{project} మీద పని చేద్దాం",
            ],
            "handler": "set_project_context",
            "params": ["project"]
        },

        # Memory Status
        "memory_status": {
            "patterns": [
                "memory status",
                "how much do you remember",
                "memory health",
            ],
            "handler": "get_memory_stats"
        },

        # Importance Marking
        "mark_important": {
            "patterns": [
                "this is important",
                "remember this well",
                "ఇది ముఖ్యం",
                "don't forget this",
            ],
            "handler": "boost_importance",
            "params": []  # Applies to current context
        }
    }
```

### Voice Command Examples

| Voice Input | Parsed Command | Action |
|------------|----------------|--------|
| "Do you remember what I said about the climax?" | `memory_search("climax")` | Search LTM + STM for climax-related memories |
| "Remember this: Ravi should hesitate before the confrontation" | `memory_store("Ravi should hesitate...")` | Store as high-importance memory |
| "ఇది ముఖ్యం, మర్చిపోకు" | `boost_importance()` | Mark last exchange as important |
| "What happened last Tuesday?" | `timeline_query("last Tuesday")` | Query timeline for that date |
| "Forget about the old ending" | `memory_delete("old ending")` | Delete with confirmation |
| "Switch to Gusagusalu" | `set_project_context("gusagusalu")` | Set project context for retrieval |

### Confirmation Flow for Destructive Actions

```
User: "Forget about the kitchen project"

Friday: "Boss, I'll delete all memories related to 'kitchen project'.
         This includes 12 memories. Confirm by saying 'Yes, delete them'
         or cancel by saying 'No, keep them'."

User: "Yes, delete them"

Friday: "Done. Deleted 12 memories about kitchen project.
         Recoverable for 7 days if you change your mind."
```

---

## Automated Operations

### Background Daemons

```python
class MemoryDaemonScheduler:
    """
    All background operations that require zero human intervention.
    """

    daemons = {
        "decay": {
            "schedule": "0 * * * *",          # Every hour
            "handler": DecayDaemon,
            "description": "Apply decay to aging memories"
        },
        "consolidation": {
            "schedule": "0 3 * * *",          # Daily at 3 AM
            "handler": ConsolidationDaemon,
            "description": "Consolidate STM → LTM"
        },
        "backup": {
            "schedule": "0 */6 * * *",        # Every 6 hours
            "handler": BackupDaemon,
            "description": "Backup all memory stores"
        },
        "embedding": {
            "schedule": "continuous",          # Always running
            "handler": EmbeddingWorker,
            "description": "Generate embeddings for new memories"
        },
        "health": {
            "schedule": "* * * * *",           # Every minute
            "handler": HealthMonitor,
            "description": "Check system health"
        },
        "pattern": {
            "schedule": "0 4 * * *",           # Daily at 4 AM
            "handler": PatternAnalyzer,
            "description": "Analyze patterns from recent data"
        },
        "cleanup": {
            "schedule": "0 5 * * 0",           # Weekly on Sunday 5 AM
            "handler": CleanupDaemon,
            "description": "Remove truly expired memories"
        }
    }
```

### Decay Algorithm (Detailed)

```python
class DecayDaemon:
    """
    Intelligent memory decay inspired by Supermemory.

    Philosophy: "Forgetting is a feature, not a bug."
    """

    def __init__(self, config: DecayConfig):
        self.config = config

    def run(self):
        """Execute decay pass on all memories."""
        memories = self.get_all_decayable_memories()

        for memory in memories:
            score = self.calculate_score(memory)

            if score < self.config.archive_threshold:
                self.archive_memory(memory)
            elif score < self.config.decay_threshold:
                self.reduce_importance(memory, score)
            else:
                # Memory is healthy, no action needed
                pass

        self.log_decay_stats()

    def calculate_score(self, memory: Memory) -> float:
        """
        Score = weighted combination of multiple factors.
        Higher score = more likely to be retained.
        """
        now = datetime.now()

        # 1. Recency factor (exponential decay)
        days_since_access = (now - memory.last_accessed).days
        recency = math.exp(-self.config.recency_decay_rate * days_since_access)
        # Example: decay_rate=0.1, 10 days old → recency = 0.37

        # 2. Frequency factor (log scale to prevent runaway)
        frequency = math.log(memory.access_count + 1) / math.log(100)
        # Caps at 1.0 when access_count = 99

        # 3. Importance factor (user-reinforced)
        importance = memory.importance

        # 4. Event relevance (upcoming events are protected)
        event_relevance = 0.0
        if memory.event_date:
            days_until = (memory.event_date - now).days
            if 0 <= days_until <= 30:
                # Upcoming events in next month get boost
                event_relevance = 1.0 - (days_until / 30)
            elif -7 <= days_until < 0:
                # Recent past events (last week) still relevant
                event_relevance = 0.3

        # 5. Profile reference (memories that reference static profile are protected)
        profile_relevance = 0.0
        if self.references_profile(memory):
            profile_relevance = 0.3

        # 6. Type bonus (certain types decay slower)
        type_bonus = self.config.type_weights.get(memory.memory_type, 0.0)
        # preferences: 0.2, decisions: 0.15, facts: 0.1, events: 0.0

        # Weighted combination
        score = (
            self.config.weights['recency'] * recency +
            self.config.weights['frequency'] * frequency +
            self.config.weights['importance'] * importance +
            self.config.weights['event'] * event_relevance +
            self.config.weights['profile'] * profile_relevance +
            type_bonus
        )

        return min(score, 1.0)  # Cap at 1.0

@dataclass
class DecayConfig:
    # Thresholds
    decay_threshold: float = 0.4    # Below this, reduce importance
    archive_threshold: float = 0.2  # Below this, archive
    delete_threshold: float = 0.05  # Below this after archive, delete

    # Decay rate
    recency_decay_rate: float = 0.1  # Per day

    # Weights (must sum to ~1.0)
    weights: Dict[str, float] = field(default_factory=lambda: {
        'recency': 0.30,
        'frequency': 0.15,
        'importance': 0.30,
        'event': 0.15,
        'profile': 0.10
    })

    # Type-specific bonuses
    type_weights: Dict[str, float] = field(default_factory=lambda: {
        'preference': 0.2,
        'decision': 0.15,
        'fact': 0.1,
        'pattern': 0.1,
        'event': 0.0
    })
```

### Consolidation Algorithm

```python
class ConsolidationDaemon:
    """
    Consolidates STM → LTM.

    Like human sleep: compress, organize, strengthen important memories.
    """

    def run(self):
        """Nightly consolidation pass."""

        # 1. Get STM entries ready for consolidation
        stm_entries = self.get_consolidation_candidates()

        # 2. Group by topic/project
        grouped = self.group_by_similarity(stm_entries)

        # 3. For each group, create consolidated LTM
        for group in grouped:
            if len(group) == 1:
                # Single entry, promote directly
                self.promote_to_ltm(group[0])
            else:
                # Multiple entries, consolidate
                consolidated = self.merge_memories(group)
                self.store_in_ltm(consolidated)
                self.archive_stm_entries(group)

        # 4. Update knowledge graph with new connections
        self.update_knowledge_graph()

        # 5. Re-calculate patterns
        self.refresh_patterns()

    def merge_memories(self, memories: List[ShortTermMemory]) -> LongTermMemory:
        """
        Merge multiple related STM entries into one LTM entry.

        Example:
        - STM1: "Boss mentioned Ravi should be more emotional"
        - STM2: "Boss said the confrontation needs more punch"
        - STM3: "Boss wants Ravi to hesitate before speaking"

        Merged LTM: "Boss wants Ravi's confrontation scene to be more
                    emotional with hesitation and punch"
        """
        # Use GLM-4.7-Flash for intelligent merging
        merge_prompt = f"""
        Merge these related memories into one coherent fact:

        {chr(10).join(m.summary for m in memories)}

        Rules:
        - Preserve all unique information
        - Remove redundancy
        - Keep the Boss's intent clear
        - Maintain temporal references

        Output: Single merged memory (1-2 sentences)
        """

        merged_content = self.glm_router.generate(merge_prompt)

        return LongTermMemory(
            content=merged_content,
            source_summary=f"Consolidated from {len(memories)} STM entries",
            memory_type=self._determine_type(memories),
            created_at=min(m.created_at for m in memories),
            event_dates=[d for m in memories for d in (m.event_dates or [])],
            importance=max(m.importance for m in memories),
            project=memories[0].project,
            topics=list(set(t for m in memories for t in m.topics)),
        )
```

### Self-Healing System

```python
class SelfHealingService:
    """
    Automatic recovery from failures.
    Goal: Team only checks logs, never intervenes.
    """

    def __init__(self):
        self.failure_handlers = {
            "database_connection": self.handle_db_failure,
            "embedding_model": self.handle_embedding_failure,
            "storage_full": self.handle_storage_failure,
            "memory_corruption": self.handle_corruption,
            "backup_failure": self.handle_backup_failure,
            "daemon_crash": self.handle_daemon_crash,
        }

    def handle_db_failure(self, error: Exception):
        """Database connection lost."""
        self.log("WARN", "Database connection lost, attempting recovery")

        # 1. Wait and retry
        for attempt in range(3):
            time.sleep(5 * (attempt + 1))  # Backoff: 5s, 10s, 15s
            if self.test_db_connection():
                self.log("INFO", f"Database recovered after {attempt+1} attempts")
                return True

        # 2. Switch to backup database
        if self.switch_to_backup_db():
            self.log("WARN", "Switched to backup database")
            self.alert("Database failover occurred - check primary")
            return True

        # 3. Enter degraded mode (memory-only operation)
        self.enter_degraded_mode()
        self.alert("CRITICAL: Running in degraded mode, no persistence")
        return False

    def handle_embedding_failure(self, error: Exception):
        """Embedding model crashed or unavailable."""
        self.log("WARN", "Embedding model unavailable")

        # 1. Try to reload model
        if self.reload_embedding_model():
            return True

        # 2. Queue items for later embedding
        self.enable_embedding_queue()

        # 3. Fall back to keyword search
        self.enable_keyword_fallback()
        self.log("INFO", "Switched to keyword search fallback")
        return True  # System continues working

    def handle_storage_failure(self, error: Exception):
        """Storage approaching or at capacity."""
        self.log("WARN", "Storage critical")

        # 1. Emergency decay pass (more aggressive)
        self.run_emergency_decay(threshold=0.5)  # More aggressive

        # 2. Archive old audio files
        self.archive_old_audio()

        # 3. Compress backups
        self.compress_old_backups()

        # 4. Alert if still critical
        if self.get_storage_percentage() > 90:
            self.alert("Storage still critical after cleanup")

        return True

    def handle_corruption(self, error: Exception):
        """Memory data corruption detected."""
        self.log("ERROR", f"Corruption detected: {error}")

        # 1. Identify corrupted entries
        corrupted_ids = self.scan_for_corruption()

        # 2. Attempt repair from backup
        for memory_id in corrupted_ids:
            if self.restore_from_backup(memory_id):
                self.log("INFO", f"Restored {memory_id} from backup")
            else:
                self.quarantine_memory(memory_id)
                self.log("WARN", f"Quarantined unrepairable memory {memory_id}")

        # 3. Run integrity check
        self.run_integrity_check()

        return True
```

---

## Failure Modes & Self-Healing

### Comprehensive Failure Matrix

| Failure Mode | Detection | Automatic Response | Alert Level |
|--------------|-----------|-------------------|-------------|
| **Database connection lost** | Connection timeout | Retry → Backup DB → Degraded mode | CRITICAL |
| **Embedding model OOM** | CUDA OOM error | Reduce batch size → CPU fallback | WARN |
| **Storage full** | >90% capacity | Emergency decay → Archive old → Alert | WARN |
| **Memory corruption** | Checksum mismatch | Restore from backup → Quarantine | ERROR |
| **Backup failure** | Job timeout | Retry → Alternative location → Alert | WARN |
| **Daemon crash** | Process monitor | Auto-restart with backoff | ERROR |
| **Slow queries** | >2s latency | Index rebuild → Cache warming | INFO |
| **Token overflow** | Context > limit | Auto-summarize → Trim old | INFO |
| **Profile inconsistency** | Version mismatch | Restore last valid → Alert | WARN |
| **Embedding drift** | Similarity degradation | Reindex all embeddings | INFO |

### Recovery Priorities

```python
class RecoveryPriority(Enum):
    """
    What to recover first in a disaster.
    """
    CRITICAL = 1   # Profile, active working memory
    HIGH = 2       # Recent LTM (7 days)
    MEDIUM = 3     # Knowledge graph, patterns
    LOW = 4        # Old LTM, archived STM
    OPTIONAL = 5   # Logs, audio files
```

### Graceful Degradation Levels

```
Level 0: FULL OPERATION
├─ All systems nominal
├─ Full semantic search
├─ All daemons running
└─ Real-time embedding

Level 1: REDUCED (Embedding unavailable)
├─ Keyword-based search
├─ New memories stored without embeddings
├─ Embedding queue growing
└─ Alert: "Embedding model down"

Level 2: LIMITED (Database degraded)
├─ Read from backup DB
├─ Write to memory queue
├─ No new LTM storage
└─ Alert: "Database failover active"

Level 3: EMERGENCY (Storage critical)
├─ Working memory only
├─ No new storage
├─ Read-only from cache
└─ Alert: "Memory system in emergency mode"

Level 4: OFFLINE (Total failure)
├─ Static responses only
├─ No memory operations
├─ Profile loaded from backup
└─ Alert: "Memory system offline"
```

---

## Telugu-English Processing

### Multilingual Memory Pipeline

```python
class TeluguEnglishProcessor:
    """
    Handle Telugu-English code-switching in memory.
    """

    def process_input(self, text: str) -> ProcessedText:
        """
        Analyze and tag language segments.
        """
        segments = []
        current_lang = None
        current_text = ""

        for char in text:
            char_lang = self.detect_char_language(char)

            if char_lang != current_lang and current_text:
                segments.append(TextSegment(
                    text=current_text,
                    language=current_lang
                ))
                current_text = ""

            current_text += char
            current_lang = char_lang

        if current_text:
            segments.append(TextSegment(current_text, current_lang))

        return ProcessedText(
            original=text,
            segments=segments,
            dominant_language=self.get_dominant(segments),
            telugu_density=self.calculate_telugu_density(text),
            code_switch_count=self.count_switches(segments)
        )

    def detect_char_language(self, char: str) -> str:
        """Detect language of a single character."""
        if '\u0c00' <= char <= '\u0c7f':
            return 'te'
        elif char.isalpha():
            return 'en'
        else:
            return 'neutral'  # Punctuation, numbers

    def calculate_telugu_density(self, text: str) -> float:
        """Calculate Telugu character ratio."""
        if not text:
            return 0.0
        telugu_chars = sum(1 for c in text if '\u0c00' <= c <= '\u0c7f')
        return telugu_chars / len(text)

@dataclass
class ProcessedText:
    original: str
    segments: List[TextSegment]
    dominant_language: str
    telugu_density: float
    code_switch_count: int
```

### Telugu-Aware Embeddings

```python
class TeluguAwareEmbedder:
    """
    Generate embeddings that understand Telugu-English mixing.
    """

    def __init__(self):
        # Use multilingual model that supports Telugu
        self.model = SentenceTransformer('paraphrase-multilingual-mpnet-base-v2')

        # Telugu stopwords to filter
        self.telugu_stopwords = {'మరియు', 'ఒక', 'ఈ', 'ఆ', 'కు', 'లో', 'తో'}

    def embed(self, text: str) -> List[float]:
        """Generate embedding for mixed text."""
        # 1. Normalize Telugu text
        normalized = self.normalize_telugu(text)

        # 2. Generate embedding
        embedding = self.model.encode(normalized)

        return embedding.tolist()

    def normalize_telugu(self, text: str) -> str:
        """
        Normalize Telugu text for better embedding.
        - Standardize vowel signs
        - Handle common variations
        """
        # Telugu-specific normalizations
        replacements = {
            'ై': 'ై',  # Standardize vowel signs
            'ౌ': 'ౌ',
        }

        for old, new in replacements.items():
            text = text.replace(old, new)

        return text

    def extract_telugu_keywords(self, text: str) -> List[str]:
        """
        Extract Telugu keywords for supplementary search.
        """
        processor = TeluguEnglishProcessor()
        processed = processor.process_input(text)

        keywords = []
        for segment in processed.segments:
            if segment.language == 'te':
                # Split Telugu text into words
                words = segment.text.split()
                # Filter stopwords
                keywords.extend(
                    w for w in words
                    if w not in self.telugu_stopwords and len(w) > 2
                )

        return keywords
```

### Bilingual Search Strategy

```python
class BilingualMemorySearch:
    """
    Search that works for Telugu, English, and mixed queries.
    """

    def search(
        self,
        query: str,
        top_k: int = 5,
        project: Optional[str] = None
    ) -> List[Memory]:
        """
        Multi-strategy search for best results.
        """
        processor = TeluguEnglishProcessor()
        processed = processor.process_input(query)

        results = []

        # 1. Semantic search (always)
        semantic_results = self.vector_search(query, top_k * 2)
        results.extend(semantic_results)

        # 2. Telugu keyword search (if Telugu present)
        if processed.telugu_density > 0.1:
            keywords = self.embedder.extract_telugu_keywords(query)
            if keywords:
                keyword_results = self.keyword_search(keywords, top_k)
                results.extend(keyword_results)

        # 3. Temporal search (if time reference detected)
        time_ref = self.detect_time_reference(query)
        if time_ref:
            temporal_results = self.temporal_search(time_ref, top_k)
            results.extend(temporal_results)

        # 4. Entity search (if named entity detected)
        entities = self.extract_entities(query)
        if entities:
            entity_results = self.entity_search(entities, top_k)
            results.extend(entity_results)

        # 5. Deduplicate and rank
        unique_results = self.deduplicate(results)
        ranked = self.rank_results(unique_results, query)

        return ranked[:top_k]
```

---

## Implementation Specifications

### Directory Structure

```
memory/
├── __init__.py
├── config.py                      # Memory configuration
├── manager.py                     # Main memory manager
├──
├── layers/
│   ├── __init__.py
│   ├── sensory.py                 # Sensory buffer
│   ├── working.py                 # Working memory
│   ├── short_term.py              # STM layer
│   ├── long_term.py               # LTM layer
│   ├── profile.py                 # Profile store
│   ├── knowledge_graph.py         # Knowledge graph
│   ├── timeline.py                # Event timeline
│   └── patterns.py                # Pattern store
│
├── operations/
│   ├── __init__.py
│   ├── search.py                  # Memory search
│   ├── store.py                   # Memory storage
│   ├── decay.py                   # Decay algorithm
│   ├── consolidate.py             # Consolidation
│   └── extract.py                 # Fact extraction
│
├── daemons/
│   ├── __init__.py
│   ├── scheduler.py               # Daemon scheduler
│   ├── decay_daemon.py            # Decay daemon
│   ├── consolidation_daemon.py    # Consolidation daemon
│   ├── backup_daemon.py           # Backup daemon
│   ├── health_monitor.py          # Health monitoring
│   └── pattern_analyzer.py        # Pattern analysis
│
├── voice/
│   ├── __init__.py
│   ├── command_registry.py        # Voice commands
│   ├── command_parser.py          # Parse voice to command
│   └── command_executor.py        # Execute commands
│
├── telugu/
│   ├── __init__.py
│   ├── processor.py               # Telugu-English processing
│   ├── embedder.py                # Telugu-aware embeddings
│   └── keywords.py                # Telugu keyword extraction
│
├── storage/
│   ├── __init__.py
│   ├── sqlite_store.py            # SQLite for STM
│   ├── postgres_store.py          # PostgreSQL for LTM
│   ├── redis_cache.py             # Redis for working memory
│   └── file_store.py              # File-based backup
│
├── healing/
│   ├── __init__.py
│   ├── self_healing.py            # Self-healing service
│   ├── degradation.py             # Graceful degradation
│   └── recovery.py                # Recovery procedures
│
├── monitoring/
│   ├── __init__.py
│   ├── metrics.py                 # Memory metrics
│   ├── logging.py                 # Structured logging
│   └── alerts.py                  # Alert system
│
└── data/
    ├── profile/
    │   ├── current.json           # Active profile
    │   └── history/               # Profile versions
    ├── backups/                   # Memory backups
    └── logs/                      # Operation logs
```

### Database Schema (Complete)

```sql
-- ============================================
-- FRIDAY MEMORY DATABASE SCHEMA
-- ============================================

-- Short-Term Memory
CREATE TABLE short_term_memories (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    session_id UUID NOT NULL,
    summary TEXT NOT NULL,
    key_facts JSONB,
    raw_turns JSONB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    event_dates JSONB,
    room TEXT,
    project TEXT,
    topics TEXT[],
    language TEXT,
    access_count INTEGER DEFAULT 0,
    last_accessed TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    importance REAL DEFAULT 0.5,
    status TEXT DEFAULT 'active',  -- active, archived, consolidated
    CONSTRAINT valid_status CHECK (status IN ('active', 'archived', 'consolidated'))
);

-- Long-Term Memory
CREATE TABLE long_term_memories (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    content TEXT NOT NULL,
    source_summary TEXT,
    memory_type TEXT NOT NULL,
    domain TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    event_date TIMESTAMP,
    valid_until TIMESTAMP,
    related_memories UUID[],
    project TEXT,
    entities TEXT[],
    embedding vector(768),
    confidence REAL DEFAULT 0.8,
    trust_level INTEGER DEFAULT 3,
    access_count INTEGER DEFAULT 0,
    last_accessed TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    importance REAL DEFAULT 0.5,
    language TEXT DEFAULT 'en',
    telugu_keywords TEXT[],
    source_stm_ids UUID[],  -- Which STM entries this came from
    CONSTRAINT valid_type CHECK (memory_type IN ('fact', 'preference', 'event', 'pattern', 'decision', 'relationship'))
);

-- Knowledge Graph Nodes
CREATE TABLE knowledge_nodes (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name TEXT NOT NULL,
    node_type TEXT NOT NULL,
    attributes JSONB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    project TEXT,
    CONSTRAINT valid_node_type CHECK (node_type IN ('character', 'scene', 'concept', 'project', 'person', 'location'))
);

-- Knowledge Graph Edges
CREATE TABLE knowledge_edges (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    source_id UUID REFERENCES knowledge_nodes(id) ON DELETE CASCADE,
    target_id UUID REFERENCES knowledge_nodes(id) ON DELETE CASCADE,
    relation TEXT NOT NULL,
    weight REAL DEFAULT 1.0,
    context TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Event Timeline
CREATE TABLE timeline_events (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    description TEXT NOT NULL,
    event_type TEXT NOT NULL,
    document_time TIMESTAMP NOT NULL,
    event_time TIMESTAMP NOT NULL,
    duration INTERVAL,
    recurrence TEXT,
    project TEXT,
    room TEXT,
    related_memories UUID[],
    status TEXT DEFAULT 'upcoming',
    importance REAL DEFAULT 0.5,
    CONSTRAINT valid_event_type CHECK (event_type IN ('conversation', 'deadline', 'milestone', 'reminder', 'decision')),
    CONSTRAINT valid_status CHECK (status IN ('upcoming', 'past', 'cancelled'))
);

-- Pattern Store
CREATE TABLE patterns (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    pattern_type TEXT NOT NULL,
    description TEXT NOT NULL,
    occurrences INTEGER DEFAULT 1,
    confidence REAL DEFAULT 0.5,
    last_observed TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    conditions JSONB,
    examples TEXT[],
    CONSTRAINT valid_pattern_type CHECK (pattern_type IN ('time_preference', 'topic_interest', 'language_switch', 'communication', 'workflow', 'emotional'))
);

-- Memory Archives (for deleted/consolidated memories)
CREATE TABLE memory_archives (
    id UUID PRIMARY KEY,
    original_table TEXT NOT NULL,
    data JSONB NOT NULL,
    archived_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    reason TEXT,
    recoverable_until TIMESTAMP  -- 7 days by default
);

-- Audit Log (all changes tracked)
CREATE TABLE memory_audit_log (
    id BIGSERIAL PRIMARY KEY,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    operation TEXT NOT NULL,  -- create, update, delete, access
    table_name TEXT NOT NULL,
    record_id UUID,
    old_data JSONB,
    new_data JSONB,
    triggered_by TEXT  -- daemon, voice_command, auto
);

-- Indexes
CREATE INDEX idx_stm_session ON short_term_memories(session_id);
CREATE INDEX idx_stm_project ON short_term_memories(project);
CREATE INDEX idx_stm_created ON short_term_memories(created_at);
CREATE INDEX idx_stm_importance ON short_term_memories(importance);

CREATE INDEX idx_ltm_embedding ON long_term_memories USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);
CREATE INDEX idx_ltm_event_date ON long_term_memories(event_date);
CREATE INDEX idx_ltm_project ON long_term_memories(project);
CREATE INDEX idx_ltm_type ON long_term_memories(memory_type);
CREATE INDEX idx_ltm_entities ON long_term_memories USING gin(entities);

CREATE INDEX idx_timeline_event_time ON timeline_events(event_time);
CREATE INDEX idx_timeline_status ON timeline_events(status);

CREATE INDEX idx_audit_timestamp ON memory_audit_log(timestamp);
CREATE INDEX idx_audit_table ON memory_audit_log(table_name);
```

### Configuration File

```yaml
# config/memory_config.yaml

memory:
  # Layer configurations
  sensory:
    buffer_size_ms: 2000
    sample_rate: 16000
    vad_threshold: 0.5

  working:
    max_turns: 10
    max_tokens: 4000
    attention_decay_rate: 0.1
    prefetch_top_k: 5

  short_term:
    retention_days: 7
    max_entries: 500
    consolidation_threshold: 0.3

  long_term:
    embedding_model: paraphrase-multilingual-mpnet-base-v2
    embedding_dim: 768
    vector_search_top_k: 10

  # Decay configuration
  decay:
    run_interval_hours: 1
    thresholds:
      decay: 0.4
      archive: 0.2
      delete: 0.05
    weights:
      recency: 0.30
      frequency: 0.15
      importance: 0.30
      event: 0.15
      profile: 0.10
    type_bonuses:
      preference: 0.2
      decision: 0.15
      fact: 0.1
      pattern: 0.1
      event: 0.0

  # Consolidation configuration
  consolidation:
    run_time: "03:00"  # 3 AM daily
    similarity_threshold: 0.8
    min_memories_to_merge: 2
    max_memories_to_merge: 10

  # Backup configuration
  backup:
    interval_hours: 6
    retention_days: 30
    locations:
      - local: /data/backups/memory
      - s3: s3://friday-backups/memory

  # Health monitoring
  health:
    check_interval_seconds: 60
    alert_thresholds:
      storage_percent: 90
      query_latency_ms: 2000
      daemon_restart_count: 3

# Voice command configuration
voice:
  confirmation_required:
    - memory_delete
    - profile_static_update
  languages:
    - en
    - te

# Telugu processing
telugu:
  stopwords_file: memory/telugu/stopwords.txt
  keyword_min_length: 2

# Storage backends
storage:
  stm:
    type: sqlite
    path: /data/memory/stm.db
  ltm:
    type: postgres
    host: localhost
    port: 5432
    database: friday_memory
    user: friday
    password: ${MEMORY_DB_PASSWORD}
  cache:
    type: redis
    host: localhost
    port: 6379
    db: 0
  profile:
    type: file
    path: memory/data/profile

# Monitoring
monitoring:
  metrics_port: 9090
  log_level: INFO
  alert_channels:
    - type: log
      path: /var/log/friday/memory_alerts.log
    - type: webhook
      url: ${ALERT_WEBHOOK_URL}
```

---

## Monitoring & Observability

### Metrics to Track

```python
class MemoryMetrics:
    """
    All metrics the team needs to monitor.
    They only check logs/dashboards, never intervene.
    """

    # Storage metrics
    stm_entry_count: int
    ltm_entry_count: int
    storage_used_bytes: int
    storage_percent: float

    # Performance metrics
    search_latency_p50_ms: float
    search_latency_p95_ms: float
    search_latency_p99_ms: float
    store_latency_ms: float
    embedding_latency_ms: float

    # Quality metrics
    search_hit_rate: float          # How often search returns results
    retrieval_relevance: float      # Average relevance score
    consolidation_rate: float       # STM → LTM rate
    decay_rate: float               # Memories decayed per hour

    # Health metrics
    daemon_uptime_seconds: Dict[str, int]
    last_backup_age_seconds: int
    embedding_queue_size: int
    error_count_last_hour: int

    # Usage metrics
    voice_commands_per_hour: int
    queries_per_hour: int
    languages_used: Dict[str, int]  # en: 50, te: 30, mixed: 20
```

### Log Format

```python
# Structured logging format
{
    "timestamp": "2026-01-29T10:30:00Z",
    "level": "INFO",
    "service": "memory",
    "component": "decay_daemon",
    "event": "decay_pass_completed",
    "details": {
        "memories_evaluated": 150,
        "memories_decayed": 12,
        "memories_archived": 3,
        "duration_ms": 450
    },
    "trace_id": "abc123"
}
```

### Alert Categories

| Category | Trigger | Action |
|----------|---------|--------|
| **CRITICAL** | System down, data loss risk | Page on-call |
| **ERROR** | Component failure, requires attention | Email + Slack |
| **WARN** | Degraded performance, self-healing engaged | Slack |
| **INFO** | Routine operations, statistics | Log only |

### Dashboard Panels

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    FRIDAY MEMORY DASHBOARD                               │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐   │
│  │   HEALTH    │  │   STORAGE   │  │   LATENCY   │  │   QUALITY   │   │
│  │   ✓ OK      │  │   45%       │  │   p95: 120ms│  │   85%       │   │
│  │             │  │   ████░░░   │  │   ▁▂▃▄▅▆▇▇  │  │   relevance │   │
│  └─────────────┘  └─────────────┘  └─────────────┘  └─────────────┘   │
│                                                                         │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                     MEMORY COUNTS OVER TIME                      │   │
│  │  STM: ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 342              │   │
│  │  LTM: ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 1,247      │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
│  ┌─────────────────────────┐  ┌─────────────────────────┐             │
│  │   DAEMON STATUS         │  │   RECENT OPERATIONS     │             │
│  │   decay:      ✓ running │  │   10:29 search (45ms)   │             │
│  │   consolidate: ✓ idle   │  │   10:28 store (12ms)    │             │
│  │   backup:     ✓ running │  │   10:27 decay pass      │             │
│  │   health:     ✓ running │  │   10:25 voice: remember │             │
│  └─────────────────────────┘  └─────────────────────────┘             │
│                                                                         │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                     LANGUAGE DISTRIBUTION                        │   │
│  │   English: ████████████████████████ 55%                          │   │
│  │   Telugu:  ████████████████ 35%                                  │   │
│  │   Mixed:   ██████ 10%                                            │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Edge Cases & Solutions

### Edge Case Matrix

| Edge Case | Problem | Solution |
|-----------|---------|----------|
| **Contradicting memories** | "Boss likes coffee" vs "Boss switched to tea" | Keep most recent, mark old as superseded |
| **Duplicate detection** | Same fact stored multiple times | Hash-based dedup + semantic similarity check |
| **Language confusion** | Telugu words that look like English | Use character-range detection, not dictionary |
| **Time zone handling** | "Tomorrow" at 11:59 PM | Always store UTC, convert on retrieval |
| **Concurrent updates** | Two processes updating same memory | Optimistic locking with version field |
| **Missing embeddings** | Model unavailable when storing | Queue for later, use keywords meanwhile |
| **Context bleed** | Kitchen memory appearing in Writers Room | Strict room/project scoping |
| **Importance inflation** | Everything marked important | Decay importance over time |
| **Query ambiguity** | "What did I say?" - about what? | Ask for clarification if context insufficient |
| **Large result sets** | Search returns 1000 results | Pagination + relevance cutoff |
| **Circular relationships** | A → B → C → A in knowledge graph | Detect cycles, allow with warning |
| **Stale event dates** | Event passed but not cleaned up | Nightly job to update statuses |
| **Profile conflicts** | Dynamic fact contradicts static | Static wins, log conflict for review |
| **Encoding issues** | Malformed Telugu Unicode | Normalize on input, validate on store |
| **Long conversations** | 100+ turn session | Progressive summarization, maintain highlights |

### Contradiction Resolution

```python
class ContradictionResolver:
    """
    Handle conflicting information in memories.
    """

    def resolve(
        self,
        new_fact: str,
        existing: List[Memory]
    ) -> ResolutionAction:
        """
        Determine how to handle potential contradiction.
        """
        # 1. Check for semantic similarity
        for memory in existing:
            similarity = self.calculate_similarity(new_fact, memory.content)

            if similarity > 0.9:
                # Very similar - likely duplicate
                return ResolutionAction(
                    action="skip",
                    reason="Duplicate detected",
                    existing_id=memory.id
                )

            elif similarity > 0.7:
                # Similar but different - potential contradiction
                contradiction = self.detect_contradiction(new_fact, memory.content)

                if contradiction:
                    # New fact contradicts old
                    return ResolutionAction(
                        action="supersede",
                        reason="Newer information supersedes",
                        existing_id=memory.id,
                        update_old=True,
                        old_status="superseded"
                    )
                else:
                    # Similar but complementary
                    return ResolutionAction(
                        action="store",
                        reason="Complementary information",
                        link_to=memory.id
                    )

        # No conflict detected
        return ResolutionAction(action="store", reason="New unique fact")
```

### Error Recovery Procedures

```python
# Procedure: Recover from total database failure

def recover_from_db_failure():
    """
    Complete recovery procedure.
    Automated - no human intervention.
    """

    # 1. Log the incident
    alert("CRITICAL", "Database failure detected, starting recovery")

    # 2. Try primary database
    if test_connection(config.primary_db):
        connect(config.primary_db)
        alert("INFO", "Primary database recovered")
        return True

    # 3. Try replica
    if test_connection(config.replica_db):
        connect(config.replica_db)
        promote_replica_to_primary()
        alert("WARN", "Promoted replica to primary")
        return True

    # 4. Restore from latest backup
    latest_backup = find_latest_backup()
    if latest_backup:
        restore_from_backup(latest_backup)
        alert("WARN", f"Restored from backup: {latest_backup.timestamp}")

        # Data loss calculation
        data_loss_hours = (datetime.now() - latest_backup.timestamp).hours
        alert("WARN", f"Potential data loss: {data_loss_hours} hours")
        return True

    # 5. Enter degraded mode
    enter_degraded_mode()
    alert("CRITICAL", "No recovery possible, running in degraded mode")
    return False
```

---

## Summary: The Complete Picture

### What Makes This Design Special

1. **Brain-Inspired**: Not just metaphor - actual mapping of human memory processes
2. **Think While Talk**: Zero-blocking architecture, memory never slows response
3. **Voice-First**: Every operation accessible via natural speech
4. **Fully Automated**: Team monitors logs, never intervenes
5. **Telugu-Native**: Built-in support for Telugu-English code-switching
6. **Self-Healing**: Every failure mode has automatic recovery
7. **Project-Aware**: Memories scoped to context (Gusagusalu vs Kitchen)
8. **Temporal Intelligence**: Dual timestamps for "when said" vs "when happens"
9. **Intelligent Decay**: Active forgetting prevents noise accumulation
10. **Observable**: Complete metrics and logging for transparency

### Key Innovation Summary

| Innovation | Inspiration | Implementation |
|------------|-------------|----------------|
| Dual Timestamps | Supermemory | Every memory tracks document_time AND event_time |
| Memory Atomicity | Supermemory | GLM-4.7-Flash extracts high-signal facts |
| Intelligent Decay | Supermemory + Brain | Score-based decay with type bonuses |
| Hierarchical Storage | Brain | Sensory → Working → STM → LTM |
| Attention Stack | Brain | 7±2 items with decay |
| Knowledge Graph | Brain (Semantic) | Entity relationships |
| Pattern Detection | Brain (Implicit) | Automatic habit learning |
| Self-Healing | Enterprise SRE | Every failure has auto-recovery |
| Voice Control | JARVIS | Natural language memory commands |

### Next Steps for Implementation

1. **Phase 1**: Core layers (Working, STM, LTM, Profile)
2. **Phase 2**: Search and storage operations
3. **Phase 3**: Decay and consolidation daemons
4. **Phase 4**: Voice command integration
5. **Phase 5**: Telugu-specific processing
6. **Phase 6**: Self-healing and monitoring
7. **Phase 7**: Knowledge graph and timeline
8. **Phase 8**: Pattern detection

---

## Appendix: Voice Command Quick Reference

| Command | Example | Action |
|---------|---------|--------|
| Search | "Do you remember the climax discussion?" | Semantic + temporal search |
| Store | "Remember this: Ravi needs hesitation" | Store with high importance |
| Important | "This is important" / "ఇది ముఖ్యం" | Boost current context importance |
| Forget | "Forget about the old ending" | Delete with confirmation |
| Timeline | "What's coming up next week?" | Query upcoming events |
| Project | "Switch to Gusagusalu" | Set project context |
| Status | "Memory status" | Show memory health |
| Profile | "Update my profile: I prefer morning work" | Update preference |

---

*"A good memory is not just about storing everything - it's about knowing what to remember, what to forget, and when to recall."*

---

**Document Version**: 1.0
**Last Updated**: January 29, 2026
**Status**: Ready for Implementation
