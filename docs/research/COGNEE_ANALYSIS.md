# Cognee Architecture Analysis for Friday AI

> Deep research on graph-based AI memory for potential adoption

**Research Date**: January 29, 2026

---

## Executive Summary

Cognee is an **11.5K star** open-source AI memory engine that combines **vector databases + knowledge graphs** in a unified memory layer. Their key innovation is the **ECL pipeline** (Extract, Cognify, Load) replacing traditional RAG, achieving **~90% accuracy vs RAG's ~60%**.

| Aspect | Cognee | Friday (Current) | Recommendation |
|--------|--------|------------------|----------------|
| Memory Storage | Graph DB + Vector DB | SQLite + Embeddings | **Adopt graph for relationships** |
| Search | Hybrid (vector + graph traversal) | Vector similarity only | **Add graph traversal** |
| Relationships | Triplet extraction (subject-relation-object) | None | **Critical to add** |
| Entity Linking | Automatic | Manual/None | **Adopt** |
| Memory Algorithms | "Memphis" (cleanup, reconnect) | Decay only | **Consider adopting** |

**Verdict**: Cognee's **graph-based relationship tracking** and **triplet extraction** are innovations we should adopt. Their hybrid search (vector + graph) significantly outperforms pure vector search.

---

## Cognee Architecture Deep Dive

### Core Philosophy: Memory-First Architecture

```
┌──────────────────────────────────────────────────────────────────────────┐
│                     COGNEE ARCHITECTURE                                   │
├──────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│   INPUT DATA                                                             │
│   ├─ Documents (PDF, HTML, text)                                         │
│   ├─ Conversations                                                       │
│   ├─ Images, Audio                                                       │
│   └─ 30+ data source connectors                                          │
│           │                                                              │
│           ▼                                                              │
│   ┌─────────────────────────────────────────────────────────────────┐   │
│   │                    ECL PIPELINE                                  │   │
│   │                                                                  │   │
│   │  EXTRACT        COGNIFY              LOAD                        │   │
│   │  ┌──────┐      ┌────────────┐       ┌──────────┐                │   │
│   │  │Ingest│ ──▶  │ 6-Stage    │  ──▶  │ Graph DB │                │   │
│   │  │Data  │      │ Processing │       │ Vector DB│                │   │
│   │  └──────┘      └────────────┘       └──────────┘                │   │
│   │                     │                                            │   │
│   │                     ▼                                            │   │
│   │              ┌────────────┐                                      │   │
│   │              │ MEMIFY     │ ← Graph enrichment algorithms        │   │
│   │              │ (Memphis)  │                                      │   │
│   │              └────────────┘                                      │   │
│   └─────────────────────────────────────────────────────────────────┘   │
│           │                                                              │
│           ▼                                                              │
│   RETRIEVAL                                                              │
│   ├─ Time-based filters                                                  │
│   ├─ Graph traversal                                                     │
│   └─ Vector similarity                                                   │
│                                                                          │
└──────────────────────────────────────────────────────────────────────────┘
```

### The 6-Stage Cognify Pipeline

1. **Classification**: Categorize input content
2. **Chunking**: Split into processable units
3. **Entity Extraction**: Identify people, places, concepts
4. **Relationship Detection**: Find connections (triplets)
5. **Embedding Generation**: Vector representations
6. **Summarization**: Compressed representations

### Key Innovation: Triplet Extraction

```
RAW TEXT:
"Boss discussed the climax scene with Ravi's father confrontation"

         ↓ TRIPLET EXTRACTION ↓

TRIPLETS (Subject-Relation-Object):
┌─────────────────────────────────────────────┐
│  ("Boss", "discussed", "climax scene")      │
│  ("climax scene", "contains", "confrontation")│
│  ("Ravi", "has_father", "father character") │
│  ("confrontation", "involves", "Ravi")      │
│  ("confrontation", "involves", "father")    │
└─────────────────────────────────────────────┘
```

**Why this matters**: These triplets enable questions like:
- "What scenes involve Ravi?" → Graph traversal finds all connections
- "Show me all family relationships" → Query edges of type "has_father", "has_mother"
- "What did Boss discuss?" → Follow edges from Boss node

### Memphis Algorithms (Graph Memory Algorithms)

Cognee's "Memphis" algorithms maintain graph health:

1. **Cleanup**: Remove orphan nodes, dead links
2. **Reconnection**: Discover implicit relationships
3. **Structure Optimization**: Improve traversal efficiency
4. **Deduplication**: Merge duplicate entities

---

## Technical Specifications

### Supported Backends

| Type | Options |
|------|---------|
| **Graph DB** | Neo4j, FalkorDB, Kuzu, NetworkX |
| **Vector DB** | Qdrant, LanceDB, PGVector, Redis |
| **LLM** | OpenAI, Anthropic, AWS Bedrock |
| **Embedding** | Multiple providers |

### Node & Edge Schema

```python
# Cognee's DataPoint approach
class EntityNode:
    id: str
    type: str           # person, object, location, event, concept
    name: str
    attributes: Dict    # Flexible metadata
    embedding: List[float]

class RelationshipEdge:
    source_id: str
    target_id: str
    relation_type: str  # by, of, at, discussed, contains, etc.
    weight: float
    context: str        # Original text context
    timestamp: datetime
```

### Hybrid Search Strategy

```python
# Cognee's retrieval approach
async def hybrid_search(query: str) -> List[Result]:
    # 1. Vector similarity search
    vector_results = await vector_db.search(
        embedding=embed(query),
        top_k=20
    )

    # 2. Graph traversal from vector results
    graph_results = []
    for result in vector_results[:5]:
        # Find connected nodes (1-2 hops)
        neighbors = await graph_db.traverse(
            start_node=result.entity_id,
            max_depth=2,
            relation_types=["discusses", "contains", "related_to"]
        )
        graph_results.extend(neighbors)

    # 3. Time-based filtering
    filtered = filter_by_time(graph_results, query)

    # 4. Merge and rank
    final = merge_and_rank(vector_results, filtered)

    return final
```

---

## Performance Comparison

### RAG vs Cognee (from their research)

| Metric | Traditional RAG | Cognee |
|--------|-----------------|--------|
| **Accuracy** | ~60% | ~90% |
| **Semantic Matching** | Prone to category mismatches | Context-aware |
| **Scalability** | 50MB causes issues | Handles larger volumes |
| **Update Speed** | Days, failure-prone | Clean delete-replace |
| **Hallucination** | High (no verification) | Low (verified relationships) |

### Why Graph Beats Pure Vector

1. **Structural Context**: Vector search finds similar content, graphs find related content
2. **Multi-hop Reasoning**: "Ravi's father's motivation" requires traversal
3. **Relationship Types**: Distinguish "discussed" from "created" from "deleted"
4. **Provenance**: Every inferred fact links back to source DocumentChunk

---

## What Friday Should Adopt

### Must-Have Additions

#### 1. Knowledge Graph Layer

```python
# Proposed addition to Friday Memory

class KnowledgeNode:
    id: str
    name: str
    node_type: str          # character, scene, concept, project, event
    attributes: Dict
    embedding: List[float]
    source_memory_ids: List[str]  # Link to LTM entries

class KnowledgeEdge:
    source_id: str
    target_id: str
    relation: str           # discusses, creates, contains, relates_to
    weight: float
    context: str
    created_at: datetime

class FridayKnowledgeGraph:
    """
    Graph layer for Friday's memory.
    Uses NetworkX locally, can scale to Neo4j.
    """

    async def add_triplets(self, triplets: List[Tuple[str, str, str]]):
        """Add subject-relation-object triplets"""
        pass

    async def traverse(self, start: str, depth: int = 2) -> List[KnowledgeNode]:
        """Graph traversal for related entities"""
        pass

    async def query_relations(self, entity: str, relation: str) -> List[KnowledgeNode]:
        """Find entities with specific relation"""
        pass
```

#### 2. Triplet Extraction

```python
TRIPLET_EXTRACTION_PROMPT = """
Extract knowledge triplets from this conversation.
Return JSON array of [subject, relation, object] tuples.

Relations to use:
- discusses (person discusses topic)
- wants (person wants something)
- creates (person creates thing)
- contains (scene contains element)
- relates_to (concept relates to concept)
- character_of (character of project)
- deadline_for (deadline for project)

Conversation:
{conversation}

Output format:
[["Boss", "discusses", "climax scene"], ["climax scene", "contains", "confrontation"]]
"""

async def extract_triplets(conversation: str) -> List[Tuple[str, str, str]]:
    """Use GLM-4.7-Flash for fast triplet extraction"""
    response = await glm_router.chat(
        messages=[{"role": "user", "content": TRIPLET_EXTRACTION_PROMPT.format(
            conversation=conversation
        )}],
        temperature=0.1,
    )
    return json.loads(response.content)
```

#### 3. Hybrid Search

```python
async def hybrid_memory_search(
    self,
    query: str,
    top_k: int = 10,
) -> List[MemoryResult]:
    """
    Combined vector + graph search.
    """
    # Step 1: Vector search in LTM
    vector_results = await self.ltm.search(query, top_k=top_k * 2)

    # Step 2: Extract entities from query
    query_entities = await self.extract_entities(query)

    # Step 3: Graph traversal from entities
    graph_results = []
    for entity in query_entities:
        neighbors = await self.knowledge_graph.traverse(
            entity, depth=2
        )
        for neighbor in neighbors:
            # Find LTM entries mentioning this entity
            related = await self.ltm.search_by_entity(neighbor.name)
            graph_results.extend(related)

    # Step 4: Merge and deduplicate
    all_results = self._merge_results(vector_results, graph_results)

    # Step 5: Re-rank by combined relevance
    return self._rerank(all_results, query)[:top_k]
```

### Nice-to-Have Additions

#### 1. Memphis-style Graph Cleanup

```python
class GraphMaintenanceDaemon:
    """
    Memphis-inspired graph maintenance.
    Runs periodically to maintain graph health.
    """

    async def cleanup_orphans(self):
        """Remove nodes with no edges"""
        orphans = await self.graph.find_orphan_nodes()
        for node in orphans:
            if node.age_days > 7:  # Grace period
                await self.graph.delete_node(node.id)

    async def discover_implicit_relations(self):
        """Find relationships that should exist based on context"""
        # If A discusses B, and B contains C, then A indirectly relates to C
        pass

    async def merge_duplicates(self):
        """Merge nodes that represent the same entity"""
        # Use embedding similarity to find duplicates
        pass
```

#### 2. Ontology Support

```python
# Define domain ontology for Telugu cinema
FRIDAY_ONTOLOGY = {
    "entities": {
        "character": {"attributes": ["name", "role", "arc"]},
        "scene": {"attributes": ["scene_code", "act", "emotion_level"]},
        "project": {"attributes": ["name", "deadline", "status"]},
        "concept": {"attributes": ["name", "domain"]},
    },
    "relations": {
        "character_in": ["character", "scene"],
        "scene_in": ["scene", "project"],
        "discusses": ["person", "*"],
        "deadline_for": ["date", "project"],
    }
}
```

---

## Implementation Plan for Friday

### Phase 1: Add NetworkX Knowledge Graph (Week 1)

```
memory/
├── layers/
│   └── knowledge_graph.py    # NEW: NetworkX-based graph
├── operations/
│   └── triplet_extractor.py  # NEW: GLM-based extraction
```

**Deliverables**:
- KnowledgeGraph class with NetworkX backend
- Triplet extraction using GLM-4.7-Flash
- Integration with MemoryManager

### Phase 2: Hybrid Search (Week 2)

**Deliverables**:
- Combined vector + graph search
- Entity extraction from queries
- Result merging and re-ranking

### Phase 3: Graph Maintenance (Week 3)

**Deliverables**:
- Orphan cleanup daemon
- Duplicate detection and merging
- Implicit relationship discovery

### Phase 4: Schema Enforcement (Week 4)

**Deliverables**:
- Telugu cinema ontology
- Schema validation for nodes/edges
- Migration to production graph DB (optional)

---

## Comparison: Supermemory vs Cognee

| Feature | Supermemory | Cognee |
|---------|-------------|--------|
| **Focus** | Temporal reasoning, decay | Relationship mapping, retrieval |
| **Storage** | Hot/warm/cold tiers | Graph + Vector hybrid |
| **Key Innovation** | Dual timestamps, intelligent decay | Triplet extraction, graph traversal |
| **Best For** | Time-based queries | Relationship queries |
| **Self-hosted** | Enterprise only | Full open source |

**Recommendation for Friday**: Adopt **BOTH** approaches:
- Supermemory's temporal reasoning + decay
- Cognee's knowledge graph + triplet extraction

---

## Code Example: Complete Integration

```python
# Proposed Friday Memory with Graph

class EnhancedMemoryManager:
    """
    Friday Memory with Cognee-inspired graph layer.
    """

    def __init__(self):
        # Existing layers
        self.working = WorkingMemory()
        self.stm = ShortTermMemory()
        self.ltm = LongTermMemory()
        self.profile = ProfileStore()

        # NEW: Knowledge graph (Cognee-inspired)
        self.graph = KnowledgeGraph()

        # NEW: Triplet extractor
        self.extractor = TripletExtractor()

    async def store_turn(self, user_msg: str, assistant_response: str):
        """Store conversation with graph extraction"""

        # 1. Store in existing layers
        turn = await super().store_turn(user_msg, assistant_response)

        # 2. Extract and store triplets (Cognee-inspired)
        triplets = await self.extractor.extract(
            f"User: {user_msg}\nFriday: {assistant_response}"
        )

        for subject, relation, obj in triplets:
            # Ensure nodes exist
            await self.graph.ensure_node(subject)
            await self.graph.ensure_node(obj)

            # Add relationship
            await self.graph.add_edge(subject, obj, relation)

        return turn

    async def search(self, query: str) -> List[MemoryResult]:
        """Hybrid search: vector + graph"""

        # Vector search
        vector_results = await self.ltm.search(query)

        # Graph traversal from top vector results
        graph_results = []
        for result in vector_results[:3]:
            entities = await self.extractor.extract_entities(result.content)
            for entity in entities:
                neighbors = await self.graph.traverse(entity, depth=2)
                for neighbor in neighbors:
                    related = await self.ltm.search_by_entity(neighbor)
                    graph_results.extend(related)

        # Merge and return
        return self._merge_results(vector_results, graph_results)
```

---

## Key Takeaways

### What Makes Cognee Special

1. **Triplet Extraction**: Converting text to subject-relation-object enables graph queries
2. **Hybrid Search**: Vector for similarity + Graph for relationships
3. **Memphis Algorithms**: Active graph maintenance, not just storage
4. **Provenance**: Every node links back to source document
5. **Multi-hop Reasoning**: Traverse relationships, not just find similar

### What Friday Should Adopt

1. **Knowledge Graph Layer** (NetworkX initially, Neo4j for scale)
2. **Triplet Extraction** (using GLM-4.7-Flash)
3. **Hybrid Search** (combine our vector with graph traversal)
4. **Relationship Types** for Telugu cinema domain
5. **Graph Maintenance Daemon**

### What We Already Have (Better than Cognee)

1. **Temporal Reasoning** (Supermemory-inspired dual timestamps)
2. **Intelligent Decay** (active forgetting)
3. **Telugu-Native Processing** (code-switching awareness)
4. **Voice Control** (natural language memory commands)
5. **Profile Store** (identity that never decays)

---

## Conclusion

Cognee's **knowledge graph approach** is the missing piece in Friday's memory. Our current system handles:
- ✅ Temporal reasoning (when things happened)
- ✅ Semantic similarity (what's related by meaning)
- ✅ Decay and consolidation (brain-inspired)
- ❌ **Structural relationships** (who discussed what, scene contains character)

Adding Cognee-inspired **triplet extraction + graph traversal** will enable:
- "What scenes involve Ravi?" → Graph query
- "Everything Boss discussed about the climax" → Relationship traversal
- "Characters in Gusagusalu" → Entity type query
- "How are these two concepts connected?" → Path finding

**Recommended Action**: Add NetworkX-based knowledge graph with GLM-powered triplet extraction in the next iteration.

---

## References

- [Cognee GitHub (11.5K stars)](https://github.com/topoteretes/cognee)
- [From RAG to Graphs: How Cognee is Building Self-Improving AI Memory](https://memgraph.com/blog/from-rag-to-graphs-cognee-ai-memory)
- [Cognee MCP Integration](https://www.cognee.ai/blog/deep-dives/model-context-protocol-cognee-llm-memory-made-simple)
- [Knowledge Graphs Explained](https://www.cognee.ai/blog/fundamentals/building-blocks-of-knowledge-graphs)
- [Research Paper: Optimizing Knowledge Graphs for LLM Reasoning (arXiv:2505.24478)](https://arxiv.org/abs/2505.24478)

---

*"Vector search finds what's similar. Graph traversal finds what's connected. You need both."*
