# Research: Vector Databases for Friday AI

**Date**: 2025-02-04
**Status**: Completed
**Recommendation**: Dual-database approach (pgvector + Qdrant)

---

## Requirements Analysis

| Use Case | Data Type | Scale | Requirements |
|----------|-----------|-------|--------------|
| **LTM (Long-Term Memory)** | Memories, facts, experiences | ~10K-100K vectors | Metadata filtering, hybrid search |
| **Document Processing** | Book chunks, PDFs | ~100K-1M vectors | Full-text + vector search |
| **Knowledge Graph** | Entity embeddings | ~10K-50K vectors | Graph-aware retrieval |
| **Scene Manager** | Screenplay scenes | ~1K-10K vectors | SQL + vector queries |
| **WhatsApp/Training Data** | Conversation embeddings | ~50K-500K vectors | Fast retrieval |

---

## Database Comparison

| Database | Best For | Latency | Throughput | Filtering | Self-Host | Stars |
|----------|----------|---------|------------|-----------|-----------|-------|
| **Qdrant** | Filtered search, RAG | p95: 37ms | 41 QPS | ⭐⭐⭐⭐⭐ | Easy (Docker) | 23K+ |
| **pgvector** | Unified SQL + vector | p95: 60ms | 471 QPS | ⭐⭐⭐ | Existing PG | 15K+ |
| **Milvus** | Enterprise scale (1B+) | p95: <30ms | 100K+ QPS | ⭐⭐⭐⭐ | K8s | 35K+ |
| **Weaviate** | Semantic + knowledge graph | p95: 50ms | Medium | ⭐⭐⭐⭐ | Docker/K8s | 14K+ |
| **ChromaDB** | Prototyping, lightweight | Varies | Low | ⭐⭐ | Embedded | 18K+ |
| **LanceDB** | Serverless, embedded | Fast | High | ⭐⭐⭐⭐ | File-based | 10K+ |

---

## Detailed Analysis

### 1. Qdrant - Best for Complex Filtering

**Strengths**:
- Rust-based, high performance
- Best-in-class metadata filtering - no overhead even with 5 filters
- Hybrid search (vector + keyword + metadata) built-in
- Real-time data updates
- Apache 2.0 license

**Limitations**:
- Scaling beyond 50M vectors requires more effort
- Smaller ecosystem than Milvus

**Good for Friday**: LTM queries with user/topic/date filters

### 2. pgvector + pgvectorscale - Best for Unified Data

**Strengths**:
- 471 QPS at 99% recall - 11.4x better throughput than Qdrant
- Works inside PostgreSQL (already in Friday stack)
- SQL queries combining vector + relational data
- Mature operational tooling (backups, replication)
- HNSW and StreamingDiskANN indexes

**Limitations**:
- Filtering overhead (2.3x)
- Needs tuning for best performance
- Can impact main DB performance

**Good for Friday**: Scene manager, screenplay data (already in PostgreSQL)

### 3. Milvus - Best for Scale

**Strengths**:
- Designed for billions of vectors
- GPU acceleration
- 100K+ QPS throughput
- Distributed by design

**Limitations**:
- Requires Kubernetes
- Operational complexity
- Overkill for Friday's current scale

### 4. LanceDB - Best for Serverless/Embedded

**Strengths**:
- File-based, S3-compatible
- Embedded (no server needed)
- Multimodal (text, images, video)
- 100x cost savings with compute-storage separation

**Limitations**:
- Newer, smaller ecosystem

**Good for Friday**: Local development, edge deployment

### 5. Weaviate - Best for Knowledge Graph + Vector

**Strengths**:
- Hybrid: vector + knowledge graph
- Built-in vectorization (auto-embed with OpenAI, HuggingFace)
- GraphQL API

**Limitations**:
- More complex setup

**Good for Friday**: Knowledge graph entity embeddings

---

## Performance Benchmarks (2025)

| Database | p50 Latency | p95 Latency | p99 Latency | QPS @ 99% Recall |
|----------|-------------|-------------|-------------|------------------|
| Qdrant | 30.75ms | 36.73ms | 38.71ms | 41 |
| pgvector+scale | 31.07ms | 60.42ms | 74.60ms | **471** |
| Milvus | <100ms | <30ms | - | 100K+ |
| Redis Vector | **<10ms** | - | - | High |

---

## Recommended Architecture for Friday

```
┌─────────────────────────────────────────────────────────────────┐
│                    Friday Vector Architecture                    │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  PostgreSQL + pgvector (Primary)                                │
│  ├── Scene Manager (scenes, projects)                           │
│  ├── Training Data Registry                                     │
│  ├── Document metadata                                          │
│  └── Structured data with vector columns                        │
│                                                                  │
│  Qdrant (Secondary - for RAG/Memory)                            │
│  ├── LTM Memories (with complex filters)                        │
│  ├── Document Chunks (hybrid search)                            │
│  ├── Knowledge Graph Embeddings                                 │
│  └── Real-time conversation context                             │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### Why This Combination?

| Need | Solution |
|------|----------|
| Already use PostgreSQL | pgvector adds vectors to existing tables |
| Complex memory filtering | Qdrant excels at filtered vector search |
| Hybrid search | Qdrant has this built-in |
| High throughput for batch | pgvector: 471 QPS vs Qdrant's 41 QPS |
| Real-time memory updates | Qdrant designed for live updates |
| Operational simplicity | Both are easy to self-host (Docker) |

---

## Quick Start Commands

### Qdrant (Docker)
```bash
docker run -p 6333:6333 -v $(pwd)/qdrant_data:/qdrant/storage qdrant/qdrant
```

### pgvector (in existing PostgreSQL)
```sql
CREATE EXTENSION vector;
ALTER TABLE ltm_memories ADD COLUMN embedding vector(384);
CREATE INDEX ON ltm_memories USING hnsw (embedding vector_cosine_ops);
```

---

## Implementation Steps

### Phase 1: pgvector Integration
1. Install pgvector extension in existing PostgreSQL
2. Add embedding columns to scene tables
3. Create HNSW indexes
4. Update scene_manager service to use vector search

### Phase 2: Qdrant Setup
1. Deploy Qdrant via Docker
2. Create collections for LTM, documents, knowledge graph
3. Configure payload indexes for filtering
4. Integrate with MemoryManager

### Phase 3: Hybrid Search
1. Implement query routing (simple queries → pgvector, complex → Qdrant)
2. Add fallback mechanisms
3. Benchmark and tune

---

## Sources

- https://www.firecrawl.dev/blog/best-vector-databases-2025
- https://www.tigerdata.com/blog/pgvector-vs-qdrant
- https://medium.com/@fendylike/top-5-open-source-vector-search-engines-a-comprehensive-comparison-guide-for-2025-e10110b47aa3
- https://latenode.com/blog/ai-frameworks-technical-infrastructure/vector-databases-embeddings/best-vector-databases-for-rag-complete-2025-comparison-guide
- https://lancedb.com/
- https://medium.com/@techlatest.net/from-milvus-to-qdrant-the-ultimate-guide-to-the-top-10-open-source-vector-databases-7d2805ed8970

---

*Research conducted: 2025-02-04*
