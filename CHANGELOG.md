# Friday AI Changelog

All notable changes to Friday are documented here.

Format based on [Keep a Changelog](https://keepachangelog.com/).

---

## [Unreleased]

### Added
- **Knowledge Graph Layer** (Cognee-inspired)
  - NetworkX-based graph for entity relationships
  - Node types: character, scene, project, person, concept, location, event
  - Relation types: discusses, contains, character_in, creates, wants, etc.
  - Graph traversal for queries like "What scenes involve Ravi?"
  - Files: `memory/layers/knowledge_graph.py`
  - Research: `docs/research/COGNEE_ANALYSIS.md`

- **Triplet Extractor**
  - GLM-4.7-Flash powered subject-relation-object extraction
  - Fallback rule-based extraction when API unavailable
  - Auto-extraction during LTM consolidation
  - Files: `memory/operations/triplet_extractor.py`

- **Decay Algorithm** (Ebbinghaus-inspired)
  - Background daemon for memory decay and consolidation
  - STM → LTM consolidation for high-importance entries
  - Configurable decay rates and thresholds
  - Files: `memory/operations/decay.py`

- **Research Documentation System**
  - Template for documenting research on external repos
  - Files: `docs/research/README.md`

### Changed
- **MemoryManager** now integrates Knowledge Graph
  - New methods: `graph_query()`, `get_related_entities()`, `get_entities_by_type()`
  - `store_fact()` now auto-extracts triplets
  - Health check includes graph stats

- **ShortTermMemory** added helper methods for decay daemon
  - `mark_accessed()`, `update_importance()`, `get_entries_before()`

---

## [2025-01-29] - Memory System Foundation

### Added
- **Brain-Inspired Memory Architecture**
  - Working Memory (7±2 items, attention stack)
  - Short-Term Memory (SQLite + FTS5, 7-day retention)
  - Long-Term Memory (vector embeddings, permanent)
  - Profile Store (identity facts, never decays)
  - Research: `docs/research/SUPERMEMORY_ANALYSIS.md`

- **Telugu-English Processor**
  - Language detection and density calculation
  - Code-switch point detection
  - Keyword extraction for search
  - Files: `memory/telugu/processor.py`

- **Memory Configuration System**
  - YAML-based config with env var overrides
  - Separate configs for each memory layer
  - Files: `memory/config.py`

### Research Completed
- Supermemory architecture (dual timestamps, memory atomicity)
- Cognee knowledge graphs (triplet extraction, hybrid search)

---

## [2025-01-28] - GLM-4.7-Flash Router

### Added
- **Intelligent Request Router**
  - GLM-4.7-Flash for task analysis
  - Task complexity detection (simple/moderate/complex)
  - Context detection (writers_room, kitchen, general)
  - Tool suggestion and ordering
  - Files: `orchestrator/inference/router.py`
  - Research: `docs/GLM-4.7-FLASH-GUIDE.md`

---

## [Earlier] - Foundation

### Core Infrastructure
- Database schema (scripts, training, memory, eval)
- MCP scene_manager service
- SageMaker training pipeline
- Iteration 1 model training (168MB LoRA)

### Training Data
- WhatsApp conversation processing (10,577 examples)
- Interview collector system
- ChatML format converters

---

## Version History Reference

| Version | Date | Focus |
|---------|------|-------|
| Iteration 5 | 2025-01-29 | Memory + Knowledge Graph |
| Iteration 4 | 2025-01-28 | GLM Router Integration |
| Iteration 3 | Earlier | Training data expansion |
| Iteration 2 | Earlier | Tool examples + WhatsApp |
| Iteration 1 | Earlier | Initial persona model |

---

## How to Read This Changelog

- **Added**: New features
- **Changed**: Modifications to existing features
- **Fixed**: Bug fixes
- **Removed**: Deleted features
- **Research**: Links to research docs that drove decisions
- **Files**: Key files involved in the change
