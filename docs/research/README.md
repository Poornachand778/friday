# Research Documentation

This folder contains research notes on external repositories, papers, and ideas that inspired or were integrated into Friday.

## Purpose

When Boss brings a new repo/idea to research:
1. Create a new file: `{REPO_NAME}_ANALYSIS.md`
2. Document findings, pros/cons, and what we adopted
3. Link to the implementation in Friday's codebase

## Research Process

```
1. Boss shares repo/idea
   ↓
2. Deep research (WebFetch, WebSearch, code analysis)
   ↓
3. Document in docs/research/{NAME}_ANALYSIS.md
   ↓
4. Decision: Adopt / Adapt / Skip
   ↓
5. If adopted: Implement + update CHANGELOG.md
   ↓
6. Git commit with reference to research doc
```

## Research Document Template

```markdown
# {Repository/Idea Name} Analysis

**Date**: YYYY-MM-DD
**Source**: {GitHub URL or Paper link}
**Status**: Researched / Adopted / Partially Adopted / Rejected

## Summary
Brief description of what this is.

## Key Concepts
- Concept 1: Description
- Concept 2: Description

## What We Can Use
1. Feature/pattern and why it's useful for Friday

## What We Skipped
1. Feature and why we didn't need it

## Implementation in Friday
- File: `path/to/file.py`
- How we adapted it for our needs

## References
- Links to related docs
```

## Completed Research

| Date | Research | Status | Implementation |
|------|----------|--------|----------------|
| 2025-01-29 | [TiDAR](TIDAR_ANALYSIS.md) | Future Consideration | Track for voice latency optimization |
| 2025-01-29 | [Supermemory](SUPERMEMORY_ANALYSIS.md) | Adopted | memory/layers/*, dual timestamps |
| 2025-01-29 | [Cognee](COGNEE_ANALYSIS.md) | Adopted | memory/layers/knowledge_graph.py |
| 2025-01-28 | [LLaMA 3.1 8B](llama_3.1_8B_model_training_research.txt) | Adopted | Base model for Friday |

## Future Research Queue

Add repos/ideas to research here:
- [ ] TiDAR implementation when NVIDIA releases training code
- [ ] (Add next research item)
