# Upskill Analysis: Agent Skills for Model Performance Transfer

**Date**: 2025-01-29
**Source**: https://huggingface.co/blog/upskill | https://github.com/huggingface/upskill
**Authors**: HuggingFace Team
**Status**: Researched - **Highly Recommended for Adoption**

---

## Summary

Upskill is a HuggingFace tool for generating and evaluating **agent skills** — portable, domain-specific instructions that transfer expertise from large models (like Claude Opus) to smaller, cheaper models. Skills are markdown files that dramatically improve smaller model performance on specialized tasks.

---

## The Problem It Solves

In agentic networks, different models have different capabilities:
- **Large models** (Claude Opus, GPT-4): High quality but expensive and slow
- **Small models** (GLM-4.7-Flash, Haiku): Fast and cheap but lower accuracy on complex tasks

**Current approach**: Use large models for complex tasks, accept the cost/latency penalty.

**Upskill approach**: Transfer expertise via skills so small models can handle complex tasks.

```
Traditional:
  Complex Task → Large Model → High Cost/Latency → Good Result

With Upskill:
  Complex Task → Small Model + Skill → Low Cost/Fast → Good Result
```

---

## How It Works

### Teacher-Student Approach

1. **Skill Generation**: Claude Opus completes a complex task and documents the process as a skill
2. **Test Case Creation**: Automatic test cases generated from task description
3. **Evaluation**: Student models tested with/without skill to measure improvement
4. **Iteration**: Skills refined based on evaluation results

### Skill Structure

```
./skills/screenplay-scene-manager/
├── SKILL.md              # Main instructions (~500 tokens)
└── skill_meta.json       # Metadata and test cases
```

**SKILL.md format**:
```markdown
---
name: screenplay-scene-manager
description: Manage Telugu screenplay scenes with proper structure
---

## Instructions
1. Scene codes follow format: PRJ-SXXX (e.g., GUS-S001)
2. Canonical text uses Telugu script with stage directions in English
3. Status transitions: draft → active → locked → archived
...
```

---

## Benchmark Results

### CUDA Kernel Writing (from HuggingFace blog)

| Model | Baseline | With Skill | Improvement |
|-------|----------|-----------|-------------|
| Claude Opus 4.5 | 60% | 95% | **+35%** |
| GLM-4.7-Flash (Q4_0) | 40% | 85% | **+45%** |
| Haiku | 80% | - | - |
| Sonnet | 100% | - | - |

**Key insight**: Skills provide the MOST benefit to smaller models. GLM-4.7-Flash improved 45% vs Opus's 35%.

---

## Relevance to Friday

### Our Current Architecture

```
User Request
    ↓
┌─────────────────────────────────┐
│ FridayOrchestrator              │
│   ↓                             │
│ GLM-4.7-Flash (Router)          │  ← Skills would help HERE
│   ↓                             │
│ Route to appropriate model      │
│   - Simple → GLM-4.7-Flash      │  ← And HERE
│   - Complex → Friday-Core (8B)  │
│   - Screenplay → MCP tools      │
└─────────────────────────────────┘
```

### Why Upskill is Perfect for Friday

1. **GLM-4.7-Flash Router**: Our router uses GLM-4.7-Flash for intent classification. Skills could improve routing accuracy significantly.

2. **Screenplay Domain**: Skills can encode:
   - Scene management conventions (GUS-S001 format)
   - Telugu-English code-switching patterns
   - Screenplay structure rules
   - Boss's communication preferences

3. **Cost Optimization**: Instead of calling expensive models for specialized tasks, smaller models + skills can handle them.

4. **Local Model Support**: Works with local models via OpenAI-compatible endpoints (our GLM setup at api.z.ai).

---

## What We Can Use

### 1. Screenplay-Specific Skills (High Priority)

| Skill Name | Purpose | Expected Benefit |
|------------|---------|------------------|
| `friday-scene-manager` | Scene CRUD operations | Better tool call accuracy |
| `friday-telugu-codegen` | Telugu-English code-switching | More natural responses |
| `friday-persona` | Boss communication style | Consistent personality |
| `friday-screenplay-structure` | Screenplay formatting | Better creative assistance |

### 2. Router Skills (High Priority)

Create skills for GLM-4.7-Flash router to:
- Classify intent more accurately
- Decide when to escalate to Friday-Core
- Handle ambiguous requests

### 3. Tool Usage Skills (Medium Priority)

Skills for MCP tool operations:
- `scene_search` query formulation
- `scene_update` field mapping
- Multi-tool workflows

---

## Implementation Plan

### Phase 1: Install and Test (1 hour)

```bash
# Install upskill
pip install upskill

# Set up API keys
export ANTHROPIC_API_KEY=sk-ant-...
export HF_TOKEN=hf_...

# Test with our GLM setup
upskill eval ./skills/test-skill/ \
    --model "unsloth/GLM-4.7-Flash-GGUF:Q4_0" \
    --base-url https://api.z.ai/api/paas/v4
```

### Phase 2: Generate Friday Skills (2-3 hours)

1. **Scene Manager Skill**:
   ```bash
   upskill generate "Manage Telugu screenplay scenes with proper scene codes, status transitions, and Telugu-English formatting"
   ```

2. **Persona Skill**:
   ```bash
   upskill generate "Respond as Friday, a Telugu screenwriter's assistant who uses Boss prefix, Telugu-English code-switching, and stays brief (max 6 lines)"
   ```

3. **Router Skill**:
   ```bash
   upskill generate "Classify user intent for screenplay assistant: tool-call, creative-help, information-query, or casual-chat"
   ```

### Phase 3: Evaluate Skills (1-2 hours)

```bash
# Test persona skill on GLM-4.7-Flash
upskill eval ./skills/friday-persona/ \
    --model "glm-4.7-flash" \
    --base-url https://api.z.ai/api/paas/v4

# Compare with/without skill
upskill eval ./skills/friday-persona/ \
    --model haiku --model sonnet
```

### Phase 4: Deploy to Friday (30 min)

1. Store skills in `friday/skills/` directory
2. Load skills into orchestrator's system prompt
3. Update GLM router to use skills
4. Monitor improvement metrics

---

## What We Skip (For Now)

1. **Skill iteration loops** - Start with basic skills, optimize later
2. **Automatic test generation** - Manual tests for screenplay domain
3. **Cross-tool compatibility** - Focus on Friday-specific skills first

---

## Technical Requirements

### Dependencies

```
upskill>=0.1.0
anthropic>=0.25.0  # Already have this
```

### API Keys Required

- `ANTHROPIC_API_KEY` - For skill generation (uses Claude Opus)
- `ZHIPU_API_KEY` - For evaluating with GLM-4.7-Flash

### Storage

Skills are small (~500 tokens each):
```
friday/
└── skills/
    ├── friday-persona/
    │   ├── SKILL.md
    │   └── skill_meta.json
    ├── friday-scene-manager/
    ├── friday-router/
    └── friday-telugu/
```

---

## Cost-Benefit Analysis

### Upfront Cost

| Item | Cost |
|------|------|
| Skill generation (4 skills × Claude Opus) | ~$2-5 |
| Evaluation runs | ~$1-2 |
| **Total** | **~$5-7** |

### Ongoing Savings

| Before Upskill | After Upskill |
|----------------|---------------|
| 30% of requests → expensive model | 10% of requests → expensive model |
| GLM-4.7-Flash accuracy: ~60% | GLM-4.7-Flash + skill accuracy: ~85% |

**Estimated savings**: 20-40% reduction in API costs

---

## Recommendation

**ADOPT IMMEDIATELY**

This tool directly addresses Friday's agentic network optimization goal:
- Improves GLM-4.7-Flash router performance without retraining
- Low cost ($5-7 one-time)
- Skills are portable and versionable
- Works with our existing infrastructure (api.z.ai endpoint)

**Action Items:**
- [x] Research and document (this analysis)
- [ ] Install upskill package
- [ ] Generate 4 Friday-specific skills
- [ ] Evaluate on GLM-4.7-Flash
- [ ] Integrate into FridayOrchestrator
- [ ] Measure improvement metrics

---

## References

- HuggingFace Blog: https://huggingface.co/blog/upskill
- GitHub Repo: https://github.com/huggingface/upskill
- Agent Skills Spec: https://agentskills.io
- Compatible tools: Claude Code, Cursor, OpenAI Codex

---

## Related Friday Components

If adopted, would affect:
- `orchestrator/friday_orchestrator.py` - Load and use skills
- `orchestrator/glm_router.py` - Router-specific skills
- `config/skills/` - New skill storage directory
- `requirements.txt` - Add upskill package

---

## Comparison: Upskill vs Fine-tuning

| Aspect | Upskill | Fine-tuning |
|--------|---------|-------------|
| Time to implement | Minutes | Hours/Days |
| Cost | $5-7 | $3-50+ |
| Reversibility | Instant | Requires retraining |
| Model-agnostic | Yes | No |
| Requires training data | No | Yes (500+ examples) |
| Updates | Edit markdown | Retrain |

**Verdict**: Use Upskill for quick domain knowledge transfer, fine-tuning for deep personality/behavior changes. Both complement each other.
