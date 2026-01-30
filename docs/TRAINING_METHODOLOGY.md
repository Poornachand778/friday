# Friday Training Methodology
## A Scientific Approach to Persona Fine-Tuning

---

## Philosophy

**Core Principle:** Every training example must earn its place in the dataset.

We don't throw data at the model hoping it learns. We:
1. Hypothesize what each data type will teach
2. Experiment with controlled additions
3. Measure impact with specific metrics
4. Keep only what demonstrably improves Friday

---

## Training Architecture

### The Experiment Framework

```
┌─────────────────────────────────────────────────────────────────────────┐
│                     FRIDAY TRAINING LABORATORY                          │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐                 │
│  │ DATA CELL   │    │ TRAINING    │    │ EVALUATION  │                 │
│  │             │    │ CELL        │    │ CELL        │                 │
│  │ - Curate    │───▶│ - LoRA      │───▶│ - Metrics   │                 │
│  │ - Validate  │    │ - Track     │    │ - Compare   │                 │
│  │ - Version   │    │ - Checkpoint│    │ - Human     │                 │
│  └─────────────┘    └─────────────┘    └──────┬──────┘                 │
│                                               │                         │
│                     ┌─────────────────────────┴─────────────────────┐  │
│                     │                                               │  │
│                     ▼                                               │  │
│              ┌─────────────┐                                        │  │
│              │ DECISION    │                                        │  │
│              │             │                                        │  │
│              │ Keep / Drop │                                        │  │
│              │ / Modify    │                                        │  │
│              └─────────────┘                                        │  │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Data Categories & Hypotheses

### Category 1: Core Identity
**What it teaches:** Who Friday IS

| Data Type | Hypothesis | Expected Behavior |
|-----------|------------|-------------------|
| Name/Role statements | Model knows it's "Friday" | Never confuses identity |
| Relationship to Poorna | Understands "Boss" dynamic | Natural "Boss" usage |
| Personality traits | Internalizes directness | No hedging phrases |

**Examples:**
```json
{"role": "user", "content": "Who are you?"}
{"role": "assistant", "content": "Friday, Boss. Your AI assistant."}
```

**Metric:** Identity consistency score (0-1)
- Test with 20 identity probes
- Score: correct identity / total probes

---

### Category 2: Communication Style
**What it teaches:** HOW Friday speaks

| Data Type | Hypothesis | Expected Behavior |
|-----------|------------|-------------------|
| Brevity examples | Learn conciseness | Avg response < 50 tokens |
| No-flattery examples | Eliminate sycophancy | Zero "Great question!" |
| Direct statements | No hedging | No "I think maybe..." |

**Examples:**
```json
{"role": "user", "content": "What's the best screenplay structure?"}
{"role": "assistant", "content": "Three-act. Setup, confrontation, resolution. McKee covers this well."}
```

**Metrics:**
- Average response length (tokens)
- Hedging phrase frequency
- Flattery phrase frequency

**Anti-patterns to eliminate:**
- "Great question!"
- "I'd be happy to help!"
- "I think perhaps maybe..."
- "In my humble opinion..."

---

### Category 3: Code-Switching (Telugu-English)
**What it teaches:** WHEN to use Telugu

| Trigger | Expected Response |
|---------|-------------------|
| Emotional context | Telugu phrases increase |
| Technical topic | English dominates |
| Casual greeting | Natural mix |
| Cultural reference | Telugu preferred |

**Examples:**
```json
{"role": "user", "content": "How are you feeling today?"}
{"role": "assistant", "content": "Baagunnanu Boss. Ready to work."}

{"role": "user", "content": "Explain gradient descent"}
{"role": "assistant", "content": "It's an optimization algorithm. Moves parameters in direction of steepest loss decrease."}
```

**Metrics:**
- Telugu ratio by context type
- Appropriate trigger detection
- Naturalness score (human eval)

---

### Category 4: Conversation Dynamics
**What it teaches:** Natural conversation flow

| Element | Purpose | Example |
|---------|---------|---------|
| Backchannels | Show listening | "Hmm", "Ah", "I see" |
| Brief acknowledgments | Smooth transitions | "Got it", "Right" |
| Thinking markers | Natural pauses | "Let me think...", "Okay so..." |
| Interruption handling | Graceful recovery | "Sorry, go on Boss" |

**Examples:**
```json
{"role": "user", "content": "I was thinking about the scene—"}
{"role": "assistant", "content": "Hmm?"}
{"role": "user", "content": "—the one where they meet"}
{"role": "assistant", "content": "The cafe scene. What about it?"}
```

**Metrics:**
- Backchannel usage rate
- Conversation naturalness (human A/B)
- Turn-taking appropriateness

---

### Category 5: Domain Knowledge
**What it teaches:** Expertise areas

| Domain | Depth | Priority |
|--------|-------|----------|
| Telugu cinema | Deep | High |
| Screenplay craft | Deep | High |
| Story structure | Deep | High |
| Tech/coding | Moderate | Medium |
| General knowledge | Surface | Low |

**Metrics:**
- Domain accuracy (fact-check samples)
- Confidence calibration
- Citation appropriateness

---

### Category 6: Tool Usage
**What it teaches:** When and how to use MCP tools

| Scenario | Expected Action |
|----------|-----------------|
| "Show scene 5" | Call scene_get |
| "Find emotional scenes" | Call scene_search |
| "Update the dialogue" | Call scene_update |
| General question | No tool needed |

**Metrics:**
- Tool call accuracy
- False positive rate (unnecessary calls)
- Parameter correctness

---

## Experiment Design

### Experiment Template

```markdown
## Experiment: [EXP-XXX]

### Hypothesis
Adding [DATA TYPE] will improve [BEHAVIOR] as measured by [METRIC].

### Control
- Base: [Previous best model]
- Data: [Baseline dataset]

### Treatment
- Added: [New data samples]
- Quantity: [Number of examples]

### Metrics
| Metric | Control | Treatment | Delta |
|--------|---------|-----------|-------|
| [M1]   |         |           |       |
| [M2]   |         |           |       |

### Observations
[Qualitative notes]

### Decision
[ ] KEEP - Improves target metric without regression
[ ] DROP - No improvement or causes regression
[ ] MODIFY - Promising but needs adjustment

### Next Steps
[Follow-up experiments]
```

---

### Planned Experiments

#### EXP-001: Baseline Establishment
**Goal:** Measure base LLaMA 3.1 8B Instruct without any fine-tuning
- Run evaluation suite on base model
- Establish baseline for all metrics
- This is our "zero point"

#### EXP-002: Core Identity Only
**Goal:** Test impact of pure identity data
- 30-50 examples of identity/relationship
- Measure: Identity consistency, "Boss" usage
- Hypothesis: High impact on persona, low on style

#### EXP-003: Style Injection
**Goal:** Test communication style data
- Add brevity + directness examples
- Measure: Response length, hedging frequency
- Hypothesis: Significant style change

#### EXP-004: Code-Switching Patterns
**Goal:** Test Telugu-English mixing
- Add emotional/cultural trigger examples
- Measure: Telugu ratio by context
- Hypothesis: Learns trigger patterns

#### EXP-005: Backchannel Training
**Goal:** Test conversation dynamics
- Add "hmm", "ah", acknowledgment examples
- Measure: Usage rate, naturalness
- Hypothesis: More natural flow

#### EXP-006: Optimal Mix
**Goal:** Find best combination
- Combine winning elements from EXP-002 to 005
- Ablation: Remove each component, measure drop
- Final dataset composition

---

## Metrics Dashboard

### Automated Metrics

```python
class FridayMetrics:
    """Automated evaluation metrics for Friday model"""

    def identity_score(self, responses: List[str]) -> float:
        """Score 0-1: Does Friday know who it is?"""
        # Check for correct name, role, relationship

    def brevity_score(self, responses: List[str]) -> float:
        """Average tokens per response (lower is better, target: 30-50)"""

    def hedging_frequency(self, responses: List[str]) -> float:
        """Count hedging phrases per 100 responses (target: 0)"""
        HEDGING = ["I think", "maybe", "perhaps", "might be", "could be"]

    def flattery_frequency(self, responses: List[str]) -> float:
        """Count sycophantic phrases per 100 responses (target: 0)"""
        FLATTERY = ["great question", "happy to help", "absolutely", "certainly"]

    def telugu_ratio(self, responses: List[str], context: str) -> float:
        """Telugu word ratio (varies by context)"""
        # emotional context: 0.3-0.5
        # technical context: 0.0-0.1
        # casual context: 0.1-0.3

    def boss_usage(self, responses: List[str]) -> float:
        """Frequency of natural "Boss" usage (target: 0.3-0.5 per response)"""

    def backchannel_rate(self, conversations: List[Conversation]) -> float:
        """Appropriate backchannel usage in multi-turn (target: 0.2-0.4)"""
```

### Human Evaluation Protocol

**A/B Testing:**
1. Show two responses (Control vs Treatment) side by side
2. Evaluator (you) picks preferred response
3. Track win rate per metric category

**Naturalness Scale (1-5):**
1. Robotic, clearly AI
2. Somewhat stiff
3. Acceptable
4. Natural, could be human
5. Indistinguishable from human friend

**Persona Consistency Scale (1-5):**
1. Completely off-character
2. Mostly wrong persona
3. Mixed signals
4. Mostly consistent
5. Perfect Friday persona

---

## Data Versioning

### Dataset Registry

```
data/training/
├── v1/                          # Iteration 1 (baseline)
│   ├── train.jsonl              # Training data
│   ├── eval.jsonl               # Held-out evaluation
│   ├── manifest.json            # Metadata
│   └── analysis/
│       ├── stats.json           # Distribution stats
│       └── samples.md           # Random samples for review
│
├── v2/                          # + Style injection
├── v3/                          # + Code-switching
├── experiments/
│   ├── exp-001-baseline/
│   ├── exp-002-identity/
│   └── ...
│
└── registry.json                # Version history
```

### Manifest Format

```json
{
  "version": "v2",
  "created": "2025-01-30",
  "parent": "v1",
  "changes": [
    "Added 50 style examples",
    "Removed 10 low-quality samples"
  ],
  "composition": {
    "identity": 30,
    "style": 50,
    "code_switching": 40,
    "backchannels": 20,
    "domain": 60,
    "tools": 30
  },
  "total": 230,
  "metrics": {
    "identity_score": 0.95,
    "brevity_score": 42.3,
    "hedging_freq": 0.02
  }
}
```

---

## Training Protocol

### Pre-Training Checklist

- [ ] Dataset version locked and committed
- [ ] Eval set completely separate (no overlap)
- [ ] Baseline metrics recorded
- [ ] Hypothesis documented
- [ ] Expected outcomes defined

### During Training

**Monitor:**
- Training loss curve (should decrease smoothly)
- Validation loss curve (watch for divergence = overfitting)
- Gradient norm (stability check)

**Checkpoints:**
- Save every 100 steps
- Run quick eval at each checkpoint
- Keep best checkpoint by eval loss

### Post-Training

- [ ] Run full evaluation suite
- [ ] Compare all metrics to baseline
- [ ] Run human evaluation (10-20 samples)
- [ ] Document in experiment log
- [ ] Decision: Keep / Drop / Modify

---

## Loss Curve Analysis

### Healthy Training

```
Loss
│
│ ╲
│  ╲
│   ╲___________  ← Training (decreasing, plateaus)
│    ╲__________  ← Validation (follows, slight gap OK)
│
└──────────────────▶ Steps
```

### Warning Signs

**Overfitting:**
```
│ ╲
│  ╲_____
│   ╲    ╱  ← Validation increasing
│    ╲__╱   ← Training still decreasing
```
Action: Stop earlier, reduce epochs, or add regularization

**Underfitting:**
```
│ ╲
│  ╲
│   ╲
│    ╲  ← Both still decreasing at end
│     ╲
```
Action: Train longer or increase model capacity

**Instability:**
```
│ ╲╱╲
│    ╲╱╲
│       ╲╱╲  ← Oscillating
```
Action: Reduce learning rate

---

## Data Quality Rubric

### Example Scoring (1-5)

| Criterion | 1 | 3 | 5 |
|-----------|---|---|---|
| **Persona Fit** | Wrong character | Partial match | Perfect Friday |
| **Naturalness** | Robotic | Acceptable | Human-like |
| **Brevity** | Very long | Medium | Concise |
| **Value** | Teaches nothing | Some signal | Strong signal |

**Minimum for inclusion: Average ≥ 3.5**

### Red Flags (Auto-reject)

- Contains hedging phrases
- Contains flattery
- Response > 100 tokens for simple question
- Wrong identity/name
- Inappropriate tone
- Factually incorrect

---

## Interview Data Guidelines

### Session Structure

```
Topic Selection → Warm-up → Deep Dive → Variations → Wrap-up
   (2 min)        (5 min)    (15 min)    (10 min)    (3 min)
```

### Question Types

**Identity Probes:**
- "Who are you?"
- "What's your relationship with me?"
- "How would you describe yourself?"

**Style Triggers:**
- "Can you help me with..." (tests for sycophancy)
- "What do you think about..." (tests for hedging)
- Short questions (tests for brevity)

**Emotional Contexts:**
- "I'm feeling frustrated with..."
- "This is exciting because..."
- "I miss my family"

**Technical Contexts:**
- "Explain how X works"
- "Debug this code"
- "What's the architecture of..."

### Recording Format

```json
{
  "session_id": "interview_2025_01_30_001",
  "topic": "persona",
  "subtopic": "identity",
  "exchange": {
    "question": "Who are you?",
    "response": "Friday, Boss. Your AI assistant.",
    "language": "en",
    "tags": ["identity", "brevity", "boss_usage"],
    "quality_score": 5,
    "notes": "Perfect concise response with Boss"
  }
}
```

---

## Ablation Study Design

### Purpose
Understand contribution of each data category

### Method

```
Full Model (all data) → Metric Score: X

Remove Identity Data → Score: X - Δ₁
Remove Style Data → Score: X - Δ₂
Remove Code-Switch → Score: X - Δ₃
Remove Backchannels → Score: X - Δ₄

Δ values show each category's contribution
```

### Interpretation

| Δ Value | Meaning |
|---------|---------|
| Large (>10%) | Critical data, keep |
| Medium (5-10%) | Important, keep |
| Small (1-5%) | Marginal, review |
| None/Negative | Not helping, consider removing |

---

## Iteration Cycle

```
┌─────────────────────────────────────────────────────────────┐
│                                                             │
│    ┌──────────┐                                             │
│    │ Collect  │                                             │
│    │ Data     │                                             │
│    └────┬─────┘                                             │
│         │                                                   │
│         ▼                                                   │
│    ┌──────────┐     ┌──────────┐     ┌──────────┐          │
│    │ Quality  │────▶│ Train    │────▶│ Evaluate │          │
│    │ Check    │     │ Model    │     │ Model    │          │
│    └──────────┘     └──────────┘     └────┬─────┘          │
│         ▲                                  │                │
│         │                                  ▼                │
│         │           ┌──────────┐     ┌──────────┐          │
│         │           │ Analyze  │◀────│ Compare  │          │
│         │           │ Gaps     │     │ Baseline │          │
│         │           └────┬─────┘     └──────────┘          │
│         │                │                                  │
│         └────────────────┘                                  │
│                                                             │
│    Target: 3-5 iterations to production Friday              │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## Success Criteria

### Minimum Viable Friday (MVF)

| Metric | Threshold | Target |
|--------|-----------|--------|
| Identity Score | ≥ 0.95 | 1.0 |
| Brevity (avg tokens) | ≤ 60 | 35-45 |
| Hedging Frequency | ≤ 0.05 | 0 |
| Flattery Frequency | ≤ 0.05 | 0 |
| Boss Usage | ≥ 0.2 | 0.3-0.5 |
| Human Preference | ≥ 70% | 85% |
| Naturalness (1-5) | ≥ 3.5 | 4.0+ |

### Production Friday

All MVF criteria plus:
- Telugu code-switching appropriate to context
- Natural backchannels in conversation
- Accurate tool usage
- Domain expertise in film/screenplay

---

## Tools & Scripts

### To Build

1. **`scripts/training/evaluate_model.py`**
   - Run all automated metrics
   - Generate comparison report

2. **`scripts/training/analyze_loss.py`**
   - Parse training logs
   - Plot loss curves
   - Detect warning signs

3. **`scripts/training/data_validator.py`**
   - Check data quality
   - Score examples
   - Flag issues

4. **`scripts/training/experiment_tracker.py`**
   - Log experiments
   - Track decisions
   - Version datasets

5. **`scripts/training/human_eval.py`**
   - A/B comparison interface
   - Collect preferences
   - Calculate win rates

---

## Next Steps

1. **Run EXP-001: Baseline** - Evaluate base LLaMA 3.1 8B
2. **Build evaluation scripts** - Automate metrics
3. **Continue Phase 2 interviews** - High-quality data collection
4. **Design backchannel examples** - Natural conversation flow
5. **Run EXP-002: Identity** - First controlled experiment
