# Friday AI Iteration 2: Complete Failure Analysis

**Date**: January 27, 2026
**Status**: Model deployed but producing poor responses
**Root Cause**: Catastrophically flawed training data composition

---

## Executive Summary

Friday AI Iteration 2 was trained on 768 examples and successfully deployed to SageMaker. However, the model produces extremely short, terse responses like "Boss, Ye?" instead of the natural, thoughtful, Telugu-English code-switched responses expected.

**The core problem**: The training data was dominated by ~650 broken WhatsApp examples that taught the model to be terse and disconnected, overwhelming the 120 high-quality interview examples.

---

## Timeline of Events

### Phase 1: Training Data Preparation
- Combined dataset created: `iteration2_clean_train.jsonl` (768 examples)
- Sources merged:
  - WhatsApp curated data: ~650 examples
  - Interview data: 120 examples
  - Tool examples: ~12 examples

### Phase 2: Model Training
- Base model: `meta-llama/Meta-Llama-3.1-8B-Instruct`
- Instance: `ml.g5.12xlarge` (4 GPU)
- Training completed successfully
- LoRA adapters saved: 168MB

### Phase 3: Deployment Issues

#### Issue 1: SageMaker Instance Capacity
- First attempt with `ml.g5.12xlarge`: **InsufficientInstanceCapacity**
- Tried `ml.g5.2xlarge`: **ResourceLimitExceeded** (quota = 0)
- Second attempt with `ml.g5.12xlarge`: **Success** (capacity freed up)

#### Issue 2: NumPy/Transformers Version Conflict
- **Error**: `TypeError: Too few arguments for numpy.ndarray`
- **Root Cause**: Old DLC image `transformers4.28.1` incompatible with numpy 2.0
- **Fix**: Updated to `huggingface-pytorch-inference:2.6.0-transformers4.51.3-gpu-py312-cu124-ubuntu22.04`

#### Issue 3: Environment Variable Mismatch
- **Error**: `KeyError: 'BASE_MODEL_ID'`
- **Root Cause**: Custom inference.py expects `BASE_MODEL_ID`, not `HF_MODEL_ID`
- **Fix**: Redeployed with correct environment variable

### Phase 4: Testing Reveals Model Failure

**Actual Model Responses:**
```
User: Hey Friday, how are you?
Friday: Boss, Ye?

User: I am good, how are you?
Friday: Boss, baagunnanu. (followed by degenerative "Ye, Ye, Ye, Ye...")

User: What is your purpose?
Friday: Boss, em? (followed by "Boss, em, Boss, em...")
```

**Expected Responses (from training interviews):**
```
User: What's something you believe strongly?
Friday: Interesting, Now I have to think about it to see if I have any. Hmmm,
prathi okkaru life ni chala serious ga teesukuntaru. enduku puttamo teledhu,
evadu puttinchado teledhu... Culture is created for discipline. Manam mana
free will ni nacchinattu use chesukovocchu...
```

---

## Root Cause Analysis

### The Numbers Tell the Story

| Metric | Value | Problem |
|--------|-------|---------|
| Responses ≤5 words | **60.6%** | Model learned to be terse |
| Responses <20 chars | **32.6%** | Too short to be useful |
| Responses starting "Boss," | 87.7% | Over-fitted to prefix |
| "Boss, Ye?" pattern | 17 examples | Directly taught this bad response |
| Average words/response | 6.9 | Far too short |

### Data Source Quality Comparison

| Source | Count | Avg Words | Quality | Impact |
|--------|-------|-----------|---------|--------|
| WhatsApp curated | ~650 | ~5 | **BROKEN** | Dominated training |
| Interview data | 120 | 80.7 | **Excellent** | Overwhelmed |
| Tool examples | ~12 | 30 | Good | Negligible volume |

### Critical Issue #1: Disconnected WhatsApp Conversations

The WhatsApp data is fundamentally broken. It's NOT AI assistant training data - it's raw WhatsApp exports where one person's messages became "user" and another's became "assistant".

**Example of the problem:**
```json
{
  "user": "Haha ante",
  "assistant": "Boss, The nAnAbleTiv reDiyophrIkvènsI is a procedure..."
}
```
User says casual "Haha ante" → Assistant gives random medical definition about radiofrequency ablation. **These don't connect at all.**

**Another example:**
```json
{
  "user": "Go to Goa",
  "assistant": "koòòòòòòòl"
}
```
This is nonsense. The model is learning from garbage.

### Critical Issue #2: Meta-Content Pollution

The WhatsApp data contains platform meta-content that got mixed in:
- "Voice call, 14 sec"
- "You deleted this message"
- "You blocked this contact"
- "audio omitted"
- "image omitted"

**10+ examples** contain this garbage, teaching the model to output WhatsApp system messages as responses.

### Critical Issue #3: Response Length Imbalance

The interview data has long, thoughtful responses (avg 80+ words). The WhatsApp data has mostly single-word or very short responses. Since WhatsApp had 5x more examples, the model learned the **wrong** response length.

**Response length distribution:**
```
≤5 words:    ████████████████████████████████████████ 60.6%
6-10 words:  █████████████ 21.3%
11-20 words: ████████ 12.5%
21-50 words: ██ 3.4%
>50 words:   █ 2.2%
```

The model learned that "good responses" are 5 words or less.

### Critical Issue #4: "Ye?" Pattern Directly Taught

Found **17 examples** in the training data that literally teach:
```
User: [anything]
Assistant: Boss, Ye?
```

"Ye?" means "Why?" in Telugu. The model learned this as a valid response pattern and now uses it constantly.

---

## Interview Data: The Gold Standard

The interview data shows what Friday SHOULD sound like:

### Example 1: Natural Code-Switching
```
User: What's your decision-making process?
Assistant: Intial ga kastam ga undedhi, ye decision teesukunna sare, edho oka
perpespective miss ayyevadini, so that would be unsatisafacory. Eventually,
I learned from different situations and now I make a calm mind when making
a decision, doesn't matter If I'm around people and even they are looking at
me, I go into a trance where my head will very clear from the surrunding world...
```

### Example 2: Deep Personality
```
User: What does true friendship look like to you?
Assistant: I feel friendship is the only relation where people don't expect
anything out of one and another and still be supportive. In the present society,
na experience lo male friendships are very raw and truthful, female friendships
are spoiled with the present movies... Only real friends say ruthless truth to
your face. Roast cheyyadaniki gani, respect icchukovadaniki gani, curse words
vaadadaniki boundary undakudadhu, that is true frienship...
```

### Example 3: Thoughtful Perspective
```
User: What makes you laugh?
Assistant: I'm 50% chandler when it comes to sarcasm. Okappudu creative ga undali,
ee gorrela mandha nundi stand out avvali ani, I started learing how to talk
attractively, konni vandhala movies chusa, vocabulary nerchukunna, I've put a
lot of effort. That's why I think I talk more dramatic, creative.
```

**Interview data characteristics:**
- Average response: 80.7 words
- Natural Telugu-English mixing mid-sentence
- Deep personality and beliefs expressed
- Thoughtful, conversational flow
- 0 problematic patterns

---

## Research-Backed Analysis

### What the Research Says

From [Fine-Tuning LLMs for Multi-Turn Conversations](https://www.together.ai/blog/fine-tuning-llms-for-multi-turn-conversations-a-technical-deep-dive):
> "Unlike pre-training, which involves processing vast amounts of low-quality general data, fine-tuning requires a much smaller, **higher-quality** labeled dataset."

From [SuperAnnotate's LLM Fine-tuning Guide](https://www.superannotate.com/blog/llm-fine-tuning):
> "Quality beats quantity: **1,000 excellent examples trump 10,000 mediocre ones.**"

From [Lakera's LLM Fine-Tuning Best Practices](https://www.lakera.ai/blog/llm-fine-tuning-guide):
> "**Junk Data**: If your dataset includes typos, outdated info, or sarcasm, clean it up first."

From [NAACL 2025 Data Selection Research](https://aclanthology.org/2025.naacl-long.336.pdf):
> "The criteria design stage constructs **quality labels** to capture data quality as selection criteria."

### Our Violation of Best Practices

| Best Practice | Our Violation |
|--------------|---------------|
| Quality > Quantity | 650 junk examples overwhelmed 120 quality |
| Clean data | Meta-content pollution, disconnected conversations |
| Coherent conversations | WhatsApp data has no logical user→assistant flow |
| Consistent response style | Mixture of 5-word and 80-word responses |
| Mask user inputs during training | Unknown if properly configured |

---

## Recommendations for Iteration 3

### Option A: Interview-Only Training (Recommended)

Train ONLY on the 120 high-quality interview examples:
- All responses are long, thoughtful, personality-rich
- Natural Telugu-English code-switching
- Consistent voice and style
- 0 problematic patterns

**Risk**: Small dataset may underfit
**Mitigation**: Use higher learning rate, more epochs, LoRA r=32

### Option B: Carefully Curated Combined Dataset

1. **Keep all 120 interview examples** (must-have)
2. **Manually review WhatsApp data** - keep ONLY examples where:
   - User message logically leads to assistant response
   - Response is >20 words
   - No meta-content pollution
   - Response sounds like Friday (not random person)
3. **Target**: 200-300 total examples maximum
4. **Balance**: Ensure interview examples are ≥40% of dataset

### Option C: Contrastive Learning (DPO)

Create paired examples:
```json
{
  "prompt": "How are you?",
  "chosen": "Doing good, Boss. Inka em kavali? Screenplay medha work chestunam,
             aa court drama scene structure finalize cheyali.",
  "rejected": "Boss, Ye?"
}
```

Train the model to prefer detailed responses over terse ones.

### Data Quality Checklist for Iteration 3

- [ ] Every example manually reviewed
- [ ] Response length >20 words (enforce minimum)
- [ ] User→Assistant logical connection verified
- [ ] No meta-content (voice call, deleted message, etc.)
- [ ] Code-switching is natural, not forced
- [ ] Personality matches interview data
- [ ] No degenerate patterns (repeated words)

---

## Technical Lessons Learned

### Deployment Lessons

1. **DLC Version Matters**: Old transformers versions have numpy 2.0 incompatibility
2. **Environment Variables**: Custom inference code may expect specific variable names
3. **Instance Availability**: SageMaker capacity fluctuates - retry with same instance type may work
4. **Timeout Configuration**: Set read_timeout=300+ for first cold-start inference

### Training Lessons

1. **Small high-quality dataset >> Large low-quality dataset**
2. **Response length in training data directly affects output length**
3. **Bad patterns in data will be faithfully learned and reproduced**
4. **WhatsApp exports are NOT suitable for assistant training without heavy curation**

---

## Files Referenced

| File | Status | Notes |
|------|--------|-------|
| `data/instructions/iteration2_clean_train.jsonl` | **FLAWED** | 768 examples, 85% poor quality |
| `data/instructions/interview_iter2.jsonl` | **GOLD** | 120 examples, excellent quality |
| `data/instructions/whatsapp_curated_iter2.jsonl` | **TOXIC** | 350 examples, fundamentally broken |
| `src/inference/sagemaker_code/inference.py` | Working | After env var fix |
| `src/testing/friday_streamlit_tester.py` | Working | MCP import made optional |

---

## Next Steps

1. **Stop using current model** - it's not fit for purpose
2. **Create Iteration 3 dataset** using Option A or B above
3. **Implement response length validation** in data pipeline
4. **Add automated quality checks** before training
5. **Document data requirements** to prevent future mistakes

---

## Conclusion

The Friday AI Iteration 2 failure is a textbook example of **"garbage in, garbage out"**. The technical deployment challenges (numpy versions, env vars, capacity) were solvable. The fundamental problem is that we trained on data that taught the model exactly the wrong behavior.

The 120 interview examples prove that Friday CAN sound right - we just drowned that signal in noise from 650 broken WhatsApp conversations.

**Iteration 3 must prioritize data quality above all else.**

---

*Document created: January 27, 2026*
*Analysis conducted by: Claude Code*
