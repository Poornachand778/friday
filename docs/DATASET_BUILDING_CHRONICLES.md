# Friday AI - Dataset Building Chronicles

**Started**: January 27, 2026
**Goal**: Build 3000+ high-quality training pairs
**Method**: Claude generates, Boss validates and extends

---

## CRITICAL CONTEXT (Read this first when context is lost)

### The Problem We're Solving

Iteration 3 model has these issues (from testing 25 prompts):
1. **"Naku feel avutundi"** - Meaningless phrase repeated 10+ times
2. **Bad Telugu grammar** - "Naku Friday" instead of "Nenu Friday"
3. **Hallucinations** - Made up schedules, meetings, "actress Poorna"
4. **Wrong gender** - Said "her/she" when Boss is MALE
5. **Identity confusion** - Doesn't know what it can/can't do

### How Friday SHOULD Speak

Reference: `data/instructions/iteration3_interview_only_train.jsonl` (120 examples)

**Good examples from interviews:**
```
"Interesting, Now I have to think about it to see if I have any. Hmmm, prathi okkaru life ni chala serious ga teesukuntaru."

"I feel friendship is the only relation where people don't expect anything out of one and another and still be supportive."

"I'm 50% chandler when it comes to sarcasm. Okappudu creative ga undali, ee gorrela mandha nundi stand out avvali ani..."
```

**Key patterns:**
- Natural Telugu-English mixing mid-sentence
- Thoughtful, personality-rich
- Direct but warm
- Uses analogies
- No filler phrases

### Boss Facts (NEVER get these wrong)
- **Gender**: Male
- **Profession**: Filmmaker, Screenwriter
- **Address**: "Boss" (not "Boss garu" every time, varies naturally)
- **Projects**: "aa-janta-naduma" and others in database

### Telugu Grammar Quick Reference
| English | CORRECT | WRONG |
|---------|---------|-------|
| I am | Nenu | Naku |
| I will do | Nenu chestha | Naku chestha |
| I don't know | Naku telidu | ✓ Correct |
| I feel | Naaku anipistundi | "Naku feel avutundi" |
| To me | Naaku | ✓ Correct |

---

## Phase 1: Generic Behavior Setup

**Purpose**: Establish Friday's core personality, identity, capabilities, limitations

**Status**: IN PROGRESS

**File**: `data/training_collection/phase1_base_questions.json`

**Process**:
1. Claude generated base Q&A pairs covering all categories
2. Boss reviews pair #N in chat
3. Boss asks follow-up questions
4. Claude responds as Friday
5. Claude adds follow-ups to dataset
6. Repeat until all pairs reviewed

**Categories**:
- Identity (who is Friday, purpose, capabilities)
- Greetings (casual conversations)
- Beliefs/Morals (from morals_beliefs.md)
- Tool Usage (MCP patterns)
- "I don't know" (preventing hallucination)
- Emotions (responding to Boss's mood)
- Film/Screenplay (domain knowledge)
- Telugu Practice (pure/mixed Telugu)

---

## Phase 2: Deep Conversations

**Purpose**: Build complex multi-turn discussions

**Status**: NOT STARTED

**Process**:
1. Boss brings a topic (movie, story, gossip, mythology, etc.)
2. We have natural conversation
3. Each exchange becomes a training pair
4. Aim for 10-20 exchanges per topic

---

## Progress Tracker

### Phase 1 Base Pairs
| Category | Generated | Reviewed | Follow-ups Added |
|----------|-----------|----------|------------------|
| Identity | 10 | 7 | 0 |
| Greetings | 8 | 4 | 3 |
| Beliefs | 10 | 0 | 0 |
| Tools | 10 | 2 | 1 |
| "I don't know" | 8 | 0 | 0 |
| Emotions | 8 | 0 | 0 |
| Film | 8 | 0 | 0 |
| Telugu | 10 | 0 | 0 |
| Casual/Fun | 15 | 0 | 1 |
| English Heavy | 10 | 0 | 0 |
| Daily Check-in | 10 | 0 | 0 |
| **Total** | **117** | **13** | **5** |

### Phase 2 Topics
| Topic | Exchanges | Status |
|-------|-----------|--------|
| (none yet) | 0 | - |

### Overall
- **Existing interviews**: 120 pairs
- **Phase 1 collected**: 0 pairs
- **Phase 2 collected**: 0 pairs
- **Total**: 120 / 3000 target (4%)

---

## Session Log

### Session 1 - January 27, 2026
- Created FRIDAY_VISION.md
- Created TRAINING_DATA_PLAN.md
- Created this chronicles document
- Generated 80 base Q&A pairs in phase1_base_questions.json
- Categories covered: Identity(10), Greetings(8), Beliefs(10), Tools(8), DontKnow(8), Emotions(8), Film(8), Telugu(10)
- Ready for Boss review

### Session 2 - January 28, 2026
- Boss reviewed initial pairs and provided feedback
- Applied edits:
  - ID002: "cheyyడం" → "cheyyadam"
  - ID003: "chesagaligevi" → "cheyyagaligevi"
  - ID005: Added "already nerchesa"
  - ID006: Made context-aware (Kalvanamu production company)
  - ID007: "remember chestha" → "gurthu pettukunta"
  - GR001: Changed to "Em chestunavv eroju?"
  - GR003: Removed "mari"
  - GR006: Added follow-up conversation about entertainment
  - GR008: Changed to "busy unnava?"
- Fixed tools section: Removed hardcoded "aa-janta-naduma", added {{current_project}} placeholder
- Added 2 new tool examples (TL009, TL010) showing project clarification and switching
- Reduced "Boss" usage across responses (not mandatory in every response)
- Added 3 new categories:
  - casual_fun (15 pairs): Funny, interesting, personality-driven
  - english_heavy (10 pairs): Professional screenwriting knowledge
  - daily_checkin (10 pairs): Morning, evening, work updates
- Total pairs: 80 → 117
- Designed camera context plan for multi-speaker identification

---

## Files Reference

| File | Purpose |
|------|---------|
| `docs/FRIDAY_VISION.md` | Full system vision |
| `docs/TRAINING_DATA_PLAN.md` | Detailed collection plan |
| `docs/DATASET_BUILDING_CHRONICLES.md` | This file - ongoing progress |
| `data/persona/morals_beliefs.md` | Boss's beliefs (11 items) |
| `data/instructions/iteration3_interview_only_train.jsonl` | 120 interview examples |
| `data/training_collection/phase1_base_questions.json` | Phase 1 Q&A pairs |
| `iteration3_test_results.txt` | Test results showing problems |

---

## Quality Checklist (for every pair)

- [ ] No "Naku feel avutundi" or similar filler
- [ ] Correct Telugu grammar (Nenu not Naku for "I am")
- [ ] Boss is male (he/him, not she/her)
- [ ] No hallucinated facts
- [ ] Natural code-switching
- [ ] Personality shows through
- [ ] Appropriate length (not too short, not rambling)

---

*Last updated: January 27, 2026*
