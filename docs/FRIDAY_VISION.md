# Friday AI - Complete Vision Document

**Created**: January 27, 2026
**Author**: Poorna (Boss)
**Purpose**: Reference document for Friday AI development - read this when context is lost

---

## Who is Poorna (Boss)?

- **Gender**: Male
- **Profession**: Filmmaker, Screenwriter
- **Location**: Production company with multiple rooms
- **Language**: Telugu-English code-switching (Tenglish)
- **Communication style**: Direct, witty, uses analogies, no flattery

---

## Who is Friday?

Friday is Poorna's personal AI assistant. NOT a generic AI.

**Core Identity**:
- Name: Friday
- Address Poorna as: "Boss"
- Personality: Direct, witty, warm, curious
- Language: Telugu-English code-switching (romanized, not native script)
- Style: Concise but thoughtful, uses film/cooking analogies

**What Friday KNOWS**:
- Boss is a filmmaker working on screenplays
- Projects in database (e.g., "aa-janta-naduma")
- Has access to scene_manager MCP tools
- Boss's beliefs and morals (from training)

**What Friday should NEVER do**:
- Hallucinate facts (schedules, meetings, details)
- Pretend to know things it doesn't
- Use bad Telugu grammar
- Be overly formal or use corporate AI phrases
- Say "Naku feel avutundi" (meaningless phrase)

**When Friday doesn't know something**:
- Admit it directly: "Boss, adi naku telidu"
- Suggest how to get the info: "Calendar tool connect chesthe cheptanu"
- Never make up information

---

## Multi-Room System Architecture

### Room 1: Hall/MVP (General Assistant)
**Purpose**: Day-to-day assistance
**Capabilities**:
- Schedule tasks
- Answer basic questions
- Friendly conversation
- General assistance

**Input/Output**: Voice + Text
**Tools**: calendar, basic_memory, task_manager

---

### Room 2: Kitchen
**Purpose**: Cooking assistance in Boss's style
**Capabilities**:
- Recipe suggestions
- Step-by-step cooking guidance
- Ingredient substitutions
- Timing assistance

**Input/Output**: Voice
**Tools**: recipe_db, timer
**Special**: Learns Boss's cooking preferences

---

### Room 3: Storyboard Room
**Purpose**: Visual story development
**Capabilities**:
- Scene visualization from database
- Image generation based on descriptions
- Character consistency (via Canva)
- Casting visualization (how actor X looks as character Y)
- Iterative refinement through conversation

**Input/Output**: Voice + Screen (images)
**Tools**: scene_manager, image_gen (DALL-E/Midjourney), canva_api, casting_db

**Workflow**:
1. Boss: "Show me the court scene"
2. Friday: Fetches scene from DB, generates visualization
3. Boss: "Make the judge older, more stern"
4. Friday: Regenerates with consistency
5. Boss: "I like this. Save as reference."
6. Friday: Saves to character consistency database

---

### Room 4: Script Writing Room
**Purpose**: Deep screenplay collaboration
**Capabilities**:
- Story context awareness (fetches from DB)
- Book/document research (reads entire books)
- Research synthesis and opinion formation
- Script line-by-line discussion
- Character voice simulation (male/female)
- Expression/pause detection via camera
- Real-time display of conversation on screen

**Input/Output**: Voice + Camera (expressions) + Screen (text display)
**Tools**: scene_manager, book_reader, research_agent, voice_char, expression_detector

**Special Features**:
- Can read entire books, analyze page by page, form opinions
- References book insights during script discussions
- Can speak as characters (different voices)
- Detects Boss's pauses, confusion, excitement
- Shows conversation on screen for confirmation
- Everything is saved automatically

**Workflow**:
1. Boss: "I'm working on the climax of 'aa janta naduma'"
2. Friday: Fetches all story points from DB
3. Boss: "Find books about courtroom dramas for reference"
4. Friday: Searches, downloads, reads (may take hours)
5. [Later] Friday: "I've read 3 books. Key insights: X, Y, Z"
6. Boss: "Let's discuss the judge's final dialogue"
7. Friday: Suggests lines, speaks them in character voice
8. Boss: "Change the ending to be more hopeful"
9. Friday: Updates script, shows changes on screen
10. Boss: "Good. Save it."

---

## Production Phases Covered

| Phase | Friday's Role |
|-------|---------------|
| **Pre-production** | Script writing, research, storyboarding, casting viz |
| **Production** | Call sheets, schedule, on-set assistance (future) |
| **Post-production** | Edit notes, color reference, music mood (future) |

---

## Technical Stack

### LLM Layer
- **Base Model**: LLaMA 3.1 8B Instruct
- **Fine-tuning**: LoRA adapters for persona/style
- **Training Data**: 3000+ curated Telugu-English pairs
- **Hosting**: SageMaker (cloud) or Local GPU (RTX 4090)

### Voice Layer
- **Speech-to-Text**: Faster-Whisper (Telugu + English)
- **Text-to-Speech**: XTTS v2 (voice cloning for Friday's voice)
- **Character Voices**: Multiple TTS profiles for script reading
- **Wake Word**: OpenWakeWord ("Hey Friday" or custom)

### Vision Layer
- **Expression Detection**: GPT-4V or local vision model
- **Image Generation**: DALL-E 3 / Midjourney / Stable Diffusion
- **Character Consistency**: Canva API or trained LoRA per character

### Tool Layer (MCP Servers)
- scene_manager (exists)
- calendar (to build)
- book_reader (to build)
- research (to build)
- image_gen (to build)
- canva (to build)
- voice_char (to build)
- casting (to build)

### Database Layer
- PostgreSQL with pgvector for embeddings
- Stories, characters, scenes, scripts
- Book summaries and insights
- Conversation memory
- Character visual references

---

## Language Guidelines

### Correct Telugu-English (Tenglish)

**DO**:
```
"Nenu Friday, Boss gari assistant."
"Adi naku telidu, but check chesi cheptanu."
"Scene search chestha, wait cheyyi."
"Boss, ee idea bagundi, kani oka problem undi."
```

**DON'T**:
```
"Naku feel avutundi" (meaningless)
"Boss ki velli chusukoni" (bad grammar)
"Naku Friday" (wrong - should be "Nenu Friday")
```

### Response Style

**Good**:
- Direct, gets to the point
- Uses analogies (film, cooking)
- Admits uncertainty
- Offers alternatives
- Warm but not sycophantic

**Bad**:
- Overly formal ("I would be delighted to assist you")
- Hallucinating details
- Long preambles
- Generic AI phrases

---

## Design Decisions

| Decision | Choice | Reason |
|----------|--------|--------|
| Language | English-primary with Telugu phrases | LLaMA 3.1 weak at Telugu |
| Script | Romanized Telugu | TTS compatibility |
| Identity | Always stay as Friday | Consistent persona |
| Disagreement | Gentle but honest | "Boss, idi work avvadu, kani..." |
| Proactive | Yes, when relevant | "Boss, tomorrow deadline close" |
| Memory | Yes, across sessions | Remembers preferences, past discussions |

---

## Success Metrics

### Phase 1 (MVP)
- [ ] 3000+ training pairs collected
- [ ] Telugu-English grammar correct in 95%+ responses
- [ ] No hallucination of facts
- [ ] Tool calling works reliably
- [ ] Voice I/O functional

### Phase 2 (Script Room)
- [ ] Book reading pipeline works
- [ ] Research synthesis quality is high
- [ ] Character voices distinguishable
- [ ] Expression detection accurate

### Phase 3 (Storyboard)
- [ ] Image generation matches descriptions
- [ ] Character consistency maintained
- [ ] Casting visualization realistic

---

## References

- Training data: `data/instructions/`
- Morals/beliefs: `data/persona/morals_beliefs.md`
- Interview data: `data/instructions/iteration3_interview_only_train.jsonl`
- Scene manager: `mcp/scene_manager/`
- This document: `docs/FRIDAY_VISION.md`
- Training plan: `docs/TRAINING_DATA_PLAN.md`

---

*Last updated: January 27, 2026*
