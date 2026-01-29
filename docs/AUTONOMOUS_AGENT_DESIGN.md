# Friday AI - Autonomous Agent Design

**Date:** January 2026
**Status:** Design Complete

---

## 1. Overview

The Autonomous Agent system allows Friday to proactively analyze scripts during idle time and generate improvement suggestions. These suggestions are stored in a "Friday Suggestions" backlog for the Boss to review when convenient.

### Key Principles

1. **Non-intrusive**: Agent works in background, never interrupts active work
2. **Helpful, not pushy**: Waits for Boss to ask before presenting suggestions
3. **Context-aware**: Understands project state, recent edits, and priorities
4. **Persistent**: Suggestions stored in database, survive restarts

---

## 2. Database Schema

### Friday Suggestions Table

```sql
-- Add to db/screenplay_schema.py
CREATE TABLE friday_suggestions (
    id SERIAL PRIMARY KEY,
    project_id INTEGER REFERENCES screenplay_projects(id) NOT NULL,
    scene_id INTEGER REFERENCES screenplay_scenes(id),  -- NULL = project-level

    -- Suggestion content
    suggestion_type VARCHAR(64) NOT NULL,  -- See types below
    title VARCHAR(256) NOT NULL,
    description TEXT NOT NULL,

    -- Proposed change (if applicable)
    proposed_change TEXT,  -- Draft scene/dialogue content
    affected_scenes INTEGER[],  -- Scene IDs affected by this suggestion

    -- Priority & categorization
    priority INTEGER DEFAULT 3,  -- 1=critical, 2=high, 3=medium, 4=low, 5=minor
    category VARCHAR(64),  -- screenplay, dialogue, pacing, character, structure
    confidence FLOAT DEFAULT 0.5,  -- 0-1, agent's confidence in the suggestion

    -- Status tracking
    status VARCHAR(32) DEFAULT 'pending',  -- pending, discussed, accepted, rejected, deferred

    -- Timestamps
    created_at TIMESTAMP DEFAULT NOW(),
    discussed_at TIMESTAMP,
    resolved_at TIMESTAMP,

    -- Resolution notes
    decision_notes TEXT,

    -- Source tracking
    analysis_run_id VARCHAR(64),  -- Which analysis run generated this
    trigger VARCHAR(64)  -- scheduled, on_edit, on_request
);

CREATE INDEX ix_suggestions_project_status ON friday_suggestions(project_id, status);
CREATE INDEX ix_suggestions_priority ON friday_suggestions(priority, created_at);
```

### Suggestion Types

| Type | Description | Example |
|------|-------------|---------|
| `plot_inconsistency` | Logic error or contradiction | "Scene 5 says Neelima is at work, but Scene 7 has her at home without transition" |
| `character_arc_gap` | Missing character development | "Ravi's transformation from skeptic to believer lacks a pivotal moment" |
| `dialogue_improvement` | Better dialogue suggestion | "This line could be more natural in Telugu" |
| `pacing_issue` | Scene too long/short, rhythm problem | "Three consecutive emotional scenes - consider adding breathing room" |
| `transition_missing` | Abrupt scene change | "Add transition between Scene 18 and 19" |
| `setup_payoff_missing` | Chekhov's gun unfired | "The gun mentioned in Scene 3 is never used" |
| `continuity_error` | Visual/temporal continuity | "Character's costume changes mid-scene" |
| `structure_suggestion` | Story structure improvement | "Consider moving Scene 12 earlier for better build-up" |
| `draft_scene` | New scene suggestion | "Friday suggests adding a scene showing..." |
| `research_finding` | Relevant research discovered | "Found reference that might help with the courtroom scene" |

---

## 3. Agent Architecture

### Agent Class

```python
# agents/script_analyzer.py

from dataclasses import dataclass
from datetime import datetime
from typing import List, Optional
from enum import Enum
import asyncio
import logging

from db.screenplay_schema import ScreenplayProject, ScreenplayScene
from db.agent_schema import FridaySuggestion

LOGGER = logging.getLogger(__name__)


class AnalysisTrigger(str, Enum):
    SCHEDULED = "scheduled"      # Nightly run
    ON_EDIT = "on_edit"          # After scene edit
    ON_REQUEST = "on_request"    # Boss asked for analysis


@dataclass
class AnalysisConfig:
    """Configuration for script analysis"""
    project_slug: str

    # What to analyze
    analyze_plot: bool = True
    analyze_characters: bool = True
    analyze_dialogue: bool = True
    analyze_pacing: bool = True

    # Scope
    scene_range: Optional[tuple[int, int]] = None  # (start, end) or None for all
    focus_scenes: Optional[List[int]] = None  # Specific scenes to prioritize

    # Thresholds
    min_confidence: float = 0.5  # Don't suggest below this confidence
    max_suggestions: int = 10  # Cap suggestions per run


class ScriptAnalyzerAgent:
    """
    Autonomous agent that analyzes screenplay for improvements.

    Runs during idle time (2 AM scheduled or when explicitly requested).
    Generates suggestions stored in friday_suggestions table.

    Flow:
    1. Load project and all scenes
    2. Run analysis modules in sequence
    3. Generate suggestions via LLM
    4. Deduplicate against existing suggestions
    5. Store new suggestions
    6. Optionally notify Boss in morning
    """

    def __init__(
        self,
        db_session,
        llm_client,
        config: Optional[AnalysisConfig] = None,
    ):
        self.db = db_session
        self.llm = llm_client
        self.config = config
        self.run_id = None

    async def analyze(
        self,
        project_slug: str,
        trigger: AnalysisTrigger = AnalysisTrigger.SCHEDULED,
    ) -> List[FridaySuggestion]:
        """
        Run full script analysis for a project.

        Returns list of new suggestions created.
        """
        self.run_id = f"{project_slug}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        LOGGER.info(f"Starting analysis run: {self.run_id}")

        # Load project
        project = await self._load_project(project_slug)
        if not project:
            LOGGER.error(f"Project not found: {project_slug}")
            return []

        scenes = await self._load_scenes(project.id)
        LOGGER.info(f"Loaded {len(scenes)} scenes for analysis")

        all_suggestions = []

        # Run analysis modules
        if self.config is None or self.config.analyze_plot:
            suggestions = await self._analyze_plot(project, scenes)
            all_suggestions.extend(suggestions)

        if self.config is None or self.config.analyze_characters:
            suggestions = await self._analyze_characters(project, scenes)
            all_suggestions.extend(suggestions)

        if self.config is None or self.config.analyze_dialogue:
            suggestions = await self._analyze_dialogue(project, scenes)
            all_suggestions.extend(suggestions)

        if self.config is None or self.config.analyze_pacing:
            suggestions = await self._analyze_pacing(project, scenes)
            all_suggestions.extend(suggestions)

        # Filter by confidence threshold
        min_conf = self.config.min_confidence if self.config else 0.5
        filtered = [s for s in all_suggestions if s.confidence >= min_conf]

        # Deduplicate against existing suggestions
        new_suggestions = await self._deduplicate(project.id, filtered)

        # Apply max limit
        max_sugg = self.config.max_suggestions if self.config else 10
        new_suggestions = new_suggestions[:max_sugg]

        # Store in database
        for suggestion in new_suggestions:
            suggestion.analysis_run_id = self.run_id
            suggestion.trigger = trigger.value
            self.db.add(suggestion)

        await self.db.commit()

        LOGGER.info(f"Analysis complete: {len(new_suggestions)} new suggestions")
        return new_suggestions

    async def _analyze_plot(
        self,
        project: ScreenplayProject,
        scenes: List[ScreenplayScene],
    ) -> List[FridaySuggestion]:
        """Analyze plot for inconsistencies and gaps"""

        # Build scene summary for LLM
        scene_summaries = []
        for scene in scenes:
            scene_summaries.append({
                "number": scene.scene_number,
                "location": scene.location,
                "summary": scene.summary or "[No summary]",
                "tags": scene.tags,
            })

        prompt = f"""Analyze this screenplay for plot issues.

Project: {project.title}
Logline: {project.logline or 'N/A'}

Scenes:
{self._format_scenes_for_prompt(scene_summaries)}

Find:
1. Plot inconsistencies (contradictions between scenes)
2. Logic gaps (missing explanations)
3. Setup without payoff (introduced elements that go nowhere)
4. Payoff without setup (events that feel unearned)

For each issue found, provide:
- Type (plot_inconsistency, setup_payoff_missing, etc.)
- Title (brief description)
- Description (detailed explanation)
- Affected scenes (list of scene numbers)
- Priority (1-5, where 1 is critical)
- Confidence (0-1, how certain you are)

Return as JSON array."""

        response = await self.llm.generate(prompt)
        issues = self._parse_llm_suggestions(response, project.id)

        return issues

    async def _analyze_characters(
        self,
        project: ScreenplayProject,
        scenes: List[ScreenplayScene],
    ) -> List[FridaySuggestion]:
        """Analyze character arcs and development"""

        # Extract character appearances per scene
        character_scenes = await self._build_character_map(scenes)

        prompt = f"""Analyze character development in this screenplay.

Project: {project.title}

Character appearances by scene:
{self._format_character_map(character_scenes)}

Find:
1. Character arc gaps (sudden changes without development)
2. Inconsistent behavior (out-of-character moments)
3. Underutilized characters (introduced but rarely seen)
4. Missing relationship development

For each issue, provide JSON with type, title, description, affected_scenes, priority, confidence."""

        response = await self.llm.generate(prompt)
        return self._parse_llm_suggestions(response, project.id)

    async def _analyze_dialogue(
        self,
        project: ScreenplayProject,
        scenes: List[ScreenplayScene],
    ) -> List[FridaySuggestion]:
        """Analyze dialogue for improvements"""

        # Focus on dialogue-heavy scenes
        dialogue_samples = await self._extract_dialogue_samples(scenes)

        prompt = f"""Analyze dialogue in this Telugu/English screenplay.

Sample dialogues:
{dialogue_samples}

Find:
1. Unnatural phrasing (could be more natural in Telugu/English)
2. Expository dialogue (too much explaining)
3. Similar voice across characters (everyone sounds the same)
4. Missed opportunities for subtext

Suggest specific improvements. Include proposed_change with better dialogue."""

        response = await self.llm.generate(prompt)
        return self._parse_llm_suggestions(response, project.id, category="dialogue")

    async def _analyze_pacing(
        self,
        project: ScreenplayProject,
        scenes: List[ScreenplayScene],
    ) -> List[FridaySuggestion]:
        """Analyze story pacing"""

        # Calculate scene lengths and emotional intensity
        pacing_data = []
        for scene in scenes:
            pacing_data.append({
                "number": scene.scene_number,
                "estimated_pages": scene.estimated_pages or 1.0,
                "tags": scene.tags,
                "location": scene.location,
            })

        prompt = f"""Analyze pacing in this screenplay.

Scene data:
{self._format_pacing_data(pacing_data)}

Find:
1. Consecutive heavy emotional scenes (need breathing room)
2. Long stretches in same location (could feel static)
3. Abrupt transitions (missing connective tissue)
4. Scenes that drag (could be tightened)

For each issue, suggest specific fixes."""

        response = await self.llm.generate(prompt)
        return self._parse_llm_suggestions(response, project.id, category="pacing")

    async def _deduplicate(
        self,
        project_id: int,
        new_suggestions: List[FridaySuggestion],
    ) -> List[FridaySuggestion]:
        """Remove suggestions that duplicate existing pending ones"""

        # Load existing pending suggestions
        existing = await self.db.query(FridaySuggestion).filter(
            FridaySuggestion.project_id == project_id,
            FridaySuggestion.status == 'pending',
        ).all()

        # Simple deduplication by title similarity
        existing_titles = {s.title.lower() for s in existing}

        unique = []
        for s in new_suggestions:
            if s.title.lower() not in existing_titles:
                unique.append(s)
                existing_titles.add(s.title.lower())

        return unique

    # Helper methods...
    def _format_scenes_for_prompt(self, scenes: list) -> str:
        return "\n".join(
            f"Scene {s['number']}: {s['location']} - {s['summary']}"
            for s in scenes
        )

    def _parse_llm_suggestions(
        self,
        response: str,
        project_id: int,
        category: str = None,
    ) -> List[FridaySuggestion]:
        """Parse LLM JSON response into suggestion objects"""
        import json

        try:
            data = json.loads(response)
            suggestions = []

            for item in data:
                s = FridaySuggestion(
                    project_id=project_id,
                    suggestion_type=item.get("type", "general"),
                    title=item.get("title", "Untitled"),
                    description=item.get("description", ""),
                    proposed_change=item.get("proposed_change"),
                    affected_scenes=item.get("affected_scenes", []),
                    priority=item.get("priority", 3),
                    confidence=item.get("confidence", 0.5),
                    category=category or item.get("category", "general"),
                )
                suggestions.append(s)

            return suggestions

        except json.JSONDecodeError:
            LOGGER.warning("Failed to parse LLM response as JSON")
            return []
```

---

## 4. Scheduler Integration

### Scheduled Nightly Analysis

```python
# agents/scheduler.py

import asyncio
from datetime import datetime, time
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger

class AgentScheduler:
    """Schedule autonomous agent runs"""

    def __init__(self, db_session, llm_client):
        self.db = db_session
        self.llm = llm_client
        self.scheduler = AsyncIOScheduler()

    def start(self):
        """Start the scheduler with configured jobs"""

        # Run script analysis at 2 AM daily
        self.scheduler.add_job(
            self._run_nightly_analysis,
            CronTrigger(hour=2, minute=0),
            id="nightly_script_analysis",
            name="Nightly Script Analysis",
        )

        self.scheduler.start()
        LOGGER.info("Agent scheduler started")

    async def _run_nightly_analysis(self):
        """Run analysis on all active projects"""

        # Get active projects
        projects = await self.db.query(ScreenplayProject).filter(
            ScreenplayProject.status.in_(["draft", "revision"])
        ).all()

        for project in projects:
            LOGGER.info(f"Running nightly analysis for: {project.slug}")

            agent = ScriptAnalyzerAgent(self.db, self.llm)
            suggestions = await agent.analyze(
                project.slug,
                trigger=AnalysisTrigger.SCHEDULED
            )

            LOGGER.info(f"Generated {len(suggestions)} suggestions for {project.slug}")
```

---

## 5. Orchestrator Integration

### Morning Greeting with Suggestions

When Boss activates Friday in the morning:

```python
# orchestrator/context_builder.py

async def build_morning_context(
    user_context: UserContext,
    db_session,
) -> dict:
    """Build context including overnight suggestions"""

    if user_context.access_level != AccessLevel.BOSS:
        return {}

    # Check for pending suggestions
    pending_count = await db_session.query(FridaySuggestion).filter(
        FridaySuggestion.status == 'pending'
    ).count()

    if pending_count > 0:
        # Get high priority suggestions
        high_priority = await db_session.query(FridaySuggestion).filter(
            FridaySuggestion.status == 'pending',
            FridaySuggestion.priority <= 2,
        ).limit(3).all()

        return {
            "suggestions_available": True,
            "pending_count": pending_count,
            "high_priority_summary": [
                f"{s.title} ({s.suggestion_type})"
                for s in high_priority
            ]
        }

    return {"suggestions_available": False}
```

### Friday's Morning Greeting

```
Boss: "Friday"
Friday: "Boss, morning. Nenu overnight analysis chesanu -
        3 suggestions ready, including one transition issue in Scene 18-19.
        Discuss cheddama?"
```

---

## 6. Suggestion Workflow

### Commands

| Command | Action |
|---------|--------|
| "Friday, show suggestions" | List all pending suggestions |
| "Friday, show high priority" | List priority 1-2 suggestions |
| "Friday, tell me about suggestion 1" | Detail on specific suggestion |
| "Friday, accept suggestion 1" | Mark as accepted, apply change |
| "Friday, reject suggestion 1" | Mark as rejected |
| "Friday, defer suggestion 1" | Mark as deferred (revisit later) |
| "Friday, analyze scene 15" | Trigger on-demand analysis |

### MCP Tool for Suggestions

```python
# mcp/suggestions/service.py

class SuggestionService:
    """MCP service for Friday suggestions"""

    async def list_suggestions(
        self,
        project_slug: str,
        status: str = "pending",
        priority_max: int = 5,
        limit: int = 10,
    ) -> List[dict]:
        """List suggestions for a project"""
        pass

    async def get_suggestion(
        self,
        suggestion_id: int,
    ) -> dict:
        """Get full details of a suggestion"""
        pass

    async def update_status(
        self,
        suggestion_id: int,
        status: str,  # accepted, rejected, deferred
        notes: Optional[str] = None,
    ) -> dict:
        """Update suggestion status"""
        pass

    async def apply_suggestion(
        self,
        suggestion_id: int,
    ) -> dict:
        """Apply proposed_change to the scene"""
        pass

    async def trigger_analysis(
        self,
        project_slug: str,
        scene_ids: Optional[List[int]] = None,
    ) -> dict:
        """Trigger on-demand analysis"""
        pass
```

---

## 7. Draft Scene Generation

When Friday suggests a new scene:

```python
async def generate_draft_scene(
    self,
    project: ScreenplayProject,
    context: dict,  # What should happen in this scene
    before_scene: Optional[int] = None,
    after_scene: Optional[int] = None,
) -> FridaySuggestion:
    """Generate a draft scene suggestion"""

    prompt = f"""Write a new scene for the screenplay "{project.title}".

Context: {context['description']}
Purpose: {context['purpose']}
Characters involved: {context.get('characters', [])}

Place this scene:
- After scene {after_scene}: {after_scene_summary}
- Before scene {before_scene}: {before_scene_summary}

Write in proper screenplay format:
- Scene heading (INT/EXT, location, time)
- Action blocks
- Dialogue (Telugu with English translation)

Keep it concise but complete."""

    response = await self.llm.generate(prompt)

    return FridaySuggestion(
        project_id=project.id,
        suggestion_type="draft_scene",
        title=f"New scene: {context['purpose'][:50]}",
        description=context['description'],
        proposed_change=response,  # The full draft scene
        affected_scenes=[after_scene, before_scene] if before_scene else [after_scene],
        priority=3,
        confidence=0.6,
        category="structure",
    )
```

---

## 8. File Structure

```
agents/
├── __init__.py
├── script_analyzer.py     # Main analysis agent
├── scheduler.py           # APScheduler integration
├── prompts/
│   ├── plot_analysis.txt
│   ├── character_analysis.txt
│   ├── dialogue_analysis.txt
│   └── pacing_analysis.txt
└── utils.py               # Helper functions

db/
└── agent_schema.py        # FridaySuggestion model

mcp/suggestions/
├── __init__.py
├── service.py             # Suggestion CRUD
└── server.py              # MCP server

config/
└── agent_config.yaml      # Scheduler, thresholds
```

---

## 9. Implementation Order

### Phase 1: Database & Basic Structure
1. [ ] Add FridaySuggestion to db/agent_schema.py
2. [ ] Create database migration
3. [ ] Implement suggestion CRUD in service

### Phase 2: Analysis Agent
1. [ ] Implement ScriptAnalyzerAgent
2. [ ] Create analysis prompts
3. [ ] Test with existing screenplay data

### Phase 3: Integration
1. [ ] Add scheduler for nightly runs
2. [ ] Integrate with orchestrator context
3. [ ] Add MCP tools for suggestion management

### Phase 4: Polish
1. [ ] Fine-tune analysis prompts
2. [ ] Add draft scene generation
3. [ ] Morning greeting integration

---

## 10. Next Steps

### Immediate (Can do now)
1. [ ] Create db/agent_schema.py with FridaySuggestion model
2. [ ] Create agents/ directory structure
3. [ ] Write analysis prompt templates

### After LLM Ready (Post iteration 2 training)
1. [ ] Implement full ScriptAnalyzerAgent
2. [ ] Test against Gusagusalu screenplay
3. [ ] Tune confidence thresholds

---

*Document generated: January 2026*
