"""
Book Understanding Data Models
==============================

Data structures for representing extracted knowledge from books.

These models capture what a book TEACHES, not just what it CONTAINS.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional
import uuid


class KnowledgeType(str, Enum):
    """Types of knowledge extracted from books"""

    CONCEPT = "concept"  # Named concept with definition
    PRINCIPLE = "principle"  # Rule or guideline
    TECHNIQUE = "technique"  # Practical method
    EXAMPLE = "example"  # Case study or reference
    FRAMEWORK = "framework"  # Multi-step process or structure
    WARNING = "warning"  # What NOT to do


class ConfidenceLevel(str, Enum):
    """How strongly the book states something"""

    ABSOLUTE = "absolute"  # "You MUST..." / "Never..."
    STRONG = "strong"  # "You should..." / "Best practice..."
    MODERATE = "moderate"  # "Consider..." / "Often works..."
    SUGGESTION = "suggestion"  # "You might..." / "Some writers..."


# =============================================================================
# Extracted Knowledge Types
# =============================================================================


@dataclass
class Concept:
    """
    A named concept defined in a book.

    Example: "Inciting Incident" from McKee's Story
    """

    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""  # "Inciting Incident"
    definition: str = ""  # How the book defines it
    importance: str = ""  # Why it matters
    source_document_id: str = ""
    source_pages: str = ""  # "pp. 45-52"

    # Relationships
    related_concepts: List[str] = field(default_factory=list)  # Other concept names
    parent_concept: Optional[str] = None  # Broader concept this belongs to
    sub_concepts: List[str] = field(default_factory=list)

    # For Knowledge Graph
    synonyms: List[str] = field(default_factory=list)  # Alternative names
    domain: str = "screenwriting"  # Domain this concept belongs to

    # Metadata
    extracted_at: datetime = field(default_factory=datetime.now)
    confidence: float = 0.9  # Extraction confidence

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "definition": self.definition,
            "importance": self.importance,
            "source_document_id": self.source_document_id,
            "source_pages": self.source_pages,
            "related_concepts": self.related_concepts,
            "parent_concept": self.parent_concept,
            "sub_concepts": self.sub_concepts,
            "synonyms": self.synonyms,
            "domain": self.domain,
            "extracted_at": self.extracted_at.isoformat(),
            "confidence": self.confidence,
        }


@dataclass
class Principle:
    """
    A rule or guideline taught by a book.

    Example: "Every scene must have a turning point" from McKee
    """

    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    statement: str = ""  # The rule itself
    rationale: str = ""  # Why this principle exists
    source_document_id: str = ""
    source_page: str = ""

    # Strength
    confidence_level: ConfidenceLevel = ConfidenceLevel.STRONG

    # Context
    applies_to: List[str] = field(default_factory=list)  # "court scenes", "dialogue"
    exceptions: List[str] = field(default_factory=list)  # When this doesn't apply
    prerequisites: List[str] = field(default_factory=list)  # What must be true first

    # Related knowledge
    related_concepts: List[str] = field(default_factory=list)
    related_techniques: List[str] = field(default_factory=list)

    # For actionable checking
    checkable: bool = True  # Can we verify if user follows this?
    check_question: str = ""  # "Does your scene have a turning point?"

    # Metadata
    extracted_at: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "statement": self.statement,
            "rationale": self.rationale,
            "source_document_id": self.source_document_id,
            "source_page": self.source_page,
            "confidence_level": self.confidence_level.value,
            "applies_to": self.applies_to,
            "exceptions": self.exceptions,
            "prerequisites": self.prerequisites,
            "related_concepts": self.related_concepts,
            "related_techniques": self.related_techniques,
            "checkable": self.checkable,
            "check_question": self.check_question,
            "extracted_at": self.extracted_at.isoformat(),
        }


@dataclass
class Technique:
    """
    A practical method or approach described in a book.

    Example: "The Slow Reveal" technique for building tension
    """

    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""  # Name of the technique
    description: str = ""  # How to use it
    steps: List[str] = field(default_factory=list)  # Step-by-step if applicable
    source_document_id: str = ""
    source_page: str = ""

    # When to use
    use_cases: List[str] = field(default_factory=list)  # "building tension", "reveal"
    when_to_use: str = ""  # Situation description
    when_not_to_use: str = ""  # Anti-patterns

    # Examples
    example_films: List[str] = field(default_factory=list)
    example_description: str = ""  # How it's used in examples

    # Related
    related_concepts: List[str] = field(default_factory=list)
    related_principles: List[str] = field(default_factory=list)
    alternative_techniques: List[str] = field(default_factory=list)

    # Metadata
    difficulty: str = "intermediate"  # beginner, intermediate, advanced
    extracted_at: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "steps": self.steps,
            "source_document_id": self.source_document_id,
            "source_page": self.source_page,
            "use_cases": self.use_cases,
            "when_to_use": self.when_to_use,
            "when_not_to_use": self.when_not_to_use,
            "example_films": self.example_films,
            "example_description": self.example_description,
            "related_concepts": self.related_concepts,
            "related_principles": self.related_principles,
            "alternative_techniques": self.alternative_techniques,
            "difficulty": self.difficulty,
            "extracted_at": self.extracted_at.isoformat(),
        }


@dataclass
class BookExample:
    """
    A case study or film reference used in a book to illustrate a point.

    Example: Analysis of "12 Angry Men" courtroom dynamics
    """

    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    work_title: str = ""  # "12 Angry Men", "A Few Good Men"
    work_type: str = "film"  # film, play, novel, tv_episode
    scene_or_section: str = ""  # Specific scene/chapter referenced
    source_document_id: str = ""
    source_page: str = ""

    # What the author teaches
    description: str = ""  # What happens in the example
    lesson: str = ""  # What the author teaches from this
    what_works: str = ""  # Why this example is effective

    # Connections
    demonstrates_concept: List[str] = field(default_factory=list)
    demonstrates_technique: List[str] = field(default_factory=list)
    demonstrates_principle: List[str] = field(default_factory=list)

    # For brainstorming reference
    situation_type: List[str] = field(default_factory=list)  # "courtroom", "tension"
    emotional_beat: str = ""  # "revelation", "confrontation"

    # Metadata
    extracted_at: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "work_title": self.work_title,
            "work_type": self.work_type,
            "scene_or_section": self.scene_or_section,
            "source_document_id": self.source_document_id,
            "source_page": self.source_page,
            "description": self.description,
            "lesson": self.lesson,
            "what_works": self.what_works,
            "demonstrates_concept": self.demonstrates_concept,
            "demonstrates_technique": self.demonstrates_technique,
            "demonstrates_principle": self.demonstrates_principle,
            "situation_type": self.situation_type,
            "emotional_beat": self.emotional_beat,
            "extracted_at": self.extracted_at.isoformat(),
        }


# =============================================================================
# Book-Level Understanding
# =============================================================================


@dataclass
class ChapterSummary:
    """Summary of a single chapter"""

    number: int
    title: str
    summary: str  # 2-3 sentence summary
    key_points: List[str] = field(default_factory=list)
    concepts_introduced: List[str] = field(default_factory=list)
    principles_taught: List[str] = field(default_factory=list)
    page_range: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "number": self.number,
            "title": self.title,
            "summary": self.summary,
            "key_points": self.key_points,
            "concepts_introduced": self.concepts_introduced,
            "principles_taught": self.principles_taught,
            "page_range": self.page_range,
        }


@dataclass
class BookUnderstanding:
    """
    Complete understanding of a single book.

    This is the "brain's model" of the book - not the raw text,
    but the structured knowledge extracted from it.
    """

    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    document_id: str = ""  # Link to raw document

    # High-level understanding
    title: str = ""
    author: str = ""
    summary: str = ""  # 2-3 paragraph summary
    main_argument: str = ""  # One sentence thesis
    target_audience: str = ""  # "Screenwriters working on..."

    # Structure
    chapters: List[ChapterSummary] = field(default_factory=list)

    # Extracted Knowledge
    concepts: List[Concept] = field(default_factory=list)
    principles: List[Principle] = field(default_factory=list)
    techniques: List[Technique] = field(default_factory=list)
    examples: List[BookExample] = field(default_factory=list)

    # Cross-book relationships (filled in when multiple books are studied)
    agrees_with: Dict[str, List[str]] = field(
        default_factory=dict
    )  # book_id -> concept names
    disagrees_with: Dict[str, List[str]] = field(default_factory=dict)
    extends: Dict[str, List[str]] = field(
        default_factory=dict
    )  # Adds to another book's idea

    # Metadata
    domains: List[str] = field(default_factory=list)  # "court drama", "dialogue"
    study_completed_at: Optional[datetime] = None
    comprehension_quality: float = 0.0  # 0-1, how well we understood it

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "document_id": self.document_id,
            "title": self.title,
            "author": self.author,
            "summary": self.summary,
            "main_argument": self.main_argument,
            "target_audience": self.target_audience,
            "chapters": [c.to_dict() for c in self.chapters],
            "concepts": [c.to_dict() for c in self.concepts],
            "principles": [p.to_dict() for p in self.principles],
            "techniques": [t.to_dict() for t in self.techniques],
            "examples": [e.to_dict() for e in self.examples],
            "agrees_with": self.agrees_with,
            "disagrees_with": self.disagrees_with,
            "extends": self.extends,
            "domains": self.domains,
            "study_completed_at": (
                self.study_completed_at.isoformat() if self.study_completed_at else None
            ),
            "comprehension_quality": self.comprehension_quality,
        }

    @property
    def total_knowledge_items(self) -> int:
        """Total extracted knowledge pieces"""
        return (
            len(self.concepts)
            + len(self.principles)
            + len(self.techniques)
            + len(self.examples)
        )


# =============================================================================
# Mentor Engine Output Types
# =============================================================================


@dataclass
class RuleCheck:
    """Result of checking a single principle against user's work"""

    principle: Principle
    status: str  # "followed", "violated", "unclear", "not_applicable"
    evidence: str  # What in user's work shows this
    suggestion: str = ""  # How to address if violated


@dataclass
class RuleCheckResult:
    """Results of checking user's work against book principles"""

    document_ids: List[str]  # Books used for checking
    scene_or_work: str  # What was checked

    rules_followed: List[RuleCheck] = field(default_factory=list)
    rules_violated: List[RuleCheck] = field(default_factory=list)
    rules_unclear: List[RuleCheck] = field(default_factory=list)
    rules_not_applicable: List[RuleCheck] = field(default_factory=list)

    overall_assessment: str = ""  # Summary
    priority_fixes: List[str] = field(default_factory=list)  # Most important issues


@dataclass
class Inspiration:
    """An example from books that might inspire the user"""

    example: BookExample
    relevance_reason: str  # Why this is relevant to user's situation
    how_to_apply: str  # How user might adapt this
    source_book: str  # Title of source book


@dataclass
class MentorAnalysis:
    """
    Complete analysis of user's scene/work against book knowledge.

    This is what Friday returns during brainstorming.
    """

    user_input: str  # What the user described
    active_books: List[str]  # Books used for analysis

    # What the user has
    elements_present: List[str] = field(default_factory=list)
    strengths: List[str] = field(default_factory=list)

    # What might be missing
    elements_missing: List[str] = field(default_factory=list)
    potential_issues: List[str] = field(default_factory=list)

    # Applicable knowledge
    relevant_principles: List[Principle] = field(default_factory=list)
    applicable_techniques: List[Technique] = field(default_factory=list)
    similar_examples: List[Inspiration] = field(default_factory=list)

    # Suggestions
    suggestions: List[str] = field(default_factory=list)
    questions_to_consider: List[str] = field(default_factory=list)

    # Cross-book insights
    book_agreements: List[str] = field(default_factory=list)  # Where books agree
    book_disagreements: List[str] = field(default_factory=list)  # Where they differ

    def to_response(self) -> str:
        """Format as natural language response for Friday"""
        parts = []

        if self.strengths:
            parts.append(f"**What's working:** {', '.join(self.strengths)}")

        if self.elements_missing:
            parts.append(f"**Consider adding:** {', '.join(self.elements_missing)}")

        if self.relevant_principles:
            principle_notes = []
            for p in self.relevant_principles[:3]:  # Top 3
                principle_notes.append(f"- {p.statement} (p. {p.source_page})")
            parts.append("**Relevant principles:**\n" + "\n".join(principle_notes))

        if self.similar_examples:
            example_notes = []
            for insp in self.similar_examples[:2]:  # Top 2
                example_notes.append(
                    f"- '{insp.example.work_title}': {insp.relevance_reason}"
                )
            parts.append("**Inspiration:**\n" + "\n".join(example_notes))

        if self.suggestions:
            parts.append(
                "**Suggestions:**\n" + "\n".join(f"- {s}" for s in self.suggestions[:3])
            )

        return "\n\n".join(parts)


@dataclass
class BrainstormIdea:
    """A single brainstorm idea grounded in book knowledge"""

    idea: str
    rationale: str  # Why this makes sense
    based_on: List[str]  # Book concepts/principles this draws from
    source_inspiration: Optional[str] = None  # Film/example if applicable
    potential_risks: List[str] = field(default_factory=list)


@dataclass
class BrainstormResult:
    """
    Results of a brainstorming session using book knowledge.
    """

    topic: str  # What user wanted to brainstorm
    constraints: List[str]  # User's requirements
    active_books: List[str]  # Books used

    # Generated ideas
    ideas: List[BrainstormIdea] = field(default_factory=list)

    # Framework suggestions
    suggested_structure: str = ""  # If books suggest a pattern

    # Applicable knowledge used
    concepts_applied: List[str] = field(default_factory=list)
    techniques_suggested: List[str] = field(default_factory=list)

    def to_response(self) -> str:
        """Format as natural language response"""
        parts = [f"**Brainstorming: {self.topic}**\n"]

        for i, idea in enumerate(self.ideas, 1):
            parts.append(f"{i}. **{idea.idea}**")
            parts.append(f"   Why: {idea.rationale}")
            if idea.based_on:
                parts.append(f"   Based on: {', '.join(idea.based_on)}")

        if self.suggested_structure:
            parts.append(f"\n**Suggested approach:** {self.suggested_structure}")

        return "\n".join(parts)
