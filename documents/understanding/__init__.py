"""
Book Understanding Layer
========================

Transforms documents from raw text into structured knowledge that Friday
can reason with during brainstorming sessions.

NOT just retrieval - this is COMPREHENSION.

Components:
    - BookUnderstanding: Complete understanding of a single book
    - Concept/Principle/Technique/Example: Extracted knowledge types
    - BookComprehensionEngine: Extracts knowledge from documents
    - MentorEngine: Applies book knowledge to user's creative work
    - BookGraphIntegrator: Links book knowledge to Knowledge Graph
    - StudyJobTracker: Live progress tracking for status queries

Status:
    DONE: BookComprehensionEngine implementation
    DONE: MentorEngine implementation
    DONE: BookUnderstandingStore (SQLite persistence)
    DONE: Knowledge Graph integration
    DONE: StudyJobTracker for live progress monitoring
    TODO: Test with sample screenwriting book
"""

from documents.understanding.models import (
    BookUnderstanding,
    ChapterSummary,
    Concept,
    Principle,
    Technique,
    BookExample,
    MentorAnalysis,
    BrainstormResult,
    BrainstormIdea,
    RuleCheckResult,
    RuleCheck,
    Inspiration,
    KnowledgeType,
    ConfidenceLevel,
)

from documents.understanding.comprehension import BookComprehensionEngine
from documents.understanding.mentor import MentorEngine
from documents.understanding.graph_integration import BookGraphIntegrator
from documents.understanding.job_tracker import (
    StudyJobTracker,
    StudyJob,
    JobStatus,
    get_job_tracker,
)

__all__ = [
    # Enums
    "KnowledgeType",
    "ConfidenceLevel",
    "JobStatus",
    # Core understanding
    "BookUnderstanding",
    "ChapterSummary",
    "Concept",
    "Principle",
    "Technique",
    "BookExample",
    # Mentor outputs
    "MentorAnalysis",
    "BrainstormResult",
    "BrainstormIdea",
    "RuleCheckResult",
    "RuleCheck",
    "Inspiration",
    # Engines
    "BookComprehensionEngine",
    "MentorEngine",
    "BookGraphIntegrator",
    # Job Tracking
    "StudyJobTracker",
    "StudyJob",
    "get_job_tracker",
]
