"""
Mentor Engine
=============

Applies book knowledge to user's creative work.

This is the "wise collaborator" - not just finding information,
but reasoning about the user's work using book frameworks.

Example:
    User: "I'm writing a court scene where the witness breaks down"

    MentorEngine:
    1. Retrieves relevant principles about witness scenes from all studied books
    2. Checks what elements the user has vs what books recommend
    3. Finds similar examples from books
    4. Generates suggestions grounded in book knowledge
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Dict, List, Optional

from documents.understanding.models import (
    BookUnderstanding,
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
)

LOGGER = logging.getLogger(__name__)


# =============================================================================
# Mentor Prompts
# =============================================================================

ANALYZE_SCENE_PROMPT = """You are a screenwriting mentor who has deeply studied these books:
{book_summaries}

The user is working on this scene/element:
"{user_input}"

Project context: {project_context}

Based on what you learned from the books, analyze this:

1. ELEMENTS PRESENT: What good elements does this scene already have?
2. ELEMENTS MISSING: What might be missing based on book principles?
3. RELEVANT PRINCIPLES: Which rules from the books apply here? (cite book and page)
4. APPLICABLE TECHNIQUES: What techniques from the books could help?
5. SIMILAR EXAMPLES: What films/examples from the books handle similar situations?
6. SUGGESTIONS: Specific recommendations based on book knowledge
7. QUESTIONS: What should the writer consider?

Respond in JSON format:
{{
    "elements_present": ["..."],
    "elements_missing": ["..."],
    "relevant_principles": [
        {{"statement": "...", "source_book": "...", "why_relevant": "..."}}
    ],
    "applicable_techniques": [
        {{"name": "...", "source_book": "...", "how_to_apply": "..."}}
    ],
    "similar_examples": [
        {{"film": "...", "scene": "...", "lesson": "...", "source_book": "..."}}
    ],
    "suggestions": ["..."],
    "questions_to_consider": ["..."]
}}"""


BRAINSTORM_PROMPT = """You are a screenwriting mentor who has deeply studied these books:
{book_summaries}

The user wants to brainstorm: "{topic}"
Constraints: {constraints}

Generate creative ideas that are GROUNDED in book knowledge. Each idea should:
- Be specific and actionable
- Reference relevant book concepts/techniques
- Include reasoning from the books

Respond in JSON format:
{{
    "ideas": [
        {{
            "idea": "...",
            "rationale": "Why this works according to the books",
            "based_on": ["concept/technique name from books"],
            "source_inspiration": "Film example if applicable"
        }}
    ],
    "suggested_structure": "If books suggest a pattern/approach for this",
    "concepts_to_apply": ["relevant concepts from the books"],
    "techniques_to_try": ["relevant techniques"]
}}"""


CHECK_RULES_PROMPT = """You are a screenwriting mentor who has deeply studied these books:
{book_summaries}

Review this scene/work against the principles from the books:

Scene/Work:
"{scene_or_work}"

For each relevant principle from the books, determine:
- Is this principle being FOLLOWED?
- Is it being VIOLATED?
- Is it UNCLEAR whether it applies?
- Is it NOT APPLICABLE to this scene?

Focus on the most important principles. Be specific about what in the scene
shows whether the principle is followed or violated.

Respond in JSON format:
{{
    "rules_followed": [
        {{"principle": "...", "evidence": "...", "source_book": "..."}}
    ],
    "rules_violated": [
        {{"principle": "...", "evidence": "...", "suggestion": "...", "source_book": "..."}}
    ],
    "rules_unclear": [
        {{"principle": "...", "question": "...", "source_book": "..."}}
    ],
    "overall_assessment": "Brief summary of how well this follows book principles",
    "priority_fixes": ["Most important issues to address"]
}}"""


FIND_INSPIRATION_PROMPT = """You are a screenwriting mentor who has deeply studied these books:
{book_summaries}

The user is looking for inspiration for: "{situation}"

Find examples from the books that handle similar situations. For each example:
- Describe what happens in that film/scene
- Explain why it's relevant
- Suggest how the user might adapt it

Respond in JSON format:
{{
    "inspirations": [
        {{
            "film": "...",
            "scene": "...",
            "description": "What happens",
            "why_relevant": "Why this helps the user's situation",
            "how_to_adapt": "How to use this inspiration",
            "source_book": "...",
            "page": "..."
        }}
    ]
}}"""


# =============================================================================
# Mentor Engine
# =============================================================================


class MentorEngine:
    """
    Applies book knowledge to user's creative work.

    The mentor doesn't just search - it REASONS about the user's work
    using the frameworks and principles from studied books.
    """

    def __init__(
        self,
        llm_complete: Callable[[str], str],
        book_store: Optional[Any] = None,  # BookUnderstandingStore
    ):
        """
        Initialize mentor engine.

        Args:
            llm_complete: Function that takes a prompt and returns LLM response
            book_store: Optional store for book understandings
        """
        self._llm_complete = llm_complete
        self._book_store = book_store
        self._active_books: Dict[str, BookUnderstanding] = {}

    def load_books(self, books: List[BookUnderstanding]) -> None:
        """Load book understandings for mentoring session"""
        self._active_books = {book.id: book for book in books}
        LOGGER.info(f"Loaded {len(books)} books for mentoring")

    def get_active_books(self) -> List[BookUnderstanding]:
        """Get currently active books"""
        return list(self._active_books.values())

    # =========================================================================
    # Core Mentor Methods
    # =========================================================================

    async def analyze_scene(
        self,
        user_input: str,
        project_context: str = "",
        book_ids: Optional[List[str]] = None,
    ) -> MentorAnalysis:
        """
        Analyze user's scene/element against book knowledge.

        Args:
            user_input: What the user is working on
            project_context: Context about the project
            book_ids: Specific books to use (or all active if None)

        Returns:
            MentorAnalysis with structured feedback
        """
        books = self._get_books(book_ids)
        if not books:
            return MentorAnalysis(
                user_input=user_input,
                active_books=[],
                suggestions=["No books loaded. Study some reference books first."],
            )

        book_summaries = self._format_book_summaries(books)

        prompt = ANALYZE_SCENE_PROMPT.format(
            book_summaries=book_summaries,
            user_input=user_input,
            project_context=project_context or "General screenplay",
        )

        response = await self._llm_complete(prompt)
        data = self._parse_json(response)

        # Build analysis from response
        analysis = MentorAnalysis(
            user_input=user_input,
            active_books=[b.title for b in books],
            elements_present=data.get("elements_present", []),
            elements_missing=data.get("elements_missing", []),
            suggestions=data.get("suggestions", []),
            questions_to_consider=data.get("questions_to_consider", []),
        )

        # Parse principles
        for p in data.get("relevant_principles", []):
            principle = self._find_principle(p.get("statement", ""), books)
            if principle:
                analysis.relevant_principles.append(principle)

        # Parse techniques
        for t in data.get("applicable_techniques", []):
            technique = self._find_technique(t.get("name", ""), books)
            if technique:
                analysis.applicable_techniques.append(technique)

        # Parse examples as inspirations
        for e in data.get("similar_examples", []):
            example = self._find_example(e.get("film", ""), books)
            if example:
                analysis.similar_examples.append(
                    Inspiration(
                        example=example,
                        relevance_reason=e.get("lesson", ""),
                        how_to_apply=e.get("lesson", ""),
                        source_book=e.get("source_book", ""),
                    )
                )

        return analysis

    async def brainstorm(
        self,
        topic: str,
        constraints: Optional[List[str]] = None,
        book_ids: Optional[List[str]] = None,
    ) -> BrainstormResult:
        """
        Generate ideas using book frameworks.

        Args:
            topic: What to brainstorm about
            constraints: User requirements/limitations
            book_ids: Specific books to use

        Returns:
            BrainstormResult with ideas grounded in book knowledge
        """
        books = self._get_books(book_ids)
        if not books:
            return BrainstormResult(
                topic=topic,
                constraints=constraints or [],
                active_books=[],
                ideas=[
                    BrainstormIdea(
                        idea="Load reference books first",
                        rationale="No books are currently loaded for reference",
                        based_on=[],
                    )
                ],
            )

        book_summaries = self._format_book_summaries(books)

        prompt = BRAINSTORM_PROMPT.format(
            book_summaries=book_summaries,
            topic=topic,
            constraints=", ".join(constraints) if constraints else "None specified",
        )

        response = await self._llm_complete(prompt)
        data = self._parse_json(response)

        result = BrainstormResult(
            topic=topic,
            constraints=constraints or [],
            active_books=[b.title for b in books],
            suggested_structure=data.get("suggested_structure", ""),
            concepts_applied=data.get("concepts_to_apply", []),
            techniques_suggested=data.get("techniques_to_try", []),
        )

        for idea_data in data.get("ideas", []):
            result.ideas.append(
                BrainstormIdea(
                    idea=idea_data.get("idea", ""),
                    rationale=idea_data.get("rationale", ""),
                    based_on=idea_data.get("based_on", []),
                    source_inspiration=idea_data.get("source_inspiration"),
                )
            )

        return result

    async def check_rules(
        self,
        scene_or_work: str,
        book_ids: Optional[List[str]] = None,
    ) -> RuleCheckResult:
        """
        Check if user's work follows book principles.

        Args:
            scene_or_work: The content to check
            book_ids: Specific books to use

        Returns:
            RuleCheckResult showing what's followed vs violated
        """
        books = self._get_books(book_ids)
        if not books:
            return RuleCheckResult(
                document_ids=[],
                scene_or_work=scene_or_work,
                overall_assessment="No books loaded for reference",
            )

        book_summaries = self._format_book_summaries(books, include_principles=True)

        prompt = CHECK_RULES_PROMPT.format(
            book_summaries=book_summaries,
            scene_or_work=scene_or_work,
        )

        response = await self._llm_complete(prompt)
        data = self._parse_json(response)

        result = RuleCheckResult(
            document_ids=[b.document_id for b in books],
            scene_or_work=scene_or_work,
            overall_assessment=data.get("overall_assessment", ""),
            priority_fixes=data.get("priority_fixes", []),
        )

        # Parse rules
        for r in data.get("rules_followed", []):
            principle = self._find_principle(r.get("principle", ""), books)
            if principle:
                result.rules_followed.append(
                    RuleCheck(
                        principle=principle,
                        status="followed",
                        evidence=r.get("evidence", ""),
                    )
                )

        for r in data.get("rules_violated", []):
            principle = self._find_principle(r.get("principle", ""), books)
            if principle:
                result.rules_violated.append(
                    RuleCheck(
                        principle=principle,
                        status="violated",
                        evidence=r.get("evidence", ""),
                        suggestion=r.get("suggestion", ""),
                    )
                )

        for r in data.get("rules_unclear", []):
            principle = self._find_principle(r.get("principle", ""), books)
            if principle:
                result.rules_unclear.append(
                    RuleCheck(
                        principle=principle,
                        status="unclear",
                        evidence=r.get("question", ""),
                    )
                )

        return result

    async def find_inspiration(
        self,
        situation: str,
        book_ids: Optional[List[str]] = None,
    ) -> List[Inspiration]:
        """
        Find examples from books that handle similar situations.

        Args:
            situation: What the user is trying to write
            book_ids: Specific books to use

        Returns:
            List of inspirational examples from books
        """
        books = self._get_books(book_ids)
        if not books:
            return []

        book_summaries = self._format_book_summaries(books, include_examples=True)

        prompt = FIND_INSPIRATION_PROMPT.format(
            book_summaries=book_summaries,
            situation=situation,
        )

        response = await self._llm_complete(prompt)
        data = self._parse_json(response)

        inspirations = []
        for insp in data.get("inspirations", []):
            example = self._find_example(insp.get("film", ""), books)
            if example:
                inspirations.append(
                    Inspiration(
                        example=example,
                        relevance_reason=insp.get("why_relevant", ""),
                        how_to_apply=insp.get("how_to_adapt", ""),
                        source_book=insp.get("source_book", ""),
                    )
                )
            else:
                # Create a new example from the response
                inspirations.append(
                    Inspiration(
                        example=BookExample(
                            work_title=insp.get("film", "Unknown"),
                            scene_or_section=insp.get("scene", ""),
                            description=insp.get("description", ""),
                            lesson=insp.get("why_relevant", ""),
                            source_page=insp.get("page", ""),
                        ),
                        relevance_reason=insp.get("why_relevant", ""),
                        how_to_apply=insp.get("how_to_adapt", ""),
                        source_book=insp.get("source_book", ""),
                    )
                )

        return inspirations

    # =========================================================================
    # Quick Methods (for common patterns)
    # =========================================================================

    async def what_would_books_say(
        self,
        question: str,
        book_ids: Optional[List[str]] = None,
    ) -> str:
        """
        Quick question answering using book knowledge.

        Returns natural language response.
        """
        books = self._get_books(book_ids)
        if not books:
            return "No books loaded. Study some reference books first."

        book_summaries = self._format_book_summaries(books)

        prompt = f"""You are a mentor who has deeply studied these books:
{book_summaries}

Answer this question based on what the books teach:
"{question}"

Cite specific books and principles in your answer."""

        return await self._llm_complete(prompt)

    async def compare_approaches(
        self,
        topic: str,
        book_ids: Optional[List[str]] = None,
    ) -> str:
        """
        Compare what different books say about a topic.

        Returns natural language comparison.
        """
        books = self._get_books(book_ids)
        if len(books) < 2:
            return "Need at least 2 books to compare approaches."

        book_summaries = self._format_book_summaries(books)

        prompt = f"""You have studied these books:
{book_summaries}

Compare what these different books say about: "{topic}"

For each book, explain:
1. Their core view on this topic
2. Where they agree with other books
3. Where they disagree or add unique perspective"""

        return await self._llm_complete(prompt)

    # =========================================================================
    # Helper Methods
    # =========================================================================

    def _get_books(self, book_ids: Optional[List[str]]) -> List[BookUnderstanding]:
        """Get books to use for mentoring"""
        if book_ids:
            return [
                self._active_books[bid] for bid in book_ids if bid in self._active_books
            ]
        return list(self._active_books.values())

    def _format_book_summaries(
        self,
        books: List[BookUnderstanding],
        include_principles: bool = False,
        include_examples: bool = False,
    ) -> str:
        """Format book knowledge for LLM prompt"""
        parts = []
        for book in books:
            book_part = [f"## {book.title} by {book.author}"]
            book_part.append(f"Main argument: {book.main_argument}")
            book_part.append(f"Summary: {book.summary[:500]}...")

            # Key concepts
            if book.concepts:
                concepts = ", ".join(c.name for c in book.concepts[:10])
                book_part.append(f"Key concepts: {concepts}")

            # Principles if requested
            if include_principles and book.principles:
                book_part.append("Key principles:")
                for p in book.principles[:10]:
                    book_part.append(f"  - {p.statement}")

            # Examples if requested
            if include_examples and book.examples:
                book_part.append("Referenced films/examples:")
                for e in book.examples[:5]:
                    book_part.append(f"  - {e.work_title}: {e.lesson[:100]}")

            parts.append("\n".join(book_part))

        return "\n\n---\n\n".join(parts)

    def _find_principle(
        self, statement: str, books: List[BookUnderstanding]
    ) -> Optional[Principle]:
        """Find a principle by statement text"""
        statement_lower = statement.lower()
        for book in books:
            for p in book.principles:
                if (
                    statement_lower in p.statement.lower()
                    or p.statement.lower() in statement_lower
                ):
                    return p
        return None

    def _find_technique(
        self, name: str, books: List[BookUnderstanding]
    ) -> Optional[Technique]:
        """Find a technique by name"""
        name_lower = name.lower()
        for book in books:
            for t in book.techniques:
                if name_lower in t.name.lower() or t.name.lower() in name_lower:
                    return t
        return None

    def _find_example(
        self, film_title: str, books: List[BookUnderstanding]
    ) -> Optional[BookExample]:
        """Find an example by film title"""
        title_lower = film_title.lower()
        for book in books:
            for e in book.examples:
                if title_lower in e.work_title.lower():
                    return e
        return None

    def _parse_json(self, response: str) -> Dict[str, Any]:
        """Parse JSON from LLM response"""
        import json

        try:
            start = response.find("{")
            end = response.rfind("}") + 1
            if start >= 0 and end > start:
                return json.loads(response[start:end])
        except json.JSONDecodeError:
            pass
        LOGGER.warning(f"Failed to parse JSON: {response[:200]}")
        return {}
