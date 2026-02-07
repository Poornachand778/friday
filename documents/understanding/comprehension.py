"""
Book Comprehension Engine
=========================

Extracts structured knowledge from documents using LLM.

This is the "reading and understanding" phase - turning raw text
into concepts, principles, techniques, and examples that Friday
can reason with.

Modes:
    - thorough_mode=True: Process every chapter, extract ALL knowledge
    - thorough_mode=False: Sample chunks, faster but may miss details

Voice Integration:
    - Progress callbacks designed for voice announcements
    - "Boss, studying Chapter 3: The Inciting Incident..."
    - "Found 5 new concepts, 3 principles so far..."
"""

from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Tuple

from documents.models import Document, Chunk, ChapterInfo
from documents.config import ComprehensionConfig, get_document_config
from documents.understanding.models import (
    BookUnderstanding,
    ChapterSummary,
    Concept,
    Principle,
    Technique,
    BookExample,
    ConfidenceLevel,
)
from documents.understanding.job_tracker import (
    StudyJobTracker,
    JobStatus,
    get_job_tracker,
)

LOGGER = logging.getLogger(__name__)


# Voice-friendly progress message templates
VOICE_MESSAGES = {
    "start": "Boss, starting to study '{title}' by {author}. {chapters} chapters to analyze.",
    "chapter_start": "Studying Chapter {num}: {title}...",
    "chapter_complete": "Chapter {num} done. Found {concepts} concepts, {principles} principles.",
    "extracting_summary": "Reading through the book to understand the main ideas...",
    "extracting_concepts": "Identifying key concepts and definitions...",
    "extracting_principles": "Finding the rules and guidelines...",
    "extracting_techniques": "Looking for practical techniques...",
    "extracting_examples": "Gathering film examples and case studies...",
    "merging": "Putting it all together...",
    "complete": "Boss, finished studying '{title}'. Found {total} knowledge items: {concepts} concepts, {principles} principles, {techniques} techniques, {examples} examples.",
    "quality_high": "Good comprehension quality.",
    "quality_medium": "Moderate comprehension - might have missed some details.",
    "quality_low": "Low comprehension quality - consider re-studying with different settings.",
}


# =============================================================================
# Extraction Prompts
# =============================================================================

SUMMARY_PROMPT = """Analyze this book excerpt and provide:

1. A 2-3 paragraph SUMMARY of what this book teaches
2. The MAIN ARGUMENT in one sentence (what is the author's thesis?)
3. The TARGET AUDIENCE (who is this book for?)

Book Title: {title}
Author: {author}

Content (excerpts from throughout the book):
{content}

Respond in JSON format:
{{
    "summary": "...",
    "main_argument": "...",
    "target_audience": "..."
}}"""


CONCEPT_EXTRACTION_PROMPT = """Extract the KEY CONCEPTS that this book defines or explains.

A concept is a named idea with a specific meaning in this domain.
Examples: "Inciting Incident", "Character Arc", "Power Shift", "Subtext"

For each concept, provide:
- name: The concept name
- definition: How the book defines it
- importance: Why it matters (according to the book)
- related_concepts: Other concepts it connects to

Book: {title} by {author}
Content:
{content}

Respond in JSON format:
{{
    "concepts": [
        {{
            "name": "...",
            "definition": "...",
            "importance": "...",
            "related_concepts": ["...", "..."]
        }}
    ]
}}

Extract 5-15 key concepts. Focus on concepts the author emphasizes or defines explicitly."""


PRINCIPLE_EXTRACTION_PROMPT = """Extract the PRINCIPLES (rules/guidelines) that this book teaches.

A principle is a rule or guideline that the author recommends following.
Examples: "Every scene must have a turning point", "Show don't tell", "The protagonist must have a clear goal"

For each principle, provide:
- statement: The rule itself (clear, actionable)
- rationale: Why this principle exists (what happens if you don't follow it)
- applies_to: What situations this applies to
- exceptions: When this rule might not apply
- confidence_level: How strongly stated (absolute/strong/moderate/suggestion)
- check_question: A question to verify if someone is following this

Book: {title} by {author}
Content:
{content}

Respond in JSON format:
{{
    "principles": [
        {{
            "statement": "...",
            "rationale": "...",
            "applies_to": ["...", "..."],
            "exceptions": ["..."],
            "confidence_level": "strong",
            "check_question": "..."
        }}
    ]
}}

Extract 10-20 key principles. Focus on actionable rules the author emphasizes."""


TECHNIQUE_EXTRACTION_PROMPT = """Extract the TECHNIQUES (methods/approaches) that this book describes.

A technique is a practical method or approach that can be applied.
Examples: "The Slow Reveal", "In Medias Res", "Planting and Payoff", "The Ticking Clock"

For each technique, provide:
- name: Name of the technique
- description: How to use it
- when_to_use: Situations where this works well
- example_films: Films/works that demonstrate this (if mentioned)

Book: {title} by {author}
Content:
{content}

Respond in JSON format:
{{
    "techniques": [
        {{
            "name": "...",
            "description": "...",
            "when_to_use": "...",
            "example_films": ["...", "..."]
        }}
    ]
}}

Extract 5-15 techniques. Focus on practical methods the author describes in detail."""


EXAMPLE_EXTRACTION_PROMPT = """Extract the EXAMPLES (film references, case studies) that this book uses.

An example is a specific film, play, or work that the author analyzes to make a point.
Examples: "12 Angry Men - the deliberation room tension", "Chinatown - the water mystery"

For each example, provide:
- work_title: Name of the film/play/work
- scene_or_section: Specific scene referenced (if applicable)
- description: What happens in this example
- lesson: What the author teaches using this example
- demonstrates: What concept/technique this demonstrates

Book: {title} by {author}
Content:
{content}

Respond in JSON format:
{{
    "examples": [
        {{
            "work_title": "...",
            "scene_or_section": "...",
            "description": "...",
            "lesson": "...",
            "demonstrates": ["...", "..."]
        }}
    ]
}}

Extract 5-15 examples. Focus on examples the author analyzes in depth."""


CHAPTER_SUMMARY_PROMPT = """Summarize this chapter and extract its key teachings.

Chapter {chapter_number}: {chapter_title}
Content:
{content}

Provide:
1. A 2-3 sentence summary
2. 3-5 key points taught in this chapter
3. Any new concepts introduced
4. Any principles/rules stated

Respond in JSON format:
{{
    "summary": "...",
    "key_points": ["...", "..."],
    "concepts_introduced": ["...", "..."],
    "principles_taught": ["...", "..."]
}}"""


# =============================================================================
# Comprehension Engine
# =============================================================================


class BookComprehensionEngine:
    """
    Extracts structured knowledge from documents.

    Takes raw document chunks and produces a BookUnderstanding with:
    - Summary and thesis
    - Concepts (named ideas with definitions)
    - Principles (rules and guidelines)
    - Techniques (practical methods)
    - Examples (case studies and film references)

    Modes:
        - thorough_mode=True: Process chapter-by-chapter for complete coverage
        - thorough_mode=False: Sample chunks for faster processing
    """

    def __init__(
        self,
        llm_complete: Callable[[str], str],
        config: Optional[ComprehensionConfig] = None,
    ):
        """
        Initialize comprehension engine.

        Args:
            llm_complete: Function that takes a prompt and returns LLM response.
                         Signature: async def complete(prompt: str) -> str
            config: Comprehension configuration (uses defaults if not provided)
        """
        self._llm_complete = llm_complete
        self._config = config or get_document_config().comprehension
        self._llm_call_count = 0  # Track for cost estimation
        self._job_tracker = get_job_tracker()
        self._current_job_id: Optional[str] = None

    async def comprehend(
        self,
        document: Document,
        chunks: List[Chunk],
        progress_callback: Optional[Callable[[str, float], None]] = None,
        voice_callback: Optional[Callable[[str], None]] = None,
    ) -> BookUnderstanding:
        """
        Fully comprehend a document.

        Args:
            document: The document to comprehend
            chunks: All chunks from the document
            progress_callback: Optional callback for progress updates (stage_name, progress 0-1)
            voice_callback: Optional callback for voice announcements (message)

        Returns:
            BookUnderstanding with all extracted knowledge
        """
        self._llm_call_count = 0
        LOGGER.info(
            f"Starting comprehension of '{document.metadata.title}' (thorough_mode={self._config.thorough_mode})"
        )

        # Start job tracking for live status queries
        total_chapters = len(document.chapters) if document.chapters else 1
        self._current_job_id = self._job_tracker.start_job(
            document_id=document.id,
            title=document.metadata.title,
            author=document.metadata.author or "Unknown",
            total_chapters=total_chapters,
            total_pages=document.total_pages or 0,
        )
        LOGGER.info(f"Started study job: {self._current_job_id}")

        understanding = BookUnderstanding(
            document_id=document.id,
            title=document.metadata.title,
            author=document.metadata.author or "Unknown",
        )

        # Voice announcement: Starting
        if voice_callback and self._config.voice_announce_start:
            voice_callback(
                VOICE_MESSAGES["start"].format(
                    title=understanding.title,
                    author=understanding.author,
                    chapters=len(document.chapters) if document.chapters else "unknown",
                )
            )

        # Update job status to studying
        self._job_tracker.update_status(self._current_job_id, JobStatus.STUDYING)

        try:
            if self._config.thorough_mode and document.chapters:
                # THOROUGH MODE: Process chapter by chapter
                understanding = await self._comprehend_thorough(
                    document, chunks, understanding, progress_callback, voice_callback
                )
            else:
                # SAMPLING MODE: Quick pass using sampled chunks
                understanding = await self._comprehend_sampled(
                    document, chunks, understanding, progress_callback, voice_callback
                )
        except Exception as e:
            # Track job failure
            self._job_tracker.fail_job(self._current_job_id, str(e))
            raise

        # Determine domains
        understanding.domains = self._infer_domains(understanding)

        # Deduplicate if enabled
        if self._config.deduplication_enabled:
            understanding = self._deduplicate_knowledge(understanding)

        # Complete
        understanding.study_completed_at = datetime.now()
        understanding.comprehension_quality = self._assess_quality(understanding)

        # Mark job as completed with final counts
        self._job_tracker.complete_job(
            self._current_job_id,
            concepts_found=len(understanding.concepts),
            principles_found=len(understanding.principles),
            techniques_found=len(understanding.techniques),
            examples_found=len(understanding.examples),
        )

        # Voice announcement: Complete
        if voice_callback and self._config.voice_announce_complete:
            quality_msg = ""
            if understanding.comprehension_quality >= 0.8:
                quality_msg = VOICE_MESSAGES["quality_high"]
            elif understanding.comprehension_quality >= 0.5:
                quality_msg = VOICE_MESSAGES["quality_medium"]
            else:
                quality_msg = VOICE_MESSAGES["quality_low"]

            voice_callback(
                VOICE_MESSAGES["complete"].format(
                    title=understanding.title,
                    total=understanding.total_knowledge_items,
                    concepts=len(understanding.concepts),
                    principles=len(understanding.principles),
                    techniques=len(understanding.techniques),
                    examples=len(understanding.examples),
                )
                + " "
                + quality_msg
            )

        if progress_callback:
            progress_callback("Complete", 1.0)

        LOGGER.info(
            f"Comprehension complete: {understanding.total_knowledge_items} items extracted "
            f"(LLM calls: {self._llm_call_count})"
        )

        return understanding

    async def _comprehend_thorough(
        self,
        document: Document,
        chunks: List[Chunk],
        understanding: BookUnderstanding,
        progress_callback: Optional[Callable[[str, float], None]] = None,
        voice_callback: Optional[Callable[[str], None]] = None,
    ) -> BookUnderstanding:
        """
        Thorough comprehension: process each chapter separately.

        This provides complete coverage but takes longer and costs more.
        """
        LOGGER.info("Using THOROUGH mode - chapter by chapter processing")

        chapters = document.chapters or []
        total_chapters = len(chapters)

        # Group chunks by chapter
        chapter_chunks = self._group_chunks_by_chapter(chunks, chapters)

        # First: Get overall summary from sampled content
        if progress_callback:
            progress_callback("Extracting summary", 0.0)
        if voice_callback:
            voice_callback(VOICE_MESSAGES["extracting_summary"])

        all_content = self._prepare_content(self._sample_chunks(chunks, 30))
        summary_data = await self._extract_summary(
            understanding.title, understanding.author, all_content
        )
        understanding.summary = summary_data.get("summary", "")
        understanding.main_argument = summary_data.get("main_argument", "")
        understanding.target_audience = summary_data.get("target_audience", "")

        # Process each chapter
        all_concepts: List[Concept] = []
        all_principles: List[Principle] = []
        all_techniques: List[Technique] = []
        all_examples: List[BookExample] = []
        chapter_summaries: List[ChapterSummary] = []

        for idx, chapter in enumerate(chapters):
            chapter_num = idx + 1
            progress = 0.1 + (0.8 * idx / total_chapters)

            if progress_callback:
                progress_callback(f"Chapter {chapter_num}/{total_chapters}", progress)

            # Update job tracker with current chapter
            if self._current_job_id:
                self._job_tracker.update_progress(
                    self._current_job_id,
                    current_chapter=chapter_num,
                    chapter_title=chapter.title,
                )

            if voice_callback and self._config.voice_progress_interval == "chapter":
                voice_callback(
                    VOICE_MESSAGES["chapter_start"].format(
                        num=chapter_num,
                        title=chapter.title,
                    )
                )

            # Get chunks for this chapter
            ch_chunks = chapter_chunks.get(chapter.title, [])
            if not ch_chunks:
                LOGGER.warning(f"No chunks found for chapter: {chapter.title}")
                continue

            # Limit chunks per chapter
            if len(ch_chunks) > self._config.max_chunks_per_chapter:
                ch_chunks = self._sample_chunks(
                    ch_chunks, self._config.max_chunks_per_chapter
                )

            content = self._prepare_content(ch_chunks)

            # Extract knowledge from this chapter
            ch_concepts, ch_principles, ch_techniques, ch_examples, ch_summary = (
                await self._extract_from_chapter(
                    understanding.title,
                    understanding.author,
                    chapter.title,
                    chapter_num,
                    content,
                    document.id,
                )
            )

            all_concepts.extend(ch_concepts)
            all_principles.extend(ch_principles)
            all_techniques.extend(ch_techniques)
            all_examples.extend(ch_examples)

            if ch_summary:
                ch_summary.page_range = (
                    f"pp. {chapter.start_page}-{chapter.end_page}"
                    if chapter.start_page
                    else ""
                )
                chapter_summaries.append(ch_summary)

            # Update job tracker with running totals
            if self._current_job_id:
                self._job_tracker.update_progress(
                    self._current_job_id,
                    concepts_found=len(all_concepts),
                    principles_found=len(all_principles),
                    techniques_found=len(all_techniques),
                    examples_found=len(all_examples),
                    llm_calls_made=self._llm_call_count,
                )

            if voice_callback and self._config.voice_progress_interval == "chapter":
                voice_callback(
                    VOICE_MESSAGES["chapter_complete"].format(
                        num=chapter_num,
                        concepts=len(ch_concepts),
                        principles=len(ch_principles),
                    )
                )

            # Safety check
            if self._llm_call_count >= self._config.max_llm_calls_per_book:
                LOGGER.warning(
                    f"Hit LLM call limit ({self._config.max_llm_calls_per_book})"
                )
                break

        understanding.concepts = all_concepts
        understanding.principles = all_principles
        understanding.techniques = all_techniques
        understanding.examples = all_examples
        understanding.chapters = chapter_summaries

        return understanding

    async def _comprehend_sampled(
        self,
        document: Document,
        chunks: List[Chunk],
        understanding: BookUnderstanding,
        progress_callback: Optional[Callable[[str, float], None]] = None,
        voice_callback: Optional[Callable[[str], None]] = None,
    ) -> BookUnderstanding:
        """
        Sampling comprehension: quick pass using sampled chunks.

        Faster and cheaper but may miss details.
        """
        LOGGER.info("Using SAMPLING mode - quick extraction")

        sampled_chunks = self._sample_chunks(
            chunks, self._config.max_chunks_per_extraction
        )
        content_sample = self._prepare_content(sampled_chunks)

        # Step 1: Extract summary and thesis
        if progress_callback:
            progress_callback("Extracting summary", 0.0)
        if voice_callback:
            voice_callback(VOICE_MESSAGES["extracting_summary"])

        summary_data = await self._extract_summary(
            understanding.title, understanding.author, content_sample
        )
        understanding.summary = summary_data.get("summary", "")
        understanding.main_argument = summary_data.get("main_argument", "")
        understanding.target_audience = summary_data.get("target_audience", "")

        # Step 2: Extract concepts
        if progress_callback:
            progress_callback("Extracting concepts", 0.1)
        if voice_callback:
            voice_callback(VOICE_MESSAGES["extracting_concepts"])

        concepts_data = await self._extract_concepts(
            understanding.title, understanding.author, content_sample
        )
        understanding.concepts = self._parse_concepts(concepts_data, document.id)

        # Step 3: Extract principles
        if progress_callback:
            progress_callback("Extracting principles", 0.3)
        if voice_callback:
            voice_callback(VOICE_MESSAGES["extracting_principles"])

        principles_data = await self._extract_principles(
            understanding.title, understanding.author, content_sample
        )
        understanding.principles = self._parse_principles(principles_data, document.id)

        # Step 4: Extract techniques
        if progress_callback:
            progress_callback("Extracting techniques", 0.5)
        if voice_callback:
            voice_callback(VOICE_MESSAGES["extracting_techniques"])

        techniques_data = await self._extract_techniques(
            understanding.title, understanding.author, content_sample
        )
        understanding.techniques = self._parse_techniques(techniques_data, document.id)

        # Step 5: Extract examples
        if progress_callback:
            progress_callback("Extracting examples", 0.7)
        if voice_callback:
            voice_callback(VOICE_MESSAGES["extracting_examples"])

        examples_data = await self._extract_examples(
            understanding.title, understanding.author, content_sample
        )
        understanding.examples = self._parse_examples(examples_data, document.id)

        return understanding

    async def _extract_from_chapter(
        self,
        book_title: str,
        author: str,
        chapter_title: str,
        chapter_num: int,
        content: str,
        document_id: str,
    ) -> Tuple[
        List[Concept],
        List[Principle],
        List[Technique],
        List[BookExample],
        Optional[ChapterSummary],
    ]:
        """Extract all knowledge types from a single chapter."""

        # Extract chapter summary + all knowledge in one call for efficiency
        prompt = f"""Analyze this chapter from "{book_title}" by {author}.

Chapter {chapter_num}: {chapter_title}

Content:
{content[:25000]}

Extract the following in JSON format:
{{
    "chapter_summary": {{
        "summary": "2-3 sentence summary of what this chapter teaches",
        "key_points": ["point 1", "point 2", ...]
    }},
    "concepts": [
        {{"name": "...", "definition": "...", "importance": "...", "related_concepts": [...]}}
    ],
    "principles": [
        {{"statement": "...", "rationale": "...", "applies_to": [...], "exceptions": [...], "confidence_level": "strong/moderate/suggestion", "check_question": "..."}}
    ],
    "techniques": [
        {{"name": "...", "description": "...", "when_to_use": "...", "example_films": [...]}}
    ],
    "examples": [
        {{"work_title": "...", "scene_or_section": "...", "description": "...", "lesson": "...", "demonstrates": [...]}}
    ]
}}

Extract everything this chapter teaches. Be thorough."""

        response = await self._llm_complete(prompt)
        self._llm_call_count += 1
        data = self._parse_json_response(response)

        # Parse results
        concepts = self._parse_concepts(
            {"concepts": data.get("concepts", [])}, document_id
        )
        principles = self._parse_principles(
            {"principles": data.get("principles", [])}, document_id
        )
        techniques = self._parse_techniques(
            {"techniques": data.get("techniques", [])}, document_id
        )
        examples = self._parse_examples(
            {"examples": data.get("examples", [])}, document_id
        )

        # Create chapter summary
        ch_summary_data = data.get("chapter_summary", {})
        chapter_summary = ChapterSummary(
            number=chapter_num,
            title=chapter_title,
            summary=ch_summary_data.get("summary", ""),
            key_points=ch_summary_data.get("key_points", []),
            concepts_introduced=[c.name for c in concepts],
            principles_taught=[p.statement[:50] for p in principles],
        )

        return concepts, principles, techniques, examples, chapter_summary

    def _group_chunks_by_chapter(
        self,
        chunks: List[Chunk],
        chapters: List[ChapterInfo],
    ) -> Dict[str, List[Chunk]]:
        """Group chunks by their chapter."""
        chapter_chunks: Dict[str, List[Chunk]] = {ch.title: [] for ch in chapters}

        for chunk in chunks:
            if chunk.chapter and chunk.chapter in chapter_chunks:
                chapter_chunks[chunk.chapter].append(chunk)
            elif chapters:
                # Try to assign by page number if chapter not set
                for ch in chapters:
                    # Check if chunk's page range falls within chapter
                    if hasattr(chunk, "page_ids") and chunk.page_ids:
                        # Simplified: assign to first chapter for now
                        chapter_chunks[chapters[0].title].append(chunk)
                        break

        return chapter_chunks

    def _deduplicate_knowledge(
        self, understanding: BookUnderstanding
    ) -> BookUnderstanding:
        """Remove duplicate concepts, principles, etc."""
        # Simple deduplication by name similarity
        seen_concepts = set()
        unique_concepts = []
        for c in understanding.concepts:
            key = c.name.lower().strip()
            if key not in seen_concepts:
                seen_concepts.add(key)
                unique_concepts.append(c)
        understanding.concepts = unique_concepts

        seen_principles = set()
        unique_principles = []
        for p in understanding.principles:
            key = p.statement[:50].lower().strip()
            if key not in seen_principles:
                seen_principles.add(key)
                unique_principles.append(p)
        understanding.principles = unique_principles

        seen_techniques = set()
        unique_techniques = []
        for t in understanding.techniques:
            key = t.name.lower().strip()
            if key not in seen_techniques:
                seen_techniques.add(key)
                unique_techniques.append(t)
        understanding.techniques = unique_techniques

        seen_examples = set()
        unique_examples = []
        for e in understanding.examples:
            key = f"{e.work_title}:{e.scene_or_section}".lower().strip()
            if key not in seen_examples:
                seen_examples.add(key)
                unique_examples.append(e)
        understanding.examples = unique_examples

        LOGGER.info(
            f"Deduplication: concepts {len(understanding.concepts)}, "
            f"principles {len(understanding.principles)}, "
            f"techniques {len(understanding.techniques)}, "
            f"examples {len(understanding.examples)}"
        )

        return understanding

    # =========================================================================
    # Extraction Methods
    # =========================================================================

    async def _extract_summary(
        self, title: str, author: str, content: str
    ) -> Dict[str, Any]:
        """Extract book summary and thesis"""
        prompt = SUMMARY_PROMPT.format(
            title=title,
            author=author,
            content=content[:15000],  # Limit content
        )
        response = await self._llm_complete(prompt)
        self._llm_call_count += 1
        return self._parse_json_response(response)

    async def _extract_concepts(
        self, title: str, author: str, content: str
    ) -> Dict[str, Any]:
        """Extract concepts from book"""
        prompt = CONCEPT_EXTRACTION_PROMPT.format(
            title=title,
            author=author,
            content=content[:20000],
        )
        response = await self._llm_complete(prompt)
        self._llm_call_count += 1
        return self._parse_json_response(response)

    async def _extract_principles(
        self, title: str, author: str, content: str
    ) -> Dict[str, Any]:
        """Extract principles from book"""
        prompt = PRINCIPLE_EXTRACTION_PROMPT.format(
            title=title,
            author=author,
            content=content[:20000],
        )
        response = await self._llm_complete(prompt)
        self._llm_call_count += 1
        return self._parse_json_response(response)

    async def _extract_techniques(
        self, title: str, author: str, content: str
    ) -> Dict[str, Any]:
        """Extract techniques from book"""
        prompt = TECHNIQUE_EXTRACTION_PROMPT.format(
            title=title,
            author=author,
            content=content[:20000],
        )
        response = await self._llm_complete(prompt)
        self._llm_call_count += 1
        return self._parse_json_response(response)

    async def _extract_examples(
        self, title: str, author: str, content: str
    ) -> Dict[str, Any]:
        """Extract examples from book"""
        prompt = EXAMPLE_EXTRACTION_PROMPT.format(
            title=title,
            author=author,
            content=content[:20000],
        )
        response = await self._llm_complete(prompt)
        self._llm_call_count += 1
        return self._parse_json_response(response)

    # =========================================================================
    # Parsing Methods
    # =========================================================================

    def _parse_concepts(self, data: Dict[str, Any], document_id: str) -> List[Concept]:
        """Parse concepts from LLM response"""
        concepts = []
        for c in data.get("concepts", []):
            try:
                concepts.append(
                    Concept(
                        name=c.get("name", ""),
                        definition=c.get("definition", ""),
                        importance=c.get("importance", ""),
                        source_document_id=document_id,
                        related_concepts=c.get("related_concepts", []),
                    )
                )
            except Exception as e:
                LOGGER.warning(f"Failed to parse concept: {e}")
        return concepts

    def _parse_principles(
        self, data: Dict[str, Any], document_id: str
    ) -> List[Principle]:
        """Parse principles from LLM response"""
        principles = []
        for p in data.get("principles", []):
            try:
                confidence_str = p.get("confidence_level", "strong").lower()
                confidence = ConfidenceLevel.STRONG
                if confidence_str == "absolute":
                    confidence = ConfidenceLevel.ABSOLUTE
                elif confidence_str == "moderate":
                    confidence = ConfidenceLevel.MODERATE
                elif confidence_str == "suggestion":
                    confidence = ConfidenceLevel.SUGGESTION

                principles.append(
                    Principle(
                        statement=p.get("statement", ""),
                        rationale=p.get("rationale", ""),
                        source_document_id=document_id,
                        confidence_level=confidence,
                        applies_to=p.get("applies_to", []),
                        exceptions=p.get("exceptions", []),
                        check_question=p.get("check_question", ""),
                    )
                )
            except Exception as e:
                LOGGER.warning(f"Failed to parse principle: {e}")
        return principles

    def _parse_techniques(
        self, data: Dict[str, Any], document_id: str
    ) -> List[Technique]:
        """Parse techniques from LLM response"""
        techniques = []
        for t in data.get("techniques", []):
            try:
                techniques.append(
                    Technique(
                        name=t.get("name", ""),
                        description=t.get("description", ""),
                        source_document_id=document_id,
                        when_to_use=t.get("when_to_use", ""),
                        example_films=t.get("example_films", []),
                    )
                )
            except Exception as e:
                LOGGER.warning(f"Failed to parse technique: {e}")
        return techniques

    def _parse_examples(
        self, data: Dict[str, Any], document_id: str
    ) -> List[BookExample]:
        """Parse examples from LLM response"""
        examples = []
        for e in data.get("examples", []):
            try:
                examples.append(
                    BookExample(
                        work_title=e.get("work_title", ""),
                        scene_or_section=e.get("scene_or_section", ""),
                        source_document_id=document_id,
                        description=e.get("description", ""),
                        lesson=e.get("lesson", ""),
                        demonstrates_concept=e.get("demonstrates", []),
                    )
                )
            except Exception as e_err:
                LOGGER.warning(f"Failed to parse example: {e_err}")
        return examples

    # =========================================================================
    # Helper Methods
    # =========================================================================

    def _sample_chunks(self, chunks: List[Chunk], max_chunks: int) -> List[Chunk]:
        """Sample chunks from throughout the document"""
        if len(chunks) <= max_chunks:
            return chunks

        # Take evenly spaced samples
        step = len(chunks) // max_chunks
        return [chunks[i * step] for i in range(max_chunks)]

    def _prepare_content(self, chunks: List[Chunk]) -> str:
        """Prepare chunk content for LLM"""
        parts = []
        for chunk in chunks:
            if chunk.page_range:
                parts.append(f"[{chunk.page_range}]\n{chunk.content}")
            else:
                parts.append(chunk.content)
        return "\n\n---\n\n".join(parts)

    def _parse_json_response(self, response: str) -> Dict[str, Any]:
        """Parse JSON from LLM response"""
        # Try to extract JSON from response
        try:
            # Look for JSON in response
            start = response.find("{")
            end = response.rfind("}") + 1
            if start >= 0 and end > start:
                return json.loads(response[start:end])
        except json.JSONDecodeError:
            pass

        # Try to find JSON array
        try:
            start = response.find("[")
            end = response.rfind("]") + 1
            if start >= 0 and end > start:
                return {"items": json.loads(response[start:end])}
        except json.JSONDecodeError:
            pass

        LOGGER.warning(f"Failed to parse JSON from response: {response[:200]}")
        return {}

    def _infer_domains(self, understanding: BookUnderstanding) -> List[str]:
        """Infer domains from extracted knowledge"""
        domains = set()

        # Check for keywords in concepts and principles
        text = " ".join(
            [
                c.name.lower() + " " + c.definition.lower()
                for c in understanding.concepts
            ]
            + [p.statement.lower() for p in understanding.principles]
        )

        domain_keywords = {
            "court_drama": ["court", "trial", "witness", "lawyer", "jury", "testimony"],
            "dialogue": ["dialogue", "conversation", "subtext", "banter"],
            "structure": ["act", "structure", "scene", "sequence", "plot"],
            "character": ["character", "protagonist", "antagonist", "arc"],
            "tension": ["tension", "suspense", "conflict", "stakes"],
            "comedy": ["comedy", "humor", "joke", "timing"],
            "action": ["action", "chase", "fight", "stunt"],
        }

        for domain, keywords in domain_keywords.items():
            if any(kw in text for kw in keywords):
                domains.add(domain)

        return list(domains) or ["screenwriting"]

    def _assess_quality(self, understanding: BookUnderstanding) -> float:
        """Assess comprehension quality (0-1)"""
        score = 0.0

        # Has summary
        if understanding.summary:
            score += 0.2
        if understanding.main_argument:
            score += 0.1

        # Has concepts (expected 5-15)
        concept_count = len(understanding.concepts)
        if 5 <= concept_count <= 20:
            score += 0.2
        elif concept_count > 0:
            score += 0.1

        # Has principles (expected 10-20)
        principle_count = len(understanding.principles)
        if 10 <= principle_count <= 25:
            score += 0.2
        elif principle_count > 0:
            score += 0.1

        # Has techniques
        if len(understanding.techniques) >= 5:
            score += 0.15
        elif len(understanding.techniques) > 0:
            score += 0.05

        # Has examples
        if len(understanding.examples) >= 5:
            score += 0.15
        elif len(understanding.examples) > 0:
            score += 0.05

        return min(1.0, score)
