"""
Citation Tracker

Handles citation management and formatting for document-grounded responses.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from documents.models import Chunk, Citation, Document, DocumentSearchResult


class CitationTracker:
    """
    Tracks and formats citations for document-grounded responses.

    Used by the orchestrator to inject document context into LLM prompts
    and format citations in responses.
    """

    def __init__(self, style: str = "inline"):
        """
        Initialize citation tracker.

        Args:
            style: Citation style ("inline", "footnote", "endnote")
        """
        self.style = style
        self._citations: List[Citation] = []
        self._citation_map: Dict[str, int] = {}  # chunk_id -> citation number

    def clear(self) -> None:
        """Clear all citations"""
        self._citations = []
        self._citation_map = {}

    def add_citation(
        self,
        chunk: Chunk,
        document: Document,
        relevance: float,
        quote: Optional[str] = None,
    ) -> int:
        """
        Add a citation and return its number (1-indexed).

        Args:
            chunk: Source chunk
            document: Source document
            relevance: Relevance score (0-1)
            quote: Relevant quote from the chunk (auto-extracted if not provided)

        Returns:
            Citation number for reference
        """
        # Check if already cited
        if chunk.id in self._citation_map:
            return self._citation_map[chunk.id]

        # Extract quote if not provided
        if not quote:
            quote = self._extract_quote(chunk.content)

        citation = Citation(
            document_id=document.id,
            document_title=document.metadata.title,
            chunk_id=chunk.id,
            page_range=chunk.page_range,
            chapter=chunk.chapter,
            section=chunk.section,
            quote=quote,
            relevance=relevance,
        )

        self._citations.append(citation)
        citation_num = len(self._citations)
        self._citation_map[chunk.id] = citation_num

        return citation_num

    def get_citation_number(self, chunk_id: str) -> Optional[int]:
        """Get citation number for a chunk"""
        return self._citation_map.get(chunk_id)

    def format_inline_reference(self, chunk_id: str) -> str:
        """Format inline citation reference [Title, p. X]"""
        num = self._citation_map.get(chunk_id)
        if not num:
            return ""

        citation = self._citations[num - 1]
        return citation.format_inline()

    def format_numbered_reference(self, chunk_id: str) -> str:
        """Format numbered reference [1]"""
        num = self._citation_map.get(chunk_id)
        if not num:
            return ""
        return f"[{num}]"

    def format_bibliography(self) -> str:
        """Format full bibliography/sources section"""
        if not self._citations:
            return ""

        lines = ["\n---", "**Sources:**"]

        for i, citation in enumerate(self._citations, 1):
            parts = [f"{i}. *{citation.document_title}*"]

            if citation.chapter:
                parts.append(f", {citation.chapter}")

            parts.append(f", {citation.page_range}")

            if self.style == "footnote" and citation.quote:
                parts.append(f'\n   > "{citation.quote[:100]}..."')

            lines.append("".join(parts))

        return "\n".join(lines)

    def get_context_with_citations(
        self,
        results: List[DocumentSearchResult],
        max_chars: int = 4000,
        include_quotes: bool = True,
    ) -> Tuple[str, List[Citation]]:
        """
        Build context string with embedded citation markers.

        Used by orchestrator to inject document knowledge into LLM context.

        Args:
            results: Search results to include
            max_chars: Maximum characters for context
            include_quotes: Whether to include quotes in context

        Returns:
            Tuple of (context_string, citation_list)
        """
        self.clear()  # Start fresh
        context_parts = []
        current_chars = 0

        for result in results:
            if current_chars >= max_chars:
                break

            # Add citation
            num = self.add_citation(
                result.chunk,
                result.document,
                result.similarity,
                result.highlight,
            )

            # Build context block
            if include_quotes:
                block = (
                    f"[{num}] From *{result.document.metadata.title}* "
                    f"({result.chunk.page_range}):\n"
                    f"{result.chunk.content}\n"
                )
            else:
                block = (
                    f"[{num}] {result.document.metadata.title}, "
                    f"{result.chunk.page_range}: "
                    f"{self._extract_quote(result.chunk.content, max_length=150)}\n"
                )

            # Check if we can fit this block
            if current_chars + len(block) > max_chars:
                # Truncate to fit
                available = max_chars - current_chars - 10
                if available > 100:
                    block = block[:available] + "...\n"
                else:
                    break

            context_parts.append(block)
            current_chars += len(block)

        return "\n".join(context_parts), self._citations.copy()

    def format_response_with_citations(
        self,
        response: str,
        add_bibliography: bool = True,
    ) -> str:
        """
        Add citation references to a response.

        Can be called after get_context_with_citations to append sources.
        """
        if not self._citations:
            return response

        if add_bibliography:
            return response + self.format_bibliography()

        return response

    def _extract_quote(self, content: str, max_length: int = 200) -> str:
        """Extract a representative quote from content"""
        # Clean up content
        quote = content.strip()

        # Remove markdown formatting
        quote = quote.replace("**", "").replace("*", "")
        quote = quote.replace("#", "").replace("`", "")

        # Get first meaningful sentence or paragraph
        lines = quote.split("\n")
        for line in lines:
            line = line.strip()
            if len(line) > 20:  # Skip short lines
                quote = line
                break

        # Truncate if needed
        if len(quote) > max_length:
            # Try to end at sentence boundary
            truncate_point = max_length
            for end in [". ", "? ", "! "]:
                idx = quote.rfind(end, 0, max_length)
                if idx > max_length // 2:
                    truncate_point = idx + 1
                    break

            quote = quote[:truncate_point].strip()
            if not quote.endswith((".", "?", "!")):
                quote += "..."

        return quote

    @property
    def citations(self) -> List[Citation]:
        """Get all citations"""
        return self._citations.copy()

    @property
    def count(self) -> int:
        """Get number of citations"""
        return len(self._citations)
