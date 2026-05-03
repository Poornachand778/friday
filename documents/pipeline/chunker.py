"""
Semantic Chunker

Implements intelligent chunking strategies for document text.
"""

from __future__ import annotations

import logging
import re
from typing import Dict, List, Optional, Tuple

from documents.config import ChunkingConfig, get_document_config
from documents.models import Chunk, ChapterInfo, Page

LOGGER = logging.getLogger(__name__)


class SemanticChunker:
    """
    Semantic chunking with chapter/section boundary detection.

    Strategies:
    - semantic: Respects natural boundaries (paragraphs, sections)
    - fixed: Fixed-size chunks with overlap
    - hybrid: Combines both approaches

    Special handling for:
    - Screenplay format (scene-based)
    - Telugu and mixed-language text
    - Tables and code blocks
    """

    def __init__(self, config: Optional[ChunkingConfig] = None):
        self.config = config or get_document_config().chunking

        # Compile chapter patterns
        self._chapter_patterns = [
            re.compile(pattern, re.IGNORECASE)
            for pattern in self.config.chapter_patterns
        ]

        # Screenplay patterns
        self._scene_pattern = re.compile(
            r"^(INT\.|EXT\.|INT/EXT\.)\s+.+", re.MULTILINE | re.IGNORECASE
        )
        self._dialogue_pattern = re.compile(r"^\s*([A-Z][A-Z\s\.]+)\s*$", re.MULTILINE)

    def chunk_document(
        self,
        pages: List[Page],
        document_id: str,
        chapters: Optional[List[ChapterInfo]] = None,
    ) -> List[Chunk]:
        """
        Create semantic chunks from document pages.

        Args:
            pages: List of Page objects with OCR text
            document_id: Document ID for chunk references
            chapters: Optional chapter info for boundary detection

        Returns:
            List of Chunk objects ready for embedding
        """
        if self.config.screenplay_mode:
            return self._chunk_screenplay(pages, document_id)

        if self.config.strategy == "fixed":
            return self._chunk_fixed(pages, document_id)
        elif self.config.strategy == "hybrid":
            return self._chunk_hybrid(pages, document_id, chapters)
        else:  # semantic
            return self._chunk_semantic(pages, document_id, chapters)

    def _chunk_semantic(
        self,
        pages: List[Page],
        document_id: str,
        chapters: Optional[List[ChapterInfo]] = None,
    ) -> List[Chunk]:
        """Semantic chunking with boundary detection"""
        chunks: List[Chunk] = []
        current_text = ""
        current_page_ids: List[str] = []
        current_page_numbers: List[int] = []
        current_chapter: Optional[str] = None
        chunk_index = 0

        # Build page number to chapter mapping
        page_to_chapter: Dict[int, str] = {}
        if chapters:
            for ch in chapters:
                for p in range(ch.start_page, ch.end_page + 1):
                    page_to_chapter[p] = ch.title

        for page in pages:
            text = page.cleaned_text

            # Check for chapter boundary
            if self.config.respect_chapters:
                chapter = page_to_chapter.get(page.page_number)
                if chapter and chapter != current_chapter:
                    # Flush current chunk on chapter change
                    if current_text.strip():
                        chunks.append(
                            self._create_chunk(
                                document_id=document_id,
                                page_ids=current_page_ids.copy(),
                                page_numbers=current_page_numbers.copy(),
                                content=current_text.strip(),
                                chunk_index=chunk_index,
                                chapter=current_chapter,
                            )
                        )
                        chunk_index += 1

                    current_text = ""
                    current_page_ids = []
                    current_page_numbers = []
                    current_chapter = chapter

            # Check for section headers
            detected_header = self._detect_section_header(text)
            if detected_header and self.config.respect_sections:
                # Flush on section change if chunk is substantial
                if len(current_text) >= self.config.min_chunk_chars:
                    chunks.append(
                        self._create_chunk(
                            document_id=document_id,
                            page_ids=current_page_ids.copy(),
                            page_numbers=current_page_numbers.copy(),
                            content=current_text.strip(),
                            chunk_index=chunk_index,
                            chapter=current_chapter,
                        )
                    )
                    chunk_index += 1
                    current_text = ""
                    current_page_ids = []
                    current_page_numbers = []

            # Add page content
            current_text += text + "\n\n"
            current_page_ids.append(page.id)
            current_page_numbers.append(page.page_number)

            # Check if we need to split
            while len(current_text) > self.config.max_chunk_chars:
                split_point = self._find_split_point(current_text)

                # Create chunk from first part
                chunk_text = current_text[:split_point].strip()
                if chunk_text:
                    chunks.append(
                        self._create_chunk(
                            document_id=document_id,
                            page_ids=current_page_ids.copy(),
                            page_numbers=current_page_numbers.copy(),
                            content=chunk_text,
                            chunk_index=chunk_index,
                            chapter=current_chapter,
                        )
                    )
                    chunk_index += 1

                # Keep overlap for continuity, ensure forward progress
                overlap_start = max(1, split_point - self.config.overlap_chars)
                current_text = current_text[overlap_start:]

        # Flush remaining text
        if (
            current_text.strip()
            and len(current_text.strip()) >= self.config.min_chunk_chars // 2
        ):
            chunks.append(
                self._create_chunk(
                    document_id=document_id,
                    page_ids=current_page_ids,
                    page_numbers=current_page_numbers,
                    content=current_text.strip(),
                    chunk_index=chunk_index,
                    chapter=current_chapter,
                )
            )

        LOGGER.info("Created %d semantic chunks from %d pages", len(chunks), len(pages))
        return chunks

    def _chunk_fixed(
        self,
        pages: List[Page],
        document_id: str,
    ) -> List[Chunk]:
        """Fixed-size chunking with overlap"""
        # Combine all text
        full_text = "\n\n".join(p.cleaned_text for p in pages)
        page_map = self._build_page_map(pages)

        chunks: List[Chunk] = []
        chunk_index = 0
        pos = 0

        while pos < len(full_text):
            # Get chunk of max size
            end = min(pos + self.config.max_chunk_chars, len(full_text))

            # Find word boundary if not at end
            if end < len(full_text):
                space = full_text.rfind(" ", pos, end)
                if space > pos:
                    end = space

            chunk_text = full_text[pos:end].strip()

            if chunk_text:
                # Find pages that contain this text
                page_ids, page_numbers = self._find_pages_for_range(pos, end, page_map)

                chunks.append(
                    self._create_chunk(
                        document_id=document_id,
                        page_ids=page_ids,
                        page_numbers=page_numbers,
                        content=chunk_text,
                        chunk_index=chunk_index,
                    )
                )
                chunk_index += 1

            # Advance position with overlap for continuity
            if end >= len(full_text):
                break  # Processed everything
            pos = end - self.config.overlap_chars
            if pos <= 0:
                pos = end

        LOGGER.info("Created %d fixed chunks from %d pages", len(chunks), len(pages))
        return chunks

    def _chunk_hybrid(
        self,
        pages: List[Page],
        document_id: str,
        chapters: Optional[List[ChapterInfo]] = None,
    ) -> List[Chunk]:
        """
        Hybrid approach: semantic at high level, fixed at low level.

        First splits by chapters/sections, then applies fixed chunking within.
        """
        # Group pages by chapter
        chapter_pages: Dict[str, List[Page]] = {"default": []}

        if chapters:
            for page in pages:
                for ch in chapters:
                    if ch.start_page <= page.page_number <= ch.end_page:
                        if ch.title not in chapter_pages:
                            chapter_pages[ch.title] = []
                        chapter_pages[ch.title].append(page)
                        break
                else:
                    chapter_pages["default"].append(page)
        else:
            chapter_pages["default"] = pages

        # Apply fixed chunking within each chapter
        all_chunks: List[Chunk] = []
        chunk_index = 0

        for chapter, ch_pages in chapter_pages.items():
            if not ch_pages:
                continue

            chunks = self._chunk_fixed(ch_pages, document_id)

            # Update chapter info and indices
            for chunk in chunks:
                chunk.chapter = chapter if chapter != "default" else None
                chunk.chunk_index = chunk_index
                chunk_index += 1

            all_chunks.extend(chunks)

        LOGGER.info(
            "Created %d hybrid chunks from %d pages", len(all_chunks), len(pages)
        )
        return all_chunks

    def _chunk_screenplay(
        self,
        pages: List[Page],
        document_id: str,
    ) -> List[Chunk]:
        """
        Screenplay-specific chunking by scenes.

        Keeps scene headings with their content.
        Optionally groups dialogue sequences.
        """
        # Combine all text
        full_text = "\n".join(p.cleaned_text for p in pages)
        page_map = self._build_page_map(pages)

        chunks: List[Chunk] = []
        chunk_index = 0

        # Split by scene headings
        scene_splits = list(self._scene_pattern.finditer(full_text))

        for i, match in enumerate(scene_splits):
            # Get scene content
            start = match.start()
            end = (
                scene_splits[i + 1].start()
                if i + 1 < len(scene_splits)
                else len(full_text)
            )

            scene_text = full_text[start:end].strip()

            if scene_text:
                # Find pages
                page_ids, page_numbers = self._find_pages_for_range(
                    start, end, page_map
                )

                # Extract scene heading as section
                section = match.group(0).strip()

                chunks.append(
                    self._create_chunk(
                        document_id=document_id,
                        page_ids=page_ids,
                        page_numbers=page_numbers,
                        content=scene_text,
                        chunk_index=chunk_index,
                        section=section,
                    )
                )
                chunk_index += 1

        LOGGER.info(
            "Created %d screenplay scene chunks from %d pages",
            len(chunks),
            len(pages),
        )
        return chunks

    def _find_split_point(self, text: str) -> int:
        """
        Find optimal split point near max_chars.

        Priority: paragraph > sentence > word
        """
        target = self.config.max_chunk_chars

        # Look for paragraph break (double newline)
        para_match = text.rfind("\n\n", self.config.min_chunk_chars, target + 200)
        if para_match > self.config.min_chunk_chars:
            return para_match + 2

        # Look for sentence end
        for punct in [". ", "? ", "! ", ".\n", "?\n", "!\n"]:
            sent = text.rfind(punct, self.config.min_chunk_chars, target)
            if sent > self.config.min_chunk_chars:
                return sent + len(punct)

        # Fall back to word boundary
        space = text.rfind(" ", self.config.min_chunk_chars, target)
        if space > 0:
            return space + 1

        # Last resort: hard cut
        return target

    def _detect_section_header(self, text: str) -> Optional[str]:
        """Detect if text starts with a section header"""
        lines = text.strip().split("\n")
        if not lines:
            return None

        first_line = lines[0].strip()

        # Check markdown headers
        if first_line.startswith("#"):
            return first_line.lstrip("#").strip()

        # Check chapter patterns
        for pattern in self._chapter_patterns:
            if pattern.match(first_line):
                return first_line

        # Check bold text that might be a header
        if first_line.startswith("**") and first_line.endswith("**"):
            header = first_line.strip("*").strip()
            if len(header) < 100:  # Reasonable header length
                return header

        return None

    def _build_page_map(self, pages: List[Page]) -> List[Tuple[int, int, str, int]]:
        """
        Build character position to page mapping.

        Returns list of (start_pos, end_pos, page_id, page_number)
        """
        page_map = []
        pos = 0

        for page in pages:
            text = page.cleaned_text
            start = pos
            end = pos + len(text) + 2  # +2 for \n\n
            page_map.append((start, end, page.id, page.page_number))
            pos = end

        return page_map

    def _find_pages_for_range(
        self,
        start: int,
        end: int,
        page_map: List[Tuple[int, int, str, int]],
    ) -> Tuple[List[str], List[int]]:
        """Find pages that overlap with character range"""
        page_ids = []
        page_numbers = []

        for p_start, p_end, p_id, p_num in page_map:
            # Check for overlap
            if p_start < end and p_end > start:
                page_ids.append(p_id)
                page_numbers.append(p_num)

        return page_ids, page_numbers

    def _create_chunk(
        self,
        document_id: str,
        page_ids: List[str],
        page_numbers: List[int],
        content: str,
        chunk_index: int,
        chapter: Optional[str] = None,
        section: Optional[str] = None,
    ) -> Chunk:
        """Create a Chunk object with page range"""
        # Format page range
        if page_numbers:
            if len(page_numbers) == 1:
                page_range = f"p. {page_numbers[0]}"
            else:
                page_range = f"pp. {min(page_numbers)}-{max(page_numbers)}"
        else:
            page_range = ""

        return Chunk.create(
            document_id=document_id,
            page_ids=page_ids,
            content=content,
            page_range=page_range,
            chunk_index=chunk_index,
            chapter=chapter,
            section=section,
        )
