"""
PDF Processor

Handles PDF to image conversion for OCR processing.
"""

from __future__ import annotations

import hashlib
import logging
import os
import shutil
from pathlib import Path
from typing import List, Optional, Tuple

from documents.config import StorageConfig, get_document_config

LOGGER = logging.getLogger(__name__)


class PDFProcessor:
    """
    PDF to image conversion for OCR.

    Handles:
    - PDF page counting
    - PDF to image conversion (using pdf2image)
    - Image optimization for OCR
    - Cleanup of temporary images
    """

    def __init__(self, config: Optional[StorageConfig] = None):
        self.config = config or get_document_config().storage
        self._images_dir = Path(self.config.images_dir)
        self._images_dir.mkdir(parents=True, exist_ok=True)

    def get_pdf_info(self, pdf_path: str) -> Tuple[int, str, int]:
        """
        Get basic PDF information.

        Returns:
            Tuple of (page_count, file_hash, file_size)
        """
        path = Path(pdf_path)
        if not path.exists():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")

        # Calculate file hash
        file_hash = self._calculate_hash(path)
        file_size = path.stat().st_size

        # Get page count
        try:
            from pdf2image import pdfinfo_from_path

            info = pdfinfo_from_path(str(path))
            page_count = info.get("Pages", 0)
        except ImportError:
            LOGGER.warning("pdf2image not installed, using PyPDF2 for page count")
            page_count = self._get_page_count_pypdf(path)
        except Exception as e:
            LOGGER.error("Failed to get PDF info: %s", e)
            page_count = self._get_page_count_pypdf(path)

        return page_count, file_hash, file_size

    def _calculate_hash(self, path: Path) -> str:
        """Calculate SHA256 hash of file"""
        sha256_hash = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                sha256_hash.update(chunk)
        return sha256_hash.hexdigest()

    def _get_page_count_pypdf(self, path: Path) -> int:
        """Fallback page count using PyPDF2"""
        try:
            from PyPDF2 import PdfReader

            reader = PdfReader(str(path))
            return len(reader.pages)
        except ImportError:
            LOGGER.error("Neither pdf2image nor PyPDF2 available")
            return 0
        except Exception as e:
            LOGGER.error("PyPDF2 failed: %s", e)
            return 0

    def convert_to_images(
        self,
        pdf_path: str,
        document_id: str,
        dpi: int = 300,
        start_page: int = 1,
        end_page: Optional[int] = None,
    ) -> List[str]:
        """
        Convert PDF pages to images.

        Args:
            pdf_path: Path to PDF file
            document_id: Document ID for organizing images
            dpi: Resolution for conversion
            start_page: First page (1-indexed)
            end_page: Last page (inclusive), None for all pages

        Returns:
            List of image file paths
        """
        path = Path(pdf_path)
        if not path.exists():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")

        # Create output directory
        output_dir = self._images_dir / document_id
        output_dir.mkdir(parents=True, exist_ok=True)

        try:
            from pdf2image import convert_from_path

            # Convert pages
            images = convert_from_path(
                str(path),
                dpi=dpi,
                first_page=start_page,
                last_page=end_page,
                fmt="png",
                thread_count=os.cpu_count() or 4,
            )

            # Save images
            image_paths = []
            for i, image in enumerate(images):
                page_num = start_page + i
                image_path = output_dir / f"page_{page_num:04d}.png"
                image.save(str(image_path), "PNG")
                image_paths.append(str(image_path))
                LOGGER.debug("Saved page %d to %s", page_num, image_path)

            LOGGER.info("Converted %d pages from %s", len(image_paths), path.name)
            return image_paths

        except ImportError:
            LOGGER.error("pdf2image not installed. Install with: pip install pdf2image")
            raise
        except Exception as e:
            LOGGER.error("PDF conversion failed: %s", e)
            raise

    def convert_page(
        self,
        pdf_path: str,
        document_id: str,
        page_number: int,
        dpi: int = 300,
    ) -> str:
        """
        Convert a single page to image.

        Returns image file path.
        """
        paths = self.convert_to_images(
            pdf_path=pdf_path,
            document_id=document_id,
            dpi=dpi,
            start_page=page_number,
            end_page=page_number,
        )
        return paths[0] if paths else ""

    def cleanup_images(self, document_id: str) -> int:
        """
        Clean up converted images for a document.

        Returns number of files deleted.
        """
        output_dir = self._images_dir / document_id
        if not output_dir.exists():
            return 0

        count = 0
        for file in output_dir.iterdir():
            try:
                file.unlink()
                count += 1
            except Exception as e:
                LOGGER.warning("Failed to delete %s: %s", file, e)

        try:
            output_dir.rmdir()
        except Exception:
            pass  # Directory not empty or other error

        LOGGER.info("Cleaned up %d images for document %s", count, document_id)
        return count

    def cleanup_all(self) -> int:
        """Clean up all image directories"""
        count = 0
        for doc_dir in self._images_dir.iterdir():
            if doc_dir.is_dir():
                count += self.cleanup_images(doc_dir.name)
        return count

    def get_image_paths(self, document_id: str) -> List[str]:
        """Get existing image paths for a document"""
        output_dir = self._images_dir / document_id
        if not output_dir.exists():
            return []

        paths = sorted(output_dir.glob("page_*.png"))
        return [str(p) for p in paths]

    def copy_pdf_to_storage(self, source_path: str, document_id: str) -> str:
        """
        Copy PDF to storage directory.

        Returns path to stored PDF.
        """
        source = Path(source_path)
        if not source.exists():
            raise FileNotFoundError(f"Source PDF not found: {source_path}")

        docs_dir = Path(self.config.documents_dir)
        docs_dir.mkdir(parents=True, exist_ok=True)

        # Use document_id + original extension
        dest = docs_dir / f"{document_id}{source.suffix}"
        shutil.copy2(source, dest)

        LOGGER.info("Copied PDF to %s", dest)
        return str(dest)
