"""
Tests for PDF Processor
============================

Tests PDF info extraction, image conversion, cleanup, storage copy,
and hash calculation -- with extensive mocking of pdf2image and PyPDF2.

Run with: pytest tests/test_pdf_processor.py -v
"""

import sys
import hashlib
import os
import shutil
from pathlib import Path
from unittest.mock import MagicMock, patch, PropertyMock

import pytest

# Add repo root to path
sys.path.insert(0, str(Path(__file__).parents[1]))

from documents.config import StorageConfig
from documents.pipeline.pdf_processor import PDFProcessor


# =========================================================================
# Helpers / Fixtures
# =========================================================================


def _make_storage_config(tmp_path):
    """Create a StorageConfig pointing at tmp_path subdirectories."""
    return StorageConfig(
        db_path=str(tmp_path / "documents.db"),
        documents_dir=str(tmp_path / "raw"),
        images_dir=str(tmp_path / "images"),
    )


def _create_fake_pdf(
    directory, name="sample.pdf", content=b"%PDF-1.4 fake pdf content"
):
    """Create a small fake PDF file and return its Path."""
    path = Path(directory) / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    return path


@pytest.fixture
def storage_config(tmp_path):
    """Provide a StorageConfig rooted under tmp_path."""
    return _make_storage_config(tmp_path)


@pytest.fixture
def processor(storage_config):
    """Provide a PDFProcessor with tmp-based config."""
    return PDFProcessor(config=storage_config)


@pytest.fixture
def fake_pdf(tmp_path):
    """Provide a fake PDF file path."""
    return _create_fake_pdf(tmp_path)


# =========================================================================
# TestPDFProcessorInit
# =========================================================================


class TestPDFProcessorInit:
    """Initialization and configuration tests."""

    def test_default_config_creates_images_dir(self, tmp_path):
        """Processor with explicit config creates the images directory."""
        cfg = _make_storage_config(tmp_path)
        proc = PDFProcessor(config=cfg)
        assert Path(proc.config.images_dir).exists()
        assert Path(proc.config.images_dir).is_dir()

    def test_custom_config_stored(self, tmp_path):
        """Custom StorageConfig is stored on the processor."""
        cfg = _make_storage_config(tmp_path)
        proc = PDFProcessor(config=cfg)
        assert proc.config is cfg

    def test_images_dir_attribute_matches_config(self, storage_config):
        """Internal _images_dir matches config.images_dir."""
        proc = PDFProcessor(config=storage_config)
        assert proc._images_dir == Path(storage_config.images_dir)

    def test_images_dir_created_even_if_nested(self, tmp_path):
        """Nested images directory is created via parents=True."""
        cfg = StorageConfig(
            db_path=str(tmp_path / "db.sqlite"),
            documents_dir=str(tmp_path / "docs"),
            images_dir=str(tmp_path / "deep" / "nested" / "images"),
        )
        proc = PDFProcessor(config=cfg)
        assert proc._images_dir.exists()

    @patch("documents.pipeline.pdf_processor.get_document_config")
    def test_none_config_falls_back_to_default(self, mock_get_cfg, tmp_path):
        """When config=None, get_document_config() is used."""
        cfg = _make_storage_config(tmp_path)
        mock_doc_cfg = MagicMock()
        mock_doc_cfg.storage = cfg
        mock_get_cfg.return_value = mock_doc_cfg

        proc = PDFProcessor(config=None)
        mock_get_cfg.assert_called_once()
        assert proc.config is cfg


# =========================================================================
# TestCalculateHash
# =========================================================================


class TestCalculateHash:
    """SHA-256 file hashing tests."""

    def test_returns_sha256_hex_string(self, processor, fake_pdf):
        """Hash is a 64-character lowercase hex string."""
        result = processor._calculate_hash(fake_pdf)
        assert isinstance(result, str)
        assert len(result) == 64
        assert all(c in "0123456789abcdef" for c in result)

    def test_correct_sha256_value(self, processor, fake_pdf):
        """Hash matches independently computed SHA-256."""
        expected = hashlib.sha256(fake_pdf.read_bytes()).hexdigest()
        assert processor._calculate_hash(fake_pdf) == expected

    def test_same_file_returns_same_hash(self, processor, fake_pdf):
        """Calling twice on same file gives identical result."""
        h1 = processor._calculate_hash(fake_pdf)
        h2 = processor._calculate_hash(fake_pdf)
        assert h1 == h2

    def test_different_content_returns_different_hash(self, processor, tmp_path):
        """Files with different content produce different hashes."""
        pdf_a = _create_fake_pdf(tmp_path, "a.pdf", b"content AAA")
        pdf_b = _create_fake_pdf(tmp_path, "b.pdf", b"content BBB")
        assert processor._calculate_hash(pdf_a) != processor._calculate_hash(pdf_b)

    def test_hash_reads_file_in_chunks(self, processor, tmp_path):
        """Large file is hashed correctly (verifies chunked reading works)."""
        big_file = tmp_path / "big.pdf"
        data = b"x" * 100_000  # > 8192 bytes so multiple chunks
        big_file.write_bytes(data)
        expected = hashlib.sha256(data).hexdigest()
        assert processor._calculate_hash(big_file) == expected


# =========================================================================
# TestGetPdfInfo
# =========================================================================


class TestGetPdfInfo:
    """Tests for get_pdf_info which returns (page_count, hash, size)."""

    def test_file_not_found_raises(self, processor):
        """Non-existent path raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError, match="PDF not found"):
            processor.get_pdf_info("/no/such/file.pdf")

    def test_returns_tuple_of_three(self, processor, fake_pdf):
        """Return value is a 3-tuple."""
        with patch(
            "documents.pipeline.pdf_processor.PDFProcessor._get_page_count_pypdf",
            return_value=5,
        ):
            with patch.dict("sys.modules", {"pdf2image": None}):
                result = processor.get_pdf_info(str(fake_pdf))
        assert isinstance(result, tuple)
        assert len(result) == 3

    @patch("documents.pipeline.pdf_processor.PDFProcessor._get_page_count_pypdf")
    def test_pdf2image_import_error_falls_back_to_pypdf(
        self, mock_pypdf, processor, fake_pdf
    ):
        """When pdf2image cannot be imported, PyPDF2 fallback is used."""
        mock_pypdf.return_value = 7

        # Force the inner import of pdf2image to raise ImportError
        original_import = (
            __builtins__.__import__
            if hasattr(__builtins__, "__import__")
            else __import__
        )

        def fake_import(name, *args, **kwargs):
            if name == "pdf2image":
                raise ImportError("No module named 'pdf2image'")
            return original_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=fake_import):
            page_count, file_hash, file_size = processor.get_pdf_info(str(fake_pdf))

        assert page_count == 7
        mock_pypdf.assert_called_once()

    @patch("documents.pipeline.pdf_processor.PDFProcessor._get_page_count_pypdf")
    def test_pdf2image_exception_falls_back_to_pypdf(
        self, mock_pypdf, processor, fake_pdf
    ):
        """When pdf2image raises a runtime error, PyPDF2 fallback is used."""
        mock_pypdf.return_value = 3

        original_import = (
            __builtins__.__import__
            if hasattr(__builtins__, "__import__")
            else __import__
        )

        # Allow import but make pdfinfo_from_path raise
        mock_pdfinfo_mod = MagicMock()
        mock_pdfinfo_mod.pdfinfo_from_path = MagicMock(
            side_effect=RuntimeError("poppler missing")
        )

        def fake_import(name, *args, **kwargs):
            if name == "pdf2image":
                return mock_pdfinfo_mod
            return original_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=fake_import):
            page_count, _, _ = processor.get_pdf_info(str(fake_pdf))

        assert page_count == 3
        mock_pypdf.assert_called_once()

    def test_hash_and_size_correct(self, processor, fake_pdf):
        """Hash and file_size are accurate regardless of page count source."""
        expected_hash = hashlib.sha256(fake_pdf.read_bytes()).hexdigest()
        expected_size = fake_pdf.stat().st_size

        with patch(
            "documents.pipeline.pdf_processor.PDFProcessor._get_page_count_pypdf",
            return_value=1,
        ):
            original_import = (
                __builtins__.__import__
                if hasattr(__builtins__, "__import__")
                else __import__
            )

            def fake_import(name, *args, **kwargs):
                if name == "pdf2image":
                    raise ImportError("nope")
                return original_import(name, *args, **kwargs)

            with patch("builtins.__import__", side_effect=fake_import):
                _, file_hash, file_size = processor.get_pdf_info(str(fake_pdf))

        assert file_hash == expected_hash
        assert file_size == expected_size

    def test_pdf2image_success_returns_page_count(self, processor, fake_pdf):
        """When pdf2image works, its page count is used."""
        original_import = (
            __builtins__.__import__
            if hasattr(__builtins__, "__import__")
            else __import__
        )

        mock_pdfinfo_mod = MagicMock()
        mock_pdfinfo_mod.pdfinfo_from_path = MagicMock(return_value={"Pages": 42})

        def fake_import(name, *args, **kwargs):
            if name == "pdf2image":
                return mock_pdfinfo_mod
            return original_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=fake_import):
            page_count, _, _ = processor.get_pdf_info(str(fake_pdf))

        assert page_count == 42

    def test_both_backends_missing_returns_zero_pages(self, processor, fake_pdf):
        """When both pdf2image and PyPDF2 are unavailable, page_count is 0."""
        original_import = (
            __builtins__.__import__
            if hasattr(__builtins__, "__import__")
            else __import__
        )

        def fake_import(name, *args, **kwargs):
            if name in ("pdf2image", "PyPDF2"):
                raise ImportError(f"No module named '{name}'")
            return original_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=fake_import):
            page_count, _, _ = processor.get_pdf_info(str(fake_pdf))

        assert page_count == 0

    def test_pdf2image_pages_key_missing_returns_zero(self, processor, fake_pdf):
        """When pdfinfo_from_path returns dict without 'Pages', page_count is 0."""
        original_import = (
            __builtins__.__import__
            if hasattr(__builtins__, "__import__")
            else __import__
        )

        mock_pdfinfo_mod = MagicMock()
        mock_pdfinfo_mod.pdfinfo_from_path = MagicMock(return_value={})

        def fake_import(name, *args, **kwargs):
            if name == "pdf2image":
                return mock_pdfinfo_mod
            return original_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=fake_import):
            page_count, _, _ = processor.get_pdf_info(str(fake_pdf))

        assert page_count == 0


# =========================================================================
# TestGetPageCountPypdf
# =========================================================================


class TestGetPageCountPypdf:
    """Tests for the PyPDF2 fallback page counter."""

    def test_returns_page_count_from_pypdf2(self, processor, fake_pdf):
        """Returns len(reader.pages) when PyPDF2 is available."""
        mock_reader = MagicMock()
        mock_reader.pages = [MagicMock(), MagicMock(), MagicMock()]

        mock_pypdf2 = MagicMock()
        mock_pypdf2.PdfReader.return_value = mock_reader

        original_import = (
            __builtins__.__import__
            if hasattr(__builtins__, "__import__")
            else __import__
        )

        def fake_import(name, *args, **kwargs):
            if name == "PyPDF2":
                return mock_pypdf2
            return original_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=fake_import):
            result = processor._get_page_count_pypdf(fake_pdf)

        assert result == 3

    def test_pypdf2_import_error_returns_zero(self, processor, fake_pdf):
        """If PyPDF2 is not installed, returns 0."""
        original_import = (
            __builtins__.__import__
            if hasattr(__builtins__, "__import__")
            else __import__
        )

        def fake_import(name, *args, **kwargs):
            if name == "PyPDF2":
                raise ImportError("No module named 'PyPDF2'")
            return original_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=fake_import):
            result = processor._get_page_count_pypdf(fake_pdf)

        assert result == 0

    def test_pypdf2_runtime_error_returns_zero(self, processor, fake_pdf):
        """If PyPDF2 raises an exception during read, returns 0."""
        mock_pypdf2 = MagicMock()
        mock_pypdf2.PdfReader.side_effect = RuntimeError("corrupt PDF")

        original_import = (
            __builtins__.__import__
            if hasattr(__builtins__, "__import__")
            else __import__
        )

        def fake_import(name, *args, **kwargs):
            if name == "PyPDF2":
                return mock_pypdf2
            return original_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=fake_import):
            result = processor._get_page_count_pypdf(fake_pdf)

        assert result == 0


# =========================================================================
# TestConvertToImages
# =========================================================================


class TestConvertToImages:
    """Tests for PDF-to-image conversion."""

    def test_file_not_found_raises(self, processor):
        """Non-existent PDF raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError, match="PDF not found"):
            processor.convert_to_images("/no/such.pdf", "doc-001")

    def test_creates_output_directory(self, processor, fake_pdf):
        """Output directory for document_id is created."""
        mock_image = MagicMock()
        mock_pdf2image = MagicMock()
        mock_pdf2image.convert_from_path.return_value = [mock_image]

        original_import = (
            __builtins__.__import__
            if hasattr(__builtins__, "__import__")
            else __import__
        )

        def fake_import(name, *args, **kwargs):
            if name == "pdf2image":
                return mock_pdf2image
            return original_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=fake_import):
            processor.convert_to_images(str(fake_pdf), "doc-abc")

        output_dir = processor._images_dir / "doc-abc"
        assert output_dir.exists()

    def test_calls_convert_from_path_with_defaults(self, processor, fake_pdf):
        """convert_from_path is called with expected arguments."""
        mock_pdf2image = MagicMock()
        mock_pdf2image.convert_from_path.return_value = []

        original_import = (
            __builtins__.__import__
            if hasattr(__builtins__, "__import__")
            else __import__
        )

        def fake_import(name, *args, **kwargs):
            if name == "pdf2image":
                return mock_pdf2image
            return original_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=fake_import):
            processor.convert_to_images(str(fake_pdf), "doc-001")

        mock_pdf2image.convert_from_path.assert_called_once_with(
            str(fake_pdf),
            dpi=300,
            first_page=1,
            last_page=None,
            fmt="png",
            thread_count=os.cpu_count() or 4,
        )

    def test_saves_images_with_correct_names(self, processor, fake_pdf):
        """Images are saved as page_NNNN.png with 1-based numbering."""
        mock_img1 = MagicMock()
        mock_img2 = MagicMock()
        mock_pdf2image = MagicMock()
        mock_pdf2image.convert_from_path.return_value = [mock_img1, mock_img2]

        original_import = (
            __builtins__.__import__
            if hasattr(__builtins__, "__import__")
            else __import__
        )

        def fake_import(name, *args, **kwargs):
            if name == "pdf2image":
                return mock_pdf2image
            return original_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=fake_import):
            paths = processor.convert_to_images(str(fake_pdf), "doc-002")

        assert len(paths) == 2
        assert paths[0].endswith("page_0001.png")
        assert paths[1].endswith("page_0002.png")

        # Verify .save() was called on each image
        output_dir = processor._images_dir / "doc-002"
        mock_img1.save.assert_called_once_with(str(output_dir / "page_0001.png"), "PNG")
        mock_img2.save.assert_called_once_with(str(output_dir / "page_0002.png"), "PNG")

    def test_custom_dpi_and_page_range(self, processor, fake_pdf):
        """Custom dpi, start_page, and end_page are forwarded correctly."""
        mock_pdf2image = MagicMock()
        mock_img = MagicMock()
        mock_pdf2image.convert_from_path.return_value = [mock_img]

        original_import = (
            __builtins__.__import__
            if hasattr(__builtins__, "__import__")
            else __import__
        )

        def fake_import(name, *args, **kwargs):
            if name == "pdf2image":
                return mock_pdf2image
            return original_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=fake_import):
            paths = processor.convert_to_images(
                str(fake_pdf), "doc-003", dpi=150, start_page=5, end_page=5
            )

        mock_pdf2image.convert_from_path.assert_called_once_with(
            str(fake_pdf),
            dpi=150,
            first_page=5,
            last_page=5,
            fmt="png",
            thread_count=os.cpu_count() or 4,
        )
        # Page numbering starts from start_page
        assert paths[0].endswith("page_0005.png")

    def test_start_page_offset_naming(self, processor, fake_pdf):
        """When start_page=3 and 2 images returned, files are page_0003 and page_0004."""
        mock_img_a = MagicMock()
        mock_img_b = MagicMock()
        mock_pdf2image = MagicMock()
        mock_pdf2image.convert_from_path.return_value = [mock_img_a, mock_img_b]

        original_import = (
            __builtins__.__import__
            if hasattr(__builtins__, "__import__")
            else __import__
        )

        def fake_import(name, *args, **kwargs):
            if name == "pdf2image":
                return mock_pdf2image
            return original_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=fake_import):
            paths = processor.convert_to_images(
                str(fake_pdf), "doc-offset", start_page=3
            )

        assert len(paths) == 2
        assert "page_0003" in paths[0]
        assert "page_0004" in paths[1]

    def test_import_error_when_pdf2image_missing(self, processor, fake_pdf):
        """ImportError is raised when pdf2image is not installed."""
        original_import = (
            __builtins__.__import__
            if hasattr(__builtins__, "__import__")
            else __import__
        )

        def fake_import(name, *args, **kwargs):
            if name == "pdf2image":
                raise ImportError("No module named 'pdf2image'")
            return original_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=fake_import):
            with pytest.raises(ImportError):
                processor.convert_to_images(str(fake_pdf), "doc-fail")

    def test_other_exception_is_raised(self, processor, fake_pdf):
        """Non-import exceptions from conversion are propagated."""
        mock_pdf2image = MagicMock()
        mock_pdf2image.convert_from_path.side_effect = RuntimeError("poppler crash")

        original_import = (
            __builtins__.__import__
            if hasattr(__builtins__, "__import__")
            else __import__
        )

        def fake_import(name, *args, **kwargs):
            if name == "pdf2image":
                return mock_pdf2image
            return original_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=fake_import):
            with pytest.raises(RuntimeError, match="poppler crash"):
                processor.convert_to_images(str(fake_pdf), "doc-crash")


# =========================================================================
# TestConvertPage
# =========================================================================


class TestConvertPage:
    """Tests for single-page conversion."""

    def test_delegates_to_convert_to_images(self, processor, fake_pdf):
        """convert_page calls convert_to_images with start=end=page_number."""
        with patch.object(
            processor, "convert_to_images", return_value=["/img/page_0005.png"]
        ) as mock_conv:
            result = processor.convert_page(
                str(fake_pdf), "doc-sp", page_number=5, dpi=200
            )

        mock_conv.assert_called_once_with(
            pdf_path=str(fake_pdf),
            document_id="doc-sp",
            dpi=200,
            start_page=5,
            end_page=5,
        )
        assert result == "/img/page_0005.png"

    def test_returns_single_path_string(self, processor, fake_pdf):
        """Return value is a single string, not a list."""
        with patch.object(processor, "convert_to_images", return_value=["/img/p.png"]):
            result = processor.convert_page(str(fake_pdf), "d", 1)

        assert isinstance(result, str)

    def test_empty_results_returns_empty_string(self, processor, fake_pdf):
        """When convert_to_images returns empty list, empty string is returned."""
        with patch.object(processor, "convert_to_images", return_value=[]):
            result = processor.convert_page(str(fake_pdf), "d", 1)

        assert result == ""


# =========================================================================
# TestCleanupImages
# =========================================================================


class TestCleanupImages:
    """Tests for per-document image cleanup."""

    def test_removes_files_and_directory(self, processor):
        """All files in document dir are removed and the dir is deleted."""
        doc_dir = processor._images_dir / "doc-cleanup"
        doc_dir.mkdir(parents=True)
        (doc_dir / "page_0001.png").write_bytes(b"img1")
        (doc_dir / "page_0002.png").write_bytes(b"img2")

        count = processor.cleanup_images("doc-cleanup")

        assert count == 2
        assert not doc_dir.exists()

    def test_returns_count_of_deleted_files(self, processor):
        """Return value is the number of files deleted."""
        doc_dir = processor._images_dir / "doc-count"
        doc_dir.mkdir(parents=True)
        for i in range(5):
            (doc_dir / f"page_{i:04d}.png").write_bytes(b"data")

        assert processor.cleanup_images("doc-count") == 5

    def test_nonexistent_dir_returns_zero(self, processor):
        """Cleanup of a non-existent document_id returns 0."""
        assert processor.cleanup_images("nonexistent-doc") == 0

    def test_handles_file_deletion_errors(self, processor):
        """Files that fail to delete are skipped, count reflects successes."""
        doc_dir = processor._images_dir / "doc-err"
        doc_dir.mkdir(parents=True)
        f1 = doc_dir / "page_0001.png"
        f2 = doc_dir / "page_0002.png"
        f1.write_bytes(b"a")
        f2.write_bytes(b"b")

        original_unlink = Path.unlink

        call_count = 0

        def flaky_unlink(self_path, *args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise PermissionError("cannot delete")
            original_unlink(self_path, *args, **kwargs)

        with patch.object(Path, "unlink", flaky_unlink):
            count = processor.cleanup_images("doc-err")

        # One succeeded, one failed
        assert count == 1

    def test_rmdir_failure_is_silenced(self, processor):
        """If rmdir fails (e.g., dir not empty), no exception is raised."""
        doc_dir = processor._images_dir / "doc-rmdir"
        doc_dir.mkdir(parents=True)
        (doc_dir / "page_0001.png").write_bytes(b"x")
        # Create a subdirectory so rmdir will fail (non-empty after file removal
        # because subdirs aren't removed by iterdir + unlink on files only)
        sub = doc_dir / "subdir"
        sub.mkdir()
        (sub / "nested.txt").write_bytes(b"nested")

        # Should not raise despite rmdir failing
        count = processor.cleanup_images("doc-rmdir")
        # Only the png and subdir entry counted (subdir unlink fails)
        assert count >= 1

    def test_empty_directory_returns_zero_and_removes_dir(self, processor):
        """Empty document dir returns 0 files deleted but directory is removed."""
        doc_dir = processor._images_dir / "doc-empty"
        doc_dir.mkdir(parents=True)

        count = processor.cleanup_images("doc-empty")

        assert count == 0
        assert not doc_dir.exists()


# =========================================================================
# TestCleanupAll
# =========================================================================


class TestCleanupAll:
    """Tests for cleaning all document image directories."""

    def test_cleans_multiple_document_dirs(self, processor):
        """All document directories under images_dir are cleaned."""
        for doc_id in ("doc-a", "doc-b", "doc-c"):
            d = processor._images_dir / doc_id
            d.mkdir(parents=True)
            (d / "page_0001.png").write_bytes(b"img")
            (d / "page_0002.png").write_bytes(b"img")

        total = processor.cleanup_all()

        assert total == 6  # 2 files x 3 dirs

    def test_empty_images_dir_returns_zero(self, processor):
        """No document dirs means 0 files cleaned."""
        assert processor.cleanup_all() == 0

    def test_skips_non_directory_entries(self, processor):
        """Files directly in images_dir are not processed as doc dirs."""
        # Place a stray file in images_dir
        stray = processor._images_dir / "stray_file.txt"
        stray.write_text("stray")

        # One valid doc dir
        d = processor._images_dir / "doc-x"
        d.mkdir()
        (d / "page_0001.png").write_bytes(b"img")

        total = processor.cleanup_all()
        assert total == 1  # Only the file inside doc-x


# =========================================================================
# TestGetImagePaths
# =========================================================================


class TestGetImagePaths:
    """Tests for retrieving existing image paths."""

    def test_returns_sorted_paths(self, processor):
        """Paths are returned in sorted order."""
        doc_dir = processor._images_dir / "doc-sorted"
        doc_dir.mkdir(parents=True)
        # Create out of order
        for name in ("page_0003.png", "page_0001.png", "page_0002.png"):
            (doc_dir / name).write_bytes(b"x")

        paths = processor.get_image_paths("doc-sorted")

        assert len(paths) == 3
        assert "page_0001.png" in paths[0]
        assert "page_0002.png" in paths[1]
        assert "page_0003.png" in paths[2]

    def test_nonexistent_dir_returns_empty_list(self, processor):
        """Non-existent document_id returns empty list."""
        result = processor.get_image_paths("no-such-doc")
        assert result == []

    def test_only_matches_page_pattern(self, processor):
        """Only files matching page_*.png are returned."""
        doc_dir = processor._images_dir / "doc-filter"
        doc_dir.mkdir(parents=True)
        (doc_dir / "page_0001.png").write_bytes(b"ok")
        (doc_dir / "page_0002.png").write_bytes(b"ok")
        (doc_dir / "thumbnail.png").write_bytes(b"no")
        (doc_dir / "page_0003.jpg").write_bytes(b"no")
        (doc_dir / "notes.txt").write_bytes(b"no")

        paths = processor.get_image_paths("doc-filter")

        assert len(paths) == 2
        filenames = [Path(p).name for p in paths]
        assert "page_0001.png" in filenames
        assert "page_0002.png" in filenames
        assert "thumbnail.png" not in filenames

    def test_returns_strings_not_paths(self, processor):
        """Each element in the returned list is a str."""
        doc_dir = processor._images_dir / "doc-str"
        doc_dir.mkdir(parents=True)
        (doc_dir / "page_0001.png").write_bytes(b"x")

        paths = processor.get_image_paths("doc-str")

        assert all(isinstance(p, str) for p in paths)


# =========================================================================
# TestCopyPdfToStorage
# =========================================================================


class TestCopyPdfToStorage:
    """Tests for copying PDFs into the storage directory."""

    def test_copies_file_to_storage(self, processor, fake_pdf):
        """PDF content is copied to documents_dir."""
        dest = processor.copy_pdf_to_storage(str(fake_pdf), "doc-copy")
        assert Path(dest).exists()
        assert Path(dest).read_bytes() == fake_pdf.read_bytes()

    def test_creates_documents_dir_if_missing(
        self, processor, fake_pdf, storage_config
    ):
        """documents_dir is created when it does not exist."""
        docs_dir = Path(storage_config.documents_dir)
        if docs_dir.exists():
            shutil.rmtree(docs_dir)

        processor.copy_pdf_to_storage(str(fake_pdf), "doc-mkdir")
        assert docs_dir.exists()

    def test_returns_destination_path(self, processor, fake_pdf):
        """Returns the full path to the stored PDF."""
        dest = processor.copy_pdf_to_storage(str(fake_pdf), "doc-ret")
        assert isinstance(dest, str)
        assert "doc-ret.pdf" in dest

    def test_source_not_found_raises(self, processor):
        """Non-existent source raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError, match="Source PDF not found"):
            processor.copy_pdf_to_storage("/no/such/source.pdf", "doc-missing")

    def test_preserves_original_extension(self, processor, tmp_path):
        """Destination uses the original file extension."""
        # Test with .pdf
        pdf_file = _create_fake_pdf(tmp_path, "report.pdf")
        dest = processor.copy_pdf_to_storage(str(pdf_file), "doc-ext")
        assert dest.endswith(".pdf")

    def test_different_extension_preserved(self, processor, tmp_path):
        """Non-.pdf extensions are preserved (e.g., .PDF uppercase)."""
        upper_pdf = tmp_path / "report.PDF"
        upper_pdf.write_bytes(b"%PDF-1.4 content")

        dest = processor.copy_pdf_to_storage(str(upper_pdf), "doc-upper")
        assert dest.endswith(".PDF")

    def test_destination_uses_document_id(self, processor, fake_pdf, storage_config):
        """Stored file is named {document_id}.{extension}."""
        dest = processor.copy_pdf_to_storage(str(fake_pdf), "my-custom-id-123")
        expected = Path(storage_config.documents_dir) / "my-custom-id-123.pdf"
        assert Path(dest) == expected

    def test_overwrites_existing_file(self, processor, tmp_path, storage_config):
        """Copying to an existing destination overwrites the file."""
        pdf_v1 = _create_fake_pdf(tmp_path, "v1.pdf", b"version 1")
        pdf_v2 = _create_fake_pdf(tmp_path, "v2.pdf", b"version 2")

        processor.copy_pdf_to_storage(str(pdf_v1), "doc-overwrite")
        processor.copy_pdf_to_storage(str(pdf_v2), "doc-overwrite")

        stored = Path(storage_config.documents_dir) / "doc-overwrite.pdf"
        assert stored.read_bytes() == b"version 2"


# =========================================================================
# Integration-Style Tests
# =========================================================================


class TestEndToEnd:
    """Integration-style tests combining multiple operations."""

    def test_convert_then_get_paths_then_cleanup(self, processor, fake_pdf):
        """Full lifecycle: convert -> list -> cleanup."""
        mock_img = MagicMock()
        mock_pdf2image = MagicMock()
        mock_pdf2image.convert_from_path.return_value = [mock_img, mock_img, mock_img]

        original_import = (
            __builtins__.__import__
            if hasattr(__builtins__, "__import__")
            else __import__
        )

        def fake_import(name, *args, **kwargs):
            if name == "pdf2image":
                return mock_pdf2image
            return original_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=fake_import):
            paths = processor.convert_to_images(str(fake_pdf), "doc-e2e")

        assert len(paths) == 3

        # Create real files since mock .save() didn't actually write
        doc_dir = processor._images_dir / "doc-e2e"
        for i in range(1, 4):
            (doc_dir / f"page_{i:04d}.png").write_bytes(b"fake image")

        listed = processor.get_image_paths("doc-e2e")
        assert len(listed) == 3

        deleted = processor.cleanup_images("doc-e2e")
        assert deleted == 3
        assert processor.get_image_paths("doc-e2e") == []

    def test_copy_then_get_info(self, processor, fake_pdf):
        """Copy PDF to storage, then get_pdf_info on the copy."""
        dest = processor.copy_pdf_to_storage(str(fake_pdf), "doc-info")

        original_import = (
            __builtins__.__import__
            if hasattr(__builtins__, "__import__")
            else __import__
        )

        def fake_import(name, *args, **kwargs):
            if name in ("pdf2image", "PyPDF2"):
                raise ImportError("unavailable")
            return original_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=fake_import):
            page_count, file_hash, file_size = processor.get_pdf_info(dest)

        assert page_count == 0  # Both backends missing
        assert file_size == fake_pdf.stat().st_size
        assert len(file_hash) == 64

    def test_multiple_documents_isolation(self, processor, tmp_path):
        """Operations on different document_ids are isolated."""
        pdf_a = _create_fake_pdf(tmp_path, "a.pdf", b"content A")
        pdf_b = _create_fake_pdf(tmp_path, "b.pdf", b"content B")

        processor.copy_pdf_to_storage(str(pdf_a), "doc-iso-a")
        processor.copy_pdf_to_storage(str(pdf_b), "doc-iso-b")

        # Create image dirs manually
        dir_a = processor._images_dir / "doc-iso-a"
        dir_b = processor._images_dir / "doc-iso-b"
        dir_a.mkdir(parents=True)
        dir_b.mkdir(parents=True)
        (dir_a / "page_0001.png").write_bytes(b"img")
        (dir_b / "page_0001.png").write_bytes(b"img")
        (dir_b / "page_0002.png").write_bytes(b"img")

        assert len(processor.get_image_paths("doc-iso-a")) == 1
        assert len(processor.get_image_paths("doc-iso-b")) == 2

        processor.cleanup_images("doc-iso-a")
        # doc-iso-b should be unaffected
        assert len(processor.get_image_paths("doc-iso-b")) == 2
