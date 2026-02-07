"""
Comprehensive tests for Cloud Storage Sync
=============================================

Tests cover StorageBackend enum, CloudFile dataclass, SyncConfig defaults,
LocalStorageAdapter (with real tmp_path files), S3StorageAdapter (mocked boto3),
CloudSyncManager (sync_once, start/stop, list_pending, get_status),
and the create_sync_manager factory function.

Run with: pytest tests/test_cloud_sync.py -v
"""

import asyncio
import hashlib
import sys
from datetime import datetime
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch, call

import pytest

# Add repo root to path
sys.path.insert(0, str(Path(__file__).parents[1]))

from documents.storage.cloud_sync import (
    StorageBackend,
    CloudFile,
    SyncConfig,
    StorageAdapter,
    LocalStorageAdapter,
    S3StorageAdapter,
    CloudSyncManager,
    create_sync_manager,
)


# =============================================================================
# Helpers
# =============================================================================


def _make_cloud_file(
    key: str = "docs/test.pdf",
    size: int = 1024,
    last_modified: datetime = None,
    etag: str = "abc123",
    content_type: str = "application/pdf",
) -> CloudFile:
    """Convenience factory for CloudFile instances."""
    return CloudFile(
        key=key,
        size=size,
        last_modified=last_modified or datetime(2025, 6, 15, 12, 0, 0),
        etag=etag,
        content_type=content_type,
    )


def _write_file(path: Path, content: bytes = b"hello world") -> Path:
    """Write content to a file, creating parent dirs if needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    return path


# =============================================================================
# 1. StorageBackend enum
# =============================================================================


class TestStorageBackend:
    """Tests for StorageBackend enum values and behaviour."""

    def test_local_value(self):
        assert StorageBackend.LOCAL.value == "local"

    def test_s3_value(self):
        assert StorageBackend.S3.value == "s3"

    def test_icloud_value(self):
        assert StorageBackend.ICLOUD.value == "icloud"

    def test_is_str_subclass(self):
        assert isinstance(StorageBackend.LOCAL, str)

    def test_from_string_local(self):
        assert StorageBackend("local") is StorageBackend.LOCAL

    def test_from_string_s3(self):
        assert StorageBackend("s3") is StorageBackend.S3

    def test_from_string_icloud(self):
        assert StorageBackend("icloud") is StorageBackend.ICLOUD

    def test_invalid_value_raises(self):
        with pytest.raises(ValueError):
            StorageBackend("azure")

    def test_members_count(self):
        assert len(StorageBackend) == 3


# =============================================================================
# 2. CloudFile dataclass
# =============================================================================


class TestCloudFile:
    """Tests for CloudFile creation and properties."""

    def test_basic_creation(self):
        cf = _make_cloud_file()
        assert cf.key == "docs/test.pdf"
        assert cf.size == 1024
        assert cf.etag == "abc123"
        assert cf.content_type == "application/pdf"

    def test_defaults_for_optional_fields(self):
        cf = CloudFile(key="a.txt", size=10, last_modified=datetime.now())
        assert cf.etag is None
        assert cf.content_type is None

    def test_filename_simple(self):
        cf = _make_cloud_file(key="folder/subfolder/readme.md")
        assert cf.filename == "readme.md"

    def test_filename_no_directory(self):
        cf = _make_cloud_file(key="report.pdf")
        assert cf.filename == "report.pdf"

    def test_extension_pdf(self):
        cf = _make_cloud_file(key="docs/report.PDF")
        assert cf.extension == ".pdf"

    def test_extension_lowercase(self):
        cf = _make_cloud_file(key="path/FILE.TxT")
        assert cf.extension == ".txt"

    def test_extension_no_ext(self):
        cf = _make_cloud_file(key="Makefile")
        assert cf.extension == ""

    def test_extension_multiple_dots(self):
        cf = _make_cloud_file(key="archive.tar.gz")
        assert cf.extension == ".gz"

    def test_last_modified_stored(self):
        dt = datetime(2024, 1, 1, 0, 0, 0)
        cf = _make_cloud_file(last_modified=dt)
        assert cf.last_modified == dt

    def test_filename_deep_path(self):
        cf = _make_cloud_file(key="a/b/c/d/e/file.epub")
        assert cf.filename == "file.epub"


# =============================================================================
# 3. SyncConfig defaults and custom values
# =============================================================================


class TestSyncConfig:
    """Tests for SyncConfig default and custom values."""

    def test_default_backend(self):
        cfg = SyncConfig()
        assert cfg.backend == StorageBackend.LOCAL

    def test_default_inbox_path(self):
        assert SyncConfig().inbox_path == ""

    def test_default_s3_bucket(self):
        assert SyncConfig().s3_bucket == ""

    def test_default_s3_prefix(self):
        assert SyncConfig().s3_prefix == "friday-inbox/"

    def test_default_s3_region(self):
        assert SyncConfig().s3_region == "us-east-1"

    def test_default_poll_interval(self):
        assert SyncConfig().poll_interval_seconds == 60

    def test_default_auto_delete(self):
        assert SyncConfig().auto_delete_after_process is False

    def test_default_supported_extensions(self):
        exts = SyncConfig().supported_extensions
        assert ".pdf" in exts
        assert ".epub" in exts
        assert ".txt" in exts
        assert ".md" in exts

    def test_default_auto_ingest(self):
        assert SyncConfig().auto_ingest is True

    def test_custom_backend(self):
        cfg = SyncConfig(backend=StorageBackend.S3)
        assert cfg.backend == StorageBackend.S3

    def test_custom_inbox_path(self):
        cfg = SyncConfig(inbox_path="/tmp/inbox")
        assert cfg.inbox_path == "/tmp/inbox"

    def test_custom_s3_bucket(self):
        cfg = SyncConfig(s3_bucket="my-bucket")
        assert cfg.s3_bucket == "my-bucket"

    def test_custom_poll_interval(self):
        cfg = SyncConfig(poll_interval_seconds=120)
        assert cfg.poll_interval_seconds == 120

    def test_custom_auto_delete(self):
        cfg = SyncConfig(auto_delete_after_process=True)
        assert cfg.auto_delete_after_process is True

    def test_custom_supported_extensions(self):
        cfg = SyncConfig(supported_extensions=[".docx", ".xlsx"])
        assert cfg.supported_extensions == [".docx", ".xlsx"]

    def test_custom_auto_ingest_false(self):
        cfg = SyncConfig(auto_ingest=False)
        assert cfg.auto_ingest is False

    def test_supported_extensions_independent_per_instance(self):
        """Default factory should give each instance its own list."""
        cfg1 = SyncConfig()
        cfg2 = SyncConfig()
        cfg1.supported_extensions.append(".docx")
        assert ".docx" not in cfg2.supported_extensions


# =============================================================================
# 4. LocalStorageAdapter
# =============================================================================


class TestLocalStorageAdapter:
    """Tests for LocalStorageAdapter using real tmp_path files."""

    @pytest.fixture
    def adapter(self, tmp_path):
        """Create a LocalStorageAdapter rooted at tmp_path."""
        return LocalStorageAdapter(str(tmp_path))

    # --- __init__ ---

    def test_init_creates_directory(self, tmp_path):
        new_dir = tmp_path / "new_inbox"
        assert not new_dir.exists()
        LocalStorageAdapter(str(new_dir))
        assert new_dir.exists()

    def test_init_base_path_is_path_object(self, adapter, tmp_path):
        assert adapter.base_path == tmp_path

    # --- list_files ---

    @pytest.mark.asyncio
    async def test_list_files_empty_dir(self, adapter):
        files = await adapter.list_files()
        assert files == []

    @pytest.mark.asyncio
    async def test_list_files_single_file(self, adapter, tmp_path):
        _write_file(tmp_path / "notes.txt", b"data")
        files = await adapter.list_files()
        assert len(files) == 1
        assert files[0].filename == "notes.txt"

    @pytest.mark.asyncio
    async def test_list_files_multiple_files(self, adapter, tmp_path):
        _write_file(tmp_path / "a.pdf", b"pdf content")
        _write_file(tmp_path / "b.epub", b"epub content")
        _write_file(tmp_path / "sub" / "c.txt", b"text content")
        files = await adapter.list_files()
        assert len(files) == 3

    @pytest.mark.asyncio
    async def test_list_files_relative_key(self, adapter, tmp_path):
        _write_file(tmp_path / "sub" / "deep" / "note.md", b"# Title")
        files = await adapter.list_files()
        assert len(files) == 1
        # key should be relative to base_path
        assert files[0].key == str(Path("sub/deep/note.md"))

    @pytest.mark.asyncio
    async def test_list_files_records_size(self, adapter, tmp_path):
        content = b"0123456789"
        _write_file(tmp_path / "sized.bin", content)
        files = await adapter.list_files()
        assert files[0].size == len(content)

    @pytest.mark.asyncio
    async def test_list_files_has_etag(self, adapter, tmp_path):
        _write_file(tmp_path / "f.txt", b"hello")
        files = await adapter.list_files()
        assert files[0].etag is not None
        assert len(files[0].etag) == 32  # MD5 hex digest length

    @pytest.mark.asyncio
    async def test_list_files_has_last_modified(self, adapter, tmp_path):
        _write_file(tmp_path / "f.txt", b"x")
        files = await adapter.list_files()
        assert isinstance(files[0].last_modified, datetime)

    @pytest.mark.asyncio
    async def test_list_files_with_prefix(self, adapter, tmp_path):
        _write_file(tmp_path / "inbox" / "a.pdf", b"pdf")
        _write_file(tmp_path / "other" / "b.txt", b"txt")
        files = await adapter.list_files(prefix="inbox")
        assert len(files) == 1
        assert "a.pdf" in files[0].key

    @pytest.mark.asyncio
    async def test_list_files_nonexistent_prefix(self, adapter):
        files = await adapter.list_files(prefix="does_not_exist")
        assert files == []

    @pytest.mark.asyncio
    async def test_list_files_ignores_directories(self, adapter, tmp_path):
        (tmp_path / "emptydir").mkdir()
        _write_file(tmp_path / "real.txt", b"data")
        files = await adapter.list_files()
        assert len(files) == 1

    # --- download_file ---

    @pytest.mark.asyncio
    async def test_download_file_success(self, adapter, tmp_path):
        content = b"file-content-bytes"
        _write_file(tmp_path / "src.pdf", content)
        dest = tmp_path / "downloads" / "copy.pdf"
        result = await adapter.download_file("src.pdf", dest)
        assert result is True
        assert dest.read_bytes() == content

    @pytest.mark.asyncio
    async def test_download_file_creates_parent_dirs(self, adapter, tmp_path):
        _write_file(tmp_path / "data.txt", b"content")
        dest = tmp_path / "a" / "b" / "c" / "data.txt"
        result = await adapter.download_file("data.txt", dest)
        assert result is True
        assert dest.exists()

    @pytest.mark.asyncio
    async def test_download_file_not_found(self, adapter, tmp_path):
        dest = tmp_path / "output" / "missing.txt"
        result = await adapter.download_file("nonexistent.txt", dest)
        assert result is False
        assert not dest.exists()

    # --- delete_file ---

    @pytest.mark.asyncio
    async def test_delete_file_success(self, adapter, tmp_path):
        _write_file(tmp_path / "to_delete.txt", b"bye")
        result = await adapter.delete_file("to_delete.txt")
        assert result is True
        assert not (tmp_path / "to_delete.txt").exists()

    @pytest.mark.asyncio
    async def test_delete_file_not_found(self, adapter):
        result = await adapter.delete_file("ghost.txt")
        assert result is False

    # --- file_exists ---

    @pytest.mark.asyncio
    async def test_file_exists_true(self, adapter, tmp_path):
        _write_file(tmp_path / "present.txt", b"here")
        assert await adapter.file_exists("present.txt") is True

    @pytest.mark.asyncio
    async def test_file_exists_false(self, adapter):
        assert await adapter.file_exists("absent.txt") is False

    # --- _compute_etag ---

    def test_compute_etag_md5(self, adapter, tmp_path):
        content = b"hello world"
        path = _write_file(tmp_path / "md5test.bin", content)
        etag = adapter._compute_etag(path)
        expected = hashlib.md5(content).hexdigest()
        assert etag == expected

    def test_compute_etag_deterministic(self, adapter, tmp_path):
        content = b"deterministic"
        path = _write_file(tmp_path / "det.bin", content)
        assert adapter._compute_etag(path) == adapter._compute_etag(path)

    def test_compute_etag_differs_for_different_content(self, adapter, tmp_path):
        path1 = _write_file(tmp_path / "a.bin", b"content_a")
        path2 = _write_file(tmp_path / "b.bin", b"content_b")
        assert adapter._compute_etag(path1) != adapter._compute_etag(path2)

    def test_compute_etag_empty_file(self, adapter, tmp_path):
        path = _write_file(tmp_path / "empty.bin", b"")
        etag = adapter._compute_etag(path)
        assert etag == hashlib.md5(b"").hexdigest()

    def test_compute_etag_large_file(self, adapter, tmp_path):
        """Test that large files are hashed correctly with chunked reads."""
        content = b"x" * 20000  # Larger than the 8192 chunk size
        path = _write_file(tmp_path / "large.bin", content)
        etag = adapter._compute_etag(path)
        assert etag == hashlib.md5(content).hexdigest()


# =============================================================================
# 5. S3StorageAdapter (mocked boto3)
# =============================================================================


class TestS3StorageAdapter:
    """Tests for S3StorageAdapter using mocked boto3."""

    def test_init_stores_bucket_and_region(self):
        adapter = S3StorageAdapter(bucket="my-bucket", region="eu-west-1")
        assert adapter.bucket == "my-bucket"
        assert adapter.region == "eu-west-1"

    def test_init_default_region(self):
        adapter = S3StorageAdapter(bucket="b")
        assert adapter.region == "us-east-1"

    def test_client_initially_none(self):
        adapter = S3StorageAdapter(bucket="b")
        assert adapter._client is None

    @pytest.mark.asyncio
    async def test_get_client_lazy_load(self):
        mock_boto3 = MagicMock()
        mock_client = MagicMock()
        mock_boto3.client.return_value = mock_client

        with patch.dict("sys.modules", {"boto3": mock_boto3}):
            adapter = S3StorageAdapter(bucket="test-bucket", region="us-west-2")
            client = await adapter._get_client()
            assert client is mock_client
            mock_boto3.client.assert_called_once_with("s3", region_name="us-west-2")

    @pytest.mark.asyncio
    async def test_get_client_cached(self):
        mock_boto3 = MagicMock()
        mock_client = MagicMock()
        mock_boto3.client.return_value = mock_client

        with patch.dict("sys.modules", {"boto3": mock_boto3}):
            adapter = S3StorageAdapter(bucket="b")
            await adapter._get_client()
            await adapter._get_client()
            # Should only call boto3.client once due to caching
            mock_boto3.client.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_client_import_error(self):
        """When boto3 is not available, should raise ImportError."""
        adapter = S3StorageAdapter(bucket="b")
        with patch.dict("sys.modules", {"boto3": None}):
            with pytest.raises(ImportError, match="boto3 required"):
                await adapter._get_client()

    @pytest.mark.asyncio
    async def test_list_files_single_page(self):
        now = datetime(2025, 6, 1, 12, 0, 0)
        mock_boto3 = MagicMock()
        mock_client = MagicMock()
        mock_paginator = MagicMock()
        mock_paginator.paginate.return_value = [
            {
                "Contents": [
                    {
                        "Key": "prefix/file.pdf",
                        "Size": 4096,
                        "LastModified": now,
                        "ETag": '"abc123"',
                    }
                ]
            }
        ]
        mock_client.get_paginator.return_value = mock_paginator
        mock_boto3.client.return_value = mock_client

        with patch.dict("sys.modules", {"boto3": mock_boto3}):
            adapter = S3StorageAdapter(bucket="test-bucket")
            files = await adapter.list_files(prefix="prefix/")

        assert len(files) == 1
        assert files[0].key == "prefix/file.pdf"
        assert files[0].size == 4096
        assert files[0].last_modified == now
        assert files[0].etag == "abc123"  # Quotes stripped

    @pytest.mark.asyncio
    async def test_list_files_multiple_pages(self):
        now = datetime.now()
        mock_boto3 = MagicMock()
        mock_client = MagicMock()
        mock_paginator = MagicMock()
        mock_paginator.paginate.return_value = [
            {
                "Contents": [
                    {"Key": "a.pdf", "Size": 100, "LastModified": now, "ETag": '"e1"'}
                ]
            },
            {
                "Contents": [
                    {"Key": "b.pdf", "Size": 200, "LastModified": now, "ETag": '"e2"'}
                ]
            },
        ]
        mock_client.get_paginator.return_value = mock_paginator
        mock_boto3.client.return_value = mock_client

        with patch.dict("sys.modules", {"boto3": mock_boto3}):
            adapter = S3StorageAdapter(bucket="b")
            files = await adapter.list_files()

        assert len(files) == 2
        assert files[0].key == "a.pdf"
        assert files[1].key == "b.pdf"

    @pytest.mark.asyncio
    async def test_list_files_empty_page(self):
        mock_boto3 = MagicMock()
        mock_client = MagicMock()
        mock_paginator = MagicMock()
        mock_paginator.paginate.return_value = [{}]  # No Contents key
        mock_client.get_paginator.return_value = mock_paginator
        mock_boto3.client.return_value = mock_client

        with patch.dict("sys.modules", {"boto3": mock_boto3}):
            adapter = S3StorageAdapter(bucket="b")
            files = await adapter.list_files()

        assert files == []

    @pytest.mark.asyncio
    async def test_download_file_success(self, tmp_path):
        mock_boto3 = MagicMock()
        mock_client = MagicMock()
        mock_boto3.client.return_value = mock_client

        with patch.dict("sys.modules", {"boto3": mock_boto3}):
            adapter = S3StorageAdapter(bucket="my-bucket")
            dest = tmp_path / "out" / "file.pdf"
            result = await adapter.download_file("key.pdf", dest)

        assert result is True
        mock_client.download_file.assert_called_once_with(
            "my-bucket", "key.pdf", str(dest)
        )

    @pytest.mark.asyncio
    async def test_download_file_creates_parent_dirs(self, tmp_path):
        mock_boto3 = MagicMock()
        mock_client = MagicMock()
        mock_boto3.client.return_value = mock_client

        with patch.dict("sys.modules", {"boto3": mock_boto3}):
            adapter = S3StorageAdapter(bucket="b")
            dest = tmp_path / "nested" / "deep" / "file.pdf"
            await adapter.download_file("key.pdf", dest)
            assert dest.parent.exists()

    @pytest.mark.asyncio
    async def test_download_file_failure(self, tmp_path):
        mock_boto3 = MagicMock()
        mock_client = MagicMock()
        mock_client.download_file.side_effect = Exception("network error")
        mock_boto3.client.return_value = mock_client

        with patch.dict("sys.modules", {"boto3": mock_boto3}):
            adapter = S3StorageAdapter(bucket="b")
            dest = tmp_path / "fail.pdf"
            result = await adapter.download_file("key.pdf", dest)

        assert result is False

    @pytest.mark.asyncio
    async def test_delete_file_success(self):
        mock_boto3 = MagicMock()
        mock_client = MagicMock()
        mock_boto3.client.return_value = mock_client

        with patch.dict("sys.modules", {"boto3": mock_boto3}):
            adapter = S3StorageAdapter(bucket="my-bucket")
            result = await adapter.delete_file("docs/old.pdf")

        assert result is True
        mock_client.delete_object.assert_called_once_with(
            Bucket="my-bucket", Key="docs/old.pdf"
        )

    @pytest.mark.asyncio
    async def test_delete_file_failure(self):
        mock_boto3 = MagicMock()
        mock_client = MagicMock()
        mock_client.delete_object.side_effect = Exception("access denied")
        mock_boto3.client.return_value = mock_client

        with patch.dict("sys.modules", {"boto3": mock_boto3}):
            adapter = S3StorageAdapter(bucket="b")
            result = await adapter.delete_file("key.pdf")

        assert result is False

    @pytest.mark.asyncio
    async def test_file_exists_true(self):
        mock_boto3 = MagicMock()
        mock_client = MagicMock()
        mock_boto3.client.return_value = mock_client

        with patch.dict("sys.modules", {"boto3": mock_boto3}):
            adapter = S3StorageAdapter(bucket="my-bucket")
            result = await adapter.file_exists("present.pdf")

        assert result is True
        mock_client.head_object.assert_called_once_with(
            Bucket="my-bucket", Key="present.pdf"
        )

    @pytest.mark.asyncio
    async def test_file_exists_false(self):
        mock_boto3 = MagicMock()
        mock_client = MagicMock()
        mock_client.head_object.side_effect = Exception("404")
        mock_boto3.client.return_value = mock_client

        with patch.dict("sys.modules", {"boto3": mock_boto3}):
            adapter = S3StorageAdapter(bucket="b")
            result = await adapter.file_exists("missing.pdf")

        assert result is False

    @pytest.mark.asyncio
    async def test_list_files_etag_without_quotes(self):
        """ETag that does not have surrounding quotes should still work."""
        now = datetime.now()
        mock_boto3 = MagicMock()
        mock_client = MagicMock()
        mock_paginator = MagicMock()
        mock_paginator.paginate.return_value = [
            {
                "Contents": [
                    {
                        "Key": "f.txt",
                        "Size": 10,
                        "LastModified": now,
                        "ETag": "noquotes",
                    }
                ]
            }
        ]
        mock_client.get_paginator.return_value = mock_paginator
        mock_boto3.client.return_value = mock_client

        with patch.dict("sys.modules", {"boto3": mock_boto3}):
            adapter = S3StorageAdapter(bucket="b")
            files = await adapter.list_files()

        assert files[0].etag == "noquotes"

    @pytest.mark.asyncio
    async def test_list_files_missing_etag(self):
        """Contents without ETag key should default to empty."""
        now = datetime.now()
        mock_boto3 = MagicMock()
        mock_client = MagicMock()
        mock_paginator = MagicMock()
        mock_paginator.paginate.return_value = [
            {"Contents": [{"Key": "f.txt", "Size": 10, "LastModified": now}]}
        ]
        mock_client.get_paginator.return_value = mock_paginator
        mock_boto3.client.return_value = mock_client

        with patch.dict("sys.modules", {"boto3": mock_boto3}):
            adapter = S3StorageAdapter(bucket="b")
            files = await adapter.list_files()

        assert files[0].etag == ""


# =============================================================================
# 6. CloudSyncManager
# =============================================================================


class TestCloudSyncManagerInit:
    """Tests for CloudSyncManager initialization."""

    def test_init_local_backend(self, tmp_path):
        config = SyncConfig(
            backend=StorageBackend.LOCAL,
            inbox_path=str(tmp_path / "inbox"),
        )
        mgr = CloudSyncManager(config, local_download_dir=str(tmp_path / "dl"))
        assert isinstance(mgr._adapter, LocalStorageAdapter)
        assert mgr._running is False
        assert mgr._task is None
        assert mgr._processed_etags == {}

    def test_init_s3_backend(self, tmp_path):
        config = SyncConfig(
            backend=StorageBackend.S3,
            s3_bucket="test-bucket",
            s3_region="us-west-2",
        )
        mgr = CloudSyncManager(config, local_download_dir=str(tmp_path / "dl"))
        assert isinstance(mgr._adapter, S3StorageAdapter)
        assert mgr._adapter.bucket == "test-bucket"
        assert mgr._adapter.region == "us-west-2"

    def test_init_icloud_backend(self, tmp_path):
        config = SyncConfig(
            backend=StorageBackend.ICLOUD,
            inbox_path=str(tmp_path / "icloud_inbox"),
        )
        mgr = CloudSyncManager(config, local_download_dir=str(tmp_path / "dl"))
        assert isinstance(mgr._adapter, LocalStorageAdapter)

    def test_init_creates_download_dir(self, tmp_path):
        dl_dir = tmp_path / "new_dl_dir"
        config = SyncConfig(
            backend=StorageBackend.LOCAL,
            inbox_path=str(tmp_path / "inbox"),
        )
        CloudSyncManager(config, local_download_dir=str(dl_dir))
        assert dl_dir.exists()

    def test_init_stores_document_manager(self, tmp_path):
        mock_dm = MagicMock()
        config = SyncConfig(
            backend=StorageBackend.LOCAL,
            inbox_path=str(tmp_path / "inbox"),
        )
        mgr = CloudSyncManager(
            config, document_manager=mock_dm, local_download_dir=str(tmp_path / "dl")
        )
        assert mgr.document_manager is mock_dm


class TestCloudSyncManagerCreateAdapter:
    """Tests for _create_adapter selecting the correct backend."""

    def test_create_adapter_local(self, tmp_path):
        config = SyncConfig(
            backend=StorageBackend.LOCAL,
            inbox_path=str(tmp_path / "local_inbox"),
        )
        mgr = CloudSyncManager(config, local_download_dir=str(tmp_path / "dl"))
        assert isinstance(mgr._adapter, LocalStorageAdapter)
        assert mgr._adapter.base_path == tmp_path / "local_inbox"

    def test_create_adapter_local_defaults_to_download_dir(self, tmp_path):
        config = SyncConfig(backend=StorageBackend.LOCAL, inbox_path="")
        dl_dir = tmp_path / "dl"
        mgr = CloudSyncManager(config, local_download_dir=str(dl_dir))
        assert isinstance(mgr._adapter, LocalStorageAdapter)
        assert mgr._adapter.base_path == dl_dir

    def test_create_adapter_s3(self, tmp_path):
        config = SyncConfig(
            backend=StorageBackend.S3,
            s3_bucket="my-bucket",
            s3_region="eu-west-1",
        )
        mgr = CloudSyncManager(config, local_download_dir=str(tmp_path / "dl"))
        assert isinstance(mgr._adapter, S3StorageAdapter)
        assert mgr._adapter.bucket == "my-bucket"
        assert mgr._adapter.region == "eu-west-1"

    def test_create_adapter_icloud_custom_path(self, tmp_path):
        custom_path = str(tmp_path / "custom_icloud")
        config = SyncConfig(
            backend=StorageBackend.ICLOUD,
            inbox_path=custom_path,
        )
        mgr = CloudSyncManager(config, local_download_dir=str(tmp_path / "dl"))
        assert isinstance(mgr._adapter, LocalStorageAdapter)
        assert mgr._adapter.base_path == Path(custom_path)

    def test_create_adapter_icloud_default_path(self, tmp_path):
        """When inbox_path is empty, iCloud backend should use ~/Library/... path."""
        config = SyncConfig(backend=StorageBackend.ICLOUD, inbox_path="")
        mgr = CloudSyncManager(config, local_download_dir=str(tmp_path / "dl"))
        assert isinstance(mgr._adapter, LocalStorageAdapter)
        assert "Friday-Inbox" in str(mgr._adapter.base_path)


class TestCloudSyncManagerSyncOnce:
    """Tests for sync_once method."""

    @pytest.fixture
    def setup(self, tmp_path):
        """Create a manager with mocked adapter."""
        config = SyncConfig(
            backend=StorageBackend.LOCAL,
            inbox_path=str(tmp_path / "inbox"),
            supported_extensions=[".pdf", ".txt", ".md", ".epub"],
            auto_ingest=False,
            auto_delete_after_process=False,
        )
        mgr = CloudSyncManager(config, local_download_dir=str(tmp_path / "dl"))
        mock_adapter = AsyncMock()
        mgr._adapter = mock_adapter
        return mgr, mock_adapter

    @pytest.mark.asyncio
    async def test_sync_once_new_files(self, setup):
        mgr, mock_adapter = setup
        cf = _make_cloud_file(key="report.pdf", etag="e1")
        mock_adapter.list_files.return_value = [cf]
        mock_adapter.download_file.return_value = True

        result = await mgr.sync_once()
        assert len(result) == 1
        assert result[0].key == "report.pdf"
        mock_adapter.download_file.assert_called_once()

    @pytest.mark.asyncio
    async def test_sync_once_skip_already_processed(self, setup):
        mgr, mock_adapter = setup
        cf = _make_cloud_file(key="report.pdf", etag="e1")
        mgr._processed_etags["report.pdf"] = "e1"
        mock_adapter.list_files.return_value = [cf]

        result = await mgr.sync_once()
        assert len(result) == 0
        mock_adapter.download_file.assert_not_called()

    @pytest.mark.asyncio
    async def test_sync_once_reprocess_changed_etag(self, setup):
        mgr, mock_adapter = setup
        cf = _make_cloud_file(key="report.pdf", etag="e2_new")
        mgr._processed_etags["report.pdf"] = "e1_old"
        mock_adapter.list_files.return_value = [cf]
        mock_adapter.download_file.return_value = True

        result = await mgr.sync_once()
        assert len(result) == 1
        assert mgr._processed_etags["report.pdf"] == "e2_new"

    @pytest.mark.asyncio
    async def test_sync_once_filters_unsupported_extension(self, setup):
        mgr, mock_adapter = setup
        cf = _make_cloud_file(key="photo.jpg", etag="e1")
        mock_adapter.list_files.return_value = [cf]

        result = await mgr.sync_once()
        assert len(result) == 0
        mock_adapter.download_file.assert_not_called()

    @pytest.mark.asyncio
    async def test_sync_once_download_failure_skips_file(self, setup):
        mgr, mock_adapter = setup
        cf = _make_cloud_file(key="report.pdf", etag="e1")
        mock_adapter.list_files.return_value = [cf]
        mock_adapter.download_file.return_value = False

        result = await mgr.sync_once()
        assert len(result) == 0
        assert "report.pdf" not in mgr._processed_etags

    @pytest.mark.asyncio
    async def test_sync_once_auto_delete(self, tmp_path):
        config = SyncConfig(
            backend=StorageBackend.LOCAL,
            inbox_path=str(tmp_path / "inbox"),
            auto_delete_after_process=True,
            auto_ingest=False,
        )
        mgr = CloudSyncManager(config, local_download_dir=str(tmp_path / "dl"))
        mock_adapter = AsyncMock()
        mgr._adapter = mock_adapter

        cf = _make_cloud_file(key="report.pdf", etag="e1")
        mock_adapter.list_files.return_value = [cf]
        mock_adapter.download_file.return_value = True
        mock_adapter.delete_file.return_value = True

        await mgr.sync_once()
        mock_adapter.delete_file.assert_called_once_with("report.pdf")

    @pytest.mark.asyncio
    async def test_sync_once_no_delete_when_disabled(self, setup):
        mgr, mock_adapter = setup
        cf = _make_cloud_file(key="report.pdf", etag="e1")
        mock_adapter.list_files.return_value = [cf]
        mock_adapter.download_file.return_value = True

        await mgr.sync_once()
        mock_adapter.delete_file.assert_not_called()

    @pytest.mark.asyncio
    async def test_sync_once_auto_ingest(self, tmp_path):
        mock_dm = AsyncMock()
        config = SyncConfig(
            backend=StorageBackend.LOCAL,
            inbox_path=str(tmp_path / "inbox"),
            auto_ingest=True,
        )
        mgr = CloudSyncManager(
            config,
            document_manager=mock_dm,
            local_download_dir=str(tmp_path / "dl"),
        )
        mock_adapter = AsyncMock()
        mgr._adapter = mock_adapter

        cf = _make_cloud_file(key="report.pdf", etag="e1")
        mock_adapter.list_files.return_value = [cf]
        mock_adapter.download_file.return_value = True

        with patch(
            "documents.storage.cloud_sync.CloudSyncManager._ingest_file",
            new_callable=AsyncMock,
        ) as mock_ingest:
            await mgr.sync_once()
            mock_ingest.assert_called_once()

    @pytest.mark.asyncio
    async def test_sync_once_auto_ingest_not_called_without_doc_manager(self, setup):
        mgr, mock_adapter = setup
        mgr.config.auto_ingest = True
        mgr.document_manager = None

        cf = _make_cloud_file(key="report.pdf", etag="e1")
        mock_adapter.list_files.return_value = [cf]
        mock_adapter.download_file.return_value = True

        # Should succeed without calling _ingest_file
        result = await mgr.sync_once()
        assert len(result) == 1

    @pytest.mark.asyncio
    async def test_sync_once_ingest_failure_does_not_stop_sync(self, tmp_path):
        mock_dm = AsyncMock()
        config = SyncConfig(
            backend=StorageBackend.LOCAL,
            inbox_path=str(tmp_path / "inbox"),
            auto_ingest=True,
        )
        mgr = CloudSyncManager(
            config,
            document_manager=mock_dm,
            local_download_dir=str(tmp_path / "dl"),
        )
        mock_adapter = AsyncMock()
        mgr._adapter = mock_adapter

        cf = _make_cloud_file(key="report.pdf", etag="e1")
        mock_adapter.list_files.return_value = [cf]
        mock_adapter.download_file.return_value = True

        with patch(
            "documents.storage.cloud_sync.CloudSyncManager._ingest_file",
            new_callable=AsyncMock,
            side_effect=Exception("ingest boom"),
        ):
            result = await mgr.sync_once()
            # File should still be marked as processed
            assert len(result) == 1
            assert "report.pdf" in mgr._processed_etags

    @pytest.mark.asyncio
    async def test_sync_once_s3_uses_prefix(self, tmp_path):
        config = SyncConfig(
            backend=StorageBackend.S3,
            s3_bucket="bucket",
            s3_prefix="my-prefix/",
            auto_ingest=False,
        )
        mgr = CloudSyncManager(config, local_download_dir=str(tmp_path / "dl"))
        mock_adapter = AsyncMock()
        mgr._adapter = mock_adapter
        mock_adapter.list_files.return_value = []

        await mgr.sync_once()
        mock_adapter.list_files.assert_called_once_with("my-prefix/")

    @pytest.mark.asyncio
    async def test_sync_once_local_uses_empty_prefix(self, setup):
        mgr, mock_adapter = setup
        mock_adapter.list_files.return_value = []
        await mgr.sync_once()
        mock_adapter.list_files.assert_called_once_with("")

    @pytest.mark.asyncio
    async def test_sync_once_multiple_files(self, setup):
        mgr, mock_adapter = setup
        files = [
            _make_cloud_file(key="a.pdf", etag="e1"),
            _make_cloud_file(key="b.txt", etag="e2"),
            _make_cloud_file(key="c.md", etag="e3"),
        ]
        mock_adapter.list_files.return_value = files
        mock_adapter.download_file.return_value = True

        result = await mgr.sync_once()
        assert len(result) == 3
        assert mock_adapter.download_file.call_count == 3

    @pytest.mark.asyncio
    async def test_sync_once_mixed_supported_unsupported(self, setup):
        mgr, mock_adapter = setup
        files = [
            _make_cloud_file(key="a.pdf", etag="e1"),
            _make_cloud_file(key="b.jpg", etag="e2"),
            _make_cloud_file(key="c.txt", etag="e3"),
            _make_cloud_file(key="d.exe", etag="e4"),
        ]
        mock_adapter.list_files.return_value = files
        mock_adapter.download_file.return_value = True

        result = await mgr.sync_once()
        assert len(result) == 2  # Only .pdf and .txt
        downloaded_keys = [c.args[0] for c in mock_adapter.download_file.call_args_list]
        assert "a.pdf" in downloaded_keys
        assert "c.txt" in downloaded_keys

    @pytest.mark.asyncio
    async def test_sync_once_updates_processed_etags(self, setup):
        mgr, mock_adapter = setup
        cf = _make_cloud_file(key="report.pdf", etag="etag_val")
        mock_adapter.list_files.return_value = [cf]
        mock_adapter.download_file.return_value = True

        await mgr.sync_once()
        assert mgr._processed_etags["report.pdf"] == "etag_val"


class TestCloudSyncManagerListPending:
    """Tests for list_pending method."""

    @pytest.fixture
    def setup(self, tmp_path):
        config = SyncConfig(
            backend=StorageBackend.LOCAL,
            inbox_path=str(tmp_path / "inbox"),
            supported_extensions=[".pdf", ".txt"],
        )
        mgr = CloudSyncManager(config, local_download_dir=str(tmp_path / "dl"))
        mock_adapter = AsyncMock()
        mgr._adapter = mock_adapter
        return mgr, mock_adapter

    @pytest.mark.asyncio
    async def test_list_pending_all_new(self, setup):
        mgr, mock_adapter = setup
        files = [
            _make_cloud_file(key="a.pdf", etag="e1"),
            _make_cloud_file(key="b.txt", etag="e2"),
        ]
        mock_adapter.list_files.return_value = files

        pending = await mgr.list_pending()
        assert len(pending) == 2

    @pytest.mark.asyncio
    async def test_list_pending_some_processed(self, setup):
        mgr, mock_adapter = setup
        mgr._processed_etags["a.pdf"] = "e1"
        files = [
            _make_cloud_file(key="a.pdf", etag="e1"),
            _make_cloud_file(key="b.txt", etag="e2"),
        ]
        mock_adapter.list_files.return_value = files

        pending = await mgr.list_pending()
        assert len(pending) == 1
        assert pending[0].key == "b.txt"

    @pytest.mark.asyncio
    async def test_list_pending_filters_extensions(self, setup):
        mgr, mock_adapter = setup
        files = [
            _make_cloud_file(key="a.pdf", etag="e1"),
            _make_cloud_file(key="b.jpg", etag="e2"),
        ]
        mock_adapter.list_files.return_value = files

        pending = await mgr.list_pending()
        assert len(pending) == 1
        assert pending[0].key == "a.pdf"

    @pytest.mark.asyncio
    async def test_list_pending_empty(self, setup):
        mgr, mock_adapter = setup
        mock_adapter.list_files.return_value = []

        pending = await mgr.list_pending()
        assert pending == []

    @pytest.mark.asyncio
    async def test_list_pending_s3_uses_prefix(self, tmp_path):
        config = SyncConfig(
            backend=StorageBackend.S3,
            s3_bucket="bucket",
            s3_prefix="inbox/",
        )
        mgr = CloudSyncManager(config, local_download_dir=str(tmp_path / "dl"))
        mock_adapter = AsyncMock()
        mgr._adapter = mock_adapter
        mock_adapter.list_files.return_value = []

        await mgr.list_pending()
        mock_adapter.list_files.assert_called_once_with("inbox/")


class TestCloudSyncManagerGetStatus:
    """Tests for get_status method."""

    def test_get_status_initial(self, tmp_path):
        config = SyncConfig(
            backend=StorageBackend.LOCAL,
            inbox_path=str(tmp_path / "inbox"),
            poll_interval_seconds=30,
            auto_ingest=True,
        )
        mgr = CloudSyncManager(config, local_download_dir=str(tmp_path / "dl"))
        status = mgr.get_status()

        assert status["running"] is False
        assert status["backend"] == "local"
        assert status["processed_count"] == 0
        assert status["poll_interval"] == 30
        assert status["auto_ingest"] is True

    def test_get_status_after_processing(self, tmp_path):
        config = SyncConfig(
            backend=StorageBackend.S3,
            s3_bucket="b",
        )
        mgr = CloudSyncManager(config, local_download_dir=str(tmp_path / "dl"))
        mgr._processed_etags = {"a.pdf": "e1", "b.txt": "e2"}

        status = mgr.get_status()
        assert status["processed_count"] == 2
        assert status["backend"] == "s3"

    def test_get_status_running_true(self, tmp_path):
        config = SyncConfig(
            backend=StorageBackend.LOCAL,
            inbox_path=str(tmp_path / "inbox"),
        )
        mgr = CloudSyncManager(config, local_download_dir=str(tmp_path / "dl"))
        mgr._running = True

        status = mgr.get_status()
        assert status["running"] is True

    def test_get_status_auto_ingest_false(self, tmp_path):
        config = SyncConfig(
            backend=StorageBackend.LOCAL,
            inbox_path=str(tmp_path / "inbox"),
            auto_ingest=False,
        )
        mgr = CloudSyncManager(config, local_download_dir=str(tmp_path / "dl"))
        status = mgr.get_status()
        assert status["auto_ingest"] is False


class TestCloudSyncManagerStartStop:
    """Tests for start and stop methods."""

    @pytest.mark.asyncio
    async def test_start_sets_running(self, tmp_path):
        config = SyncConfig(
            backend=StorageBackend.LOCAL,
            inbox_path=str(tmp_path / "inbox"),
            poll_interval_seconds=1,
        )
        mgr = CloudSyncManager(config, local_download_dir=str(tmp_path / "dl"))
        mock_adapter = AsyncMock()
        mock_adapter.list_files.return_value = []
        mgr._adapter = mock_adapter

        await mgr.start()
        assert mgr._running is True
        assert mgr._task is not None
        await mgr.stop()

    @pytest.mark.asyncio
    async def test_stop_clears_running(self, tmp_path):
        config = SyncConfig(
            backend=StorageBackend.LOCAL,
            inbox_path=str(tmp_path / "inbox"),
            poll_interval_seconds=1,
        )
        mgr = CloudSyncManager(config, local_download_dir=str(tmp_path / "dl"))
        mock_adapter = AsyncMock()
        mock_adapter.list_files.return_value = []
        mgr._adapter = mock_adapter

        await mgr.start()
        await mgr.stop()
        assert mgr._running is False

    @pytest.mark.asyncio
    async def test_start_is_idempotent(self, tmp_path):
        config = SyncConfig(
            backend=StorageBackend.LOCAL,
            inbox_path=str(tmp_path / "inbox"),
            poll_interval_seconds=1,
        )
        mgr = CloudSyncManager(config, local_download_dir=str(tmp_path / "dl"))
        mock_adapter = AsyncMock()
        mock_adapter.list_files.return_value = []
        mgr._adapter = mock_adapter

        await mgr.start()
        task1 = mgr._task
        await mgr.start()  # Should not create a new task
        assert mgr._task is task1
        await mgr.stop()

    @pytest.mark.asyncio
    async def test_stop_when_not_started(self, tmp_path):
        config = SyncConfig(
            backend=StorageBackend.LOCAL,
            inbox_path=str(tmp_path / "inbox"),
        )
        mgr = CloudSyncManager(config, local_download_dir=str(tmp_path / "dl"))
        # Should not raise
        await mgr.stop()
        assert mgr._running is False


class TestCloudSyncManagerIngestFile:
    """Tests for _ingest_file method."""

    @pytest.mark.asyncio
    async def test_ingest_file_no_document_manager(self, tmp_path):
        config = SyncConfig(
            backend=StorageBackend.LOCAL,
            inbox_path=str(tmp_path / "inbox"),
        )
        mgr = CloudSyncManager(config, local_download_dir=str(tmp_path / "dl"))
        mgr.document_manager = None

        cf = _make_cloud_file(key="test.pdf")
        local_path = tmp_path / "test.pdf"
        # Should return without error
        await mgr._ingest_file(local_path, cf)

    @pytest.mark.asyncio
    async def test_ingest_file_pdf_type(self, tmp_path):
        mock_dm = AsyncMock()
        config = SyncConfig(
            backend=StorageBackend.LOCAL,
            inbox_path=str(tmp_path / "inbox"),
        )
        mgr = CloudSyncManager(
            config,
            document_manager=mock_dm,
            local_download_dir=str(tmp_path / "dl"),
        )

        cf = _make_cloud_file(key="book.pdf")
        local_path = tmp_path / "book.pdf"

        # _ingest_file does `from documents.models import DocumentType` locally,
        # so we patch at the source module level.
        MockDocType = MagicMock()
        MockDocType.BOOK = "BOOK"
        MockDocType.ARTICLE = "ARTICLE"
        MockDocType.REFERENCE = "REFERENCE"
        mock_models = MagicMock(DocumentType=MockDocType)
        with patch.dict("sys.modules", {"documents.models": mock_models}):
            await mgr._ingest_file(local_path, cf)
            mock_dm.ingest_document.assert_called_once()
            call_kwargs = mock_dm.ingest_document.call_args
            assert call_kwargs[1]["title"] == "book"
            assert call_kwargs[1]["document_type"] == "BOOK"

    @pytest.mark.asyncio
    async def test_ingest_file_txt_type(self, tmp_path):
        mock_dm = AsyncMock()
        config = SyncConfig(
            backend=StorageBackend.LOCAL,
            inbox_path=str(tmp_path / "inbox"),
        )
        mgr = CloudSyncManager(
            config,
            document_manager=mock_dm,
            local_download_dir=str(tmp_path / "dl"),
        )

        cf = _make_cloud_file(key="notes.txt")
        local_path = tmp_path / "notes.txt"

        MockDocType = MagicMock()
        MockDocType.BOOK = "BOOK"
        MockDocType.ARTICLE = "ARTICLE"
        MockDocType.REFERENCE = "REFERENCE"
        mock_models = MagicMock(DocumentType=MockDocType)
        with patch.dict("sys.modules", {"documents.models": mock_models}):
            await mgr._ingest_file(local_path, cf)
            call_kwargs = mock_dm.ingest_document.call_args
            assert call_kwargs[1]["document_type"] == "ARTICLE"

    @pytest.mark.asyncio
    async def test_ingest_file_epub_type(self, tmp_path):
        """Epub files should be ingested as REFERENCE type."""
        mock_dm = AsyncMock()
        config = SyncConfig(
            backend=StorageBackend.LOCAL,
            inbox_path=str(tmp_path / "inbox"),
        )
        mgr = CloudSyncManager(
            config,
            document_manager=mock_dm,
            local_download_dir=str(tmp_path / "dl"),
        )

        cf = _make_cloud_file(key="guide.epub")
        local_path = tmp_path / "guide.epub"

        MockDocType = MagicMock()
        MockDocType.BOOK = "BOOK"
        MockDocType.ARTICLE = "ARTICLE"
        MockDocType.REFERENCE = "REFERENCE"
        mock_models = MagicMock(DocumentType=MockDocType)
        with patch.dict("sys.modules", {"documents.models": mock_models}):
            await mgr._ingest_file(local_path, cf)
            call_kwargs = mock_dm.ingest_document.call_args
            assert call_kwargs[1]["document_type"] == "REFERENCE"

    @pytest.mark.asyncio
    async def test_ingest_file_md_type(self, tmp_path):
        """Markdown files should be ingested as ARTICLE type."""
        mock_dm = AsyncMock()
        config = SyncConfig(
            backend=StorageBackend.LOCAL,
            inbox_path=str(tmp_path / "inbox"),
        )
        mgr = CloudSyncManager(
            config,
            document_manager=mock_dm,
            local_download_dir=str(tmp_path / "dl"),
        )

        cf = _make_cloud_file(key="readme.md")
        local_path = tmp_path / "readme.md"

        MockDocType = MagicMock()
        MockDocType.BOOK = "BOOK"
        MockDocType.ARTICLE = "ARTICLE"
        MockDocType.REFERENCE = "REFERENCE"
        mock_models = MagicMock(DocumentType=MockDocType)
        with patch.dict("sys.modules", {"documents.models": mock_models}):
            await mgr._ingest_file(local_path, cf)
            call_kwargs = mock_dm.ingest_document.call_args
            assert call_kwargs[1]["document_type"] == "ARTICLE"


# =============================================================================
# 7. create_sync_manager factory
# =============================================================================


class TestCreateSyncManager:
    """Tests for the create_sync_manager factory function."""

    def test_default_creates_local_manager(self, tmp_path):
        with patch(
            "documents.storage.cloud_sync.CloudSyncManager.__init__",
            return_value=None,
        ) as mock_init:
            mgr = create_sync_manager()
            # Verify it was called with a SyncConfig using LOCAL backend
            args = mock_init.call_args
            config = args[0][0]
            assert config.backend == StorageBackend.LOCAL

    def test_factory_local_backend(self, tmp_path):
        mgr = create_sync_manager(backend="local", inbox_path=str(tmp_path / "inbox"))
        assert isinstance(mgr, CloudSyncManager)
        assert mgr.config.backend == StorageBackend.LOCAL
        assert mgr.config.inbox_path == str(tmp_path / "inbox")

    def test_factory_s3_backend(self, tmp_path):
        mgr = create_sync_manager(
            backend="s3",
            s3_bucket="my-bucket",
            s3_prefix="custom-prefix/",
        )
        assert isinstance(mgr, CloudSyncManager)
        assert mgr.config.backend == StorageBackend.S3
        assert mgr.config.s3_bucket == "my-bucket"
        assert mgr.config.s3_prefix == "custom-prefix/"

    def test_factory_icloud_backend(self, tmp_path):
        mgr = create_sync_manager(
            backend="icloud",
            inbox_path=str(tmp_path / "icloud"),
        )
        assert isinstance(mgr, CloudSyncManager)
        assert mgr.config.backend == StorageBackend.ICLOUD

    def test_factory_invalid_backend(self):
        with pytest.raises(ValueError):
            create_sync_manager(backend="gcs")

    def test_factory_passes_document_manager(self, tmp_path):
        mock_dm = MagicMock()
        mgr = create_sync_manager(
            backend="local",
            inbox_path=str(tmp_path / "inbox"),
            document_manager=mock_dm,
        )
        assert mgr.document_manager is mock_dm

    def test_factory_default_prefix(self):
        mgr = create_sync_manager(backend="s3", s3_bucket="b")
        assert mgr.config.s3_prefix == "friday-inbox/"

    def test_factory_returns_cloud_sync_manager_type(self, tmp_path):
        mgr = create_sync_manager(
            backend="local",
            inbox_path=str(tmp_path / "inbox"),
        )
        assert type(mgr).__name__ == "CloudSyncManager"


# =============================================================================
# 8. StorageAdapter ABC enforcement
# =============================================================================


class TestStorageAdapterABC:
    """Verify StorageAdapter cannot be instantiated directly."""

    def test_cannot_instantiate_abc(self):
        with pytest.raises(TypeError):
            StorageAdapter()

    def test_subclass_must_implement_all_methods(self):
        class IncompleteAdapter(StorageAdapter):
            pass

        with pytest.raises(TypeError):
            IncompleteAdapter()


# =============================================================================
# 9. Edge cases and integration-like scenarios
# =============================================================================


class TestEdgeCases:
    """Additional edge cases for thorough coverage."""

    @pytest.mark.asyncio
    async def test_local_adapter_list_files_nested_deeply(self, tmp_path):
        adapter = LocalStorageAdapter(str(tmp_path))
        _write_file(tmp_path / "a" / "b" / "c" / "d" / "deep.txt", b"deep")
        files = await adapter.list_files()
        assert len(files) == 1
        assert "deep.txt" in files[0].key

    @pytest.mark.asyncio
    async def test_sync_once_download_path_uses_filename(self, tmp_path):
        """Verify the download local_path is based on cloud_file.filename."""
        config = SyncConfig(
            backend=StorageBackend.LOCAL,
            inbox_path=str(tmp_path / "inbox"),
            auto_ingest=False,
        )
        dl_dir = tmp_path / "dl"
        mgr = CloudSyncManager(config, local_download_dir=str(dl_dir))
        mock_adapter = AsyncMock()
        mgr._adapter = mock_adapter

        cf = _make_cloud_file(key="sub/folder/report.pdf", etag="e1")
        mock_adapter.list_files.return_value = [cf]
        mock_adapter.download_file.return_value = True

        await mgr.sync_once()
        # download_file should be called with (key, local_download_dir / filename)
        call_args = mock_adapter.download_file.call_args
        assert call_args[0][0] == "sub/folder/report.pdf"
        assert call_args[0][1] == dl_dir / "report.pdf"

    @pytest.mark.asyncio
    async def test_sync_once_processes_all_before_returning(self, tmp_path):
        """All files should be attempted even if some fail."""
        config = SyncConfig(
            backend=StorageBackend.LOCAL,
            inbox_path=str(tmp_path / "inbox"),
            auto_ingest=False,
        )
        mgr = CloudSyncManager(config, local_download_dir=str(tmp_path / "dl"))
        mock_adapter = AsyncMock()
        mgr._adapter = mock_adapter

        files = [
            _make_cloud_file(key="a.pdf", etag="e1"),
            _make_cloud_file(key="b.pdf", etag="e2"),
            _make_cloud_file(key="c.pdf", etag="e3"),
        ]
        mock_adapter.list_files.return_value = files
        # First fails, second succeeds, third fails
        mock_adapter.download_file.side_effect = [False, True, False]

        result = await mgr.sync_once()
        assert len(result) == 1  # Only b.pdf succeeded
        assert result[0].key == "b.pdf"
        assert mock_adapter.download_file.call_count == 3

    def test_cloud_file_filename_with_spaces(self):
        cf = _make_cloud_file(key="path/my document file.pdf")
        assert cf.filename == "my document file.pdf"

    def test_cloud_file_extension_hidden_file(self):
        cf = _make_cloud_file(key=".gitignore")
        assert cf.extension == ""

    @pytest.mark.asyncio
    async def test_local_adapter_download_preserves_content(self, tmp_path):
        """Ensure shutil.copy2 preserves byte-accurate content."""
        content = bytes(range(256)) * 100  # 25600 bytes of varied content
        adapter = LocalStorageAdapter(str(tmp_path / "src"))
        _write_file(tmp_path / "src" / "binary.dat", content)
        dest = tmp_path / "dest" / "binary.dat"
        result = await adapter.download_file("binary.dat", dest)
        assert result is True
        assert dest.read_bytes() == content

    @pytest.mark.asyncio
    async def test_poll_loop_calls_sync_once(self, tmp_path):
        """Verify _poll_loop calls sync_once repeatedly."""
        config = SyncConfig(
            backend=StorageBackend.LOCAL,
            inbox_path=str(tmp_path / "inbox"),
            poll_interval_seconds=0,  # No delay for test
        )
        mgr = CloudSyncManager(config, local_download_dir=str(tmp_path / "dl"))

        call_count = 0

        async def mock_sync_once():
            nonlocal call_count
            call_count += 1
            if call_count >= 3:
                mgr._running = False
            return []

        mgr.sync_once = mock_sync_once
        mgr._running = True
        await mgr._poll_loop()
        assert call_count >= 3

    @pytest.mark.asyncio
    async def test_poll_loop_handles_sync_exception(self, tmp_path):
        """Verify _poll_loop continues even when sync_once raises."""
        config = SyncConfig(
            backend=StorageBackend.LOCAL,
            inbox_path=str(tmp_path / "inbox"),
            poll_interval_seconds=0,
        )
        mgr = CloudSyncManager(config, local_download_dir=str(tmp_path / "dl"))

        call_count = 0

        async def mock_sync_once():
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise RuntimeError("transient error")
            if call_count >= 3:
                mgr._running = False
            return []

        mgr.sync_once = mock_sync_once
        mgr._running = True
        await mgr._poll_loop()
        assert call_count >= 3  # Continued past the error
