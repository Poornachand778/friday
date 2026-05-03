"""
Cloud Storage Sync for Friday AI
================================

Handles file synchronization between cloud storage (S3/iCloud) and Friday server.
This enables the workflow where users drop files in a shared folder and Friday
automatically processes them.

Supported backends:
- AWS S3 (primary for server deployment)
- Local directory (for development/local mode)
- iCloud Drive (via mounted path on Mac)
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import os
import shutil
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import AsyncIterator, Dict, List, Optional

LOGGER = logging.getLogger(__name__)


class StorageBackend(str, Enum):
    """Supported storage backends"""

    LOCAL = "local"  # Local directory (dev mode)
    S3 = "s3"  # AWS S3 bucket
    ICLOUD = "icloud"  # iCloud Drive (mounted)


@dataclass
class CloudFile:
    """Represents a file in cloud storage"""

    key: str  # Relative path/key
    size: int
    last_modified: datetime
    etag: Optional[str] = None  # For change detection
    content_type: Optional[str] = None

    @property
    def filename(self) -> str:
        return Path(self.key).name

    @property
    def extension(self) -> str:
        return Path(self.key).suffix.lower()


@dataclass
class SyncConfig:
    """Configuration for cloud sync"""

    backend: StorageBackend = StorageBackend.LOCAL

    # Local/iCloud settings
    inbox_path: str = ""  # Watch this directory

    # S3 settings
    s3_bucket: str = ""
    s3_prefix: str = "friday-inbox/"
    s3_region: str = "us-east-1"

    # Sync behavior
    poll_interval_seconds: int = 60
    auto_delete_after_process: bool = False
    supported_extensions: List[str] = field(
        default_factory=lambda: [".pdf", ".epub", ".txt", ".md"]
    )

    # Processing
    auto_ingest: bool = True  # Auto-process new files


class StorageAdapter(ABC):
    """Abstract base for storage backends"""

    @abstractmethod
    async def list_files(self, prefix: str = "") -> List[CloudFile]:
        """List files in storage"""
        pass

    @abstractmethod
    async def download_file(self, key: str, local_path: Path) -> bool:
        """Download file to local path"""
        pass

    @abstractmethod
    async def delete_file(self, key: str) -> bool:
        """Delete file from storage"""
        pass

    @abstractmethod
    async def file_exists(self, key: str) -> bool:
        """Check if file exists"""
        pass


class LocalStorageAdapter(StorageAdapter):
    """Local directory storage (for dev/local mode)"""

    def __init__(self, base_path: str):
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)

    async def list_files(self, prefix: str = "") -> List[CloudFile]:
        """List files in local directory"""
        files = []
        search_path = self.base_path / prefix if prefix else self.base_path

        if not search_path.exists():
            return files

        for path in search_path.rglob("*"):
            if path.is_file():
                stat = path.stat()
                rel_path = path.relative_to(self.base_path)
                files.append(
                    CloudFile(
                        key=str(rel_path),
                        size=stat.st_size,
                        last_modified=datetime.fromtimestamp(stat.st_mtime),
                        etag=self._compute_etag(path),
                    )
                )

        return files

    async def download_file(self, key: str, local_path: Path) -> bool:
        """Copy file to local path"""
        source = self.base_path / key
        if not source.exists():
            return False

        local_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, local_path)
        return True

    async def delete_file(self, key: str) -> bool:
        """Delete file"""
        path = self.base_path / key
        if path.exists():
            path.unlink()
            return True
        return False

    async def file_exists(self, key: str) -> bool:
        """Check if file exists"""
        return (self.base_path / key).exists()

    def _compute_etag(self, path: Path) -> str:
        """Compute MD5 hash for change detection"""
        hasher = hashlib.md5()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                hasher.update(chunk)
        return hasher.hexdigest()


class S3StorageAdapter(StorageAdapter):
    """AWS S3 storage adapter"""

    def __init__(self, bucket: str, region: str = "us-east-1"):
        self.bucket = bucket
        self.region = region
        self._client = None

    async def _get_client(self):
        """Lazy load boto3 client"""
        if self._client is None:
            try:
                import boto3

                self._client = boto3.client("s3", region_name=self.region)
            except ImportError:
                raise ImportError("boto3 required for S3 storage: pip install boto3")
        return self._client

    async def list_files(self, prefix: str = "") -> List[CloudFile]:
        """List files in S3 bucket"""
        client = await self._get_client()
        files = []

        paginator = client.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=self.bucket, Prefix=prefix):
            for obj in page.get("Contents", []):
                files.append(
                    CloudFile(
                        key=obj["Key"],
                        size=obj["Size"],
                        last_modified=obj["LastModified"],
                        etag=obj.get("ETag", "").strip('"'),
                    )
                )

        return files

    async def download_file(self, key: str, local_path: Path) -> bool:
        """Download file from S3"""
        client = await self._get_client()
        local_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            client.download_file(self.bucket, key, str(local_path))
            return True
        except Exception as e:
            LOGGER.error(f"S3 download failed for {key}: {e}")
            return False

    async def delete_file(self, key: str) -> bool:
        """Delete file from S3"""
        client = await self._get_client()
        try:
            client.delete_object(Bucket=self.bucket, Key=key)
            return True
        except Exception as e:
            LOGGER.error(f"S3 delete failed for {key}: {e}")
            return False

    async def file_exists(self, key: str) -> bool:
        """Check if file exists in S3"""
        client = await self._get_client()
        try:
            client.head_object(Bucket=self.bucket, Key=key)
            return True
        except:
            return False


class CloudSyncManager:
    """
    Manages file synchronization between cloud storage and Friday.

    Workflow:
    1. User drops PDF in shared folder (S3/iCloud/local inbox)
    2. CloudSyncManager polls for new files
    3. New files are downloaded to Friday's processing directory
    4. DocumentManager ingests the files
    5. Optionally, original file is deleted from inbox

    Usage:
        sync = CloudSyncManager(config)
        await sync.start()  # Starts background polling

        # Or manual sync
        new_files = await sync.sync_once()
    """

    def __init__(
        self,
        config: SyncConfig,
        document_manager=None,  # Optional: for auto-ingest
        local_download_dir: str = "documents/data/inbox",
    ):
        self.config = config
        self.document_manager = document_manager
        self.local_download_dir = Path(local_download_dir)
        self.local_download_dir.mkdir(parents=True, exist_ok=True)

        # Track processed files
        self._processed_etags: Dict[str, str] = {}
        self._running = False
        self._task: Optional[asyncio.Task] = None

        # Initialize storage adapter
        self._adapter = self._create_adapter()

    def _create_adapter(self) -> StorageAdapter:
        """Create appropriate storage adapter"""
        if self.config.backend == StorageBackend.S3:
            return S3StorageAdapter(
                bucket=self.config.s3_bucket,
                region=self.config.s3_region,
            )
        elif self.config.backend == StorageBackend.ICLOUD:
            # iCloud is just a special local path on Mac
            icloud_path = self.config.inbox_path or os.path.expanduser(
                "~/Library/Mobile Documents/com~apple~CloudDocs/Friday-Inbox"
            )
            return LocalStorageAdapter(icloud_path)
        else:
            # Local storage
            return LocalStorageAdapter(
                self.config.inbox_path or str(self.local_download_dir)
            )

    async def start(self) -> None:
        """Start background sync polling"""
        if self._running:
            return

        self._running = True
        self._task = asyncio.create_task(self._poll_loop())
        LOGGER.info(
            f"CloudSync started: backend={self.config.backend.value}, "
            f"interval={self.config.poll_interval_seconds}s"
        )

    async def stop(self) -> None:
        """Stop background sync"""
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        LOGGER.info("CloudSync stopped")

    async def _poll_loop(self) -> None:
        """Background polling loop"""
        while self._running:
            try:
                await self.sync_once()
            except Exception as e:
                LOGGER.error(f"Sync error: {e}")

            await asyncio.sleep(self.config.poll_interval_seconds)

    async def sync_once(self) -> List[CloudFile]:
        """
        Perform one sync cycle.

        Returns:
            List of newly processed files
        """
        prefix = (
            self.config.s3_prefix if self.config.backend == StorageBackend.S3 else ""
        )

        # List files in cloud storage
        files = await self._adapter.list_files(prefix)

        # Filter to supported extensions
        files = [f for f in files if f.extension in self.config.supported_extensions]

        new_files = []
        for cloud_file in files:
            # Skip if already processed (same etag)
            if cloud_file.key in self._processed_etags:
                if self._processed_etags[cloud_file.key] == cloud_file.etag:
                    continue

            # Download file
            local_path = self.local_download_dir / cloud_file.filename
            success = await self._adapter.download_file(cloud_file.key, local_path)

            if not success:
                LOGGER.warning(f"Failed to download: {cloud_file.key}")
                continue

            LOGGER.info(f"Downloaded: {cloud_file.filename}")
            new_files.append(cloud_file)

            # Auto-ingest if configured
            if self.config.auto_ingest and self.document_manager:
                try:
                    await self._ingest_file(local_path, cloud_file)
                except Exception as e:
                    LOGGER.error(f"Auto-ingest failed for {cloud_file.filename}: {e}")

            # Mark as processed
            self._processed_etags[cloud_file.key] = cloud_file.etag

            # Delete from cloud if configured
            if self.config.auto_delete_after_process:
                await self._adapter.delete_file(cloud_file.key)
                LOGGER.info(f"Deleted from cloud: {cloud_file.key}")

        if new_files:
            LOGGER.info(f"Synced {len(new_files)} new files")

        return new_files

    async def _ingest_file(self, local_path: Path, cloud_file: CloudFile) -> None:
        """Auto-ingest file into document system"""
        if not self.document_manager:
            return

        # Determine document type from extension
        from documents.models import DocumentType

        ext = cloud_file.extension
        if ext == ".pdf":
            doc_type = DocumentType.BOOK  # Default for PDFs
        elif ext in [".txt", ".md"]:
            doc_type = DocumentType.ARTICLE
        else:
            doc_type = DocumentType.REFERENCE

        # Ingest
        await self.document_manager.ingest_document(
            file_path=str(local_path),
            title=cloud_file.filename.rsplit(".", 1)[0],  # Filename without extension
            document_type=doc_type,
        )

    async def list_pending(self) -> List[CloudFile]:
        """List files in cloud storage that haven't been processed"""
        prefix = (
            self.config.s3_prefix if self.config.backend == StorageBackend.S3 else ""
        )
        files = await self._adapter.list_files(prefix)

        return [
            f
            for f in files
            if f.extension in self.config.supported_extensions
            and f.key not in self._processed_etags
        ]

    def get_status(self) -> Dict:
        """Get sync status"""
        return {
            "running": self._running,
            "backend": self.config.backend.value,
            "processed_count": len(self._processed_etags),
            "poll_interval": self.config.poll_interval_seconds,
            "auto_ingest": self.config.auto_ingest,
        }


def create_sync_manager(
    backend: str = "local",
    inbox_path: str = "",
    s3_bucket: str = "",
    s3_prefix: str = "friday-inbox/",
    document_manager=None,
) -> CloudSyncManager:
    """
    Factory function to create CloudSyncManager.

    Args:
        backend: "local", "s3", or "icloud"
        inbox_path: Local path for local/icloud backend
        s3_bucket: S3 bucket name
        s3_prefix: S3 key prefix
        document_manager: Optional DocumentManager for auto-ingest

    Returns:
        Configured CloudSyncManager
    """
    config = SyncConfig(
        backend=StorageBackend(backend),
        inbox_path=inbox_path,
        s3_bucket=s3_bucket,
        s3_prefix=s3_prefix,
    )

    return CloudSyncManager(config, document_manager)
