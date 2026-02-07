"""
Study Job Tracker for Live Progress Monitoring
===============================================

Tracks active book study jobs with real-time progress updates,
ETA calculation, and voice-friendly status messages.

Example usage:
    tracker = StudyJobTracker()
    job_id = tracker.start_job(document_id, title, total_chapters=15)

    # During processing:
    tracker.update_progress(job_id, current_chapter=5, chapter_title="The Inciting Incident")

    # When user asks "what's the status?":
    status = tracker.get_status(job_id)
    # Returns: "Boss, I'm on Chapter 5 of 15 - 'The Inciting Incident'. About 8 minutes remaining."
"""

from __future__ import annotations

import time
import uuid
import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Any
from threading import Lock

LOGGER = logging.getLogger(__name__)


class JobStatus(str, Enum):
    """Status of a study job."""

    PENDING = "pending"
    INITIALIZING = "initializing"
    STUDYING = "studying"
    EXTRACTING = "extracting"
    INTEGRATING = "integrating"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class StudyJobProgress:
    """Detailed progress information for a study job."""

    current_chapter: int = 0
    total_chapters: int = 0
    current_chapter_title: str = ""
    concepts_found: int = 0
    principles_found: int = 0
    techniques_found: int = 0
    examples_found: int = 0
    llm_calls_made: int = 0
    pages_processed: int = 0
    total_pages: int = 0


@dataclass
class StudyJob:
    """Represents an active book study job."""

    id: str
    document_id: str
    title: str
    author: str
    status: JobStatus
    progress: StudyJobProgress
    start_time: float  # time.time()
    last_update: float
    error_message: Optional[str] = None
    chapter_times: List[float] = field(default_factory=list)  # Time per chapter for ETA

    @property
    def elapsed_seconds(self) -> float:
        """Total elapsed time in seconds."""
        return time.time() - self.start_time

    @property
    def elapsed_formatted(self) -> str:
        """Human-readable elapsed time."""
        elapsed = self.elapsed_seconds
        if elapsed < 60:
            return f"{int(elapsed)} seconds"
        elif elapsed < 3600:
            mins = int(elapsed // 60)
            secs = int(elapsed % 60)
            return f"{mins} minute{'s' if mins > 1 else ''}" + (
                f" {secs} seconds" if secs > 0 else ""
            )
        else:
            hours = int(elapsed // 3600)
            mins = int((elapsed % 3600) // 60)
            return f"{hours} hour{'s' if hours > 1 else ''} {mins} minute{'s' if mins > 1 else ''}"

    def estimate_remaining_seconds(self) -> Optional[float]:
        """Estimate remaining time based on chapter processing times."""
        if not self.chapter_times or self.progress.current_chapter == 0:
            return None

        # Average time per chapter
        avg_time = sum(self.chapter_times) / len(self.chapter_times)
        remaining_chapters = (
            self.progress.total_chapters - self.progress.current_chapter
        )

        return avg_time * remaining_chapters

    @property
    def eta_formatted(self) -> str:
        """Human-readable ETA."""
        remaining = self.estimate_remaining_seconds()
        if remaining is None:
            return "calculating..."

        if remaining < 60:
            return f"about {int(remaining)} seconds"
        elif remaining < 3600:
            mins = int(remaining // 60)
            return f"about {mins} minute{'s' if mins > 1 else ''}"
        else:
            hours = int(remaining // 3600)
            mins = int((remaining % 3600) // 60)
            if mins > 0:
                return f"about {hours} hour{'s' if hours > 1 else ''} {mins} minute{'s' if mins > 1 else ''}"
            return f"about {hours} hour{'s' if hours > 1 else ''}"

    @property
    def progress_percentage(self) -> float:
        """Progress as percentage (0-100)."""
        if self.progress.total_chapters == 0:
            return 0.0
        return (self.progress.current_chapter / self.progress.total_chapters) * 100

    def get_voice_status(self) -> str:
        """Get a voice-friendly status message."""
        if self.status == JobStatus.PENDING:
            return f"Boss, '{self.title}' is queued and waiting to start."

        elif self.status == JobStatus.INITIALIZING:
            return f"Boss, I'm preparing to study '{self.title}' by {self.author}. Setting things up now."

        elif self.status == JobStatus.STUDYING:
            chapter_info = ""
            if self.progress.current_chapter_title:
                chapter_info = f" - '{self.progress.current_chapter_title}'"

            progress_msg = (
                f"Boss, I'm on Chapter {self.progress.current_chapter} of "
                f"{self.progress.total_chapters}{chapter_info}. "
                f"{self.eta_formatted.capitalize()} remaining."
            )

            # Add knowledge count if interesting
            total_knowledge = (
                self.progress.concepts_found
                + self.progress.principles_found
                + self.progress.techniques_found
            )
            if total_knowledge > 0:
                progress_msg += f" Found {total_knowledge} knowledge items so far."

            return progress_msg

        elif self.status == JobStatus.EXTRACTING:
            return (
                f"Boss, I've read all chapters of '{self.title}'. "
                f"Now extracting and organizing the knowledge."
            )

        elif self.status == JobStatus.INTEGRATING:
            return (
                f"Boss, almost done with '{self.title}'. "
                f"Integrating the knowledge into my memory now."
            )

        elif self.status == JobStatus.COMPLETED:
            total = (
                self.progress.concepts_found
                + self.progress.principles_found
                + self.progress.techniques_found
                + self.progress.examples_found
            )
            return (
                f"Boss, I've finished studying '{self.title}'! "
                f"Extracted {total} knowledge items in {self.elapsed_formatted}. "
                f"Ready to mentor you on this material."
            )

        elif self.status == JobStatus.FAILED:
            return (
                f"Boss, I ran into a problem studying '{self.title}': "
                f"{self.error_message or 'Unknown error'}. "
                f"Want me to try again?"
            )

        return f"Boss, '{self.title}' status: {self.status.value}"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API responses."""
        return {
            "job_id": self.id,
            "document_id": self.document_id,
            "title": self.title,
            "author": self.author,
            "status": self.status.value,
            "progress": {
                "current_chapter": self.progress.current_chapter,
                "total_chapters": self.progress.total_chapters,
                "current_chapter_title": self.progress.current_chapter_title,
                "percentage": round(self.progress_percentage, 1),
                "concepts_found": self.progress.concepts_found,
                "principles_found": self.progress.principles_found,
                "techniques_found": self.progress.techniques_found,
                "examples_found": self.progress.examples_found,
                "llm_calls_made": self.progress.llm_calls_made,
            },
            "timing": {
                "start_time": datetime.fromtimestamp(self.start_time).isoformat(),
                "elapsed": self.elapsed_formatted,
                "eta": (
                    self.eta_formatted if self.status == JobStatus.STUDYING else None
                ),
            },
            "voice_status": self.get_voice_status(),
            "error": self.error_message,
        }


class StudyJobTracker:
    """
    Singleton tracker for all active and recent study jobs.

    Thread-safe for concurrent job updates.
    """

    _instance: Optional["StudyJobTracker"] = None
    _lock = Lock()

    def __new__(cls) -> "StudyJobTracker":
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return

        self._jobs: Dict[str, StudyJob] = {}
        self._document_to_job: Dict[str, str] = {}  # document_id -> job_id mapping
        self._job_lock = Lock()
        self._max_completed_jobs = 10  # Keep last N completed jobs
        self._initialized = True

        LOGGER.info("StudyJobTracker initialized")

    def start_job(
        self,
        document_id: str,
        title: str,
        author: str = "Unknown",
        total_chapters: int = 0,
        total_pages: int = 0,
    ) -> str:
        """
        Start tracking a new study job.

        Returns:
            Job ID for tracking
        """
        job_id = f"study_{uuid.uuid4().hex[:12]}"

        with self._job_lock:
            # Check if document already being studied
            if document_id in self._document_to_job:
                existing_job_id = self._document_to_job[document_id]
                existing_job = self._jobs.get(existing_job_id)
                if existing_job and existing_job.status in (
                    JobStatus.PENDING,
                    JobStatus.STUDYING,
                    JobStatus.INITIALIZING,
                ):
                    LOGGER.warning(
                        f"Document {document_id} already being studied: {existing_job_id}"
                    )
                    return existing_job_id

            job = StudyJob(
                id=job_id,
                document_id=document_id,
                title=title,
                author=author,
                status=JobStatus.INITIALIZING,
                progress=StudyJobProgress(
                    total_chapters=total_chapters,
                    total_pages=total_pages,
                ),
                start_time=time.time(),
                last_update=time.time(),
            )

            self._jobs[job_id] = job
            self._document_to_job[document_id] = job_id

            LOGGER.info(f"Started study job {job_id} for '{title}'")

        return job_id

    def update_status(self, job_id: str, status: JobStatus) -> None:
        """Update job status."""
        with self._job_lock:
            if job_id not in self._jobs:
                LOGGER.warning(f"Job {job_id} not found for status update")
                return

            job = self._jobs[job_id]
            job.status = status
            job.last_update = time.time()

            LOGGER.debug(f"Job {job_id} status: {status.value}")

    def update_progress(
        self,
        job_id: str,
        current_chapter: Optional[int] = None,
        chapter_title: Optional[str] = None,
        concepts_found: Optional[int] = None,
        principles_found: Optional[int] = None,
        techniques_found: Optional[int] = None,
        examples_found: Optional[int] = None,
        llm_calls_made: Optional[int] = None,
        pages_processed: Optional[int] = None,
    ) -> None:
        """Update job progress with partial updates."""
        with self._job_lock:
            if job_id not in self._jobs:
                LOGGER.warning(f"Job {job_id} not found for progress update")
                return

            job = self._jobs[job_id]
            now = time.time()

            # Track chapter completion time for ETA
            if (
                current_chapter is not None
                and current_chapter > job.progress.current_chapter
            ):
                chapter_time = now - job.last_update
                job.chapter_times.append(chapter_time)
                job.progress.current_chapter = current_chapter

            if chapter_title is not None:
                job.progress.current_chapter_title = chapter_title
            if concepts_found is not None:
                job.progress.concepts_found = concepts_found
            if principles_found is not None:
                job.progress.principles_found = principles_found
            if techniques_found is not None:
                job.progress.techniques_found = techniques_found
            if examples_found is not None:
                job.progress.examples_found = examples_found
            if llm_calls_made is not None:
                job.progress.llm_calls_made = llm_calls_made
            if pages_processed is not None:
                job.progress.pages_processed = pages_processed

            job.last_update = now

            LOGGER.debug(
                f"Job {job_id} progress: chapter {job.progress.current_chapter}/"
                f"{job.progress.total_chapters}"
            )

    def complete_job(
        self,
        job_id: str,
        concepts_found: int = 0,
        principles_found: int = 0,
        techniques_found: int = 0,
        examples_found: int = 0,
    ) -> None:
        """Mark job as completed."""
        with self._job_lock:
            if job_id not in self._jobs:
                LOGGER.warning(f"Job {job_id} not found for completion")
                return

            job = self._jobs[job_id]
            job.status = JobStatus.COMPLETED
            job.progress.concepts_found = concepts_found
            job.progress.principles_found = principles_found
            job.progress.techniques_found = techniques_found
            job.progress.examples_found = examples_found
            job.progress.current_chapter = job.progress.total_chapters
            job.last_update = time.time()

            LOGGER.info(
                f"Job {job_id} completed: {concepts_found} concepts, "
                f"{principles_found} principles, {techniques_found} techniques"
            )

            # Cleanup old completed jobs
            self._cleanup_old_jobs()

    def fail_job(self, job_id: str, error_message: str) -> None:
        """Mark job as failed."""
        with self._job_lock:
            if job_id not in self._jobs:
                LOGGER.warning(f"Job {job_id} not found for failure")
                return

            job = self._jobs[job_id]
            job.status = JobStatus.FAILED
            job.error_message = error_message
            job.last_update = time.time()

            LOGGER.error(f"Job {job_id} failed: {error_message}")

    def get_job(self, job_id: str) -> Optional[StudyJob]:
        """Get job by ID."""
        with self._job_lock:
            return self._jobs.get(job_id)

    def get_job_by_document(self, document_id: str) -> Optional[StudyJob]:
        """Get active job for a document."""
        with self._job_lock:
            job_id = self._document_to_job.get(document_id)
            if job_id:
                return self._jobs.get(job_id)
            return None

    def get_status(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Get job status as dictionary."""
        job = self.get_job(job_id)
        if job:
            return job.to_dict()
        return None

    def get_voice_status(self, job_id: str) -> Optional[str]:
        """Get voice-friendly status message."""
        job = self.get_job(job_id)
        if job:
            return job.get_voice_status()
        return None

    def get_active_jobs(self) -> List[StudyJob]:
        """Get all active (non-completed, non-failed) jobs."""
        with self._job_lock:
            return [
                job
                for job in self._jobs.values()
                if job.status not in (JobStatus.COMPLETED, JobStatus.FAILED)
            ]

    def get_all_jobs(self) -> List[Dict[str, Any]]:
        """Get all jobs as dictionaries."""
        with self._job_lock:
            return [job.to_dict() for job in self._jobs.values()]

    def get_active_status_summary(self) -> str:
        """Get a summary of all active jobs for voice response."""
        active = self.get_active_jobs()

        if not active:
            return "Boss, no books are currently being studied."

        if len(active) == 1:
            return active[0].get_voice_status()

        # Multiple active jobs
        summaries = []
        for job in active:
            if job.status == JobStatus.STUDYING:
                summaries.append(
                    f"'{job.title}' ({job.progress.current_chapter}/"
                    f"{job.progress.total_chapters} chapters)"
                )
            else:
                summaries.append(f"'{job.title}' ({job.status.value})")

        return f"Boss, I'm studying {len(active)} books: {', '.join(summaries)}"

    def _cleanup_old_jobs(self) -> None:
        """Remove old completed jobs to prevent memory growth."""
        completed = [
            (job_id, job)
            for job_id, job in self._jobs.items()
            if job.status in (JobStatus.COMPLETED, JobStatus.FAILED)
        ]

        if len(completed) > self._max_completed_jobs:
            # Sort by completion time, oldest first
            completed.sort(key=lambda x: x[1].last_update)
            to_remove = completed[: -self._max_completed_jobs]

            for job_id, job in to_remove:
                del self._jobs[job_id]
                if job.document_id in self._document_to_job:
                    if self._document_to_job[job.document_id] == job_id:
                        del self._document_to_job[job.document_id]

            LOGGER.debug(f"Cleaned up {len(to_remove)} old jobs")


# Convenience function to get the singleton
def get_job_tracker() -> StudyJobTracker:
    """Get the global StudyJobTracker instance."""
    return StudyJobTracker()
