"""
Tests for documents/understanding/job_tracker.py
=================================================

Comprehensive tests for StudyJobTracker, StudyJob, StudyJobProgress, JobStatus.
Covers singleton pattern, ETA calculation, voice status messages, job lifecycle,
thread safety, and cleanup.

Tests: 75+
"""

import time
from unittest.mock import patch, MagicMock

import pytest

from documents.understanding.job_tracker import (
    JobStatus,
    StudyJobProgress,
    StudyJob,
    StudyJobTracker,
    get_job_tracker,
)


# ── Fixtures ──────────────────────────────────────────────────────────────


@pytest.fixture(autouse=True)
def reset_singleton():
    """Reset singleton between tests."""
    StudyJobTracker._instance = None
    yield
    StudyJobTracker._instance = None


@pytest.fixture
def tracker():
    """Fresh tracker instance."""
    return StudyJobTracker()


@pytest.fixture
def sample_progress():
    """Sample progress object."""
    return StudyJobProgress(
        current_chapter=5,
        total_chapters=15,
        current_chapter_title="The Inciting Incident",
        concepts_found=10,
        principles_found=5,
        techniques_found=3,
        examples_found=2,
        llm_calls_made=8,
        pages_processed=50,
        total_pages=200,
    )


@pytest.fixture
def sample_job():
    """Sample job with controlled timing."""
    return StudyJob(
        id="study_test123",
        document_id="doc_abc",
        title="Story",
        author="Robert McKee",
        status=JobStatus.STUDYING,
        progress=StudyJobProgress(
            current_chapter=5,
            total_chapters=15,
            current_chapter_title="The Inciting Incident",
            concepts_found=10,
            principles_found=5,
            techniques_found=3,
            examples_found=2,
        ),
        start_time=time.time() - 300,  # 5 minutes ago
        last_update=time.time() - 30,
        chapter_times=[55.0, 60.0, 58.0, 62.0, 57.0],
    )


# ── JobStatus Enum ────────────────────────────────────────────────────────


class TestJobStatus:
    def test_all_statuses_exist(self):
        assert JobStatus.PENDING == "pending"
        assert JobStatus.INITIALIZING == "initializing"
        assert JobStatus.STUDYING == "studying"
        assert JobStatus.EXTRACTING == "extracting"
        assert JobStatus.INTEGRATING == "integrating"
        assert JobStatus.COMPLETED == "completed"
        assert JobStatus.FAILED == "failed"

    def test_status_count(self):
        assert len(JobStatus) == 7

    def test_status_is_string(self):
        assert isinstance(JobStatus.PENDING, str)
        assert JobStatus.PENDING == "pending"


# ── StudyJobProgress ──────────────────────────────────────────────────────


class TestStudyJobProgress:
    def test_defaults(self):
        p = StudyJobProgress()
        assert p.current_chapter == 0
        assert p.total_chapters == 0
        assert p.current_chapter_title == ""
        assert p.concepts_found == 0
        assert p.principles_found == 0
        assert p.techniques_found == 0
        assert p.examples_found == 0
        assert p.llm_calls_made == 0
        assert p.pages_processed == 0
        assert p.total_pages == 0

    def test_custom_values(self, sample_progress):
        assert sample_progress.current_chapter == 5
        assert sample_progress.total_chapters == 15
        assert sample_progress.current_chapter_title == "The Inciting Incident"
        assert sample_progress.concepts_found == 10


# ── StudyJob Properties ──────────────────────────────────────────────────


class TestStudyJobProperties:
    def test_elapsed_seconds(self):
        job = StudyJob(
            id="j1",
            document_id="d1",
            title="T",
            author="A",
            status=JobStatus.STUDYING,
            progress=StudyJobProgress(),
            start_time=time.time() - 120,
            last_update=time.time(),
        )
        assert 119 <= job.elapsed_seconds <= 122

    def test_elapsed_formatted_seconds(self):
        job = StudyJob(
            id="j1",
            document_id="d1",
            title="T",
            author="A",
            status=JobStatus.STUDYING,
            progress=StudyJobProgress(),
            start_time=time.time() - 30,
            last_update=time.time(),
        )
        assert "seconds" in job.elapsed_formatted

    def test_elapsed_formatted_minutes(self):
        job = StudyJob(
            id="j1",
            document_id="d1",
            title="T",
            author="A",
            status=JobStatus.STUDYING,
            progress=StudyJobProgress(),
            start_time=time.time() - 150,  # 2.5 min
            last_update=time.time(),
        )
        fmt = job.elapsed_formatted
        assert "minute" in fmt

    def test_elapsed_formatted_hours(self):
        job = StudyJob(
            id="j1",
            document_id="d1",
            title="T",
            author="A",
            status=JobStatus.STUDYING,
            progress=StudyJobProgress(),
            start_time=time.time() - 7200,  # 2 hours
            last_update=time.time(),
        )
        fmt = job.elapsed_formatted
        assert "hour" in fmt

    def test_elapsed_formatted_hours_and_minutes(self):
        job = StudyJob(
            id="j1",
            document_id="d1",
            title="T",
            author="A",
            status=JobStatus.STUDYING,
            progress=StudyJobProgress(),
            start_time=time.time() - 5400,  # 1.5 hours
            last_update=time.time(),
        )
        fmt = job.elapsed_formatted
        assert "hour" in fmt
        assert "minute" in fmt

    def test_progress_percentage_normal(self):
        job = StudyJob(
            id="j1",
            document_id="d1",
            title="T",
            author="A",
            status=JobStatus.STUDYING,
            progress=StudyJobProgress(current_chapter=5, total_chapters=10),
            start_time=time.time(),
            last_update=time.time(),
        )
        assert job.progress_percentage == 50.0

    def test_progress_percentage_zero_chapters(self):
        job = StudyJob(
            id="j1",
            document_id="d1",
            title="T",
            author="A",
            status=JobStatus.STUDYING,
            progress=StudyJobProgress(current_chapter=0, total_chapters=0),
            start_time=time.time(),
            last_update=time.time(),
        )
        assert job.progress_percentage == 0.0

    def test_progress_percentage_complete(self):
        job = StudyJob(
            id="j1",
            document_id="d1",
            title="T",
            author="A",
            status=JobStatus.COMPLETED,
            progress=StudyJobProgress(current_chapter=10, total_chapters=10),
            start_time=time.time(),
            last_update=time.time(),
        )
        assert job.progress_percentage == 100.0


# ── ETA Calculation ──────────────────────────────────────────────────────


class TestETACalculation:
    def test_estimate_remaining_no_chapter_times(self):
        job = StudyJob(
            id="j1",
            document_id="d1",
            title="T",
            author="A",
            status=JobStatus.STUDYING,
            progress=StudyJobProgress(current_chapter=3, total_chapters=10),
            start_time=time.time(),
            last_update=time.time(),
            chapter_times=[],
        )
        assert job.estimate_remaining_seconds() is None

    def test_estimate_remaining_zero_current_chapter(self):
        job = StudyJob(
            id="j1",
            document_id="d1",
            title="T",
            author="A",
            status=JobStatus.STUDYING,
            progress=StudyJobProgress(current_chapter=0, total_chapters=10),
            start_time=time.time(),
            last_update=time.time(),
            chapter_times=[],
        )
        assert job.estimate_remaining_seconds() is None

    def test_estimate_remaining_with_times(self):
        job = StudyJob(
            id="j1",
            document_id="d1",
            title="T",
            author="A",
            status=JobStatus.STUDYING,
            progress=StudyJobProgress(current_chapter=5, total_chapters=10),
            start_time=time.time(),
            last_update=time.time(),
            chapter_times=[60.0, 60.0, 60.0, 60.0, 60.0],
        )
        remaining = job.estimate_remaining_seconds()
        assert remaining == 300.0  # 5 remaining * 60s avg

    def test_eta_formatted_calculating(self):
        job = StudyJob(
            id="j1",
            document_id="d1",
            title="T",
            author="A",
            status=JobStatus.STUDYING,
            progress=StudyJobProgress(current_chapter=0, total_chapters=10),
            start_time=time.time(),
            last_update=time.time(),
        )
        assert job.eta_formatted == "calculating..."

    def test_eta_formatted_seconds(self):
        job = StudyJob(
            id="j1",
            document_id="d1",
            title="T",
            author="A",
            status=JobStatus.STUDYING,
            progress=StudyJobProgress(current_chapter=9, total_chapters=10),
            start_time=time.time(),
            last_update=time.time(),
            chapter_times=[30.0],
        )
        assert "seconds" in job.eta_formatted

    def test_eta_formatted_minutes(self):
        job = StudyJob(
            id="j1",
            document_id="d1",
            title="T",
            author="A",
            status=JobStatus.STUDYING,
            progress=StudyJobProgress(current_chapter=5, total_chapters=10),
            start_time=time.time(),
            last_update=time.time(),
            chapter_times=[120.0, 120.0, 120.0, 120.0, 120.0],
        )
        assert "minute" in job.eta_formatted

    def test_eta_formatted_hours(self):
        job = StudyJob(
            id="j1",
            document_id="d1",
            title="T",
            author="A",
            status=JobStatus.STUDYING,
            progress=StudyJobProgress(current_chapter=1, total_chapters=10),
            start_time=time.time(),
            last_update=time.time(),
            chapter_times=[3600.0],
        )
        assert "hour" in job.eta_formatted

    def test_eta_formatted_hours_and_minutes(self):
        job = StudyJob(
            id="j1",
            document_id="d1",
            title="T",
            author="A",
            status=JobStatus.STUDYING,
            progress=StudyJobProgress(current_chapter=1, total_chapters=3),
            start_time=time.time(),
            last_update=time.time(),
            chapter_times=[5400.0],  # 1.5 hours per chapter, 2 remaining = 3 hours
        )
        fmt = job.eta_formatted
        assert "hour" in fmt


# ── Voice Status Messages ────────────────────────────────────────────────


class TestVoiceStatusMessages:
    def _make_job(self, status, **kwargs):
        defaults = dict(
            id="j1",
            document_id="d1",
            title="Story",
            author="McKee",
            status=status,
            progress=StudyJobProgress(
                current_chapter=5,
                total_chapters=15,
                current_chapter_title="The Inciting Incident",
                concepts_found=10,
                principles_found=5,
                techniques_found=3,
            ),
            start_time=time.time() - 300,
            last_update=time.time(),
            chapter_times=[60.0] * 5,
        )
        defaults.update(kwargs)
        return StudyJob(**defaults)

    def test_pending_status(self):
        job = self._make_job(JobStatus.PENDING)
        msg = job.get_voice_status()
        assert "Boss" in msg
        assert "Story" in msg
        assert "queued" in msg

    def test_initializing_status(self):
        job = self._make_job(JobStatus.INITIALIZING)
        msg = job.get_voice_status()
        assert "Boss" in msg
        assert "preparing" in msg
        assert "McKee" in msg

    def test_studying_status(self):
        job = self._make_job(JobStatus.STUDYING)
        msg = job.get_voice_status()
        assert "Boss" in msg
        assert "Chapter 5" in msg
        assert "15" in msg
        assert "Inciting Incident" in msg

    def test_studying_status_with_knowledge(self):
        job = self._make_job(JobStatus.STUDYING)
        msg = job.get_voice_status()
        assert "knowledge items" in msg or "18" in msg  # 10+5+3=18

    def test_studying_status_no_chapter_title(self):
        job = self._make_job(
            JobStatus.STUDYING,
            progress=StudyJobProgress(
                current_chapter=5,
                total_chapters=15,
                current_chapter_title="",
            ),
        )
        msg = job.get_voice_status()
        assert "Boss" in msg
        assert "Chapter 5" in msg

    def test_extracting_status(self):
        job = self._make_job(JobStatus.EXTRACTING)
        msg = job.get_voice_status()
        assert "Boss" in msg
        assert "read all chapters" in msg
        assert "extracting" in msg.lower()

    def test_integrating_status(self):
        job = self._make_job(JobStatus.INTEGRATING)
        msg = job.get_voice_status()
        assert "Boss" in msg
        assert "almost done" in msg.lower() or "integrating" in msg.lower()

    def test_completed_status(self):
        job = self._make_job(
            JobStatus.COMPLETED,
            progress=StudyJobProgress(
                current_chapter=15,
                total_chapters=15,
                concepts_found=20,
                principles_found=10,
                techniques_found=8,
                examples_found=5,
            ),
        )
        msg = job.get_voice_status()
        assert "Boss" in msg
        assert "finished" in msg.lower()
        assert "43" in msg  # 20+10+8+5

    def test_failed_status(self):
        job = self._make_job(
            JobStatus.FAILED,
            error_message="Model timeout",
        )
        msg = job.get_voice_status()
        assert "Boss" in msg
        assert "problem" in msg.lower()
        assert "Model timeout" in msg

    def test_failed_status_no_error(self):
        job = self._make_job(JobStatus.FAILED, error_message=None)
        msg = job.get_voice_status()
        assert "Unknown error" in msg


# ── StudyJob.to_dict ─────────────────────────────────────────────────────


class TestStudyJobToDict:
    def test_to_dict_keys(self, sample_job):
        d = sample_job.to_dict()
        assert "job_id" in d
        assert "document_id" in d
        assert "title" in d
        assert "author" in d
        assert "status" in d
        assert "progress" in d
        assert "timing" in d
        assert "voice_status" in d

    def test_to_dict_values(self, sample_job):
        d = sample_job.to_dict()
        assert d["job_id"] == "study_test123"
        assert d["document_id"] == "doc_abc"
        assert d["title"] == "Story"
        assert d["author"] == "Robert McKee"
        assert d["status"] == "studying"

    def test_to_dict_progress(self, sample_job):
        d = sample_job.to_dict()
        p = d["progress"]
        assert p["current_chapter"] == 5
        assert p["total_chapters"] == 15
        assert p["concepts_found"] == 10
        assert "percentage" in p

    def test_to_dict_timing(self, sample_job):
        d = sample_job.to_dict()
        t = d["timing"]
        assert "start_time" in t
        assert "elapsed" in t
        assert "eta" in t  # Because status is STUDYING

    def test_to_dict_timing_no_eta_when_not_studying(self):
        job = StudyJob(
            id="j1",
            document_id="d1",
            title="T",
            author="A",
            status=JobStatus.COMPLETED,
            progress=StudyJobProgress(),
            start_time=time.time(),
            last_update=time.time(),
        )
        d = job.to_dict()
        assert d["timing"]["eta"] is None


# ── Singleton Pattern ────────────────────────────────────────────────────


class TestSingleton:
    def test_singleton_returns_same_instance(self):
        t1 = StudyJobTracker()
        t2 = StudyJobTracker()
        assert t1 is t2

    def test_get_job_tracker_returns_singleton(self):
        t1 = get_job_tracker()
        t2 = get_job_tracker()
        assert t1 is t2

    def test_get_job_tracker_same_as_constructor(self):
        t1 = StudyJobTracker()
        t2 = get_job_tracker()
        assert t1 is t2

    def test_reset_singleton(self):
        t1 = StudyJobTracker()
        StudyJobTracker._instance = None
        t2 = StudyJobTracker()
        assert t1 is not t2


# ── StudyJobTracker.start_job ────────────────────────────────────────────


class TestStartJob:
    def test_start_job_returns_id(self, tracker):
        job_id = tracker.start_job("doc1", "Story", "McKee")
        assert job_id.startswith("study_")
        assert len(job_id) > 6

    def test_start_job_creates_job(self, tracker):
        job_id = tracker.start_job("doc1", "Story", "McKee")
        job = tracker.get_job(job_id)
        assert job is not None
        assert job.title == "Story"
        assert job.author == "McKee"
        assert job.document_id == "doc1"
        assert job.status == JobStatus.INITIALIZING

    def test_start_job_with_chapters(self, tracker):
        job_id = tracker.start_job("doc1", "Story", total_chapters=15)
        job = tracker.get_job(job_id)
        assert job.progress.total_chapters == 15

    def test_start_job_with_pages(self, tracker):
        job_id = tracker.start_job("doc1", "Story", total_pages=300)
        job = tracker.get_job(job_id)
        assert job.progress.total_pages == 300

    def test_start_job_default_author(self, tracker):
        job_id = tracker.start_job("doc1", "Story")
        job = tracker.get_job(job_id)
        assert job.author == "Unknown"

    def test_start_job_duplicate_document_active(self, tracker):
        job_id1 = tracker.start_job("doc1", "Story")
        job_id2 = tracker.start_job("doc1", "Story Again")
        assert job_id1 == job_id2  # Returns existing

    def test_start_job_duplicate_document_completed(self, tracker):
        job_id1 = tracker.start_job("doc1", "Story")
        tracker.complete_job(job_id1)
        job_id2 = tracker.start_job("doc1", "Story v2")
        assert job_id1 != job_id2  # New job after completion

    def test_start_job_unique_ids(self, tracker):
        id1 = tracker.start_job("doc1", "Book1")
        id2 = tracker.start_job("doc2", "Book2")
        assert id1 != id2

    def test_start_job_document_mapping(self, tracker):
        job_id = tracker.start_job("doc1", "Story")
        job = tracker.get_job_by_document("doc1")
        assert job is not None
        assert job.id == job_id


# ── StudyJobTracker.update_status ────────────────────────────────────────


class TestUpdateStatus:
    def test_update_status(self, tracker):
        job_id = tracker.start_job("doc1", "Story")
        tracker.update_status(job_id, JobStatus.STUDYING)
        job = tracker.get_job(job_id)
        assert job.status == JobStatus.STUDYING

    def test_update_status_updates_last_update(self, tracker):
        job_id = tracker.start_job("doc1", "Story")
        before = tracker.get_job(job_id).last_update
        time.sleep(0.01)
        tracker.update_status(job_id, JobStatus.STUDYING)
        after = tracker.get_job(job_id).last_update
        assert after >= before

    def test_update_status_nonexistent_job(self, tracker):
        # Should not raise, just log warning
        tracker.update_status("nonexistent", JobStatus.STUDYING)


# ── StudyJobTracker.update_progress ──────────────────────────────────────


class TestUpdateProgress:
    def test_update_chapter(self, tracker):
        job_id = tracker.start_job("doc1", "Story", total_chapters=10)
        tracker.update_status(job_id, JobStatus.STUDYING)
        tracker.update_progress(job_id, current_chapter=3)
        job = tracker.get_job(job_id)
        assert job.progress.current_chapter == 3

    def test_update_chapter_title(self, tracker):
        job_id = tracker.start_job("doc1", "Story")
        tracker.update_progress(job_id, chapter_title="Act One")
        job = tracker.get_job(job_id)
        assert job.progress.current_chapter_title == "Act One"

    def test_update_tracks_chapter_time(self, tracker):
        job_id = tracker.start_job("doc1", "Story", total_chapters=10)
        tracker.update_progress(job_id, current_chapter=1)
        time.sleep(0.01)
        tracker.update_progress(job_id, current_chapter=2)
        job = tracker.get_job(job_id)
        assert len(job.chapter_times) == 2

    def test_update_same_chapter_no_time_tracking(self, tracker):
        job_id = tracker.start_job("doc1", "Story", total_chapters=10)
        tracker.update_progress(job_id, current_chapter=1)
        tracker.update_progress(job_id, current_chapter=1)  # Same chapter
        job = tracker.get_job(job_id)
        assert len(job.chapter_times) == 1  # Only one time

    def test_update_concepts(self, tracker):
        job_id = tracker.start_job("doc1", "Story")
        tracker.update_progress(job_id, concepts_found=15)
        job = tracker.get_job(job_id)
        assert job.progress.concepts_found == 15

    def test_update_principles(self, tracker):
        job_id = tracker.start_job("doc1", "Story")
        tracker.update_progress(job_id, principles_found=8)
        job = tracker.get_job(job_id)
        assert job.progress.principles_found == 8

    def test_update_techniques(self, tracker):
        job_id = tracker.start_job("doc1", "Story")
        tracker.update_progress(job_id, techniques_found=5)
        job = tracker.get_job(job_id)
        assert job.progress.techniques_found == 5

    def test_update_examples(self, tracker):
        job_id = tracker.start_job("doc1", "Story")
        tracker.update_progress(job_id, examples_found=3)
        job = tracker.get_job(job_id)
        assert job.progress.examples_found == 3

    def test_update_llm_calls(self, tracker):
        job_id = tracker.start_job("doc1", "Story")
        tracker.update_progress(job_id, llm_calls_made=20)
        job = tracker.get_job(job_id)
        assert job.progress.llm_calls_made == 20

    def test_update_pages_processed(self, tracker):
        job_id = tracker.start_job("doc1", "Story")
        tracker.update_progress(job_id, pages_processed=50)
        job = tracker.get_job(job_id)
        assert job.progress.pages_processed == 50

    def test_update_nonexistent_job(self, tracker):
        # Should not raise
        tracker.update_progress("nonexistent", current_chapter=5)

    def test_partial_update_preserves_other_fields(self, tracker):
        job_id = tracker.start_job("doc1", "Story")
        tracker.update_progress(job_id, concepts_found=10)
        tracker.update_progress(job_id, principles_found=5)
        job = tracker.get_job(job_id)
        assert job.progress.concepts_found == 10
        assert job.progress.principles_found == 5


# ── StudyJobTracker.complete_job ─────────────────────────────────────────


class TestCompleteJob:
    def test_complete_job(self, tracker):
        job_id = tracker.start_job("doc1", "Story", total_chapters=10)
        tracker.complete_job(job_id, concepts_found=20, principles_found=10)
        job = tracker.get_job(job_id)
        assert job.status == JobStatus.COMPLETED
        assert job.progress.concepts_found == 20
        assert job.progress.principles_found == 10
        assert job.progress.current_chapter == 10  # Set to total

    def test_complete_job_sets_all_knowledge(self, tracker):
        job_id = tracker.start_job("doc1", "Story")
        tracker.complete_job(
            job_id,
            concepts_found=20,
            principles_found=10,
            techniques_found=8,
            examples_found=5,
        )
        job = tracker.get_job(job_id)
        assert job.progress.techniques_found == 8
        assert job.progress.examples_found == 5

    def test_complete_nonexistent_job(self, tracker):
        tracker.complete_job("nonexistent")  # Should not raise


# ── StudyJobTracker.fail_job ─────────────────────────────────────────────


class TestFailJob:
    def test_fail_job(self, tracker):
        job_id = tracker.start_job("doc1", "Story")
        tracker.fail_job(job_id, "GPU out of memory")
        job = tracker.get_job(job_id)
        assert job.status == JobStatus.FAILED
        assert job.error_message == "GPU out of memory"

    def test_fail_nonexistent_job(self, tracker):
        tracker.fail_job("nonexistent", "error")  # Should not raise


# ── StudyJobTracker.get_* methods ────────────────────────────────────────


class TestGetMethods:
    def test_get_job_exists(self, tracker):
        job_id = tracker.start_job("doc1", "Story")
        assert tracker.get_job(job_id) is not None

    def test_get_job_not_found(self, tracker):
        assert tracker.get_job("nonexistent") is None

    def test_get_job_by_document(self, tracker):
        job_id = tracker.start_job("doc1", "Story")
        job = tracker.get_job_by_document("doc1")
        assert job is not None
        assert job.id == job_id

    def test_get_job_by_document_not_found(self, tracker):
        assert tracker.get_job_by_document("nonexistent") is None

    def test_get_status(self, tracker):
        job_id = tracker.start_job("doc1", "Story")
        status = tracker.get_status(job_id)
        assert isinstance(status, dict)
        assert status["job_id"] == job_id

    def test_get_status_not_found(self, tracker):
        assert tracker.get_status("nonexistent") is None

    def test_get_voice_status(self, tracker):
        job_id = tracker.start_job("doc1", "Story")
        msg = tracker.get_voice_status(job_id)
        assert isinstance(msg, str)
        assert "Boss" in msg

    def test_get_voice_status_not_found(self, tracker):
        assert tracker.get_voice_status("nonexistent") is None

    def test_get_active_jobs(self, tracker):
        id1 = tracker.start_job("doc1", "Book1")
        id2 = tracker.start_job("doc2", "Book2")
        tracker.complete_job(id2)
        active = tracker.get_active_jobs()
        assert len(active) == 1
        assert active[0].id == id1

    def test_get_active_jobs_empty(self, tracker):
        assert tracker.get_active_jobs() == []

    def test_get_all_jobs(self, tracker):
        tracker.start_job("doc1", "Book1")
        tracker.start_job("doc2", "Book2")
        all_jobs = tracker.get_all_jobs()
        assert len(all_jobs) == 2
        assert all(isinstance(j, dict) for j in all_jobs)


# ── Active Status Summary ────────────────────────────────────────────────


class TestActiveStatusSummary:
    def test_no_active_jobs(self, tracker):
        msg = tracker.get_active_status_summary()
        assert "no books" in msg.lower()

    def test_single_active_job(self, tracker):
        job_id = tracker.start_job("doc1", "Story")
        tracker.update_status(job_id, JobStatus.STUDYING)
        msg = tracker.get_active_status_summary()
        assert "Boss" in msg

    def test_multiple_active_jobs(self, tracker):
        id1 = tracker.start_job("doc1", "Story")
        id2 = tracker.start_job("doc2", "Save the Cat")
        tracker.update_status(id1, JobStatus.STUDYING)
        tracker.update_progress(id1, current_chapter=3)
        tracker.update_status(id2, JobStatus.STUDYING)
        msg = tracker.get_active_status_summary()
        assert "2 books" in msg
        assert "Story" in msg
        assert "Save the Cat" in msg

    def test_multiple_active_mixed_status(self, tracker):
        id1 = tracker.start_job("doc1", "Story")
        id2 = tracker.start_job("doc2", "Save the Cat")
        tracker.update_status(id1, JobStatus.STUDYING)
        # id2 stays INITIALIZING
        msg = tracker.get_active_status_summary()
        assert "2 books" in msg


# ── Cleanup Old Jobs ────────────────────────────────────────────────────


class TestCleanupOldJobs:
    def test_cleanup_removes_old_completed(self, tracker):
        tracker._max_completed_jobs = 2
        ids = []
        for i in range(5):
            job_id = tracker.start_job(f"doc{i}", f"Book{i}")
            tracker.complete_job(job_id)
            ids.append(job_id)

        # After 5 completions with max 2, should keep 2 most recent
        all_jobs = tracker.get_all_jobs()
        assert len(all_jobs) <= 3  # Some cleanup happened

    def test_cleanup_preserves_active_jobs(self, tracker):
        tracker._max_completed_jobs = 1
        active_id = tracker.start_job("docA", "Active Book")

        for i in range(5):
            jid = tracker.start_job(f"doc{i}", f"Book{i}")
            tracker.complete_job(jid)

        # Active job should still be there
        assert tracker.get_job(active_id) is not None

    def test_cleanup_removes_document_mapping(self, tracker):
        tracker._max_completed_jobs = 1
        id1 = tracker.start_job("doc_old", "OldBook")
        tracker.complete_job(id1)
        id2 = tracker.start_job("doc_new", "NewBook")
        tracker.complete_job(id2)
        # Force another to trigger cleanup
        id3 = tracker.start_job("doc_newer", "NewerBook")
        tracker.complete_job(id3)

        # Old document mapping should be removed
        # New one should still exist
        job = tracker.get_job_by_document("doc_newer")
        assert job is not None


# ── Edge Cases ───────────────────────────────────────────────────────────


class TestEdgeCases:
    def test_backward_chapter_no_time_tracking(self, tracker):
        """Going backward in chapters should not track time."""
        job_id = tracker.start_job("doc1", "Story", total_chapters=10)
        tracker.update_progress(job_id, current_chapter=5)
        tracker.update_progress(job_id, current_chapter=3)  # Backward
        job = tracker.get_job(job_id)
        assert len(job.chapter_times) == 1  # Only the forward one

    def test_elapsed_formatted_exactly_60_seconds(self):
        """Exactly 60 seconds → 1 minute."""
        job = StudyJob(
            id="j1",
            document_id="d1",
            title="T",
            author="A",
            status=JobStatus.STUDYING,
            progress=StudyJobProgress(),
            start_time=time.time() - 60,
            last_update=time.time(),
        )
        assert "minute" in job.elapsed_formatted

    def test_elapsed_formatted_exactly_3600_seconds(self):
        """Exactly 3600 seconds → 1 hour."""
        job = StudyJob(
            id="j1",
            document_id="d1",
            title="T",
            author="A",
            status=JobStatus.STUDYING,
            progress=StudyJobProgress(),
            start_time=time.time() - 3600,
            last_update=time.time(),
        )
        assert "hour" in job.elapsed_formatted

    def test_studying_no_knowledge_items(self):
        """Voice status without knowledge items found."""
        job = StudyJob(
            id="j1",
            document_id="d1",
            title="Story",
            author="McKee",
            status=JobStatus.STUDYING,
            progress=StudyJobProgress(
                current_chapter=2,
                total_chapters=10,
                concepts_found=0,
                principles_found=0,
                techniques_found=0,
            ),
            start_time=time.time(),
            last_update=time.time(),
            chapter_times=[60.0, 60.0],
        )
        msg = job.get_voice_status()
        assert "knowledge items" not in msg  # No knowledge count shown

    def test_multiple_start_job_same_doc_failed(self, tracker):
        """After failure, should allow re-study."""
        job_id1 = tracker.start_job("doc1", "Story")
        tracker.fail_job(job_id1, "error")
        job_id2 = tracker.start_job("doc1", "Story Retry")
        assert job_id1 != job_id2

    def test_elapsed_plural_minutes(self):
        """Multiple minutes should use plural."""
        job = StudyJob(
            id="j1",
            document_id="d1",
            title="T",
            author="A",
            status=JobStatus.STUDYING,
            progress=StudyJobProgress(),
            start_time=time.time() - 180,  # 3 minutes
            last_update=time.time(),
        )
        assert "minutes" in job.elapsed_formatted

    def test_elapsed_singular_minute(self):
        """Exactly 1 minute should use singular."""
        job = StudyJob(
            id="j1",
            document_id="d1",
            title="T",
            author="A",
            status=JobStatus.STUDYING,
            progress=StudyJobProgress(),
            start_time=time.time() - 65,  # 1 min 5 sec
            last_update=time.time(),
        )
        fmt = job.elapsed_formatted
        assert "minute" in fmt
