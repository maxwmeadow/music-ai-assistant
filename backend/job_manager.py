"""
Simple in-memory job manager for async processing
Handles long-running inference tasks that exceed HTTP timeout
"""

import asyncio
import uuid
from datetime import datetime, timedelta
from typing import Dict, Any, Optional
from enum import Enum


class JobStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class Job:
    def __init__(self, job_id: str):
        self.job_id = job_id
        self.status = JobStatus.PENDING
        self.created_at = datetime.now()
        self.updated_at = datetime.now()
        self.result: Optional[Dict[str, Any]] = None
        self.error: Optional[str] = None
        self.progress: int = 0  # 0-100

    def to_dict(self) -> Dict[str, Any]:
        return {
            "job_id": self.job_id,
            "status": self.status.value,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "result": self.result,
            "error": self.error,
            "progress": self.progress
        }


class JobManager:
    """
    In-memory job manager for async processing.
    Jobs expire after 1 hour.
    """

    def __init__(self):
        self.jobs: Dict[str, Job] = {}
        self.cleanup_interval = 3600  # 1 hour
        print("[JobManager] Initialized")

    def create_job(self) -> str:
        """Create a new job and return job_id"""
        job_id = str(uuid.uuid4())
        self.jobs[job_id] = Job(job_id)
        print(f"[JobManager] Created job: {job_id}")
        return job_id

    def get_job(self, job_id: str) -> Optional[Job]:
        """Get job by ID"""
        return self.jobs.get(job_id)

    def update_status(self, job_id: str, status: JobStatus, progress: int = 0):
        """Update job status"""
        if job_id in self.jobs:
            job = self.jobs[job_id]
            job.status = status
            job.progress = progress
            job.updated_at = datetime.now()
            print(f"[JobManager] Job {job_id}: {status.value} ({progress}%)")

    def set_result(self, job_id: str, result: Dict[str, Any]):
        """Set job result and mark as completed"""
        if job_id in self.jobs:
            job = self.jobs[job_id]
            job.result = result
            job.status = JobStatus.COMPLETED
            job.progress = 100
            job.updated_at = datetime.now()
            print(f"[JobManager] Job {job_id}: Completed")

    def set_error(self, job_id: str, error: str):
        """Set job error and mark as failed"""
        if job_id in self.jobs:
            job = self.jobs[job_id]
            job.error = error
            job.status = JobStatus.FAILED
            job.updated_at = datetime.now()
            print(f"[JobManager] Job {job_id}: Failed - {error}")

    def cleanup_expired(self, max_age_hours: int = 1):
        """Remove jobs older than max_age_hours"""
        now = datetime.now()
        expired = []

        for job_id, job in self.jobs.items():
            age = now - job.created_at
            if age > timedelta(hours=max_age_hours):
                expired.append(job_id)

        for job_id in expired:
            del self.jobs[job_id]

        if expired:
            print(f"[JobManager] Cleaned up {len(expired)} expired jobs")

        return len(expired)

    def get_stats(self) -> Dict[str, Any]:
        """Get job manager statistics"""
        statuses = {status.value: 0 for status in JobStatus}
        for job in self.jobs.values():
            statuses[job.status.value] += 1

        return {
            "total_jobs": len(self.jobs),
            "by_status": statuses
        }


# Global instance
_job_manager: Optional[JobManager] = None


def get_job_manager() -> JobManager:
    """Get or create global job manager instance"""
    global _job_manager
    if _job_manager is None:
        _job_manager = JobManager()
    return _job_manager
