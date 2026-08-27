import heapq

from dataclasses import dataclass, field
from typing import Optional

from app.jobs.models import (
    JobNotFoundError,
    JobPriority,
    JobStateError,
    JobStatus,
    WorkflowJob,
)


@dataclass(
    order=True
)
class _QueueEntry:
    priority: int

    sequence: int

    job_id: str = field(
        compare=False
    )


class PriorityJobQueue:
    """
    Deterministic in-memory priority queue
    for NEXUS workflow jobs.

    Ordering rules:
    - lower numeric priority runs first
    - equal priority preserves FIFO order
    """

    def __init__(
        self,
    ):
        self._heap: list[
            _QueueEntry
        ] = []

        self._jobs: dict[
            str,
            WorkflowJob
        ] = {}

        self._sequence = 0

    def enqueue(
        self,
        job: WorkflowJob,
    ) -> None:
        if job.id in self._jobs:
            raise JobStateError(
                "Job already exists in queue: "
                f"{job.id}"
            )

        if (
            job.status
            != JobStatus.QUEUED
        ):
            raise JobStateError(
                "Only queued jobs can "
                "be enqueued."
            )

        entry = _QueueEntry(
            priority=(
                job.priority.value
            ),
            sequence=(
                self._sequence
            ),
            job_id=job.id,
        )

        self._sequence += 1

        self._jobs[
            job.id
        ] = job

        heapq.heappush(
            self._heap,
            entry,
        )

    def dequeue(
        self,
    ) -> Optional[
        WorkflowJob
    ]:
        while self._heap:
            entry = heapq.heappop(
                self._heap
            )

            job = self._jobs.get(
                entry.job_id
            )

            if job is None:
                continue

            if (
                job.status
                != JobStatus.QUEUED
            ):
                continue

            del self._jobs[
                job.id
            ]

            return job

        return None

    def peek(
        self,
    ) -> Optional[
        WorkflowJob
    ]:
        while self._heap:
            entry = self._heap[0]

            job = self._jobs.get(
                entry.job_id
            )

            if (
                job is None
                or job.status
                != JobStatus.QUEUED
            ):
                heapq.heappop(
                    self._heap
                )

                continue

            return job

        return None

    def remove(
        self,
        job_id: str,
    ) -> WorkflowJob:
        job = self._jobs.get(
            job_id
        )

        if job is None:
            raise JobNotFoundError(
                "Job not found in queue: "
                f"{job_id}"
            )

        del self._jobs[
            job_id
        ]

        return job

    def get(
        self,
        job_id: str,
    ) -> Optional[
        WorkflowJob
    ]:
        return self._jobs.get(
            job_id
        )

    def contains(
        self,
        job_id: str,
    ) -> bool:
        return (
            job_id
            in self._jobs
        )

    def size(
        self,
    ) -> int:
        return len(
            self._jobs
        )

    def empty(
        self,
    ) -> bool:
        return (
            self.size()
            == 0
        )

    def clear(
        self,
    ) -> int:
        count = len(
            self._jobs
        )

        self._heap.clear()
        self._jobs.clear()

        return count

    def queued_jobs(
        self,
    ) -> list[
        WorkflowJob
    ]:
        valid_entries = []

        for entry in self._heap:
            job = self._jobs.get(
                entry.job_id
            )

            if (
                job is not None
                and job.status
                == JobStatus.QUEUED
            ):
                valid_entries.append(
                    entry
                )

        valid_entries.sort()

        return [
            self._jobs[
                entry.job_id
            ]
            for entry
            in valid_entries
        ]
