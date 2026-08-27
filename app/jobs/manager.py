from typing import Optional

from app.core.state import (
    NexusState,
)

from app.jobs.models import (
    JobExecutionResult,
    JobNotFoundError,
    JobPriority,
    JobSnapshot,
    JobStateError,
    JobStatus,
    WorkflowJob,
)
from app.jobs.queue import (
    PriorityJobQueue,
)
from app.jobs.worker import (
    WorkflowWorker,
)


class JobManager:
    """
    Coordinates NEXUS asynchronous
    workflow jobs.

    Responsibilities:
    - submit jobs
    - track workflow states
    - execute queued work
    - inspect job status
    - cancel queued jobs
    - retry failed jobs
    """

    def __init__(
        self,
        queue: PriorityJobQueue,
        worker: WorkflowWorker,
    ):
        self.queue = queue
        self.worker = worker

        self._jobs: dict[
            str,
            WorkflowJob,
        ] = {}

        self._states: dict[
            str,
            NexusState,
        ] = {}

        self._results: dict[
            str,
            JobExecutionResult,
        ] = {}

    def submit(
        self,
        state: NexusState,
        *,
        priority: JobPriority = (
            JobPriority.NORMAL
        ),
        max_attempts: int = 1,
        metadata: Optional[
            dict
        ] = None,
    ) -> WorkflowJob:
        job = WorkflowJob(
            run_id=state.run_id,
            priority=priority,
            max_attempts=(
                max_attempts
            ),
            metadata=(
                metadata
                or {}
            ),
        )

        self._jobs[
            job.id
        ] = job

        self._states[
            job.id
        ] = state

        self.queue.enqueue(
            job
        )

        return job

    def get_job(
        self,
        job_id: str,
    ) -> WorkflowJob:
        job = self._jobs.get(
            job_id
        )

        if job is None:
            raise JobNotFoundError(
                "Job not found: "
                f"{job_id}"
            )

        return job

    def get_state(
        self,
        job_id: str,
    ) -> NexusState:
        state = self._states.get(
            job_id
        )

        if state is None:
            raise JobNotFoundError(
                "Workflow state not found "
                f"for job: {job_id}"
            )

        return state

    def get_result(
        self,
        job_id: str,
    ) -> Optional[
        JobExecutionResult
    ]:
        self.get_job(
            job_id
        )

        return self._results.get(
            job_id
        )

    def snapshot(
        self,
        job_id: str,
    ) -> JobSnapshot:
        job = self.get_job(
            job_id
        )

        terminal = (
            job.status
            in {
                JobStatus.COMPLETED,
                JobStatus.FAILED,
                JobStatus.CANCELLED,
            }
        )

        return JobSnapshot(
            job_id=job.id,
            run_id=job.run_id,
            status=job.status,
            priority=job.priority,
            attempt=job.attempt,
            max_attempts=(
                job.max_attempts
            ),
            queued=(
                job.status
                == JobStatus.QUEUED
            ),
            running=(
                job.status
                == JobStatus.RUNNING
            ),
            terminal=terminal,
        )

    def execute_next(
        self,
    ) -> Optional[
        JobExecutionResult
    ]:
        job = self.queue.dequeue()

        if job is None:
            return None

        state = self.get_state(
            job.id
        )

        result = (
            self.worker.execute(
                job,
                state,
            )
        )

        self._results[
            job.id
        ] = result

        return result

    def execute_job(
        self,
        job_id: str,
    ) -> JobExecutionResult:
        job = self.get_job(
            job_id
        )

        if (
            job.status
            != JobStatus.QUEUED
        ):
            raise JobStateError(
                "Only queued jobs can "
                "be executed."
            )

        if self.queue.contains(
            job_id
        ):
            self.queue.remove(
                job_id
            )

        state = self.get_state(
            job_id
        )

        result = (
            self.worker.execute(
                job,
                state,
            )
        )

        self._results[
            job.id
        ] = result

        return result

    def cancel(
        self,
        job_id: str,
    ) -> WorkflowJob:
        job = self.get_job(
            job_id
        )

        if self.queue.contains(
            job_id
        ):
            self.queue.remove(
                job_id
            )

        return self.worker.cancel(
            job
        )

    def retry(
        self,
        job_id: str,
    ) -> WorkflowJob:
        job = self.get_job(
            job_id
        )

        prepared = (
            self.worker
            .prepare_retry(
                job
            )
        )

        self.queue.enqueue(
            prepared
        )

        self._results.pop(
            job_id,
            None,
        )

        return prepared

    def pending_jobs(
        self,
    ) -> list[
        WorkflowJob
    ]:
        return (
            self.queue
            .queued_jobs()
        )

    def job_count(
        self,
    ) -> int:
        return len(
            self._jobs
        )

    def result_count(
        self,
    ) -> int:
        return len(
            self._results
        )
