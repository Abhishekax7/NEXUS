from dataclasses import dataclass
from typing import Optional

from app.core.engine import (
    NexusEngine,
)
from app.core.state import (
    NexusState,
)

from app.jobs.models import (
    JobExecutionResult,
    JobStateError,
    JobStatus,
    WorkflowJob,
    utc_now,
)


@dataclass
class WorkflowWorker:
    """
    Executes one queued NEXUS workflow job
    against a NexusEngine.

    The worker is responsible for:
    - validating job lifecycle state
    - marking execution timestamps
    - invoking the engine
    - capturing completion or failure
    """

    engine: NexusEngine

    def execute(
        self,
        job: WorkflowJob,
        state: NexusState,
    ) -> JobExecutionResult:
        if (
            job.status
            != JobStatus.QUEUED
        ):
            raise JobStateError(
                "Only queued jobs can "
                "be executed."
            )

        if (
            job.run_id
            != state.run_id
        ):
            raise JobStateError(
                "Job run_id does not match "
                "the supplied NexusState."
            )

        job.status = (
            JobStatus.RUNNING
        )

        job.started_at = (
            utc_now()
        )

        job.attempt += 1

        try:
            result_state = (
                self.engine.run(
                    state
                )
            )

            if result_state.failed:
                raise JobStateError(
                    "NEXUS engine returned "
                    "a failed workflow state."
                )

            job.status = (
                JobStatus.COMPLETED
            )

            job.completed_at = (
                utc_now()
            )

            return JobExecutionResult(
                job_id=job.id,
                run_id=job.run_id,
                status=(
                    JobStatus.COMPLETED
                ),
                success=True,
                started_at=(
                    job.started_at
                ),
                completed_at=(
                    job.completed_at
                ),
                metadata={
                    "iteration":
                        result_state.iteration,
                    "artifact_count":
                        len(
                            result_state.artifacts
                        ),
                    "task_count":
                        len(
                            result_state.tasks
                        ),
                },
            )

        except Exception as exc:
            job.status = (
                JobStatus.FAILED
            )

            job.completed_at = (
                utc_now()
            )

            return JobExecutionResult(
                job_id=job.id,
                run_id=job.run_id,
                status=(
                    JobStatus.FAILED
                ),
                success=False,
                started_at=(
                    job.started_at
                ),
                completed_at=(
                    job.completed_at
                ),
                error=str(
                    exc
                ),
                metadata={
                    "attempt":
                        job.attempt,
                },
            )

    def can_retry(
        self,
        job: WorkflowJob,
    ) -> bool:
        return (
            job.status
            == JobStatus.FAILED
            and job.attempt
            < job.max_attempts
        )

    def prepare_retry(
        self,
        job: WorkflowJob,
    ) -> WorkflowJob:
        if not self.can_retry(
            job
        ):
            raise JobStateError(
                "Job cannot be retried."
            )

        job.status = (
            JobStatus.QUEUED
        )

        job.started_at = None
        job.completed_at = None

        return job

    def cancel(
        self,
        job: WorkflowJob,
    ) -> WorkflowJob:
        if (
            job.status
            not in {
                JobStatus.QUEUED,
            }
        ):
            raise JobStateError(
                "Only queued jobs can "
                "be cancelled."
            )

        job.status = (
            JobStatus.CANCELLED
        )

        job.completed_at = (
            utc_now()
        )

        return job
